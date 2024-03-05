// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <openvino/opsets/opset8.hpp>
#include "musicgen_decoder_model.h"
#include "musicgen_utils.h"
#include "musicgen_config.h"

namespace ov_musicgen
{
   class MusicgenDecoderModelFullStaticBatch2 : public MusicgenDecoderModel
   {
   public:

      const size_t N_LAYERS = 24;

      MusicgenDecoderModelFullStaticBatch2(ov::Core& core, MusicGenConfig& config)
      {
         auto model_folder = config.model_folder;

         std::string device = config.musicgen_decode_device0;

         auto tensortype = device == "CPU" ? ov::element::f32 : ov::element::f16;

         if (config.bStereo)
         {
            model_folder = FullPath(model_folder, "stereo");
         }
         else
         {
            model_folder = FullPath(model_folder, "mono");
         }

         //prep decoder model
         {
            std::string decoder_model_path, binfile;
            switch (config.model_selection)
            {
            case MusicGenConfig::ModelSelection::MUSICGEN_SMALL_FP16:
               decoder_model_path = FullPath(model_folder, "musicgen_decoder_static.xml");
               binfile = FullPath(model_folder, "musicgen_decoder_combined_weights.bin");
               break;

            case  MusicGenConfig::ModelSelection::MUSICGEN_SMALL_INT8:
               decoder_model_path = FullPath(model_folder, "musicgen_decoder_static_int8.xml");
               binfile = FullPath(model_folder, "musicgen_decoder_combined_weights_int8.bin");
               break;

            default:
               throw std::runtime_error("Invalid model selection");
               break;
            }

            std::cout << " Using model=" << decoder_model_path << ", " << binfile << std::endl;

            std::shared_ptr<ov::Model> model = core.read_model(decoder_model_path, binfile);

            {
               size_t max_tokens = 1004;
               std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
               port_to_shape[model->input("input_hidden_states")] = { 2, 1, 1024 };
               port_to_shape[model->input("encoder_attention_mask")] = { 2, 1, 1, 64 };
               port_to_shape[model->input("custom_attention_mask")] = { 1, max_tokens + 1 };

               for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
               {
                  for (size_t i = 0; i < 4; i++)
                  {
                     std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);

                     if (i < 2)
                     {
                        port_to_shape[model->input(tensorname)] = { 2, 16, max_tokens, 64 };
                     }
                     else
                     {
                        port_to_shape[model->input(tensorname)] = { 2, 16, 64, 64 };
                     }
                  }
               }

               model->reshape(port_to_shape);
            }

            ov::preprocess::PrePostProcessor ppp(model);

            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (size_t i = 0; i < 4; i++)
               {
                  std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
                  ppp.input(tensorname).tensor().set_element_type(tensortype);
               }

               for (int i = 0; i < 2; i++)
               {
                  std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
                  ppp.output(tensorname).tensor().set_element_type(tensortype);

               }
            }

            model = ppp.build();

            logBasicModelInfo(model);

            using namespace std::chrono;
            using Clock = std::chrono::high_resolution_clock;

            uint64_t  t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            ov::CompiledModel compiledModel = core.compile_model(model, device);
            uint64_t  t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            std::cout << "    compile time = " << (t1 - t0) << " ms" << std::endl;

            _infer_request = compiledModel.create_infer_request();

            //first inference is usually a bit longer due to some lazy initialization, so trigger a warm up inference.
            _infer_request.infer();

            _hidden_states = wrap_ov_tensor_as_torch(_infer_request.get_tensor("input_hidden_states"));
            _encoder_attention_mask_ov = _infer_request.get_tensor("encoder_attention_mask");
            _encoder_attention_mask = wrap_ov_tensor_as_torch(_encoder_attention_mask_ov);

            _custom_attention_mask = _infer_request.get_tensor("custom_attention_mask");

            _past_key_values_ov.resize(N_LAYERS);
            _past_key_values_torch.resize(N_LAYERS);
            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               _past_key_values_ov[layeri].resize(4);
               _past_key_values_torch[layeri].resize(4);

               for (size_t i = 0; i < 4; i++)
               {
                  std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);

                  _past_key_values_ov[layeri][i] = _infer_request.get_tensor(tensorname);

                  //wrap ov tensor as torch tensor
                  _past_key_values_torch[layeri][i] = wrap_ov_tensor_as_torch(_past_key_values_ov[layeri][i]);
               }
            }

            //get last hidden state tensor
            _last_hidden_state_ov = _infer_request.get_tensor("last_hidden_state");

            //wrap it as a torch::Tensor
            _last_hidden_state = wrap_ov_tensor_as_torch(_last_hidden_state_ov);

            _new_key_values_ov.resize(N_LAYERS);
            _new_key_values.resize(N_LAYERS);
            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               _new_key_values[layeri].resize(2);
               _new_key_values_ov[layeri].resize(2);

               for (int i = 0; i < 2; i++)
               {
                  std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);

                  auto ov_tensor = _infer_request.get_tensor(tensorname);

                  _new_key_values_ov[layeri][i] = ov_tensor;

                  //wrap ov tensor as torch tensor
                  _new_key_values[layeri][i] = wrap_ov_tensor_as_torch(ov_tensor);
               }
            }

            std::cout << "Batch2 Decoder" << std::endl;
            std::cout << "    tensortype = " << tensortype << std::endl;
            std::cout << "    max token length = " << MaxNewTokens() << std::endl;
         }

         //prep initial model
         {
            auto model_path = FullPath(model_folder, "initial_cross_attn_kv_producer.xml");
            std::shared_ptr<ov::Model> model = core.read_model(model_path);

            model->reshape({ {2, _past_key_values_ov[0][2].get_shape()[2], 1024} });

            ov::preprocess::PrePostProcessor ppp(model);

            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (int i = 0; i < 2; i++)
               {
                  std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
                  ppp.output(tensorname).tensor().set_element_type(tensortype);

               }
            }

            model = ppp.build();

            ov::CompiledModel compiledModel = core.compile_model(model, device);

            _infer_request_initial = compiledModel.create_infer_request();
            _infer_request_initial.infer();

            _intiial_encoder_hidden_state = wrap_ov_tensor_as_torch(_infer_request_initial.get_input_tensor());
            _initial_past_key_values.resize(N_LAYERS);
            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               _initial_past_key_values[layeri].resize(2);
               for (int i = 0; i < 2; i++)
               {
                  std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);

                  //link the initial model with the decode model
                  _infer_request_initial.set_tensor(tensorname, _past_key_values_ov[layeri][i + 2]);
               }
            }
         }

         {
            std::string model_path;
            switch (config.m_continuation_context)
            {
            case MusicGenConfig::ContinuationContext::FIVE_SECONDS:
            {
               model_path = FullPath(model_folder, "musicgen_decoder_static0_5s.xml");
               auto attn_mask_raw_file = FullPath(config.model_folder, "attention_mask_from_prepare_4d_causal_5s.raw");
               _attention_mask = read_tensor(attn_mask_raw_file, { 2, 1, 251, 251 });
            }
            break;

            case MusicGenConfig::ContinuationContext::TEN_SECONDS:
            {
               model_path = FullPath(model_folder, "musicgen_decoder_static0_10s.xml");
               auto attn_mask_raw_file = FullPath(config.model_folder, "attention_mask_from_prepare_4d_causal_10s.raw");
               _attention_mask = read_tensor(attn_mask_raw_file, { 2, 1, 501, 501 });
            }
            break;
            }

            auto binfile = FullPath(model_folder, "musicgen_decoder_combined_weights.bin");

            std::cout << "reading model as " << model_path << ", " << binfile << std::endl;
            std::shared_ptr<ov::Model> model = core.read_model(model_path, binfile);

            ov::preprocess::PrePostProcessor ppp(model);

            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (size_t i = 0; i < 2; i++)
               {
                  std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i + 2);
                  ppp.input(tensorname).tensor().set_element_type(tensortype);
               }

               for (int i = 0; i < 2; i++)
               {
                  std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);
                  ppp.output(tensorname).tensor().set_element_type(tensortype);

               }
            }

            model = ppp.build();

            std::cout << "large context model" << std::endl;
            logBasicModelInfo(model);

            ov::CompiledModel compiledModel = core.compile_model(model, device);

            _infer_request_large_context = compiledModel.create_infer_request();
            _infer_request_large_context.infer();

            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (int i = 0; i < 2; i++)
               {
                  std::string tensorname = "past_key_value_" + std::to_string(layeri) + "_" + std::to_string(i + 2);

                  //link the initial model with the this model
                  _infer_request_large_context.set_tensor(tensorname, _past_key_values_ov[layeri][i + 2]);
               }
            }
         }

         Reset();
      }

      virtual int64_t MaxNewTokens() override
      {
         return _past_key_values_ov[0][0].get_shape()[2];
      }

      virtual void Reset() override
      {
         _past_length = 0;

         memset(_custom_attention_mask.data<float>(), 0, _custom_attention_mask.get_byte_size());
         for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
         {
            for (size_t i = 0; i < 4; i++)
            {
               void* pData = _past_key_values_ov[layeri][i].data();
               memset(pData, 0, _past_key_values_ov[layeri][i].get_byte_size());
            }
         }

         memset(_infer_request_initial.get_input_tensor().data<float>(), 0, _infer_request_initial.get_input_tensor().get_byte_size());

         {
            float* pAttnMask = _custom_attention_mask.data<float>();
            //std::cout << "_custom_attention_mask.get_size() = " << _custom_attention_mask.get_size() << std::endl;
            for (int i = 0; i < _custom_attention_mask.get_size(); i++)
            {
               pAttnMask[i] = -INFINITY;
            }
         }

         {
            float* pAttnMask = _encoder_attention_mask_ov.data<float>();
            for (int i = 0; i < _encoder_attention_mask_ov.get_size(); i++)
            {
               pAttnMask[i] = -INFINITY;
            }
         }
      }

      virtual void ShiftLeft(int64_t ntokens) override
      {

      }

      virtual ov::Tensor run(torch::Tensor hidden_states, std::optional<torch::Tensor> encoder_hidden_states, std::optional<torch::Tensor> encoder_attention_mask) override
      {
         ITT_SCOPED_TASK(MusicgenModelStatic_run)
            ov::Tensor last_hidden_states_ret;
         if (_past_length == 0)
         {
            ITT_SCOPED_TASK(initial_infer)
               using namespace torch::indexing;
            _intiial_encoder_hidden_state.index_put_({ Slice(), Slice(0, encoder_hidden_states->sizes()[1]), Slice() }, *encoder_hidden_states);

            _infer_request_initial.infer();
         }

         if (hidden_states.sizes()[1] > 1)
         {
            ITT_SCOPED_TASK(large_context_path)
               using namespace torch::indexing;

            auto input_hidden_states = wrap_ov_tensor_as_torch(_infer_request_large_context.get_tensor("input_hidden_states"));
            auto input_attention_mask = wrap_ov_tensor_as_torch(_infer_request_large_context.get_tensor("attention_mask"));
            auto input_encoder_attention_mask = wrap_ov_tensor_as_torch(_infer_request_large_context.get_tensor("encoder_attention_mask"));

            input_hidden_states.copy_(hidden_states);
            input_attention_mask.copy_(_attention_mask);

            //first, fill with -INF
            input_encoder_attention_mask.copy_(torch::full(input_encoder_attention_mask.sizes(), -INFINITY));

            //then slice the valid values in.
            if (encoder_attention_mask)
            {
               input_encoder_attention_mask.index_put_({ Slice(), Slice(), Slice(), Slice(0, encoder_attention_mask->sizes()[3]) }, *encoder_attention_mask);
            }

            {
               ITT_SCOPED_TASK(infer)
                  _infer_request_large_context.infer();
            }

            auto ov_last_hidden_state = _infer_request_large_context.get_tensor("last_hidden_state");

            last_hidden_states_ret = ov_last_hidden_state;

            ITT_SCOPED_TASK(update_past_key_values);

            int64_t valid_height = ov_last_hidden_state.get_shape()[1];

            //insert the new key / value tensors into the past keys tensor
            for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
            {
               for (int i = 0; i < 2; i++)
               {
                  using namespace torch::indexing;
                  std::string tensorname = "new_key_value_" + std::to_string(layeri) + "_" + std::to_string(i);

                  auto new_key_val_torch = wrap_ov_tensor_as_torch(_infer_request_large_context.get_tensor(tensorname));
                  _past_key_values_torch[layeri][i].index_put_({ Slice(), Slice(), Slice(0, valid_height), Slice() }, new_key_val_torch);
               }
            }

            float* pAttnMask = _custom_attention_mask.data<float>();
            memset(pAttnMask, 0, valid_height * sizeof(float));
            _past_length = valid_height;
         }
         else
         {
            //set attention mask
            float* pAttnMask = _custom_attention_mask.data<float>();
            if (_past_length > 0)
            {
               pAttnMask[_past_length - 1] = 0.f;
            }

            pAttnMask[_custom_attention_mask.get_size() - 1] = 0.f;

            //set input tensors
            {
               using namespace torch::indexing;
               _hidden_states.index_put_({ Slice(), Slice(), Slice() }, hidden_states);

               if (encoder_attention_mask)
               {
                  _encoder_attention_mask.index_put_({ Slice(), Slice(), Slice(), Slice(0, encoder_attention_mask->sizes()[3]) }, *encoder_attention_mask);
               }
            }

            {
               ITT_SCOPED_TASK(infer)
                  _infer_request.infer();
            }

            last_hidden_states_ret = _last_hidden_state_ov;

            if (_past_length < _past_key_values_torch[0][0].sizes()[2])
            {
               ITT_SCOPED_TASK(update_past_key_values);

               //insert the new key / value tensors into the past keys tensor
               for (size_t layeri = 0; layeri < N_LAYERS; layeri++)
               {
                  for (int i = 0; i < 2; i++)
                  {
                     using namespace torch::indexing;

                     _past_key_values_torch[layeri][i].index_put_({ Slice(), Slice(), Slice(_past_length, _past_length + 1), Slice() }, _new_key_values[layeri][i]);
                  }
               }

               _past_length++;
            }
            else
            {
               throw std::runtime_error("past key length exceeded it's max length");
            }
         }

         return last_hidden_states_ret;
      }

      virtual ov::Tensor get_last_hidden_state() override
      {
         return _last_hidden_state_ov;
      }

      virtual int64_t PastLength() override
      {
         return _past_length;
      }

   private:

      ov::InferRequest _infer_request;

      //input tensors (OpenVINO tensors & OV Tensors wrapped as torch::Tensor's)
      torch::Tensor _hidden_states;
      ov::Tensor _encoder_attention_mask_ov;
      torch::Tensor _encoder_attention_mask;
      ov::Tensor _custom_attention_mask;

      std::vector< std::vector< ov::Tensor > > _past_key_values_ov;
      std::vector< std::vector< torch::Tensor > > _past_key_values_torch;

      //output tensors
      ov::Tensor _last_hidden_state_ov;
      torch::Tensor _last_hidden_state;  //simply a wrapper around _last_hidden_state_ov (pointing to same underlying buffer)
      std::vector< std::vector< ov::Tensor > > _new_key_values_ov;
      std::vector< std::vector< torch::Tensor > > _new_key_values;

      int64_t _past_length = 0;

      //model that needs to run when _past_length is 0 to produce initial key/vals
      ov::InferRequest _infer_request_initial;
      torch::Tensor _intiial_encoder_hidden_state;
      std::vector< std::vector< torch::Tensor > > _initial_past_key_values;

      //todo, this should go away:
      torch::Tensor _attention_mask;

      //large context stuff
      ov::InferRequest _infer_request_large_context;
   };
}
