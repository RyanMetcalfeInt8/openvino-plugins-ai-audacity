#include "musicgen_for_conditional_generation.h"

void MusicgenForConditionalGeneration::SetSeed(unsigned int seed)
{
    torch::Generator generator = at::detail::createCPUGenerator();
    {
        std::lock_guard<std::mutex> lock(generator.mutex());
        generator.set_current_seed(seed);
    }

    _generator = generator;
}



MusicgenForConditionalGeneration::MusicgenForConditionalGeneration(MusicGenConfig& config)
{
    auto model_folder = config.model_folder;
    std::string cache_dir = *config.cache_folder;

    _core = std::make_shared< ov::Core >();
    auto& core = *_core;

    //set various device configuration
    {
        std::map<std::string, ov::AnyMap> config;

    }


    if (config.cache_folder)
    {
        core.set_property(ov::cache_dir(*config.cache_folder));
    }

    //hack! to line up results between CPP and python version. It should be removed
    // once everything is working (although we probably want to preserve ability to
    // set seed for consistency).
    torch::Generator generator = at::detail::createCPUGenerator();
    {
        std::lock_guard<std::mutex> lock(generator.mutex());
        generator.set_current_seed(1);
    }

    _generator = generator;


    {
        //prep text encoder
        auto modelpath = FullPath(model_folder, "t5.xml");
        std::shared_ptr<ov::Model> model = core.read_model(modelpath);

        //logBasicModelInfo(model);
        //TODO: Expose text encoder device?
        auto compiled_model = core.compile_model(model, "CPU");

        _text_encoder_infer_request = compiled_model.create_infer_request();
        _text_encoder_infer_request.infer();

    }

    _decoder = std::make_shared< MusicgenForCausalLM >(core, config);

    {
        //prep enc-to-dec proj model

        std::string enc_to_dec_model_folder = model_folder;
        if (config.bStereo)
        {
            enc_to_dec_model_folder = FullPath(enc_to_dec_model_folder, "Stereo");
        }

        auto modelpath = FullPath(enc_to_dec_model_folder, "enc_to_dec_proj.xml");
        std::shared_ptr<ov::Model> model = core.read_model(modelpath);

        model->reshape({ {2, ov::Dimension(1, 64), 768} });

        //logBasicModelInfo(model);

        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

        _enc_to_dec_proj_infer_request = compiled_model.create_infer_request();
    }

    {
        size_t num_encodec_secs = 20;

        auto modelpath = FullPath(model_folder, "encodec_" + std::to_string(num_encodec_secs) + "s.xml");
        auto binfile = FullPath(model_folder, "encodec_combined_weights.bin");

        std::shared_ptr<ov::Model> model = core.read_model(modelpath, binfile);

        
        model->reshape({ 1, 1, 4, num_encodec_secs*50 });

        std::cout << "encodec: " << std::endl;
        //logBasicModelInfo(model);

        auto compiled_model = core.compile_model(model, config.encoded_dec_device);

        _encodec_infer_request = compiled_model.create_infer_request();
    }

    _encoder = std::make_shared< MusicGenEncodecEncoder >(core, config);
}

torch::Tensor MusicgenForConditionalGeneration::apply_delay_pattern_mask(torch::Tensor input_ids, torch::Tensor decoder_pad_token_mask)
{
    using namespace torch::indexing;
    auto seq_len = input_ids.sizes().back();
    decoder_pad_token_mask = decoder_pad_token_mask.index({ "...", Slice(None, seq_len) });
    input_ids = torch::where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask);
    return input_ids;
}

torch::Tensor MusicgenForConditionalGeneration::prepare_inputs_for_generation(torch::Tensor decoder_input_ids, torch::Tensor decoder_delay_pattern_mask, std::optional<float> guidance_scale)
{
    ITT_SCOPED_TASK(prepare_inputs_for_generation)
        //todo:
        // if decoder_delay_pattern_mask is None:
        //    self.decoder.build_delay_pattern_mask(...)
        using namespace torch::indexing;

    //apply delay pattern mask
#if 0
    {
        auto seq_len = decoder_input_ids.sizes().back();
        auto decoder_pad_token_mask = decoder_delay_pattern_mask.index({ "...", Slice(None, seq_len) });
        //std::cout << "decoder_pad_token_mask.shape = " << decoder_pad_token_mask.sizes() << std::endl;
        //std::cout << "decoder_input_ids.shape = " << decoder_input_ids.sizes() << std::endl;
        //dump_tensor(decoder_pad_token_mask, "ov_decoder_pad_token_mask_into_where.raw");
        //dump_tensor(decoder_input_ids, "ov_decoder_input_ids_into_where.raw");
        decoder_input_ids = torch::where(decoder_pad_token_mask == -1, decoder_input_ids, decoder_pad_token_mask);
        //dump_tensor(decoder_input_ids, "ov_input_ids_after_where.raw");
    }
#else
    decoder_input_ids = apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask);
#endif

    if (guidance_scale && (*guidance_scale > 1))
    {
        decoder_input_ids = decoder_input_ids.repeat({ 2, 1 });

        //todo:
        //if decoder_attention_mask is not None:
        //    decoder_attention_mask = decoder_attention_mask.repeat((2, 1))
    }


    auto past_length = _decoder->PastLength();
    if (past_length >= 1)
    {
        int64_t remove_prefix_length;
        if (decoder_input_ids.sizes()[1] > past_length)
        {
            remove_prefix_length = past_length;
        }
        else
        {
            remove_prefix_length = decoder_input_ids.sizes()[1] - 1;
        }

        // std::cout << "remove_prefix_length = " << remove_prefix_length << std::endl;

         //decoder_input_ids = decoder_input_ids[:, remove_prefix_length: ];
        decoder_input_ids = decoder_input_ids.index({ Slice(), Slice(remove_prefix_length, None) });
    }

    return decoder_input_ids;

}

std::pair<torch::Tensor, torch::Tensor> MusicgenForConditionalGeneration::forward(std::optional<torch::Tensor> input_ids,
    std::optional<torch::Tensor> attention_mask,
    std::optional<torch::Tensor> input_values,
    std::optional<torch::Tensor> padding_mask,
    std::optional<torch::Tensor> decoder_input_ids,
    std::optional< BaseModelOutput > encoder_outputs,
    std::optional< torch::Tensor > encoder_hidden_states_in)
{
    ITT_SCOPED_TASK(MusicgenForConditionalGeneration_forward)
        _nforward_calls++;

    torch::Tensor encoder_hidden_states;
    if (!encoder_hidden_states_in)
    {
        encoder_hidden_states = *(*encoder_outputs).last_hidden_state;
        //todo?
        //if (
        //    self.text_encoder.config.hidden_size != self.decoder.config.hidden_size
        //    and self.decoder.config.cross_attention_hidden_size is None
        //): 
        //    encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        //todo: If encoder_outputs).last_hidden_state doesn't change per each call to forward,
        // why are we running this enc_to_dec_proj each time? Seems like we can run it once up
        // front, and then just keep reusing it...
        encoder_hidden_states = _enc_to_dec_proj(encoder_hidden_states);
    }
    else
    {
        encoder_hidden_states = *encoder_hidden_states_in;
    }


    if (attention_mask)
    {
        using namespace torch::indexing;
        encoder_hidden_states = encoder_hidden_states * (*attention_mask).index({ "...", None });
    }

    //todo?
    /*
    if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    elif decoder_input_ids is None and decoder_inputs_embeds is None:

        audio_encoder_outputs = self.audio_encoder(
            input_values=input_values,
            padding_mask=padding_mask,
            **kwargs_audio_encoder,
        )
        audio_codes = audio_encoder_outputs.audio_codes
        frames, bsz, codebooks, seq_len = audio_codes.shape
        if frames != 1:
            raise ValueError(
                f"Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is "
                "disabled by setting `chunk_length=None` in the audio encoder."
            )

        if self.config.decoder.audio_channels == 2 and audio_codes.shape[2] == self.decoder.num_codebooks // 2:
            # mono input through encodec that we convert to stereo
            audio_codes = audio_codes.repeat_interleave(2, dim=2)

        decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)

    */

    return { _decoder->forward(*decoder_input_ids, //input_ids
        {},  //attention_mask
        encoder_hidden_states,
        attention_mask, //encoder attention mask
        {}, //head mask
        {}, //cross_attn_head_mask
        {} //inputs_embeds
        ),
        encoder_hidden_states };
}

MusicgenForConditionalGeneration::GenerateReturn MusicgenForConditionalGeneration::generate(torch::Tensor inputs_tensor,
    int64_t max_token_length,
    torch::Tensor attention_mask,
    CallbackTracking& tracking,
    std::optional< torch::Tensor > audio_to_continue,
    std::optional< torch::Tensor > input_ids_to_continue,
    float guidance_scale,
    int64_t top_k,
    std::optional< CallbackParams > callback_params)
{
    ITT_SCOPED_TASK(generate)

    GenerateReturn ret;

    if (max_token_length <= 0)
    {
        throw std::invalid_argument("max_token_length needs to be > 0");
    }

    _decoder->Reset();

    max_token_length += 4;
    //size_t max_length = 504;
    int64_t pad_token_id = 2048;

    int64_t batch_size = 1;
    int64_t num_codebooks = _decoder->NumCodebooks();

    //auto inputs_tensor = read_tensor("raw_input_tensors\\inputs_tensor.raw", { 1, 12 }, torch::kInt64);
    auto input_ids = torch::full({ num_codebooks, 1 }, pad_token_id, torch::kInt64);

    //std::optional< torch::Tensor > decoder_input_ids;
    if (audio_to_continue)
    {
        std::cout << "encoding audio to continue.." << std::endl;
        torch::Tensor audio_codes;
        int64_t frames, bsz, codebooks, seq_len;
        if (_decoder->AudioChannels() == 1)
        {
            if (audio_to_continue->size(1) != 1)
            {
                throw std::runtime_error("Models are configured for mono, but audio-to-continue != 1 channel");
            }

            audio_codes = _encoder->encode(*audio_to_continue);
          
            frames = audio_codes.size(0);
            bsz = audio_codes.size(1);
            codebooks = audio_codes.size(2);
            seq_len = audio_codes.size(3);

            if (frames != 1)
            {
                throw std::runtime_error("generate: expected frames to be 1");
            }

            
        }
        else if (_decoder->AudioChannels() == 2)
        {
            if (audio_to_continue->size(1) != 2)
            {
                throw std::runtime_error("Models are configured for stereo, but audio-to-continue != 2 channels");
            }

            using namespace torch::indexing;

            auto input_vals_left = audio_to_continue->index({ Slice(), Slice(None, 1), Slice() });
            
            //careful! audio_codes_left is a thin wrapper around the encoder output tensor. 
            // this is why we do the index_put below before generating the right one (as it will implicitly)
            // change the values of audio_codes_left!
            auto audio_codes_left = _encoder->encode(input_vals_left);

            frames = audio_codes_left.size(0);
            bsz = audio_codes_left.size(1);
            codebooks = audio_codes_left.size(2);
            seq_len = audio_codes_left.size(3);

            // copy alternating left / right channel codes into stereo codebook
            audio_codes = audio_codes_left.new_ones({ frames, bsz, 2 * codebooks, seq_len });

            audio_codes.index_put_({ Slice(), Slice(), Slice(None, None, 2), Slice() }, audio_codes_left);

            auto input_vals_right = audio_to_continue->index({ Slice(), Slice(1, None), Slice() });
            
            auto audio_codes_right = _encoder->encode(input_vals_right);

            audio_codes.index_put_({ Slice(), Slice(), Slice(1, None, 2), Slice() }, audio_codes_right);

            

            std::cout << "done producing audio_codes" << std::endl;
        }
        else
        {
            throw std::runtime_error("AudioChannels() is not 1 or 2... (something is very wrong)");
        }

        auto decoder_input_ids = audio_codes.index({ 0, "..." }).reshape({ bsz * num_codebooks , seq_len });

        decoder_input_ids = torch::cat({ input_ids, decoder_input_ids }, -1);

        input_ids = decoder_input_ids;
    }
    else if (input_ids_to_continue)
    {
        input_ids = torch::cat({ input_ids, *input_ids_to_continue }, -1);
    }

    std::cout << "input_ids.shape = " << input_ids.sizes() << std::endl;

    //run text encoder
    std::optional< BaseModelOutput > encoder_outputs;
    {
        auto txt_encode_input = wrap_torch_tensor_as_ov(inputs_tensor);
        _text_encoder_infer_request.set_input_tensor(txt_encode_input);
        _text_encoder_infer_request.infer();

        auto txt_encode_out = _text_encoder_infer_request.get_output_tensor();
        //save_tensor_to_disk(txt_encode_out, "ov_txt_encode_out.raw");

        auto last_hidden_state = wrap_ov_tensor_as_torch(txt_encode_out);

        if (guidance_scale > 1)
        {
            last_hidden_state = torch::concatenate({ last_hidden_state, torch::zeros_like(last_hidden_state) }, 0);
        }

        BaseModelOutput output;
        output.last_hidden_state = last_hidden_state;

        encoder_outputs = output;

        //std::cout << "last_hidden_state shape = " << last_hidden_state.sizes() << std::endl;
    }

    //std::cout << "input_ids shape into build_delay_pattern_mask = " << input_ids.sizes() << std::endl;

    auto build_delay_pattern_mask_ret = _decoder->build_delay_pattern_mask(input_ids, pad_token_id, max_token_length);
    input_ids = build_delay_pattern_mask_ret.first;
    auto decoder_delay_pattern_mask = build_delay_pattern_mask_ret.second;

    std::cout << "calling sample.." << std::endl;
    auto output_ids = sample(input_ids, 
        attention_mask, 
        decoder_delay_pattern_mask, 
        encoder_outputs, 
        max_token_length,
        guidance_scale,
        top_k,
        tracking,
        callback_params);

    ret.input_ids = output_ids;

    //dump_tensor(output_ids, "ov_output_ids.raw");

#if 0
    {
        using namespace torch::indexing;
        auto golden = read_tensor("output_ids.raw", output_ids.sizes(), torch::kInt64);

        for (int64_t i = 0; i < golden.sizes()[1]; i++)
        {
            auto gi = golden.index({ Slice(), Slice(i, i + 1) });
            auto our_i = output_ids.index({ Slice(), Slice(i, i + 1) });

            if (!gi.equal(our_i))
            {
                std::cout << "mismatch between golden at index " << i << std::endl;
                break;
            }
        }
    }
#endif

    //std::cout << "output_ids size = " << output_ids.sizes() << std::endl;

    // apply the pattern mask to the final ids
    output_ids = apply_delay_pattern_mask(output_ids, decoder_delay_pattern_mask);

    // revert the pattern delay mask by filtering the pad token id
    torch::Tensor mask = output_ids != pad_token_id;
    output_ids = torch::masked_select(output_ids, mask);

    output_ids = torch::reshape(output_ids, { batch_size, num_codebooks, -1 });

    //  append the frame dimension back to the audio codes
    using namespace torch::indexing;
    output_ids = output_ids.index({ None, "..." });

    if (_decoder->AudioChannels() == 1)
    {
        ret.wav = ids_to_wav(output_ids);
    }
    else
    {
        auto left_input = output_ids.index({ Slice(), Slice(), Slice(None, None, 2), Slice() });
        ret.wav = ids_to_wav(left_input);

        auto right_input = output_ids.index({ Slice(), Slice(), Slice(1, None, 2), Slice() });
        ret.wav1 = ids_to_wav(right_input);
    }

    return ret;
}

std::shared_ptr<std::vector<float>> MusicgenForConditionalGeneration::ids_to_wav(torch::Tensor ids)
{
    using namespace torch::indexing;

    auto encodec_input_tensor = wrap_ov_tensor_as_torch(_encodec_infer_request.get_input_tensor());

    int64_t tokens_left_to_decode = ids.sizes()[3];

    //Conversion from number of tokens to number of audio samples is as follows..
    // There are 50 tokens per second
    // The EnCodec decoder produces 32 khz audio samples. (32000 samples for each second)
    // So, conversion is, (# of tokens)/ 50.0 * 32000.
    size_t number_of_output_samples = (size_t)std::ceil(((double)tokens_left_to_decode / 50.0) * 32000);

    std::cout << "total number of output samples for " << tokens_left_to_decode << "tokens = " << number_of_output_samples << std::endl;

    std::shared_ptr<std::vector<float>> wav = std::make_shared< std::vector<float> >(number_of_output_samples);

    int64_t encodec_input_token_size = encodec_input_tensor.sizes()[3];
    int64_t num_tokens_decoded_so_far = 0;
    size_t num_samples_produced_so_far = 0;

    //if we are going to be calling encodec multiple times back to back, we want to establish
    // some overlap so that we don't produce any noticeable pops or clicks at the wav
    // concatenation point.
    size_t overlap_tokens = 5; //should be even
    if (tokens_left_to_decode <= encodec_input_token_size)
    {
        //if we can decode all tokens in a single pass then, we don't worry about overlap.
        overlap_tokens = 0;
    }

    size_t overlap_samples = (size_t)std::ceil(((double)overlap_tokens / 50.0) * 32000);

    //todo: EEK! This one got away from me a bit -- this code is a mess and is way more complicated than it has to be, but seems to be working
    // for now. Clean it up, and also rework it so that we can run multiple iterations of the loop in parallel
    // (as we can have multiple infer requests & use async API).
    size_t decodei = 0;
    while (tokens_left_to_decode > 0)
    {
        int64_t num_tokens_to_decode_this_it = std::min(encodec_input_token_size, tokens_left_to_decode);
        
        encodec_input_tensor.index_put_({ Slice(), Slice(), Slice(), Slice(0, num_tokens_to_decode_this_it) },
            ids.index({ Slice(), Slice(), Slice(), Slice(num_tokens_decoded_so_far,
                num_tokens_decoded_so_far + num_tokens_to_decode_this_it) }));

        //todo: if num_tokens_to_decode_this_it <= encodec_input_token_size, we should probably fill the remainder
        // tensor with some padding values...
        
        {
            ITT_SCOPED_TASK(encodec_decode)
                _encodec_infer_request.infer();
        }

        auto audio_values = _encodec_infer_request.get_output_tensor();

        size_t num_samples_this_it = (size_t)std::ceil(((double)num_tokens_to_decode_this_it / 50.0) * 32000);
 
        if (tokens_left_to_decode > encodec_input_token_size)
        {
            num_tokens_to_decode_this_it -= overlap_tokens;
            num_samples_this_it -= overlap_samples / 2;
        }

        size_t offset = 0;
        if (decodei > 0)
        {
            offset += overlap_samples / 2;
            num_samples_this_it -= overlap_samples / 2;
        }

        num_tokens_decoded_so_far += num_tokens_to_decode_this_it;
        tokens_left_to_decode -= num_tokens_to_decode_this_it;

        if ((num_samples_produced_so_far + num_samples_this_it) > wav->size())
        {
            throw std::runtime_error("Unexpectedly, the output wav vector doesn't have enough elements to hold decoded output samples.");
        }

        std::memcpy(wav->data() + num_samples_produced_so_far, audio_values.data<float>() + offset, num_samples_this_it * sizeof(float));

        num_samples_produced_so_far += num_samples_this_it;

        decodei++;
    }

    return wav;
}

torch::Tensor MusicgenForConditionalGeneration::sample(torch::Tensor input_ids, 
    torch::Tensor attention_mask, 
    torch::Tensor decoder_delay_pattern_mask, 
    std::optional< BaseModelOutput > encoder_outputs, 
    size_t max_length,
    float guidance_scale,
    int64_t top_k,
    CallbackTracking& tracking,
    std::optional< CallbackParams > callback_params)
{
    ITT_SCOPED_TASK(MusicgenForConditionalGeneration_sample)

    //auto attention_mask = read_tensor("raw_input_tensors\\attention_mask.raw", { 2, 12 }, torch::kInt64);

    //std::cout << "attention_mask = " << attention_mask << std::endl;
#if 0
    BaseModelOutput encoder_outputs;
    encoder_outputs.last_hidden_state = read_tensor("raw_input_tensors\\encoder_hidden_states.raw", { 2, 12, 768 });
#endif

    using namespace std::chrono;
    using Clock = std::chrono::high_resolution_clock;

    uint64_t  t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    std::optional< torch::Tensor > encoder_hidden_states;

    std::cout << "going into while loop" << std::endl;
    int64_t tokens_generated_count = -4;

    int nforward_calls = 0;
    while (true)
    {
        nforward_calls++;

        auto decoder_input_ids = prepare_inputs_for_generation(input_ids, decoder_delay_pattern_mask, guidance_scale);

        //std::cout << "decoder_input_ids.dtype = " << decoder_input_ids.dtype() << ", decoder_input_ids shape = " << decoder_input_ids.sizes() << std::endl;

        //std::cout << "decoder_input_ids shape = " << decoder_input_ids.sizes() << std::endl;
        auto fwd_ret = forward(input_ids,
            attention_mask,
            {}, //input_values
            {}, //padding mask
            decoder_input_ids,
            encoder_outputs,
            encoder_hidden_states);

        auto logits = fwd_ret.first;
        encoder_hidden_states = fwd_ret.second;

        using namespace torch::indexing;
        auto next_token_logits = logits.index({ Slice(), -1, Slice() });

        auto next_token_scores = _logits_processor(input_ids, next_token_logits, guidance_scale);

        next_token_scores = _logits_warper(input_ids, next_token_scores, top_k, -INFINITY);


        //dump_tensor(next_token_scores, "ov_next_token_scores_into_softmax.raw");
        torch::Tensor probs;
        {
            ITT_SCOPED_TASK(softmax)
                probs = torch::softmax(next_token_scores, -1);
        }

        //std::cout << "probs shape into multinomial" << probs.sizes() << std::endl;

        //dump_tensor(probs, "ov_probs.raw
        torch::Tensor next_tokens;
        {
            ITT_SCOPED_TASK(multinomial)
                next_tokens = torch::multinomial(probs, 1, false, _generator).squeeze(1);
        }

        // std::cout << "next_tokens = " << next_tokens << std::endl;
        using namespace torch::indexing;

        //std::cout << "next_tokens = " << next_tokens << std::endl;
        //std::exit(0);

        input_ids = torch::cat({ input_ids, next_tokens.index({Slice(), None}) }, -1);

#if 0
        if (input_ids.sizes()[1] == 3)
        {
            //std::cout << "input_ids = " << input_ids.index({ Slice(), Slice(65, None) }) << std::endl;
            std::cout << "input_ids = " << input_ids << std::endl;
            std::cout << "exiting!" << std::endl;
            std::exit(0);
        }
#endif


        tokens_generated_count++;

        if (tokens_generated_count > 0)
        {
           tracking.total_tokens_generated_so_far++;

           if (callback_params)
           {
              if ((tracking.total_tokens_generated_so_far % callback_params->every_n_new_tokens) == 0)
              {
                 if (callback_params->callback)
                 {
                    float perc = (float)tracking.total_tokens_generated_so_far / tracking.total_tokens_to_generate;
                    callback_params->callback(perc, callback_params->user);
                 }
              }
           }
        }

        //std::cout << "input_ids shape = " << input_ids.sizes() << std::endl;
        if (input_ids.sizes()[1] >= max_length)
        {
            std::cout << ": breaking out of while loop" << std::endl;
            break;
        }

        
    }

    uint64_t  t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::cout << "loop time = " << t1 - t0 << std::endl;

    std::cout << "Final input_ids shape = " << input_ids.sizes() << std::endl;
    //dump_tensor(input_ids, "ov_input_ids_final.raw");

    _current_input_ids = input_ids;
    _current_attention_mask = attention_mask;
    _current_encoder_hidden_states = *encoder_hidden_states;
    _current_decoder_delay_pattern_mask = decoder_delay_pattern_mask;

    return input_ids;
}



torch::Tensor MusicgenForConditionalGeneration::_enc_to_dec_proj(torch::Tensor encoder_hidden_states)
{
    ITT_SCOPED_TASK(_enc_to_dec_proj)
        using namespace torch::indexing;
#if 0
    auto ov_input_tensor = _enc_to_dec_proj_infer_request.get_input_tensor();

    //wrap input tensor as a torch::Tensor
    auto input_tensor_wrapped = wrap_ov_tensor_as_torch(ov_input_tensor);

    auto ov_tensor_padded_size = input_tensor_wrapped.sizes()[1];
    auto hidden_state_valid_size = encoder_hidden_states.sizes()[1];

    if (ov_tensor_padded_size < hidden_state_valid_size)
        throw std::runtime_error("_enc_to_dec_proj: ov_tensor_padded_size < hidden_state_valid_size");

    //std::cout << "ov_tensor_padded_size = " << ov_tensor_padded_size << std::endl;
    //std::cout << "hidden_state_valid_size = " << hidden_state_valid_size << std::endl;



    // zero out ov input tensor to account for possible padding.
    // todo: This should be done during initialization.. no reason to add this 
    // to the critical path (not that I think it'll take very much time..)
    memset(ov_input_tensor.data<float>(), 0, ov_input_tensor.get_byte_size());

    //fill the valid portion of this tensor
    input_tensor_wrapped.index_put_({ Slice(), Slice(0, hidden_state_valid_size), Slice() }, encoder_hidden_states);

#else
    //std::cout << "encoder_hidden_states shape = " << encoder_hidden_states.sizes() << std::endl;
    auto ov_input_tensor = wrap_torch_tensor_as_ov(encoder_hidden_states);
    _enc_to_dec_proj_infer_request.set_input_tensor(ov_input_tensor);

#endif
    //run inference.
    //std::cout << "calling infer " << std::endl;
    _enc_to_dec_proj_infer_request.infer();

    //wrap output tensor as a torch tensor
    auto output_tensor_wrapped = wrap_ov_tensor_as_torch(_enc_to_dec_proj_infer_request.get_output_tensor());

    //std::cout << "output_tensor_wrapped shape  = " << output_tensor_wrapped.sizes() << std::endl;

    //std::exit(1);

    return output_tensor_wrapped;



    //return valid portion of this tensor
    //todo: we probably will end up sending a padded tensor to decode model,
    // so we'll most likely return full padded model... or even better, the 
    // output tensor of _enc_to_dec_proj_infer_request *is* the input tensor
    // for the decode model...
    //return output_tensor_wrapped.index({ Slice(), Slice(0, hidden_state_valid_size) , Slice() });
}

torch::Tensor MusicgenForConditionalGeneration::_logits_processor(torch::Tensor input_ids, torch::Tensor next_token_logits, float guidance_scale)
{
    ITT_SCOPED_TASK(_logits_processor)
        using namespace torch::indexing;

    size_t unguided_bsz = next_token_logits.sizes()[0] / 2;

    auto next_token_logits_split = next_token_logits.split(unguided_bsz, 0);
    auto cond_logits = next_token_logits_split[0];
    auto uncond_logits = next_token_logits_split[1];
    auto scores = uncond_logits + (cond_logits - uncond_logits) * guidance_scale;
    //{ cond_logits, uncond_logits } = next_token_logits.split(unguided_bsz, 0);
    //cond_logits = next_token_logits_split.get<0>;

    return scores;
}

torch::Tensor MusicgenForConditionalGeneration::_logits_warper(torch::Tensor input_ids, torch::Tensor next_token_scores, int64_t top_k, float filter_value)
{
    ITT_SCOPED_TASK(_logits_warper)
        top_k = std::min(top_k, next_token_scores.sizes().back());

    auto topk_values = std::get<0>(torch::topk(next_token_scores, top_k));

    using namespace torch::indexing;
    auto selected = topk_values.index({ "...", -1, None });
    //std::cout << "selected shape = " << selected.sizes() << std::endl;

    auto indices_to_remove = next_token_scores < selected;

    next_token_scores = next_token_scores.masked_fill(indices_to_remove, filter_value);

    return next_token_scores;
}
