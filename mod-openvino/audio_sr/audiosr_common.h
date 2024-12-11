#pragma once

#include <torch/torch.h>
#include <openvino/openvino.hpp>
#include <fstream>

#include "musicgen_utils.h"
namespace ov_audiosr
{

   struct Batch
   {
      torch::Tensor waveform;
      torch::Tensor stft;
      torch::Tensor log_mel_spec;
      int64_t sampling_rate;
      torch::Tensor waveform_lowpass;
      torch::Tensor lowpass_mel;
      double duration;
      int64_t target_frame;

      double cutoff_freq;
   };

   enum class AudioSRModel {
      BASIC,
      SPEECH
   };

   enum class AudioSRModelChunkSize {
      FIVE_SEC,
      TEN_SEC
   };

   struct AudioSR_Config
   {
      std::string model_folder;
      std::string first_stage_encoder_device = "CPU";
      std::string vae_feature_extract_device = "CPU";
      std::string ddpm__device = "CPU";
      std::string vocoder_device = "CPU";

      AudioSRModel model_selection;
      AudioSRModelChunkSize chunk_size;

      ov::Core core;
   };

   struct CallbackParams
   {
      //return false if you want to cancel. 
      typedef bool (*CallbackFunc)(int ddpm_ith_step_complete,
         void* user);

      CallbackFunc callback;
      void* user;
   };
}
