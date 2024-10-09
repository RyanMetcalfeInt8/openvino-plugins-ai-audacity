#pragma once
#include <memory>
#include <vector>
#include "audiosr_common.h"

class DDPMLatentDiffusion;
class AudioSR
{
public:

   AudioSR(std::string model_folder,
      std::string first_stage_encoder_device,
      std::string vae_feature_extract_device,
      std::string ddpm__device,
      std::string vocoder_device,
      AudioSRModel model_selection,
      std::string cache_dir="");

   void normalize_and_pad(float* pSamples, size_t num_samples, Batch& batch);

   torch::Tensor run_audio_sr(Batch& batch,
      double unconditional_guidance_scale=3.5,
      int ddim_steps = 50,
      int64_t seed = 42,
      std::optional< CallbackParams > callback_params = {}
      );

   // How many samples we apply super res pipeline to at once.
   size_t nchunk_samples() { return 491520; };

   AudioSR_Config config()
   {
      if (_audio_sr_config)
      {
         return *_audio_sr_config;
      }

      throw std::runtime_error("config not set.");
   }

   void set_config(AudioSR_Config config);
   
private:

   struct Impl;
   std::shared_ptr< Impl > _impl;

   //move this stuff to Impl?
   torch::Tensor _mel_basis;
   std::pair< torch::Tensor, torch::Tensor> _mel_spectrogram_train(torch::Tensor y);
   std::pair<torch::Tensor, torch::Tensor> _wav_feature_extraction(torch::Tensor waveform, int64_t target_frame);

   std::shared_ptr< AudioSR_Config > _audio_sr_config;
   std::shared_ptr< DDPMLatentDiffusion > _ddpm_latent_diffusion;

};
