#pragma once

#include "noise_suppression_model.h"
#include "deepfilternet3/deepfilter.h"

class NoiseSuppressionDFModel : public NoiseSuppressionModel
{
public:

   NoiseSuppressionDFModel(std::string model_path, std::string device, std::string cache_dir)
   {
      _df = std::make_shared< ov_deepfilternet3::DeepFilter >(model_path, device);
   }

   virtual int sample_rate() override
   {
      return 48000;
   }

   void SetAttenLimit(float atten_limit)
   {
      _atten_limit = atten_limit;
   }

   virtual bool run(std::shared_ptr<WaveChannel> pChannel, sampleCount start, size_t total_samples, ProgressCallbackFunc callback = nullptr, void* callback_user = nullptr) override
   {
      bool ret = true;

      Floats entire_input{ total_samples };
      bool bOkay = pChannel->GetFloats(entire_input.get(), start, total_samples);
      if (!bOkay)
      {
         throw std::runtime_error("Unable to get " + std::to_string(total_samples) + " samples.");
      }

      torch::Tensor input_wav_tensor = torch::from_blob(entire_input.get(), { 1, (int64_t)total_samples });

      auto wav = _df->filter(input_wav_tensor, _atten_limit, 20, callback, callback_user);

      if (!wav)
      {
         std::cout << "!wav -- returning false" << std::endl;
         return false;
      }

      ret = pChannel->Set((samplePtr)(wav->data()), floatSample, start, total_samples);

      return ret;
   }

private:

   std::shared_ptr<ov_deepfilternet3::DeepFilter> _df;
   float _atten_limit = 100.f;

};
