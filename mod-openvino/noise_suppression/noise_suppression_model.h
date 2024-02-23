#pragma once

#include <vector>
#include <memory>

#include "WaveTrack.h"

class NoiseSuppressionModel
{
public:

   // Return true to cancel.
   typedef bool (*ProgressCallbackFunc)(float perc_complete, //<- range 0 to 1
      void* user);

   virtual int sample_rate() = 0;

   virtual bool run(std::shared_ptr<WaveChannel> pChannel,
      sampleCount start,
      size_t total_samples,
      ProgressCallbackFunc callback = nullptr,
      void *callback_user = nullptr) = 0;
};
