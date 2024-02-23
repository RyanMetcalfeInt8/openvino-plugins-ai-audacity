// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include "OVNoiseSuppression.h"
#include "WaveTrack.h"
#include "EffectOutputTracks.h"
#include "effects/EffectEditor.h"
#include <math.h>
#include <iostream>

#include <wx/log.h>

#include "ViewInfo.h"
#include "TimeWarper.h"

#include <wx/intl.h>
#include <wx/valgen.h>

#include "ShuttleGui.h"

#include <wx/choice.h>
#include "FileNames.h"
#include "CodeConversions.h"

#include "LoadEffects.h"
#include <future>

#include <openvino/openvino.hpp>

#include "widgets/valnum.h"

#include "noise_suppression_omz_model.h"
#include "noise_suppression_df_model.h"

const ComponentInterfaceSymbol EffectOVNoiseSuppression::Symbol{ XO("OpenVINO Noise Suppression") };

namespace { BuiltinEffectsModule::Registration< EffectOVNoiseSuppression > reg; }


EffectOVNoiseSuppression::EffectOVNoiseSuppression()
{
   ov::Core core;

   auto ov_supported_device = core.get_available_devices();
   for (auto d : ov_supported_device)
   {
      //GNA devices are not supported
      if (d.find("GNA") != std::string::npos) continue;

      mSupportedDevices.push_back(d);
      mGuiDeviceSelections.push_back({ TranslatableString{ wxString(d), {}} });

   }
   
   mSupportedModels = { "deepfilternet3", "noise-suppression-denseunet-ll-0001" };

   for (auto m : mSupportedModels)
   {
      mGuiModelSelections.push_back({ TranslatableString{ wxString(m), {}} });
   }

}

EffectOVNoiseSuppression::~EffectOVNoiseSuppression()
{

}

// ComponentInterface implementation
ComponentInterfaceSymbol EffectOVNoiseSuppression::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVNoiseSuppression::GetDescription() const
{
   return XO("Applies AI Background Noise Suppression using OpenVINO");
}

VendorSymbol EffectOVNoiseSuppression::GetVendor() const
{
   return XO("OpenVINO AI Effects");
}

// EffectDefinitionInterface implementation

EffectType EffectOVNoiseSuppression::GetType() const
{
   return EffectTypeProcess;
}

bool EffectOVNoiseSuppression::IsInteractive() const
{
   return true;
}

std::unique_ptr<EffectEditor> EffectOVNoiseSuppression::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess&,
   const EffectOutputs*)
{
   S.AddSpace(0, 5);
   S.StartVerticalLay();
   {
      S.StartMultiColumn(4, wxCENTER);
      {
         //m_deviceSelectionChoice
         mTypeChoiceDeviceCtrl = S.Id(ID_Type)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_deviceSelectionChoice)
            .AddChoice(XXO("OpenVINO Inference Device:"),
               Msgids(mGuiDeviceSelections.data(), mGuiDeviceSelections.size()));
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         auto attn = S.Name(XO("Attenuation Limit (dB)"))
            .Validator<FloatingPointValidator<float>>(
               6, &mAttenuationLimit,
               NumValidatorStyle::NO_TRAILING_ZEROES,
               0.0f,
               100.0f)
            .AddTextBox(XO("Attenuation Limit (dB)"), L"", 12);
      }

      S.StartMultiColumn(4, wxCENTER);
      {
         //m_deviceSelectionChoice
         mTypeChoiceModelCtrl = S.Id(ID_Type_Model)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_modelSelectionChoice)
            .AddChoice(XXO("Noise Suppression Model:"),
               Msgids(mGuiModelSelections.data(), mGuiModelSelections.size()));
      }
      S.EndMultiColumn();
   }
   S.EndVerticalLay();

   return nullptr;
}

bool EffectOVNoiseSuppression::UpdateProgress(double perc)
{
   return TotalProgress(perc);
}

static bool NSProgressCallback(float perc_complete, void* user)
{
   EffectOVNoiseSuppression* pThis = (EffectOVNoiseSuppression*)user;

   return pThis->UpdateProgress(perc_complete);
}

bool EffectOVNoiseSuppression::Process(EffectInstance&, EffectSettings&)
{
   EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };
   bool bGoodResult = true;

   ov::CompiledModel compiledModel;
   std::shared_ptr< NoiseSuppressionModel > ns_model;
   try
   {
      auto compile_compiledModel_fut = std::async(std::launch::async, [this, &compiledModel]() {
         std::shared_ptr< NoiseSuppressionModel > ret;
         try {

            //CompileNoiseSuppression(compiledModel);
            FilePath model_folder = FileNames::MkDir(wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath());
            FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());
            std::string cache_path = audacity::ToUTF8(wxFileName(cache_folder).GetFullPath());

            if (m_modelSelectionChoice > 0)
            {
               auto model_file = audacity::ToUTF8(mTypeChoiceModelCtrl->GetString(m_modelSelectionChoice)) + ".xml";
               std::string  model_path = audacity::ToUTF8(wxFileName(model_folder, wxString(model_file))
                  .GetFullPath());

               std::cout << "Using model path = " << model_path << std::endl;
               ret = std::make_shared< NoiseSuppressionOMZModel >(model_path, mSupportedDevices[m_deviceSelectionChoice], cache_path);
            }
            else
            {
               auto ns_df = std::make_shared< NoiseSuppressionDFModel >(audacity::ToUTF8(wxFileName(model_folder).GetFullPath()), mSupportedDevices[m_deviceSelectionChoice], cache_path);
               std::cout << "setting attn limit of " << mAttenuationLimit << std::endl;
               ns_df->SetAttenLimit(mAttenuationLimit);

               ret = ns_df;
               
            }

            return ret;
         }
         catch (const std::exception& error) {
            wxLogError("In Noise Suppression Compilation, exception: %s", error.what());
            EffectUIServices::DoMessageBox(*this,
               XO("Noise Suppression failed. See details in Help->Diagnostics->Show Log..."),
               wxICON_STOP,
               XO("Error"));
            return ret;
         }
         });

      std::future_status status;
      float total_time = 0.f;
      do {
         using namespace std::chrono_literals;
         status = compile_compiledModel_fut.wait_for(0.5s);
         {
            std::string message = "Compiling Noise Suppression AI Model for " + mSupportedDevices[m_deviceSelectionChoice] + "...";
            if (total_time > 10)
            {
               message += " (This could take a while if this is the first time running this feature with this device)";
            }
            TotalProgress(0.01, TranslatableString{ wxString(message), {} });
         }

         total_time += 0.5;

      } while (status != std::future_status::ready);

      ns_model = compile_compiledModel_fut.get();

      if (!ns_model)
      {
         std::cout << "CompileNoiseSuppression routine failed." << std::endl;
         return false;
      }
   }
   catch (const std::exception& error) {
      std::cout << "CompileNoiseSuppression routine failed: " << error.what() << std::endl;
      return false;
   }

   try
   {
      //mCurTrackNum = 0;
      size_t wavetracki = 0;
      for (auto pOutWaveTrack : outputs.Get().Selected<WaveTrack>())
      {
         //Get start and end times from track
         double trackStart = pOutWaveTrack->GetStartTime();
         double trackEnd = pOutWaveTrack->GetEndTime();

         // Set the current bounds to whichever left marker is
         // greater and whichever right marker is less:
         const double curT0 = std::max(trackStart, mT0);
         const double curT1 = std::min(trackEnd, mT1);

         // Process only if the right marker is to the right of the left marker
         if (curT1 > curT0) {
            double origRate = pOutWaveTrack->GetRate();

            int model_sample_rate = ns_model->sample_rate();

            if (origRate != model_sample_rate)
            {
               std::cout << "resampling from " << origRate << " to " << model_sample_rate << std::endl;
               pOutWaveTrack->Resample(model_sample_rate);
            }
            
            //Transform the marker timepoints to samples
            auto start = pOutWaveTrack->TimeToLongSamples(curT0);
            auto end = pOutWaveTrack->TimeToLongSamples(curT1);

            size_t total_samples = (end - start).as_size_t();

            for (size_t channeli = 0; channeli < pOutWaveTrack->Channels().size(); channeli++)
            {
               std::string message = "Running Noise Suppression on Track " + std::to_string(wavetracki) + ", channel " + std::to_string(channeli);
               if (TotalProgress(0.01, TranslatableString{ wxString(message), {} }))
               {
                  return false;
               }

               auto pChannel = pOutWaveTrack->GetChannel(channeli);

               if (!ns_model->run(pChannel, start, total_samples, NSProgressCallback, this) )
               {
                  return false;
               }

            }

            //resample back to original rate.
            if (origRate != model_sample_rate)
            {
               pOutWaveTrack->Resample(origRate);
            }
         }

         wavetracki++;
      }

      if (bGoodResult)
         outputs.Commit();

   }
   catch (const std::exception& error) {
      std::cout << "CompileNoiseSuppression routine failed: " << error.what() << std::endl;
      return false;
   }

   return bGoodResult;
}

