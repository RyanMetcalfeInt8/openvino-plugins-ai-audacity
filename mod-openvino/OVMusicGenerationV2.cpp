// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include <openvino/openvino.hpp>
#include "OVMusicGenerationV2.h"
#include "WaveTrack.h"
#include "EffectOutputTracks.h"
#include "effects/EffectEditor.h"
#include <math.h>
#include <iostream>
#include <wx/log.h>

#include "BasicUI.h"
#include "ViewInfo.h"
#include "TimeWarper.h"
#include "LoadEffects.h"
#include "htdemucs.h"
#include "widgets/NumericTextCtrl.h"

#include <wx/intl.h>
#include <wx/valgen.h>
#include <wx/textctrl.h>
#include <wx/button.h>
#include <wx/checkbox.h>
#include <wx/wrapsizer.h>
#include <wx/stattext.h>

#include "ShuttleGui.h"

#include "widgets/valnum.h"
#include <wx/choice.h>
#include "FileNames.h"
#include "CodeConversions.h"
#include "SyncLock.h"
#include "ConfigInterface.h"

#include <future>

#include "InterpolateAudio.h"

const ComponentInterfaceSymbol EffectOVMusicGenerationV2::Symbol{ XO("OpenVINO Music Generation V2") };

namespace { BuiltinEffectsModule::Registration< EffectOVMusicGenerationV2 > reg; }

BEGIN_EVENT_TABLE(EffectOVMusicGenerationV2, wxEvtHandler)
EVT_BUTTON(ID_Type_UnloadModelsButton, EffectOVMusicGenerationV2::OnUnloadModelsButtonClicked)
EVT_CHOICE(ID_Type_ContextLength, EffectOVMusicGenerationV2::OnContextLengthChanged)
EVT_CHECKBOX(ID_Type_AudioContinuationCheckBox, EffectOVMusicGenerationV2::OnContextLengthChanged)
EVT_CHECKBOX(ID_Type_AudioContinuationAsNewTrackCheckBox, EffectOVMusicGenerationV2::OnContextLengthChanged)
END_EVENT_TABLE()

EffectOVMusicGenerationV2::EffectOVMusicGenerationV2()
{

   Parameters().Reset(*this);

   ov::Core core;

   //Find all supported devices on this system
   std::vector<std::string> devices = core.get_available_devices();

   for (auto d : devices)
   {
      //GNA devices are not supported
      if (d.find("GNA") != std::string::npos) continue;

      mGuiDeviceVPUSupportedSelections.push_back(wxString(d));

      if (d == "NPU") continue;

      mGuiDeviceNonVPUSupportedSelections.push_back(wxString(d));
   }

   std::vector<std::string> context_length_choices = { "5 Seconds", "10 Seconds" };
   for (auto d : context_length_choices)
   {
      mGuiContextLengthSelections.push_back({ TranslatableString{ wxString(d), {}} });
   }

   std::vector<std::string> model_selection_choices = { "musicgen-small-fp16-stereo",
                                                        "musicgen-small-int8-stereo",
                                                        "musicgen-small-fp16-mono",
                                                        "musicgen-small-int8-mono" };
   for (auto d : model_selection_choices)
   {
      mGuiModelSelections.push_back({ TranslatableString{ wxString(d), {}} });
   }

}

EffectOVMusicGenerationV2::~EffectOVMusicGenerationV2()
{
   _musicgen = {};
}

// ComponentInterface implementation
ComponentInterfaceSymbol EffectOVMusicGenerationV2::GetSymbol() const
{
   return Symbol;
}

TranslatableString EffectOVMusicGenerationV2::GetDescription() const
{
   return XO("Generates an audio track from a set of text prompts");
}

VendorSymbol EffectOVMusicGenerationV2::GetVendor() const
{
   return XO("OpenVINO AI Effects");
}

// EffectDefinitionInterface implementation
EffectType EffectOVMusicGenerationV2::GetType() const
{
   return EffectTypeGenerate;
}

static void NormalizeSamples(std::shared_ptr<std::vector<float>> samples, WaveTrack* base, float target_rms)
{
   auto tmp_tracklist = base->WideEmptyCopy();

   auto iter =
      (*tmp_tracklist->Any<WaveTrack>().begin())->Channels().begin();

   auto& tmp = **iter++;
   tmp.Append((samplePtr)samples->data(), floatSample, samples->size());

   auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();
   pTmpTrack->Flush();

   float tmp_rms = pTmpTrack->GetRMS(pTmpTrack->GetStartTime(), pTmpTrack->GetEndTime());

   float rms_ratio = target_rms / tmp_rms;
   {
      float* pSamples = samples->data();
      for (size_t si = 0; si < samples->size(); si++)
      {
         pSamples[si] *= rms_ratio;
      }
   }
}

bool EffectOVMusicGenerationV2::MusicGenCallback(float perc_complete)
{
   std::lock_guard<std::mutex> guard(mProgMutex);

   mProgressFrac = perc_complete;

   if (mIsCancelled)
   {
      return false;
   }

   return true;
}

static bool musicgen_callback(float perc_complete, void* user)
{
   EffectOVMusicGenerationV2* music_gen = (EffectOVMusicGenerationV2*)user;

   if (music_gen)
   {
      return music_gen->MusicGenCallback(perc_complete);
   }

   return true;
}

// Effect implementation
bool EffectOVMusicGenerationV2::Process(EffectInstance&, EffectSettings& settings)
{
   if (!mDurationT || (mDurationT->GetValue() <= 0))
   {
      std::cout << "Duration <= 0... returning" << std::endl;
      return false;
   }

   mIsCancelled = false;

   EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };
   bool bGoodResult = true;

   {
      FilePath model_folder = FileNames::MkDir(wxFileName(FileNames::BaseDir(), wxT("openvino-models")).GetFullPath());
      std::string musicgen_model_folder = audacity::ToUTF8(wxFileName(model_folder, wxString("musicgen"))
         .GetFullPath());

      auto encodec_device = audacity::ToUTF8(mTypeChoiceDeviceCtrl_EnCodec->GetString(m_deviceSelectionChoice_EnCodec));

      auto musicgen_dec0_device = audacity::ToUTF8(mTypeChoiceDeviceCtrl_Decode0->GetString(m_deviceSelectionChoice_MusicGenDecode0));
      auto musicgen_dec1_device = audacity::ToUTF8(mTypeChoiceDeviceCtrl_Decode1->GetString(m_deviceSelectionChoice_MusicGenDecode1));

      ov_musicgen::MusicGenConfig::ContinuationContext continuation_context;
      if (m_contextLengthChoice == 0)
      {
         std::cout << "continuation context of 5 seconds..." << std::endl;
         continuation_context = ov_musicgen::MusicGenConfig::ContinuationContext::FIVE_SECONDS;
      }
      else
      {
         std::cout << "continuation context of 10 seconds..." << std::endl;
         continuation_context = ov_musicgen::MusicGenConfig::ContinuationContext::TEN_SECONDS;
      }

      ov_musicgen::MusicGenConfig::ModelSelection model_selection;
      if ((m_modelSelectionChoice % 2) == 0)
      {
         model_selection = ov_musicgen::MusicGenConfig::ModelSelection::MUSICGEN_SMALL_FP16;
      }
      else
      {
         model_selection = ov_musicgen::MusicGenConfig::ModelSelection::MUSICGEN_SMALL_INT8;
      }

      bool bStereo = (m_modelSelectionChoice < 2);

      std::cout << "encodec_device = " << encodec_device << std::endl;
      std::cout << "MusicGen Decode Device 0 = " << musicgen_dec0_device << std::endl;
      std::cout << "MusicGen Decode Device 1 = " << musicgen_dec1_device << std::endl;

      FilePath cache_folder = FileNames::MkDir(wxFileName(FileNames::DataDir(), wxT("openvino-model-cache")).GetFullPath());
      std::string cache_path = audacity::ToUTF8(wxFileName(cache_folder).GetFullPath());
      std::cout << "cache path = " << cache_path << std::endl;

      wxString added_trackName;
      try
      {
         mProgress->SetMessage(TranslatableString{ wxString("Creating MusicGen Pipeline"), {} });

         auto musicgen_pipeline_creation_future = std::async(std::launch::async,
            [this, &musicgen_model_folder, &cache_path, &encodec_device, &musicgen_dec0_device, &musicgen_dec1_device,
            &continuation_context, &model_selection, &bStereo]
            {

               try
               {
                  //TODO: This should be much more efficient. No need to destroy *everything*, just update the
                  // pieces of the pipelines that have changed.
                  if ((musicgen_dec0_device != _musicgen_config.musicgen_decode_device0) ||
                     (musicgen_dec1_device != _musicgen_config.musicgen_decode_device1))
                  {
                     //force destroy music gen if config has changed.
                     _musicgen = {};
                  }

                  if (continuation_context != _musicgen_config.m_continuation_context)
                  {
                     _musicgen = {};
                  }

                  if (model_selection != _musicgen_config.model_selection)
                  {
                     _musicgen = {};
                  }

                  if ((encodec_device != _musicgen_config.encodec_enc_device) ||
                     (encodec_device != _musicgen_config.encodec_dec_device))
                  {
                     //force destroy music gen if config has changed.
                     _musicgen = {};
                  }

                  if (bStereo != _musicgen_config.bStereo)
                  {
                     _musicgen = {};
                  }

                  if (!_musicgen)
                  {
                     _musicgen_config.musicgen_decode_device0 = musicgen_dec0_device;
                     _musicgen_config.musicgen_decode_device1 = musicgen_dec1_device;

                     _musicgen_config.encodec_enc_device = encodec_device;
                     _musicgen_config.encodec_dec_device = encodec_device;

                     _musicgen_config.cache_folder = cache_path;
                     _musicgen_config.model_folder = musicgen_model_folder;

                     _musicgen_config.m_continuation_context = continuation_context;

                     _musicgen_config.bStereo = bStereo;

                     _musicgen_config.model_selection = model_selection;

                     _musicgen = std::make_shared< ov_musicgen::MusicGen >(_musicgen_config);
                  }
               }
               catch (const std::exception& error) {
                  std::cout << "In Music Generation V2, exception: " << error.what() << std::endl;
                  wxLogError("In Music Generation V2, exception: %s", error.what());
                  EffectUIServices::DoMessageBox(*this,
                     XO("Music Generation failed. See details in Help->Diagnostics->Show Log..."),
                     wxICON_STOP,
                     XO("Error"));
                  _musicgen = {};
               }
            });

         float total_time = 0.f;
         std::future_status status;
         do {
            using namespace std::chrono_literals;
            status = musicgen_pipeline_creation_future.wait_for(0.5s);

            {
               std::string message = "Loading Music Generation AI Models... ";
               if (total_time > 30)
               {
                  message += " (This could take a while if this is the first time running this feature with this device)";
               }
               if (TotalProgress(0.01, TranslatableString{ wxString(message), {} }))
               {
                  mIsCancelled = true;
               }
            }
         } while (status != std::future_status::ready);

         if (mIsCancelled)
         {
            return false;
         }

         if (!_musicgen)
         {
            wxLogError("MusicGen pipeline could not be created.");
            return false;
         }

         _prompt = audacity::ToUTF8(mTextPrompt->GetLineText(0));

         std::string seed_str = audacity::ToUTF8(mSeed->GetLineText(0));
         _seed_str = seed_str;

         std::optional<unsigned int> seed;
         if (!seed_str.empty() && seed_str != "")
         {
            seed = std::stoul(seed_str);
         }
         else
         {
            //seed is not set.. set it to time.
            time_t t;
            seed = (unsigned)time(&t);
         }

         std::cout << "Guidance Scale = " << mGuidanceScale << std::endl;
         std::cout << "TopK = " << mTopK << std::endl;

         std::string descriptor_str = "prompt: " + _prompt;
         descriptor_str += ", seed: " + std::to_string(*seed);
         descriptor_str += ", Guidance Scale = " + std::to_string(mGuidanceScale);
         descriptor_str += ", TopK = " + std::to_string(mTopK);

         added_trackName = wxString("Generated: (" + descriptor_str + ")");

         std::cout << "Duration = " << (float)mDurationT->GetValue() << std::endl;

         std::optional<ov_musicgen::MusicGen::AudioContinuationParams> audio_to_continue_params;

         double audio_to_continue_start, audio_to_continue_end;
         size_t audio_to_continue_samples = 0;
         if (_AudioContinuationCheckBox->GetValue())
         {
            ov_musicgen::MusicGen::AudioContinuationParams params;
            auto track = *(outputs.Get().Selected<WaveTrack>().begin());

            auto left = track->GetChannel(0);

            // create a temporary track list to append samples to
            auto tmp_tracklist = track->WideEmptyCopy();

            auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();

            auto iter =
               pTmpTrack->Channels().begin();

            {
               // If an existing (non-empty) track is highlighted, and the highlighted region
                // overlaps with existing segment.
               if (((mT0 >= left->GetStartTime()) && (mT0 < left->GetEndTime())) || ((mT1 >= left->GetStartTime()) && (mT1 < left->GetEndTime())))
               {
                  audio_to_continue_start = mT0;
                  audio_to_continue_end = mT1;
               }
               else
               {
                  audio_to_continue_start = mT0 - 10.f;
                  audio_to_continue_end = mT0;
               }

               double max_context_length = 0.f;
               switch (_musicgen_config.m_continuation_context)
               {
               case ov_musicgen::MusicGenConfig::ContinuationContext::FIVE_SECONDS:
                  max_context_length = 5.;
                  break;

               case ov_musicgen::MusicGenConfig::ContinuationContext::TEN_SECONDS:
                  max_context_length = 10.;
                  break;
               }

               if ((audio_to_continue_end - audio_to_continue_start) > max_context_length)
               {
                  audio_to_continue_start = audio_to_continue_end - max_context_length;
               }

               std::cout << "audio_to_continue_start = " << audio_to_continue_start << std::endl;
               std::cout << "audio_to_continue_end = " << audio_to_continue_end << std::endl;

               if (audio_to_continue_start < left->GetStartTime())
                  audio_to_continue_start = left->GetStartTime();

               auto start_s = left->TimeToLongSamples(audio_to_continue_start);
               auto end_s = left->TimeToLongSamples(audio_to_continue_end);

               size_t audio_to_continue_samples = (end_s - start_s).as_size_t();

               Floats entire_input{ audio_to_continue_samples };

               bool bOkay = left->GetFloats(entire_input.get(), start_s, audio_to_continue_samples);
               if (!bOkay)
               {
                  throw std::runtime_error("unable to get all left samples. GetFloats() failed for " +
                     std::to_string(audio_to_continue_samples) + "samples");
               }

               auto& tmpLeft = **iter++;
               tmpLeft.Append((samplePtr)entire_input.get(), floatSample, audio_to_continue_samples);

               //flush it
               auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();

               if (track->Channels().size() > 1)
               {
                  auto right = track->GetChannel(1);
                  bOkay = right->GetFloats(entire_input.get(), start_s, audio_to_continue_samples);
                  if (!bOkay)
                  {
                     throw std::runtime_error("unable to get all right samples. GetFloats() failed for " +
                        std::to_string(audio_to_continue_samples) + " samples");
                  }
                  auto& tmpRight = **iter;
                  tmpRight.Append((samplePtr)entire_input.get(), floatSample, audio_to_continue_samples);
               }

               pTmpTrack->Flush();

               if (pTmpTrack->GetRate() != 32000)
               {
                  pTmpTrack->Resample(32000, mProgress);
               }

               std::pair<std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>>> wav_pair;
               {
                  auto pResampledTrack = pTmpTrack->GetChannel(0);

                  auto start = pResampledTrack->GetStartTime();
                  auto end = pResampledTrack->GetEndTime();

                  auto start_s = pResampledTrack->TimeToLongSamples(start);
                  auto end_s = pResampledTrack->TimeToLongSamples(end);

                  audio_to_continue_samples = (end_s - start_s).as_size_t();

                  std::shared_ptr< std::vector<float> > resampled_samples_left = std::make_shared< std::vector<float> >(audio_to_continue_samples);
                  bool bOkay = pResampledTrack->GetFloats(resampled_samples_left->data(), start_s, audio_to_continue_samples);
                  if (!bOkay)
                  {
                     throw std::runtime_error("unable to get all left samples. GetFloats() failed for " +
                        std::to_string(audio_to_continue_samples) + "samples");
                  }


                  wav_pair.first = resampled_samples_left;

                  if (_musicgen_config.bStereo)
                  {
                     if (pTmpTrack->Channels().size() > 1)
                     {
                        auto pResampledTrackR = pTmpTrack->GetChannel(1);
                        std::shared_ptr< std::vector<float> > resampled_samples_right = std::make_shared< std::vector<float> >(audio_to_continue_samples);
                        bool bOkay = pResampledTrackR->GetFloats(resampled_samples_right->data(), start_s, audio_to_continue_samples);
                        if (!bOkay)
                        {
                           throw std::runtime_error("unable to get all right samples. GetFloats() failed for " +
                              std::to_string(audio_to_continue_samples) + "samples");
                        }

                        wav_pair.second = resampled_samples_right;
                     }
                     else
                     {
                        //we're setting both R & L channels to the same thing -- so divide by 2.
                        float* pResampled = resampled_samples_left->data();
                        for (size_t i = 0; i < resampled_samples_left->size(); i++)
                        {
                           pResampled[i] /= 2.f;
                        }
                        wav_pair.second = resampled_samples_left;
                     }
                  }

                  std::cout << "okay, set audio to continue to " << audio_to_continue_samples << " samples..." << std::endl;
               }

               params.audio_to_continue = wav_pair;
               params.bReturnAudioToContinueInOutput = _AudioContinuationAsNewTrackCheckBox->GetValue();
               audio_to_continue_params = params;
            }
         }

         mProgressFrac = 0.0;
         mProgMessage = "Running Music Generation";

         auto musicgen_pipeline_run_future = std::async(std::launch::async,
            [this, &seed, &audio_to_continue_params]
            {

               ov_musicgen::CallbackParams callback_params;
               callback_params.user = this;
               callback_params.every_n_new_tokens = 5;
               callback_params.callback = musicgen_callback;

               auto wav = _musicgen->Generate(_prompt,
                  audio_to_continue_params,
                  (float)mDurationT->GetValue(),
                  seed,
                  mGuidanceScale,
                  mTopK,
                  callback_params);

               return wav;
            }
            );

         do {
            using namespace std::chrono_literals;
            status = musicgen_pipeline_run_future.wait_for(0.5s);
            {
               std::lock_guard<std::mutex> guard(mProgMutex);
               mProgress->SetMessage(TranslatableString{ wxString(mProgMessage), {} });
               if (TotalProgress(mProgressFrac))
               {
                  mIsCancelled = true;
               }
            }

         } while (status != std::future_status::ready);

         auto res_wav_pair = musicgen_pipeline_run_future.get();
         auto generated_samples_L = res_wav_pair.first;
         if (!generated_samples_L)
            return false;

         auto generated_samples_R = res_wav_pair.second;

         auto duration = settings.extra.GetDuration();
         if (_AudioContinuationCheckBox->GetValue() && _AudioContinuationAsNewTrackCheckBox->GetValue())
         {
            duration += audio_to_continue_end - audio_to_continue_start;
         }

         settings.extra.SetDuration(duration);

         //clip samples to max duration
         size_t max_output_samples = duration * 32000;
         if (generated_samples_L)
         {
            std::cout << "Clipping Left from " << generated_samples_L->size() << " to " << max_output_samples << " samples." << std::endl;
            if (generated_samples_L->size() > max_output_samples)
            {
               generated_samples_L->resize(max_output_samples);
            }
         }
         else
         {
            std::cout << "No L samples" << std::endl;
            return false;
         }

         if (generated_samples_R)
         {
            std::cout << "Clipping Right from " << generated_samples_R->size() << " to " << max_output_samples << " samples." << std::endl;
            if (generated_samples_R->size() > max_output_samples)
            {
               generated_samples_R->resize(max_output_samples);
            }
         }

         bool bNormalized = false;
         for (auto track : outputs.Get().Selected<WaveTrack>())
         {
            bool editClipCanMove = GetEditClipsCanMove();

            //Don't normalize until we figure out how we want this to work
            // (as you need to keep in mind audio continuation, etc.).
#if 0
            if (!bNormalized)
            {
#define GEN_DB_TO_LINEAR(x) (pow(10.0, (x) / 20.0))
               float target_rms = GEN_DB_TO_LINEAR(std::clamp<double>(mRMSLevel, -145.0, 0.0));
               NormalizeSamples(generated_samples_L, track, target_rms);
               if (generated_samples_R)
               {
                  NormalizeSamples(generated_samples_R, track, target_rms);
               }
               bNormalized = true;
            }
#endif

            if (!_AudioContinuationAsNewTrackCheckBox->GetValue())
            {
               //if we can't move clips, and we're generating into an empty space,
               //make sure there's room.
               if (!editClipCanMove &&
                  track->IsEmpty(mT0, mT1 + 1.0 / track->GetRate()) &&
                  !track->IsEmpty(mT0,
                     mT0 + duration - (mT1 - mT0) - 1.0 / track->GetRate()))
               {
                  EffectUIServices::DoMessageBox(*this,
                     XO("There is not enough room available to generate the audio"),
                     wxICON_STOP,
                     XO("Error"));
                  return false;
               }
            }

            if (duration > 0.0)
            {
               auto pProject = FindProject();

               // Create a temporary track
               track->SetName(added_trackName);

               // create a temporary track list to append samples to
               auto tmp_tracklist = track->WideEmptyCopy();

               auto iter =
                  (*tmp_tracklist->Any<WaveTrack>().begin())->Channels().begin();

               if (track->NChannels() > 1)
               {
                  //append output samples to L & R channels.
                  auto& tmpLeft = **iter++;
                  tmpLeft.Append((samplePtr)generated_samples_L->data(), floatSample, generated_samples_L->size());

                  auto& tmpRight = **iter;

                  if (generated_samples_R)
                  {
                     tmpRight.Append((samplePtr)generated_samples_R->data(), floatSample, generated_samples_R->size());
                  }
                  else
                  {
                     tmpRight.Append((samplePtr)generated_samples_L->data(), floatSample, generated_samples_L->size());
                  }
               }
               else
               {
                  auto& tmpMono = **iter++;

                  //if we're populating a mono track, but we had stereo track selected which produced L & R. Convert to mono quick.
                  if (generated_samples_L && generated_samples_R)
                  {
                     Floats entire_input{ generated_samples_L->size() };
                     float* pL = generated_samples_L->data();
                     float* pR = generated_samples_R->data();
                     float* pMono = entire_input.get();
                     for (size_t i = 0; i < generated_samples_L->size(); i++)
                     {
                        pMono[i] = pL[i] + pR[i] / 2.f;
                     }

                     tmpMono.Append((samplePtr)pMono, floatSample, generated_samples_L->size());
                  }
                  else
                  {
                     tmpMono.Append((samplePtr)generated_samples_L->data(), floatSample, generated_samples_L->size());
                  }

               }

               //flush it
               auto pTmpTrack = *tmp_tracklist->Any<WaveTrack>().begin();
               pTmpTrack->Flush();

               // The track we just populated with samples is 32000 Hz.
               pTmpTrack->SetRate(32000);

               if (_AudioContinuationAsNewTrackCheckBox->GetValue())
               {
                  auto newOutputTrackList = track->WideEmptyCopy();

                  auto newOutputTrack = *newOutputTrackList->Any<WaveTrack>().begin();

                  newOutputTrack->SetRate(32000);

                  std::cout << "Pasting " << audio_to_continue_start << " : " << audio_to_continue_start + duration << std::endl;
                  // Clear & Paste into new output track
                  newOutputTrack->ClearAndPaste(audio_to_continue_start,
                     audio_to_continue_start + duration, *tmp_tracklist);

                  if (TracksBehaviorsSolo.ReadEnum() == SoloBehaviorSimple)
                  {
                     //If in 'simple' mode, if original track is solo,
                     // mute the new track and set it to *not* be solo.
                     if (newOutputTrack->GetSolo())
                     {
                        newOutputTrack->SetMute(true);
                        newOutputTrack->SetSolo(false);
                     }
                  }

                  newOutputTrack->Split(audio_to_continue_start, audio_to_continue_end);

                  //auto v = *newOutputTrackList;
                 // Add the new track to the output.
                  outputs.AddToOutputTracks(std::move(*newOutputTrackList));

                  // audio_to_continue_start may have been moved forward a bit, so update it.
                  //mT0 = audio_to_continue_start;

                  //in audio continuation case, original outputs size is exactly 1. We need to break out of
                  // the loop here, as the loop is iterating over outputs... this is ugly. Clean it up.
                  break;
               }
               else
               {
                  PasteTimeWarper warper{ mT1, mT0 + duration };
                  const auto& selectedRegion =
                     ViewInfo::Get(*pProject).selectedRegion;

                  std::cout << "Pasting " << selectedRegion.t0() << " : " << selectedRegion.t1() << std::endl;
                  track->ClearAndPaste(
                     selectedRegion.t0(), selectedRegion.t1(),
                     *tmp_tracklist, true, false, &warper);
               }

               if (!bGoodResult) {
                  return false;
               }
            }
            else
            {
               // If the duration is zero, there's no need to actually
               // generate anything
               track->Clear(mT0, mT1);
            }
         }

         if (mIsCancelled)
         {
            return false;
         }

         if (bGoodResult) {

            outputs.Commit();

            //mT1 = mT0 + duration; // Update selection.
         }

      }
      catch (const std::exception& error) {
         wxLogError("In Music Generation V2, exception: %s", error.what());
         EffectUIServices::DoMessageBox(*this,
            XO("Music Generation failed. See details in Help->Diagnostics->Show Log..."),
            wxICON_STOP,
            XO("Error"));
         return false;
      }

      std::cout << "returning!" << std::endl;
      return bGoodResult;
   }
}

bool EffectOVMusicGenerationV2::UpdateProgress(double perc)
{
   if (!TotalProgress(perc / 100.0))
   {
      std::cout << "Total Progress returned false" << std::endl;
      return false;
   }

   return true;
}

std::unique_ptr<EffectEditor> EffectOVMusicGenerationV2::PopulateOrExchange(
   ShuttleGui& S, EffectInstance&, EffectSettingsAccess& access,
   const EffectOutputs*)
{
   DoPopulateOrExchange(S, access);
   return nullptr;
}


void EffectOVMusicGenerationV2::DoPopulateOrExchange(
   ShuttleGui& S, EffectSettingsAccess& access)
{
   mUIParent = S.GetParent();

   //EnablePreview(false); //Port this



   S.StartVerticalLay(wxLEFT);
   {
      S.StartMultiColumn(3, wxLEFT);
      {
         mUnloadModelsButton = S.Id(ID_Type_UnloadModelsButton).AddButton(XO("Unload Models"));

         if (!_musicgen)
         {
            mUnloadModelsButton->Enable(false);
         }
      }
      S.EndMultiColumn();

      // Disable Normalization option until we figure out how we want this to work..
#if 0
      S.StartMultiColumn(4, wxLEFT);
      {
         S.AddVariableText(XO("Normalize "), false,
            wxALIGN_CENTER_VERTICAL | wxALIGN_LEFT);

         S.Name(XO("RMS dB"))
            .Validator<FloatingPointValidator<double>>(
               2, &mRMSLevel,
               NumValidatorStyle::ONE_TRAILING_ZERO,
               -145.0, 0.0)
            .AddTextBox({}, L"", 10);

         S.AddVariableText(XO("dB"), false,
            wxALIGN_CENTER_VERTICAL | wxALIGN_LEFT);
      }
      S.EndMultiColumn();
#endif


      S.StartMultiColumn(2, wxLEFT);
      {
         S.AddPrompt(XXO("&Duration:"));

         auto& extra = access.Get().extra;
         std::cout << "Creating prompt with duration = " << extra.GetDuration() << std::endl;

         mDurationT = safenew
            NumericTextCtrl(FormatterContext::SampleRateContext(mProjectRate),
               S.GetParent(), wxID_ANY,
               NumericConverterType_TIME(),
               extra.GetDurationFormat(),
               extra.GetDuration(),
               NumericTextCtrl::Options{}
         .AutoPos(true));
         S.Id(ID_Type_Duration).Name(XO("Duration"))
            .Position(wxALIGN_LEFT | wxALL)
            .AddWindow(mDurationT);

      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         mTypeChoiceModelSelection = S.Id(ID_Type_ModelSelection)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_modelSelectionChoice)
            .AddChoice(XXO("Model Selection:"),
               Msgids(mGuiModelSelections.data(), mGuiModelSelections.size()));


      }
      S.EndMultiColumn();

      S.StartMultiColumn(4, wxLEFT);
      {
         S.StartMultiColumn(2, wxCENTER);
         {
            mTextPrompt = S.Id(ID_Type_Prompt)
               .Style(wxTE_LEFT)
               .AddTextBox(XXO("Prompt:"), wxString(_prompt), 60);
         }
         S.EndMultiColumn();

         S.StartMultiColumn(2, wxCENTER);
         {

            mTypeChoiceDeviceCtrl_EnCodec = S.Id(ID_Type_EnCodec)
               .MinSize({ -1, -1 })
               .Validator<wxGenericValidator>(&m_deviceSelectionChoice_EnCodec)
               .AddChoice(XXO("EnCodec Device:"),
                  Msgids(mGuiDeviceNonVPUSupportedSelections.data(), mGuiDeviceNonVPUSupportedSelections.size()));

            mTypeChoiceDeviceCtrl_Decode0 = S.Id(ID_Type_MusicGenDecodeDevice0)
               .MinSize({ -1, -1 })
               .Validator<wxGenericValidator>(&m_deviceSelectionChoice_MusicGenDecode0)
               .AddChoice(XXO("MusicGen Decode Device:"),
                  Msgids(mGuiDeviceVPUSupportedSelections.data(), mGuiDeviceVPUSupportedSelections.size()));

            mTypeChoiceDeviceCtrl_Decode1 = S.Id(ID_Type_MusicGenDecodeDevice1)
               .MinSize({ -1, -1 })
               .Validator<wxGenericValidator>(&m_deviceSelectionChoice_MusicGenDecode1)
               .AddChoice(XXO("MusicGen Decode Device:"),
                  Msgids(mGuiDeviceVPUSupportedSelections.data(), mGuiDeviceVPUSupportedSelections.size()));

            //mTypeChoiceScheduler->Hide();
         }
         S.EndMultiColumn();
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         mSeed = S.Id(ID_Type_Seed)
            .Style(wxTE_LEFT)
            .AddNumericTextBox(XXO("Seed:"), wxString(_seed_str), 10);

         auto t0 = S.Name(XO("Guidance Scale"))
            .Validator<FloatingPointValidator<float>>(
               6, &mGuidanceScale,
               NumValidatorStyle::NO_TRAILING_ZEROES,
               0.0f,
               10.0f)
            .AddTextBox(XO("Guidance Scale"), L"", 12);

         mTopKCtl = S.Id(ID_Type_TopK)
            .Validator<IntegerValidator<int>>(&mTopK,
               NumValidatorStyle::DEFAULT,
               10,
               1000)
            .AddTextBox(XO("TopK"), L"", 12);
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         mTypeChoiceContextLength = S.Id(ID_Type_ContextLength)
            .MinSize({ -1, -1 })
            .Validator<wxGenericValidator>(&m_contextLengthChoice)
            .AddChoice(XXO("Context Length:"),
               Msgids(mGuiContextLengthSelections.data(), mGuiContextLengthSelections.size()));
      }
      S.EndMultiColumn();

      S.StartMultiColumn(2, wxLEFT);
      {
         _AudioContinuationCheckBox = S.Id(ID_Type_AudioContinuationCheckBox).AddCheckBox(XXO("&Audio Continuation"), false);
         _AudioContinuationAsNewTrackCheckBox = S.Id(ID_Type_AudioContinuationAsNewTrackCheckBox).AddCheckBox(XXO("&Audio Continuation on New Track"), false);

         EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };
         auto track_selection_size = outputs.Get().Selected<WaveTrack>().size();
         std::cout << "Track Selection Size = " << track_selection_size << std::endl;

         if (track_selection_size != 1)
         {
            _AudioContinuationCheckBox->Enable(false);
            _AudioContinuationAsNewTrackCheckBox->Enable(false);
         }
         else
         {
            auto track = *(outputs.Get().Selected<WaveTrack>().begin());

            auto t0 = track->GetStartTime();
            auto t1 = track->GetEndTime();

            std::cout << "track end time = " << t1 << std::endl;
            std::cout << "mT0 = " << mT0 << std::endl;
            if (track->IsEmpty(t0, t1))
            {
               _AudioContinuationCheckBox->Enable(false);
               _AudioContinuationAsNewTrackCheckBox->Enable(false);
            }
            else
            {
               // Not sure if I totally love this idea, but if a user's selection starts within 0.01 seconds
               // of the end of this track, let's just snap it exactly to the end as this is *probably* what
               // they intended?
               if (std::abs(mT0 - t1) <= 0.01)
               {
                  mT0 = t1;
               }

               //same with the end selection.
               if (std::abs(mT1 - t1) <= 0.01)
               {
                  mT1 = t1;
               }

               //if the start position of the user's track selection is within 0.1 seconds of the end time, we can
               // make a pretty safe assumption that their intention is to perform music continuation.
               if (((mT0 - t1) >= 0.) && ((mT0 - t1) <= 0.1))
               {
                  _AudioContinuationCheckBox->SetValue(true);
               }
               else
               {
                  // If an existing (non-empty) track is highlighted, and the highlighted region
                  // overlaps with existing segment.
                  if (((mT0 >= t0) && (mT0 < t1)) || ((mT1 >= t0) && (mT1 < t1)))
                  {
                     double duration = 0;
                     GetConfig(GetDefinition(), PluginSettings::Private,
                        CurrentSettingsGroup(),
                        EffectSettingsExtra::DurationKey(), duration, 30.);

                     if (duration > 0)
                     {
                        std::cout << "Setting last used duration of " << duration << std::endl;
                        mDurationT->SetValue(duration);
                     }

                     _AudioContinuationCheckBox->SetValue(true);
                     _AudioContinuationAsNewTrackCheckBox->SetValue(true);
                  }
               }
            }
         }
      }
      S.EndMultiColumn();



      S.StartMultiColumn(2, wxLEFT);
      {
         _continuationContextWarning = S.AddVariableText(XO("Some default text"), false,
            wxALIGN_CENTER_VERTICAL | wxALIGN_LEFT);

         _continuationContextWarning->SetFont(_continuationContextWarning->GetFont().Scale(1.5));

         SetContinuationContextWarning();

      }
      S.EndMultiColumn();

      }

   S.EndVerticalLay();

   }

void EffectOVMusicGenerationV2::SetContinuationContextWarning()
{
   if (!_AudioContinuationCheckBox->GetValue())
   {
      _AudioContinuationAsNewTrackCheckBox->SetValue(false);
      _AudioContinuationAsNewTrackCheckBox->Enable(false);
      _continuationContextWarning->Hide();
      return;
   }
   else
   {
      if (!_AudioContinuationAsNewTrackCheckBox->IsEnabled())
      {
         _AudioContinuationAsNewTrackCheckBox->Enable(true);
      }
   }

   EffectOutputTracks outputs{ *mTracks, GetType(), {{ mT0, mT1 }} };
   auto track = *(outputs.Get().Selected<WaveTrack>().begin());

   auto t0 = track->GetStartTime();
   auto t1 = track->GetEndTime();
   if (((mT0 >= t0) && (mT0 < t1)) || ((mT1 >= t0) && (mT1 < t1)))
   {
      //std::string warning_message =
      //

      double current_selection_size = mT1 - mT0;
      std::cout << "mTypeChoiceContextLength->GetSelection() = " << mTypeChoiceContextLength->GetCurrentSelection() << std::endl;

      int context_length_choice = mTypeChoiceContextLength->GetCurrentSelection() == -1 ? m_contextLengthChoice : mTypeChoiceContextLength->GetCurrentSelection();
      double context_selection_size = (context_length_choice == 1) ? 10. : 5;

      if (current_selection_size > context_selection_size)
      {
         std::string warning_message = "Note: Given the 'Context Length' chosen above, only the last " + std::to_string((int)context_selection_size) + " seconds of your selection will be used as continuation context.";
         std::cout << "Setting message to: " << warning_message << std::endl;
         auto warn_msg = wxString(warning_message);
         _continuationContextWarning->SetLabelText(warn_msg);
      }
      else
      {
         std::string warning_message = "Note: The selected audio segment of " + std::to_string(current_selection_size) + " seconds will be used as continuation context.";
         auto warn_msg = wxString(warning_message);
         _continuationContextWarning->SetLabelText(warn_msg);
      }

   }
   else
   {
      int context_selection_size = (m_contextLengthChoice == 0) ? 5 : 10;
      std::string warning_message = "Note: Given the chosen 'Context Length' above, the previous " + std::to_string(context_selection_size) + " seconds prior to your selection will be used as continuation context.";
      auto warn_msg = wxString(warning_message);
      _continuationContextWarning->SetLabelText(warn_msg);
   }

   _continuationContextWarning->Hide();
   _continuationContextWarning->Show();
}

void EffectOVMusicGenerationV2::OnContextLengthChanged(wxCommandEvent& evt)
{
   std::cout << "OnContextLengthChanged called" << std::endl;
   SetContinuationContextWarning();
}

void EffectOVMusicGenerationV2::OnUnloadModelsButtonClicked(wxCommandEvent& evt)
{
   _musicgen = {};

   if (mUnloadModelsButton)
   {
      mUnloadModelsButton->Enable(false);
   }
}

bool EffectOVMusicGenerationV2::TransferDataToWindow(const EffectSettings& settings)
{
   std::cout << "TransferDataToWindow. settings.extra.GetDuration() = " << settings.extra.GetDuration() << std::endl;

   if (!mUIParent->TransferDataToWindow())
   {
      return false;
   }

   EffectEditor::EnablePreview(mUIParent, false);

   if (mDurationT)
   {
      std::cout << "EffectOVMusicGenerationV2::TransferDataToWindow: Setting mDurationT to " << settings.extra.GetDuration() << std::endl;
      //mDurationT->SetValue(settings.extra.GetDuration());
   }

   return true;
}

bool EffectOVMusicGenerationV2::TransferDataFromWindow(EffectSettings& settings)
{
   std::cout << "TransferDataFromWindow. settings.extra.GetDuration() = " << settings.extra.GetDuration() << std::endl;
   if (!mUIParent->Validate() || !mUIParent->TransferDataFromWindow())
   {
      return false;
   }

   if (mDurationT)
   {
      std::cout << "EffectOVMusicGenerationV2::TransferDataFromWindow: Setting settings.extra.SetDuration to " << mDurationT->GetValue() << std::endl;
      settings.extra.SetDuration(mDurationT->GetValue());
   }

   return true;
}
