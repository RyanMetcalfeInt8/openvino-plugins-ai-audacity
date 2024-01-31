#include "musicgen.h"
#include <ittutils.h>
#include "musicgen_for_conditional_generation.h"
#include <sentencepiece_processor.h>

struct MusicGen::Impl
{
	Impl(MusicGenConfig& config)
	{
		_gen = std::make_shared< MusicgenForConditionalGeneration >(config);
		
		auto tokenizer_model_file = FullPath(config.model_folder, "musicgen_small_spiece.model");
		const auto status = processor.Load(tokenizer_model_file);
		if (!status.ok()) {
			throw std::runtime_error("Error loading sentencepiece model file. Error = " + status.ToString());
		}
	}

	sentencepiece::SentencePieceProcessor processor;
	std::shared_ptr< MusicgenForConditionalGeneration > _gen;
};

MusicGen::MusicGen(MusicGenConfig& config)
{
	_config = config;

	_impl = std::make_shared< Impl >(_config);
}

std::shared_ptr<std::vector<float>> MusicGen::Generate(std::optional<std::string> prompt,
	std::optional<std::shared_ptr<std::vector<float>>> audio_to_continue,
	float total_desired_length_seconds,
	std::optional< unsigned int > seed,
	float guidance_scale,
	float top_k,
   std::optional< CallbackParams > callback_params)
{
	if (!prompt)
	{
		throw std::runtime_error("Prompt is required (right now)");
	}

	if (seed)
	{
		std::cout << "Setting seed of " << *seed << std::endl;
		_impl->_gen->SetSeed(*seed);
	}
	else
	{
		std::cout << "Seed not set. Defaulting to 1 " << std::endl;
	}

	//tokenize
	std::vector<int> ids;
	_impl->processor.Encode(*prompt, &ids);
	std::vector<int64_t> ids_64;
	for (auto id : ids)
		ids_64.push_back(id);
	ids_64.push_back(1);

	if(ids_64.size() > 64)
	{
		throw std::runtime_error("Given prompt is too long (it cannot exceed 64 tokens after tokenization)");
	}

	torch::Tensor input_tensor = torch::from_blob(ids_64.data(), { 1, (int64_t)ids_64.size() }, torch::kInt64);
	torch::Tensor attention_mask = torch::cat({ torch::ones(input_tensor.sizes()),  torch::zeros(input_tensor.sizes()) });

	//generare will +4 to the tokens, so we need to subtract it here (yuck)
	const int64_t max_new_tokens_per_generate = std::max((int64_t)1000, _impl->_gen->MaxNewTokens() - 4);
	//const int64_t max_new_tokens_per_generate = 1000;

	// 50 samples / sec
	int64_t total_tokens_left_to_generate = (int64_t)(std::ceilf(total_desired_length_seconds * 50));

   MusicgenForConditionalGeneration::CallbackTracking tracking;
   tracking.total_tokens_generated_so_far = 0;
   tracking.total_tokens_to_generate = total_tokens_left_to_generate;

	std::cout << "total_tokens_left_to_generate = " << total_tokens_left_to_generate << std::endl;

	int64_t ncontext_samples;
	int64_t ncontext_tokens;
	switch (_config.m_continuation_context)
	{
	    case MusicGenConfig::ContinuationContext::FIVE_SECONDS:
		ncontext_samples = 160000;
		ncontext_tokens = 250;
		break;

		case MusicGenConfig::ContinuationContext::TEN_SECONDS:
		ncontext_samples = 320000;
		ncontext_tokens = 500;
		break;
	}
	
	std::optional< torch::Tensor > audio_to_continue_tensor;

	if (audio_to_continue)
	{
		audio_to_continue_tensor = torch::zeros({ 1, 1, ncontext_samples });
		auto atc = *audio_to_continue;

		int64_t atc_num_samples = atc->size();
		int64_t tensor_offset = 0;
		int64_t atc_offset = 0;
		int64_t size_to_copy = ncontext_samples;

		if (atc_num_samples < ncontext_samples)
		{
			tensor_offset = ncontext_samples - atc->size();
			size_to_copy = atc_num_samples;
		}
		else if (atc_num_samples > ncontext_samples)
		{
			atc_offset = atc_num_samples - ncontext_samples;
			size_to_copy = ncontext_samples;
		}

		memcpy((float*)(audio_to_continue_tensor->data_ptr()) + tensor_offset,
			atc->data() + atc_offset, size_to_copy * sizeof(float));
	}

   std::shared_ptr< std::vector<float> > output_wav;

   //this will cause us to *not* copy the re-encoded audio that was
   // passed in (although the user may want that in some mode).
   if (audio_to_continue)
   {
      output_wav = std::make_shared<std::vector<float>>();
   }
  
	size_t iterationi = 0;
	while (total_tokens_left_to_generate > 0)
	{
		int64_t max_new_tokens_we_can_generate_this_it = max_new_tokens_per_generate;
		int64_t context_tokens_this_it = 0;

		//if we are passing in some audio_to_continue, the max_length_this_iteration needs to include that.
		if (audio_to_continue_tensor)
		{
			max_new_tokens_we_can_generate_this_it -= ncontext_tokens;
			context_tokens_this_it = ncontext_tokens;
		}

		int64_t max_length_this_iteration = std::min(max_new_tokens_we_can_generate_this_it,
			total_tokens_left_to_generate);


		std::cout << "max_length_this_iteration = " << max_length_this_iteration << std::endl;
		std::cout << "context_tokens_this_it = " << context_tokens_this_it << std::endl;
		auto gen_ret = _impl->_gen->generate(input_tensor, max_length_this_iteration + context_tokens_this_it, attention_mask, tracking, audio_to_continue_tensor, {}, guidance_scale, top_k,
                                           callback_params);

		if (!output_wav)
		{
			output_wav = gen_ret.wav;
		}
		else
		{
			auto wav = gen_ret.wav;

			//todo: crossfade between the start of the new clip and the end of the old clip.
			// Right now there are sometimes 'pops' or 'clicks' introduced at the point where
			// we merge these together.
			output_wav->insert(output_wav->end(), wav->begin() + ncontext_samples, wav->end());
		}

		total_tokens_left_to_generate -= max_length_this_iteration;

		if (total_tokens_left_to_generate > 0)
		{
			audio_to_continue_tensor = torch::from_blob(output_wav->data() + (output_wav->size() - ncontext_samples), { 1, 1, ncontext_samples });
		}

		iterationi++;
	}

#if 0
	//generate first snippet
	auto gen_ret = _impl->_gen->generate(input_tensor, total_tokens_left_to_generate, attention_mask, {}, {}, guidance_scale, top_k);

	auto wav = gen_ret.wav;

	torch::Tensor wav_tensor = torch::from_blob(wav->data() + (wav->size() - ncontext_samples), { 1, 1, ncontext_samples });


	for (int i = 0; i < 0; i++)
	{
		std::cout << i << ": generate" << std::endl;

		std::cout << "wav_tensor shape = " << wav_tensor.sizes() << std::endl;
		gen_ret = _impl->_gen->generate(input_tensor, attention_mask, wav_tensor);

		auto wav1 = gen_ret.wav;

		wav->insert(wav->end(), wav1->begin() + ncontext_samples, wav1->end());

		wav_tensor = torch::from_blob(wav->data() + (wav->size() - ncontext_samples), { 1, 1, ncontext_samples });
	}
#endif

	return output_wav;
}