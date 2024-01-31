#pragma once
#include <optional>
#include <vector>
#include <string>
#include <memory>
#include "musicgen_config.h"

class MusicGen
{
public:

	MusicGen(MusicGenConfig &config);


	std::shared_ptr<std::vector<float>> Generate(std::optional<std::string> prompt,
		std::optional<std::shared_ptr<std::vector<float>>> audio_to_continue,
		float total_desired_length_seconds,
		std::optional< unsigned int > seed,
		float guidance_scale = 3.f,
		float top_k = 250,
      std::optional< CallbackParams > callback_params = {});

private:

	struct Impl;
	std::shared_ptr<Impl> _impl;

	MusicGenConfig _config;
};