#include "deepfilter.h"
#include "musicgen_utils.h"

namespace ov_deepfilternet3
{
	DeepFilter::DeepFilter(std::string model_folder, std::string device, int64_t sr, int64_t fft_size, int64_t hop_size, int64_t nb_bands, int64_t min_nb_freqs, int64_t nb_df, double alpha)
		: _df(nb_df, 5, 2)
	{
		_fft_size = fft_size;
		_frame_size = hop_size;
		_window_size = fft_size;
		_window_size_h = fft_size / 2;
		_freq_size = fft_size / 2 + 1;
		_wnorm = 1.f / ((_window_size * _window_size) / (2 * _frame_size));

		// Initialize the vorbis window: sin(pi/2*sin^2(pi*n/N))
		_reg_window = torch::zeros({ _fft_size });
		_window = torch::sin(0.5 * M_PI * (torch::arange(_fft_size) + 0.5) / _window_size_h);
		_window = torch::sin(0.5 * M_PI * (_window * _window));

		_sr = sr;
		_min_nb_freqs = min_nb_freqs;
		_nb_df = nb_df;

		// Initializing erb features
		_erb_indices = torch::tensor({
		2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8,
		10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67
			}, torch::dtype(torch::kInt64));
		_n_erb_features = nb_bands;

		// Create the convolutional layer
		torch::nn::Conv1dOptions conv_options(_freq_size, _n_erb_features, 1);
		conv_options.bias(false); // Set bias to false
		_erb_conv = std::make_shared<torch::nn::Conv1d>(conv_options);

		// Set requires_grad to false for all parameters
		for (auto& param : (*_erb_conv)->named_parameters()) {
			param.value().set_requires_grad(false);
		}

		// Initialize out_weight tensor
		auto out_weight = torch::zeros({ _n_erb_features, _freq_size, 1 });

		// Update out_weight based on erb_indices
		int64_t start_index = 0;
		for (int64_t i = 0; i < _erb_indices.size(0); ++i) {
			int64_t num = _erb_indices[i].item<int64_t>();
			out_weight.index({ i, torch::indexing::Slice(start_index, start_index + num), 0 }) = 1.0 / static_cast<double>(num);
			start_index += num;
		}

		// Copy out_weight to erb_conv's weight
		(*_erb_conv)->weight.copy_(out_weight);

		_mean_norm_init = { -60., -90. };
		_unit_norm_init = { 0.001, 0.0001 };

		_alpha = alpha;

		// Init buffers
		_reg_analysis_mem = torch::zeros(_fft_size - _frame_size);
		_reg_synthesis_mem = torch::zeros(_fft_size - _frame_size);
		_reg_band_unit_norm_state = torch::linspace(_unit_norm_init[0], _unit_norm_init[1], _nb_df);
		_reg_erb_norm_state = torch::linspace(_mean_norm_init[0], _mean_norm_init[1], _n_erb_features);


		{
			ov::Core core;
			auto model_fullpath = FullPath(model_folder, "deepfilternet3.xml");
			auto model = core.read_model(model_fullpath);
			auto compiledModel = core.compile_model(model, device);
			_infer_request = compiledModel.create_infer_request();

		}

	}

	std::shared_ptr<std::vector<float>> DeepFilter::filter(torch::Tensor noisy_audio, std::optional<float> atten_lim_db, float normalize_atten_lim, ProgressCallbackFunc callback, void* callback_user)
	{
		// the number of samples that we will *always* pass into forward (10s of audio)
		const int64_t chunk_size = 480000;

		//take the mean first, for the entire audio segment.
		noisy_audio = noisy_audio.mean(0);


		int64_t samples_left_to_process = noisy_audio.size(0);
      int64_t total_samples = samples_left_to_process;

		//auto chunk_tensor = torch::zeros({ 1, chunk_size });

		std::shared_ptr< std::vector<float> > output_wav = std::make_shared < std::vector<float> >();

		int64_t samples_processed = 0;
		while (samples_left_to_process > 0)
		{
			int64_t valid_samples = std::min(samples_left_to_process, chunk_size);

			using namespace torch::indexing;


			torch::Tensor chunk_tensor;
			if (valid_samples < chunk_size)
			{
				chunk_tensor = torch::zeros({ chunk_size });
				chunk_tensor.index_put_({ Slice(0, valid_samples) }, noisy_audio.index({ Slice(samples_processed, samples_processed + valid_samples) }));
			}
			else
			{
				chunk_tensor = noisy_audio.index({ Slice(samples_processed, samples_processed + chunk_size) });
			}

			auto wav = forward(chunk_tensor, true, atten_lim_db, normalize_atten_lim);
			output_wav->insert(output_wav->end(), wav->begin(), wav->begin() + valid_samples);


			samples_left_to_process -= valid_samples;
			samples_processed += valid_samples;

         if (callback)
         {
            float perc_complete = (float)samples_processed / (float)total_samples;
            if (callback(perc_complete, callback_user))
            {
               std::cout << "callback returned true -- returning {}" << std::endl;
               return {};
            }
         }
		}


		return output_wav;
	}

	std::shared_ptr<std::vector<float>> DeepFilter::forward(torch::Tensor noisy_audio, bool pad, std::optional<float> atten_lim_db, float normalize_atten_lim)
	{
		auto audio = noisy_audio;
		auto orig_len = audio.sizes().back();

		if (pad)
		{
			// Pad audio to compensate for the delay due to the real-time STFT implementation
			auto hop_size_divisible_padding_size = (_fft_size - orig_len % _fft_size) % _fft_size;
			orig_len += hop_size_divisible_padding_size;

			audio = torch::nn::functional::pad(audio, torch::nn::functional::PadFuncOptions({ 0, 2 * _fft_size + hop_size_divisible_padding_size }));

		}

		auto df_ret = _df_features(audio);
		auto spec = std::get<0>(df_ret);
		auto erb_feat = std::get<1>(df_ret);
		auto spec_feat = std::get<2>(df_ret);

		//enhanced = self.model(spec, erb_feat, spec_feat)[0] # [B = 1, CH = 1, T, F, 2]

		//_infer_request
		spec = spec.contiguous();
		erb_feat = erb_feat.contiguous();
		spec_feat = spec_feat.contiguous();

		auto in_spec_ov = wrap_ov_tensor_as_torch(_infer_request.get_tensor("in_spec"));
		auto in_feat_erb_ov = wrap_ov_tensor_as_torch(_infer_request.get_tensor("in_feat_erb"));
		auto in_feat_spec_ov = wrap_ov_tensor_as_torch(_infer_request.get_tensor("in_feat_spec"));

		in_spec_ov.copy_(spec);
		in_feat_erb_ov.copy_(erb_feat);
		in_feat_spec_ov.copy_(spec_feat);

		_infer_request.infer();

		auto spec_ret = wrap_ov_tensor_as_torch(_infer_request.get_tensor("out_spec"));
		auto df_coefs_ret = wrap_ov_tensor_as_torch(_infer_request.get_tensor("out_df_coefs"));
		auto spec_m_ret = wrap_ov_tensor_as_torch(_infer_request.get_tensor("out_spec_m"));

		//dump_tensor(spec_ret, "ov_spec_ret.raw");
		//dump_tensor(df_coefs_ret, "ov_df_coefs_ret.raw");
		//dump_tensor(spec_m_ret, "ov_spec_m_ret.raw");
      auto spec_orig = spec;
		spec = _df.forward(spec_ret, df_coefs_ret);
		//dump_tensor(spec, "ov_spec_after_df_op.raw");
		using namespace torch::indexing;
		spec.index_put_({ "...", Slice(_nb_df, None),  Slice(None) }, spec_m_ret.index({ "...", Slice(_nb_df, None),  Slice(None) }));

		auto enhanced = spec;

		if (atten_lim_db)
		{
			float lim = std::powf(10, (-std::abs(*atten_lim_db) / normalize_atten_lim));
			enhanced = torch::lerp(enhanced, spec_orig, lim);
		}

		{
			std::vector< int64_t > view_shape;
			for (size_t i = 2; i < enhanced.sizes().size(); i++)
				view_shape.push_back(enhanced.size(i));
			enhanced = torch::view_as_complex(enhanced.view(view_shape));
		}

		audio = _synthesis_time(enhanced).unsqueeze(0);

		if (pad)
		{
			auto d = _fft_size + _frame_size;
			audio = audio.index({ Slice(None), Slice(d, orig_len + d) });
		}

		std::shared_ptr<std::vector<float>> wav = std::make_shared< std::vector<float> >(audio.size(1));
		std::memcpy(wav->data(), audio.data_ptr<float>(), audio.size(1) * sizeof(float));
		return wav;
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> DeepFilter::_df_features(torch::Tensor audio)
	{
		auto spec = _analysis_time(audio);

		auto erb_feat = _erb_norm_time(_erb(spec), _alpha);

		using namespace torch::indexing;
		auto spec_feat = torch::view_as_real(_unit_norm_time(spec.index({ "...", Slice(None, _nb_df) }), _alpha));
		//dump_tensor(spec_feat, "ov_spec_feat.raw");
		spec = torch::view_as_real(spec);
		//dump_tensor(spec, "ov_spec.raw");

		auto spec_ret = spec.unsqueeze(0).unsqueeze(0);
		auto erb_feat_ret = erb_feat.unsqueeze(0).unsqueeze(0);
		auto spec_feat_ret = spec_feat.unsqueeze(0).unsqueeze(0);

		return { spec_ret , erb_feat_ret, spec_feat_ret };
	}

	torch::Tensor DeepFilter::_analysis_time(torch::Tensor input_data)
	{
		auto in_chunks = torch::split(input_data, _frame_size);

		// time chunks iteration
		std::vector< torch::Tensor > output;
		for (auto ichunk : in_chunks)
		{
			output.push_back(_frame_analysis(ichunk));
		}

		return torch::stack(output, 0);
	}

	torch::Tensor DeepFilter::_frame_analysis(torch::Tensor input_frame)
	{
		//First part of the window on the previous frame
		//Second part of the window on the new input frame
		auto buf = torch::cat({ _reg_analysis_mem, input_frame }) * _window;
		auto buf_fft = torch::fft::rfft(buf, {}, -1, "backward") * _wnorm;

		// Copy input to analysis_mem for next iteration
		_reg_analysis_mem = input_frame;

		return buf_fft;
	}

	torch::Tensor DeepFilter::_erb_norm_time(torch::Tensor input_data, float alpha)
	{
		std::vector<torch::Tensor> output;

		for (int64_t i = 0; i < input_data.size(0); ++i) {
			torch::Tensor in_step = input_data[i];
			output.push_back(_band_mean_norm_erb(in_step, alpha));
		}

		return torch::stack(output, 0);
	}

	torch::Tensor DeepFilter::_band_mean_norm_erb(torch::Tensor xs, float alpha, float denominator)
	{
		_reg_erb_norm_state = torch::lerp(xs, _reg_erb_norm_state, alpha);
		auto output = (xs - _reg_erb_norm_state) / denominator;
		return output;
	}

	torch::Tensor DeepFilter::_erb(const torch::Tensor& input_data, bool db) {
		torch::Tensor magnitude_squared = torch::real(input_data).pow(2) + torch::imag(input_data).pow(2);
		torch::Tensor erb_features = (*_erb_conv)->forward(magnitude_squared.unsqueeze(-1)).squeeze(-1);

		if (db) {
			erb_features = 10.0 * torch::log10(erb_features + 1e-10);
		}

		return erb_features;
	}

	torch::Tensor DeepFilter::_unit_norm_time(torch::Tensor input_data, float alpha)
	{
		std::vector<torch::Tensor> output;

		for (int64_t i = 0; i < input_data.size(0); ++i) {
			torch::Tensor in_step = input_data[i];
			output.push_back(_band_unit_norm(in_step, alpha));
		}
		return torch::stack(output, 0);
	}

	torch::Tensor DeepFilter::_band_unit_norm(torch::Tensor xs, float alpha)
	{
		_reg_band_unit_norm_state = torch::lerp(xs.abs(), _reg_band_unit_norm_state, alpha);
		auto output = xs / _reg_band_unit_norm_state.sqrt();

		return output;
	}

	torch::Tensor DeepFilter::_synthesis_time(torch::Tensor input_data)
	{
		std::vector<torch::Tensor> out_chunks;
		for (int64_t i = 0; i < input_data.size(0); ++i) {
			torch::Tensor ichunk = input_data[i];
			auto output_frame = _frame_synthesis(ichunk);
			out_chunks.push_back(output_frame);
		}

		return torch::cat(out_chunks);
	}

	torch::Tensor DeepFilter::_frame_synthesis(torch::Tensor input_data)
	{
		auto x = torch::fft::irfft(input_data, {}, -1, "forward") * _window;

		auto split_ret = torch::split(x, { _frame_size, x.size(0) - _frame_size });
		auto x_first = split_ret[0];
		auto x_second = split_ret[1];

		auto output = x_first + _reg_synthesis_mem;

		_reg_synthesis_mem = x_second;

		return output;
	}

}
