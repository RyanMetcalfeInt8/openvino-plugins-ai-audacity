#pragma once

#include <torch/torch.h>
#include <openvino/openvino.hpp>
#include <optional>

#include "multiframe.h"

namespace ov_deepfilternet3
{
	// Ported from pyTorch implementation here:
	// https://github.com/grazder/DeepFilterNet/blob/7fc312645d1c468f8a910acb9d0acb3b77637ed7/torchDF/torch_df_offline.py#L9
	// thank you @grazder for the pytorch implementation!
	class DeepFilter
	{
	public:

		DeepFilter(std::string model_folder, std::string device, int64_t sr = 48000, int64_t fft_size = 960, int64_t hop_size = 480, int64_t nb_bands = 32, int64_t min_nb_freqs = 2, int64_t nb_df = 96, double alpha = 0.99);

      typedef bool (*ProgressCallbackFunc)(float perc_complete, //<- range 0 to 1
         void* user);

      std::shared_ptr<std::vector<float>> filter(torch::Tensor noisy_audio, std::optional<float> atten_lim_db = {},
         float normalize_atten_lim = 20,
         ProgressCallbackFunc callback = nullptr, void *callback_user = nullptr);
		

	private:

      std::shared_ptr<std::vector<float>> forward(torch::Tensor noisy_audio, bool pad = true, std::optional<float> atten_lim_db = {}, float normalize_atten_lim = 20);
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _df_features(torch::Tensor audio);
		torch::Tensor _analysis_time(torch::Tensor input_data);
		torch::Tensor _frame_analysis(torch::Tensor input_frame);
		torch::Tensor _erb_norm_time(torch::Tensor input_data, float alpha = 0.9);
		torch::Tensor _band_mean_norm_erb(torch::Tensor xs, float alpha, float denominator = 40.0);
		torch::Tensor _erb(const torch::Tensor& input_data, bool db = true);
		torch::Tensor _unit_norm_time(torch::Tensor input_data, float alpha = 0.9);
		torch::Tensor _band_unit_norm(torch::Tensor xs, float alpha);
		torch::Tensor _synthesis_time(torch::Tensor input_data);
		torch::Tensor _frame_synthesis(torch::Tensor input_data);

		int64_t _fft_size;
		int64_t _frame_size;
		int64_t _window_size;
		int64_t _window_size_h;
		int64_t _freq_size;
		double _wnorm;
		int64_t _sr;
		int64_t _min_nb_freqs;
		int64_t _nb_df;
		int64_t _n_erb_features;
		double _alpha;

		std::vector< double > _mean_norm_init;
		std::vector< double > _unit_norm_init;

		torch::Tensor _window;
		torch::Tensor _erb_indices;


		std::shared_ptr<torch::nn::Conv1d> _erb_conv;

		//registered buffers
		torch::Tensor _reg_window;
		torch::Tensor _reg_analysis_mem;
		torch::Tensor _reg_synthesis_mem;
		torch::Tensor _reg_band_unit_norm_state;
		torch::Tensor _reg_erb_norm_state;

		ov::InferRequest _infer_request;

		DF _df;
	};
}
