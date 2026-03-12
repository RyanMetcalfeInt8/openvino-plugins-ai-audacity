// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include <demix/apollo.h>
#include "utils/openvino_utils.h"

namespace ov_demix
{

   static const char* status_to_cstr(ov::ProfilingInfo::Status s) {
      using S = ov::ProfilingInfo::Status;
      switch (s) {
      case S::NOT_RUN:       return "NOT_RUN";
      case S::OPTIMIZED_OUT: return "OPTIMIZED_OUT";
      case S::EXECUTED:      return "EXECUTED";
      default:               return "UNKNOWN";
      }
   }

   static inline double us_to_ms(const std::chrono::microseconds& us) {
      return static_cast<double>(us.count()) / 1000.0;
   }

   static void print_perf_counts(const ov::InferRequest& req) {
      std::vector<ov::ProfilingInfo> prof = req.get_profiling_info();

      std::sort(prof.begin(), prof.end(),
         [](const ov::ProfilingInfo& a, const ov::ProfilingInfo& b) {
            return a.real_time.count() > b.real_time.count();
         });

      // Header
      std::cout << std::left
         << std::setw(55) << "node_name"
         << "  " << std::setw(18) << "node_type"
         << "  " << std::setw(20) << "exec_type"
         << "  " << std::right << std::setw(10) << "real_ms"
         << "  " << std::setw(10) << "cpu_ms"
         << "  " << std::left << "status"
         << "\n";

      for (const auto& p : prof) {
         const double real_ms = us_to_ms(p.real_time);
         const double cpu_ms = us_to_ms(p.cpu_time);

         std::cout << std::left
            << std::setw(55) << p.node_name
            << "  " << std::setw(18) << p.node_type
            << "  " << std::setw(20) << p.exec_type
            << "  " << std::right << std::setw(10) << std::fixed << std::setprecision(3) << real_ms
            << "  " << std::setw(10) << std::fixed << std::setprecision(3) << cpu_ms
            << "  " << std::left << status_to_cstr(p.status)
            << "\n";
      }
   }
 
    Apollo::~Apollo() {
       print_perf_counts(_pre_ir);
    }

    Apollo::Apollo(const std::string& model_dir,
        const std::string& device,
        const std::string& cache_dir)
    {
        ov::Core core;

        {
            {
                auto xml_path = FullPath(model_dir, "apollo_pre.xml");
                auto model = core.read_model(xml_path);
                std::cout << "PRE:" << std::endl;
                logBasicModelInfo(model);

                auto properties = ov::AnyMap{ {"PERF_COUNT", "YES"} };
                auto compiled_model = core.compile_model(model, "CPU", properties);
                _pre_ir = compiled_model.create_infer_request();
            }

            {
                auto xml_path = FullPath(model_dir, "apollo_fwd.xml");
                auto model = core.read_model(xml_path);
                std::cout << "FWD:" << std::endl;
                logBasicModelInfo(model);

                ov::AnyMap properties = { ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY) };

                if (!cache_dir.empty())
                {
                   properties.insert(ov::cache_dir(cache_dir));
                }

                auto compiled_model = core.compile_model(model, device, properties);
                _forward_ir = compiled_model.create_infer_request();
            }

            {
                auto xml_path = FullPath(model_dir, "apollo_post.xml");
                auto model = core.read_model(xml_path);
                std::cout << "POST:" << std::endl;
                logBasicModelInfo(model);

                auto compiled_model = core.compile_model(model, "CPU");
                _post_ir = compiled_model.create_infer_request();
            }

            //connect pre -> fwd -> post
            _forward_ir.set_input_tensor(_pre_ir.get_output_tensor());
            _post_ir.set_input_tensor(_forward_ir.get_output_tensor());

            auto out_shape = _forward_ir.get_output_tensor().get_shape();
            if (out_shape.size() != 4)
            {
                throw std::runtime_error("Apollo: Expected output tensor to have rank 4, but it is "
                    + std::to_string(out_shape.size()));
            }

            _hop_len = out_shape[2] - 1;
            _win_len = _hop_len * 2;

            int64_t num_frames = static_cast<int64_t>(out_shape[3]);
            _chunk_len = (num_frames - 1) * _hop_len - 2 * (_win_len / 2) + _win_len;

            std::cout << "Apollo: _hop_len = " << _hop_len << std::endl;
            std::cout << "Apollo: _win_len = " << _win_len << std::endl;
            std::cout << "Apollo: _chunk_len = " << _chunk_len << std::endl;

            _window = torch::hann_window(_win_len, torch::TensorOptions().dtype(torch::kFloat32));
        }
    }

    torch::Tensor Apollo::run(torch::Tensor arr)
    {
       using namespace std::chrono;
       using Clock = std::chrono::high_resolution_clock;
#if 0
        uint64_t pre_t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        auto arr_sizes = arr.sizes();
        TORCH_CHECK(arr_sizes.size() == 3, "Expected input of shape [B, nch, nsample]");
        int64_t B = arr_sizes[0];
        int64_t nch = arr_sizes[1];
        int64_t nsample = arr_sizes[2];
        int64_t win_len = _win_len;
        int64_t hop_len = _hop_len;

        torch::Tensor spec;
        {
            torch::Tensor input_reshaped = arr.view({ B * nch, nsample });

            spec = torch::stft(input_reshaped,
                /*n_fft=*/win_len,
                /*hop_length=*/hop_len,
                /*win_length=*/win_len,
                /*window=*/_window,
                /*center=*/true,
                /*pad_mode=*/"reflect",
                /*normalized=*/false,
                /*onesided=*/true,
                /*return_complex=*/true);

            torch::Tensor spec_real = torch::view_as_real(spec);  // shape [B*nch, F, T, 2]
            spec = spec_real.contiguous();
        }
        uint64_t pre_t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        std::cout << "pre time = " << pre_t1 - pre_t0 << std::endl;

        _forward_ir.set_input_tensor(wrap_torch_tensor_as_ov(spec));

        {
           uint64_t t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
           _forward_ir.infer();
           uint64_t t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

           std::cout << "fwd time = " << t1 - t0 << std::endl;
        }

        uint64_t post_t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        auto est_spec = wrap_ov_tensor_as_torch(_forward_ir.get_output_tensor());

        // est_spec shape: [B*nch, 2, F, T]
        // Split real and imag
        torch::Tensor real = est_spec.select(1, 0);
        torch::Tensor imag = est_spec.select(1, 1);
        auto stft_repr = torch::complex(real, imag);  // shape [B*nch, F, T]

        torch::Tensor output = torch::istft(stft_repr, /*n_fft=*/win_len,
            /*hop_length=*/hop_len,
            /*win_length=*/win_len,
            /*window=*/_window,
            /*center=*/true,
            /*normalized=*/false,
            /*onesided=*/true,
            /*length=*/c10::nullopt);  // Use nsample if needed

        output = output.view({ B, nch, -1 });
        uint64_t post_t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        std::cout << "post time = " << post_t1 - post_t0 << std::endl;
        std::cout << std::endl;
        return output;
#else
       _pre_ir.set_input_tensor(wrap_torch_tensor_as_ov(arr.contiguous()));
       uint64_t pre_t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
       _pre_ir.infer();
       uint64_t pre_t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
       _forward_ir.infer();
       uint64_t fwd_t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
       _post_ir.infer();
       uint64_t post_t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
       auto ret = wrap_ov_tensor_as_torch(_post_ir.get_output_tensor());
       std::cout << "pre time = " << pre_t1 - pre_t0 << std::endl;
       std::cout << "fwd time = " << fwd_t1 - pre_t1 << std::endl;
       std::cout << "post time = " << post_t1 - fwd_t1 << std::endl;
       std::cout << std::endl;
       return ret;

#endif
    }
}
