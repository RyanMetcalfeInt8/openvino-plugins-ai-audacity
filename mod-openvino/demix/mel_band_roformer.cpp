// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: GPL-3.0-only

#include <demix/mel_band_roformer.h>
#include "utils/openvino_utils.h"

namespace ov_demix
{
    MelBandRoformer::MelBandRoformer(const std::string& model_dir,
        const std::string& device,
        const std::string& cache_dir,
        DemixModel::PadMode pad_mode)
        : _pad_mode(pad_mode)
    {
        ov::Core core;

        {
            auto xml_path = FullPath(model_dir, "mel_band_pre.xml");
            auto model = core.read_model(xml_path);
            std::cout << "PRE:" << std::endl;
            logBasicModelInfo(model);

            auto compiled_model = core.compile_model(model, "CPU");
            _pre_ir = compiled_model.create_infer_request();
        }

        {
            auto xml_path = FullPath(model_dir, "mel_band_fwd.xml");
            auto model = core.read_model(xml_path);
            std::cout << "FWD:" << std::endl;
            logBasicModelInfo(model);

            ov::AnyMap properties = { ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY) };

            if (!cache_dir.empty())
            {
                properties.insert(ov::cache_dir(cache_dir));
            }

            properties.insert({ "PERF_COUNT", "YES" });

            auto compiled_model = core.compile_model(model, device, properties);

            auto runtime_graph = compiled_model.get_runtime_model();
            ov::save_model(runtime_graph, "runtime_graph.xml");

            _forward_ir = compiled_model.create_infer_request();
        }


        {
            auto xml_path = FullPath(model_dir, "mel_band_post.xml");
            auto model = core.read_model(xml_path);
            std::cout << "POST:" << std::endl;
            logBasicModelInfo(model);

            auto compiled_model = core.compile_model(model, "CPU");
            _post_ir = compiled_model.create_infer_request();
        }
    }

    

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

    MelBandRoformer::~MelBandRoformer() {
       print_perf_counts(_forward_ir);
    }

    torch::Tensor MelBandRoformer::run(torch::Tensor arr)
    {
        using namespace std::chrono;
        using Clock = std::chrono::high_resolution_clock;
        auto arr_ov = wrap_torch_tensor_as_ov(arr);

        _pre_ir.set_input_tensor(arr_ov);
        _pre_ir.infer();

        auto stft_repr_ov = _pre_ir.get_tensor("stft_repr");
        auto stft_window_ov = _pre_ir.get_tensor("stft_window");

        _forward_ir.set_input_tensor(stft_repr_ov);
        uint64_t t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        _forward_ir.infer();
        uint64_t t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        std::cout << "fwd time = " << t1 - t0 << std::endl;

        auto masks_ov = _forward_ir.get_tensor("masks");

        _post_ir.set_tensor("stft_repr", stft_repr_ov);
        _post_ir.set_tensor("masks", masks_ov);
        _post_ir.set_tensor("stft_window", stft_window_ov);
        _post_ir.infer();

        auto recon_ov = _post_ir.get_tensor("recon");

        //TODO: I think we can remove this clone.
        return wrap_ov_tensor_as_torch(recon_ov).clone();
    }
}
