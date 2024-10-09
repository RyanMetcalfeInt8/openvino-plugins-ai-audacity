#pragma once

#include "audiosr_common.h"

class DiagonalGaussianDistribution
{
public:

    DiagonalGaussianDistribution(torch::Tensor parameters)
    {
        auto chynk_ret = torch::chunk(parameters, 2, 1);
        _mean = chynk_ret.at(0);
        _logvar = chynk_ret.at(1);

        _logvar = torch::clamp(_logvar, -30.f, 20.f);
        _std = torch::exp(0.5f * _logvar);
        _var = torch::exp(_logvar);
    }

    torch::Tensor sample()
    {
        auto rand_tensor = torch::randn(_mean.sizes());
        auto x = _mean + _std * rand_tensor;
        return x;
    }

private:

    torch::Tensor _mean;
    torch::Tensor _logvar;
    torch::Tensor _parameters;
    torch::Tensor _std;
    torch::Tensor _var;
};