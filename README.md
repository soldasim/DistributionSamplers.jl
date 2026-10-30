# DistributionSamplers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://soldasim.github.io/DistributionSamplers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://soldasim.github.io/DistributionSamplers.jl/dev/)
[![Build Status](https://github.com/soldasim/DistributionSamplers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/soldasim/DistributionSamplers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/soldasim/DistributionSamplers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/soldasim/DistributionSamplers.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Samplers for drawing (weighted) samples from a probability distribution given by a (possibly unnormalized)
log-density on a box domain. Originally part of [BOSIP.jl](https://github.com/soldasim/BOSIP.jl); usable on its own.

| Sampler | Type | Notes |
|---|---|---|
| `RejectionSampler` | `PureSampler` | Rejection sampling from the uniform distribution on the box. The maximum of the log-density is found by multistart optimization. Only practical in low dimensions. |
| `AMISSampler` | `WeightedSampler` | Adaptive Metropolis Importance Sampling. Starts from a Gaussian mixture found by Laplace approximations of the modes, then adapts a Gaussian proposal. Returns weighted samples. |
| `TuringSampler` | `PureSampler` | MCMC via [Turing.jl](https://github.com/TuringLang/Turing.jl). Available once `Turing` is loaded (package extension). |

Pure samplers return uniformly weighted samples. `sample_posterior_pure` down-samples the output of weighted samplers
(by `resample`) so that every sampler can return unweighted samples.

The domain is the box `(lb, ub)` (a tuple of two vectors). Discrete variables and nonlinear constraints are not supported.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/soldasim/DistributionSamplers.jl")
```

## Usage

```julia
using DistributionSamplers, Distributions, OptimizationPRIMA

bounds = ([-5., -5.], [5., 5.])
target = MvNormal([1., -1.], [0.5 0.1; 0.1 0.3])

# The log-density has to return `-Inf` outside of the box.
logpost(x::AbstractVector) = all(bounds[1] .<= x .<= bounds[2]) ? logpdf(target, x) : -Inf
# `AMISSampler` evaluates the log-density on a matrix of samples (one sample per column).
logpost(X::AbstractMatrix) = logpost.(eachcol(X))

sampler = AMISSampler(;
    iters = 5,
    gauss_mix_options = GaussMixOptions(; algorithm=BOBYQA(), multistart=20),
)

xs, ws = sample_posterior(sampler, logpost, bounds, 1000)  # weighted samples (columns of `xs`)
xs = sample_posterior_pure(sampler, logpost, bounds, 1000) # unweighted samples
```

The keyword `info=true` prints progress messages (e.g. the AMIS progress bar).
`RejectionSampler` is created as
`RejectionSampler(; logpdf_maximizer=LogpdfMaximizer(; algorithm=BOBYQA(), multistart=10))`.
`TuringSampler(; sampler=NUTS(0, 0.65), warmup=1000, chain_count=6, leap_size=5)` requires `using Turing`.
