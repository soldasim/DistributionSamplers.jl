include("proposal_distributions/abstract_distribution.jl")
include("proposal_distributions/normal.jl")

include("distribution_fitters/abstract_fitter.jl")
include("distribution_fitters/analytical.jl")
include("distribution_fitters/optimization.jl")

include("amis_method.jl")

"""
    AMISSampler(; kwargs...)

Adaptive Metropolis Importance Sampling (AMIS) sampler for posterior distributions.

The sampler first approximates the posterior distribution by a Laplace approximation
centered on the maximum of the posterior, or with a Gaussian mixture model,
and draws samples from it in the 0th iteration.

Afterwards, the AMIS algorithm is run for `iters` iterations with a simple Gaussian proposal distribution
re-fitted in each iteration.

The target log-density is evaluated on matrices of samples (one sample per column),
so it has to accept both a vector and a matrix and return a scalar or a vector of values, respectively.
The proposal distributions are not truncated to the domain bounds, so the log-density
has to return `-Inf` outside of its support (e.g. through a truncated prior).

## Keywords
- `iters::Int`: Number of iterations of the AMIS algorithm.
- `proposal_fitter::DistributionFitter`: The algorithm used to re-fit the proposal distribution
    in each iteration. Defaults to the `AnalyticalFitter`.
- `gauss_mix_options::GaussMixOptions`: Options for the Gaussian mixture approximation
    used for the 0th iteration.
"""
@kwdef struct AMISSampler <: WeightedSampler
    iters::Int
    proposal_fitter::DistributionFitter = AnalyticalFitter()
    gauss_mix_options::GaussMixOptions
end

function sample_posterior(sampler::AMISSampler, logpost::Function, bounds::AbstractBounds, count::Int;
    info::Bool = false,
    debug::Bool = false,
)
    iters = sampler.iters
    samples_total = iters * count # more samples than needed to allow for efficient down-sampling
    samples_per_iter = samples_total / iters |> ceil |> Int
    (samples_per_iter < 50) && @warn "AMIS: Low sample count ($samples_per_iter) per iteration!"

    # Initialize the proposal distribution
    lb, ub = bounds
    μ = mean(bounds)
    Σ = Diagonal((ub .- lb) ./ 5)
    # the initial parameters `μ, Σ` are important if `approx_by_gauss_mix` returns `nothing`
    q = NormalProposal(MvNormal(μ, Σ))
    init_q = approx_by_gauss_mix(logpost, bounds, sampler.gauss_mix_options; info, debug)

    amis = AMIS(;
        T = iters,
        N = samples_per_iter,
        init_q,
    )
    xs, ws = amis(logpost, q, sampler.proposal_fitter; info, debug)

    return xs, ws
end
