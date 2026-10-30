module TuringExt

using DistributionSamplers
using Distributions
using Random
using Turing

import DistributionSamplers: AbstractBounds

"""
Implementation of the abstract `DistributionSamplers.TuringSampler`. See the docs `? TuringSampler`.
"""
@kwdef struct TuringSampler{S} <: DistributionSamplers.TuringSampler
    sampler::S = NUTS(0, 0.65)
    warmup::Int = 1000
    chain_count::Int = 6
    leap_size::Int = 5
    parallel::Bool = true
end

DistributionSamplers.TuringSampler(args...; kwargs...) = TuringSampler(args...; kwargs...)

function DistributionSamplers.sample_posterior(sampler::TuringSampler, logpost::Function, bounds::AbstractBounds, count::Int; kwargs...)
    model = turing_model(logpost, bounds)
    return _sample_posterior_turing(model, sampler, count)
end
function DistributionSamplers.sample_posterior(sampler::TuringSampler, loglike::Function, prior::MultivariateDistribution, bounds::AbstractBounds, count::Int; kwargs...)
    @assert extrema(prior) == bounds

    model = turing_model(loglike, prior)
    return _sample_posterior_turing(model, sampler, count)
end

@model function turing_model(logpost, bounds::AbstractBounds)
    x ~ product_distribution(Uniform.(bounds...))
    Turing.@addlogprob! logpost(x)
end
@model function turing_model(loglike, x_prior::MultivariateDistribution)
    x ~ x_prior
    Turing.@addlogprob! loglike(x)
end

function _sample_posterior_turing(model, sampler::TuringSampler, count::Int)
    # This count will possibly result in a few extra samples. They are discarded later.
    count_per_chain = (count / sampler.chain_count) |> ceil |> Int
    samples_in_chain = sampler.warmup + (sampler.leap_size * count_per_chain)

    if sampler.parallel
        chains = Turing.sample(model, sampler.sampler, MCMCThreads(), samples_in_chain, sampler.chain_count; progress=false)
    else
        chains = mapreduce(_ -> Turing.sample(model, sampler.sampler, samples_in_chain; progress=false), Turing.AbstractMCMC.chainscat, 1:sampler.chain_count)
    end

    # VNChain (Turing ≥0.45): _data maps Parameter(x) -> Matrix{Vector{Float64}}(n_iters, n_chains)
    x_key = only(k for k in keys(chains._data) if k isa Turing.FlexiChains.Parameter)
    x_matrix = chains._data[x_key]  # (n_iters, n_chains)

    # skip warmup samples and apply leap
    idx = (sampler.warmup + sampler.leap_size):sampler.leap_size:size(x_matrix, 1)
    # collect into x_dim × n_samples matrix
    samples = reduce(hcat, [x_matrix[i, j] for j in axes(x_matrix, 2) for i in idx])

    # shuffle & discard extra samples
    keep = randperm(size(samples, 2))[1:count]
    samples = samples[:, keep]

    ws = fill(1 / size(samples, 2), size(samples, 2))
    return samples, ws
end

end # module TuringExt
