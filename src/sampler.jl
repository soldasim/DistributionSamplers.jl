"""
    DistributionSampler

Subtypes of `DistributionSampler` are used to sample from a probability distribution
defined by a (possibly unnormalized) log-density on a box domain.

Each subtype *should* implement:
- `sample_posterior(::DistributionSampler, logpost::Function, bounds, count::Int; info=false, debug=false) -> (X, ws)`

Each subtype *may* additionally implement:
- `sample_posterior(::DistributionSampler, loglike::Function, prior::MultivariateDistribution, bounds, count::Int; info=false, debug=false) -> (X, ws)`

See also: [`PureSampler`](@ref), [`WeightedSampler`](@ref)
"""
abstract type DistributionSampler end

"""
    PureSampler <: DistributionSampler

A `DistributionSampler` which samples directly from the provided pdf,
and always returns samples with uniform weights.
"""
abstract type PureSampler <: DistributionSampler end

"""
    WeightedSampler <: DistributionSampler

A `DistributionSampler` which does not sample directly from the pdf,
but instead returns samples with non-uniform weights correcting for the sampling bias.
"""
abstract type WeightedSampler <: DistributionSampler end

"""
    X, ws = sample_posterior(sampler, logpost, bounds, count; kwargs...)
    X, ws = sample_posterior(sampler, loglike, prior, bounds, count; kwargs...)

Draw `count` samples (columns of `X`) with weights `ws` from the given (possibly unnormalized)
log-density `logpost` on the box `bounds = (lb, ub)`.
The second method samples from `loglike + logpdf(prior, x)`.

## Keywords
- `info::Bool=false`: Print progress messages.
- `debug::Bool=false`: Rethrow errors of failed optimization runs.
"""
function sample_posterior end

# default implementation for `DistributionSampler`s not implementing the second method
function sample_posterior(sampler::DistributionSampler, loglike::Function, prior::MultivariateDistribution, bounds::AbstractBounds, count::Int; kwargs...)
    function logpost(x)
        lp = logpdf(prior, x)
        ll = loglike(x)
        return ll + lp
    end

    return sample_posterior(sampler, logpost, bounds, count; kwargs...)
end

"""
    xs = sample_posterior_pure(sampler, logpost, bounds, count; supersample_ratio=20, kwargs...)

Sample `count` samples from the posterior distribution defined by the `logpost` log-density.
Assures that the returned samples are "pure" (unweighted).

In case of a [`WeightedSampler`](@ref), `supersample_ratio` × `count` samples are drawn,
and subsequently down-sampled to `count` samples according to their weights.
"""
function sample_posterior_pure(sampler::PureSampler, logpost::Function, bounds::AbstractBounds, count::Int;
    supersample_ratio = 20,
    kwargs...
)
    xs, ws = sample_posterior(sampler, logpost, bounds, count; kwargs...)
    @assert allequal(ws)
    return xs
end
function sample_posterior_pure(sampler::WeightedSampler, logpost::Function, bounds::AbstractBounds, count::Int;
    supersample_ratio = 20,
    kwargs...
)
    xs, ws = sample_posterior(sampler, logpost, bounds, supersample_ratio * count; kwargs...)
    xs = resample(xs, ws, count)
    return xs
end

"""
    xs = resample(xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real}, count::Int)

Resample `count` samples from the given data set `xs` weighted by the given weights `ws`
with replacement to obtain a new un-weighted data set.

Some data points may repeat in the resampled data set. Increasing the sample size
of the initial data set may help to reduce the number of repetitions.
"""
function resample(xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real}, count::Int)
    return hcat(wsample(eachcol(xs), ws, count; replace=true)...)
end
