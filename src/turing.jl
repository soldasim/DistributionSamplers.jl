"""
    TuringSampler <: PureSampler

Samples from the posterior using the Turing.jl package.
The concrete implementation is loaded together with Turing.jl (the `TuringExt` extension),
and is constructed by `TuringSampler(; kwargs...)`.

## Keywords
- `sampler::Any`: The sampling algorithm used to draw the samples. Defaults to `NUTS(0, 0.65)`.
- `warmup::Int`: The amount of initial unused 'warmup' samples in each chain.
- `chain_count::Int`: The amount of independent chains sampled.
- `leap_size::Int`: Every `leap_size`-th sample is used from each chain. (To avoid correlated samples.)
- `parallel::Bool`: If `true` then the chains are sampled in parallel.

## Sampling process

In each sampled chain;
  - The first `warmup` samples are discarded.
  - From the following `leap_size * samples_in_chain` samples each `leap_size`-th is kept.
Then the samples from all chains are concatenated and returned.

Total drawn samples:    `chain_count * (warmup + leap_size * samples_in_chain)`
Total returned samples: `chain_count * samples_in_chain`
"""
abstract type TuringSampler <: PureSampler end

# The sampling is implemented in the `TuringExt` extension.
