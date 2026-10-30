```@meta
CurrentModule = DistributionSamplers
```

# DistributionSamplers

Samplers for drawing (weighted) samples from a probability distribution given by a (possibly unnormalized)
log-density on a box domain `(lb, ub)`. See the [README](https://github.com/soldasim/DistributionSamplers.jl) for an overview and an example.

## Sampling

```@docs
DistributionSampler
PureSampler
WeightedSampler
sample_posterior
sample_posterior_pure
resample
```

## Samplers

```@docs
RejectionSampler
LogpdfMaximizer
AMISSampler
TuringSampler
```

## AMIS components

```@docs
GaussMixOptions
approx_by_gauss_mix
ProposalDistribution
NormalProposal
DistributionFitter
AnalyticalFitter
OptimizationFitter
```
