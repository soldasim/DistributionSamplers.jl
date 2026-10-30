"""
    LogpdfMaximizer(; kwargs...)

A helper struct for the [`RejectionSampler`](@ref) that defines the optimization algorithm
used to maximize the log-density. The `RejectionSampler` needs to know the maximum
value for the rejection process.

## Keywords
- `algorithm::Any`: The optimization algorithm used to maximize the log-density.
- `multistart::Int64`: The number of random starting points used in the optimization.
- `parallel::Bool`: If `true`, the optimization is performed in parallel. Defaults to `false`.
- `static_schedule::Bool`: If `true` then the `:static` schedule is used for parallelization.
    This makes the parallel tasks sticky (non-migrating), but can decrease performance.
- `kwargs...`: Additional keyword arguments passed to the optimization algorithm.
"""
struct LogpdfMaximizer
    algorithm::Any
    multistart::Int64
    parallel::Bool
    static_schedule::Bool
    kwargs::Base.Pairs{Symbol, <:Any}
end
function LogpdfMaximizer(;
    algorithm,
    multistart,
    parallel = false,
    static_schedule = false,
    kwargs...
)
    return LogpdfMaximizer(algorithm, multistart, parallel, static_schedule, kwargs)
end

"""
    RejectionSampler(; kwargs...)

A sampler that uses trivial rejection sampling to draw samples from the posterior distribution.

## Keywords
- `logpdf_maximizer::LogpdfMaximizer`: The optimizer used to find the maximum log-density value.
"""
@kwdef struct RejectionSampler <: PureSampler
    logpdf_maximizer::LogpdfMaximizer
end

function sample_posterior(sampler::RejectionSampler, logpost::Function, bounds::AbstractBounds, count::Int;
    info::Bool = false,
    debug::Bool = false,
)
    # uniform prior
    prior = product_distribution(Uniform.(bounds...))

    # likelihood := posterior
    max_logpost = opt_logpdf(sampler.logpdf_maximizer, logpost, bounds; info, debug)

    return sample_posterior_rej(prior, logpost, max_logpost, count)
end

function sample_posterior(sampler::RejectionSampler, loglike::Function, prior::MultivariateDistribution, bounds::AbstractBounds, count::Int;
    info::Bool = false,
    debug::Bool = false,
)
    @assert extrema(prior) == bounds

    max_loglike = opt_logpdf(sampler.logpdf_maximizer, loglike, bounds; info, debug)

    return sample_posterior_rej(prior, loglike, max_loglike, count)
end

function sample_posterior_rej(prior, loglike, max_loglike, count)
    x_dim = length(prior)

    prog = Progress(count; desc="Sampling the posterior: ")
    xs = zeros(x_dim, count)
    drawn = 0
    rejected = 0

    while drawn < count
        x = rand(prior)
        log_p = loglike(x)

        # if max_like * rand() < p
        if max_loglike + log(rand()) < log_p
            next!(prog)
            drawn += 1
            xs[:, drawn] .= x
        else
            rejected += 1
        end
    end

    @info "Rejection rate: $(rejected / (drawn + rejected))"
    ws = fill(1 / size(xs, 2), size(xs, 2))
    return xs, ws
end

function opt_logpdf(opt::LogpdfMaximizer, loglike::Function, bounds::AbstractBounds; info::Bool=true, debug::Bool=false)
    x_dim = length(bounds[1])
    objective = OptimizationFunction((x, _) -> -loglike(x), AutoForwardDiff())
    problem(start) = OptimizationProblem(objective, start, nothing;
        lb = bounds[1],
        ub = bounds[2],
        int = fill(false, x_dim),
    )

    function optim(start)
        x = Optimization.solve(problem(start), opt.algorithm; opt.kwargs...).u
        val = loglike(x)
        return x, val
    end

    starts = generate_LHC(bounds, opt.multistart)
    best_x, best_val = optimize_multistart(
        optim,
        starts;
        opt.parallel,
        opt.static_schedule,
        info,
        debug,
    )

    return best_val
end
