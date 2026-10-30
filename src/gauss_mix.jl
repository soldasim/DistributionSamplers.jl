"""
    GaussMixOptions(; kwargs...)

Contains all hyperparameters for the function [`approx_by_gauss_mix`](@ref).

## Keywords
- `algorithm`: Optimization algorithm used to find the modes.
- `multistart::Int`: Number of optimization restarts.
- `parallel::Bool`: Controls whether the individual optimization runs are performed in parallel.
- `static_schedule::Bool`: If `true` then the `:static` schedule is used for parallelization.
    This makes the parallel tasks sticky (non-migrating), but can decrease performance.
- `autodiff::ADTypes.AbstractADType`: The autodiff library used for the optimization
    and the Hessian. (Only relevant if a gradient-based optimizer is used as `algorithm`.)
- `cluster_ϵs::Union{Nothing, Vector{Float64}}`: The minimum distance between modes.
    Modes which are too close to a "more important" mode are discarded.
- `cons_atol::Real`: Unused. Kept for backwards compatibility.
- `rel_min_weight::Float64`: The minimum pdf value of a mode to be considered
    relative to the highest pdf value among all found modes.
- `kwargs...`: Other keywords are passed to the optimization algorithm.
"""
struct GaussMixOptions{
    A<:Any,
}
    algorithm::A
    multistart::Int
    parallel::Bool
    static_schedule::Bool
    autodiff::ADTypes.AbstractADType
    cluster_ϵs::Union{Nothing, Vector{Float64}}
    cons_atol::Real
    rel_min_weight::Float64
    kwargs::Base.Pairs{Symbol, <:Any}
end
GaussMixOptions(;
    algorithm,
    multistart = 200,
    parallel = false,
    static_schedule = false,
    autodiff = AutoForwardDiff(),
    cluster_ϵs = nothing,
    cons_atol = 0.05,
    rel_min_weight = 1e-8,
    kwargs...
) = GaussMixOptions(algorithm, multistart, parallel, static_schedule, autodiff, cluster_ϵs, cons_atol, rel_min_weight, kwargs)

cluster_ϵs(opt::GaussMixOptions, bounds::AbstractBounds) =
    isnothing(opt.cluster_ϵs) ? _default_cluster_ϵs(bounds) : opt.cluster_ϵs

_default_cluster_ϵs(bounds::AbstractBounds) = (bounds[2] .- bounds[1]) ./ (ℯ^3)

"""
    gauss_mix = approx_by_gauss_mix(logpost, bounds, opt::GaussMixOptions; info=true, debug=false)

Approximate the given (possibly unnormalized) log-density by a Gaussian mixture.

Find all modes via Optimization.jl, then approximate each mode with a multivariate Gaussian
with mean in the mode and covariance according to the second derivative of the log-density in the mode.
Falls back on the uniform distribution on `bounds` if no mode is found.
"""
function approx_by_gauss_mix(logpost, bounds::AbstractBounds, opt::GaussMixOptions; info::Bool=true, debug::Bool=false)
    μs = opt_for_modes(logpost, bounds, opt; info, debug)

    gausses = laplace_approx.(Ref(logpost), μs, Ref(bounds); opt.autodiff)
    gausses = [filter(!isnothing, gausses)...] # init new vector to correct the eltype

    # discard gausses which are too wide
    max_stds = (bounds[2] .- bounds[1]) ./ 2 # arbitrarily chosen
    gausses = filter(g -> all(std(g) .<= max_stds), gausses)

    if isempty(gausses)
        @warn "No modes found! Falling back on a Uniform distribution."
        return product_distribution(Uniform.(bounds...))
    end

    log_weights = logpost.(getfield.(gausses, Ref(:μ)))
    log_weights .-= maximum(log_weights)
    weights = exp.(log_weights)

    return MixtureModel(gausses, weights ./ sum(weights))
end

function opt_for_modes(logpost, bounds::AbstractBounds, opt::GaussMixOptions; info::Bool=true, debug::Bool=false)
    starts = get_starts(opt.multistart, bounds)

    # objective function
    obj_func = logpost
    objective = OptimizationFunction((x, _) -> -obj_func(x), opt.autodiff)

    # constraints
    lb, ub = bounds
    x_dim = length(lb)

    # optimization problem
    function problem(start)
        return OptimizationProblem(objective, start, nothing;
            lb,
            ub,
            lcons = fill(0., 0),
            ucons = fill(Inf, 0),
            int = fill(false, x_dim),
        )
    end

    # optimization run
    function optimize_(start)
        x = Optimization.solve(problem(start), opt.algorithm; opt.kwargs...).u
        val = obj_func(x)  # correct sign
        return x, val
    end

    # --- MAIN ----
    θs, vals = optimize_multistart(optimize_, starts; opt.parallel, opt.static_schedule, return_all=true, info, debug)
    θs = cluster_modes(θs, vals; opt.rel_min_weight, cluster_ϵs=cluster_ϵs(opt, bounds))
    return θs
end

function cluster_modes(θs::AbstractVector{<:AbstractVector{<:Real}}, vals::AbstractVector{<:Real}; rel_min_weight::Real, cluster_ϵs::AbstractVector{<:Real})
    isempty(θs) && return θs  # no modes found

    # sort by descending posterior value
    score = sortperm(vals; rev=true)
    θs = θs[score]
    vals = vals[score]

    # keep only a single sample from each mode
    keep = [first(θs)]
    v1 = first(vals)

    for (θ, v) in zip(θs, vals)
        (v / v1 < rel_min_weight) && continue
        is_new_mode(θ, keep, cluster_ϵs) && push!(keep, θ)
    end

    return keep
end

function is_new_mode(θ::AbstractVector{<:Real}, keep::AbstractVector{<:AbstractVector{<:Real}}, ϵs::AbstractVector{<:Float64})
    for θ_ in keep
        any(abs.(θ .- θ_) .> ϵs) || return false
    end
    return true
end

function laplace_approx(logpost, μ, bounds; autodiff=AutoForwardDiff(), ϵ=0.)
    second_derivative = hessian(logpost, autodiff, μ)

    if any(isnan.(second_derivative)) || any(isinf.(second_derivative))
        @warn "Laplace approx.: Failed due to numerical issues in Hessian."
        return nothing
    end

    # construct Σ
    Σ = (-1) * second_derivative
    if (det(Σ) ≈ 0)
        @warn "Laplace approx.: Hessian is singular! Using diagonal covariance matrix instead."
        Σ = Diagonal(Σ)
    end
    Σ = inv(Σ)

    # solve numerical issues
    #   - Σ should already be symmetric, but might not be due to small numerical imprecisions
    Σ = Symmetric(Σ) + (ϵ * I) # add a small diagonal matrix to ensure positive definiteness

    if !isposdef(Σ)
        @warn "Laplace approx.: Failed due to the covariance matrix not being positive definite."
        return nothing
    end

    return MvNormal(μ, Σ)
end
