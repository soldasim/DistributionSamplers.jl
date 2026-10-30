using DistributionSamplers
using Test
using Aqua
using Distributions
using LinearAlgebra
using OptimizationPRIMA
using Random
using Statistics
using Turing

import DistributionSamplers: generate_LHC, get_starts, optimize_multistart, resample
import DistributionSamplers: initial_params, set_params!, loglikelihood, estimate_parameters!, fit_distribution!

const BOUNDS = ([-5., -5.], [5., 5.])
const TRUE_DIST = MvNormal([1., -1.], Diagonal([0.7^2, 0.5^2]))

# truncated to the bounds through the `-Inf` outside of the support
function logpost(x::AbstractVector)
    all(BOUNDS[1] .<= x .<= BOUNDS[2]) || return -Inf
    return logpdf(TRUE_DIST, x)
end
# the `AMISSampler` evaluates the log-density on a matrix of samples
logpost(X::AbstractMatrix) = logpost.(eachcol(X))

weighted_mean(xs, ws) = vec(sum(xs .* ws'; dims=2))
function weighted_cov(xs, ws)
    μ = weighted_mean(xs, ws)
    δ = xs .- μ
    return (δ .* ws') * δ'
end

@testset "DistributionSamplers.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DistributionSamplers)
    end

    @testset "utils" begin
        @testset "generate_LHC" begin
            X = generate_LHC(BOUNDS, 50)
            @test size(X) == (2, 50)
            @test all(BOUNDS[1] .<= X .<= BOUNDS[2])
            @test_throws AssertionError generate_LHC(BOUNDS, 1)
        end

        @testset "get_starts" begin
            @test get_starts(1, BOUNDS) == [0.; 0.;;]
            @test size(get_starts(7, BOUNDS)) == (2, 7)
        end

        @testset "optimize_multistart" begin
            f(x) = (x, -sum(abs2, x .- 1.))
            starts = [0. 3. 1.; 0. 3. 1.]
            x, v = optimize_multistart(f, starts; info=false)
            @test x == [1., 1.] && v == 0.

            xs, vs = optimize_multistart(f, starts; info=false, return_all=true)
            @test length(xs) == 3 && length(vs) == 3

            function flaky(x)
                x[1] == 3. && error("failed")
                return f(x)
            end
            xs, vs = optimize_multistart(flaky, starts; info=false, return_all=true)
            @test length(xs) == 2
            @test_throws ErrorException optimize_multistart(x -> error("failed"), starts; info=false)
            @test_throws ErrorException optimize_multistart(flaky, starts; info=false, debug=true)

            for kws in ((; parallel=true), (; parallel=true, static_schedule=true))
                x, v = optimize_multistart(f, starts; info=false, kws...)
                @test x == [1., 1.]
            end
        end

        @testset "resample" begin
            xs = [1. 2. 3.; 4. 5. 6.]
            @test resample(xs, [0., 1., 0.], 10) == repeat([2.; 5.], 1, 10)
            ys = resample(xs, [1/3, 1/3, 1/3], 30)
            @test size(ys) == (2, 30)
            @test all(in(eachcol(xs)), eachcol(ys))
        end
    end

    @testset "RejectionSampler" begin
        Random.seed!(1)
        maximizer = LogpdfMaximizer(; algorithm=BOBYQA(), multistart=5)
        sampler = RejectionSampler(; logpdf_maximizer=maximizer)
        @test sampler isa PureSampler

        xs, ws = sample_posterior(sampler, logpost, BOUNDS, 4000)
        @test size(xs) == (2, 4000)
        @test all(ws .== 1/4000)
        @test all(BOUNDS[1] .<= xs .<= BOUNDS[2])
        @test vec(mean(xs; dims=2)) ≈ mean(TRUE_DIST) atol=0.1
        @test vec(std(xs; dims=2)) ≈ sqrt.(diag(cov(TRUE_DIST))) atol=0.1

        @testset "likelihood and prior" begin
            prior = product_distribution(Uniform.(BOUNDS...))
            xs, ws = sample_posterior(sampler, x -> logpdf(TRUE_DIST, x), prior, BOUNDS, 2000)
            @test size(xs) == (2, 2000)
            @test vec(mean(xs; dims=2)) ≈ mean(TRUE_DIST) atol=0.15

            wrong_prior = product_distribution(Uniform.([-1., -1.], [1., 1.]))
            @test_throws AssertionError sample_posterior(sampler, x -> logpdf(TRUE_DIST, x), wrong_prior, BOUNDS, 10)
        end

        @testset "sample_posterior_pure" begin
            xs = sample_posterior_pure(sampler, logpost, BOUNDS, 500)
            @test size(xs) == (2, 500)
        end
    end

    @testset "GaussMixOptions & approx_by_gauss_mix" begin
        opt = GaussMixOptions(; algorithm=BOBYQA(), multistart=20)
        @test opt.multistart == 20 && opt.cons_atol == 0.05 && opt.rel_min_weight == 1e-8
        @test haskey(GaussMixOptions(; algorithm=BOBYQA(), rhoend=1e-3).kwargs, :rhoend)

        Random.seed!(2)
        gm = approx_by_gauss_mix(logpost, BOUNDS, opt; info=false)
        @test gm isa MixtureModel
        @test length(components(gm)) == 1
        @test mean(gm) ≈ mean(TRUE_DIST) atol=0.05
        @test cov(gm) ≈ cov(TRUE_DIST) atol=0.05

        @testset "bimodal" begin
            bi = MixtureModel([MvNormal([-3., -3.], 0.5), MvNormal([3., 3.], 0.5)], [0.5, 0.5])
            bi_logpost(x) = all(BOUNDS[1] .<= x .<= BOUNDS[2]) ? logpdf(bi, x) : -Inf
            gm = approx_by_gauss_mix(bi_logpost, BOUNDS, GaussMixOptions(; algorithm=BOBYQA(), multistart=40); info=false)
            modes = sort(collect(mean.(components(gm))); by=first)
            @test length(modes) == 2
            @test modes[1] ≈ [-3., -3.] atol=0.1
            @test modes[2] ≈ [3., 3.] atol=0.1
            @test probs(gm) ≈ [0.5, 0.5] atol=0.1
        end

        @testset "no modes: uniform fallback" begin
            flat(x) = logpdf(MvNormal(zeros(2), 100.), x)  # too wide to be a mode
            @test_logs (:warn, r"No modes found") match_mode=:any begin
                gm = approx_by_gauss_mix(flat, BOUNDS, GaussMixOptions(; algorithm=BOBYQA(), multistart=5); info=false)
                @test gm isa Product
            end
        end
    end

    @testset "NormalProposal & fitters" begin
        Random.seed!(3)
        true_d = MvNormal([1., 2.], [1. 0.3; 0.3 0.5])
        xs = rand(true_d, 3000)
        ws = fill(1/3000, 3000)

        q = NormalProposal(2)
        @test DistributionSamplers.x_dim(q) == 2
        @test size(rand(q, 5)) == (2, 5)
        @test pdf(q, [0., 0.]) ≈ exp(logpdf(q, [0., 0.]))

        fit_distribution!(AnalyticalFitter(), q, xs, ws)
        @test mean(q.dist) ≈ [1., 2.] atol=0.1
        @test cov(q.dist) ≈ cov(true_d) atol=0.1

        q2 = NormalProposal(2)
        fitter = OptimizationFitter(; algorithm=NEWUOA(), multistart=10)
        fit_distribution!(fitter, q2, xs, ws; info=false)
        @test mean(q2.dist) ≈ [1., 2.] atol=0.15
        @test cov(q2.dist) ≈ cov(true_d) atol=0.15

        @testset "parametrization" begin
            θ = initial_params(NormalProposal(3), 4)
            @test size(θ) == (3 + 9, 4)
            set_params!(q, [0., 0., 1., 0., 0., 1.])
            @test mean(q.dist) == [0., 0.] && cov(q.dist) ≈ I(2)
            ll = loglikelihood(q, xs[:, 1:5], ones(5))
            @test ll([0., 0., 1., 0., 0., 1.]) ≈ sum(logpdf(MvNormal(zeros(2), I(2)), x) for x in eachcol(xs[:, 1:5]))
        end
    end

    @testset "AMISSampler" begin
        Random.seed!(4)
        sampler = AMISSampler(;
            iters = 5,
            gauss_mix_options = GaussMixOptions(; algorithm=BOBYQA(), multistart=10),
        )
        @test sampler isa WeightedSampler
        @test sampler.proposal_fitter isa AnalyticalFitter

        xs, ws = sample_posterior(sampler, logpost, BOUNDS, 400)
        @test size(xs) == (2, 5 * 400)
        @test length(ws) == 5 * 400
        @test sum(ws) ≈ 1
        @test all(ws .>= 0)
        @test weighted_mean(xs, ws) ≈ mean(TRUE_DIST) atol=0.1
        @test weighted_cov(xs, ws) ≈ cov(TRUE_DIST) atol=0.1

        @testset "likelihood and prior" begin
            prior = product_distribution(Uniform.(BOUNDS...))
            xs, ws = sample_posterior(sampler, x -> logpdf(TRUE_DIST, x), prior, BOUNDS, 400)
            @test sum(ws) ≈ 1
            @test weighted_mean(xs, ws) ≈ mean(TRUE_DIST) atol=0.15
        end

        @testset "sample_posterior_pure" begin
            xs = sample_posterior_pure(sampler, logpost, BOUNDS, 300)
            @test size(xs) == (2, 300)
            @test vec(mean(xs; dims=2)) ≈ mean(TRUE_DIST) atol=0.15
        end

        @testset "optimization fitter" begin
            sampler_opt = AMISSampler(;
                iters = 3,
                proposal_fitter = OptimizationFitter(; algorithm=NEWUOA(), multistart=5),
                gauss_mix_options = GaussMixOptions(; algorithm=BOBYQA(), multistart=10),
            )
            xs, ws = sample_posterior(sampler_opt, logpost, BOUNDS, 300)
            @test weighted_mean(xs, ws) ≈ mean(TRUE_DIST) atol=0.15
        end

        @testset "numerical issues" begin
            neg_inf(x::AbstractVector) = -Inf
            neg_inf(X::AbstractMatrix) = fill(-Inf, size(X, 2))
            @test_throws ErrorException sample_posterior(sampler, neg_inf, BOUNDS, 100)
        end
    end

    @testset "TuringSampler (TuringExt)" begin
        Random.seed!(5)
        sampler = TuringSampler(; sampler=NUTS(0, 0.65), warmup=500, chain_count=2, leap_size=2, parallel=false)
        @test sampler isa PureSampler
        @test sampler isa DistributionSamplers.TuringSampler

        xs, ws = sample_posterior(sampler, logpost, BOUNDS, 200)
        @test size(xs) == (2, 200)
        @test all(ws .== 1/200)
        @test all(BOUNDS[1] .<= xs .<= BOUNDS[2])
        @test vec(mean(xs; dims=2)) ≈ mean(TRUE_DIST) atol=0.5

        prior = product_distribution(Uniform.(BOUNDS...))
        xs, ws = sample_posterior(sampler, x -> logpdf(TRUE_DIST, x), prior, BOUNDS, 100)
        @test size(xs) == (2, 100)
        @test_throws AssertionError sample_posterior(sampler, x -> 0., product_distribution(Uniform.([-1., -1.], [1., 1.])), BOUNDS, 10)
    end
end
