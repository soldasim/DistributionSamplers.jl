using DistributionSamplers
using Test
using Aqua

@testset "DistributionSamplers.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DistributionSamplers)
    end
    # Write your tests here.
end
