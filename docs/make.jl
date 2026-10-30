using DistributionSamplers
using Documenter

DocMeta.setdocmeta!(DistributionSamplers, :DocTestSetup, :(using DistributionSamplers); recursive=true)

makedocs(;
    modules=[DistributionSamplers],
    authors="Šimon Soldát",
    sitename="DistributionSamplers.jl",
    format=Documenter.HTML(;
        canonical="https://soldasim.github.io/DistributionSamplers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/soldasim/DistributionSamplers.jl",
    devbranch="main",
)
