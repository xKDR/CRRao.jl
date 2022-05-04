using CRRao
using Documenter

DocMeta.setdocmeta!(CRRao, :DocTestSetup, :(using CRRao); recursive=true)

makedocs(;
    modules=[CRRao],
    authors="xKDR Forum",
    repo="https://github.com/xKDR/CRRao.jl/blob/{commit}{path}#{line}",
    sitename="CRRao.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://xKDR.github.io/CRRao.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => "examples.md"
    ],
)

deploydocs(;
    repo="github.com/xKDR/CRRao.jl",
    target = "build",
    devbranch = "main",
    strict = :doctest
)
