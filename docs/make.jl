using Elysivm
using Documenter

makedocs(;
    modules=[Elysivm],
    authors="Edwin Bedolla",
    repo="https://github.com/edwinb-ai/Elysivm.jl/blob/{commit}{path}#L{line}",
    sitename="Elysivm.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://edwinb-ai.github.io/Elysivm.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/edwinb-ai/Elysivm.jl",
)
