using Elysivm
using Documenter

# * Build the complete documentation
makedocs(;
    modules=[Elysivm],
    authors="Edwin Bedolla",
    repo="https://github.com/edwinb-ai/Elysivm/blob/{commit}{path}#L{line}",
    sitename="Elysivm",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://edwinb-ai.github.io/Elysivm/stable/",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "example1.md",
            "example2.md",
            "example3.md"
        ],
        "Reference" => "reference.md",
    ]
)

deploydocs(;
    repo="github.com/edwinb-ai/Elysivm.git",
    devbranch="main",
    push_preview=true
)
