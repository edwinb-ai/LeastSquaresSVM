using LeastSquaresSVM
using Documenter

# * Build the complete documentation
makedocs(;
    modules=[LeastSquaresSVM],
    authors="Edwin Bedolla",
    repo="https://github.com/edwinb-ai/LeastSquaresSVM/blob/{commit}{path}#L{line}",
    sitename="LeastSquaresSVM",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://edwinb-ai.github.io/LeastSquaresSVM/stable/",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "example1.md",
            "example2.md",
            "example3.md",
            "example4.md"
        ],
        "Reference" => "reference.md",
    ]
)

deploydocs(;
    repo="github.com/edwinb-ai/LeastSquaresSVM.git",
    devbranch="main",
    push_preview=true
)
