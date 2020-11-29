using Elysivm
using Documenter
using Literate

# ! Convert scripts to markdown using Literate
files = ["example1.jl", "example2.jl"]

function lit_to_md(file)
    examples_path = joinpath(@__DIR__, "src", "examples")
    out_md_path = "src"
    Literate.markdown(
        joinpath(examples_path, file),
        out_md_path;
        documenter=true
    )
end

map(lit_to_md, files)

# * Build the complete documentation
makedocs(;
    modules=[Elysivm],
    authors="Edwin Bedolla",
    repo="https://github.com/edwinb-ai/Elysivm/blob/{commit}{path}#L{line}",
    sitename="Elysivm",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://edwinb-ai.github.io/Elysivm/stable/",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "example1.md",
            "example2.md",
        ],
        "Reference" => "reference.md",

    ],
)

deploydocs(;
    repo="github.com/edwinb-ai/Elysivm.git",
    devbranch="main",
    push_preview=true,
)
