using Literate

# ! Convert scripts to markdown using Literate
files = ["example1.jl", "example2.jl", "example3.jl"]

function lit_to_md(file)
    examples_path = joinpath("docs", "src", "examples")
    out_md_path = joinpath("docs", "src")
    Literate.markdown(
        joinpath(examples_path, file),
        out_md_path;
        documenter=true,
        evaluate=true
    )
end

map(lit_to_md, files)
