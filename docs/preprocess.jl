using Literate

# * If true, the code will be executed by Literate and the output will be captured.
# * If false, the code will be executed by Documenter as an example block.
files = Dict(
    "example1.jl" => true,
    "example2.jl" => false,
    "example3.jl" => true,
    "example4.jl" => true
)

function lit_to_md(file; execute=false)
    examples_path = joinpath("src", "examples")
    out_md_path = "src/"
    Literate.markdown(
        joinpath(examples_path, file),
        out_md_path;
        documenter=true,
        execute=execute
    )

    return nothing
end

function parse_files(some_files)
    for (k, v) in some_files
        lit_to_md(k; execute=v)
    end

    return nothing
end

parse_files(files)
