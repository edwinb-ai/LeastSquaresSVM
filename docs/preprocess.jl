using Literate

# ! Convert scripts to markdown using Literate
files = Dict("example1.jl" => false, "example2.jl" => false, "example3.jl" => true)

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
