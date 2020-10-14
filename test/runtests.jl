using Elysivm
using Test
using MLJBase, MLJ

@testset "Elysivm.jl" begin
    include("typetests.jl")
    include("trainingtests.jl")
    include("integrationtests.jl")
end
