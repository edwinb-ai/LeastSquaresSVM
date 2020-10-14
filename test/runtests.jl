using Elysivm
using Test
using MLJ

@testset "Elysivm.jl" begin
    include("typetests.jl")
    include("trainingtests.jl")
    include("integrationtests.jl")
end
