using Elysivm
using Test

@testset "Elysivm.jl" begin
    include("typetests.jl")
    include("trainingtests.jl")
end
