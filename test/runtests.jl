using LeastSquaresSVM
using Test
using MLJ
using Random

@testset "LeastSquaresSVM.jl" begin
    include("typetests.jl")
    include("trainingtests.jl")
    include("fixedsizetests.jl")
    include("integrationtests.jl")
end
