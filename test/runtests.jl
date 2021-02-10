using LeastSquaresSVM
using Test
using MLJ

@testset "LeastSquaresSVM.jl" begin
    # include("typetests.jl")
    # include("trainingtests.jl")
    include("integrationtests.jl")
end
