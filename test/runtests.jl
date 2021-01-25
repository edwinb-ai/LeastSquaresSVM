using LeastSquaresSVM
using Test
using MLJBase, MLJ, MLJModels

@testset "LeastSquaresSVM.jl" begin
    include("typetests.jl")
    include("trainingtests.jl")
    include("integrationtests.jl")
end
