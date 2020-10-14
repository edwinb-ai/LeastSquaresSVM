@testset "Types" begin
    svm = LSSVC()
    @test svm.kernel == "rbf"
    @test svm.γ == 1.0
    @test svm.σ == 1.0
end
