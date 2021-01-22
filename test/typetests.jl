@testset "Types" begin
    # Test the classifier
    svm = LSSVC()
    @test svm.kernel == :rbf
    @test svm.γ == 1.0
    @test svm.σ == 1.0
    @test svm.degree == 0.0

    # Test the regressor
    svr = LSSVR()
    @test svm.kernel == :rbf
    @test svm.γ == 1.0
    @test svm.σ == 1.0
    @test svm.degree == 0.0
end
