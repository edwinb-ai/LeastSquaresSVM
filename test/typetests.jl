@testset "Types" begin
    # Test the classifier
    svm = LSSVC()
    @test svm.kernel == "rbf"
    @test svm.γ == 1.0
    @test svm.σ == 1.0
    @test svm.degree == 0.0

    # Test the regressor
    svr = LSSVR()
    @test svm.kernel == "rbf"
    @test svm.γ == 1.0
    @test svm.σ == 1.0
    @test svm.degree == 0.0

    # Test the kernels
    # x = [[0.0, 0.0] [1.0, 1.0]]
    # y = [-1.0, 1.0]
    # true_kernel = [[1.0, exp(-4.0)] [exp(-4.0), 1.0]]
end
