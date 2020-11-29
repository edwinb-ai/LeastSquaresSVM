@testset "Types" begin
    # Test the classifier
    svm = LSSVC()
    @test svm.kernel == "rbf"
    @test svm.γ == 1.0
    @test svm.σ == 1.0

    # Test the regressor
    svr = LSSVR()
    @test svm.kernel == "rbf"
    @test svm.γ == 1.0
    @test svm.σ == 1.0

    # Test the kernels
    x = [[0.0, 0.0] [1.0, 1.0]]
    y = [-1.0, 1.0]
    true_kernel = [[1.0, exp(-4.0)] [exp(-4.0), 1.0]]
    la_kernel = KernelRBF(x, 2.0)

    @test all(true_kernel .== la_kernel)

    loop_omega = zeros(size(x)...)

    for i in axes(x, 1)
        for j in axes(x, 2)
            loop_omega[i, j] = y[i] * y[j] * true_kernel[i, j]
        end
    end

    la_omega = @. y * y' * la_kernel
    @test all(loop_omega .== la_omega)

    omega = build_omega(x, y; sigma=2.0)
    @test all(omega .== la_omega)
    @test all(omega .== loop_omega)
end
