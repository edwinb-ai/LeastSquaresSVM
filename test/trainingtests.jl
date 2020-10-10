@testset "Training" begin
    x = [[0.0, 1.0] [1.0, 0.0]]
    y = [1.0, -1.0]
    svm = LSSVC(size(x)...)
    true_kernel = [[1.0, exp(-4.0)] [exp(-4.0), 1.0]]
    loop_omega = zeros(size(x)...)

    for i in axes(x, 1)
        for j in axes(x, 2)
            loop_omega[i, j] = y[i] * y[j] * true_kernel[i, j]
        end
    end

    la_omega = (y * y') .* true_kernel
    @test all(loop_omega .== la_omega)

    omega = fit!(svm, x, y)
    @test all(omega .== la_omega)
    @test all(omega .== loop_omega)
end
