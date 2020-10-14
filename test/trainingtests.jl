@testset "Training" begin
    # * Setup the problem
    x = [[0.0, 0.0] [1.0, 1.0]]
    y = [-1.0, 1.0]
    svm = LSSVC()

    x_test = [2.0, 2.0]
    x_test = reshape(x_test, 2, :)
    fitted = svmtrain(svm, x, y)
    result = svmpredict(svm, fitted, x_test)
    true_result = [1.0]

    @test all(result .== true_result)

    # ! Test for multiple prediction points
    x_test = [[2.0, 2.0] [3.0, 3.0] [-1.0, -1.0]]
    result = svmpredict(svm, fitted, x_test)
    true_result = [1.0, 1.0, -1.0]

    @test all(result .== true_result)

    # ! Change the hyperparameters
    svm = LSSVC(; γ=5.0, σ=0.5)
    fitted = svmtrain(svm, x, y)
    result = svmpredict(svm, fitted, x_test)
    true_result = [1.0, 1.0, -1.0]

    @test all(result .== true_result)
end
