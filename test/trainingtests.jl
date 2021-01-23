@testset "Classifier" begin
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

@testset "Regression" begin
    # * Setup the problem
    X_train = [[0.0, 0.0] [2.0, 2.0]]
    y = [0.5, 2.5]
    svr = LSSVR()
    fitted = svmtrain(svr, X_train, y)

    # Create some testing data
    X_test = [1.0, 1.0]
    X_test = reshape(X_test, 2, :)
    y_test = [1.5]
    result = svmpredict(svr, fitted, X_test)
    @test isapprox(result, y_test)
end

function train_and_predict(x, xtest, y, kernel, deg)
    svm = LSSVC(; kernel=kernel, degree=deg)
    fitted = svmtrain(svm, x, y)
    result = svmpredict(svm, fitted, xtest)

    return result
end

@testset "Kernels" begin
    # * Setup the problem
    x = [[0.0, 0.0] [1.0, 1.0]]
    y = [-1.0, 1.0]
    kernels = Dict(:linear => 1, :poly => 2)

    x_test = [2.0, 2.0]
    x_test = reshape(x_test, 2, :)
    true_result = [1.0]

    for (k, v) in kernels
        result = train_and_predict(x, x_test, y, k, v)
        @test all(result .== true_result)
    end
end
