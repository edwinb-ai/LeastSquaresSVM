# Core-level correctness tests for the Fixed-Size LS-SVM regressor.
#
# Following Suykens et al. (Least Squares Support Vector Machines, §6.2), the
# primal ridge problem is solved over ALL N training points projected into the
# m-dimensional Nyström feature space. These tests guard that solve: a previous
# version regressed on only the m prototype points (an underdetermined system)
# and produced unbounded, meaningless predictions.

_rmse(a, b) = sqrt(sum(abs2, a .- b) / length(b))

@testset "Fixed Size regressor: primal solve over all N" begin
    rng = MersenneTwister(1234)
    n = 400
    X = randn(rng, 2, n)                              # features × samples
    f(c) = sinpi(0.5c[1]) + 0.5c[2]^2                 # smooth nonlinear target
    y = [f(view(X, :, i)) for i in 1:n] .+ 0.05 .* randn(rng, n)

    # Standardize the features (zero mean, unit std along the observations).
    μ = sum(X; dims=2) ./ n
    σ = sqrt.(sum(abs2, X .- μ; dims=2) ./ (n - 1))
    X = (X .- μ) ./ σ

    tr, te = 1:300, 301:400
    Xtr, ytr = X[:, tr], y[tr]
    Xte, yte = X[:, te], y[te]
    naive = _rmse(fill(sum(ytr) / length(ytr), length(te)), yte)

    # Full LS-SVR baseline (uses every training point) for comparison.
    base_model = LSSVR(γ=10.0, σ=1.0)
    base_fit = svmtrain(base_model, Xtr, ytr)
    base = _rmse(svmpredict(base_model, base_fit, Xte), yte)

    function fixed(m)
        Random.seed!(2024)                            # deterministic SV selection
        svm = FixedSizeSVR(γ=10.0, σ=1.0, subsample=m, iters=400)
        info = LeastSquaresSVM.factorization_entropy(svm, Xtr, ytr)
        fits = svmtrain(svm, info.matrices, info.target)
        ŷ = svmpredict(svm, fits, Xte)
        return (ŷ, _rmse(ŷ, yte))
    end

    ŷ15, rmse15 = fixed(15)
    ŷ30, rmse30 = fixed(30)
    ŷ60, rmse60 = fixed(60)

    # Predictions are finite and bounded — no 1/√λ blow-up.
    @test all(isfinite, ŷ30)
    @test maximum(abs, ŷ30) < 5 * maximum(abs, yte)

    # The regressor actually learns: far better than predicting the mean.
    @test rmse30 < 0.5 * naive

    # With only m = 60 prototypes it approaches the full LS-SVR baseline.
    @test rmse60 < 2 * base

    # Nyström convergence: more prototypes reduce the error.
    @test rmse60 < rmse15
end

@testset "Fixed Size: entropy-based active selection" begin
    rng = MersenneTwister(7)
    n, m = 300, 25
    X = randn(rng, 2, n)
    k = LeastSquaresSVM._choose_kernel(; kernel=:rbf, sigma=1.0, degree=0)
    ent(idx) = LeastSquaresSVM._quadratic_renyi_entropy(
        LeastSquaresSVM.kernelmatrix(k, view(X, :, idx); obsdim=2)
    )

    Random.seed!(99)
    H_sel, Ω_sel, idxs = LeastSquaresSVM._active_selection(k, X, n, m; iters=2000)

    # The selection returns a valid working set of m distinct training points.
    @test length(idxs) == m
    @test length(unique(idxs)) == m
    @test all(i -> 1 <= i <= n, idxs)

    # The returned entropy and Ω are consistent with the selected working set.
    @test Ω_sel ≈ LeastSquaresSVM.kernelmatrix(k, view(X, :, idxs); obsdim=2)
    @test H_sel ≈ ent(idxs)

    # Active selection maximizes entropy: it beats the average random subset.
    H_rand = sum(ent(randperm(n)[1:m]) for _ in 1:20) / 20
    @test H_sel > H_rand
end
