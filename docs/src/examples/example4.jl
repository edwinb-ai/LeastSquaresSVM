using MLJ, MLJBase
using Random
using Elysivm

rng = MersenneTwister(95)

X, y = @load_iris
train, test = partition(eachindex(y), 0.6, shuffle=true, rng=rng)
model = LSSVClassifier()
r1 = range(model, :σ, lower=12, upper=15)
r2 = range(model, :γ, lower=1000, upper=2000)
self_tuning_model = TunedModel(
    model=model,
    # tuning=Grid(goal=400, rng=rng),
    tuning=RandomSearch(),
    resampling=StratifiedCV(nfolds=5),
    range=[r1, r2],
    measure=accuracy,
    acceleration=CPUThreads(),
    n=500
)
pipe = @pipeline(Standardizer(), self_tuning_model)
mach = MLJBase.machine(pipe, X, y)
MLJBase.fit!(mach, rows=train)
display(fitted_params(mach).deterministic_tuned_model.best_model)

results = MLJBase.predict(mach, rows=test)
acc = MLJBase.accuracy(results, y[test])
@show acc
