using MLJ, MLJBase
using Random
using Elysivm

rng = MersenneTwister(951)

X, y = @load_iris
train, test = partition(eachindex(y), 0.6, shuffle=true, rng=rng)
model = LSSVClassifier()
r1 = range(model, :σ, lower=1, upper=20)
r2 = range(model, :γ, lower=1, upper=1000)
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
mach = MLJ.machine(pipe, X, y)
MLJ.fit!(mach, rows=train)
display(fitted_params(mach).deterministic_tuned_model.best_model)

results = MLJ.predict(mach, rows=test)
acc = MLJ.accuracy(results, y[test])
@show acc
