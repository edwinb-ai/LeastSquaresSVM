using MLJ, MLJBase
using Random
using Elysivm

rng = MersenneTwister(953)

X, y = @load_iris
train, test = partition(eachindex(y), 0.6, shuffle=true, rng=rng)
model = LSSVClassifier()
r1 = range(model, :σ, lower=1e-3, upper=5e-3)
r2 = range(model, :γ, lower=140, upper=150)
self_tuning_model = TunedModel(
    model=model,
    tuning=Grid(goal=400, rng=rng),
    resampling=StratifiedCV(nfolds=5),
    range=[r1, r2],
    measure=accuracy,
    acceleration=CPUThreads(),
    repeats=10 # Add more resampling to be sure we are not biased
)
pipe = @pipeline(Standardizer(), self_tuning_model)
mach = MLJ.machine(pipe, X, y)
MLJ.fit!(mach, rows=train)
display(fitted_params(mach).deterministic_tuned_model.best_model)

results = MLJ.predict(mach, rows=test)
acc = MLJ.accuracy(results, y[test])
@show acc
