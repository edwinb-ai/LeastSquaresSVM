using MLJ, MLJBase
using Random
using Elysivm

rng = MersenneTwister(95)

X, y = @load_iris
train, test = partition(eachindex(y), 0.6, shuffle=true, rng=rng)
model = LSSVClassifier()
sigma_values = [0.5, 5.0, 10.0, 15.0, 25.0, 50.0, 100.0, 250.0, 500.0]
gamma_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
r1 = range(model, :σ, values=sigma_values)
r2 = range(model, :γ, values=gamma_values)
self_tuning_model = TunedModel(
    model=model,
    # tuning=Grid(goal=400, rng=rng),
    resampling=StratifiedCV(nfolds=5),
    range=[r1, r2],
    measure=accuracy,
    acceleration=CPUThreads()
)
pipe = @pipeline(Standardizer(), self_tuning_model)
mach = MLJBase.machine(pipe, X, y)
MLJBase.fit!(mach, rows=train)
results = MLJBase.predict(mach, rows=test)
acc = MLJBase.accuracy(results, y[test])
@show acc
