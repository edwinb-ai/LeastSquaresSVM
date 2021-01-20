# # Using different kernels
#
# Starting with version v0.6, two new kernels can be chosen: a **linear** kernel and a
# **polynomial** kernel.
# In this document we will see how to handle choosing a different kernel, and we'll
# showcase an example.
#
# First, we need to import all the necessary packages.
import Elysivm
using MLJ, MLJBase
using DataFrames, CSV
using CategoricalArrays, Random
using Plots
gr();
rng = MersenneTwister(812);

# For this example, we will create a very large classification problem. It is actually
# inspired from a similar classification problem from [`scikit-learn`](https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-unbalanced-py).
#
# The idea is to have a very large number of samples (10_000), and a large number of
# features (20).
X, y = make_blobs(10_000, 20; centers=2, cluster_std=[1.5, 0.5])

# The `make_blobs` function is very similar to that of [`scikit-learn`s](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs).
# The idea is to create circles, or clusters, and to classify between them.
#
# We'll use them to test our _linear_ kernel.

# We need to construct a `DataFrame` with the arrays created to better handle the data,
# as well as a better integration with `MLJ`.
df = DataFrame(X);
df.y = y;

# Recall that we need to change the primitive types of `Julia` to `scitypes`.
dfnew = coerce(df, autotype(df));

# We can then observe the first three columns, together with their new types.
first(dfnew, 3) |> pretty

# We should also check out the basic statistics of the dataset.
describe(dfnew, :mean, :std, :eltype)

# Recall that we also need to standardize the dataset, we can see here that the mean is
# close to zero, but not quite, and we also need an unitary standard deviation.

# Split the dataset into training and testing sets.
y, X = unpack(dfnew, ==(:y), colname -> true);
train, test = partition(eachindex(y), 0.75, shuffle=true, rng=rng);
stand1 = Standardizer();
X = transform(fit!(machine(stand1, X)), X);

# We should make sure that the features have mean close to zero and an unitary standard
# deviation.
describe(X |> DataFrame, :mean, :std, :eltype)

# For the case of a _linear_ kernel, no hyperparameter is needed. Instead, the only
# hyperparameter that needs to be adjusted is the ``\gamma`` value that is intrinsic
# of the least-squares formulation.

model = Elysivm.LSSVRegressor();
r1 = MLJBase.range(model, :σ, lower=7e-4, upper=1e-3);
r2 = MLJBase.range(model, :γ, lower=120.0, upper=130.0);
self_tuning_model = TunedModel(
    model=model,
    tuning=Grid(goal=400, rng=rng),
    resampling=CV(nfolds=5),
    range=[r1, r2],
    measure=MLJBase.rms,
    acceleration=CPUThreads(), # We use this to enable multithreading
);

# And now we proceed to train all the models and find the best one!
mach = MLJ.machine(self_tuning_model, X, y);
MLJBase.fit!(mach, rows=train, verbosity=0);
fitted_params(mach).best_model

# Having found the best hyperparameters for the regressor model we proceed to check how the
# model generalizes and we use the test set to check the performance.
ŷ = MLJBase.predict(mach, rows=test);
result = round(MLJBase.rms(ŷ, y[test]), sigdigits=4);
@show result # Check the result

# We can see that we did quite well. A value of 1, or close enough, is good. We expect it
# to reach a lower value, closer to zero, but maybe we needed more refinement in the grid
# search.
#
# We can also see a plot of the predicted and true values. The closer these dots are to the
# diagonal means that the model performed well.
scatter(ŷ, y[test], markersize=9)
r = range(minimum(y[test]), maximum(y[test]); length=length(test))
plot!(r, r, linewidth=9)
plot!(size=(3000, 2100))

# We can actually see that we are not that far off, maybe a little more search could
# definitely improve the performance of our model.
