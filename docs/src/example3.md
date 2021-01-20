```@meta
EditURL = "<unknown>/src/examples/example3.jl"
```

# Using different kernels

Starting with version v0.6, two new kernels can be chosen: a **linear** kernel and a
**polynomial** kernel.
In this document we will see how to handle choosing a different kernel, and we'll
showcase an example.

First, we need to import all the necessary packages.

```julia
using Elysivm
using MLJ, MLJBase
using DataFrames, CSV
using CategoricalArrays, Random
using Plots
gr();
rng = MersenneTwister(812);
nothing #hide
```

For this example, we will create a large classification problem. It is actually
inspired from a similar classification problem from [`scikit-learn`](https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-unbalanced-py).

The idea is to have a very large number of features (5000), and a small number of
instances.
This has been [reported](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) as
as being a good use case or rule of thumb

> Whenever the number of features is _larger_ than the number of instances, use a
> linear kernel.

```julia
X, y = MLJ.make_blobs(500, 2_000; centers=2, cluster_std=[1.5, 0.5]);
nothing #hide
```

Of course, this is just to showcase the implementation within `Elysivm`. There are
actually better ways to handle this kind of problem, e.g. dimensionality-reduction
algorithms.

The `make_blobs` function is very similar to that of [`scikit-learn`s](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs).
The idea is to create circles, or clusters, and to classify between them.

We'll use them to test our _linear_ kernel.

We need to construct a `DataFrame` with the arrays created to better handle the data,
as well as a better integration with `MLJ`.

```julia
df = DataFrame(X);
df.y = y;
nothing #hide
```

Recall that we need to change the primitive types of `Julia` to `scitypes`.

```julia
dfnew = coerce(df, autotype(df));
nothing #hide
```

We can then observe the first three columns, together with their new types.
We'll just look at the first 8 features to avoid cluttering the space.

```julia
first(dfnew[:, 1:8], 3) |> pretty
```

```
┌────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│ x1         │ x2         │ x3         │ x4         │ x5         │ x6         │ x7         │ x8         │
│ Float64    │ Float64    │ Float64    │ Float64    │ Float64    │ Float64    │ Float64    │ Float64    │
│ Continuous │ Continuous │ Continuous │ Continuous │ Continuous │ Continuous │ Continuous │ Continuous │
├────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
│ 4.6069     │ -11.0476   │ 9.02517    │ 2.06608    │ 10.3952    │ 8.74835    │ 0.534613   │ -0.493674  │
│ -7.47441   │ 1.80263    │ -0.47754   │ 5.36791    │ -9.11338   │ -2.67125   │ -4.20552   │ -1.68881   │
│ -6.90708   │ 1.61288    │ -0.852302  │ 5.24683    │ -9.36584   │ -3.15882   │ -6.1735    │ -1.85193   │
└────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘

```

We should also check out the basic statistics of the dataset. We'll only use a small
subset as the data frame it quite large.

```julia
describe(dfnew[1:20, 1:10], :mean, :std, :eltype)
```

```
10×4 DataFrame
 Row │ variable  mean       std      eltype
     │ Symbol    Float64    Float64  DataType
─────┼────────────────────────────────────────
   1 │ x1         0.768153  6.07188  Float64
   2 │ x2        -5.57183   5.79931  Float64
   3 │ x3         5.38725   4.93904  Float64
   4 │ x4         3.94973   1.804    Float64
   5 │ x5         3.24621   9.60158  Float64
   6 │ x6         3.47164   4.90996  Float64
   7 │ x7        -1.08106   3.11007  Float64
   8 │ x8        -0.9725    1.10477  Float64
   9 │ x9         7.51147   1.06876  Float64
  10 │ x10       -0.735586  7.56975  Float64
```

Recall that we also need to standardize the dataset, we can see here that the mean is
close to zero, but not quite, and we also need an unitary standard deviation.

Split the dataset into training and testing sets.

```julia
y, X = unpack(dfnew, ==(:y), colname -> true);
train, test = partition(eachindex(y), 0.75, shuffle=true, rng=rng);
stand1 = Standardizer();
X = MLJBase.transform(MLJBase.fit!(MLJBase.machine(stand1, X)), X);
nothing #hide
```

```
┌ Info: Training [34mMachine{Standardizer} @835[39m.
└ @ MLJBase /home/edwin/.julia/packages/MLJBase/vwzmG/src/machines.jl:319

```

We should make sure that the features have mean close to zero and an unitary standard
deviation. Again, using only a small subset.

```julia
X_df = DataFrame(X)
describe(X_df[1:20, 1:10], :mean, :std, :eltype)
```

```
10×4 DataFrame
 Row │ variable  mean       std       eltype
     │ Symbol    Float64    Float64   DataType
─────┼─────────────────────────────────────────
   1 │ x1         0.29838   0.97679   Float64
   2 │ x2        -0.258993  0.964678  Float64
   3 │ x3         0.253288  1.00743   Float64
   4 │ x4        -0.341693  1.09002   Float64
   5 │ x5         0.311695  0.967953  Float64
   6 │ x6         0.390866  1.06769   Float64
   7 │ x7         0.248482  1.00987   Float64
   8 │ x8         0.299902  1.04662   Float64
   9 │ x9         0.335588  0.900707  Float64
  10 │ x10       -0.317464  0.977763  Float64
```

For the case of a _linear_ kernel, no hyperparameter is needed. Instead, the only
hyperparameter that needs to be adjusted is the ``\gamma`` value that is intrinsic
of the least-squares formulation. We will search for a good hyperparameter now.

We will use the `accuracy` as a metric. The accuracy is simply defined as

```math
\text{accuracy} = \frac{\text{number of correct predictions}}{\text{total number of predictions}}
```

Note that the accuracy is not always a good measure of classification, but it will do
fine on this dataset.

!!! warning
    Remember that the least-squares formulation uses **all** the data samples, so the
    following will actually consume at least > 6 GB of RAM. Do not run this on your
    hardware if you are not sure you have this kind of resources available.
    At the very least, replace `CPUThreads()` with `CPU1()` to disable multithreading.
    Methods to handle memory more efficiently will be available in future
    versions.

```julia
model = LSSVClassifier(kernel="linear");
r1 = range(model, :σ, lower=1.0, upper=1000.0);
self_tuning_model = TunedModel(
    model=model,
    tuning=Grid(goal=400, rng=rng),
    resampling=StratifiedCV(nfolds=5),
    range=[r1],
    measure=accuracy,
    acceleration=CPUThreads(), # We use this to enable multithreading
);
nothing #hide
```

And now we proceed to train all the models and find the best one!

```julia
mach = machine(self_tuning_model, X, y);
fit!(mach, rows=train, verbosity=0);
fitted_params(mach).best_model
```

```
LSSVClassifier(
    kernel = "linear",
    γ = 1.0,
    σ = 283.9248120300752,
    degree = 0)[34m @161[39m
```

Having found the best hyperparameters for the regressor model we proceed to check how the
model generalizes and we use the test set to check the performance.

```julia
ŷ = MLJBase.predict(mach, rows=test);
result = accuracy(ŷ, y[test])
@show result # Check th
```

```
1.0
```

We can see that we did quite well. A value of 1, or close enough, means the classifier
is _perfect._ That is, it can classify correctly between each class.

Finally, let us look at the so-called _confusion matrix._ This table shows us useful
information about the performance of our classifier.

Let us compute it first, and then we'll analyse it. Notice, however, that we need to
first coerce the types to `OrderedFactor` _scitypes_ in order for the confusion matrix
to be computed correctly.

```julia
ŷ = coerce(ŷ, OrderedFactor);
y_ordered = coerce(y[test], OrderedFactor);
confusion_matrix(ŷ, y_ordered)
```

```
              ┌───────────────────────────┐
              │       Ground Truth        │
┌─────────────┼─────────────┬─────────────┤
│  Predicted  │      1      │      2      │
├─────────────┼─────────────┼─────────────┤
│      1      │     52      │      0      │
├─────────────┼─────────────┼─────────────┤
│      2      │      0      │     73      │
└─────────────┴─────────────┴─────────────┘

```

The way you read the confusion matrix is the following. The main diagonal tells us how
many correct predictions were obtained by the classifier for both classes.
On the other hand, the other values are the following

- The _upper right_ value is known as the **false positive**. This is the number of instances that were classified as belonging to a given class, when actually they were instances of the other one. An example would be if we had an instance ``x_1`` which belonged to the class `b`, but the classifier would have predicted it actually belonged to class `a`.
- The _lower left_ value is known as the **false negative**. This is the number of instances classified as _not_ belonging to a given class, when they actually belonged to a class. An example would be if we had an instance ``x_2`` which belonged to class `a`, and the classifier actually predicted it belonged to class `b`.

It might be a little bit confusing, so a good starting point for more information on the
subject is the excellent [Wikipedia article.](https://en.wikipedia.org/wiki/Confusion_matrix)
You might also be interested in the following [document](https://developers.google.com/machine-learning/crash-course/classification/accuracy) from a Google's Machine Learning Crash Course.

In this case, we can see that no false negative or positive values were found, which
means that the classifier did outstandingly good.
Normally, we can expect to have at least some percentage of false negative or positives.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

