```@meta
EditURL = "<unknown>/src/examples/example1.jl"
```

# Classification of the Wisconsin breast cancer dataset

In this case study we will deal with the Wisconsin breast cancer dataset which can be
browsed freely on the [UCI website](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

In particular, this dataset contains *10 features* and 699 instances. In the work we
will do here, however, we will skip some instances due to some missing values.

The dataset contains only two classes, and the purpose is to use all ten features to
answer a simple question:
> Does the subject have a benign or malign tumor?
To answer this question, we will train a Least Squares Support Vector Machine as
implemented in `Elysivm`.

First, we need to import all the necessary packages.

```@example example1
using MLJ, MLJBase
using DataFrames, CSV
using CategoricalArrays
using Random, Statistics
using Elysivm
```

We then need to specify a seed to enable reproducibility of the results.

```@example example1
rng = MersenneTwister(801239);
nothing #hide
```

Here we are creating a list with all the headers.

```@example example1
headers = [
	"id", "Clump Thickness",
	"Uniformity of Cell Size", "Uniformity of Cell Shape",
	"Marginal Adhesion", "Single Epithelial Cell Size",
	"Bare Nuclei", "Bland Chromatin",
	"Normal Nucleoli", "Mitoses", "class"
];
nothing #hide
```

We define the path were the dataset is located

```@example example1
path = joinpath("examples", "wbc.csv");
nothing #hide
```

We load the csv file and convert it to a `DataFrame`. Note that we are specifying
to the file reader to replace the string `?` to a `missing` value. This dataset contains
the the string `?` when there is a value missing.

```@example example1
data = CSV.File(path; header=headers, missingstring="?") |> DataFrame;
nothing #hide
```

We can display the first 10 rows from the dataset

```@example example1
first(data, 10)
```

We can see that all the features have been added correctly, we can see that we have
an unncessary feature called `id`, so we will remove it.

```@example example1
select!(data, Not(:id));
nothing #hide
```

We also need to remove all the missing data from the `DataFrame`

```@example example1
data = dropmissing(data);
nothing #hide
```

The `class` column should be of type `categorical`, following the `MLJ` API, so we
encode it here.

```@example example1
transform!(data, :class => categorical, renamecols=false);
nothing #hide
```

Check statistics per column.

```@example example1
describe(data)
```

Split the dataset into training and testing.

```@example example1
y, X = unpack(data, ==(:class), colname -> true);
nothing #hide
```

We will use only 2/3 for training.

```@example example1
train, test = partition(eachindex(y), 2 / 3, shuffle=true, rng=rng);
nothing #hide
```

Always remove mean and set the standard deviation to 1.0 when dealing with SVMs.

```@example example1
stand1 = Standardizer(count=true);
X = MLJBase.transform(fit!(machine(stand1, X)), X);
nothing #hide
```

Check statistics per column again to ensure standardization, but remember to do it now
with the `X` matrix.

```@example example1
describe(X)
```

Good, now every column has a mean very close to zero, so the standardization was
done correctly.

We now create our model with `Elysivm`

```@example example1
model = Elysivm.LSSVClassifier();
nothing #hide
```

These are the values for the hyperparameter grid search. We need to find the best subset
from this set of parameters.
Although I will not do this here, the best approach is to find a set of good hyperparameters
and then refine the search space around that set. That way we can ensure we will always get
the best results.

```@example example1
sigma_values = [0.5, 5.0, 10.0, 15.0, 25.0, 50.0, 100.0, 250.0, 500.0];
r1 = MLJBase.range(model, :σ, values=sigma_values);
gamma_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0];
r2 = MLJBase.range(model, :γ, values=gamma_values);
nothing #hide
```

We now create a `TunedModel` that will use a 10-folds stratified cross validation scheme
in order to find the best set of hyperparameters. The stratification is needed because
the classes are somewhat imbalanced:
- Benign: 458 (65.5%)
- Malignant: 241 (34.5%)

```@example example1
self_tuning_model = TunedModel(
    model=model,
    tuning=Grid(rng=rng),
    resampling=StratifiedCV(nfolds=10),
    range=[r1, r2],
    measure=accuracy,
    acceleration=CPUThreads(), # We use this to enable multithreading
);
nothing #hide
```

Once the best model is found, we create a `machine` with it, and fit it

```@example example1
mach = machine(self_tuning_model, X, y);
fit!(mach, rows=train, verbosity=0);
nothing #hide
```

We can now show the best hyperparameters found.

```@example example1
fitted_params(mach).best_model
```

And we test the trained model. We expect somewhere around 94%-96% accuracy.

```@example example1
results = predict(mach, rows=test);
acc = accuracy(results, y[test]);
nothing #hide
```

Show the accuracy for the testing set

```@example example1
println(acc * 100.0)
```

As you can see, it is fairly easy to use `Elysivm` together with MLJ. We got a good
accuracy result and this proves that the implementation is actually correct. This
dataset is commonly used as a benchmark dataset to test new algorithms.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

