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

```julia
using MLJ, MLJBase
using DataFrames, CSV
using CategoricalArrays
using Random, Statistics
using Elysivm
```

We then need to specify a seed to enable reproducibility of the results.

```julia
rng = MersenneTwister(801239);

```

Here we are creating a list with all the headers.

```julia
headers = [
	"id", "Clump Thickness",
	"Uniformity of Cell Size", "Uniformity of Cell Shape",
	"Marginal Adhesion", "Single Epithelial Cell Size",
	"Bare Nuclei", "Bland Chromatin",
	"Normal Nucleoli", "Mitoses", "class"
];

```

We define the path were the dataset is located

```julia
path = joinpath("src", "examples", "wbc.csv");

```

We load the csv file and convert it to a `DataFrame`. Note that we are specifying
to the file reader to replace the string `?` to a `missing` value. This dataset contains
the the string `?` when there is a value missing.

```julia
data = CSV.File(path; header=headers, missingstring="?") |> DataFrame;

```

We can display the first 10 rows from the dataset

```julia
first(data, 10)
```

```
10×11 DataFrame
 Row │ id       Clump Thickness  Uniformity of Cell Size  Uniformity of Cell Shape  Marginal Adhesion  Single Epithelial Cell Size  Bare Nuclei  Bland Chromatin  Normal Nucleoli  Mitoses  class
     │ Int64    Int64            Int64                    Int64                     Int64              Int64                        Int64?       Int64            Int64            Int64    Int64
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 1000025                5                        1                         1                  1                            2            1                3                1        1      2
   2 │ 1002945                5                        4                         4                  5                            7           10                3                2        1      2
   3 │ 1015425                3                        1                         1                  1                            2            2                3                1        1      2
   4 │ 1016277                6                        8                         8                  1                            3            4                3                7        1      2
   5 │ 1017023                4                        1                         1                  3                            2            1                3                1        1      2
   6 │ 1017122                8                       10                        10                  8                            7           10                9                7        1      4
   7 │ 1018099                1                        1                         1                  1                            2           10                3                1        1      2
   8 │ 1018561                2                        1                         2                  1                            2            1                3                1        1      2
   9 │ 1033078                2                        1                         1                  1                            2            1                1                1        5      2
  10 │ 1033078                4                        2                         1                  1                            2            1                2                1        1      2
```

We can see that all the features have been added correctly, we can see that we have
an unncessary feature called `id`, so we will remove it.

```julia
select!(data, Not(:id));

```

We also need to remove all the missing data from the `DataFrame`

```julia
data = dropmissing(data);

```

The `class` column should be of type `categorical`, following the `MLJ` API, so we
encode it here.

```julia
transform!(data, :class => categorical, renamecols=false);

```

Check statistics per column.

```julia
describe(data)
```

```
10×7 DataFrame
 Row │ variable                     mean     min  median  max  nmissing  eltype
     │ Symbol                       Union…   Any  Union…  Any  Int64     DataType
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Clump Thickness              4.44217  1    4.0     10          0  Int64
   2 │ Uniformity of Cell Size      3.15081  1    1.0     10          0  Int64
   3 │ Uniformity of Cell Shape     3.21523  1    1.0     10          0  Int64
   4 │ Marginal Adhesion            2.83016  1    1.0     10          0  Int64
   5 │ Single Epithelial Cell Size  3.23426  1    2.0     10          0  Int64
   6 │ Bare Nuclei                  3.54466  1    1.0     10          0  Int64
   7 │ Bland Chromatin              3.4451   1    3.0     10          0  Int64
   8 │ Normal Nucleoli              2.86969  1    1.0     10          0  Int64
   9 │ Mitoses                      1.60322  1    1.0     10          0  Int64
  10 │ class                                 2            4           0  CategoricalValue{Int64,UInt32}
```

Split the dataset into training and testing.

```julia
y, X = unpack(data, ==(:class), colname -> true);

```

We will use only 2/3 for training.

```julia
train, test = partition(eachindex(y), 2 / 3, shuffle=true, rng=rng);

```

Always remove mean and set the standard deviation to 1.0 when dealing with SVMs.

```julia
stand1 = Standardizer(count=true);
X = MLJBase.transform(fit!(machine(stand1, X)), X);

```

```
┌ Info: Training [34mMachine{Standardizer} @314[39m.
└ @ MLJBase /home/edwin/.julia/packages/MLJBase/5TNcr/src/machines.jl:319

```

Check statistics per column again to ensure standardization, but remember to do it now
with the `X` matrix.

```julia
describe(X)
```

```
9×7 DataFrame
 Row │ variable                     mean          min        median     max      nmissing  eltype
     │ Symbol                       Float64       Float64    Float64    Float64  Int64     DataType
─────┼──────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Clump Thickness               6.90029e-17  -1.2203    -0.156754  1.97033         0  Float64
   2 │ Uniformity of Cell Size      -3.5111e-17   -0.701698  -0.701698  2.23454         0  Float64
   3 │ Uniformity of Cell Shape     -7.44483e-17  -0.74123   -0.74123   2.27023         0  Float64
   4 │ Marginal Adhesion             9.96437e-17  -0.638897  -0.638897  2.50294         0  Float64
   5 │ Single Epithelial Cell Size  -5.29103e-17  -1.00503   -0.555202  3.0434          0  Float64
   6 │ Bare Nuclei                  -2.77962e-17  -0.698341  -0.698341  1.77157         0  Float64
   7 │ Bland Chromatin               6.11192e-17  -0.998122  -0.181694  2.6758          0  Float64
   8 │ Normal Nucleoli              -1.24677e-16  -0.612478  -0.612478  2.33576         0  Float64
   9 │ Mitoses                       2.17818e-17  -0.348145  -0.348145  4.84614         0  Float64
```

Good, now every column has a mean very close to zero, so the standardization was
done correctly.

We now create our model with `Elysivm`

```julia
model = Elysivm.LSSVClassifier();

```

These are the values for the hyperparameter grid search. We need to find the best subset
from this set of parameters.
Although I will not do this here, the best approach is to find a set of good hyperparameters
and then refine the search space around that set. That way we can ensure we will always get
the best results.

```julia
sigma_values = [0.5, 5.0, 10.0, 15.0, 25.0, 50.0, 100.0, 250.0, 500.0];
r1 = MLJBase.range(model, :σ, values=sigma_values);
gamma_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0];
r2 = MLJBase.range(model, :γ, values=gamma_values);

```

We now create a `TunedModel` that will use a 10-folds stratified cross validation scheme
in order to find the best set of hyperparameters. The stratification is needed because
the classes are somewhat imbalanced:
- Benign: 458 (65.5%)
- Malignant: 241 (34.5%)

```julia
self_tuning_model = TunedModel(
    model=model,
    tuning=Grid(rng=rng),
    resampling=StratifiedCV(nfolds=10),
    range=[r1, r2],
    measure=accuracy,
    acceleration=CPUThreads(), # We use this to enable multithreading
);

```

Once the best model is found, we create a `machine` with it, and fit it

```julia
mach = machine(self_tuning_model, X, y);
fit!(mach, rows=train, verbosity=0);

```

We can now show the best hyperparameters found.

```julia
fitted_params(mach).best_model
```

```
LSSVClassifier(
    kernel = :rbf,
    γ = 0.01,
    σ = 0.5,
    degree = 0)[34m @717[39m
```

And we test the trained model. We expect somewhere around 94%-96% accuracy.

```julia
results = predict(mach, rows=test);
acc = accuracy(results, y[test]);

```

Show the accuracy for the testing set

```julia
println(acc * 100.0)
```

```
94.73684210526316

```

As you can see, it is fairly easy to use `Elysivm` together with MLJ. We got a good
accuracy result and this proves that the implementation is actually correct. This
dataset is commonly used as a benchmark dataset to test new algorithms.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

