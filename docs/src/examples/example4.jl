# # Multiclass classification
#
# As of version `v0.7` and over, the `LSSVClassifier` can handle multiclass classification
# problems.
# It does so by using the so-called _one-vs-one approach_. This is described as follows:
#
# - For a classification problem with ``k`` classes, one trains ``k(k-1)/2`` binary classifiers.
# - Each binary classifier deals with every possible class pair combination. For instance, for a three class classification problem, we need to train three classifiers. The first will deal with the (1, 2) binary classification problem. The second with the (1, 3), while the third classifier will deal with the (2, 3) class pair.
# - We then save all these parameters, and on the prediction step, we ask every classifier to predict a given data instance. Care must be taken to ensure that the class pairs are decoded correctly.
# - A _voting scheme_ is then applied to these predictions. The class that has the largest number of votes when a pediction is carried out is the class that is taken as the predicted class. In the important case of a tie, the predicted class is always that of the lowest index. This is obviously not a great strategy, but it's simple enough to break the tie. See[^1] to read more about this strategy.
#
# Here, we deal with the binary classification problems with the underlying implementation
# of the `LSSVClassifier` and finally pool all of our results together. We then apply
# the voting scheme and we are done.
#
# In this example we will showcase this ability on the very famous and not so difficult
# problem of the [Iris flower dataset.](https://en.wikipedia.org/wiki/Iris_flower_data_set)
#
# This problem is a 3-class classification problem, with 4 features and a total of 150
# samples; 50 for each class.
# It is expected that with a good set of hyperparameters, our implementation should easily
# obtain a 100% accuracy classification rate on this problem.
#
# Let us begin by importing our packages and setting an RNG to ensure reproducibility.
#
using MLJ, MLJBase
using Random
using Elysivm

rng = MersenneTwister(957);

# The Iris dataset is so famous, it is included in the `MLJ.jl` package. By using the
# very convenient macro `@load_iris` we can obtain the full dataset, as follows.
X, y = @load_iris;

# Then, we split the dataset. We have said before that the datset is small, so we will
# keep this in mind. A 60-40 train-test split should suffice.
train, test = partition(eachindex(y), 0.6, shuffle=true, rng=rng);

# We will now instantiate our model. Recall that the default kernel is the RBF kernel.
model = LSSVClassifier();

# Now we set some ranges for the hyperparameters and also instantiate a self-tuning model
# Note however that a `repeats` option is now used. What this does is to repeat the
# resampling of the dataset that many times.
# This is just to ensure that the model is learning correctly and not being biased.
r1 = range(model, :σ, lower=1e-2, upper=1e-1);
r2 = range(model, :γ, lower=140, upper=150);
self_tuning_model = TunedModel(
    model=model,
    tuning=Grid(goal=200, rng=rng),
    resampling=StratifiedCV(nfolds=5),
    range=[r1, r2],
    measure=accuracy,
    acceleration=CPUThreads(),
    repeats=10 # Add more resampling to be sure we are not biased
);

# We will use a pipeline to first standardize the data and then feed it to the model.
pipe = @pipeline(Standardizer(), self_tuning_model);
mach = MLJ.machine(pipe, X, y);

# It's time to train. Remeber that if you are going to use multithreading for the
# hyperparameter search, you need to start your `Julia` session with the environment
# variable `JULIA_NUM_THREADS` set to a different number than 1.
MLJ.fit!(mach, rows=train, verbosity=0);

# We want to see the best set of hyperparameters.
fitted_params(mach).deterministic_tuned_model.best_model

# Great! Let us see how good was the model. Let's check with the test set.
results = MLJ.predict(mach, rows=test);
acc = MLJ.accuracy(results, y[test]);
acc

# Fantastic! A 100% accuracy, as was expected.
# As a good measure, let us inspect the confusion matrix of the classification problem.
results = coerce(results, OrderedFactor);
y_ordered = coerce(y[test], OrderedFactor);
confusion_matrix(results, y_ordered)

# ## Conclusions
#
# Again, just as expected, the result is perfect. But do note that this does not mean that
# the implementation or the model is the best there is. There are other models and
# implementations that should be checked along with this one.
#
# As a good rule of thumb, when doing machine learning, always try to train _at least_
# one more model to compare performance.
#
# [^1]: Chih-Wei Hsu and Chih-Jen Lin (2002) ‘A comparison of methods for multiclass support vector machines’, IEEE Transactions on Neural Networks, 13(2), pp. 416. doi: 10.1109/72.991427.
#
