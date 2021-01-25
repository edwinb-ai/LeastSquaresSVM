```@meta
CurrentModule = LeastSquaresSVM
```

# LeastSquaresSVM

This is `LeastSquaresSVM`, a Least Squares Support Vector Machine (LSSVM) implementation in pure Julia.
It is meant to be used together with the fantastic [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/)
Machine Learning framework.

# Formulation

It is a re-formulation of the classical Support Vector Machine (SVM) formalism. In this case we attempt
to solve a least squares problem which is faster[^1], instead of the classic quadratic, convex optimization problem
that is solved in the original Support Vector Machine.

In the case of `LeastSquaresSVM` we use the conjugate gradient method, in particular the Lanczos version[^2] due to the fact
that we solve several linear systems which have the the following structure

```math
A \mathbf{x} = \mathbf{b}
```

where the matrix $A$ is symmetric.

This fact makes it a great candidate for the Lanczos algorithm, a
very fast, iterative procedure based on Krylov subspace methods. The implementation used here is that from the
[Krylov.jl](https://juliasmoothoptimizers.github.io/Krylov.jl/dev/) package.

# Rationale

SVM has been the most known and used formulation, but here are some pros and cons for using LSSVMs and `LeastSquaresSVM`.

## Advantages

The LSSVM is a great alternative to the classic SVM in the following things:

- Solving a linear system is much easier and faster than solving a quadratic optimization problem.
- Some useful properties from numerical linear algebra can be exploited in order to solve the new optimization problem.
- One can potentialy train of thousands or millions of instances using LSSVM, something that the classic SVM cannot do. This is possible using the _fixed size LSSVM_[^3].
- Less hyperparemeters to tune. LSSVM only has one intrinsic hyperparamter, whereas the SVM has at least two. This is without taking into account the kernel's hyperparameter.

## Disadvantages

But there are some important **shortcommings** for the LSSVM, namely:

- In contrast with the classic SVM, the decision function lack all _sparseness._ Every single dataset instance must be used to train the LSSVM. This can become troublesome for very large problems because all the instances must fit into memory.
- There is complete lack of interpretation for the _support vectors,_ which are the data instances that are used to construct the decision function. Because all data instances are used, every instance is effectively a support vector and this removes any interpretation for the importance of each instance on the model's performance.

# Bibliography

[^1]: Suykens, J. A., & Vandewalle, J. (1999). Least squares support vector machine classifiers. Neural processing letters, 9(3), 293-300.

[^2]: Fasano, G. (2007). Lanczos conjugate-gradient method and pseudoinverse computation on indefinite and singular systems. Journal of optimization theory and applications, 132(2), 267-285.

[^3]: Espinoza, M., Suykens, J. A., & De Moor, B. (2006). Fixed-size least squares support vector machines: A large scale application in electrical load forecasting. Computational Management Science, 3(2), 113-129.
