module LeastSquaresSVM

using LinearAlgebra
using KernelFunctions
using Krylov
using Tullio
using StatsBase
import LinearAlgebra.BLAS as BL
import MLJModelInterface
import ScientificTypes

# We declare a shorter name because there appears to be a bug with the `as` keyword
# for the `MLJModelInterface` module
const MMI = MLJModelInterface
# The same for `ScientificTypesBase`
const ST = ScientificTypes

export SVM,
    LSSVC,
    LSSVR,
    KernelRBF,
    svmtrain,
    svmtrain_mc,
    svmpredict,
    LSSVClassifier,
    LSSVRegressor,
    FixedSizeSVR,
    FixedSizeRegressor

include("types.jl")
include("utils.jl")
include("tools.jl")
include("training.jl")
include("fixed_training.jl")
include("mlj_interface.jl")
# Some precompilation for the Tullio tools
include("precompile.jl")
_precompile_tullio()

end
