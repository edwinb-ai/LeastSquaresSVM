module Elysivm

using LinearAlgebra
using KernelFunctions
using Krylov
import MLJModelInterface

const MMI = MLJModelInterface

export SVM, LSSVC, LSSVR, KernelRBF
export svmtrain, svmtrain_mc, svmpredict
export LSSVClassifier, LSSVRegressor

include("types.jl")
include("utils.jl")
include("training.jl")
include("mlj_interface.jl")

end
