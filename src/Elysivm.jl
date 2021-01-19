module Elysivm

using LinearAlgebra
using KernelFunctions
using Krylov
import MLJModelInterface

const MMI = MLJModelInterface

export LSSVC, LSSVR, KernelRBF
export svmtrain, svmpredict
export LSSVClassifier, LSSVRegressor

include("types.jl")
include("utils.jl")
include("training.jl")
include("mlj_interface.jl")

end
