module Elysivm

using LinearAlgebra
using KernelFunctions
using Krylov
import MLJModelInterface

const MMI = MLJModelInterface

include("types.jl")
export LSSVC, LSSVR, KernelRBF

include("training.jl")
export svmtrain, svmpredict

include("mlj_interface.jl")
export LSSVClassifier, LSSVRegressor

end
