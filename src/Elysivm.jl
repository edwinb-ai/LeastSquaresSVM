module Elysivm

using LinearAlgebra
using Distances
using Krylov
import MLJModelInterface

const MMI = MLJModelInterface

include("types.jl")
export LSSVC, KernelRBF

include("training.jl")
export svmtrain, build_omega, svmpredict

include("mlj_interface.jl")
export LSSVClassifier

end
