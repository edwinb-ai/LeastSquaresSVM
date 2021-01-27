module LeastSquaresSVM

using LinearAlgebra
using KernelFunctions
using Krylov
using Tullio
import MLJModelInterface

const MMI = MLJModelInterface
const BL = LinearAlgebra.BLAS

export SVM, LSSVC, LSSVR, KernelRBF, svmtrain, svmtrain_mc, svmpredict, LSSVClassifier,
LSSVRegressor

include("types.jl")
include("utils.jl")
include("tools.jl")
include("training.jl")
include("mlj_interface.jl")

end
