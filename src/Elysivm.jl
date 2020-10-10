module Elysivm

using LinearAlgebra
using Distances

include("types.jl")
export LSSVC, KernelRBF

include("training.jl")
export fit!

end
