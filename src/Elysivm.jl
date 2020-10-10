module Elysivm

using LinearAlgebra
using Distances
using Krylov

include("types.jl")
export LSSVC, KernelRBF

include("training.jl")
export fit!, build_omega, predict!

end
