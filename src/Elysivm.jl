module Elysivm

using KernelFunctions

include("types.jl")
export LSSVC

include("training.jl")
fit!

end
