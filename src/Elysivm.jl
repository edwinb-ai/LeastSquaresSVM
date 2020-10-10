module Elysivm

using KernelFunctions

include("types.jl")
export LSSVM

include("training.jl")
fit!

end
