function _precompile_tullio()
    precompile(extern_prod, (Vector{Float64}, Vector{Float64},))
    precompile(pairwise_mul!, (Matrix{Float64}, Matrix{Float64},))
    precompile(pairwise_diff, (Vector{Float64}, Vector{Float64},))
    precompile(prod_reduction, (Matrix{Float64}, Vector{Float64}, Vector{Float64},))
    precompile(prod_reduction, (Matrix{Float64}, Vector{Float64},))
end
