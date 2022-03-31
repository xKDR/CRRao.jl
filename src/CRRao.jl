module CRRao

using DataFrames, CSV, HTTP, ShiftedArrays, Plots, GLM, Turing, StatsModels, StatsBase, StatsPlots, LazyArrays

export CRRao

# Write your package code here.
include("general_stats.jl")
include("mcmc.jl")
include("optimisation.jl")
include("crrao/crrao.jl")

end
