module CRRao

using DataFrames, GLM, Turing, StatsModels, StatsBase, Distributions

export fitmodel

include("general_stats.jl")
include("mcmc.jl")
include("optimisation.jl")
include("fitmodel.jl")

end



