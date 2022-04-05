module CRRao

using DataFrames, GLM, Turing, StatsModels, StatsBase, Distributions

export CRRao

include("general_stats.jl")
include("mcmc.jl")
include("optimisation.jl")
include("crrao/crrao.jl")

end



