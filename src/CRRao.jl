module CRRao

using DataFrames, GLM, Turing, StatsModels
using StatsBase, Distributions, LinearAlgebra
using Optim, NLSolversBase


struct PoissonRegression end
struct LinearRegression end
struct LogisticRegression end
struct Prior_Ridge end
struct Prior_Laplace end
struct Prior_Cauchy end
struct Prior_TDist end
struct Prior_Uniform end
struct Logit end
struct Probit end
struct Cloglog end
struct Cauchit end



export LogisticRegression, LinearRegression, PoissonRegression
export Prior_Ridge, Prior_Laplace, Prior_Cauchy, Prior_TDist, Prior_Uniform
export Logit, Probit, Cloglog, Cauchit, fitmodel, analysis_Poisson_Regression

include("general_stats.jl")
include("mcmc.jl")
include("optimisation.jl")
include("logistic_regression.jl")
include("fitmodel.jl")

end



