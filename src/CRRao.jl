"""
CRRao is a Julia package that implements the Statistical models. The implementation 
   of Statistical models become straightforward for most Julia users 
   with the help of this package. This is going to be wrapper package;
   leveraging the strength of wonderful Julia packages that already exists, 
   such as StatsBase, StatsModels, Distributions,GLM, Turing, DataFrames,
   LinearAlgebra, etc.

   
   CRRao is a consistent framework through which callers interact with 
   a large suite of models. For the end-user, it reduces the cost and complexity 
   of estimating/training statistical models. It offers convenient guidelines through 
   which development of additional statistical models can take place 
   in the future.

   We follow framework which makes contribution to this package easy.

   **Note**: You can read more about **Prof C.R. Rao** [Click Here](https://en.wikipedia.org/wiki/C._R._Rao)
   
"""
module CRRao

using DataFrames, GLM, Turing, StatsModels
using StatsBase, Distributions, LinearAlgebra
using Optim, NLSolversBase, Random

struct NegBinomRegression end
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


export LinearRegression, LogisticRegression, PoissonRegression, NegBinomRegression
export Prior_Ridge, Prior_Laplace, Prior_Cauchy, Prior_TDist, Prior_Uniform
export Logit, Probit, Cloglog, Cauchit, fitmodel, @fitmodel

include("random_number_generator.jl")
include("general_stats.jl")
include("fitmodel.jl")

end



