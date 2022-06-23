module CRRao

using DataFrames, GLM, Turing, StatsModels, StatsBase
using StatsBase, Distributions, LinearAlgebra
using Optim, NLSolversBase, Random

"""
```julia
LinearRegression
```

Type representing the Linear Regression model class.
"""
struct LinearRegression end

"""
```julia
LogisticRegression
```

Type representing the Logistic Regression model class.
"""
struct LogisticRegression end

"""
```julia
NegBinomRegression
```

Type representing the Negative Binomial Regression model class.
"""
struct NegBinomRegression end

"""
```julia
PoissonRegression
```

Type representing the Poisson Regression model class.
"""
struct PoissonRegression end



struct Prior_Ridge end
struct Prior_Laplace end
struct Prior_Cauchy end
struct Prior_TDist end
struct Prior_Uniform end

"""
```julia
CRRaoLink
```

Abstract type representing link functions which are used to dispatch to appropriate calls.
"""
abstract type CRRaoLink end

"""
```julia
Logit <: CRRaoLink
```

A type representing the Logit link function, which is defined by the formula

```math
z\\mapsto \\dfrac{1}{1 + \\exp(-z)}
```
"""
struct Logit <: CRRaoLink
    link::Function
end

Logit() = Logit(Logit_Link)

"""
```julia
Probit <: CRRaoLink
```

A type representing the Probit link function, which is defined by the formula

```math
z\\mapsto \\mathbb{P}[Z\\le z]
```

where ``Z\\sim \\text{Normal}(0, 1)``.
"""
struct Probit <: CRRaoLink
    link::Function
end

Probit() = Probit(Probit_Link)

"""
```julia
Cloglog <: CRRaoLink
```

A type representing the Cloglog link function, which is defined by the formula 

```math
z\\mapsto 1 - \\exp(-\\exp(z))
```
"""
struct Cloglog <: CRRaoLink
    link::Function
end

Cloglog() = Cloglog(Cloglog_Link)

"""
```julia
Cauchit <: CRRaoLink
```

A type representing the Cauchit link function, which is defined by the formula

```math
z\\mapsto \\dfrac{1}{2} + \\dfrac{\\text{atan}(z)}{\\pi}
```
"""
struct Cauchit <: CRRaoLink
    link::Function
end

Cauchit() = Cauchit(Cauchit_Link)

export LinearRegression, LogisticRegression, PoissonRegression, NegBinomRegression
export Prior_Ridge, Prior_Laplace, Prior_Cauchy, Prior_TDist, Prior_Uniform
export CRRaoLink, Logit, Probit, Cloglog, Cauchit, fitmodel, @fitmodel
export coeftable, r2, adjr2, loglikelihood, aic, bic, sigma, predict, residuals, cooksdistance
export FrequentistRegression, BayesianRegression

include("random_number_generator.jl")
include("general_stats.jl")
include("fitmodel.jl")

end
