"""
Type to represent frequentist regression models returned by `fitmodel` functions. This type is used internally by the package to represent all frequentist regression models.
"""
struct FrequentistRegression{RegressionType}
    model
    formula
end

"""
```julia
FrequentistRegression(::Symbol, model, formula)
```

Constructor for `FrequentistRegression`. `model` can be any regression model. Used by `fitmodel` functions to return a frequentist regression model containers.
"""
FrequentistRegression(RegressionType::Symbol, model, formula) = FrequentistRegression{RegressionType}(model, formula)

"""
Type to represent bayesian regression models returned by `fitmodel` functions. This type is used internally by the package to represent all bayesian regression models.
"""
struct BayesianRegression{RegressionType}
    chain
end

"""
```julia
BayesianRegression(::Symbol, chain)
```

Constructor for `BayesianRegression`. `model` can be any regression model. Used by `fitmodel` functions to return a frequentist regression model containers.
"""
BayesianRegression(RegressionType::Symbol, chain) = BayesianRegression{RegressionType}(chain)

# Print Messages
include("print.jl")

# Frequentist getter functions
include("frequentist/getter.jl")

# Frequentist Regression Models
include("frequentist/linear_regression.jl")
include("frequentist/logistic_regression.jl")
include("frequentist/negativebinomial_regression.jl")
include("frequentist/poisson_regression.jl")

# Bayesian Regression Models
include("bayesian/linear_regression.jl")
include("bayesian/logistic_regression.jl")
include("bayesian/negativebinomial_regression.jl")
include("bayesian/poisson_regression.jl")

"""
```julia
   @fitmodel(formula, args...)
```

Macro for calling `fitmodel` without using `@formula` to create the formula. 

# Example
```julia
using CRRao, RDatasets
sanction = dataset("Zelig", "sanction")
model = @fitmodel(Num ~ Target + Coop + NCost, sanction, NegBinomRegression())
```
"""
macro fitmodel(formula, args...)
    quote
        fitmodel(@formula($formula), $(map(esc, args)...))
    end
end
