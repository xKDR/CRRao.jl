"""
Type to represent frequentist models returned by `fitmodel` functions. This type is used internally by the package to represent all frequentist regression models.
"""
struct FrequentistRegression{RegressionType}
   model
end

"""
```julia
FrequentistRegression(::Symbol, model)
```

Constructor for `FrequentistRegression`. `model` can be any regression model. Used by `fitmodel` functions to return a frequentist regression model containers.
"""
FrequentistRegression(RegressionType::Symbol, model) = FrequentistRegression{RegressionType}(model)

# Print Messages
include("print.jl")

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
