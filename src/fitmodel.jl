"""
Type to represent frequentist models returned by `fitmodel` functions. This type is used internally by the package to represent all frequentist regression models.
"""
struct FrequentistRegressionModel{RegressionType}
   model::StatsBase.RegressionModel
end

"""
```julia
FrequentistRegressionModel(::Symbol, ::StatsBase.RegressionModel)
```

Constructor for `FrequentistRegressionModel`. Used by `fitmodel` functions to return a frequentist regression model.
"""
FrequentistRegressionModel(RegressionType::Symbol, model::StatsBase.RegressionModel) = FrequentistRegressionModel{RegressionType}(model)

# Including fitmodel definitions

## Regression Models
include("fitmodel_defs/linear_regression.jl")
include("fitmodel_defs/logistic_regression.jl")
include("fitmodel_defs/negativebinomial_regression.jl")
include("fitmodel_defs/poisson_regression.jl")

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
