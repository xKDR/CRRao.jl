"""
```julia
FrequentistRegression{RegressionType}
```

Type to represent frequentist regression models returned by `fit` functions. This type is used internally by the package to represent all frequentist regression models. `RegressionType` is a `Symbol` representing the model class.
"""
struct FrequentistRegression{RegressionType} <: RegressionModel
    model
    formula::FormulaTerm
    link
end

"""
```julia
FrequentistRegression(::Symbol, model, formula, link = GLM.IdentityLink)
```

Constructor for `FrequentistRegression`. `model` can be any regression model. Used by `fit` functions to return a frequentist regression model container.
"""
FrequentistRegression(RegressionType::Symbol, model, formula, link = GLM.IdentityLink) = FrequentistRegression{RegressionType}(model, formula, link)

"""
```julia
BayesianRegressionMCMC{RegressionType}
```

Type to represent bayesian regression models (using MCMC) returned by `fit` functions. This type is used internally by the package to represent all bayesian regression models using MCMC. `RegressionType` is a `Symbol` representing the model class.
"""
struct BayesianRegressionMCMC{RegressionType} <: RegressionModel
    chain
    formula::FormulaTerm
    link
end

"""
```julia
BayesianRegressionMCMC(::Symbol, chain)
```

Constructor for `BayesianRegressionMCMC`. `model` can be any regression model. Used by `fit` functions to return a bayesian regression model container.
"""
BayesianRegressionMCMC(RegressionType::Symbol, chain, formula, link = Identity()) = BayesianRegressionMCMC{RegressionType}(chain, formula, link)

"""
```julia
BayesianRegressionVI{RegressionType}
```

Type to represent bayesian regression models (using VI) returned by `fit` functions. This type is used internally by the package to represent all bayesian regression models using VI. `RegressionType` is a `Symbol` representing the model class.
"""
struct BayesianRegressionVI{RegressionType} <: RegressionModel
    dist
    formula::FormulaTerm
    symbol_to_range
    link
end

"""
```julia
BayesianRegressionVI(::Symbol, dist)
```

Constructor for `BayesianRegressionVI`. `model` can be any regression model. Used by `fit` functions to return a bayesian regression model container.
"""
BayesianRegressionVI(RegressionType::Symbol, dist, formula, symbol_to_range, link = Identity()) = BayesianRegressionVI{RegressionType}(dist, formula, symbol_to_range, link)

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

# Bayesian getter functions
include("bayesian/getter.jl")
