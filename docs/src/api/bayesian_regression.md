# Bayesian Regression Models

```@docs
BayesianRegression
```

## Linear Regression
```@docs
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Ridge, h::Float64 = 0.01, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Laplace, h::Float64 = 0.01, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Cauchy, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_TDist, h::Float64 = 2.0, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Uniform, h::Float64 = 0.01, sim_size::Int64 = 10000)
```

## Logistic Regression
```@docs
```

## Negative Binomial Regression
```@docs
```

## Poisson Regression
```@docs
```