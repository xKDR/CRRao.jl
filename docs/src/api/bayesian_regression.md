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
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Ridge, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Laplace, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Cauchy, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_TDist, h::Float64 = 1.0, level::Float64 = 0.95, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Uniform, h::Float64 = 0.01, level::Float64 = 0.95, sim_size::Int64 = 10000)
```

## Negative Binomial Regression
```@docs
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Ridge, h::Float64 = 0.1, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Laplace, h::Float64 = 0.01, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Cauchy, h::Float64 = 1.0, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_TDist, h::Float64 = 1.0, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Uniform, h::Float64 = 0.1, sim_size::Int64 = 10000)
```

## Poisson Regression
```@docs
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_Ridge, h::Float64 = 0.1, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_Laplace, h::Float64 = 0.1, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_Cauchy, h::Float64 = 1.0, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_TDist, h::Float64 = 2.0, sim_size::Int64 = 10000)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_Uniform, h::Float64 = 1.0, sim_size::Int64 = 10000)
```