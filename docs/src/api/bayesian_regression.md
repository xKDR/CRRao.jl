# Bayesian Regression Models

```@docs
BayesianRegression
```

## Bayesian Algorithms

```@docs
BayesianAlgorithm
MCMC
VI
```

## Linear Regression

### Linear Regression with User Specific Gaussian Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Gauss, alpha_prior_mean::Float64, beta_prior_mean::Vector{Float64}, algorithm::BayesianAlgorithm = MCMC())
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Gauss, alpha_prior_mean::Float64, alpha_prior_sd::Float64, beta_prior_mean::Vector{Float64}, beta_prior_sd::Vector{Float64}, algorithm::BayesianAlgorithm = MCMC())
```

### Linear Regression with Ridge Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Ridge, algorithm::BayesianAlgorithm = MCMC(), h::Float64 = 0.01)
```

### Linear Regression with Laplace Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Laplace, algorithm::BayesianAlgorithm = MCMC(), h::Float64 = 0.01)
```
### Linear Regression with Cauchy Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Cauchy, algorithm::BayesianAlgorithm = MCMC())
```
### Linear Regression with T-distributed Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_TDist, algorithm::BayesianAlgorithm = MCMC(), h::Float64 = 2.0)
```
### Linear Regression with Horse Shoe Prior
```@docs
fit(formula::FormulaTerm,data::DataFrame,modelClass::LinearRegression,prior::Prior_HorseShoe,algorithm::BayesianAlgorithm = MCMC())
```

## Logistic Regression

### Logistic Regression with Ridge Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Ridge, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```
### Logistic Regression with Laplace Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Laplace, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```
### Logistic Regression with Cauchy Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Cauchy, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```
### Logistic Regression with T-Distributed Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_TDist, h::Float64 = 1.0, level::Float64 = 0.95, sim_size::Int64 = 1000)
```

### Logistic Regression with Horse Shoe Prior
```@docs
fit(formula::FormulaTerm,data::DataFrame,modelClass::LogisticRegression,Link::CRRaoLink,prior::Prior_HorseShoe,level::Float64 = 0.95,sim_size::Int64 = 1000)
```

## Negative Binomial Regression

### Negative Binomial Regression with Ridge Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Ridge, h::Float64 = 0.1, sim_size::Int64 = 1000)
```

### Negative Binomial Regression with Laplace Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Laplace, h::Float64 = 0.01, sim_size::Int64 = 1000)
```
### Negative Binomial Regression with Cauchy Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Cauchy, h::Float64 = 1.0, sim_size::Int64 = 1000)
```

### Negative Binomial Regression with T-Distributed Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_TDist, h::Float64 = 1.0, sim_size::Int64 = 1000)
```

### Negative Binomial Regression with HorseShoe Prior
```@docs
fit(formula::FormulaTerm,data::DataFrame,modelClass::NegBinomRegression,prior::Prior_HorseShoe, h::Float64 = 1.0, sim_size::Int64 = 1000)
```

## Poisson Regression

### Poisson Regression with Ridge Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_Ridge, h::Float64 = 0.1, sim_size::Int64 = 1000)
```
### Poisson Regression with Laplace Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_Laplace, h::Float64 = 0.1, sim_size::Int64 = 1000)
```
### Poisson Regression with Cauchy Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_Cauchy, h::Float64 = 1.0, sim_size::Int64 = 1000)
```
### Poisson Regression with T-Distributed Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_TDist, h::Float64 = 2.0, sim_size::Int64 = 1000)
```

### Poisson Regression with Horse Shoe Prior
```@docs
fit(formula::FormulaTerm,data::DataFrame,modelClass::PoissonRegression,prior::Prior_HorseShoe,sim_size::Int64 = 1000)
```
