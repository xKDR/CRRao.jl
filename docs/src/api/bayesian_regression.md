# Bayesian Regression Models

```@docs
BayesianRegression
```

## Linear Regression

### Linear Regression with User Specific Gaussian Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Gauss, alpha_prior_mean::Float64, alpha_prior_sd::Float64, beta_prior_mean::Vector{Float64}, beta_prior_sd::Vector{Float64}, sim_size::Int64 = 1000)
```

### Linear Regression with Ridge Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Ridge, h::Float64 = 0.01, sim_size::Int64 = 1000)
```

### Linear Regression with Laplace Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Laplace, h::Float64 = 0.01, sim_size::Int64 = 1000)
```
### Linear Regression with Cauchy Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Cauchy, sim_size::Int64 = 1000)
```
### Linear Regression with T-distributed Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::TDist, h::Float64 = 2.0, sim_size::Int64 = 1000)
```
### Linear Regression with Horse Shoe Prior
```@docs
fit(formula::FormulaTerm,data::DataFrame,modelClass::LinearRegression,prior::HorseShoe,sim_size::Int64 = 1000)
```

## Logistic Regression

### Logistic Regression with Ridge Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Ridge, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```
### Logistic Regression with Laplace Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Laplace, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```
### Logistic Regression with Cauchy Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Cauchy, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```
### Logistic Regression with T-Distributed Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::TDist, h::Float64 = 1.0, level::Float64 = 0.95, sim_size::Int64 = 1000)
```

### Logistic Regression with Horse Shoe Prior
```@docs
fit(formula::FormulaTerm,data::DataFrame,modelClass::LogisticRegression,Link::CRRaoLink,prior::HorseShoe,level::Float64 = 0.95,sim_size::Int64 = 1000)
```

## Negative Binomial Regression

### Negative Binomial Regression with Ridge Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Ridge, h::Float64 = 0.1, sim_size::Int64 = 1000)
```

### Negative Binomial Regression with Laplace Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Laplace, h::Float64 = 0.01, sim_size::Int64 = 1000)
```
### Negative Binomial Regression with Cauchy Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Cauchy, h::Float64 = 1.0, sim_size::Int64 = 1000)
```

### Negative Binomial Regression with T-Distributed Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::TDist, h::Float64 = 1.0, sim_size::Int64 = 1000)
```

### Negative Binomial Regression with HorseShoe Prior
```@docs
fit(formula::FormulaTerm,data::DataFrame,modelClass::NegBinomRegression,prior::HorseShoe,sim_size::Int64 = 1000)
```

## Poisson Regression

### Poisson Regression with Ridge Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Ridge, h::Float64 = 0.1, sim_size::Int64 = 1000)
```
### Poisson Regression with Laplace Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Laplace, h::Float64 = 0.1, sim_size::Int64 = 1000)
```
### Poisson Regression with Cauchy Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Cauchy, h::Float64 = 1.0, sim_size::Int64 = 1000)
```
### Poisson Regression with T-Distributed Prior
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::TDist, h::Float64 = 2.0, sim_size::Int64 = 1000)
```

### Poisson Regression with Horse Shoe Prior
```@docs
fit(formula::FormulaTerm,data::DataFrame,modelClass::PoissonRegression,prior::HorseShoe,sim_size::Int64 = 1000)
```