# Frequentist Regression Models

```@docs
FrequentistRegression
```

## Linear Regression 
```@docs
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression)
```

## Logistic Regression
```@docs
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog)
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit)
```

## Negative Binomial Regression
```@docs
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression)
```

## Poisson Regression
```@docs
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression)
```

## Extended function from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl)

```@docs
coeftable(container::FrequentistRegression)
r2(container::FrequentistRegression)
adjr2(container::FrequentistRegression)
loglikelihood(container::FrequentistRegression)
aic(container::FrequentistRegression)
bic(container::FrequentistRegression)
sigma(container::FrequentistRegression)
predict(container::FrequentistRegression)
residuals(container::FrequentistRegression)
cooksdistance(container::FrequentistRegression)
```