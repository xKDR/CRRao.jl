# Frequentist Regression Models

```@docs
FrequentistRegression
```

## Linear Regression 
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression)
```

## Logistic Regression
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit)
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit)
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog)
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit)
```

## Negative Binomial Regression
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression)
```

## Poisson Regression
```@docs
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression)
```

## Extended functions from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl)

```@docs
coeftable(container::FrequentistRegression)
r2(container::FrequentistRegression)
adjr2(container::FrequentistRegression)
loglikelihood(container::FrequentistRegression)
aic(container::FrequentistRegression)
bic(container::FrequentistRegression)
predict(container::FrequentistRegression)
residuals(container::FrequentistRegression)
cooksdistance(container::FrequentistRegression)
```