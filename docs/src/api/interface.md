# General Interface

## Understanding the interface

CRRao exports the [`fit`](@ref) function, which is used to train all types of models supported by the package. As of now, the function supports the following signatures.

```julia
fit(formula, data, modelClass)
fit(formula, data, modelClass, link)
fit(formula, data, modelClass, prior)
fit(formula, data, modelClass, link, prior)
```

It should be noted that not all model classes support every type of signature. The parameters passed above mean the following.

1. The parameter `formula` must be a formula of type [`StatsModels.FormulaTerm`](https://juliastats.org/StatsModels.jl/stable/api/#StatsModels.FormulaTerm). Any formula has an LHS and an RHS. The LHS represents the response variable, and the RHS represents the independent variables.

2. The parameter `data` must be a [`DataFrame`](https://dataframes.juliadata.org/stable/lib/types/#DataFrames.DataFrame). This variable represents the dataset on which the model must be trained.

3. `modelClass` represents the type of the statistical model to be used. Currently, CRRao supports four regression models, and the type of `modelClass` must be one of the following:
    - [`LinearRegression`](@ref)
    - [`LogisticRegression`](@ref)
    - [`NegBinomRegression`](@ref)
    - [`PoissonRegression`](@ref)

4. Certain model classes (like Logistic Regression) support link functions; this is represented by the `link` parameter. Currently four link functions are supported: Logit, Probit, Cloglog and Cauchit. So, the type of `link` must be one of the following:
    - [`Logit`](@ref)
    - [`Probit`](@ref)
    - [`Cloglog`](@ref)
    - [`Cauchit`](@ref)

5. CRRao also supports Bayesian models, and the priors to be can be specified while calling `fit`. Currently CRRao supports six different kinds of priors, and the type of the `prior` parameter must be one of the following.
    - [`Gauss`](@ref)
    - [`Ridge`](@ref)
    - [`Laplace`](@ref)
    - [`Cauchy`](@ref)
    - [`TDist`](@ref)
    - [`HorseShoe`](@ref)


## Model Classes and Data Models

```@docs
LinearRegression
LogisticRegression
NegBinomRegression
PoissonRegression
```

## Link functions.

```@docs
CRRaoLink
Logit
Probit
Cloglog
Cauchit
```
## Prior Distributions

```@docs
Gauss
Ridge
Laplace
Cauchy
TDist
HorseShoe
```

## Setting Random Number Generators

```@docs
CRRao.set_rng
```
