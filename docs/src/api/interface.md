# General Interface

## Understanding the interface

CRRao exports the [`@fitmodel`](@ref) macro, which is used to train all types of models supported by the package. As of now, the macro supports the following signatures.

```julia
@fitmodel(formula, data, modelClass)
@fitmodel(formula, data, modelClass, link)
@fitmodel(formula, data, modelClass, prior)
@fitmodel(formula, data, modelClass, link, prior)
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

5. CRRao also supports Bayesian models, and the priors to be can be specified while calling `@fitmodel`. Currently CRRao supports five different kinds of priors, and the type of the `prior` parameter must be one of the following.
    - [`Prior_Ridge`](@ref)
    - [`Prior_Laplace`](@ref)
    - [`Prior_Cauchy`](@ref)
    - [`Prior_TDist`](@ref)
    - [`Prior_Uniform`](@ref)

```@docs
@fitmodel
```

## Model Classes

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

## Setting Random Number Generators

```@docs
CRRao.set_rng
```
