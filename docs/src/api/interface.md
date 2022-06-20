# Understanding the interface

CRRao exports the `@fitmodel` macro, which is used to train all types of models supported by the package. As of now, the macro supports the following signatures.

```julia
@fitmodel(formula, data, modelClass)
@fitmodel(formula, data, modelClass, link)
@fitmodel(formula, data, modelClass, prior)
@fitmodel(formula, data, modelClass, link, prior)
```

It should be noted that not all model classes support every type of signature. The parameters passed above mean the following.

1. The parameter `formula` must be a formula of type `StatsModels.FormulaTerm`. Any formula has an LHS and an RHS. The LHS represents the response variable, and the RHS represents the independent variables.

2. The parameter `data` must be a `DataFrame`. This variable represents the dataset on which the model must be trained.

3. `modelClass` represents the type of the statistical model to be used. Currently, CRRao supports four regression models, and `modelClass` must be one of the following objects:
    - `LinearRegression()`
    - `LogisticRegression()`
    - `NegBinomialRegression()`
    - `PoissonRegression()`

4. Certain model classes (like Logistic Regression) support link functions; this is represented by the `link` parameter. Currently four link functions are supported: Logit, Probit, Cloglog and Cauchit. So, `link` must be equal to one of the following:
    - `Logit()`
    - `Probit()`
    - `Cloglog()`
    - `Cauchit`

5. CRRao also supports Bayesian models, and the priors to be can be specified while calling `@fitmodel`. Currently CRRao supports five different kinds of priors, and the `prior` parameter must be equal to one of the following.
    - `Prior_Ridge()`
    - `Prior_Laplace()`
    - `Prior_Cauchy()`
    - `Prior_TDist()`
    - `Prior_Uniform()`
