function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link)
    model = glm(formula, data, Binomial(), Link)
    return FrequentistRegression(:LogisticRegression, model, formula, typeof(Link))
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit)
```

Fit a Logistic Regression model on the input data using the Logit link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:LogisticRegression}`.

# Example
```julia-repl
julia> using CRRao, RDatasets, StatsModels
julia> turnout = dataset("Zelig", "turnout")
2000×5 DataFrame
  Row │ Race   Age    Educate  Income   Vote  
      │ Cat…   Int32  Float64  Float64  Int32 
──────┼───────────────────────────────────────
    1 │ white     60     14.0   3.3458      1
    2 │ white     51     10.0   1.8561      0
    3 │ white     24     12.0   0.6304      0
    4 │ white     38      8.0   3.4183      1
    5 │ white     25     12.0   2.7852      1
    6 │ white     67     12.0   2.3866      1
  ⋮   │   ⋮      ⋮       ⋮        ⋮       ⋮
 1995 │ white     22      7.0   0.2364      0
 1996 │ white     26     16.0   3.3834      0
 1997 │ white     34     12.0   2.917       1
 1998 │ white     51     16.0   7.8949      1
 1999 │ white     22     10.0   2.4811      0
 2000 │ white     59     10.0   0.5523      0
                             1988 rows omitted
julia> container = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Logit())
Model Class: Logistic Regression
Likelihood Mode: Binomial
Link Function: Identity
Computing Method: Optimization
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  -3.03426    0.325927    -9.31    <1e-19  -3.67307    -2.39546
Age           0.0283543  0.00346034   8.19    <1e-15   0.0215722   0.0351365
Race: white   0.250798   0.146457     1.71    0.0868  -0.0362521   0.537847
Income        0.177112   0.0271516    6.52    <1e-10   0.123896    0.230328
Educate       0.175634   0.0203308    8.64    <1e-17   0.135786    0.215481
────────────────────────────────────────────────────────────────────────────
julia> coeftable(container)
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  -3.03426    0.325927    -9.31    <1e-19  -3.67307    -2.39546
Age           0.0283543  0.00346034   8.19    <1e-15   0.0215722   0.0351365
Race: white   0.250798   0.146457     1.71    0.0868  -0.0362521   0.537847
Income        0.177112   0.0271516    6.52    <1e-10   0.123896    0.230328
Educate       0.175634   0.0203308    8.64    <1e-17   0.135786    0.215481
────────────────────────────────────────────────────────────────────────────
julia> loglikelihood(container)
-1011.9906318515575
julia> aic(container)
2033.981263703115
julia> bic(container)
2061.9857760008254
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::Logit
)
    return logistic_reg(formula, data, LogitLink())
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit)
```

Fit a Logistic Regression model on the input data using the Probit link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:LogisticRegression}`.
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::Probit
)
    return logistic_reg(formula, data, ProbitLink())
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog)
```

Fit a Logistic Regression model on the input data using the Cloglog link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:LogisticRegression}`.
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::Cloglog
)
    return logistic_reg(formula, data, CloglogLink())
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit)
```

Fit a Logistic Regression model on the input data using the Cauchit link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:LogisticRegression}`.
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::Cauchit
)
    return logistic_reg(formula, data, CauchitLink())
end
