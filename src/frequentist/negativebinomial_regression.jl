function negativebinomial_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link; kwargs...)
    model = negbin(formula, data, Link; kwargs...)
    return FrequentistRegression(:NegativeBinomialRegression, model, formula, typeof(Link))
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression; kwargs...)
```

Fit a Negative Binomial Regression model on the input data (with the default link function being the Log link). Uses the [negbin](https://juliastats.org/GLM.jl/stable/api/#GLM.negbin) method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:NegativeBinomialRegression}`. Supports the same keyword arguments as negbin.

# Example
```julia-repl
julia> using CRRao, RDatasets, StatsModels
julia> sanction = dataset("Zelig", "sanction")
78×8 DataFrame
 Row │ Mil    Coop   Target  Import  Export  Cost   Num    NCost         
     │ Int32  Int32  Int32   Int32   Int32   Int32  Int32  Cat…          
─────┼───────────────────────────────────────────────────────────────────
   1 │     1      4       3       1       1      4     15  major loss
   2 │     0      2       3       0       1      3      4  modest loss
   3 │     0      1       3       1       0      2      1  little effect
   4 │     1      1       3       1       1      2      1  little effect
   5 │     0      1       3       1       1      2      1  little effect
   6 │     0      1       3       0       1      2      1  little effect
  ⋮  │   ⋮      ⋮      ⋮       ⋮       ⋮       ⋮      ⋮          ⋮
  73 │     1      3       1       1       1      2     14  little effect
  74 │     0      2       1       0       0      1      2  net gain
  75 │     0      1       3       0       1      2      1  little effect
  76 │     0      4       3       1       0      2     13  little effect
  77 │     0      1       2       0       0      1      1  net gain
  78 │     1      3       1       1       1      2     10  little effect
                                                          66 rows omitted
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression())
Model Class: Count Regression
Likelihood Mode: Negative Binomial
Link Function: Log
Computing Method: Optimization
──────────────────────────────────────────────────────────────────────────────────
                          Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────────────
(Intercept)         -1.14517       0.480887  -2.38    0.0172  -2.0877    -0.202652
Target               0.00862527    0.145257   0.06    0.9527  -0.276074   0.293324
Coop                 1.06397       0.115995   9.17    <1e-19   0.836621   1.29131
NCost: major loss   -0.23511       0.511443  -0.46    0.6457  -1.23752    0.7673
NCost: modest loss   1.30767       0.276012   4.74    <1e-05   0.766698   1.84865
NCost: net gain      0.183453      0.275387   0.67    0.5053  -0.356296   0.723202
──────────────────────────────────────────────────────────────────────────────────
```
"""
function fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression; kwargs...)
    return negativebinomial_reg(formula, data, LogLink(); kwargs...)
end
