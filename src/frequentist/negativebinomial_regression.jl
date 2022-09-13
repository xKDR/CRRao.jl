function negativebinomial_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link)
    formula = apply_schema(formula, schema(formula, data))
    model = glm(formula, data, NegativeBinomial(), Link)
    return FrequentistRegression(:NegativeBinomialRegression, model, formula, typeof(Link))
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression)
```

Fit a Negative Binomial Regression model on the input data (with the default link function being the Log link). Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:NegativeBinomialRegression}`.

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
─────────────────────────────────────────────────────────────────────────────────
                         Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────────────
(Intercept)         -1.10939      0.459677  -2.41    0.0158  -2.01034   -0.208444
Target               0.0117398    0.142779   0.08    0.9345  -0.268101   0.291581
Coop                 1.0506       0.111556   9.42    <1e-20   0.831949   1.26924
NCost: major loss   -0.204244     0.508156  -0.40    0.6877  -1.20021    0.791723
NCost: modest loss   1.27142      0.290427   4.38    <1e-04   0.702197   1.84065
NCost: net gain      0.176797     0.254291   0.70    0.4869  -0.321604   0.675197
─────────────────────────────────────────────────────────────────────────────────
```
"""
function fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression)
    return negativebinomial_reg(formula, data, LogLink())
end
