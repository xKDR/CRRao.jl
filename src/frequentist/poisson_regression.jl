function poisson_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link; kwargs...)
    model = glm(formula, data, Poisson(), Link; kwargs...)
    return FrequentistRegression(:PoissonRegression, model, formula, typeof(Link))
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression; kwargs...)
```

Fit a Poisson Regression model on the input data (with the default link function being the Log link). Uses the [glm](https://juliastats.org/GLM.jl/stable/api/#GLM.glm) method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:PoissonRegression}`. Supports the same keyword arguments as glm.

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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression())
Model Class: Poisson Regression
Likelihood Mode: Poison
Link Function: Log
Computing Method: Optimization
─────────────────────────────────────────────────────────────────────────────────
                        Coef.  Std. Error      z  Pr(>|z|)   Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────────────
(Intercept)         -1.91392    0.261667   -7.31    <1e-12  -2.42678    -1.40106
Target               0.157769   0.0653822   2.41    0.0158   0.0296218   0.285915
Coop                 1.15127    0.0561861  20.49    <1e-92   1.04114     1.26139
NCost: major loss   -0.324051   0.230055   -1.41    0.1590  -0.774951    0.126848
NCost: modest loss   1.71973    0.100518   17.11    <1e-64   1.52272     1.91674
NCost: net gain      0.463907   0.16992     2.73    0.0063   0.13087     0.796944
─────────────────────────────────────────────────────────────────────────────────
```
"""
function fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression; kwargs...)
    return poisson_reg(formula, data, LogLink(); kwargs...)
end
