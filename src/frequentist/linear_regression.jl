"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression)
```

Fit an OLS Linear Regression model on the input data. Uses the `lm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Linear Regression in our case.

```julia-repl
julia> using CRRao, RDatasets, StatsModels, StatsPlots, StatsBase

julia> df = dataset("datasets", "mtcars");

julia> container = @fitmodel(MPG ~ HP + WT + Gear, df, LinearRegression());

julia> coeftable(container.model)
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  32.0137     4.63226      6.91    <1e-06  22.5249     41.5024
HP           -0.0367861  0.00989146  -3.72    0.0009  -0.0570478  -0.0165243
WT           -3.19781    0.846546    -3.78    0.0008  -4.93188    -1.46374
Gear          1.01998    0.851408     1.20    0.2410  -0.72405     2.76401
────────────────────────────────────────────────────────────────────────────

julia> loglikelihood(container.model)
-73.52638935960971

julia> aic(container.model)
157.05277871921942

julia> bic(container.model)
164.38145823321804

julia> r2(container.model)
0.8352309600685555

julia> adjr2(container.model)
0.8175771343616149

julia> predict(container.model)
32-element Vector{Float64}:
 23.668849952338718
 22.85340824320634
 25.253556140740894
 20.746171762311384
 17.635570543830177
 20.14663845388644
 14.644831040166633
  ⋮
 26.922715039574857
 28.11844900519874
 17.264981908248554
 21.818065399379595
 13.374047477198516
 23.193986311384343

julia> residuals(container.model)
-2.668849952338718
-1.8534082432063386
-2.4535561407408935
 0.6538282376886144
 1.0644294561698224
-2.0466384538864375
-0.3448310401666319
 ⋮
-0.9227150395748573
 2.2815509948012576
-1.4649819082485536
-2.1180653993795957
 1.6259525228014837
-1.7939863113843444

julia> cooksdistance(container.model)
32-element Vector{Float64}:
 0.013342034282302798
 0.006887282667312197
 0.015495847517059161
 0.0014309089637597765
 0.004471979213923591
 0.014588985833725164
 0.001540100419881934
 ⋮
 0.006862929526074502
 0.04703889945177857
 0.038120451318087265
 0.035404694590360615
 0.1371534135504359
 0.006145660329519691

julia> plot(cooksdistance(container.model))
```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression)
   formula = apply_schema(formula, schema(formula, data))
   model = lm(formula,data)
   return FrequentistRegression{:LinearRegression}(model)
end