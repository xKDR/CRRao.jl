"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression)
```

Fit an OLS Linear Regression model on the input data. Uses the `lm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood. Returns an object of type `FrequentistRegression{:LinearRegression}`.

# Example
```julia-repl
julia> using CRRao, RDatasets, StatsPlots
julia> df = dataset("datasets", "mtcars")
32×12 DataFrame
 Row │ Model              MPG      Cyl    Disp     HP     DRat     WT       QSec     VS     AM     Gear   Carb  
     │ String31           Float64  Int64  Float64  Int64  Float64  Float64  Float64  Int64  Int64  Int64  Int64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Mazda RX4             21.0      6    160.0    110     3.9     2.62     16.46      0      1      4      4
   2 │ Mazda RX4 Wag         21.0      6    160.0    110     3.9     2.875    17.02      0      1      4      4
   3 │ Datsun 710            22.8      4    108.0     93     3.85    2.32     18.61      1      1      4      1
   4 │ Hornet 4 Drive        21.4      6    258.0    110     3.08    3.215    19.44      1      0      3      1
   5 │ Hornet Sportabout     18.7      8    360.0    175     3.15    3.44     17.02      0      0      3      2
   6 │ Valiant               18.1      6    225.0    105     2.76    3.46     20.22      1      0      3      1
  ⋮  │         ⋮             ⋮       ⋮       ⋮       ⋮       ⋮        ⋮        ⋮       ⋮      ⋮      ⋮      ⋮
  27 │ Porsche 914-2         26.0      4    120.3     91     4.43    2.14     16.7       0      1      5      2
  28 │ Lotus Europa          30.4      4     95.1    113     3.77    1.513    16.9       1      1      5      2
  29 │ Ford Pantera L        15.8      8    351.0    264     4.22    3.17     14.5       0      1      5      4
  30 │ Ferrari Dino          19.7      6    145.0    175     3.62    2.77     15.5       0      1      5      6
  31 │ Maserati Bora         15.0      8    301.0    335     3.54    3.57     14.6       0      1      5      8
  32 │ Volvo 142E            21.4      4    121.0    109     4.11    2.78     18.6       1      1      4      2
                                                                                                 20 rows omitted
julia> container = @fitmodel(MPG ~ HP + WT + Gear, df, LinearRegression())
Model Class: Linear Regression
Likelihood Mode: Gauss
Link Function: Identity
Computing Method: Optimization
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  32.0137     4.63226      6.91    <1e-06  22.5249     41.5024
HP           -0.0367861  0.00989146  -3.72    0.0009  -0.0570478  -0.0165243
WT           -3.19781    0.846546    -3.78    0.0008  -4.93188    -1.46374
Gear          1.01998    0.851408     1.20    0.2410  -0.72405     2.76401
────────────────────────────────────────────────────────────────────────────
julia> coeftable(container)
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  32.0137     4.63226      6.91    <1e-06  22.5249     41.5024
HP           -0.0367861  0.00989146  -3.72    0.0009  -0.0570478  -0.0165243
WT           -3.19781    0.846546    -3.78    0.0008  -4.93188    -1.46374
Gear          1.01998    0.851408     1.20    0.2410  -0.72405     2.76401
────────────────────────────────────────────────────────────────────────────
julia> sigma(container)
2.5741691724978972
julia> aic(container)
157.05277871921942
julia> predict(container)
32-element Vector{Float64}:
 23.668849952338718
 22.85340824320634
 25.253556140740894
 20.746171762311384
 17.635570543830177
 20.14663845388644
 14.644831040166633
 23.61182872351372
  ⋮
 16.340457241090512
 27.47793682112109
 26.922715039574857
 28.11844900519874
 17.264981908248554
 21.818065399379595
 13.374047477198516
 23.193986311384343
julia> residuals(container)
32-element Vector{Float64}:
 -2.668849952338718
 -1.8534082432063386
 -2.4535561407408935
  0.6538282376886144
  1.0644294561698224
 -2.0466384538864375
 -0.3448310401666319
  0.7881712764862776
  ⋮
  2.8595427589094875
 -0.1779368211210901
 -0.9227150395748573
  2.2815509948012576
 -1.4649819082485536
 -2.1180653993795957
  1.6259525228014837
 -1.7939863113843444
julia> plot(cooksdistance(container))
```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression)
    formula = apply_schema(formula, schema(formula, data))
    model = lm(formula, data)
    return FrequentistRegression(:LinearRegression, model, formula)
end
