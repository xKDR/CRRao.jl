
"""
```julia

fit(formula, data::DataFrame, modelClass::GaussianProcessesRegression, IndexVar, mean, kern::Kernel, 
   DistanceClass::Euclidean)
```

Fit a Gaussian Process Regression model on the input data with a Gaussian Process prior and user-specific mean and kernel functions.

#Example

```julia-repl

julia> using CRRao, RDatasets, StatsModels, StatsPlots, GaussianProcesses, Distances
julia> df = dataset("datasets", "mtcars")
32×12 DataFrame
 Row │ Model              MPG      Cyl    Disp     HP     DRat     WT       QSec     VS     AM     Gear   Carb  
     │ String31           Float64  Int64  Float64  Int64  Float64  Float64  Float64  Int64  Int64  Int64  Int64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Mazda RX4             21.0      6    160.0    110     3.9     2.62     16.46      0      1      4      4
   2 │ Mazda RX4 Wag         21.0      6    160.0    110     3.9     2.875    17.02      0      1      4      4
  ⋮  │         ⋮             ⋮       ⋮       ⋮       ⋮       ⋮        ⋮        ⋮       ⋮      ⋮      ⋮      ⋮
  31 │ Maserati Bora         15.0      8    301.0    335     3.54    3.57     14.6       0      1      5      8
  32 │ Volvo 142E            21.4      4    121.0    109     4.11    2.78     18.6       1      1      4      2
                                                                                             28 rows omitted

julia> container=fit(@formula(MPG ~0+ HP),df,GaussianProcessesRegression(),[:MPG, :HP],MeanZero(), SE(0.0,0.0),Euclidean())

Formula: MPG ~ 0 + HP
Link: CRRao.Identity(CRRao.Identity_Link)
Chain: GP Exact object:
  Dim = 1
  Number of observations = 32
  Mean function:
    Type: MeanZero, Params: Float64[]
  Kernel:
    Type: SEIso{Float64}, Params: [5.464908573213355, 3.3936838718120708]
  Input observations = 
[110.0 110.0 … 335.0 109.0]
  Output observations = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2  …  15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4]
  Variance of observation noise = 9.667961411202336
  Marginal Log-Likelihood = -89.745
julia> plot(container.chain)
```
"""
function fit(formula,
     data::DataFrame, 
     modelClass::GaussianProcessesRegression,
     IndexVar, 
     mean, 
     kern::Kernel, 
     DistanceClass::Euclidean)

 formula = apply_schema(formula, schema(formula, data), RegressionModel)
 select!(data, IndexVar)
 y, X = modelcols(formula, data)
 logObsNoise = -1.0
 gp = GP(X', y, mean, kern, logObsNoise)
 optimize!(gp)
 return BayesianRegression(:GaussianProcessesRegression, gp, formula)

end

"""

```julia

fit(formula, data::DataFrame, modelClass::GaussianProcessesRegression, IndexVar, mean 
   DistanceClass::Euclidean)
```

Fit a Gaussian Process Regression model on the input data with a Gaussian Process prior. The Zero Mean and Squared Exponential kernel function is implemented by default.

#Example

```julia-repl
julia> using CRRao, RDatasets, StatsModels, StatsPlots, GaussianProcesses, Distances

julia> df = dataset("datasets", "mtcars")
32×12 DataFrame
 Row │ Model              MPG      Cyl    Disp     HP     DRat     WT       QSec     VS     AM     Gear   Carb  
     │ String31           Float64  Int64  Float64  Int64  Float64  Float64  Float64  Int64  Int64  Int64  Int64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Mazda RX4             21.0      6    160.0    110     3.9     2.62     16.46      0      1      4      4
   2 │ Mazda RX4 Wag         21.0      6    160.0    110     3.9     2.875    17.02      0      1      4      4
  ⋮  │         ⋮             ⋮       ⋮       ⋮       ⋮       ⋮        ⋮        ⋮       ⋮      ⋮      ⋮      ⋮
  31 │ Maserati Bora         15.0      8    301.0    335     3.54    3.57     14.6       0      1      5      8
  32 │ Volvo 142E            21.4      4    121.0    109     4.11    2.78     18.6       1      1      4      2
                                                                                                 28 rows omitted

julia> container=fit(@formula(MPG ~0+ HP),df,GaussianProcessesRegression(),[:MPG, :HP],Euclidean())

Formula: MPG ~ 0 + HP
Link: CRRao.Identity(CRRao.Identity_Link)
Chain: GP Exact object:
  Dim = 1
  Number of observations = 32
  Mean function:
    Type: MeanZero, Params: Float64[]
  Kernel:
    Type: SEIso{Float64}, Params: [5.464908573213355, 3.3936838718120708]
  Input observations = 
[110.0 110.0 … 335.0 109.0]
  Output observations = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2  …  15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4]
  Variance of observation noise = 9.667961411202336
  Marginal Log-Likelihood = -89.745



julia> plot(container.chain)
```
"""


function fit(formula, 
    data::DataFrame, 
    modelClass::GaussianProcessesRegression,
    IndexVar,
    DistanceClass::Euclidean)

 formula = apply_schema(formula, schema(formula, data), RegressionModel)
 select!(data, IndexVar)
 y, X = modelcols(formula, data)
 logObsNoise = -1.0
 mean= MeanZero()
 kern=SE(0.0,0.0)
 gp = GP(X', y, mean, kern, logObsNoise)
 optimize!(gp)
 return BayesianRegression(:GaussianProcessesRegression, gp, formula)

end
