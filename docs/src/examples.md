```meta
using CRRao,  RDatasets, StatsModels, StatsPlots, NLSolversBase; 
```

### Example 1: Linear Regression
```jldoctest
julia> df = dataset("datasets", "mtcars");
julia> first(df, 10)
10×12 DataFrame
 Row │ Model              MPG      Cyl    Disp     HP     DRat     WT       QSec     VS     AM     Gear   Carb  
     │ String31           Float64  Int64  Float64  Int64  Float64  Float64  Float64  Int64  Int64  Int64  Int64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Mazda RX4             21.0      6    160.0    110     3.9     2.62     16.46      0      1      4      4
   2 │ Mazda RX4 Wag         21.0      6    160.0    110     3.9     2.875    17.02      0      1      4      4
   3 │ Datsun 710            22.8      4    108.0     93     3.85    2.32     18.61      1      1      4      1
   4 │ Hornet 4 Drive        21.4      6    258.0    110     3.08    3.215    19.44      1      0      3      1
   5 │ Hornet Sportabout     18.7      8    360.0    175     3.15    3.44     17.02      0      0      3      2
   6 │ Valiant               18.1      6    225.0    105     2.76    3.46     20.22      1      0      3      1
   7 │ Duster 360            14.3      8    360.0    245     3.21    3.57     15.84      0      0      3      4
   8 │ Merc 240D             24.4      4    146.7     62     3.69    3.19     20.0       1      0      4      2
   9 │ Merc 230              22.8      4    140.8     95     3.92    3.15     22.9       1      0      4      2
  10 │ Merc 280              19.2      6    167.6    123     3.92    3.44     18.3       1      0      4      4
julia> m1_1 = fitmodel(@formula(MPG ~ HP + WT+Gear), df, LinearRegression());

julia> m1_1.fit  
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  32.0137     4.63226      6.91    <1e-06  22.5249     41.5024
HP           -0.0367861  0.00989146  -3.72    0.0009  -0.0570478  -0.0165243
WT           -3.19781    0.846546    -3.78    0.0008  -4.93188    -1.46374
Gear          1.01998    0.851408     1.20    0.2410  -0.72405     2.76401
────────────────────────────────────────────────────────────────────────────
julia> m1_1.sigma
2.5741691724978977

julia> m1_1.LogLike
-73.52638935960971

julia> m1_1.AIC
157.05277871921942
julia> m1_1.BIC
164.38145823321804
julia> m1_1.R_sqr
0.8352309600685555
julia> m1_1.Adjusted_R_sqr
0.8175771343616149
julia> m1_1.fittedResponse;
julia> first(m1_1.fittedResponse, 5)
10-element Vector{Float64}:
 23.668849952338704
 22.853408243206346
 25.253556140740866
 20.746171762311327
 17.63557054383011
julia> m1_1.residuals;
julia> first(m1_1.residuals, 5)
10-element Vector{Float64}:
 -2.6688499523387037
 -1.8534082432063457
 -2.453556140740865
  0.6538282376886713
  1.06442945616989
julia> m1_1.Cooks_distance
julia> first(m1_1.Cooks_distance, 5)
5-element Vector{Float64}:
 0.013342034282302684
 0.00688728266731234
 0.015495847517058797
 0.0014309089637600369
 0.004471979213924145
```