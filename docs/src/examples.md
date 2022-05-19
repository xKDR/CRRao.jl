## Examples: Setting up the code

```jldoctest examples
julia> using RDatasets, NLSolversBase, CRRao, Logging, StableRNGs;

julia> Logging.disable_logging(Logging.Warn); CRRao.setprogress!(false);

julia> CRRao.set_rng(StableRNG(1234))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000009a5)
```

## Example 1: Linear Regression

```jldoctest examples
julia> df = dataset("datasets", "mtcars")
32×12 DataFrame
 Row │ Model              MPG      Cyl    Disp     HP     DRat     WT       QS ⋯
     │ String31           Float64  Int64  Float64  Int64  Float64  Float64  Fl ⋯
─────┼──────────────────────────────────────────────────────────────────────────
   1 │ Mazda RX4             21.0      6    160.0    110     3.9     2.62      ⋯
   2 │ Mazda RX4 Wag         21.0      6    160.0    110     3.9     2.875
   3 │ Datsun 710            22.8      4    108.0     93     3.85    2.32
   4 │ Hornet 4 Drive        21.4      6    258.0    110     3.08    3.215
   5 │ Hornet Sportabout     18.7      8    360.0    175     3.15    3.44      ⋯
   6 │ Valiant               18.1      6    225.0    105     2.76    3.46
   7 │ Duster 360            14.3      8    360.0    245     3.21    3.57
   8 │ Merc 240D             24.4      4    146.7     62     3.69    3.19
  ⋮  │         ⋮             ⋮       ⋮       ⋮       ⋮       ⋮        ⋮        ⋱
  26 │ Fiat X1-9             27.3      4     79.0     66     4.08    1.935     ⋯
  27 │ Porsche 914-2         26.0      4    120.3     91     4.43    2.14
  28 │ Lotus Europa          30.4      4     95.1    113     3.77    1.513
  29 │ Ford Pantera L        15.8      8    351.0    264     4.22    3.17
  30 │ Ferrari Dino          19.7      6    145.0    175     3.62    2.77      ⋯
  31 │ Maserati Bora         15.0      8    301.0    335     3.54    3.57
  32 │ Volvo 142E            21.4      4    121.0    109     4.11    2.78
                                                   5 columns and 17 rows omitted

julia> m1_1 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression());

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
2.5741691724978972

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

julia> m1_1.fittedResponse
32-element Vector{Float64}:
 23.668849952338718
 22.85340824320634
 25.253556140740894
 20.746171762311384
 17.635570543830177
 20.14663845388644
 14.644831040166633
 23.61182872351372
 22.525801204993822
 20.568426475004856
  ⋮
 13.781422171673526
 16.340457241090512
 27.47793682112109
 26.922715039574857
 28.11844900519874
 17.264981908248554
 21.818065399379595
 13.374047477198516
 23.193986311384343

julia> m1_1.residuals
32-element Vector{Float64}:
 -2.668849952338718
 -1.8534082432063386
 -2.4535561407408935
  0.6538282376886144
  1.0644294561698224
 -2.0466384538864375
 -0.3448310401666319
  0.7881712764862776
  0.2741987950061784
 -1.3684264750048563
  ⋮
 -0.4814221716735254
  2.8595427589094875
 -0.1779368211210901
 -0.9227150395748573
  2.2815509948012576
 -1.4649819082485536
 -2.1180653993795957
  1.6259525228014837
 -1.7939863113843444

julia> m1_1.Cooks_distance
32-element Vector{Float64}:
 0.013342034282302798
 0.006887282667312197
 0.015495847517059161
 0.0014309089637597765
 0.004471979213923591
 0.014588985833725164
 0.001540100419881934
 0.005826402580871439
 0.00030743156824582164
 0.00701180372448546
 ⋮
 0.002076825609693457
 0.022039704192128577
 0.0001378106083285506
 0.006862929526074502
 0.04703889945177857
 0.038120451318087265
 0.035404694590360615
 0.1371534135504359
 0.006145660329519691

```

 **Linear Regression - Ridge Prior**

```jldoctest examples
julia> m1_2 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Ridge());

julia> m1_2.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           v    6.7693    3.9279     0.0393    0.0610   3574.4104    1.0000    ⋯
           σ    2.6884    0.3984     0.0040    0.0062   3466.2021    1.0002    ⋯
           α   28.4603    5.4967     0.0550    0.1052   2208.7354    1.0008    ⋯
        β[1]   -0.0399    0.0108     0.0001    0.0002   3733.4487    1.0005    ⋯
        β[2]   -2.6629    0.9680     0.0097    0.0180   2467.8793    1.0011    ⋯
        β[3]    1.6338    0.9939     0.0099    0.0183   2342.6812    1.0006    ⋯
                                                                1 column omitted

julia> m1_2.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    2.3216    4.4133    5.9258    8.0968   16.1426
           σ    2.0474    2.4028    2.6364    2.9225    3.5785
           α   16.7786   25.1108   28.6936   32.1160   38.5610
        β[1]   -0.0616   -0.0470   -0.0398   -0.0328   -0.0183
        β[2]   -4.5039   -3.3021   -2.6899   -2.0502   -0.6440
        β[3]   -0.2071    0.9672    1.5988    2.2439    3.7647

```

**Linear Regression - Laplace Prior**

```jldoctest examples
julia> m1_3 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Laplace());

julia> m1_3.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           v    4.3182    3.3442     0.0334    0.0490   3968.7204    0.9999    ⋯
           σ    2.6657    0.3792     0.0038    0.0056   4690.6502    1.0000    ⋯
           α   29.0672    5.1669     0.0517    0.0918   3367.7350    1.0001    ⋯
        β[1]   -0.0399    0.0105     0.0001    0.0002   4116.7824    1.0005    ⋯
        β[2]   -2.7069    0.9341     0.0093    0.0170   3286.4012    1.0002    ⋯
        β[3]    1.5082    0.9373     0.0094    0.0163   3601.6346    1.0002    ⋯
                                                                1 column omitted

julia> m1_3.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    1.2224    2.3903    3.4576    5.1836   12.4068
           σ    2.0359    2.3975    2.6246    2.8949    3.5178
           α   18.0169   25.8440   29.3179   32.5968   38.4400
        β[1]   -0.0609   -0.0466   -0.0396   -0.0330   -0.0199
        β[2]   -4.4455   -3.3403   -2.7409   -2.1152   -0.7174
        β[3]   -0.2418    0.8759    1.4648    2.1019    3.4821

```

 **Linear Regression - Cauchy Prior**
```jldoctest examples
julia> m1_4 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Cauchy(),20000);

julia> m1_4.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           σ    2.5855    0.3416     0.0024    0.0036   9218.6691    1.0001    ⋯
           α   30.3875    4.6394     0.0328    0.0678   4559.8857    1.0001    ⋯
        β[1]   -0.0394    0.0099     0.0001    0.0001   7652.1219    1.0000    ⋯
        β[2]   -2.8435    0.8542     0.0060    0.0116   4998.6993    1.0001    ⋯
        β[3]    1.2513    0.8428     0.0060    0.0120   5011.2306    1.0000    ⋯
                                                                1 column omitted

julia> m1_4.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           σ    2.0120    2.3452    2.5484    2.7877    3.3675
           α   20.9628   27.4262   30.4704   33.5255   39.1492
        β[1]   -0.0591   -0.0459   -0.0393   -0.0328   -0.0201
        β[2]   -4.4925   -3.4133   -2.8494   -2.2897   -1.1335
        β[3]   -0.3345    0.6759    1.2385    1.7936    2.9585

```

 **Linear Regression - T-Distributed Prior**

```jldoctest examples
julia> m1_5 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_TDist());

julia> m1_5.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           ν    1.0539    0.5597     0.0056    0.0070   5800.1893    1.0003    ⋯
           σ    2.6265    0.3686     0.0037    0.0047   6165.5244    0.9999    ⋯
           α   30.2167    4.8679     0.0487    0.1012   2225.6405    0.9999    ⋯
        β[1]   -0.0393    0.0103     0.0001    0.0002   3319.3510    1.0005    ⋯
        β[2]   -2.8300    0.8976     0.0090    0.0187   2396.9552    1.0001    ⋯
        β[3]    1.2837    0.8841     0.0088    0.0179   2334.0136    0.9999    ⋯
                                                                1 column omitted

julia> m1_5.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           ν    0.3731    0.6686    0.9233    1.2896    2.4911
           σ    2.0385    2.3621    2.5841    2.8463    3.4736
           α   20.4434   27.0806   30.3379   33.4157   39.6471
        β[1]   -0.0597   -0.0461   -0.0393   -0.0324   -0.0192
        β[2]   -4.5979   -3.4317   -2.8360   -2.2500   -1.0505
        β[3]   -0.4012    0.6970    1.2552    1.8472    3.0717

```

 **Linear Regression - Uniform Prior**
```jldoctest examples
julia> m1_6 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_TDist());

julia> m1_6.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           ν    1.0665    0.5900     0.0059    0.0081   5791.3987    1.0007    ⋯
           σ    2.6276    0.3678     0.0037    0.0056   3788.8270    0.9999    ⋯
           α   30.3304    4.7387     0.0474    0.0881   2697.4202    0.9999    ⋯
        β[1]   -0.0394    0.0102     0.0001    0.0002   3969.3250    0.9999    ⋯
        β[2]   -2.8421    0.8679     0.0087    0.0159   2813.1886    0.9999    ⋯
        β[3]    1.2646    0.8642     0.0086    0.0157   2858.6901    0.9999    ⋯
                                                                1 column omitted

julia> m1_6.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           ν    0.3749    0.6698    0.9298    1.3095    2.5508
           σ    2.0306    2.3712    2.5893    2.8418    3.4723
           α   20.4677   27.3179   30.5001   33.5657   39.1292
        β[1]   -0.0596   -0.0460   -0.0392   -0.0324   -0.0198
        β[2]   -4.5194   -3.4288   -2.8534   -2.2704   -1.1239
        β[3]   -0.3362    0.6886    1.2226    1.8201    3.0601

```