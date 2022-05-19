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

## Example 2: Logistic Regression 
```jldoctest examples
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
    7 │ white     40     12.0   4.2857      0
    8 │ white     56     10.0   9.3205      1
  ⋮   │   ⋮      ⋮       ⋮        ⋮       ⋮
 1994 │ white     58     12.0   0.1936      0
 1995 │ white     22      7.0   0.2364      0
 1996 │ white     26     16.0   3.3834      0
 1997 │ white     34     12.0   2.917       1
 1998 │ white     51     16.0   7.8949      1
 1999 │ white     22     10.0   2.4811      0
 2000 │ white     59     10.0   0.5523      0
                             1985 rows omitted

julia> m2_1 = @fitmodel((Vote ~ Age + Race +Income + Educate)
                       ,turnout,LogisticRegression(),Logit());

julia> m2_1.fit
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  -3.03426    0.325927    -9.31    <1e-19  -3.67307    -2.39546
Age           0.0283543  0.00346034   8.19    <1e-15   0.0215722   0.0351365
Race: white   0.250798   0.146457     1.71    0.0868  -0.0362521   0.537847
Income        0.177112   0.0271516    6.52    <1e-10   0.123896    0.230328
Educate       0.175634   0.0203308    8.64    <1e-17   0.135786    0.215481
────────────────────────────────────────────────────────────────────────────

julia> m2_1.modelClass
"LogisticReg"

julia> m2_1.LogLike
-1011.9906318515575

julia> m2_1.AIC
2033.981263703115

julia> m2_1.BIC
2061.9857760008254

julia> m2_2 = @fitmodel((Vote ~ Age + Race +Income + Educate)
                       ,turnout,LogisticRegression(),Probit());

julia> m2_2.fit
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  -1.76141    0.188556    -9.34    <1e-20  -2.13097    -1.39185
Age           0.0164973  0.00199897   8.25    <1e-15   0.0125794   0.0204152
Race: white   0.162856   0.0876885    1.86    0.0633  -0.0090108   0.334722
Income        0.0963117  0.0149675    6.43    <1e-09   0.066976    0.125647
Educate       0.10417    0.0116713    8.93    <1e-18   0.0812949   0.127046
────────────────────────────────────────────────────────────────────────────

julia> m2_2.BIC
2062.201026236795

julia> m2_3 = @fitmodel((Vote ~ Age + Race +Income + Educate)
                       ,turnout,LogisticRegression(),Cloglog());

julia> m2_3.fit
─────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error       z  Pr(>|z|)   Lower 95%   Upper 95%
─────────────────────────────────────────────────────────────────────────────
(Intercept)  -1.94617    0.184123    -10.57    <1e-25  -2.30704    -1.58529
Age           0.0147857  0.00184088    8.03    <1e-15   0.0111776   0.0183937
Race: white   0.185139   0.087101      2.13    0.0335   0.014424    0.355854
Income        0.0768268  0.0126411     6.08    <1e-08   0.0520506   0.101603
Educate       0.0983976  0.0108857     9.04    <1e-18   0.077062    0.119733
─────────────────────────────────────────────────────────────────────────────

julia> m2_3.BIC
2064.69463374921

julia> m2_4 = @fitmodel((Vote ~ Age + Race +Income + Educate)
                       ,turnout,LogisticRegression(),Cauchit());

julia> m2_4.fit
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  -3.16889    0.384429    -8.24    <1e-15  -3.92235    -2.41542
Age           0.0304105  0.00413473   7.35    <1e-12   0.0223066   0.0385144
Race: white   0.181839   0.144766     1.26    0.2091  -0.101898    0.465576
Income        0.235267   0.038152     6.17    <1e-09   0.16049     0.310043
Educate       0.169276   0.0240098    7.05    <1e-11   0.122217    0.216334
────────────────────────────────────────────────────────────────────────────

julia> m2_4.BIC
2078.946461750509

```

 **Logistic Regression - with Ridge Prior**

```jldoctest examples
julia> m2_5 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_Ridge());

julia> m2_5.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    1.5314    0.6655     0.0067    0.0088   4736.5514    1.0003    ⋯
        β[1]   -2.8619    0.3365     0.0034    0.0047   4505.8853    1.0004    ⋯
        β[2]    0.0270    0.0035     0.0000    0.0000   6544.5015    1.0002    ⋯
        β[3]    0.2279    0.1446     0.0014    0.0015   6968.9017    1.0000    ⋯
        β[4]    0.1773    0.0274     0.0003    0.0004   6716.7864    0.9999    ⋯
        β[5]    0.1676    0.0208     0.0002    0.0003   5279.6598    1.0001    ⋯
                                                                1 column omitted

julia> m2_5.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.7570    1.1039    1.3921    1.7719    3.2092
        β[1]   -3.5306   -3.0918   -2.8651   -2.6338   -2.2062
        β[2]    0.0203    0.0247    0.0270    0.0295    0.0338
        β[3]   -0.0540    0.1311    0.2280    0.3264    0.5073
        β[4]    0.1243    0.1587    0.1767    0.1956    0.2327
        β[5]    0.1273    0.1535    0.1675    0.1815    0.2085

julia> m2_6 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Ridge(),1.0);

julia> m2_6.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.9025    0.3651     0.0037    0.0054   4480.4540    1.0000    ⋯
        β[1]   -1.6606    0.1915     0.0019    0.0030   4852.0370    0.9999    ⋯
        β[2]    0.0158    0.0020     0.0000    0.0000   7009.3493    1.0001    ⋯
        β[3]    0.1496    0.0856     0.0009    0.0009   7761.4054    0.9999    ⋯
        β[4]    0.0967    0.0148     0.0001    0.0001   7444.0661    1.0000    ⋯
        β[5]    0.0994    0.0118     0.0001    0.0002   5268.9316    0.9999    ⋯
                                                                1 column omitted

julia> m2_6.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.4607    0.6606    0.8182    1.0396    1.8407
        β[1]   -2.0383   -1.7904   -1.6583   -1.5311   -1.2863
        β[2]    0.0118    0.0144    0.0158    0.0171    0.0197
        β[3]   -0.0173    0.0925    0.1488    0.2073    0.3180
        β[4]    0.0679    0.0865    0.0967    0.1066    0.1257
        β[5]    0.0768    0.0914    0.0993    0.1072    0.1231

julia> m2_7 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Ridge(),1.0);

julia> m2_7.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.9928    0.3745     0.0037    0.0054   5503.9612    0.9999    ⋯
        β[1]   -1.8666    0.1845     0.0018    0.0026   5165.8451    1.0001    ⋯
        β[2]    0.0142    0.0018     0.0000    0.0000   7814.5837    1.0002    ⋯
        β[3]    0.1730    0.0860     0.0009    0.0009   6881.1222    0.9999    ⋯
        β[4]    0.0771    0.0120     0.0001    0.0001   7628.3991    1.0000    ⋯
        β[5]    0.0947    0.0109     0.0001    0.0002   5465.7776    1.0000    ⋯
                                                                1 column omitted

julia> m2_7.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.5248    0.7395    0.9092    1.1465    1.9612
        β[1]   -2.2325   -1.9919   -1.8664   -1.7424   -1.5047
        β[2]    0.0107    0.0130    0.0142    0.0154    0.0178
        β[3]    0.0038    0.1149    0.1732    0.2303    0.3426
        β[4]    0.0542    0.0687    0.0770    0.0851    0.1006
        β[5]    0.0728    0.0875    0.0947    0.1021    0.1162

julia> m2_8 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Ridge(),1.0);

julia> m2_8.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    1.5164    0.6110     0.0061    0.0088   5073.8844    0.9999    ⋯
        β[1]   -2.9665    0.3808     0.0038    0.0055   3900.2655    1.0001    ⋯
        β[2]    0.0286    0.0043     0.0000    0.0001   4943.5487    1.0002    ⋯
        β[3]    0.1604    0.1508     0.0015    0.0018   6891.3045    0.9999    ⋯
        β[4]    0.2395    0.0397     0.0004    0.0005   6038.7928    0.9999    ⋯
        β[5]    0.1592    0.0237     0.0002    0.0003   4328.8957    1.0000    ⋯
                                                                1 column omitted

julia> m2_8.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.7799    1.1154    1.3821    1.7577    3.0617
        β[1]   -3.7283   -3.2133   -2.9612   -2.7075   -2.2404
        β[2]    0.0205    0.0257    0.0286    0.0315    0.0374
        β[3]   -0.1449    0.0613    0.1635    0.2610    0.4474
        β[4]    0.1640    0.2126    0.2385    0.2651    0.3212
        β[5]    0.1148    0.1431    0.1583    0.1748    0.2070

```

 **Logistic Regression - with Laplace Prior**
```jldoctest examples
julia> m2_9 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_Laplace());

julia> m2_9.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.8729    0.5109     0.0051    0.0070   4168.8950    0.9999    ⋯
        β[1]   -2.8745    0.3293     0.0033    0.0043   4525.5742    0.9999    ⋯
        β[2]    0.0273    0.0035     0.0000    0.0000   6601.7518    0.9999    ⋯
        β[3]    0.2103    0.1424     0.0014    0.0013   7660.9194    1.0000    ⋯
        β[4]    0.1767    0.0271     0.0003    0.0003   7474.8745    1.0001    ⋯
        β[5]    0.1690    0.0202     0.0002    0.0003   4860.4472    1.0001    ⋯
                                                                1 column omitted

julia> m2_9.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.3280    0.5428    0.7460    1.0415    2.1841
        β[1]   -3.5072   -3.1004   -2.8769   -2.6541   -2.2327
        β[2]    0.0206    0.0250    0.0273    0.0297    0.0342
        β[3]   -0.0614    0.1128    0.2082    0.3091    0.4884
        β[4]    0.1252    0.1582    0.1762    0.1947    0.2305
        β[5]    0.1297    0.1552    0.1691    0.1827    0.2079

julia> m2_10 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Laplace());

julia> m2_10.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.8677    0.5061     0.0051    0.0066   5223.5887    0.9999    ⋯
        β[1]   -2.8772    0.3324     0.0033    0.0051   3975.5078    0.9999    ⋯
        β[2]    0.0273    0.0035     0.0000    0.0000   6116.3599    0.9999    ⋯
        β[3]    0.2075    0.1449     0.0014    0.0017   6674.4916    0.9999    ⋯
        β[4]    0.1772    0.0274     0.0003    0.0003   6721.6931    1.0000    ⋯
        β[5]    0.1693    0.0205     0.0002    0.0003   4354.2790    0.9999    ⋯
                                                                1 column omitted

julia> m2_10.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.3292    0.5522    0.7445    1.0321    2.1142
        β[1]   -3.5240   -3.1023   -2.8816   -2.6542   -2.2191
        β[2]    0.0205    0.0250    0.0273    0.0297    0.0341
        β[3]   -0.0676    0.1086    0.2058    0.3058    0.4905
        β[4]    0.1248    0.1584    0.1768    0.1960    0.2308
        β[5]    0.1296    0.1554    0.1693    0.1830    0.2104

julia> m2_11 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Laplace(),1.0);

julia> m2_11.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.9072    0.4890     0.0049    0.0069   4738.4116    1.0004    ⋯
        β[1]   -2.8796    0.3286     0.0033    0.0041   5837.4510    1.0000    ⋯
        β[2]    0.0273    0.0035     0.0000    0.0000   7668.8459    1.0001    ⋯
        β[3]    0.2121    0.1436     0.0014    0.0015   7916.5774    1.0002    ⋯
        β[4]    0.1777    0.0268     0.0003    0.0003   8707.6308    0.9999    ⋯
        β[5]    0.1691    0.0204     0.0002    0.0002   6188.7181    0.9999    ⋯
                                                                1 column omitted

julia> m2_11.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.3746    0.5961    0.7889    1.0716    2.1763
        β[1]   -3.5249   -3.1023   -2.8795   -2.6541   -2.2457
        β[2]    0.0207    0.0249    0.0273    0.0297    0.0341
        β[3]   -0.0624    0.1129    0.2114    0.3114    0.4930
        β[4]    0.1253    0.1597    0.1772    0.1962    0.2305
        β[5]    0.1290    0.1553    0.1691    0.1827    0.2089

julia> m2_12 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Laplace(),1.0);

julia> m2_12.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.8988    0.4514     0.0045    0.0063   5520.6699    1.0006    ⋯
        β[1]   -2.8904    0.3304     0.0033    0.0043   4419.1261    0.9999    ⋯
        β[2]    0.0274    0.0034     0.0000    0.0000   5962.6203    0.9999    ⋯
        β[3]    0.2133    0.1421     0.0014    0.0017   6693.5568    1.0008    ⋯
        β[4]    0.1773    0.0271     0.0003    0.0003   7664.6326    0.9999    ⋯
        β[5]    0.1698    0.0203     0.0002    0.0003   4712.2368    1.0001    ⋯
                                                                1 column omitted

julia> m2_12.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.3770    0.5991    0.7909    1.0748    2.0284
        β[1]   -3.5400   -3.1123   -2.8919   -2.6659   -2.2510
        β[2]    0.0208    0.0250    0.0274    0.0297    0.0343
        β[3]   -0.0553    0.1161    0.2109    0.3058    0.4987
        β[4]    0.1243    0.1587    0.1771    0.1953    0.2310
        β[5]    0.1308    0.1558    0.1695    0.1834    0.2095

```

 **Logistic Regression - with Cauchy Prior**
```jldoctest examples
julia> m2_13 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_Cauchy(),1.0);

julia> m2_13.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.3060    0.2319     0.0023    0.0028   6670.6339    0.9999    ⋯
        β[1]   -2.9196    0.3286     0.0033    0.0044   4636.5902    0.9999    ⋯
        β[2]    0.0280    0.0034     0.0000    0.0000   6407.9414    0.9999    ⋯
        β[3]    0.1791    0.1358     0.0014    0.0019   5612.7276    1.0000    ⋯
        β[4]    0.1773    0.0273     0.0003    0.0004   6538.4075    1.0014    ⋯
        β[5]    0.1725    0.0205     0.0002    0.0003   5017.4150    1.0000    ⋯
                                                                1 column omitted

julia> m2_13.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.0648    0.1582    0.2433    0.3780    0.9132
        β[1]   -3.5551   -3.1405   -2.9220   -2.6936   -2.2841
        β[2]    0.0213    0.0257    0.0279    0.0303    0.0348
        β[3]   -0.0580    0.0786    0.1723    0.2712    0.4634
        β[4]    0.1232    0.1586    0.1774    0.1957    0.2306
        β[5]    0.1326    0.1585    0.1724    0.1866    0.2122

julia> m2_14 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Cauchy(),2.0,30000);

julia> m2_14.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.3083    0.2464     0.0025    0.0030   6527.1307    0.9999    ⋯
        β[1]   -2.9164    0.3268     0.0033    0.0054   4001.0802    0.9999    ⋯
        β[2]    0.0279    0.0034     0.0000    0.0000   5853.3102    0.9999    ⋯
        β[3]    0.1778    0.1384     0.0014    0.0017   6653.8555    0.9999    ⋯
        β[4]    0.1766    0.0270     0.0003    0.0004   5995.1423    1.0002    ⋯
        β[5]    0.1727    0.0202     0.0002    0.0003   4212.0545    1.0000    ⋯
                                                                1 column omitted

julia> m2_14.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.0642    0.1553    0.2436    0.3806    0.9498
        β[1]   -3.5544   -3.1398   -2.9155   -2.6942   -2.2855
        β[2]    0.0211    0.0256    0.0280    0.0303    0.0348
        β[3]   -0.0658    0.0766    0.1706    0.2721    0.4679
        β[4]    0.1239    0.1584    0.1769    0.1946    0.2298
        β[5]    0.1332    0.1592    0.1727    0.1860    0.2127
           
julia> m2_15 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Cauchy(),1.0);

julia> m2_15.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.3077    0.2368     0.0024    0.0031   5706.3441    1.0003    ⋯
        β[1]   -2.9128    0.3295     0.0033    0.0042   5051.3410    0.9999    ⋯
        β[2]    0.0279    0.0035     0.0000    0.0000   6723.5109    0.9999    ⋯
        β[3]    0.1785    0.1376     0.0014    0.0017   6301.5177    0.9999    ⋯
        β[4]    0.1772    0.0270     0.0003    0.0003   7536.6524    0.9999    ⋯
        β[5]    0.1724    0.0203     0.0002    0.0002   5509.0935    0.9999    ⋯
                                                                1 column omitted

julia> m2_15.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.0647    0.1573    0.2431    0.3769    0.9521
        β[1]   -3.5601   -3.1328   -2.9126   -2.6877   -2.2779
        β[2]    0.0213    0.0255    0.0278    0.0302    0.0349
        β[3]   -0.0625    0.0791    0.1714    0.2699    0.4611
        β[4]    0.1243    0.1590    0.1768    0.1952    0.2306
        β[5]    0.1329    0.1587    0.1723    0.1860    0.2117
              
julia> m2_16 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Cauchy(),1.0);

julia> m2_16.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.3049    0.2370     0.0024    0.0029   5684.3206    1.0000    ⋯
        β[1]   -2.9134    0.3314     0.0033    0.0046   5120.8859    1.0000    ⋯
        β[2]    0.0279    0.0034     0.0000    0.0000   6977.9876    0.9999    ⋯
        β[3]    0.1752    0.1372     0.0014    0.0015   6830.2857    1.0000    ⋯
        β[4]    0.1770    0.0267     0.0003    0.0003   8103.9753    0.9999    ⋯
        β[5]    0.1726    0.0204     0.0002    0.0003   5655.1416    0.9999    ⋯
                                                                1 column omitted

julia> m2_16.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.0667    0.1556    0.2404    0.3788    0.9164
        β[1]   -3.5774   -3.1275   -2.9059   -2.6959   -2.2603
        β[2]    0.0212    0.0255    0.0279    0.0303    0.0347
        β[3]   -0.0675    0.0764    0.1688    0.2667    0.4589
        β[4]    0.1254    0.1584    0.1769    0.1950    0.2298
        β[5]    0.1331    0.1588    0.1724    0.1864    0.2138

```

 **Logistic Regression - with T-Dist Prior**
```jldoctest examples
julia> m2_17 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_TDist(),1.0);

julia> m2_17.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.5289    0.3511     0.0035    0.0056   4448.5392    0.9999    ⋯
           ν    2.2155   10.7265     0.1073    0.1899   3067.8091    1.0003    ⋯
        β[1]   -2.9351    0.3287     0.0033    0.0040   5364.8317    1.0002    ⋯
        β[2]    0.0278    0.0035     0.0000    0.0000   7461.5560    0.9999    ⋯
        β[3]    0.2065    0.1429     0.0014    0.0017   7208.6730    0.9999    ⋯
        β[4]    0.1768    0.0278     0.0003    0.0003   7523.2925    0.9999    ⋯
        β[5]    0.1727    0.0206     0.0002    0.0002   5830.2383    1.0002    ⋯
                                                                1 column omitted

julia> m2_17.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.1780    0.3040    0.4273    0.6348    1.4809
           ν    0.3755    0.7694    1.1673    1.8724    7.4220
        β[1]   -3.5790   -3.1586   -2.9311   -2.7106   -2.3021
        β[2]    0.0210    0.0255    0.0278    0.0302    0.0350
        β[3]   -0.0652    0.1084    0.2039    0.2996    0.4975
        β[4]    0.1230    0.1579    0.1765    0.1955    0.2328
        β[5]    0.1325    0.1587    0.1727    0.1864    0.2135

julia> m2_18 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_TDist(),1.0);

julia> m2_18.summaries
Summary Statistics
  parameters       mean       std   naive_se      mcse       ess      rhat   e ⋯
      Symbol    Float64   Float64    Float64   Float64   Float64   Float64     ⋯

           λ     4.8118    0.0029     0.0000    0.0003   21.0934    1.3204     ⋯
           ν     3.7561    0.0016     0.0000    0.0002   22.4348    1.1310     ⋯
        β[1]   -22.6598    0.0004     0.0000    0.0000   22.2429    1.2323     ⋯
        β[2]     0.8228    0.0000     0.0000    0.0000   66.1383    1.1389     ⋯
        β[3]    -2.0240    0.0004     0.0000    0.0000   21.0386    2.6936     ⋯
        β[4]    -8.7439    0.0027     0.0000    0.0003   20.2520    2.8354     ⋯
        β[5]    -6.9804    0.0062     0.0001    0.0006   20.2317    2.7320     ⋯
                                                                1 column omitted

julia> m2_18.quantiles
Quantiles
  parameters       2.5%      25.0%      50.0%      75.0%      97.5%
      Symbol    Float64    Float64    Float64    Float64    Float64

           λ     4.8048     4.8110     4.8128     4.8138     4.8151
           ν     3.7527     3.7552     3.7562     3.7574     3.7584
        β[1]   -22.6608   -22.6600   -22.6598   -22.6596   -22.6592
        β[2]     0.8228     0.8228     0.8228     0.8228     0.8229
        β[3]    -2.0247    -2.0243    -2.0240    -2.0237    -2.0234
        β[4]    -8.7480    -8.7470    -8.7436    -8.7415    -8.7392
        β[5]    -6.9900    -6.9861    -6.9801    -6.9747    -6.9706

julia> m2_19 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_TDist(),1.0);

julia> m2_19.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse       ess      rhat   es ⋯
      Symbol   Float64   Float64    Float64   Float64   Float64   Float64      ⋯

           λ    0.7808    0.4602     0.0046    0.0453   23.5174    1.2619      ⋯
           ν    3.7505    4.1940     0.0419    0.4159   21.9030    1.2467      ⋯
        β[1]   -2.7421    0.3762     0.0038    0.0363   26.8105    1.1348      ⋯
        β[2]    0.0415    0.0723     0.0007    0.0072   27.8089    1.0384      ⋯
        β[3]    0.1168    0.4182     0.0042    0.0408   26.9065    1.0372      ⋯
        β[4]   -0.0110    0.8895     0.0089    0.0882   27.0833    1.0456      ⋯
        β[5]    0.1890    0.1184     0.0012    0.0112   29.1688    1.0274      ⋯
                                                                1 column omitted

julia> m2_19.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.2057    0.4125    0.6491    1.1041    1.8468
           ν    0.6845    1.3588    2.1210    3.7455   17.0191
        β[1]   -3.3513   -3.0447   -2.7642   -2.4708   -2.0116
        β[2]    0.0195    0.0242    0.0269    0.0295    0.3964
        β[3]   -1.6313    0.1010    0.2012    0.2958    0.4656
        β[4]   -4.3298    0.1543    0.1750    0.1952    0.2313
        β[5]    0.1229    0.1497    0.1663    0.1821    0.6713

julia> m2_20 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_TDist(),1.0);

julia> m2_20.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.5314    0.3468     0.0035    0.0044   4487.2447    1.0004    ⋯
           ν    1.9974    5.5248     0.0552    0.0980   3013.3594    0.9999    ⋯
        β[1]   -2.9315    0.3305     0.0033    0.0040   5717.6278    1.0002    ⋯
        β[2]    0.0278    0.0035     0.0000    0.0000   7735.5418    1.0001    ⋯
        β[3]    0.2045    0.1403     0.0014    0.0016   8011.0127    1.0006    ⋯
        β[4]    0.1772    0.0273     0.0003    0.0003   7454.0591    0.9999    ⋯
        β[5]    0.1724    0.0206     0.0002    0.0003   5544.3491    1.0011    ⋯
                                                                1 column omitted

julia> m2_20.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.1763    0.3082    0.4331    0.6374    1.4639
           ν    0.3746    0.7738    1.1719    1.8817    7.2365
        β[1]   -3.5830   -3.1509   -2.9295   -2.7051   -2.2834
        β[2]    0.0210    0.0255    0.0278    0.0302    0.0345
        β[3]   -0.0670    0.1086    0.2025    0.2996    0.4815
        β[4]    0.1252    0.1586    0.1770    0.1952    0.2317
        β[5]    0.1327    0.1586    0.1724    0.1863    0.2136

```

 **Logistic Regression - with Uniform Prior**
```jldoctest examples
julia> m2_21 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_Uniform(),1.0);

julia> m2_21.summaries
Summary Statistics
  parameters      mean        std   naive_se      mcse         ess      rhat   ⋯
      Symbol   Float64    Float64    Float64   Float64     Float64   Float64   ⋯

           v   13.7852   180.2962     1.8030    2.6824   4217.9790    1.0006   ⋯
        β[1]   -0.6624     0.0000     0.0000    0.0000     20.5530    0.9999   ⋯
        β[2]    0.0103     0.0023     0.0000    0.0000   8889.1797    0.9999   ⋯
        β[3]   -0.0034     0.1519     0.0015    0.0019   7495.4943    0.9999   ⋯
        β[4]    0.1710     0.0259     0.0003    0.0003   7518.2286    1.0000   ⋯
        β[5]    0.0637     0.0126     0.0001    0.0002   6785.0123    1.0000   ⋯
                                                                1 column omitted

julia> m2_21.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    0.8333    1.5379    2.6688    6.2389   61.2451
        β[1]   -0.6624   -0.6624   -0.6624   -0.6624   -0.6624
        β[2]    0.0058    0.0087    0.0103    0.0118    0.0147
        β[3]   -0.3015   -0.1034   -0.0040    0.0998    0.2916
        β[4]    0.1211    0.1531    0.1708    0.1884    0.2225
        β[5]    0.0383    0.0554    0.0639    0.0722    0.0882

julia> m2_22 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Uniform(),1.0);

julia> m2_22.summaries
Summary Statistics
  parameters      mean        std   naive_se      mcse         ess      rhat   ⋯
      Symbol   Float64    Float64    Float64   Float64     Float64   Float64   ⋯

           v   15.3310   222.0068     2.2201    4.0557   3034.3611    1.0003   ⋯
        β[1]   -0.3434     0.0000     0.0000    0.0000     20.5530    0.9999   ⋯
        β[2]    0.0099     0.0021     0.0000    0.0000   9458.2021    0.9999   ⋯
        β[3]   -0.3434     0.0000     0.0000    0.0000     20.5530    0.9999   ⋯
        β[4]    0.1789     0.0266     0.0003    0.0003   8784.2763    0.9999   ⋯
        β[5]    0.0608     0.0117     0.0001    0.0001   9076.6331    0.9999   ⋯
                                                                1 column omitted

julia> m2_22.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    0.5648    1.1828    2.2363    5.0195   54.2485
        β[1]   -0.3434   -0.3434   -0.3434   -0.3434   -0.3434
        β[2]    0.0057    0.0085    0.0099    0.0113    0.0141
        β[3]   -0.3434   -0.3434   -0.3434   -0.3434   -0.3434
        β[4]    0.1281    0.1610    0.1784    0.1968    0.2315
        β[5]    0.0378    0.0529    0.0609    0.0689    0.0832

julia> m2_23 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Uniform(),1.0);

julia> m2_23.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           v    7.3458   35.8188     0.3582    0.5261   4241.8175    0.9999    ⋯
        β[1]   -0.3274    0.0000     0.0000    0.0000         NaN       NaN    ⋯
        β[2]    0.0097    0.0021     0.0000    0.0000   9364.4462    0.9999    ⋯
        β[3]   -0.3274    0.0000     0.0000    0.0000         NaN       NaN    ⋯
        β[4]    0.1791    0.0261     0.0003    0.0002   9621.3881    0.9999    ⋯
        β[5]    0.0590    0.0113     0.0001    0.0001   9168.4454    0.9999    ⋯
                                                                1 column omitted

julia> m2_23.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    0.5398    1.1335    2.0818    4.7264   43.0439
        β[1]   -0.3274   -0.3274   -0.3274   -0.3274   -0.3274
        β[2]    0.0055    0.0083    0.0097    0.0111    0.0137
        β[3]   -0.3274   -0.3274   -0.3274   -0.3274   -0.3274
        β[4]    0.1298    0.1612    0.1788    0.1962    0.2318
        β[5]    0.0370    0.0515    0.0590    0.0665    0.0812
                
julia> m2_24 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Uniform(),1.0);

julia> m2_24.summaries
Summary Statistics
  parameters      mean         std   naive_se      mcse         ess      rhat  ⋯
      Symbol   Float64     Float64    Float64   Float64     Float64   Float64  ⋯

           v   50.7212   2454.3435    24.5434   39.2977   4005.7567    1.0001  ⋯
        β[1]   -0.5073      0.0000     0.0000    0.0000     20.5530    0.9999  ⋯
        β[2]    0.0123      0.0021     0.0000    0.0000   9294.7660    0.9999  ⋯
        β[3]   -0.5073      0.0000     0.0000    0.0000     20.5530    0.9999  ⋯
        β[4]    0.1826      0.0263     0.0003    0.0003   8350.3257    0.9999  ⋯
        β[5]    0.0761      0.0116     0.0001    0.0001   8318.5904    1.0000  ⋯
                                                                1 column omitted

julia> m2_24.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    0.7552    1.5016    2.7477    6.3105   68.6481
        β[1]   -0.5073   -0.5073   -0.5073   -0.5073   -0.5073
        β[2]    0.0082    0.0109    0.0123    0.0137    0.0165
        β[3]   -0.5073   -0.5073   -0.5073   -0.5073   -0.5073
        β[4]    0.1307    0.1651    0.1824    0.1998    0.2345
        β[5]    0.0533    0.0682    0.0760    0.0838    0.0985

```

## Example 3: Poisson Regression

 **Poisson Regression - Likelihood analysis**
```jldoctest examples
sanction = dataset("Zelig", "sanction")
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
   7 │     1      2       2       0       1      2      3  little effect
   8 │     0      1       3       0       0      2      3  little effect
  ⋮  │   ⋮      ⋮      ⋮       ⋮       ⋮       ⋮      ⋮          ⋮
  72 │     0      2       2       0       0      1      8  net gain
  73 │     1      3       1       1       1      2     14  little effect
  74 │     0      2       1       0       0      1      2  net gain
  75 │     0      1       3       0       1      2      1  little effect
  76 │     0      4       3       1       0      2     13  little effect
  77 │     0      1       2       0       0      1      1  net gain
  78 │     1      3       1       1       1      2     10  little effect
                                                          63 rows omitted

julia> m3_1 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression());

julia> m3_1.fit
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

julia> m3_1.LogLike
-284.33693448347356

julia> m3_1.AIC
580.6738689669471

julia> m3_1.BIC
594.8141219270847

```

 **Poisson Regression with Ridge Prior**
```jldoctest examples
julia> m3_2 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Ridge());

julia> m3_2.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    1.3047    0.4939     0.0049    0.0080   4614.5363    1.0003    ⋯
           α   -1.7955    0.2534     0.0025    0.0039   4454.7752    1.0000    ⋯
        β[1]    0.1390    0.0654     0.0007    0.0008   6391.9793    0.9999    ⋯
        β[2]    1.1322    0.0556     0.0006    0.0008   5857.7599    1.0002    ⋯
        β[3]   -0.3259    0.2278     0.0023    0.0028   7027.6031    1.0000    ⋯
        β[4]    1.6974    0.0996     0.0010    0.0012   7459.8841    0.9999    ⋯
        β[5]    0.4036    0.1676     0.0017    0.0024   6099.1495    1.0001    ⋯
                                                                1 column omitted

julia> m3_2.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.7118    0.9816    1.2004    1.4923    2.5232
           α   -2.2925   -1.9675   -1.7963   -1.6226   -1.2942
        β[1]    0.0123    0.0950    0.1384    0.1840    0.2668
        β[2]    1.0240    1.0943    1.1326    1.1705    1.2402
        β[3]   -0.7938   -0.4734   -0.3194   -0.1702    0.1013
        β[4]    1.5019    1.6310    1.6971    1.7643    1.8944
        β[5]    0.0683    0.2928    0.4049    0.5185    0.7256

```

 **Poisson Regression with Laplace Prior**
```jldoctest examples
julia> m3_3 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Laplace());

julia> m3_3.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    1.0855    0.5268     0.0053    0.0077   5271.8825    1.0005    ⋯
           α   -1.7913    0.2674     0.0027    0.0040   4231.1524    1.0000    ⋯
        β[1]    0.1358    0.0662     0.0007    0.0009   5785.6833    1.0000    ⋯
        β[2]    1.1320    0.0567     0.0006    0.0008   5446.3525    0.9999    ⋯
        β[3]   -0.2906    0.2195     0.0022    0.0026   7570.4683    0.9999    ⋯
        β[4]    1.7025    0.0998     0.0010    0.0011   7436.8894    1.0000    ⋯
        β[5]    0.3949    0.1730     0.0017    0.0021   6648.4939    1.0001    ⋯
                                                                1 column omitted

julia> m3_3.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.4543    0.7290    0.9624    1.2917    2.4373
           α   -2.3164   -1.9738   -1.7953   -1.6079   -1.2586
        β[1]    0.0048    0.0922    0.1364    0.1796    0.2644
        β[2]    1.0202    1.0935    1.1321    1.1708    1.2427
        β[3]   -0.7434   -0.4336   -0.2820   -0.1369    0.1038
        β[4]    1.5084    1.6358    1.7020    1.7679    1.8996
        β[5]    0.0503    0.2775    0.3959    0.5131    0.7315

```

**Poisson Regression with Cauchy Prior**
```jldoctest examples
julia> m3_4 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Cauchy());

julia> m3_4.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.8456    0.4396     0.0044    0.0047   6775.8311    1.0001    ⋯
           α   -1.7969    0.2640     0.0026    0.0037   5127.9411    0.9999    ⋯
        β[1]    0.1388    0.0654     0.0007    0.0007   7076.7634    0.9999    ⋯
        β[2]    1.1314    0.0562     0.0006    0.0008   6216.6410    0.9999    ⋯
        β[3]   -0.2934    0.2153     0.0022    0.0026   6756.2170    0.9999    ⋯
        β[4]    1.7055    0.0983     0.0010    0.0011   7390.4902    0.9999    ⋯
        β[5]    0.3928    0.1675     0.0017    0.0019   6509.6692    0.9999    ⋯
                                                                1 column omitted

julia> m3_4.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.2957    0.5448    0.7464    1.0367    1.9870
           α   -2.3106   -1.9751   -1.7946   -1.6238   -1.2737
        β[1]    0.0120    0.0944    0.1382    0.1830    0.2674
        β[2]    1.0237    1.0934    1.1317    1.1692    1.2417
        β[3]   -0.7479   -0.4322   -0.2864   -0.1464    0.1089
        β[4]    1.5156    1.6389    1.7055    1.7721    1.8945
        β[5]    0.0585    0.2811    0.3952    0.5066    0.7126

```

 **Poisson Regression with TDist Prior**
```jldoctest examples
julia> m3_5 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_TDist());

julia> m3_5.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.9972    0.4285     0.0043    0.0047   8603.3650    0.9999    ⋯
           ν    2.9413    5.2859     0.0529    0.0916   3402.3197    1.0000    ⋯
           α   -1.8112    0.2604     0.0026    0.0031   5515.2271    0.9999    ⋯
        β[1]    0.1410    0.0655     0.0007    0.0007   7259.2984    0.9999    ⋯
        β[2]    1.1344    0.0557     0.0006    0.0006   7189.8249    0.9999    ⋯
        β[3]   -0.3121    0.2224     0.0022    0.0026   7858.4358    0.9999    ⋯
        β[4]    1.7029    0.0998     0.0010    0.0011   7737.1039    1.0000    ⋯
        β[5]    0.4039    0.1692     0.0017    0.0019   6815.5865    1.0004    ⋯
                                                                1 column omitted

julia> m3_5.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.3985    0.7021    0.9209    1.2049    2.0244
           ν    0.5626    1.1544    1.8296    3.0831   12.0174
           α   -2.3161   -1.9881   -1.8143   -1.6302   -1.3049
        β[1]    0.0134    0.0961    0.1410    0.1859    0.2687
        β[2]    1.0236    1.0970    1.1348    1.1720    1.2409
        β[3]   -0.7690   -0.4572   -0.3059   -0.1591    0.1062
        β[4]    1.5070    1.6357    1.7014    1.7695    1.8992
        β[5]    0.0714    0.2891    0.4055    0.5202    0.7335

```

 **Poisson Regression with Uniform Prior**
```jldoctest examples
julia> m3_6 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Uniform());

julia> m3_6.summaries
Summary Statistics
  parameters      mean        std   naive_se      mcse          ess      rhat  ⋯
      Symbol   Float64    Float64    Float64   Float64      Float64   Float64  ⋯

           λ   31.7237   269.4096     2.6941    4.6703    3141.7386    1.0002  ⋯
           α   -1.6312     0.0000     0.0000    0.0000          NaN       NaN  ⋯
        β[1]    0.1113     0.0484     0.0005    0.0005    7517.0556    0.9999  ⋯
        β[2]    1.1174     0.0351     0.0004    0.0004    7513.8645    1.0000  ⋯
        β[3]   -0.3724     0.2208     0.0022    0.0020   10488.8748    0.9999  ⋯
        β[4]    1.6312     0.0000     0.0000    0.0000          NaN       NaN  ⋯
        β[5]    0.3333     0.1345     0.0013    0.0012   10310.8481    1.0000  ⋯
                                                                1 column omitted

julia> m3_6.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%      97.5%
      Symbol   Float64   Float64   Float64   Float64    Float64

           λ    2.3312    4.1421    7.0190   15.1146   160.2559
           α   -1.6312   -1.6312   -1.6312   -1.6312    -1.6312
        β[1]    0.0177    0.0783    0.1108    0.1441     0.2055
        β[2]    1.0485    1.0932    1.1182    1.1415     1.1844
        β[3]   -0.8226   -0.5151   -0.3640   -0.2234     0.0370
        β[4]    1.6312    1.6312    1.6312    1.6312     1.6312
        β[5]    0.0619    0.2449    0.3343    0.4241     0.5893

```

## Example 4: Negative Binomial Regression

```jldoctest examples
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
   7 │     1      2       2       0       1      2      3  little effect
   8 │     0      1       3       0       0      2      3  little effect
  ⋮  │   ⋮      ⋮      ⋮       ⋮       ⋮       ⋮      ⋮          ⋮
  72 │     0      2       2       0       0      1      8  net gain
  73 │     1      3       1       1       1      2     14  little effect
  74 │     0      2       1       0       0      1      2  net gain
  75 │     0      1       3       0       1      2      1  little effect
  76 │     0      4       3       1       0      2     13  little effect
  77 │     0      1       2       0       0      1      1  net gain
  78 │     1      3       1       1       1      2     10  little effect
                                                          63 rows omitted

```

**Negative Binomial Regression - Likelihood method** 
```jldoctest examples
julia> m4_1 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression());

julia> m4_1.fit
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

julia> m4_1.AIC
363.85804286542685

julia> m4_1.BIC
377.9982958255644

```

 **NegativeBinomial Regression with Ridge Prior**
```jldoctest examples
julia> m4_2 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Ridge());

julia> m4_2.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    2.0413    0.4480     0.0045    0.0051   8307.4837    0.9999    ⋯
           α   -1.0893    0.5193     0.0052    0.0087   3899.6757    0.9999    ⋯
        β[1]   -0.0048    0.1616     0.0016    0.0023   5386.0221    0.9999    ⋯
        β[2]    1.0642    0.1311     0.0013    0.0018   5604.7062    0.9999    ⋯
        β[3]   -0.1729    0.5552     0.0056    0.0058   8708.3320    1.0000    ⋯
        β[4]    1.2807    0.3178     0.0032    0.0034   8557.2897    0.9999    ⋯
        β[5]    0.1556    0.2840     0.0028    0.0036   6126.1145    0.9999    ⋯
                                                                1 column omitted

julia> m4_2.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    1.3137    1.7227    1.9925    2.3068    3.0644
           α   -2.1030   -1.4363   -1.0925   -0.7387   -0.0708
        β[1]   -0.3276   -0.1114   -0.0035    0.1031    0.3128
        β[2]    0.8102    0.9757    1.0640    1.1518    1.3243
        β[3]   -1.2103   -0.5564   -0.1824    0.1904    0.9515
        β[4]    0.6475    1.0692    1.2835    1.4937    1.9101
        β[5]   -0.3947   -0.0331    0.1557    0.3436    0.7122

```

 **NegativeBinomial Regression with Laplace Prior**
```jldoctest examples
julia> m4_3 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Laplace());

julia> m4_3.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    2.1062    0.4582     0.0046    0.0057   5683.7921    1.0001    ⋯
           α   -1.0101    0.5154     0.0052    0.0084   3795.0884    1.0002    ⋯
        β[1]   -0.0204    0.1590     0.0016    0.0022   5324.1385    1.0001    ⋯
        β[2]    1.0495    0.1329     0.0013    0.0018   4926.4231    1.0002    ⋯
        β[3]   -0.1473    0.5059     0.0051    0.0054   7542.6944    1.0000    ⋯
        β[4]    1.2814    0.3226     0.0032    0.0038   6496.8709    1.0000    ⋯
        β[5]    0.1210    0.2777     0.0028    0.0039   6228.0313    1.0000    ⋯
                                                                1 column omitted

julia> m4_3.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    1.3411    1.7873    2.0594    2.3793    3.1542
           α   -2.0197   -1.3645   -1.0063   -0.6555   -0.0143
        β[1]   -0.3337   -0.1265   -0.0194    0.0874    0.2908
        β[2]    0.7878    0.9605    1.0493    1.1380    1.3108
        β[3]   -1.1584   -0.4804   -0.1489    0.1745    0.8704
        β[4]    0.6527    1.0626    1.2827    1.4935    1.9379
        β[5]   -0.4316   -0.0628    0.1198    0.3065    0.6654

```


**Negative Binomial Regression with Cauchy Prior**
```jldoctest examples
m4_4 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Cauchy())

m4_4.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    2.0321    0.4414     0.0044    0.0046   7966.3509    1.0000    ⋯
           α   -1.0429    0.5163     0.0052    0.0092   3481.2902    0.9999    ⋯
        β[1]   -0.0115    0.1626     0.0016    0.0024   4862.1372    0.9999    ⋯
        β[2]    1.0545    0.1318     0.0013    0.0019   5091.0562    0.9999    ⋯
        β[3]   -0.1563    0.5484     0.0055    0.0058   7173.1471    0.9999    ⋯
        β[4]    1.2755    0.3303     0.0033    0.0036   7331.3107    1.0000    ⋯
        β[5]    0.1370    0.2782     0.0028    0.0038   5766.5502    0.9999    ⋯
                                                                1 column omitted

m4_4.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    1.3087    1.7197    1.9826    2.2938    3.0432
           α   -2.0846   -1.3881   -1.0360   -0.6882   -0.0508
        β[1]   -0.3277   -0.1201   -0.0135    0.1004    0.3104
        β[2]    0.8001    0.9644    1.0524    1.1424    1.3192
        β[3]   -1.1840   -0.5340   -0.1662    0.1949    0.9594
        β[4]    0.6424    1.0565    1.2674    1.4899    1.9446
        β[5]   -0.4143   -0.0500    0.1407    0.3254    0.6775

```

 **Negative Binomial Regression with TDist Prior**
```jldoctest examples
julia> m4_5 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_TDist());
ERROR: DomainError with Dual{ForwardDiff.Tag{Turing.Essential.var"#f#4"{DynamicPPL.TypedVarInfo{NamedTuple{(:λ, :ν, :α, :β), Tuple{DynamicPPL.Metadata{Dict{AbstractPPL.VarName{:λ, Setfield.IdentityLens}, Int64}, Vector{Distributions.InverseGamma{Float64}}, Vector{AbstractPPL.VarName{:λ, Setfield.IdentityLens}}, Vector{Float64}, Vector{Set{DynamicPPL.Selector}}}, DynamicPPL.Metadata{Dict{AbstractPPL.VarName{:ν, Setfield.IdentityLens}, Int64}, Vector{Distributions.InverseGamma{Float64}}, Vector{AbstractPPL.VarName{:ν, Setfield.IdentityLens}}, Vector{Float64}, Vector{Set{DynamicPPL.Selector}}}, DynamicPPL.Metadata{Dict{AbstractPPL.VarName{:α, Setfield.IdentityLens}, Int64}, Vector{Distributions.LocationScale{Float64, Distributions.Continuous, Distributions.TDist{Float64}}}, Vector{AbstractPPL.VarName{:α, Setfield.IdentityLens}}, Vector{Float64}, Vector{Set{DynamicPPL.Selector}}}, DynamicPPL.Metadata{Dict{AbstractPPL.VarName{:β, Setfield.IdentityLens}, Int64}, Vector{Distributions.Product{Distributions.Continuous, Distributions.LocationScale{Float64, Distributions.Continuous, Distributions.TDist{Float64}}, FillArrays.Fill{Distributions.LocationScale{Float64, Distributions.Continuous, Distributions.TDist{Float64}}, 1, Tuple{Base.OneTo{Int64}}}}}, Vector{AbstractPPL.VarName{:β, Setfield.IdentityLens}}, Vector{Float64}, Vector{Set{DynamicPPL.Selector}}}}}, Float64}, DynamicPPL.Model{CRRao.var"#NegBinomReg#19"{Float64}, (:X, :y), (), (), Tuple{Matrix{Float64}, Vector{Int32}}, Tuple{}, DynamicPPL.DefaultContext}, DynamicPPL.Sampler{Turing.Inference.NUTS{Turing.Essential.ForwardDiffAD{0}, (), AdvancedHMC.DiagEuclideanMetric}}, DynamicPPL.DefaultContext}, Float64}}(0.0,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN):
NegativeBinomial: the condition zero(p) < p <= one(p) is not satisfied.

julia> m4_5.summaries
ERROR: UndefVarError: m4_5 not defined

julia> m4_5.quantiles
ERROR: UndefVarError: m4_5 not defined

```

 **Negative Binomial Regression with Uniform Prior**
```jldoctest examples
julia> m4_6 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Uniform(),1.0);

julia> m4_6.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    3.2065    0.4036     0.0040    0.0054   4236.5829    1.0011    ⋯
           α   -2.5288    0.0000     0.0000    0.0000         NaN       NaN    ⋯
        β[1]    0.3328    0.0984     0.0010    0.0027    204.3503    1.0604    ⋯
        β[2]    1.3577    0.0936     0.0009    0.0037     85.4193    1.1485    ⋯
        β[3]   -1.5072    1.0686     0.0107    0.0999     21.9705    3.2882    ⋯
        β[4]    1.0054    0.3172     0.0032    0.0167     49.1963    1.3045    ⋯
        β[5]    0.5403    0.2348     0.0023    0.0097     75.8215    1.1720    ⋯
                                                                1 column omitted

julia> m4_6.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    2.6474    2.9139    3.1328    3.4106    4.1948
           α   -2.5288   -2.5288   -2.5288   -2.5288   -2.5288
        β[1]    0.1367    0.2674    0.3346    0.3987    0.5259
        β[2]    1.1746    1.2938    1.3580    1.4202    1.5452
        β[3]   -2.5288   -2.5288   -1.5590   -0.5322    0.3807
        β[4]    0.3983    0.7860    1.0072    1.2188    1.6254
        β[5]    0.0954    0.3776    0.5400    0.6997    1.0010

```