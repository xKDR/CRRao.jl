## Examples: Setting up the code

```jldoctest examples
julia> using RDatasets, NLSolversBase, CRRao, Logging;

julia> Logging.disable_logging(Logging.Warn); CRRao.setprogress!(false);

julia> CRRao_seed(123);
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

           v    6.8472    3.9651     0.0397    0.0666   3261.1590    1.0002    ⋯
           σ    2.6820    0.3926     0.0039    0.0072   3565.1579    1.0002    ⋯
           α   28.5844    5.4028     0.0540    0.1130   2224.4727    1.0000    ⋯
        β[1]   -0.0398    0.0107     0.0001    0.0002   3518.9607    1.0005    ⋯
        β[2]   -2.6764    0.9619     0.0096    0.0202   2321.2765    1.0005    ⋯
        β[3]    1.6066    0.9842     0.0098    0.0197   2394.6319    0.9999    ⋯
                                                                1 column omitted

julia> m1_2.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    2.4615    4.4044    5.9078    8.1748   16.7856
           σ    2.0447    2.4085    2.6358    2.9049    3.5792
           α   17.4931   25.1660   28.8441   32.2267   38.6689
        β[1]   -0.0618   -0.0468   -0.0396   -0.0329   -0.0188
        β[2]   -4.5193   -3.3150   -2.7025   -2.0792   -0.6935
        β[3]   -0.2176    0.9400    1.5686    2.2308    3.6497
```


 **Linear Regression - Laplace Prior**

```jldoctest examples
julia> m1_3 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Laplace());

julia> m1_3.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           v    4.3153    3.1487     0.0315    0.0473   3853.2427    0.9999    ⋯
           σ    2.6640    0.3842     0.0038    0.0054   3801.4191    1.0001    ⋯
           α   29.2166    5.0280     0.0503    0.0954   2739.9530    0.9999    ⋯
        β[1]   -0.0394    0.0103     0.0001    0.0002   4450.9936    0.9999    ⋯
        β[2]   -2.7473    0.9074     0.0091    0.0173   2956.2937    0.9999    ⋯
        β[3]    1.4796    0.9122     0.0091    0.0169   2863.6450    1.0000    ⋯
                                                                1 column omitted

julia> m1_3.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    1.2608    2.4252    3.4891    5.1536   12.5245
           σ    2.0428    2.3938    2.6188    2.8862    3.5394
           α   18.8463   26.1502   29.4403   32.6286   38.3664
        β[1]   -0.0599   -0.0460   -0.0392   -0.0325   -0.0197
        β[2]   -4.4117   -3.3681   -2.7783   -2.1923   -0.8503
        β[3]   -0.1671    0.8518    1.4508    2.0578    3.3588
```

 **Linear Regression - Cauchy Prior**
```jldoctest examples
julia> m1_4 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Cauchy(),20000);

julia> m1_4.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           σ    2.5888    0.3475     0.0025    0.0038   8583.3523    1.0003    ⋯
           α   30.4212    4.6081     0.0326    0.0643   4846.4270    1.0006    ⋯
        β[1]   -0.0392    0.0099     0.0001    0.0001   6723.4938    1.0008    ⋯
        β[2]   -2.8533    0.8478     0.0060    0.0120   5117.1757    1.0008    ⋯
        β[3]    1.2459    0.8391     0.0059    0.0113   5144.2713    1.0004    ⋯
                                                                1 column omitted

julia> m1_4.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           σ    2.0137    2.3431    2.5519    2.7934    3.3670
           α   21.0866   27.4450   30.5285   33.5397   39.2914
        β[1]   -0.0590   -0.0457   -0.0391   -0.0327   -0.0202
        β[2]   -4.4846   -3.4182   -2.8619   -2.2933   -1.1317
        β[3]   -0.3612    0.6764    1.2254    1.7792    2.9538
```

 **Linear Regression - T-Distributed Prior**

```jldoctest examples
julia> m1_5 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_TDist());

julia> m1_5.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           ν    1.0574    0.6103     0.0061    0.0095   3655.5585    0.9999    ⋯
           σ    2.6252    0.3634     0.0036    0.0058   4149.1617    0.9999    ⋯
           α   30.2842    4.8008     0.0480    0.0979   2564.6609    1.0001    ⋯
        β[1]   -0.0394    0.0103     0.0001    0.0002   3974.7879    0.9999    ⋯
        β[2]   -2.8365    0.8780     0.0088    0.0165   2894.8736    0.9999    ⋯
        β[3]    1.2741    0.8715     0.0087    0.0174   2675.3506    1.0001    ⋯
                                                                1 column omitted

julia> m1_5.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           ν    0.3717    0.6671    0.9174    1.2935    2.5072
           σ    2.0280    2.3683    2.5859    2.8448    3.4403
           α   20.4646   27.2122   30.4015   33.5678   39.4157
        β[1]   -0.0596   -0.0461   -0.0393   -0.0326   -0.0193
        β[2]   -4.5542   -3.4196   -2.8514   -2.2667   -1.0681
        β[3]   -0.3839    0.6881    1.2483    1.8371    3.0702
```

 **Linear Regression - Uniform Prior**
```jldoctest examples
julia> m1_6 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_TDist());

julia> m1_6.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           ν    1.0574    0.6103     0.0061    0.0095   3655.5585    0.9999    ⋯
           σ    2.6252    0.3634     0.0036    0.0058   4149.1617    0.9999    ⋯
           α   30.2842    4.8008     0.0480    0.0979   2564.6609    1.0001    ⋯
        β[1]   -0.0394    0.0103     0.0001    0.0002   3974.7879    0.9999    ⋯
        β[2]   -2.8365    0.8780     0.0088    0.0165   2894.8736    0.9999    ⋯
        β[3]    1.2741    0.8715     0.0087    0.0174   2675.3506    1.0001    ⋯
                                                                1 column omitted

julia> m1_6.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           ν    0.3717    0.6671    0.9174    1.2935    2.5072
           σ    2.0280    2.3683    2.5859    2.8448    3.4403
           α   20.4646   27.2122   30.4015   33.5678   39.4157
        β[1]   -0.0596   -0.0461   -0.0393   -0.0326   -0.0193
        β[2]   -4.5542   -3.4196   -2.8514   -2.2667   -1.0681
        β[3]   -0.3839    0.6881    1.2483    1.8371    3.0702
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

           λ    1.5372    0.6825     0.0068    0.0090   4699.5179    0.9999    ⋯
        β[1]   -2.8659    0.3285     0.0033    0.0044   4940.2410    1.0002    ⋯
        β[2]    0.0271    0.0035     0.0000    0.0000   6757.3608    0.9999    ⋯
        β[3]    0.2293    0.1449     0.0014    0.0018   6894.9720    1.0001    ⋯
        β[4]    0.1774    0.0272     0.0003    0.0004   7279.5100    1.0002    ⋯
        β[5]    0.1678    0.0205     0.0002    0.0003   5484.0619    1.0002    ⋯
                                                                1 column omitted

julia> m2_5.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.7565    1.0976    1.3837    1.7861    3.2992
        β[1]   -3.5009   -3.0890   -2.8642   -2.6411   -2.2094
        β[2]    0.0203    0.0247    0.0271    0.0295    0.0340
        β[3]   -0.0578    0.1331    0.2302    0.3272    0.5135
        β[4]    0.1253    0.1587    0.1772    0.1957    0.2314
        β[5]    0.1276    0.1541    0.1679    0.1820    0.2077


julia> m2_6 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Ridge(),1.0);

julia> m2_6.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.8974    0.3516     0.0035    0.0055   4766.0718    1.0004    ⋯
        β[1]   -1.6640    0.1915     0.0019    0.0030   4483.8131    1.0003    ⋯
        β[2]    0.0158    0.0020     0.0000    0.0000   7100.6926    1.0002    ⋯
        β[3]    0.1501    0.0870     0.0009    0.0011   6954.9861    0.9999    ⋯
        β[4]    0.0964    0.0143     0.0001    0.0002   8092.4537    0.9999    ⋯
        β[5]    0.0997    0.0117     0.0001    0.0002   5151.0238    1.0002    ⋯
                                                                1 column omitted

julia> m2_6.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.4628    0.6641    0.8191    1.0405    1.7562
        β[1]   -2.0392   -1.7949   -1.6625   -1.5332   -1.2890
        β[2]    0.0118    0.0144    0.0158    0.0171    0.0197
        β[3]   -0.0224    0.0905    0.1507    0.2100    0.3175
        β[4]    0.0680    0.0868    0.0964    0.1059    0.1244
        β[5]    0.0764    0.0918    0.0996    0.1075    0.1228


julia> m2_7 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Ridge(),1.0);

julia> m2_7.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.9863    0.3754     0.0038    0.0047   5487.6972    0.9999    ⋯
        β[1]   -1.8649    0.1860     0.0019    0.0025   5954.0953    1.0001    ⋯
        β[2]    0.0142    0.0019     0.0000    0.0000   9071.5557    0.9999    ⋯
        β[3]    0.1730    0.0851     0.0009    0.0010   8023.9162    1.0002    ⋯
        β[4]    0.0769    0.0123     0.0001    0.0001   7536.1264    0.9999    ⋯
        β[5]    0.0947    0.0112     0.0001    0.0001   6251.6050    0.9999    ⋯
                                                                1 column omitted

julia> m2_7.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.5239    0.7336    0.9009    1.1478    1.9601
        β[1]   -2.2303   -1.9903   -1.8638   -1.7379   -1.5141
        β[2]    0.0107    0.0129    0.0142    0.0155    0.0179
        β[3]    0.0074    0.1144    0.1727    0.2305    0.3396
        β[4]    0.0526    0.0687    0.0768    0.0854    0.1007
        β[5]    0.0733    0.0871    0.0946    0.1024    0.1170


julia> m2_8 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Ridge(),1.0);

julia> m2_8.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    1.5233    0.6058     0.0061    0.0090   5251.4092    1.0000    ⋯
        β[1]   -2.9883    0.3942     0.0039    0.0060   4750.7406    1.0001    ⋯
        β[2]    0.0289    0.0044     0.0000    0.0001   5952.1391    1.0002    ⋯
        β[3]    0.1589    0.1528     0.0015    0.0018   5618.7281    0.9999    ⋯
        β[4]    0.2395    0.0399     0.0004    0.0005   6082.9861    1.0005    ⋯
        β[5]    0.1604    0.0241     0.0002    0.0004   5048.3073    1.0003    ⋯
                                                                1 column omitted

julia> m2_8.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.7718    1.1194    1.3885    1.7735    3.1323
        β[1]   -3.8019   -3.2411   -2.9768   -2.7227   -2.2400
        β[2]    0.0208    0.0259    0.0287    0.0317    0.0379
        β[3]   -0.1462    0.0572    0.1599    0.2635    0.4511
        β[4]    0.1639    0.2123    0.2385    0.2653    0.3206
        β[5]    0.1152    0.1440    0.1596    0.1762    0.2097
```

 **Logistic Regression - with Laplace Prior**
```jldoctest examples
julia> m2_9 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_Laplace());

julia> m2_9.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.8658    0.4932     0.0049    0.0069   4709.3634    1.0001    ⋯
        β[1]   -2.8704    0.3336     0.0033    0.0048   5134.2137    1.0000    ⋯
        β[2]    0.0273    0.0035     0.0000    0.0000   6552.4899    0.9999    ⋯
        β[3]    0.2106    0.1463     0.0015    0.0018   6941.0532    0.9999    ⋯
        β[4]    0.1776    0.0271     0.0003    0.0003   6854.0396    0.9999    ⋯
        β[5]    0.1687    0.0207     0.0002    0.0003   5680.7532    1.0000    ⋯
                                                                1 column omitted

julia> m2_9.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.3334    0.5500    0.7447    1.0356    2.1764
        β[1]   -3.5208   -3.0940   -2.8733   -2.6430   -2.2124
        β[2]    0.0205    0.0249    0.0273    0.0296    0.0341
        β[3]   -0.0654    0.1072    0.2088    0.3097    0.5061
        β[4]    0.1263    0.1590    0.1770    0.1952    0.2320
        β[5]    0.1275    0.1549    0.1686    0.1828    0.2097

julia> m2_10 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Laplace());

julia> m2_10.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.8658    0.4932     0.0049    0.0069   4709.3634    1.0001    ⋯
        β[1]   -2.8704    0.3336     0.0033    0.0048   5134.2137    1.0000    ⋯
        β[2]    0.0273    0.0035     0.0000    0.0000   6552.4899    0.9999    ⋯
        β[3]    0.2106    0.1463     0.0015    0.0018   6941.0532    0.9999    ⋯
        β[4]    0.1776    0.0271     0.0003    0.0003   6854.0396    0.9999    ⋯
        β[5]    0.1687    0.0207     0.0002    0.0003   5680.7532    1.0000    ⋯
                                                                1 column omitted

julia> m2_10.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.3334    0.5500    0.7447    1.0356    2.1764
        β[1]   -3.5208   -3.0940   -2.8733   -2.6430   -2.2124
        β[2]    0.0205    0.0249    0.0273    0.0296    0.0341
        β[3]   -0.0654    0.1072    0.2088    0.3097    0.5061
        β[4]    0.1263    0.1590    0.1770    0.1952    0.2320
        β[5]    0.1275    0.1549    0.1686    0.1828    0.2097


julia> m2_11 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Laplace(),1.0);

julia> m2_11.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.9026    0.4653     0.0047    0.0064   4863.7878    1.0000    ⋯
        β[1]   -2.8813    0.3311     0.0033    0.0045   5456.6584    1.0005    ⋯
        β[2]    0.0273    0.0035     0.0000    0.0000   7030.4545    1.0000    ⋯
        β[3]    0.2133    0.1413     0.0014    0.0016   8023.5642    1.0002    ⋯
        β[4]    0.1771    0.0267     0.0003    0.0003   8059.5351    1.0000    ⋯
        β[5]    0.1695    0.0203     0.0002    0.0003   6003.4945    1.0003    ⋯
                                                                1 column omitted

julia> m2_11.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.3833    0.6002    0.7948    1.0682    2.1101
        β[1]   -3.5374   -3.1016   -2.8841   -2.6589   -2.2367
        β[2]    0.0207    0.0249    0.0273    0.0297    0.0342
        β[3]   -0.0606    0.1179    0.2129    0.3079    0.4917
        β[4]    0.1256    0.1591    0.1771    0.1948    0.2294
        β[5]    0.1296    0.1558    0.1695    0.1833    0.2092


julia> m2_12 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Laplace(),1.0);

julia> m2_12.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.9026    0.4653     0.0047    0.0064   4863.7878    1.0000    ⋯
        β[1]   -2.8813    0.3311     0.0033    0.0045   5456.6584    1.0005    ⋯
        β[2]    0.0273    0.0035     0.0000    0.0000   7030.4545    1.0000    ⋯
        β[3]    0.2133    0.1413     0.0014    0.0016   8023.5642    1.0002    ⋯
        β[4]    0.1771    0.0267     0.0003    0.0003   8059.5351    1.0000    ⋯
        β[5]    0.1695    0.0203     0.0002    0.0003   6003.4945    1.0003    ⋯
                                                                1 column omitted

julia> m2_12.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.3833    0.6002    0.7948    1.0682    2.1101
        β[1]   -3.5374   -3.1016   -2.8841   -2.6589   -2.2367
        β[2]    0.0207    0.0249    0.0273    0.0297    0.0342
        β[3]   -0.0606    0.1179    0.2129    0.3079    0.4917
        β[4]    0.1256    0.1591    0.1771    0.1948    0.2294
        β[5]    0.1296    0.1558    0.1695    0.1833    0.2092
```

 **Logistic Regression - with Cauchy Prior**
```jldoctest examples
julia> m2_13 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_Cauchy(),1.0);

julia> m2_13.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.3100    0.2478     0.0025    0.0034   5605.2824    1.0013    ⋯
        β[1]   -2.9090    0.3257     0.0033    0.0049   5376.8054    1.0008    ⋯
        β[2]    0.0279    0.0035     0.0000    0.0000   7004.5181    1.0006    ⋯
        β[3]    0.1768    0.1384     0.0014    0.0018   5821.5948    1.0000    ⋯
        β[4]    0.1770    0.0273     0.0003    0.0004   7017.1793    1.0000    ⋯
        β[5]    0.1723    0.0204     0.0002    0.0003   5482.9126    1.0003    ⋯
                                                                1 column omitted

julia> m2_13.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.0650    0.1570    0.2446    0.3821    0.9595
        β[1]   -3.5631   -3.1231   -2.9068   -2.6885   -2.2846
        β[2]    0.0212    0.0255    0.0278    0.0302    0.0347
        β[3]   -0.0695    0.0765    0.1684    0.2695    0.4671
        β[4]    0.1240    0.1585    0.1770    0.1952    0.2313
        β[5]    0.1328    0.1585    0.1720    0.1861    0.2131


julia> m2_14 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Cauchy(),2.0,30000);

julia> m2_14.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.3100    0.2478     0.0025    0.0034   5605.2824    1.0013    ⋯
        β[1]   -2.9090    0.3257     0.0033    0.0049   5376.8054    1.0008    ⋯
        β[2]    0.0279    0.0035     0.0000    0.0000   7004.5181    1.0006    ⋯
        β[3]    0.1768    0.1384     0.0014    0.0018   5821.5948    1.0000    ⋯
        β[4]    0.1770    0.0273     0.0003    0.0004   7017.1793    1.0000    ⋯
        β[5]    0.1723    0.0204     0.0002    0.0003   5482.9126    1.0003    ⋯
                                                                1 column omitted

julia> m2_14.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.0650    0.1570    0.2446    0.3821    0.9595
        β[1]   -3.5631   -3.1231   -2.9068   -2.6885   -2.2846
        β[2]    0.0212    0.0255    0.0278    0.0302    0.0347
        β[3]   -0.0695    0.0765    0.1684    0.2695    0.4671
        β[4]    0.1240    0.1585    0.1770    0.1952    0.2313
        β[5]    0.1328    0.1585    0.1720    0.1861    0.2131
                
julia> m2_15 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Cauchy(),1.0);

julia> m2_15.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.3100    0.2478     0.0025    0.0034   5605.2824    1.0013    ⋯
        β[1]   -2.9090    0.3257     0.0033    0.0049   5376.8054    1.0008    ⋯
        β[2]    0.0279    0.0035     0.0000    0.0000   7004.5181    1.0006    ⋯
        β[3]    0.1768    0.1384     0.0014    0.0018   5821.5948    1.0000    ⋯
        β[4]    0.1770    0.0273     0.0003    0.0004   7017.1793    1.0000    ⋯
        β[5]    0.1723    0.0204     0.0002    0.0003   5482.9126    1.0003    ⋯
                                                                1 column omitted

julia> m2_15.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.0650    0.1570    0.2446    0.3821    0.9595
        β[1]   -3.5631   -3.1231   -2.9068   -2.6885   -2.2846
        β[2]    0.0212    0.0255    0.0278    0.0302    0.0347
        β[3]   -0.0695    0.0765    0.1684    0.2695    0.4671
        β[4]    0.1240    0.1585    0.1770    0.1952    0.2313
        β[5]    0.1328    0.1585    0.1720    0.1861    0.2131
                

julia> m2_16 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Cauchy(),1.0);

julia> m2_16.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.3100    0.2478     0.0025    0.0034   5605.2824    1.0013    ⋯
        β[1]   -2.9090    0.3257     0.0033    0.0049   5376.8054    1.0008    ⋯
        β[2]    0.0279    0.0035     0.0000    0.0000   7004.5181    1.0006    ⋯
        β[3]    0.1768    0.1384     0.0014    0.0018   5821.5948    1.0000    ⋯
        β[4]    0.1770    0.0273     0.0003    0.0004   7017.1793    1.0000    ⋯
        β[5]    0.1723    0.0204     0.0002    0.0003   5482.9126    1.0003    ⋯
                                                                1 column omitted

julia> m2_16.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.0650    0.1570    0.2446    0.3821    0.9595
        β[1]   -3.5631   -3.1231   -2.9068   -2.6885   -2.2846
        β[2]    0.0212    0.0255    0.0278    0.0302    0.0347
        β[3]   -0.0695    0.0765    0.1684    0.2695    0.4671
        β[4]    0.1240    0.1585    0.1770    0.1952    0.2313
        β[5]    0.1328    0.1585    0.1720    0.1861    0.2131
```

 **Logistic Regression - with T-Dist Prior**
```jldoctest examples
julia> m2_17 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_TDist(),1.0);

julia> m2_17.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.5359    0.3600     0.0036    0.0052   4597.0424    0.9999    ⋯
           ν    2.3246   15.5231     0.1552    0.2815   3060.4891    1.0002    ⋯
        β[1]   -2.9360    0.3329     0.0033    0.0051   5185.3889    0.9999    ⋯
        β[2]    0.0279    0.0035     0.0000    0.0000   7333.0907    1.0000    ⋯
        β[3]    0.2057    0.1420     0.0014    0.0015   8778.7328    0.9999    ⋯
        β[4]    0.1770    0.0274     0.0003    0.0003   6953.1952    0.9999    ⋯
        β[5]    0.1726    0.0205     0.0002    0.0003   5407.3816    0.9999    ⋯
                                                                1 column omitted

julia> m2_17.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.1714    0.3041    0.4354    0.6468    1.5183
           ν    0.3738    0.7694    1.1794    1.8559    7.8037
        β[1]   -3.5985   -3.1605   -2.9304   -2.7117   -2.2988
        β[2]    0.0210    0.0254    0.0278    0.0303    0.0348
        β[3]   -0.0641    0.1088    0.2014    0.3020    0.4906
        β[4]    0.1240    0.1583    0.1769    0.1955    0.2313
        β[5]    0.1331    0.1587    0.1725    0.1865    0.2134

julia> m2_18 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_TDist(),1.0);

julia> m2_18.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.5359    0.3600     0.0036    0.0052   4597.0424    0.9999    ⋯
           ν    2.3246   15.5231     0.1552    0.2815   3060.4891    1.0002    ⋯
        β[1]   -2.9360    0.3329     0.0033    0.0051   5185.3889    0.9999    ⋯
        β[2]    0.0279    0.0035     0.0000    0.0000   7333.0907    1.0000    ⋯
        β[3]    0.2057    0.1420     0.0014    0.0015   8778.7328    0.9999    ⋯
        β[4]    0.1770    0.0274     0.0003    0.0003   6953.1952    0.9999    ⋯
        β[5]    0.1726    0.0205     0.0002    0.0003   5407.3816    0.9999    ⋯
                                                                1 column omitted

julia> m2_18.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.1714    0.3041    0.4354    0.6468    1.5183
           ν    0.3738    0.7694    1.1794    1.8559    7.8037
        β[1]   -3.5985   -3.1605   -2.9304   -2.7117   -2.2988
        β[2]    0.0210    0.0254    0.0278    0.0303    0.0348
        β[3]   -0.0641    0.1088    0.2014    0.3020    0.4906
        β[4]    0.1240    0.1583    0.1769    0.1955    0.2313
        β[5]    0.1331    0.1587    0.1725    0.1865    0.2134


julia> m2_19 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_TDist(),1.0);

julia> m2_19.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.5359    0.3600     0.0036    0.0052   4597.0424    0.9999    ⋯
           ν    2.3246   15.5231     0.1552    0.2815   3060.4891    1.0002    ⋯
        β[1]   -2.9360    0.3329     0.0033    0.0051   5185.3889    0.9999    ⋯
        β[2]    0.0279    0.0035     0.0000    0.0000   7333.0907    1.0000    ⋯
        β[3]    0.2057    0.1420     0.0014    0.0015   8778.7328    0.9999    ⋯
        β[4]    0.1770    0.0274     0.0003    0.0003   6953.1952    0.9999    ⋯
        β[5]    0.1726    0.0205     0.0002    0.0003   5407.3816    0.9999    ⋯
                                                                1 column omitted

julia> m2_19.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.1714    0.3041    0.4354    0.6468    1.5183
           ν    0.3738    0.7694    1.1794    1.8559    7.8037
        β[1]   -3.5985   -3.1605   -2.9304   -2.7117   -2.2988
        β[2]    0.0210    0.0254    0.0278    0.0303    0.0348
        β[3]   -0.0641    0.1088    0.2014    0.3020    0.4906
        β[4]    0.1240    0.1583    0.1769    0.1955    0.2313
        β[5]    0.1331    0.1587    0.1725    0.1865    0.2134

julia> m2_20 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_TDist(),1.0);

julia> m2_20.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.5359    0.3600     0.0036    0.0052   4597.0424    0.9999    ⋯
           ν    2.3246   15.5231     0.1552    0.2815   3060.4891    1.0002    ⋯
        β[1]   -2.9360    0.3329     0.0033    0.0051   5185.3889    0.9999    ⋯
        β[2]    0.0279    0.0035     0.0000    0.0000   7333.0907    1.0000    ⋯
        β[3]    0.2057    0.1420     0.0014    0.0015   8778.7328    0.9999    ⋯
        β[4]    0.1770    0.0274     0.0003    0.0003   6953.1952    0.9999    ⋯
        β[5]    0.1726    0.0205     0.0002    0.0003   5407.3816    0.9999    ⋯
                                                                1 column omitted

julia> m2_20.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.1714    0.3041    0.4354    0.6468    1.5183
           ν    0.3738    0.7694    1.1794    1.8559    7.8037
        β[1]   -3.5985   -3.1605   -2.9304   -2.7117   -2.2988
        β[2]    0.0210    0.0254    0.0278    0.0303    0.0348
        β[3]   -0.0641    0.1088    0.2014    0.3020    0.4906
        β[4]    0.1240    0.1583    0.1769    0.1955    0.2313
        β[5]    0.1331    0.1587    0.1725    0.1865    0.2134
```

 **Logistic Regression - with Uniform Prior**
```jldoctest examples
julia> m2_21 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Logit(),Prior_Uniform(),1.0);

julia> m2_21.summaries
Summary Statistics
  parameters      mean        std   naive_se      mcse         ess      rhat   ⋯
      Symbol   Float64    Float64    Float64   Float64     Float64   Float64   ⋯

           v   14.9507   144.9283     1.4493    2.1998   4323.3028    1.0006   ⋯
        β[1]   -1.0886     0.0000     0.0000    0.0000     20.5530    0.9999   ⋯
        β[2]    0.0135     0.0023     0.0000    0.0000   8486.4851    1.0001   ⋯
        β[3]    0.0494     0.1478     0.0015    0.0017   6222.2810    0.9999   ⋯
        β[4]    0.1719     0.0256     0.0003    0.0003   7382.3021    1.0000   ⋯
        β[5]    0.0832     0.0124     0.0001    0.0002   6599.2571    0.9999   ⋯
                                                                1 column omitted

julia> m2_21.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    1.3158    2.2473    3.8181    8.1392   74.7266
        β[1]   -1.0886   -1.0886   -1.0886   -1.0886   -1.0886
        β[2]    0.0090    0.0119    0.0134    0.0150    0.0181
        β[3]   -0.2424   -0.0513    0.0513    0.1512    0.3315
        β[4]    0.1221    0.1546    0.1715    0.1890    0.2228
        β[5]    0.0591    0.0750    0.0833    0.0914    0.1076


julia> m2_22 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Probit(),Prior_Uniform(),1.0);

julia> m2_22.summaries
Summary Statistics
  parameters      mean        std   naive_se      mcse         ess      rhat   ⋯
      Symbol   Float64    Float64    Float64   Float64     Float64   Float64   ⋯

           v   14.9507   144.9283     1.4493    2.1998   4323.3028    1.0006   ⋯
        β[1]   -1.0886     0.0000     0.0000    0.0000     20.5530    0.9999   ⋯
        β[2]    0.0135     0.0023     0.0000    0.0000   8486.4851    1.0001   ⋯
        β[3]    0.0494     0.1478     0.0015    0.0017   6222.2810    0.9999   ⋯
        β[4]    0.1719     0.0256     0.0003    0.0003   7382.3021    1.0000   ⋯
        β[5]    0.0832     0.0124     0.0001    0.0002   6599.2571    0.9999   ⋯
                                                                1 column omitted

julia> m2_22.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    1.3158    2.2473    3.8181    8.1392   74.7266
        β[1]   -1.0886   -1.0886   -1.0886   -1.0886   -1.0886
        β[2]    0.0090    0.0119    0.0134    0.0150    0.0181
        β[3]   -0.2424   -0.0513    0.0513    0.1512    0.3315
        β[4]    0.1221    0.1546    0.1715    0.1890    0.2228
        β[5]    0.0591    0.0750    0.0833    0.0914    0.1076

julia> m2_23 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cloglog(),Prior_Uniform(),1.0);

julia> m2_23.summaries
Summary Statistics
  parameters      mean        std   naive_se      mcse         ess      rhat   ⋯
      Symbol   Float64    Float64    Float64   Float64     Float64   Float64   ⋯

           v   14.9507   144.9283     1.4493    2.1998   4323.3028    1.0006   ⋯
        β[1]   -1.0886     0.0000     0.0000    0.0000     20.5530    0.9999   ⋯
        β[2]    0.0135     0.0023     0.0000    0.0000   8486.4851    1.0001   ⋯
        β[3]    0.0494     0.1478     0.0015    0.0017   6222.2810    0.9999   ⋯
        β[4]    0.1719     0.0256     0.0003    0.0003   7382.3021    1.0000   ⋯
        β[5]    0.0832     0.0124     0.0001    0.0002   6599.2571    0.9999   ⋯
                                                                1 column omitted

julia> m2_23.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    1.3158    2.2473    3.8181    8.1392   74.7266
        β[1]   -1.0886   -1.0886   -1.0886   -1.0886   -1.0886
        β[2]    0.0090    0.0119    0.0134    0.0150    0.0181
        β[3]   -0.2424   -0.0513    0.0513    0.1512    0.3315
        β[4]    0.1221    0.1546    0.1715    0.1890    0.2228
        β[5]    0.0591    0.0750    0.0833    0.0914    0.1076
                
julia> m2_24 = @fitmodel((Vote ~ Age + Race +Income + Educate),turnout
                       ,LogisticRegression(),Cauchit(),Prior_Uniform(),1.0);

julia> m2_24.summaries
Summary Statistics
  parameters      mean        std   naive_se      mcse         ess      rhat   ⋯
      Symbol   Float64    Float64    Float64   Float64     Float64   Float64   ⋯

           v   14.9507   144.9283     1.4493    2.1998   4323.3028    1.0006   ⋯
        β[1]   -1.0886     0.0000     0.0000    0.0000     20.5530    0.9999   ⋯
        β[2]    0.0135     0.0023     0.0000    0.0000   8486.4851    1.0001   ⋯
        β[3]    0.0494     0.1478     0.0015    0.0017   6222.2810    0.9999   ⋯
        β[4]    0.1719     0.0256     0.0003    0.0003   7382.3021    1.0000   ⋯
        β[5]    0.0832     0.0124     0.0001    0.0002   6599.2571    0.9999   ⋯
                                                                1 column omitted

julia> m2_24.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    1.3158    2.2473    3.8181    8.1392   74.7266
        β[1]   -1.0886   -1.0886   -1.0886   -1.0886   -1.0886
        β[2]    0.0090    0.0119    0.0134    0.0150    0.0181
        β[3]   -0.2424   -0.0513    0.0513    0.1512    0.3315
        β[4]    0.1221    0.1546    0.1715    0.1890    0.2228
        β[5]    0.0591    0.0750    0.0833    0.0914    0.1076
```

## Example 3: Poisson Regression

 **Poisson Regression - Likelihood analysis**
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

           λ    1.3035    0.4759     0.0048    0.0060   5792.0856    1.0000    ⋯
           α   -1.7996    0.2611     0.0026    0.0040   4057.0885    1.0002    ⋯
        β[1]    0.1384    0.0651     0.0007    0.0008   5683.9844    0.9999    ⋯
        β[2]    1.1340    0.0564     0.0006    0.0008   5294.9292    1.0003    ⋯
        β[3]   -0.3301    0.2260     0.0023    0.0032   5773.7915    1.0000    ⋯
        β[4]    1.6967    0.1002     0.0010    0.0012   5452.7214    1.0002    ⋯
        β[5]    0.4065    0.1693     0.0017    0.0023   5568.8862    1.0006    ⋯
                                                                1 column omitted

julia> m3_2.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.7155    0.9826    1.1971    1.5081    2.4875
           α   -2.3196   -1.9734   -1.7948   -1.6245   -1.2966
        β[1]    0.0104    0.0955    0.1378    0.1820    0.2673
        β[2]    1.0247    1.0960    1.1339    1.1717    1.2451
        β[3]   -0.7949   -0.4772   -0.3237   -0.1768    0.0987
        β[4]    1.5022    1.6287    1.6954    1.7638    1.8937
        β[5]    0.0735    0.2906    0.4075    0.5218    0.7383

```

 **Poisson Regression with Laplace Prior**
```jldoctest examples
julia> m3_3 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Laplace());

julia> m3_3.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    1.0888    0.5691     0.0057    0.0067   6034.6151    0.9999    ⋯
           α   -1.7810    0.2650     0.0027    0.0038   4713.4915    1.0000    ⋯
        β[1]    0.1340    0.0654     0.0007    0.0008   6785.6872    1.0001    ⋯
        β[2]    1.1307    0.0570     0.0006    0.0008   5775.8048    0.9999    ⋯
        β[3]   -0.2916    0.2199     0.0022    0.0026   8178.4736    1.0000    ⋯
        β[4]    1.7001    0.1001     0.0010    0.0013   6869.9510    1.0000    ⋯
        β[5]    0.3900    0.1733     0.0017    0.0024   5567.0883    0.9999    ⋯
                                                                1 column omitted

julia> m3_3.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.4561    0.7238    0.9534    1.2941    2.4814
           α   -2.3089   -1.9581   -1.7783   -1.5996   -1.2690
        β[1]    0.0074    0.0901    0.1334    0.1779    0.2648
        β[2]    1.0181    1.0924    1.1307    1.1696    1.2425
        β[3]   -0.7499   -0.4350   -0.2833   -0.1369    0.1059
        β[4]    1.5050    1.6349    1.6991    1.7666    1.8988
        β[5]    0.0448    0.2748    0.3902    0.5059    0.7298

```

**Poisson Regression with Cauchy Prior**
```jldoctest examples
julia> m3_4 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Cauchy());

julia> m3_4.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.8407    0.4405     0.0044    0.0052   7620.0858    0.9999    ⋯
           α   -1.7985    0.2668     0.0027    0.0040   4794.9791    1.0001    ⋯
        β[1]    0.1382    0.0661     0.0007    0.0008   7074.1508    0.9999    ⋯
        β[2]    1.1327    0.0562     0.0006    0.0008   5743.9180    1.0001    ⋯
        β[3]   -0.2939    0.2214     0.0022    0.0026   7493.5982    1.0003    ⋯
        β[4]    1.7023    0.0995     0.0010    0.0011   7280.7799    1.0001    ⋯
        β[5]    0.3905    0.1697     0.0017    0.0021   6036.1766    1.0001    ⋯
                                                                1 column omitted

julia> m3_4.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.2987    0.5350    0.7381    1.0340    1.9455
           α   -2.3260   -1.9766   -1.7955   -1.6153   -1.2873
        β[1]    0.0096    0.0939    0.1378    0.1829    0.2688
        β[2]    1.0233    1.0945    1.1326    1.1703    1.2419
        β[3]   -0.7489   -0.4378   -0.2850   -0.1417    0.1182
        β[4]    1.5121    1.6344    1.7019    1.7698    1.8990
        β[5]    0.0634    0.2715    0.3913    0.5052    0.7191

```

 **Poisson Regression with TDist Prior**
```jldoctest examples
julia> m3_5 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_TDist());

julia> m3_5.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    0.9931    0.4263     0.0043    0.0051   7130.9985    0.9999    ⋯
           ν    2.9270    5.6485     0.0565    0.0743   5213.5802    1.0002    ⋯
           α   -1.8091    0.2562     0.0026    0.0031   5384.3847    0.9999    ⋯
        β[1]    0.1411    0.0644     0.0006    0.0006   7198.9055    1.0000    ⋯
        β[2]    1.1341    0.0564     0.0006    0.0007   6479.3772    0.9999    ⋯
        β[3]   -0.3090    0.2218     0.0022    0.0025   8771.7465    1.0003    ⋯
        β[4]    1.7011    0.1006     0.0010    0.0011   8387.8752    1.0005    ⋯
        β[5]    0.4029    0.1677     0.0017    0.0018   7190.0471    1.0002    ⋯
                                                                1 column omitted

julia> m3_5.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    0.4067    0.7024    0.9134    1.1924    2.0507
           ν    0.5677    1.1442    1.7835    3.0446   11.6848
           α   -2.3164   -1.9833   -1.8034   -1.6359   -1.3175
        β[1]    0.0154    0.0981    0.1405    0.1840    0.2686
        β[2]    1.0240    1.0961    1.1340    1.1722    1.2450
        β[3]   -0.7610   -0.4543   -0.3032   -0.1569    0.1106
        β[4]    1.5079    1.6314    1.7006    1.7686    1.8983
        β[5]    0.0725    0.2891    0.4032    0.5156    0.7335

```

 **Poisson Regression with Uniform Prior**
```jldoctest examples
julia> m3_6 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Uniform());

julia> m3_6.summaries
Summary Statistics
  parameters      mean        std   naive_se      mcse          ess      rhat  ⋯
      Symbol   Float64    Float64    Float64   Float64      Float64   Float64  ⋯

           λ   20.7320   168.8424     1.6884    2.1490    5720.1240    0.9999  ⋯
           α   -1.0886     0.0000     0.0000    0.0000      20.5530    0.9999  ⋯
        β[1]    0.0704     0.0183     0.0002    0.0002   10135.9926    1.0002  ⋯
        β[2]    1.0886     0.0000     0.0000    0.0000      20.5530    0.9999  ⋯
        β[3]   -1.0886     0.0000     0.0000    0.0000      20.5530    0.9999  ⋯
        β[4]    1.0886     0.0000     0.0000    0.0000      20.5530    0.9999  ⋯
        β[5]   -0.0809     0.1332     0.0013    0.0013    9996.2268    0.9999  ⋯
                                                                1 column omitted

julia> m3_6.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%      97.5%
      Symbol   Float64   Float64   Float64   Float64    Float64

           λ    1.8045    3.3858    5.8117   12.0525   104.4603
           α   -1.0886   -1.0886   -1.0886   -1.0886    -1.0886
        β[1]    0.0339    0.0583    0.0706    0.0827     0.1063
        β[2]    1.0886    1.0886    1.0886    1.0886     1.0886
        β[3]   -1.0886   -1.0886   -1.0886   -1.0886    -1.0886
        β[4]    1.0886    1.0886    1.0886    1.0886     1.0886
        β[5]   -0.3485   -0.1694   -0.0772    0.0102     0.1736
        
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

           λ    2.0313    0.4409     0.0044    0.0050   7647.2593    1.0000    ⋯
           α   -1.0821    0.5148     0.0051    0.0078   3989.9298    1.0000    ⋯
        β[1]   -0.0057    0.1617     0.0016    0.0021   5310.4953    0.9999    ⋯
        β[2]    1.0636    0.1323     0.0013    0.0017   5847.4460    0.9999    ⋯
        β[3]   -0.1637    0.5580     0.0056    0.0062   7316.3762    1.0000    ⋯
        β[4]    1.2756    0.3206     0.0032    0.0046   5971.4021    1.0001    ⋯
        β[5]    0.1544    0.2804     0.0028    0.0037   5542.0961    1.0004    ⋯
                                                                1 column omitted

julia> m4_2.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    1.2991    1.7153    1.9857    2.2929    3.0073
           α   -2.1119   -1.4265   -1.0798   -0.7350   -0.0891
        β[1]   -0.3314   -0.1116   -0.0030    0.1030    0.3036
        β[2]    0.8116    0.9738    1.0631    1.1524    1.3291
        β[3]   -1.2359   -0.5395   -0.1747    0.1994    0.9628
        β[4]    0.6414    1.0614    1.2729    1.4889    1.9101
        β[5]   -0.3895   -0.0373    0.1524    0.3421    0.7095

```

 **NegativeBinomial Regression with Laplace Prior**
```jldoctest examples
julia> m4_3 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Laplace());

julia> m4_3.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           λ    2.1206    0.4655     0.0047    0.0063   5901.9259    1.0000    ⋯
           α   -1.0132    0.5072     0.0051    0.0076   4295.9489    1.0000    ⋯
        β[1]   -0.0192    0.1541     0.0015    0.0021   5628.5021    0.9999    ⋯
        β[2]    1.0494    0.1321     0.0013    0.0016   5694.6372    1.0000    ⋯
        β[3]   -0.1506    0.5087     0.0051    0.0053   6510.9094    1.0000    ⋯
        β[4]    1.2814    0.3195     0.0032    0.0038   6534.2472    0.9999    ⋯
        β[5]    0.1238    0.2707     0.0027    0.0034   6383.3028    1.0000    ⋯
                                                                1 column omitted

julia> m4_3.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    1.3526    1.7891    2.0684    2.3960    3.1627
           α   -2.0185   -1.3561   -1.0156   -0.6604   -0.0498
        β[1]   -0.3245   -0.1231   -0.0176    0.0853    0.2786
        β[2]    0.7930    0.9581    1.0475    1.1373    1.3140
        β[3]   -1.1415   -0.4847   -0.1486    0.1667    0.9080
        β[4]    0.6616    1.0669    1.2795    1.4899    1.9206
        β[5]   -0.3974   -0.0588    0.1156    0.3042    0.6689

```


<!-- **Negative Binomial Regression with Cauchy Prior**
```@repl examples
m4_4 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Cauchy())
m4_4.summaries
m4_4.quantiles
``` -->

 **Negative Binomial Regression with TDist Prior**
```jldoctest examples
julia> m4_5 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_TDist());

julia> m4_5.summaries
Summary Statistics
  parameters      mean        std   naive_se      mcse         ess      rhat   ⋯
      Symbol   Float64    Float64    Float64   Float64     Float64   Float64   ⋯

           λ    2.0084     0.4362     0.0044    0.0053   7773.0961    0.9999   ⋯
           ν   24.2506   407.7773     4.0778    6.2329   4167.5690    1.0000   ⋯
           α   -1.0466     0.5191     0.0052    0.0082   3895.6805    1.0000   ⋯
        β[1]   -0.0137     0.1621     0.0016    0.0022   5415.7432    0.9999   ⋯
        β[2]    1.0575     0.1339     0.0013    0.0017   5411.7782    0.9999   ⋯
        β[3]   -0.1591     0.5468     0.0055    0.0064   8569.9356    0.9999   ⋯
        β[4]    1.2786     0.3272     0.0033    0.0037   7649.1266    0.9999   ⋯
        β[5]    0.1393     0.2835     0.0028    0.0039   6478.0610    0.9999   ⋯
                                                                1 column omitted

julia> m4_5.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    1.3012    1.6984    1.9586    2.2643    3.0070
           ν    0.6875    1.9456    3.9010    9.3369   97.3373
           α   -2.0729   -1.3939   -1.0505   -0.6923   -0.0339
        β[1]   -0.3353   -0.1196   -0.0129    0.0972    0.2977
        β[2]    0.8006    0.9660    1.0557    1.1477    1.3255
        β[3]   -1.1903   -0.5278   -0.1792    0.1962    0.9472
        β[4]    0.6391    1.0602    1.2775    1.4927    1.9255
        β[5]   -0.4201   -0.0512    0.1391    0.3308    0.6965

```

 **Negative Binomial Regression with Uniform Prior**
```jldoctest examples
julia> m4_6 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Uniform(),1.0);

julia> m4_6.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   e ⋯
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64     ⋯

           λ    2.6899    0.5663     0.0057    0.0173   354.9071    1.0011     ⋯
           α   -1.0886    0.0000     0.0000    0.0000    20.5530    0.9999     ⋯
        β[1]    0.0017    0.0610     0.0006    0.0038    71.2556    1.0084     ⋯
        β[2]    1.0886    0.0000     0.0000    0.0000    20.5530    0.9999     ⋯
        β[3]   -0.5521    0.9382     0.0094    0.0911    24.5058    1.0292     ⋯
        β[4]    1.0886    0.0000     0.0000    0.0000    20.5530    0.9999     ⋯
        β[5]    0.1461    0.3154     0.0032    0.0230    69.7307    1.0061     ⋯
                                                                1 column omitted

julia> m4_6.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           λ    1.7611    2.2807    2.6335    3.0232    3.9710
           α   -1.0886   -1.0886   -1.0886   -1.0886   -1.0886
        β[1]   -0.1395   -0.0329    0.0071    0.0429    0.1071
        β[2]    1.0886    1.0886    1.0886    1.0886    1.0886
        β[3]   -1.0886   -1.0886   -1.0886   -1.0886    1.0886
        β[4]    1.0886    1.0886    1.0886    1.0886    1.0886
        β[5]   -0.3274   -0.0455    0.1012    0.2621    1.0886

```