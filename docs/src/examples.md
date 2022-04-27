## Example 1: Linear Regression

```jldoctest examples
julia> using RDatasets, NLSolversBase, CRRao, Logging; 

julia> Logging.disable_logging(Logging.Warn); CRRao.setprogress!(false); 

julia> CRRao_seed(123); 

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

julia> m1_1.residuals
32-element Vector{Float64}:
 -2.6688499523387037
 -1.8534082432063457
 -2.453556140740865
  0.6538282376886713
  1.06442945616989
 -2.046638453886395
 -0.3448310401665502
  0.7881712764862279
  0.27419879500614286
 -1.3684264750049024
  ⋮
 -0.481422171673465
  2.8595427589095195
 -0.17793682112103681
 -0.9227150395749106
  2.281550994801261
 -1.4649819082486246
 -2.1180653993796668
  1.6259525228014091
 -1.7939863113843444

julia> m1_1.Cooks_distance
32-element Vector{Float64}:
 0.013342034282302684
 0.00688728266731234
 0.015495847517058797
 0.0014309089637600369
 0.004471979213924145
 0.014588985833724696
 0.0015401004198812064
 0.005826402580870707
 0.0003074315682457445
 0.007011803724485943
 ⋮
 0.0020768256096929424
 0.02203970419212919
 0.0001378106083284689
 0.006862929526075293
 0.047038899451778936
 0.0381204513180911
 0.03540469459036285
 0.13715341355042346
 0.006145660329519638
```

 **Linear Regression - Ridge Prior**

```jldoctest examples
julia> m1_2 = @fitmodel((MPG ~ HP + WT+Gear), df, LinearRegression(), Prior_Ridge()); 

julia> m1_2.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           v    6.7959    3.8426     0.0384    0.0656   3504.6565    1.0000    ⋯
           σ    2.6845    0.3933     0.0039    0.0066   3519.8321    0.9999    ⋯
           α   28.3443    5.3753     0.0538    0.0993   2412.5004    0.9999    ⋯
        β[1]   -0.0400    0.0105     0.0001    0.0002   3626.7198    0.9999    ⋯
        β[2]   -2.6492    0.9481     0.0095    0.0176   2590.8328    0.9999    ⋯
        β[3]    1.6538    0.9761     0.0098    0.0175   2629.2168    0.9999    ⋯
                                                                1 column omitted

julia> m1_2.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    2.4776    4.3942    5.9531    8.0658   16.4623
           σ    2.0531    2.4095    2.6403    2.9047    3.5856
           α   16.9646   25.0338   28.5895   31.8601   38.3976
        β[1]   -0.0608   -0.0470   -0.0399   -0.0330   -0.0195
        β[2]   -4.4404   -3.2875   -2.6778   -2.0376   -0.6931
        β[3]   -0.1610    1.0006    1.6097    2.2526    3.7245
```


 **Linear Regression - Laplace Prior**

```jldoctest examples
julia> m1_3 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Laplace()); 

julia> m1_3.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           v    4.3681    3.5844     0.0358    0.0509   3935.6187    0.9999    ⋯
           σ    2.6666    0.3809     0.0038    0.0059   4034.9336    1.0000    ⋯
           α   29.1630    5.1674     0.0517    0.0805   3118.7281    1.0000    ⋯
        β[1]   -0.0398    0.0105     0.0001    0.0002   4170.8923    0.9999    ⋯
        β[2]   -2.7284    0.9316     0.0093    0.0151   3161.8637    1.0000    ⋯
        β[3]    1.4945    0.9379     0.0094    0.0145   3299.8195    0.9999    ⋯
                                                                1 column omitted

julia> m1_3.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           v    1.2592    2.4031    3.4896    5.2107   12.5745
           σ    2.0453    2.3982    2.6273    2.8867    3.5336
           α   18.7995   25.8276   29.3558   32.6159   38.8971
        β[1]   -0.0605   -0.0468   -0.0398   -0.0328   -0.0190
        β[2]   -4.5024   -3.3491   -2.7417   -2.1278   -0.8679
        β[3]   -0.2803    0.8657    1.4662    2.1077    3.3809

```

 **Linear Regression - Cauchy Prior**
```jldoctest examples
julia> m1_4 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Cauchy(),20000); 

julia> m1_4.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           σ    2.5858    0.3486     0.0025    0.0039   9681.3301    1.0001    ⋯
           α   30.3606    4.6081     0.0326    0.0601   5086.7259    1.0000    ⋯
        β[1]   -0.0395    0.0099     0.0001    0.0001   6944.3903    1.0000    ⋯
        β[2]   -2.8396    0.8538     0.0060    0.0109   5127.7097    1.0000    ⋯
        β[3]    1.2596    0.8380     0.0059    0.0107   5509.6327    1.0000    ⋯
                                                                1 column omitted

julia> m1_4.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           σ    2.0142    2.3370    2.5468    2.7920    3.3841
           α   21.0188   27.3578   30.4396   33.4542   39.2363
        β[1]   -0.0589   -0.0460   -0.0394   -0.0329   -0.0200
        β[2]   -4.5079   -3.4098   -2.8391   -2.2857   -1.1349
        β[3]   -0.3206    0.7001    1.2473    1.7952    2.9440

```

 **Linear Regression - T-Distributed Prior**

```jldoctest examples
julia> m1_5 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_TDist()); 

julia> m1_5.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           ν    1.0435    0.5361     0.0054    0.0067   6299.0784    0.9999    ⋯
           σ    2.6215    0.3595     0.0036    0.0056   4014.6383    0.9999    ⋯
           α   30.2491    4.6744     0.0467    0.0714   3829.7770    0.9999    ⋯
        β[1]   -0.0395    0.0101     0.0001    0.0001   4682.2786    1.0000    ⋯
        β[2]   -2.8195    0.8609     0.0086    0.0138   3756.4596    0.9999    ⋯
        β[3]    1.2744    0.8481     0.0085    0.0126   4089.8676    0.9999    ⋯
                                                                1 column omitted

julia> m1_5.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           ν    0.3843    0.6722    0.9193    1.2795    2.4194
           σ    2.0277    2.3677    2.5766    2.8377    3.4500
           α   20.8821   27.1538   30.2773   33.4444   39.4333
        β[1]   -0.0595   -0.0460   -0.0395   -0.0328   -0.0198
        β[2]   -4.4847   -3.3954   -2.8202   -2.2444   -1.0906
        β[3]   -0.3638    0.7009    1.2560    1.8346    2.9985

```

 **Linear Regression - Uniform Prior**
```jldoctest examples
julia> m1_6 = @fitmodel((MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_TDist()); 

julia> m1_6.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat    ⋯
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64    ⋯

           ν    1.0435    0.5361     0.0054    0.0067   6299.0784    0.9999    ⋯
           σ    2.6215    0.3595     0.0036    0.0056   4014.6383    0.9999    ⋯
           α   30.2491    4.6744     0.0467    0.0714   3829.7770    0.9999    ⋯
        β[1]   -0.0395    0.0101     0.0001    0.0001   4682.2786    1.0000    ⋯
        β[2]   -2.8195    0.8609     0.0086    0.0138   3756.4596    0.9999    ⋯
        β[3]    1.2744    0.8481     0.0085    0.0126   4089.8676    0.9999    ⋯
                                                                1 column omitted

julia> m1_6.quantiles
Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%
      Symbol   Float64   Float64   Float64   Float64   Float64

           ν    0.3843    0.6722    0.9193    1.2795    2.4194
           σ    2.0277    2.3677    2.5766    2.8377    3.4500
           α   20.8821   27.1538   30.2773   33.4444   39.4333
        β[1]   -0.0595   -0.0460   -0.0395   -0.0328   -0.0198
        β[2]   -4.4847   -3.3954   -2.8202   -2.2444   -1.0906
        β[3]   -0.3638    0.7009    1.2560    1.8346    2.9985

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
2064.694633749211


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
2078.9464617505087
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

sanction = dataset("Zelig", "sanction")

m3_1 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression()); 

m3_1.fit

m3_1.LogLike

m3_1.AIC

m3_1.BIC

```

 **Poisson Regression with Ridge Prior**
```jldoctest examples

m3_2 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Ridge()) 

m3_2.summaries

m3_2.quantiles

```

 **Poisson Regression with Laplace Prior**
```jldoctest examples

m3_3 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Laplace()); 

m3_3.summaries

m3_3.quantiles

```

 **Poisson Regression with Cauchy Prior**
```jldoctest examples
m3_4 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Cauchy());

m3_4.summaries

m3_4.quantiles

```

 **Poisson Regression with TDist Prior**
```jldoctest examples
m3_5 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_TDist()); 

m3_5.summaries

m3_5.quantiles

```

 **Poisson Regression with Uniform Prior**
```jldoctest examples
m3_6 = @fitmodel((Num ~ Target + Coop + NCost), sanction,PoissonRegression(),Prior_Uniform()); 

m3_6.summaries

m3_6.quantiles

```
## Example 4: Negative Binomial Regression

**Negative Binomial Regression - Likelihood method** 
```jldoctest examples
m4_1 = @fitmodel((Num ~ Target + Coop + NCost), sanction, NegBinomRegression()); 

m4_1.fit

m4_1.AIC

m4_1.BIC

m4_1.lambda_hat
```

 **NegativeBinomial Regression with Ridge Prior**
```jldoctest examples

m4_2 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Ridge()); 

m4_2.summaries

m4_2.quantiles

```

 **NegativeBinomial Regression with Laplace Prior**
```jldoctest examples
m4_3 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Laplace()); 

m4_3.summaries

m4_3.quantiles

```


 **Negative Binomial Regression with Cauchy Prior**
```jldoctest examples
m4_4 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Cauchy()); 

m4_4.summaries

m4_4.quantiles

```

 **Negative Binomial Regression with TDist Prior**
```jldoctest examples
m4_5 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_TDist()); 

m4_5.summaries

m4_5.quantiles

```

 **Negative Binomial Regression with Uniform Prior**
```jldoctest examples
m4_6 = @fitmodel((Num ~ Target + Coop + NCost), sanction,NegBinomRegression(),Prior_Uniform(),1.0); 

m4_6.summaries

m4_6.quantiles

```


