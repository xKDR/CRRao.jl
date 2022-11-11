function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::CRRaoLink, turingModel::Function, sim_size::Int64)
    formula = apply_schema(formula, schema(formula, data),RegressionModel)
    y, X = modelcols(formula, data)

    if sim_size < 500
        @warn "Simulation size should generally be atleast 500."
    end
    chain = sample(CRRao_rng, turingModel(X, y), NUTS(), sim_size)
    return BayesianRegression(:LogisticRegression, chain, formula, Link)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Ridge, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```

Fit a Bayesian Logistic Regression model on the input data with a Ridge prior with the provided `Link` function.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> turnout = dataset("Zelig", "turnout")
2000×5 DataFrame
  Row │ Race   Age    Educate  Income   Vote  
      │ Cat…   Int32  Float64  Float64  Int32 
──────┼───────────────────────────────────────
    1 │ white     60     14.0   3.3458      1
    2 │ white     51     10.0   1.8561      0
    3 │ white     24     12.0   0.6304      0
    4 │ white     38      8.0   3.4183      1
  ⋮   │   ⋮      ⋮       ⋮        ⋮       ⋮
 1998 │ white     51     16.0   7.8949      1
 1999 │ white     22     10.0   2.4811      0
 2000 │ white     59     10.0   0.5523      0

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> container_logit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Logit(), Prior_Ridge())
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 46.7 seconds
Compute duration  = 46.7 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    1.4907    0.5918     0.0187    0.0236   593.6860    0.9990       12.7136
        β[1]   -2.8684    0.3406     0.0108    0.0151   502.0358    1.0001       10.7509
        β[2]    0.0271    0.0036     0.0001    0.0001   617.1201    1.0001       13.2154
        β[3]    0.2266    0.1434     0.0045    0.0047   830.1625    0.9996       17.7776
        β[4]    0.1793    0.0273     0.0009    0.0010   830.0115    1.0018       17.7744
        β[5]    0.1677    0.0203     0.0006    0.0009   530.2735    1.0002       11.3556

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.7846    1.0730    1.3581    1.7063    3.0609
        β[1]   -3.5602   -3.1138   -2.8687   -2.6160   -2.2454
        β[2]    0.0202    0.0247    0.0270    0.0294    0.0347
        β[3]   -0.0541    0.1323    0.2315    0.3264    0.4980
        β[4]    0.1281    0.1597    0.1783    0.1989    0.2336
        β[5]    0.1276    0.1548    0.1675    0.1804    0.2104

julia> CRRao.set_rng(StableRNG(123))

julia> container_probit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Probit(), Prior_Ridge())
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 35.35 seconds
Compute duration  = 35.35 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    0.9120    0.3882     0.0123    0.0137   592.9548    1.0005       16.7719
        β[1]   -1.6775    0.1843     0.0058    0.0094   452.5107    0.9996       12.7994
        β[2]    0.0159    0.0020     0.0001    0.0001   699.2797    0.9996       19.7794
        β[3]    0.1473    0.0868     0.0027    0.0032   609.4328    0.9990       17.2380
        β[4]    0.0966    0.0153     0.0005    0.0006   556.8599    0.9991       15.7510
        β[5]    0.1004    0.0119     0.0004    0.0006   408.9705    0.9993       11.5679

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.4408    0.6582    0.8190    1.0636    1.8731
        β[1]   -2.0316   -1.8051   -1.6756   -1.5634   -1.2959
        β[2]    0.0121    0.0145    0.0159    0.0172    0.0201
        β[3]   -0.0286    0.0907    0.1496    0.2102    0.3104
        β[4]    0.0665    0.0863    0.0969    0.1070    0.1269
        β[5]    0.0771    0.0919    0.1004    0.1084    0.1238

julia> CRRao.set_rng(StableRNG(123))
julia> container_cloglog = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Cloglog(), Prior_Ridge())
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 44.56 seconds
Compute duration  = 44.56 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    1.0207    0.4636     0.0147    0.0250   250.5938    1.0007        5.6241
        β[1]   -1.8739    0.1944     0.0061    0.0082   486.4861    0.9992       10.9183
        β[2]    0.0143    0.0018     0.0001    0.0001   693.4025    0.9990       15.5621
        β[3]    0.1715    0.0822     0.0026    0.0029   668.5177    0.9993       15.0037
        β[4]    0.0775    0.0114     0.0004    0.0003   767.3234    0.9991       17.2212
        β[5]    0.0950    0.0110     0.0003    0.0004   523.2168    0.9999       11.7426

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.5094    0.7421    0.9210    1.1688    2.1113
        β[1]   -2.2663   -1.9963   -1.8787   -1.7465   -1.5043
        β[2]    0.0106    0.0131    0.0143    0.0155    0.0180
        β[3]    0.0140    0.1159    0.1730    0.2232    0.3280
        β[4]    0.0562    0.0700    0.0777    0.0849    0.1009
        β[5]    0.0728    0.0879    0.0948    0.1019    0.1164
        
julia> CRRao.set_rng(StableRNG(123))
julia> container_cauchit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Cauchit(), Prior_Ridge())
┌ Info: Found initial step size
└   ϵ = 0.025
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 32.18 seconds
Compute duration  = 32.18 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    1.6145    0.8633     0.0273    0.0605   263.9458    1.0083        8.2032
        β[1]   -2.9672    0.4041     0.0128    0.0194   425.5258    0.9996       13.2249
        β[2]    0.0287    0.0043     0.0001    0.0002   514.5857    1.0017       15.9928
        β[3]    0.1590    0.1530     0.0048    0.0072   635.2277    0.9990       19.7423
        β[4]    0.2406    0.0399     0.0013    0.0014   654.7789    0.9992       20.3499
        β[5]    0.1590    0.0248     0.0008    0.0011   470.3448    0.9992       14.6179

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.7688    1.1244    1.3980    1.8607    3.5488
        β[1]   -3.7871   -3.2405   -2.9510   -2.6879   -2.2261
        β[2]    0.0201    0.0256    0.0285    0.0315    0.0374
        β[3]   -0.1352    0.0549    0.1552    0.2700    0.4495
        β[4]    0.1679    0.2128    0.2384    0.2667    0.3222
        β[5]    0.1142    0.1420    0.1580    0.1754    0.2121

```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::CRRaoLink,
    prior::Prior_Ridge,
    h::Float64 = 0.1,
    level::Float64 = 0.95,
    sim_size::Int64 = 1000
)
    @model LogisticRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        #α ~ Normal(0, λ)
        β ~ filldist(Normal(0, λ), p)

        #z = α .+ X * β
        z =  X * β

        ## Link Function

        prob = Link.link_function.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, Link, LogisticRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Laplace, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```

Fit a Bayesian Logistic Regression model on the input data with a Laplace prior with the provided `Link` function.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
julia> turnout = dataset("Zelig", "turnout")
2000×5 DataFrame
  Row │ Race   Age    Educate  Income   Vote  
      │ Cat…   Int32  Float64  Float64  Int32 
──────┼───────────────────────────────────────
    1 │ white     60     14.0   3.3458      1
    2 │ white     51     10.0   1.8561      0
    3 │ white     24     12.0   0.6304      0
    4 │ white     38      8.0   3.4183      1
  ⋮   │   ⋮      ⋮       ⋮        ⋮       ⋮
 1998 │ white     51     16.0   7.8949      1
 1999 │ white     22     10.0   2.4811      0
 2000 │ white     59     10.0   0.5523      0

julia> container_logit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Logit(), Prior_Laplace())
┌ Info: Found initial step size
└   ϵ = 0.0015625
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 47.58 seconds
Compute duration  = 47.58 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

          λ    0.8459    0.4712     0.0149    0.0148   821.7483    0.9990       17.2709
        β[1]   -2.8796    0.3230     0.0102    0.0168   358.1394    1.0043        7.5271
        β[2]    0.0273    0.0032     0.0001    0.0001   629.0124    0.9999       13.2201
        β[3]    0.2138    0.1418     0.0045    0.0049   697.8744    0.9995       14.6674
        β[4]    0.1774    0.0264     0.0008    0.0009   779.0814    0.9998       16.3741
        β[5]    0.1692    0.0201     0.0006    0.0010   363.7691    1.0071        7.6454

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

          λ    0.3275    0.5375    0.7244    1.0189    2.1072
        β[1]   -3.5527   -3.0956   -2.8824   -2.6539   -2.2616
        β[2]    0.0212    0.0250    0.0273    0.0294    0.0336
        β[3]   -0.0401    0.1143    0.2130    0.3094    0.4914
        β[4]    0.1283    0.1597    0.1768    0.1953    0.2282
        β[5]    0.1282    0.1562    0.1693    0.1828    0.2087
                             

julia> container_probit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Probit(), Prior_Laplace())
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 44.42 seconds
Compute duration  = 44.42 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    0.5137    0.2746     0.0087    0.0090   654.5174    0.9991       14.7357
        β[1]   -1.6793    0.2013     0.0064    0.0086   640.0050    0.9991       14.4090
        β[2]    0.0160    0.0020     0.0001    0.0001   848.1151    0.9990       19.0944
        β[3]    0.1451    0.0878     0.0028    0.0037   804.2314    1.0003       18.1064
        β[4]    0.0963    0.0143     0.0005    0.0005   750.7575    1.0029       16.9025
        β[5]    0.1005    0.0118     0.0004    0.0004   696.8244    0.9991       15.6882

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.1955    0.3289    0.4496    0.6183    1.1915
        β[1]   -2.0608   -1.8105   -1.6898   -1.5374   -1.2636
        β[2]    0.0123    0.0147    0.0160    0.0174    0.0199
        β[3]   -0.0209    0.0832    0.1466    0.2063    0.3211
        β[4]    0.0686    0.0869    0.0964    0.1054    0.1258
        β[5]    0.0772    0.0925    0.1004    0.1091    0.1219

julia> CRRao.set_rng(StableRNG(123))        
julia> container_cloglog = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Cloglog(), Prior_Laplace())
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 45.03 seconds
Compute duration  = 45.03 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    0.5670    0.3381     0.0107    0.0133   643.3993    0.9995       14.2882
        β[1]   -1.8734    0.1839     0.0058    0.0072   622.0146    0.9990       13.8133
        β[2]    0.0143    0.0017     0.0001    0.0001   869.5863    0.9992       19.3113
        β[3]    0.1671    0.0802     0.0025    0.0028   580.9503    0.9997       12.9014
        β[4]    0.0772    0.0114     0.0004    0.0004   854.8669    0.9993       18.9844
        β[5]    0.0955    0.0109     0.0003    0.0004   758.8162    0.9993       16.8513

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.2079    0.3661    0.4862    0.6688    1.3563
        β[1]   -2.2418   -1.9906   -1.8718   -1.7539   -1.5159
        β[2]    0.0109    0.0131    0.0143    0.0154    0.0176
        β[3]    0.0194    0.1086    0.1660    0.2231    0.3207
        β[4]    0.0549    0.0696    0.0772    0.0849    0.0999
        β[5]    0.0738    0.0882    0.0959    0.1031    0.1151

julia> container_cauchit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Cauchit(), Prior_Laplace())
┌ Info: Found initial step size
└   ϵ = 0.00078125
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 42.23 seconds
Compute duration  = 42.23 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

            λ    0.8480    0.4143     0.0131    0.0144   675.5220    0.9998       15.9970
        β[1]   -3.0014    0.3675     0.0116    0.0179   512.9194    0.9995       12.1464
        β[2]    0.0291    0.0042     0.0001    0.0002   668.6412    0.9994       15.8341
        β[3]    0.1403    0.1468     0.0046    0.0059   732.6106    0.9993       17.3489
        β[4]    0.2394    0.0383     0.0012    0.0019   517.9500    1.0015       12.2656
        β[5]    0.1622    0.0233     0.0007    0.0011   552.8157    0.9999       13.0912

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

            λ    0.3261    0.5543    0.7534    1.0416    1.8282
        β[1]   -3.7419   -3.2460   -2.9995   -2.7679   -2.2990
        β[2]    0.0210    0.0260    0.0293    0.0319    0.0380
        β[3]   -0.1352    0.0447    0.1375    0.2297    0.4357
        β[4]    0.1680    0.2130    0.2378    0.2663    0.3196
        β[5]    0.1178    0.1461    0.1615    0.1769    0.2070     
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::CRRaoLink,
    prior::Prior_Laplace,
    h::Float64 = 0.1,
    level::Float64 = 0.95,
    sim_size::Int64 = 1000
)
    @model LogisticRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        #α ~ Normal(0, λ)
        β ~ filldist(Laplace(0, λ), p)

        #z = α .+ X * β
        z = X * β

        ## Link Function

        prob = Link.link_function.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, Link, LogisticRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Cauchy, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```

Fit a Bayesian Logistic Regression model on the input data with a Cauchy prior with the provided `Link` function.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
julia> turnout = dataset("Zelig", "turnout")
2000×5 DataFrame
  Row │ Race   Age    Educate  Income   Vote  
      │ Cat…   Int32  Float64  Float64  Int32 
──────┼───────────────────────────────────────
    1 │ white     60     14.0   3.3458      1
    2 │ white     51     10.0   1.8561      0
    3 │ white     24     12.0   0.6304      0
    4 │ white     38      8.0   3.4183      1
  ⋮   │   ⋮      ⋮       ⋮        ⋮       ⋮
 1998 │ white     51     16.0   7.8949      1
 1999 │ white     22     10.0   2.4811      0
 2000 │ white     59     10.0   0.5523      0
                             1993 rows omitted
julia> container_logit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Logit(), Prior_Cauchy())
┌ Info: Found initial step size
└   ϵ = 0.0015625
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 32.88 seconds
Compute duration  = 32.88 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    0.3048    0.2245     0.0071    0.0092   732.5949    0.9994       22.2829
        β[1]   -2.9536    0.3307     0.0105    0.0151   501.8962    1.0078       15.2659
        β[2]    0.0282    0.0035     0.0001    0.0001   814.6017    1.0013       24.7773
        β[3]    0.1818    0.1379     0.0044    0.0049   641.2468    1.0105       19.5044
        β[4]    0.1781    0.0279     0.0009    0.0009   873.2256    0.9992       26.5604
        β[5]    0.1738    0.0201     0.0006    0.0008   612.2187    1.0022       18.6215

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0701    0.1564    0.2435    0.3783    0.9120
        β[1]   -3.5915   -3.1683   -2.9647   -2.7330   -2.2954
        β[2]    0.0214    0.0258    0.0282    0.0307    0.0350
        β[3]   -0.0673    0.0823    0.1762    0.2728    0.4755
        β[4]    0.1237    0.1595    0.1787    0.1962    0.2341
        β[5]    0.1358    0.1608    0.1735    0.1866    0.2164

julia> container_probit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Probit(), Prior_Cauchy())
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 92.1 seconds
Compute duration  = 92.1 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.1889    0.1676     0.0053    0.0102    213.5688    1.0055        2.3189
        β[1]   -1.6968    0.1900     0.0060    0.0059    668.5132    0.9997        7.2586
        β[2]    0.0164    0.0020     0.0001    0.0001    844.5946    0.9994        9.1704
        β[3]    0.1155    0.0820     0.0026    0.0050    351.2848    1.0087        3.8142
        β[4]    0.0955    0.0151     0.0005    0.0004   1072.1826    0.9992       11.6415
        β[5]    0.1029    0.0117     0.0004    0.0004    781.5659    0.9991        8.4861

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0368    0.0847    0.1335    0.2284    0.6325
        β[1]   -2.0732   -1.8312   -1.6971   -1.5656   -1.3258
        β[2]    0.0124    0.0150    0.0163    0.0177    0.0203
        β[3]   -0.0366    0.0566    0.1128    0.1706    0.2800
        β[4]    0.0657    0.0852    0.0958    0.1058    0.1245
        β[5]    0.0798    0.0949    0.1028    0.1111    0.1251
julia> container_cloglog = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Cloglog(), Prior_Cauchy())
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 41.19 seconds
Compute duration  = 41.19 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    0.1839    0.2072     0.0066    0.0085   681.7783    0.9991       16.5520
        β[1]   -1.8712    0.1883     0.0060    0.0068   568.8922    0.9993       13.8114
        β[2]    0.0146    0.0019     0.0001    0.0001   866.0645    0.9990       21.0261
        β[3]    0.1383    0.0857     0.0027    0.0037   500.3015    1.0037       12.1462
        β[4]    0.0765    0.0117     0.0004    0.0004   955.1365    0.9991       23.1886
        β[5]    0.0965    0.0114     0.0004    0.0004   612.5153    1.0007       14.8705

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0346    0.0889    0.1368    0.2232    0.5814
        β[1]   -2.2534   -2.0036   -1.8581   -1.7428   -1.5302
        β[2]    0.0112    0.0133    0.0145    0.0159    0.0184
        β[3]   -0.0102    0.0751    0.1357    0.1937    0.3198
        β[4]    0.0538    0.0689    0.0768    0.0842    0.1003
        β[5]    0.0750    0.0883    0.0960    0.1042    0.1206

julia> container_cauchit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Cauchit(), Prior_Cauchy())
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 26.07 seconds
Compute duration  = 26.07 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    0.3142    0.2296     0.0073    0.0065   775.1497    1.0000       29.7334
        β[1]   -3.0635    0.3940     0.0125    0.0142   616.9777    1.0020       23.6662
        β[2]    0.0299    0.0045     0.0001    0.0001   758.2408    1.0016       29.0848
        β[3]    0.1211    0.1321     0.0042    0.0050   551.1313    0.9992       21.1404
        β[4]    0.2345    0.0408     0.0013    0.0014   670.1954    1.0003       25.7075
        β[5]    0.1671    0.0249     0.0008    0.0009   631.3726    1.0014       24.2184

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0654    0.1638    0.2555    0.3933    0.9454
        β[1]   -3.8138   -3.3366   -3.0547   -2.7857   -2.3285
        β[2]    0.0211    0.0270    0.0301    0.0329    0.0384
        β[3]   -0.1105    0.0282    0.1117    0.2168    0.3907
        β[4]    0.1603    0.2064    0.2324    0.2627    0.3157
        β[5]    0.1166    0.1501    0.1669    0.1839    0.2152
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::CRRaoLink,
    prior::Prior_Cauchy,
    h::Float64 = 0.1,
    level::Float64 = 0.95,
    sim_size::Int64 = 1000
)
    @model LogisticRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ Truncated(TDist(1), 0, Inf)
        #α ~ TDist(1) * λ
        β ~ filldist(TDist(1) * λ, p)

        #z = α .+ X * β
        z = X * β

        ## Link Function

        prob = Link.link_function.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, Link, LogisticRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_TDist, h::Float64 = 1.0, level::Float64 = 0.95, sim_size::Int64 = 1000)
```

Fit a Bayesian Logistic Regression model on the input data with a T-Dist prior with the provided `Link` function.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
julia> turnout = dataset("Zelig", "turnout")
2000×5 DataFrame
  Row │ Race   Age    Educate  Income   Vote  
      │ Cat…   Int32  Float64  Float64  Int32 
──────┼───────────────────────────────────────
    1 │ white     60     14.0   3.3458      1
    2 │ white     51     10.0   1.8561      0
    3 │ white     24     12.0   0.6304      0
    4 │ white     38      8.0   3.4183      1
  ⋮   │   ⋮      ⋮       ⋮        ⋮       ⋮
 1998 │ white     51     16.0   7.8949      1
 1999 │ white     22     10.0   2.4811      0
 2000 │ white     59     10.0   0.5523      0
                             1993 rows omitted
julia> CRRao.set_rng(StableRNG(7740));
StableRNGs.LehmerRNG(state=0x00000000000000000000000000003c79)
julia> container_logit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Logit(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 0.003125
┌ Warning: The current proposal will be rejected due to numerical error(s).
│   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
└ @ AdvancedHMC ~/.julia/packages/AdvancedHMC/iWHPQ/src/hamiltonian.jl:47
Chains MCMC chain (1000×19×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 39.94 seconds
Compute duration  = 39.94 seconds
parameters        = λ, ν, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    0.7011    0.3630     0.0115    0.0144   545.8105    0.9991       13.6671
           ν    1.4459    0.9166     0.0290    0.0334   691.7171    1.0066       17.3206
        β[1]   -2.9624    0.3175     0.0100    0.0120   727.4172    0.9999       18.2146
        β[2]    0.0279    0.0034     0.0001    0.0001   939.5146    0.9993       23.5255
        β[3]    0.2331    0.1479     0.0047    0.0046   742.9274    0.9992       18.6029
        β[4]    0.1750    0.0259     0.0008    0.0009   695.4691    0.9994       17.4146
        β[5]    0.1729    0.0197     0.0006    0.0008   697.3021    0.9994       17.4605

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.2980    0.4622    0.6229    0.8441    1.6138
           ν    0.4603    0.8479    1.2170    1.7611    3.9754
        β[1]   -3.5783   -3.1642   -2.9738   -2.7651   -2.3367
        β[2]    0.0210    0.0257    0.0279    0.0302    0.0347
        β[3]   -0.0383    0.1349    0.2251    0.3318    0.5290
        β[4]    0.1262    0.1577    0.1741    0.1920    0.2272
        β[5]    0.1349    0.1604    0.1723    0.1861    0.2119

julia> CRRao.set_rng(StableRNG(7740))
julia> container_probit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Probit(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 0.00078125
Chains MCMC chain (1000×19×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 47.56 seconds
Compute duration  = 47.56 seconds
parameters        = λ, ν, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    0.6284    0.2643     0.0084    0.0085   759.3697    0.9997       15.9656
           ν    1.7113    1.3044     0.0412    0.0491   715.9368    0.9996       15.0524
        β[1]   -1.7098    0.1909     0.0060    0.0064   609.6431    0.9994       12.8176
        β[2]    0.0162    0.0020     0.0001    0.0001   943.5295    0.9992       19.8375
        β[3]    0.1525    0.0878     0.0028    0.0026   945.7419    0.9990       19.8840
        β[4]    0.0966    0.0145     0.0005    0.0005   942.0548    0.9990       19.8065
        β[5]    0.1017    0.0116     0.0004    0.0004   580.8348    0.9993       12.2119

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.2854    0.4398    0.5684    0.7640    1.2603
           ν    0.4877    0.9461    1.3453    2.0327    5.1304
        β[1]   -2.0824   -1.8335   -1.7198   -1.5816   -1.3325
        β[2]    0.0120    0.0148    0.0162    0.0176    0.0198
        β[3]   -0.0282    0.0960    0.1541    0.2147    0.3217
        β[4]    0.0687    0.0869    0.0970    0.1063    0.1240
        β[5]    0.0787    0.0937    0.1015    0.1095    0.1241

julia> CRRao.set_rng(StableRNG(7740))
julia> container_cloglog = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Cloglog(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 0.0015625
Chains MCMC chain (10000×19×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 666.27 seconds
Compute duration  = 666.27 seconds
parameters        = λ, ν, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse          ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64      Float64   Float64       Float64 

           λ    0.6312    0.2782     0.0028    0.0040    4453.1117    0.9999        6.6837
           ν    1.6104    1.3172     0.0132    0.0241    2233.9301    1.0000        3.3529
        β[1]   -1.9118    0.1841     0.0018    0.0019    8957.3062    0.9999       13.4440
        β[2]    0.0146    0.0018     0.0000    0.0000    9811.4797    0.9999       14.7260
        β[3]    0.1736    0.0851     0.0009    0.0013    3552.7097    0.9999        5.3323
        β[4]    0.0769    0.0119     0.0001    0.0001   11861.3612    1.0002       17.8027
        β[5]    0.0971    0.0109     0.0001    0.0001   11028.1160    1.0002       16.5521

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.2858    0.4424    0.5693    0.7569    1.3451
           ν    0.5054    0.8957    1.2752    1.8894    4.6403
        β[1]   -2.2751   -2.0344   -1.9134   -1.7884   -1.5464
        β[2]    0.0110    0.0134    0.0146    0.0158    0.0182
        β[3]    0.0072    0.1164    0.1742    0.2325    0.3366
        β[4]    0.0534    0.0689    0.0768    0.0849    0.1003
        β[5]    0.0760    0.0897    0.0970    0.1044    0.1187

julia> CRRao.set_rng(StableRNG(7740))
StableRNGs.LehmerRNG(state=0x00000000000000000000000000003c79)
        
julia> container_cauchit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Cauchit(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 0.8
Chains MCMC chain (1000×19×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 68.68 seconds
Compute duration  = 68.68 seconds
parameters        = λ, ν, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    0.6907    0.3474     0.0110    0.0126   712.4682    0.9990       10.3736
           ν    1.5930    1.7268     0.0546    0.1194   179.2215    0.9991        2.6095
        β[1]   -3.1058    0.3806     0.0120    0.0152   645.0360    0.9992        9.3918
        β[2]    0.0301    0.0043     0.0001    0.0002   824.3969    0.9991       12.0033
        β[3]    0.1578    0.1461     0.0046    0.0045   903.3324    1.0037       13.1526
        β[4]    0.2375    0.0421     0.0013    0.0019   471.7863    1.0106        6.8692
        β[5]    0.1668    0.0235     0.0007    0.0010   645.8576    0.9990        9.4037

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.2894    0.4587    0.6085    0.8172    1.5353
           ν    0.4751    0.8676    1.2220    1.7524    5.1716
        β[1]   -3.8599   -3.3612   -3.1010   -2.8637   -2.3669
        β[2]    0.0218    0.0271    0.0300    0.0330    0.0388
        β[3]   -0.1336    0.0643    0.1646    0.2552    0.4311
        β[4]    0.1592    0.2092    0.2368    0.2650    0.3271
        β[5]    0.1218    0.1508    0.1661    0.1819    0.2157
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::CRRaoLink,
    prior::Prior_TDist,
    h::Float64 = 3.0,
    level::Float64 = 0.95,
    sim_size::Int64 = 1000
)
    @model LogisticRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        ν ~ InverseGamma(h, h)
        #α ~ TDist(ν) * λ
        β ~ filldist(TDist(ν) * λ, p)

        #z = α .+ X * β
        z =  X * β

        ## Link Function

        prob = Link.link_function.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, Link, LogisticRegression, sim_size)
end



"""
```julia
fit(formula::FormulaTerm,data::DataFrame,modelClass::LogisticRegression,Link::CRRaoLink,prior::Prior_HorseShoe,level::Float64 = 0.95,sim_size::Int64 = 1000)
```

Fit a Bayesian Logistic Regression model on the input data with a HorseShoe prior with the provided `Link` function.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> turnout = dataset("Zelig", "turnout");
julia> CRRao.set_rng(StableRNG(7740))
julia> container_logit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Logit(), Prior_HorseShoe())
Chains MCMC chain (1000×24×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 78.98 seconds
Compute duration  = 78.98 seconds
parameters        = τ, λ[1], λ[2], λ[3], λ[4], λ[5], σ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           τ    0.9411    1.0672     0.0337    0.0303    817.3843    0.9990       10.3489
        λ[1]   12.1657   20.9749     0.6633    0.6373    839.0119    0.9992       10.6227
        λ[2]    0.6175    1.2944     0.0409    0.0530    564.7242    0.9997        7.1499
        λ[3]    1.0698    1.6054     0.0508    0.0564    755.8639    1.0007        9.5700
        λ[4]    1.2037    1.5510     0.0490    0.0626    601.4558    0.9993        7.6150
        λ[5]    1.4052    3.8216     0.1208    0.2157    309.1836    1.0036        3.9146
           σ    1.0562    1.7528     0.0554    0.0775    458.5741    1.0000        5.8060
        β[1]   -2.9262    0.3572     0.0113    0.0234    199.5082    1.0052        2.5260
        β[2]    0.0280    0.0037     0.0001    0.0002    335.2483    1.0029        4.2446
        β[3]    0.1541    0.1360     0.0043    0.0072    308.0926    1.0027        3.9007
        β[4]    0.1783    0.0259     0.0008    0.0009   1061.7391    0.9994       13.4426
        β[5]    0.1747    0.0217     0.0007    0.0016    150.7165    1.0082        1.9082

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           τ    0.0894    0.3362    0.6462    1.1511    3.5586
        λ[1]    1.1212    3.7091    7.0026   13.2362   59.5366
        λ[2]    0.0271    0.1222    0.2984    0.6426    3.0533
        λ[3]    0.0216    0.2845    0.6136    1.2534    5.1424
        λ[4]    0.1336    0.4165    0.7458    1.3820    5.6334
        λ[5]    0.1210    0.4467    0.8046    1.3866    5.8171
           σ    0.0902    0.3277    0.5955    1.1698    4.3552
        β[1]   -3.5907   -3.1452   -2.9250   -2.6843   -2.1970
        β[2]    0.0209    0.0256    0.0279    0.0305    0.0352
        β[3]   -0.0579    0.0413    0.1396    0.2505    0.4221
        β[4]    0.1287    0.1613    0.1793    0.1962    0.2313
        β[5]    0.1345    0.1600    0.1736    0.1888    0.2162

julia> CRRao.set_rng(StableRNG(7750))
julia> container_probit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Probit(), Prior_HorseShoe())
Chains MCMC chain (1000×24×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 90.76 seconds
Compute duration  = 90.76 seconds
parameters        = τ, λ[1], λ[2], λ[3], λ[4], λ[5], σ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           τ    0.9361    1.4286     0.0452    0.0594    518.1317    0.9991        5.7086
        λ[1]   10.9980   15.4067     0.4872    0.6976    466.4646    1.0033        5.1394
        λ[2]    0.4934    0.7068     0.0223    0.0227    907.8114    1.0021       10.0020
        λ[3]    1.1991    2.0495     0.0648    0.0643    981.7976    1.0035       10.8172
        λ[4]    1.1391    1.4539     0.0460    0.0650    639.5463    0.9997        7.0463
        λ[5]    1.2285    2.2036     0.0697    0.0733    771.8797    1.0034        8.5043
           σ    0.8506    1.3755     0.0435    0.0594    685.4138    1.0002        7.5517
        β[1]   -1.6771    0.1910     0.0060    0.0064    920.9217    0.9993       10.1464
        β[2]    0.0162    0.0020     0.0001    0.0001   1142.6505    0.9990       12.5894
        β[3]    0.1114    0.0865     0.0027    0.0031    718.5753    0.9992        7.9171
        β[4]    0.0959    0.0153     0.0005    0.0004   1132.1662    0.9993       12.4739
        β[5]    0.1021    0.0118     0.0004    0.0004    869.2085    0.9999        9.5767

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           τ    0.0513    0.2551    0.5591    1.0551    4.1911
        λ[1]    1.1021    3.5425    6.6721   13.0485   47.5960
        λ[2]    0.0281    0.1149    0.2683    0.5840    2.3417
        λ[3]    0.0650    0.3625    0.7063    1.3983    4.6179
        λ[4]    0.1094    0.4109    0.7536    1.3183    4.7938
        λ[5]    0.1194    0.4239    0.7649    1.2889    5.0621
           σ    0.0507    0.2487    0.4977    0.9753    3.1844
        β[1]   -2.0502   -1.8040   -1.6787   -1.5443   -1.3105
        β[2]    0.0122    0.0148    0.0162    0.0176    0.0200
        β[3]   -0.0269    0.0436    0.1048    0.1703    0.2917
        β[4]    0.0684    0.0852    0.0959    0.1058    0.1272
        β[5]    0.0782    0.0946    0.1021    0.1099    0.1258

julia> CRRao.set_rng(StableRNG(7750))
julia> container_cloglog = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Cloglog(), Prior_HorseShoe())
Chains MCMC chain (1000×24×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 94.09 seconds
Compute duration  = 94.09 seconds
parameters        = τ, λ[1], λ[2], λ[3], λ[4], λ[5], σ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           τ    1.6055   10.8894     0.3444    0.4607    563.7698    1.0009        5.9916
        λ[1]   12.0709   18.6874     0.5909    0.6065    606.1712    1.0007        6.4422
        λ[2]    0.5384    0.9729     0.0308    0.0282    934.6438    0.9991        9.9331
        λ[3]    1.2516    2.0053     0.0634    0.0708    719.1385    1.0009        7.6428
        λ[4]    0.9365    1.0546     0.0333    0.0375    706.7728    0.9990        7.5113
        λ[5]    1.1632    1.8148     0.0574    0.0621    637.0480    1.0002        6.7703
           σ    0.8427    1.4313     0.0453    0.0634    743.6152    1.0109        7.9029
        β[1]   -1.8752    0.1799     0.0057    0.0067    685.7310    0.9990        7.2877
        β[2]    0.0147    0.0018     0.0001    0.0001   1010.6479    0.9990       10.7408
        β[3]    0.1313    0.0869     0.0027    0.0029    635.0589    0.9996        6.7492
        β[4]    0.0770    0.0116     0.0004    0.0004    803.1570    0.9992        8.5357
        β[5]    0.0969    0.0108     0.0003    0.0004    631.0117    0.9994        6.7062

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           τ    0.0679    0.2685    0.5739    1.0915    8.0842
        λ[1]    1.2178    3.5342    6.6865   13.2847   58.1702
        λ[2]    0.0232    0.1050    0.2616    0.6176    2.7919
        λ[3]    0.0779    0.3932    0.7325    1.3764    5.5664
        λ[4]    0.0972    0.3363    0.6103    1.1374    3.4275
        λ[5]    0.1142    0.3990    0.7305    1.2619    4.7179
           σ    0.0319    0.2244    0.4916    0.9191    4.2356
        β[1]   -2.2334   -1.9932   -1.8766   -1.7558   -1.5183
        β[2]    0.0111    0.0134    0.0147    0.0159    0.0180
        β[3]   -0.0100    0.0650    0.1297    0.1920    0.3038
        β[4]    0.0530    0.0690    0.0767    0.0847    0.0999
        β[5]    0.0765    0.0894    0.0966    0.1039    0.1198

julia> CRRao.set_rng(StableRNG(7750))
julia> container_cauchit = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), Cauchit(), Prior_HorseShoe())
┌ Info: Found initial step size
└   ϵ = 0.8
Chains MCMC chain (1000×24×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 58.31 seconds
Compute duration  = 58.31 seconds
parameters        = τ, λ[1], λ[2], λ[3], λ[4], λ[5], σ, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           τ    1.0708    1.5386     0.0487    0.0643    593.4742    0.9991       10.1776
        λ[1]   15.3511   26.4777     0.8373    1.3810    350.8906    0.9991        6.0175
        λ[2]    0.6148    0.9366     0.0296    0.0477    424.3585    1.0011        7.2774
        λ[3]    0.9767    1.4798     0.0468    0.0740    452.3447    1.0003        7.7573
        λ[4]    1.6912    2.8264     0.0894    0.1299    409.0444    1.0014        7.0148
        λ[5]    1.2917    1.8591     0.0588    0.0643    601.0381    1.0004       10.3073
           σ    1.0216    1.8927     0.0599    0.0726    636.1415    0.9990       10.9093
        β[1]   -3.0273    0.3875     0.0123    0.0199    451.7525    0.9996        7.7472
        β[2]    0.0297    0.0044     0.0001    0.0002    471.3925    1.0005        8.0840
        β[3]    0.1006    0.1311     0.0041    0.0050    639.1553    0.9990       10.9610
        β[4]    0.2376    0.0382     0.0012    0.0012   1088.5251    0.9992       18.6673
        β[5]    0.1653    0.0243     0.0008    0.0012    505.5200    0.9991        8.6692

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           τ    0.0703    0.3114    0.6561    1.2853    4.6674
        λ[1]    1.1598    3.8901    7.8699   15.9998   76.0001
        λ[2]    0.0306    0.1398    0.3299    0.6663    3.1801
        λ[3]    0.0238    0.2307    0.5136    1.1709    4.5934
        λ[4]    0.1611    0.5298    0.9386    1.7458    7.8814
        λ[5]    0.1182    0.4297    0.8235    1.4823    5.0292
           σ    0.0713    0.3226    0.5813    1.1611    4.1744
        β[1]   -3.7724   -3.2934   -3.0226   -2.7394   -2.3257
        β[2]    0.0216    0.0266    0.0297    0.0327    0.0390
        β[3]   -0.1005    0.0049    0.0767    0.1826    0.4019
        β[4]    0.1614    0.2127    0.2382    0.2638    0.3140
        β[5]    0.1201    0.1472    0.1657    0.1804    0.2148

```
"""
function fit(
  formula::FormulaTerm,
  data::DataFrame,
  modelClass::LogisticRegression,
  Link::CRRaoLink,
  prior::Prior_HorseShoe,
  level::Float64 = 0.95,
  sim_size::Int64 = 1000
)
  @model LogisticRegression(X, y) = begin
      p = size(X, 2)
      n = size(X, 1)
      #priors
      #v ~ InverseGamma(h, h)
      #α ~ TDist(1)
      #β ~ filldist(Uniform(-v, v), p)

      halfcauchy = Truncated(TDist(1), 0, Inf)
    
      τ ~ halfcauchy    ## Global Shrinkage
      λ ~ filldist(halfcauchy, p) ## Local Shrinkage
      σ ~ halfcauchy
      #α ~ Normal(0, τ * σ)
      β0 = repeat([0], p)  ## prior mean
      β ~ MvNormal(β0, λ * τ *σ)


      #z = α .+ X * β
      z = X * β

      ## Link Function

      #prob = Link.link.(z)
      prob = Link.link_function.(z)

      #likelihood
      for i = 1:n
          y[i] ~ Bernoulli(prob[i])
      end
  end

  return logistic_reg(formula, data, Link, LogisticRegression, sim_size)
end