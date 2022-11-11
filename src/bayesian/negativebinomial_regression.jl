function negativebinomial_reg(
    formula::FormulaTerm,
    data::DataFrame,
    turingModel::Function,
    sim_size::Int64
)
    formula = apply_schema(formula, schema(formula, data),RegressionModel)
    y, X = modelcols(formula, data)

    if sim_size < 500
        @warn "Simulation size should generally be atleast 500."
    end
    chain = sample(CRRao_rng, turingModel(X, y), NUTS(), sim_size)
    return BayesianRegression(:NegativeBinomialRegression, chain, formula)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Ridge, h::Float64 = 0.1, sim_size::Int64 = 1000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a Ridge prior.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> sanction = dataset("Zelig", "sanction")
78×8 DataFrame
 Row │ Mil    Coop   Target  Import  Export  Cost   Num    NCost         
     │ Int32  Int32  Int32   Int32   Int32   Int32  Int32  Cat…          
─────┼───────────────────────────────────────────────────────────────────
   1 │     1      4       3       1       1      4     15  major loss
   2 │     0      2       3       0       1      3      4  modest loss
   3 │     0      1       3       1       0      2      1  little effect
   4 │     1      1       3       1       1      2      1  little effect
  ⋮  │   ⋮      ⋮      ⋮       ⋮       ⋮       ⋮      ⋮          ⋮
  76 │     0      4       3       1       0      2     13  little effect
  77 │     0      1       2       0       0      1      1  net gain
  78 │     1      3       1       1       1      2     10  little effect

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
  
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), Prior_Ridge())
┌ Info: Found initial step size
└   ϵ = 0.05
Chains MCMC chain (1000×19×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 2.1 seconds
Compute duration  = 2.1 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5], β[6]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    2.0335    0.4164     0.0132    0.0144   602.6989    1.0028      287.2731
        β[1]   -1.1009    0.5388     0.0170    0.0349   216.8922    0.9992      103.3804
        β[2]    0.0011    0.1636     0.0052    0.0095   304.5594    0.9995      145.1665
        β[3]    1.0603    0.1306     0.0041    0.0074   370.3104    0.9994      176.5064
        β[4]   -0.1579    0.5449     0.0172    0.0214   884.5495    1.0033      421.6156
        β[5]    1.2946    0.3216     0.0102    0.0120   734.1990    1.0016      349.9519
        β[6]    0.1590    0.2913     0.0092    0.0143   426.8561    0.9990      203.4586

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    1.3742    1.7265    2.0057    2.2686    2.9704
        β[1]   -2.1760   -1.4541   -1.0944   -0.7251   -0.0849
        β[2]   -0.3026   -0.1087   -0.0005    0.1103    0.3231
        β[3]    0.7923    0.9728    1.0539    1.1497    1.3159
        β[4]   -1.1527   -0.5150   -0.1671    0.1970    1.0099
        β[5]    0.6724    1.0801    1.2830    1.5037    1.9353
        β[6]   -0.4259   -0.0374    0.1497    0.3484    0.7325

```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    prior::Prior_Ridge,
    h::Float64 = 0.1,
    sim_size::Int64 = 1000
)
    @model NegativeBinomialRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)

        #priors
        λ ~ InverseGamma(h, h)
        #α ~ Normal(0, λ)
        β ~ filldist(Normal(0, λ), p)

        ## link
        #z = α .+ X * β
        z = X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ NegativeBinomial2(mu[i], λ)
        end
    end

    return negativebinomial_reg(formula, data, NegativeBinomialRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Laplace, h::Float64 = 0.01, sim_size::Int64 = 1000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a Laplace prior.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> sanction = dataset("Zelig", "sanction")
78×8 DataFrame
 Row │ Mil    Coop   Target  Import  Export  Cost   Num    NCost         
     │ Int32  Int32  Int32   Int32   Int32   Int32  Int32  Cat…          
─────┼───────────────────────────────────────────────────────────────────
   1 │     1      4       3       1       1      4     15  major loss
   2 │     0      2       3       0       1      3      4  modest loss
   3 │     0      1       3       1       0      2      1  little effect
   4 │     1      1       3       1       1      2      1  little effect
  ⋮  │   ⋮      ⋮      ⋮       ⋮       ⋮       ⋮      ⋮          ⋮
  76 │     0      4       3       1       0      2     13  little effect
  77 │     0      1       2       0       0      1      1  net gain
  78 │     1      3       1       1       1      2     10  little effect

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), Prior_Laplace())
┌ Info: Found initial step size
└   ϵ = 0.05
Chains MCMC chain (1000×19×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 2.06 seconds
Compute duration  = 2.06 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5], β[6]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    2.1157    0.4443     0.0141    0.0154   961.8068    0.9990      465.9917
        β[1]   -1.0085    0.5101     0.0161    0.0250   289.3129    1.0108      140.1710
        β[2]   -0.0248    0.1554     0.0049    0.0064   476.0466    1.0043      230.6427
        β[3]    1.0521    0.1305     0.0041    0.0055   342.3108    1.0075      165.8482
        β[4]   -0.1528    0.5179     0.0164    0.0183   630.1413    1.0002      305.3010
        β[5]    1.2862    0.3096     0.0098    0.0113   576.8927    1.0023      279.5023
        β[6]    0.1316    0.2634     0.0083    0.0114   491.5701    1.0014      238.1638

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    1.3970    1.8074    2.0635    2.3653    3.1272
        β[1]   -2.0432   -1.3326   -0.9808   -0.6653   -0.0397
        β[2]   -0.3230   -0.1281   -0.0271    0.0741    0.2848
        β[3]    0.8077    0.9677    1.0496    1.1381    1.3186
        β[4]   -1.1888   -0.4738   -0.1480    0.1743    0.8940
        β[5]    0.6787    1.0819    1.3003    1.4830    1.8838
        β[6]   -0.3786   -0.0462    0.1339    0.3186    0.6422
``` 
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    prior::Prior_Laplace,
    h::Float64 = 0.1,
    sim_size::Int64 = 1000
)
    @model NegativeBinomialRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)

        #priors
        λ ~ InverseGamma(h, h)
        #α ~ Laplace(0, λ)
        β ~ filldist(Laplace(0, λ), p)

        ## link
        #z = α .+ X * β
        z =  X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ NegativeBinomial2(mu[i], λ)
        end
    end

    return negativebinomial_reg(formula, data, NegativeBinomialRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Cauchy, h::Float64 = 1.0, sim_size::Int64 = 1000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a Cauchy prior.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> sanction = dataset("Zelig", "sanction")
78×8 DataFrame
 Row │ Mil    Coop   Target  Import  Export  Cost   Num    NCost         
     │ Int32  Int32  Int32   Int32   Int32   Int32  Int32  Cat…          
─────┼───────────────────────────────────────────────────────────────────
   1 │     1      4       3       1       1      4     15  major loss
   2 │     0      2       3       0       1      3      4  modest loss
   3 │     0      1       3       1       0      2      1  little effect
   4 │     1      1       3       1       1      2      1  little effect
  ⋮  │   ⋮      ⋮      ⋮       ⋮       ⋮       ⋮      ⋮          ⋮
  76 │     0      4       3       1       0      2     13  little effect
  77 │     0      1       2       0       0      1      1  net gain
  78 │     1      3       1       1       1      2     10  little effect
                                                          71 rows omitted

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), Prior_Cauchy())
┌ Info: Found initial step size
└   ϵ = 0.05
┌ Warning: The current proposal will be rejected due to numerical error(s).
│   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
└ @ AdvancedHMC ~/.julia/packages/AdvancedHMC/iWHPQ/src/hamiltonian.jl:47
Chains MCMC chain (1000×19×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 2.5 seconds
Compute duration  = 2.5 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5], β[6]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    2.0094    0.4380     0.0139    0.0169   690.8506    1.0028      276.7831
        β[1]   -1.0783    0.5492     0.0174    0.0254   373.3703    1.0000      149.5875
        β[2]   -0.0077    0.1702     0.0054    0.0075   431.5668    0.9991      172.9034
        β[3]    1.0621    0.1340     0.0042    0.0052   498.3356    0.9999      199.6537
        β[4]   -0.1812    0.5431     0.0172    0.0239   653.5156    1.0022      261.8252
        β[5]    1.2693    0.3302     0.0104    0.0128   624.9936    0.9991      250.3981
        β[6]    0.1551    0.2951     0.0093    0.0106   548.9336    1.0019      219.9253

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    1.3395    1.7025    1.9368    2.2640    3.0577
        β[1]   -2.2001   -1.4242   -1.0651   -0.7256   -0.0066
        β[2]   -0.3558   -0.1220   -0.0026    0.0946    0.3230
        β[3]    0.7907    0.9789    1.0606    1.1386    1.3336
        β[4]   -1.1899   -0.5662   -0.1798    0.1778    0.9467
        β[5]    0.6384    1.0495    1.2673    1.4884    1.9266
        β[6]   -0.4275   -0.0388    0.1676    0.3465    0.7206

```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    prior::Prior_Cauchy,
    h::Float64 = 1.0,
    sim_size::Int64 = 1000
)
    @model NegativeBinomialRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        #α ~ TDist(1) * λ
        β ~ filldist(TDist(1) * λ, p)

        ## link
        #z = α .+ X * β
        z =  X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ NegativeBinomial2(mu[i], λ)
        end
    end

    return negativebinomial_reg(formula, data, NegativeBinomialRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_TDist, h::Float64 = 1.0, sim_size::Int64 = 1000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a t(ν) distributed prior.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> sanction = dataset("Zelig", "sanction")
78×8 DataFrame
 Row │ Mil    Coop   Target  Import  Export  Cost   Num    NCost         
     │ Int32  Int32  Int32   Int32   Int32   Int32  Int32  Cat…          
─────┼───────────────────────────────────────────────────────────────────
   1 │     1      4       3       1       1      4     15  major loss
   2 │     0      2       3       0       1      3      4  modest loss
   3 │     0      1       3       1       0      2      1  little effect
   4 │     1      1       3       1       1      2      1  little effect
  ⋮  │   ⋮      ⋮      ⋮       ⋮       ⋮       ⋮      ⋮          ⋮
  76 │     0      4       3       1       0      2     13  little effect
  77 │     0      1       2       0       0      1      1  net gain
  78 │     1      3       1       1       1      2     10  little effect
                                                          71 rows omitted
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 0.05
Chains MCMC chain (1000×20×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 2.71 seconds
Compute duration  = 2.71 seconds
parameters        = λ, ν, β[1], β[2], β[3], β[4], β[5], β[6]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean        std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64    Float64    Float64   Float64    Float64   Float64       Float64 

           λ    2.0088     0.4446     0.0141    0.0161   819.9421    0.9990      302.1157
           ν   21.7233   116.4227     3.6816    4.9619   685.3526    0.9996      252.5249
        β[1]   -1.0604     0.5311     0.0168    0.0283   397.1943    0.9997      146.3502
        β[2]   -0.0109     0.1620     0.0051    0.0068   654.7909    1.0005      241.2642
        β[3]    1.0601     0.1326     0.0042    0.0071   449.7489    0.9992      165.7144
        β[4]   -0.1635     0.5340     0.0169    0.0187   724.3698    0.9996      266.9012
        β[5]    1.2760     0.3239     0.0102    0.0119   679.5304    1.0006      250.3797
        β[6]    0.1456     0.2862     0.0091    0.0133   500.7724    0.9991      184.5145

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%      97.5% 
      Symbol   Float64   Float64   Float64   Float64    Float64 

           λ    1.2838    1.6993    1.9651    2.2594     3.0105
           ν    0.7203    1.9770    4.0830   10.1954   152.7865
        β[1]   -2.1498   -1.4146   -1.0534   -0.6976    -0.0150
        β[2]   -0.3198   -0.1253   -0.0024    0.1027     0.3046
        β[3]    0.8080    0.9688    1.0538    1.1519     1.3261
        β[4]   -1.2265   -0.5194   -0.1622    0.2064     0.8424
        β[5]    0.6213    1.0451    1.2792    1.4909     1.8955
        β[6]   -0.4084   -0.0453    0.1477    0.3433     0.7080

```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    prior::Prior_TDist,
    h::Float64 = 1.0,
    sim_size::Int64 = 1000
)
    @model NegativeBinomialRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        ν ~ InverseGamma(h, h)
        #α ~ TDist(ν) * λ
        β ~ filldist(TDist(ν) * λ, p)

        ## link
        #z = α .+ X * β
        z =  X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ NegativeBinomial2(mu[i], λ)
        end
    end

    return negativebinomial_reg(formula, data, NegativeBinomialRegression, sim_size)
end


"""
```julia
fit(formula::FormulaTerm,data::DataFrame,modelClass::NegBinomRegression,prior::Prior_HorseShoe,sim_size::Int64 = 1000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a HorseShoe prior. 

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsPlots, StatsModels
julia> sanction = dataset("Zelig", "sanction");
julia> CRRao.set_rng(StableRNG(123))
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), Prior_HorseShoe())
┌ Info: Found initial step size
└   ϵ = 0.05
Chains MCMC chain (1000×26×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 3.74 seconds
Compute duration  = 3.74 seconds
parameters        = τ, λ[1], λ[2], λ[3], λ[4], λ[5], λ[6], σ, β[1], β[2], β[3], β[4], β[5], β[6]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           τ    0.4838    0.4157     0.0131    0.0308   173.3105    1.0086       46.3397
        λ[1]    1.5987    2.6514     0.0838    0.1408   419.8568    1.0017      112.2612
        λ[2]    0.5508    0.8382     0.0265    0.0342   585.4782    1.0002      156.5450
        λ[3]    2.0960    2.2974     0.0726    0.1275   359.7626    1.0205       96.1932
        λ[4]    0.9183    1.2790     0.0404    0.0631   447.9905    1.0010      119.7835
        λ[5]    2.5239    3.4791     0.1100    0.1755   369.0944    1.0009       98.6883
        λ[6]    0.7963    3.1672     0.1002    0.1031   906.7498    0.9990      242.4465
           σ    2.3692    0.5405     0.0171    0.0335   160.7206    1.0116       42.9734
        β[1]   -0.6991    0.4640     0.0147    0.0403    74.1265    1.0119       19.8199
        β[2]   -0.0675    0.1221     0.0039    0.0084   157.2105    1.0021       42.0349
        β[3]    0.9701    0.1417     0.0045    0.0129    45.9995    1.0279       12.2993
        β[4]   -0.0838    0.3660     0.0116    0.0201   289.6569    1.0192       77.4484
        β[5]    1.2784    0.3110     0.0098    0.0193   190.0338    1.0033       50.8112
        β[6]   -0.0028    0.2110     0.0067    0.0169    95.8876    1.0065       25.6384

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           τ    0.0879    0.2295    0.3679    0.5927    1.4404
        λ[1]    0.0227    0.4536    0.9371    1.8603    6.6116
        λ[2]    0.0162    0.1229    0.2742    0.6572    2.5594
        λ[3]    0.2818    0.9145    1.5319    2.3517    7.9027
        λ[4]    0.0562    0.2805    0.5363    1.0203    4.5210
        λ[5]    0.3047    0.9296    1.5168    2.8306   10.3197
        λ[6]    0.0241    0.1982    0.3462    0.8242    3.1959
           σ    1.5715    1.9489    2.3025    2.6799    3.6985
        β[1]   -1.6118   -1.0219   -0.7054   -0.3626    0.0332
        β[2]   -0.2998   -0.1666   -0.0557    0.0150    0.1638
        β[3]    0.7268    0.8657    0.9770    1.0755    1.2406
        β[4]   -0.9419   -0.2720   -0.0234    0.1072    0.6187
        β[5]    0.6108    1.0708    1.2936    1.5336    1.8016
        β[6]   -0.3937   -0.1448   -0.0029    0.1130    0.4378
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    prior::Prior_HorseShoe,
    sim_size::Int64 = 1000
)
    @model NegativeBinomialRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)

        #priors
        
        halfcauchy = Truncated(TDist(1), 0, Inf)
        
        τ ~ halfcauchy    ## Global Shrinkage
        λ ~ filldist(halfcauchy, p) ## Local Shrinkage
        σ ~ halfcauchy
        #α ~ Normal(0, τ * σ)
        β0 = repeat([0], p)  ## prior mean
        β ~ MvNormal(β0, λ * τ *σ)


        ## link
        #z = α .+ X * β
        z =  X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ NegativeBinomial2(mu[i], σ)
        end
    end

    return negativebinomial_reg(formula, data, NegativeBinomialRegression, sim_size)
end