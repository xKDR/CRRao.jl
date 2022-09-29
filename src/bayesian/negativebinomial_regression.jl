function negativebinomial_reg(
    formula::FormulaTerm,
    data::DataFrame,
    turingModel::Function,
    sim_size::Int64
)
    formula = apply_schema(formula, schema(formula, data))
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
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), Prior_Ridge())
┌ Info: Found initial step size
└   ϵ = 0.025
Chains MCMC chain (10000×19×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 26.52 seconds
Compute duration  = 26.52 seconds
parameters        = λ, α, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    2.0416    0.4460     0.0045    0.0045   8499.3498    0.9999      320.5246
           α   -1.0792    0.5148     0.0051    0.0089   3405.4069    1.0010      128.4235
        β[1]   -0.0049    0.1614     0.0016    0.0023   4627.1117    1.0009      174.4960
        β[2]    1.0615    0.1319     0.0013    0.0020   5046.9022    1.0001      190.3270
        β[3]   -0.1757    0.5563     0.0056    0.0063   8056.2338    1.0001      303.8139
        β[4]    1.2810    0.3214     0.0032    0.0035   6779.1552    0.9999      255.6532
        β[5]    0.1493    0.2799     0.0028    0.0036   6164.9114    1.0004      232.4890

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    1.3159    1.7243    1.9928    2.3049    3.0445
           α   -2.0865   -1.4300   -1.0908   -0.7306   -0.0721
        β[1]   -0.3180   -0.1136   -0.0044    0.1053    0.3146
        β[2]    0.8046    0.9738    1.0594    1.1483    1.3262
        β[3]   -1.2332   -0.5561   -0.1992    0.2020    0.9502
        β[4]    0.6571    1.0654    1.2744    1.4900    1.9274
        β[5]   -0.4064   -0.0370    0.1501    0.3388    0.6903
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
        α ~ Normal(0, λ)
        β ~ filldist(Normal(0, λ), p)

        ## link
        z = α .+ X * β
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
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), Prior_Laplace())
┌ Info: Found initial step size
└   ϵ = 0.05
Chains MCMC chain (10000×19×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 26.96 seconds
Compute duration  = 26.96 seconds
parameters        = λ, α, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    2.1058    0.4611     0.0046    0.0052   8213.6672    0.9999      304.6048
           α   -1.0014    0.5020     0.0050    0.0084   3465.0499    1.0000      128.5018
        β[1]   -0.0207    0.1583     0.0016    0.0021   5223.4434    0.9999      193.7120
        β[2]    1.0465    0.1301     0.0013    0.0017   5029.9415    1.0000      186.5359
        β[3]   -0.1426    0.4996     0.0050    0.0057   7487.9201    0.9999      277.6903
        β[4]    1.2832    0.3245     0.0032    0.0035   6912.6238    0.9999      256.3554
        β[5]    0.1198    0.2656     0.0027    0.0039   5505.7699    1.0000      204.1821

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    1.3431    1.7782    2.0523    2.3788    3.1662
           α   -2.0082   -1.3266   -1.0000   -0.6730   -0.0202
        β[1]   -0.3373   -0.1240   -0.0190    0.0823    0.2921
        β[2]    0.7927    0.9595    1.0454    1.1337    1.3056
        β[3]   -1.1412   -0.4702   -0.1379    0.1801    0.8557
        β[4]    0.6480    1.0707    1.2824    1.4966    1.9203
        β[5]   -0.4026   -0.0558    0.1158    0.2980    0.6499
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
        α ~ Laplace(0, λ)
        β ~ filldist(Laplace(0, λ), p)

        ## link
        z = α .+ X * β
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
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), Prior_Cauchy())
┌ Info: Found initial step size
└   ϵ = 0.2
Chains MCMC chain (10000×19×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 27.58 seconds
Compute duration  = 27.58 seconds
parameters        = λ, α, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    2.0219    0.4304     0.0043    0.0047   7839.1614    1.0001      284.1923
           α   -1.0233    0.5192     0.0052    0.0091   3193.5541    1.0010      115.7756
        β[1]   -0.0192    0.1632     0.0016    0.0025   4320.9927    1.0006      156.6485
        β[2]    1.0535    0.1327     0.0013    0.0021   4739.9448    1.0008      171.8367
        β[3]   -0.1552    0.5453     0.0055    0.0069   7763.7273    1.0002      281.4576
        β[4]    1.2743    0.3250     0.0032    0.0041   6655.6093    1.0008      241.2851
        β[5]    0.1298    0.2822     0.0028    0.0036   5253.2578    1.0000      190.4458

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    1.3226    1.7126    1.9731    2.2757    2.9804
           α   -2.0538   -1.3647   -1.0180   -0.6733   -0.0207
        β[1]   -0.3375   -0.1285   -0.0189    0.0881    0.3042
        β[2]    0.8001    0.9647    1.0516    1.1418    1.3138
        β[3]   -1.1825   -0.5301   -0.1676    0.2010    0.9589
        β[4]    0.6478    1.0553    1.2704    1.4870    1.9319
        β[5]   -0.4131   -0.0613    0.1305    0.3166    0.6901
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
        α ~ TDist(1) * λ
        β ~ filldist(TDist(1) * λ, p)

        ## link
        z = α .+ X * β
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
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 0.05
Chains MCMC chain (10000×20×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 32.4 seconds
Compute duration  = 32.4 seconds
parameters        = λ, ν, α, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean        std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64    Float64    Float64   Float64     Float64   Float64       Float64 

           λ    2.0021     0.4262     0.0043    0.0048   7946.6182    0.9999      245.2887
           ν   20.4978   213.5274     2.1353    2.7473   6455.1193    0.9999      199.2505
           α   -1.0562     0.5154     0.0052    0.0076   4162.0565    1.0010      128.4704
        β[1]   -0.0096     0.1617     0.0016    0.0022   5232.3275    1.0005      161.5065
        β[2]    1.0581     0.1308     0.0013    0.0016   5850.3314    1.0004      180.5825
        β[3]   -0.1725     0.5396     0.0054    0.0056   7961.1718    0.9999      245.7379
        β[4]    1.2762     0.3222     0.0032    0.0036   7541.0855    0.9999      232.7711
        β[5]    0.1400     0.2822     0.0028    0.0037   6538.0847    1.0009      201.8114

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%      97.5% 
      Symbol   Float64   Float64   Float64   Float64    Float64 

           λ    1.2972    1.6983    1.9536    2.2528     2.9704
           ν    0.6529    1.9144    3.8831    9.7303   108.9970
           α   -2.0694   -1.4014   -1.0565   -0.7076    -0.0696
        β[1]   -0.3338   -0.1176   -0.0090    0.0979     0.3056
        β[2]    0.8046    0.9695    1.0576    1.1482     1.3157
        β[3]   -1.2170   -0.5366   -0.1879    0.1806     0.9140
        β[4]    0.6519    1.0551    1.2762    1.4910     1.9133
        β[5]   -0.4045   -0.0511    0.1377    0.3314     0.6957
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
        α ~ TDist(ν) * λ
        β ~ filldist(TDist(ν) * λ, p)

        ## link
        z = α .+ X * β
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
fit(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Uniform, h::Float64 = 0.1, sim_size::Int64 = 1000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a Uniform prior. Ibrahim and Laud (JASA, 1990) showed that the uniform flat priors for GLMs can lead to improper posterior distributions thus making them undesirable. In such cases, the Markov Chain struggles to converge. Even if it converges, results are unreliable.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), Prior_Uniform())
Chains MCMC chain (10000×19×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 294.79 seconds
Compute duration  = 294.79 seconds
parameters        = λ, α, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.9182    0.1461     0.0015    0.0016   9715.0842    1.0003       32.9562
           α    0.2792    0.0000     0.0000    0.0000     20.5530    0.9999        0.0697
        β[1]    0.2792    0.0000     0.0000    0.0000     20.5530    0.9999        0.0697
        β[2]    0.2792    0.0000     0.0000    0.0000     20.5530    0.9999        0.0697
        β[3]   -0.1238    0.2502     0.0025    0.0234     24.2396    1.5013        0.0822
        β[4]    0.2792    0.0000     0.0000    0.0000     20.5530    0.9999        0.0697
        β[5]   -0.2643    0.0899     0.0009    0.0075     81.7627    1.0276        0.2774

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.6696    0.8140    0.9070    1.0073    1.2411
           α    0.2792    0.2792    0.2792    0.2792    0.2792
        β[1]    0.2792    0.2792    0.2792    0.2792    0.2792
        β[2]    0.2792    0.2792    0.2792    0.2792    0.2792
        β[3]   -0.2792   -0.2792   -0.2792    0.2792    0.2792
        β[4]    0.2792    0.2792    0.2792    0.2792    0.2792
        β[5]   -0.2792   -0.2792   -0.2792   -0.2792    0.2792
``` 
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    PriorMod::Prior_Uniform,
    h::Float64 = 0.1,
    sim_size::Int64 = 1000
)
    @model NegativeBinomialRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)

        #priors
        λ ~ InverseGamma(h, h)
        α ~ Uniform(-λ, λ)
        β ~ filldist(Uniform(-λ, λ), p)

        ## link
        z = α .+ X * β
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
fit(formula::FormulaTerm,data::DataFrame,modelClass::NegBinomRegression,PriorMod::Prior_HorseShoe,sim_size::Int64 = 1000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a HorseShoe prior. 

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsPlots, StatsModels
julia> CRRao.set_rng(StableRNG(123))
julia> sanction = dataset("Zelig", "sanction");
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), Prior_HorseShoe())
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    PriorMod::Prior_HorseShoe,
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
        α ~ Normal(0, τ * σ)
        β0 = repeat([0], p)  ## prior mean
        β ~ MvNormal(β0, λ * τ *σ)


        ## link
        z = α .+ X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ NegativeBinomial2(mu[i], σ)
        end
    end

    return negativebinomial_reg(formula, data, NegativeBinomialRegression, sim_size)
end