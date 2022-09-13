function poisson_reg(formula::FormulaTerm, data::DataFrame, turingModel::Function, sim_size::Int64)
    formula = apply_schema(formula, schema(formula, data))
    y, X = modelcols(formula, data)

    if sim_size < 500
        @warn "Simulation size should generally be atleast 500."
    end
    chain = sample(CRRao_rng, turingModel(X, y), NUTS(), sim_size)
    return BayesianRegression(:PoissonRegression, chain, formula)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_Ridge, h::Float64 = 0.1, sim_size::Int64 = 1000)
```

Fit a Bayesian Poisson Regression model on the input data with a Ridge prior.

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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), Prior_Ridge())
┌ Info: Found initial step size
└   ϵ = 0.025
Chains MCMC chain (10000×19×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 28.3 seconds
Compute duration  = 28.3 seconds
parameters        = λ, α, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    1.3118    0.4894     0.0049    0.0066   5733.9828    1.0000      202.6214
           α   -1.8003    0.2607     0.0026    0.0038   4247.2367    1.0000      150.0843
        β[1]    0.1392    0.0656     0.0007    0.0008   5949.9827    1.0000      210.2542
        β[2]    1.1334    0.0563     0.0006    0.0007   5344.6101    1.0003      188.8622
        β[3]   -0.3259    0.2281     0.0023    0.0026   7065.4440    0.9999      249.6712
        β[4]    1.6983    0.0988     0.0010    0.0012   6534.2641    1.0001      230.9009
        β[5]    0.4053    0.1688     0.0017    0.0023   5330.2762    1.0006      188.3556

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.7113    0.9826    1.2040    1.5098    2.5620
           α   -2.3202   -1.9764   -1.7971   -1.6229   -1.3003
        β[1]    0.0115    0.0950    0.1399    0.1825    0.2690
        β[2]    1.0246    1.0950    1.1331    1.1712    1.2451
        β[3]   -0.7923   -0.4776   -0.3205   -0.1703    0.1022
        β[4]    1.5095    1.6308    1.6977    1.7645    1.8936
        β[5]    0.0755    0.2930    0.4068    0.5190    0.7331
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::PoissonRegression,
    prior::Prior_Ridge,
    h::Float64 = 0.1,
    sim_size::Int64 = 1000
)
    @model PoissonRegression(X, y) = begin
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
            y[i] ~ Poisson(mu[i])
        end
    end

    return poisson_reg(formula, data, PoissonRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_Laplace, h::Float64 = 0.1, sim_size::Int64 = 1000)
```

Fit a Bayesian Poisson Regression model on the input data with a Laplace prior.

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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), Prior_Laplace())
┌ Info: Found initial step size
└   ϵ = 0.025
Chains MCMC chain (10000×19×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 26.38 seconds
Compute duration  = 26.38 seconds
parameters        = λ, α, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    1.1036    0.5676     0.0057    0.0086   5101.9856    1.0003      193.4109
           α   -1.7912    0.2625     0.0026    0.0041   4611.2398    1.0002      174.8072
        β[1]    0.1360    0.0649     0.0006    0.0008   6345.1220    0.9999      240.5369
        β[2]    1.1324    0.0561     0.0006    0.0008   6267.6347    1.0006      237.5994
        β[3]   -0.2965    0.2234     0.0022    0.0027   7304.0984    1.0001      276.8906
        β[4]    1.7010    0.1012     0.0010    0.0013   7420.3061    0.9999      281.2960
        β[5]    0.3928    0.1730     0.0017    0.0021   6264.6983    0.9999      237.4881

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.4544    0.7383    0.9620    1.3049    2.5869
           α   -2.3130   -1.9684   -1.7862   -1.6133   -1.2838
        β[1]    0.0093    0.0924    0.1354    0.1801    0.2627
        β[2]    1.0241    1.0943    1.1313    1.1698    1.2448
        β[3]   -0.7542   -0.4437   -0.2889   -0.1370    0.1132
        β[4]    1.5029    1.6331    1.6994    1.7690    1.9002
        β[5]    0.0581    0.2740    0.3946    0.5113    0.7309
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::PoissonRegression,
    prior::Prior_Laplace,
    h::Float64 = 0.1,
    sim_size::Int64 = 1000
)
    @model PoissonRegression(X, y) = begin
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
            y[i] ~ Poisson(mu[i])
        end
    end

    return poisson_reg(formula, data, PoissonRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Cauchy, h::Float64 = 1.0, sim_size::Int64 = 1000)
```

Fit a Bayesian Poisson Regression model on the input data with a Cauchy prior.

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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), Prior_Cauchy())
┌ Info: Found initial step size
└   ϵ = 0.025
Chains MCMC chain (10000×19×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 27.23 seconds
Compute duration  = 27.23 seconds
parameters        = λ, α, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.8558    0.4620     0.0046    0.0050   7120.2358    0.9999      261.5138
           α   -1.7984    0.2622     0.0026    0.0038   4736.5277    0.9999      173.9644
        β[1]    0.1383    0.0649     0.0006    0.0008   6989.3372    1.0001      256.7061
        β[2]    1.1322    0.0573     0.0006    0.0008   5442.3181    0.9999      199.8868
        β[3]   -0.2928    0.2169     0.0022    0.0025   6830.7146    1.0000      250.8802
        β[4]    1.7040    0.0974     0.0010    0.0011   6738.4680    0.9999      247.4921
        β[5]    0.3945    0.1673     0.0017    0.0023   5730.9957    0.9999      210.4894

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.2927    0.5424    0.7551    1.0504    1.9964
           α   -2.3125   -1.9749   -1.7957   -1.6220   -1.2893
        β[1]    0.0112    0.0950    0.1366    0.1813    0.2677
        β[2]    1.0198    1.0937    1.1315    1.1709    1.2457
        β[3]   -0.7403   -0.4351   -0.2887   -0.1398    0.1058
        β[4]    1.5135    1.6384    1.7053    1.7704    1.8926
        β[5]    0.0677    0.2823    0.3952    0.5066    0.7253
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::PoissonRegression,
    prior::Prior_Cauchy,
    h::Float64 = 1.0,
    sim_size::Int64 = 1000
)
    @model PoissonRegression(X, y) = begin
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
            y[i] ~ Poisson(mu[i])
        end
    end

    return poisson_reg(formula, data, PoissonRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_TDist, h::Float64 = 2.0, sim_size::Int64 = 1000)
```

Fit a Bayesian Poisson Regression model on the input data with a t(ν) distributed prior.

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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 0.0125
Chains MCMC chain (10000×20×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 28.44 seconds
Compute duration  = 28.44 seconds
parameters        = λ, ν, α, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.9887    0.4187     0.0042    0.0046   6943.4330    1.0001      244.1861
           ν    3.0837    7.9963     0.0800    0.1140   4422.8043    1.0000      155.5409
           α   -1.8065    0.2648     0.0026    0.0042   3384.5428    0.9999      119.0274
        β[1]    0.1399    0.0656     0.0007    0.0009   5242.0449    1.0001      184.3518
        β[2]    1.1339    0.0565     0.0006    0.0009   4397.9611    1.0004      154.6672
        β[3]   -0.3097    0.2208     0.0022    0.0029   5930.8888    1.0000      208.5771
        β[4]    1.7026    0.1000     0.0010    0.0012   5706.3129    0.9999      200.6792
        β[5]    0.4025    0.1701     0.0017    0.0024   4239.8288    0.9999      149.1060

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.4024    0.7020    0.9159    1.1889    2.0151
           ν    0.5755    1.1693    1.8160    3.0737   12.7800
           α   -2.3383   -1.9843   -1.8024   -1.6222   -1.3155
        β[1]    0.0116    0.0952    0.1392    0.1836    0.2704
        β[2]    1.0255    1.0953    1.1331    1.1717    1.2464
        β[3]   -0.7635   -0.4518   -0.3017   -0.1559    0.1005
        β[4]    1.5112    1.6334    1.7023    1.7700    1.9025
        β[5]    0.0680    0.2864    0.4016    0.5174    0.7395
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::PoissonRegression,
    prior::Prior_TDist,
    h::Float64 = 2.0,
    sim_size::Int64 = 1000
)
    @model PoissonRegression(X, y) = begin
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
            y[i] ~ Poisson(mu[i])
        end
    end

    return poisson_reg(formula, data, PoissonRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Prior_Uniform, h::Float64 = 1.0, sim_size::Int64 = 1000)
```

Fit a Bayesian Poisson Regression model on the input data with a Uniform prior.

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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), Prior_Uniform())
Chains MCMC chain (10000×19×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 226.71 seconds
Compute duration  = 226.71 seconds
parameters        = λ, α, β[1], β[2], β[3], β[4], β[5]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse          ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64      Float64   Float64       Float64 

           λ    6.8751   14.2909     0.1429    0.9915     139.5154    1.0000        0.6154
           α    0.2792    0.0000     0.0000    0.0000      20.5530    0.9999        0.0907
        β[1]    0.2792    0.0000     0.0000    0.0000      20.5530    0.9999        0.0907
        β[2]    0.2792    0.0000     0.0000    0.0000      20.5530    0.9999        0.0907
        β[3]    0.2791    0.0056     0.0001    0.0001   10004.0032    1.0000       44.1261
        β[4]    0.2792    0.0000     0.0000    0.0000      20.5530    0.9999        0.0907
        β[5]   -0.2792    0.0000     0.0000    0.0000      20.5530    0.9999        0.0907

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.6280    1.3666    2.6401    6.3038   39.3119
           α    0.2792    0.2792    0.2792    0.2792    0.2792
        β[1]    0.2792    0.2792    0.2792    0.2792    0.2792
        β[2]    0.2792    0.2792    0.2792    0.2792    0.2792
        β[3]    0.2792    0.2792    0.2792    0.2792    0.2792
        β[4]    0.2792    0.2792    0.2792    0.2792    0.2792
        β[5]   -0.2792   -0.2792   -0.2792   -0.2792   -0.2792
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::PoissonRegression,
    prior::Prior_Uniform,
    h::Float64 = 1.0,
    sim_size::Int64 = 1000
)
    @model PoissonRegression(X, y) = begin
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
            y[i] ~ Poisson(mu[i])
        end
    end

    return poisson_reg(formula, data, PoissonRegression, sim_size)
end
