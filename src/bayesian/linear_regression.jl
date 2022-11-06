function linear_reg(formula::FormulaTerm, data::DataFrame, turingModel::Function, sim_size::Int64)
    formula = apply_schema(formula, schema(formula, data),RegressionModel)
    y, X = modelcols(formula, data)

    if sim_size < 500
        @warn "Simulation size should generally be atleast 500."
    end
    chain = sample(CRRao_rng, turingModel(X, y), NUTS(), sim_size)
    return BayesianRegression(:LinearRegression, chain, formula)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Ridge, h::Float64 = 0.01, sim_size::Int64 = 1000)
```

Fit a Bayesian Linear Regression model on the input data with a Ridge prior.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
julia> df = dataset("datasets", "mtcars")
32×12 DataFrame
 Row │ Model              MPG      Cyl    Disp     HP     DRat     WT       QSec     VS     AM     Gear   Carb  
     │ String31           Float64  Int64  Float64  Int64  Float64  Float64  Float64  Int64  Int64  Int64  Int64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Mazda RX4             21.0      6    160.0    110     3.9     2.62     16.46      0      1      4      4
   2 │ Mazda RX4 Wag         21.0      6    160.0    110     3.9     2.875    17.02      0      1      4      4
   3 │ Datsun 710            22.8      4    108.0     93     3.85    2.32     18.61      1      1      4      1
   4 │ Hornet 4 Drive        21.4      6    258.0    110     3.08    3.215    19.44      1      0      3      1
  ⋮  │         ⋮             ⋮       ⋮       ⋮       ⋮       ⋮        ⋮        ⋮       ⋮      ⋮      ⋮      ⋮
  30 │ Ferrari Dino          19.7      6    145.0    175     3.62    2.77     15.5       0      1      5      6
  31 │ Maserati Bora         15.0      8    301.0    335     3.54    3.57     14.6       0      1      5      8
  32 │ Volvo 142E            21.4      4    121.0    109     4.11    2.78     18.6       1      1      4      2
                                                                                                 25 rows omitted
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_Ridge())
┌ Info: Found initial step size
└   ϵ = 0.00078125
Chains MCMC chain (10000×18×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 30.44 seconds
Compute duration  = 30.44 seconds
parameters        = v, σ, α, β[1], β[2], β[3]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           v    6.9097    3.7793     0.0378    0.0609   3848.1626    0.9999      126.4013
           σ    2.6726    0.3878     0.0039    0.0061   3787.1472    1.0000      124.3972
        β[1]   28.6866    5.4205     0.0542    0.1106   2431.5304    1.0001       79.8690
        β[2]   -0.0395    0.0106     0.0001    0.0002   4057.7267    0.9999      133.2849
        β[3]   -2.7056    0.9635     0.0096    0.0176   2897.6230    1.0001       95.1788
        β[4]    1.5912    0.9825     0.0098    0.0198   2538.0548    1.0001       83.3680

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           v    2.5200    4.5014    6.0397    8.2382   16.6220
           σ    2.0519    2.4030    2.6297    2.8942    3.5482
        β[1]   17.6034   25.2623   28.9229   32.3360   38.7343
        β[2]   -0.0612   -0.0464   -0.0393   -0.0325   -0.0191
        β[3]   -4.5163   -3.3443   -2.7385   -2.1041   -0.7211
        β[4]   -0.2205    0.9158    1.5520    2.2028    3.6202
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LinearRegression,
    prior::Prior_Ridge,
    h::Float64 = 0.01,
    sim_size::Int64 = 1000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)

        #priors
        a0 = 0.1
        b0 = 0.1

        v ~ InverseGamma(h, h)
        σ ~ InverseGamma(a0, b0)
        #α ~ Normal(0, v * σ)
        β ~ filldist(Normal(0, v * σ), p)

        #likelihood
        #y ~ MvNormal(α .+ X * β, σ)
        y ~ MvNormal(X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Laplace, h::Float64 = 0.01, sim_size::Int64 = 1000)
```

Fit a Bayesian Linear Regression model on the input data with a Laplace prior.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
julia> df = dataset("datasets", "mtcars")
32×12 DataFrame
 Row │ Model              MPG      Cyl    Disp     HP     DRat     WT       QSec     VS     AM     Gear   Carb  
     │ String31           Float64  Int64  Float64  Int64  Float64  Float64  Float64  Int64  Int64  Int64  Int64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Mazda RX4             21.0      6    160.0    110     3.9     2.62     16.46      0      1      4      4
   2 │ Mazda RX4 Wag         21.0      6    160.0    110     3.9     2.875    17.02      0      1      4      4
   3 │ Datsun 710            22.8      4    108.0     93     3.85    2.32     18.61      1      1      4      1
   4 │ Hornet 4 Drive        21.4      6    258.0    110     3.08    3.215    19.44      1      0      3      1
  ⋮  │         ⋮             ⋮       ⋮       ⋮       ⋮       ⋮        ⋮        ⋮       ⋮      ⋮      ⋮      ⋮
  30 │ Ferrari Dino          19.7      6    145.0    175     3.62    2.77     15.5       0      1      5      6
  31 │ Maserati Bora         15.0      8    301.0    335     3.54    3.57     14.6       0      1      5      8
  32 │ Volvo 142E            21.4      4    121.0    109     4.11    2.78     18.6       1      1      4      2
                                                                                                 25 rows omitted
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_Laplace())
┌ Info: Found initial step size
└   ϵ = 0.00078125
Chains MCMC chain (10000×18×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 28.55 seconds
Compute duration  = 28.55 seconds
parameters        = v, σ, α, β[1], β[2], β[3]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           v    4.2213    3.0653     0.0307    0.0506   3799.4211    0.9999      133.0609
           σ    2.6713    0.3829     0.0038    0.0068   3782.5307    1.0001      132.4694
        β[1]   29.0523    5.2589     0.0526    0.1032   3144.5864    1.0004      110.1277
        β[2]   -0.0398    0.0106     0.0001    0.0002   4429.6471    1.0005      155.1323
        β[3]   -2.7161    0.9506     0.0095    0.0182   3299.1828    1.0009      115.5419
        β[4]    1.5129    0.9530     0.0095    0.0180   3383.7096    1.0002      118.5021

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           v    1.2438    2.3788    3.4110    5.1138   11.7423
           σ    2.0692    2.4016    2.6226    2.8897    3.5602
        β[1]   17.8056   25.8001   29.2866   32.5385   38.8889
        β[2]   -0.0614   -0.0466   -0.0395   -0.0326   -0.0194
        β[3]   -4.5559   -3.3384   -2.7407   -2.1204   -0.7254
        β[4]   -0.2790    0.8794    1.4691    2.1092    3.5245
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LinearRegression,
    prior::Prior_Laplace,
    h::Float64 = 0.01,
    sim_size::Int64 = 1000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)

        #priors
        a0 = 0.1
        b0 = 0.1
        v ~ InverseGamma(h, h)
        σ ~ InverseGamma(a0, b0)
        #α ~ Laplace(0, σ * v)
        β ~ filldist(Laplace(0, σ * v), p)

        #likelihood
        #y ~ MvNormal(α .+ X * β, σ)
        y ~ MvNormal(X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Cauchy, sim_size::Int64 = 1000)
```

Fit a Bayesian Linear Regression model on the input data with a Cauchy prior.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
julia> df = dataset("datasets", "mtcars")
32×12 DataFrame
 Row │ Model              MPG      Cyl    Disp     HP     DRat     WT       QSec     VS     AM     Gear   Carb  
     │ String31           Float64  Int64  Float64  Int64  Float64  Float64  Float64  Int64  Int64  Int64  Int64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Mazda RX4             21.0      6    160.0    110     3.9     2.62     16.46      0      1      4      4
   2 │ Mazda RX4 Wag         21.0      6    160.0    110     3.9     2.875    17.02      0      1      4      4
   3 │ Datsun 710            22.8      4    108.0     93     3.85    2.32     18.61      1      1      4      1
   4 │ Hornet 4 Drive        21.4      6    258.0    110     3.08    3.215    19.44      1      0      3      1
  ⋮  │         ⋮             ⋮       ⋮       ⋮       ⋮       ⋮        ⋮        ⋮       ⋮      ⋮      ⋮      ⋮
  30 │ Ferrari Dino          19.7      6    145.0    175     3.62    2.77     15.5       0      1      5      6
  31 │ Maserati Bora         15.0      8    301.0    335     3.54    3.57     14.6       0      1      5      8
  32 │ Volvo 142E            21.4      4    121.0    109     4.11    2.78     18.6       1      1      4      2
                                                                                                 25 rows omitted
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_Cauchy(), 20000)
┌ Info: Found initial step size
└   ϵ = 0.000390625
Chains MCMC chain (20000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:21000
Number of chains  = 1
Samples per chain = 20000
Wall duration     = 34.1 seconds
Compute duration  = 34.1 seconds
parameters        = σ, α, β[1], β[2], β[3]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           σ    2.5891    0.3413     0.0024    0.0036   8611.3030    1.0001      252.5087
        β[1]   30.2926    4.6666     0.0330    0.0590   5600.5552    1.0013      164.2247
        β[2]   -0.0394    0.0100     0.0001    0.0001   7985.0944    1.0009      234.1464
        β[3]   -2.8393    0.8638     0.0061    0.0106   6031.2854    1.0012      176.8550
        β[4]    1.2738    0.8524     0.0060    0.0107   5814.5026    1.0014      170.4983

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           σ    2.0266    2.3485    2.5547    2.7908    3.3512
        β[1]   20.8140   27.3265   30.3854   33.4168   39.1369
        β[2]   -0.0595   -0.0458   -0.0393   -0.0328   -0.0197
        β[3]   -4.5172   -3.4069   -2.8485   -2.2786   -1.1244
        β[4]   -0.3576    0.7039    1.2568    1.8199    3.0201
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LinearRegression,
    prior::Prior_Cauchy,
    sim_size::Int64 = 1000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)

        #priors
        σ ~ Truncated(TDist(1), 0, Inf)
        #α ~ TDist(1) * σ
        β ~ filldist(TDist(1) * σ, p)

        #likelihood
        #y ~ MvNormal(α .+ X * β, σ)
        y ~ MvNormal(X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_TDist, h::Float64 = 2.0, sim_size::Int64 = 1000)
```

Fit a Bayesian Linear Regression model on the input data with a t(ν) distributed prior.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsPlots, StatsModels
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
julia> df = dataset("datasets", "mtcars")
32×12 DataFrame
 Row │ Model              MPG      Cyl    Disp     HP     DRat     WT       QSec     VS     AM     Gear   Carb  
     │ String31           Float64  Int64  Float64  Int64  Float64  Float64  Float64  Int64  Int64  Int64  Int64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Mazda RX4             21.0      6    160.0    110     3.9     2.62     16.46      0      1      4      4
   2 │ Mazda RX4 Wag         21.0      6    160.0    110     3.9     2.875    17.02      0      1      4      4
   3 │ Datsun 710            22.8      4    108.0     93     3.85    2.32     18.61      1      1      4      1
   4 │ Hornet 4 Drive        21.4      6    258.0    110     3.08    3.215    19.44      1      0      3      1
  ⋮  │         ⋮             ⋮       ⋮       ⋮       ⋮       ⋮        ⋮        ⋮       ⋮      ⋮      ⋮      ⋮
  30 │ Ferrari Dino          19.7      6    145.0    175     3.62    2.77     15.5       0      1      5      6
  31 │ Maserati Bora         15.0      8    301.0    335     3.54    3.57     14.6       0      1      5      8
  32 │ Volvo 142E            21.4      4    121.0    109     4.11    2.78     18.6       1      1      4      2
                                                                                                 25 rows omitted
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 1.1920928955078126e-8
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 2.95 seconds
Compute duration  = 2.95 seconds
parameters        = ν, σ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           ν    1.0568    0.6520     0.0206    0.0260   502.7967    1.0003      170.4974
           σ    2.6259    0.3419     0.0108    0.0138   703.9042    0.9992      238.6925
        β[1]   30.0379    4.5994     0.1454    0.2797   269.7244    0.9999       91.4630
        β[2]   -0.0397    0.0102     0.0003    0.0004   461.0848    0.9990      156.3529
        β[3]   -2.7886    0.8671     0.0274    0.0448   313.0010    0.9993      106.1380
        β[4]    1.3133    0.8340     0.0264    0.0482   283.8113    0.9998       96.2399

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           ν    0.3782    0.6342    0.8964    1.2648    2.6659
           σ    2.0382    2.3839    2.5998    2.8364    3.3586
        β[1]   20.3671   27.1314   30.1112   33.2593   38.6807
        β[2]   -0.0602   -0.0459   -0.0399   -0.0328   -0.0196
        β[3]   -4.4999   -3.3668   -2.7715   -2.1960   -1.1627
        β[4]   -0.2606    0.7745    1.3046    1.8201    3.1602
        
julia> plot(container.chain)
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LinearRegression,
    prior::Prior_TDist,
    h::Float64 = 2.0,
    sim_size::Int64 = 1000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)

        #priors
        a0 = 0.1
        b0 = 0.1
        ν ~ InverseGamma(h, h)
        σ ~ InverseGamma(a0, b0)
        #α ~ TDist(ν) * σ
        β ~ filldist(TDist(ν) * σ, p)

        #likelihood
        #y ~ MvNormal(α .+ X * β, σ)
        y ~ MvNormal(X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end


"""
```julia
fit(formula::FormulaTerm,data::DataFrame,modelClass::LinearRegression,prior::Prior_HorseShoe,sim_size::Int64 = 1000)
```

Fit a Bayesian Linear Regression model on the input data with a HorseShoe prior.
# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsPlots, StatsModels
julia> CRRao.set_rng(StableRNG(123));
julia> df = dataset("datasets", "mtcars");                                                                                                 
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_HorseShoe())
┌ Info: Found initial step size
└   ϵ = 0.00078125
Chains MCMC chain (10000×22×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 6.03 seconds
Compute duration  = 6.03 seconds
parameters        = τ, λ[1], λ[2], λ[3], λ[4], σ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           τ    1.5165    1.4686     0.0147    0.0226   3890.9574    1.0002      644.9457
        λ[1]   20.3086   44.1555     0.4416    0.6318   4612.1706    0.9999      764.4904
        λ[2]    0.3844    1.0610     0.0106    0.0127   5516.8694    1.0000      914.4488
        λ[3]    2.3296    3.8670     0.0387    0.0649   4003.2481    0.9999      663.5584
        λ[4]    1.0810    2.3547     0.0235    0.0313   5438.8620    1.0005      901.5186
           σ    2.6166    0.3554     0.0036    0.0049   5482.7207    0.9999      908.7884
        β[1]   31.7408    4.8161     0.0482    0.0896   2609.4658    1.0001      432.5320
        β[2]   -0.0377    0.0104     0.0001    0.0002   4158.0492    1.0006      689.2175
        β[3]   -3.0310    0.9029     0.0090    0.0153   2960.2086    1.0003      490.6694
        β[4]    0.9813    0.8712     0.0087    0.0162   2585.6064    1.0003      428.5772

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%      97.5% 
      Symbol   Float64   Float64   Float64   Float64    Float64 

           τ    0.2005    0.6313    1.0816    1.8738     5.5363
        λ[1]    1.6126    5.2145    9.9806   20.1750   101.7374
        λ[2]    0.0056    0.0365    0.1223    0.3860     2.1596
        λ[3]    0.2267    0.7353    1.3384    2.5024    10.4504
        λ[4]    0.0406    0.2855    0.5983    1.2022     4.9044
           σ    2.0225    2.3627    2.5797    2.8283     3.4198
        β[1]   21.2706   28.6013   32.1666   35.2770    39.7922
        β[2]   -0.0587   -0.0443   -0.0375   -0.0308    -0.0174
        β[3]   -4.7043   -3.6342   -3.0709   -2.4655    -1.1592
        β[4]   -0.3857    0.2907    0.9011    1.5545     2.8715

julia> plot(container.chain)
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LinearRegression,
    prior::Prior_HorseShoe,
    sim_size::Int64 = 1000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)

        #priors
    
        halfcauchy = Truncated(TDist(1), 0, Inf)
    
        τ ~ halfcauchy    ## Global Shrinkage
        λ ~ filldist(halfcauchy, p) ## Local Shrinkage
        σ ~ halfcauchy
        #α ~ Normal(0, τ * σ)
        β0 = repeat([0], p)  ## prior mean
        β ~ MvNormal(β0, λ * τ *σ)
        
        #likelihood
        #y ~ MvNormal(α .+ X * β, σ)
        y ~ MvNormal( X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Gauss,alpha_prior_mean::Float64 = 0.0, beta_prior_mean::Float64, sim_size::Int64 = 1000, h::Float64 = 0.1)
```

Fit a Bayesian Linear Regression model on the input data with a Gaussian prior with user specific prior mean for α and β. User doesnot have
    idea about the prior sd of α and β. User ignore the specification for sd of α and β.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> CRRao.set_rng(StableRNG(123));
julia> df = dataset("datasets", "mtcars");                                                                                                
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_Gauss(),30.0,[0.0,-3.0,1.0],1000)
┌ Info: Found initial step size
└   ϵ = 0.000390625
Chains MCMC chain (1000×17×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 0.4 seconds
Compute duration  = 0.4 seconds
parameters        = σ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           σ    2.4817    0.3419     0.0108    0.0164   442.9220    1.0038     1115.6726
        β[1]   30.6898    2.2222     0.0703    0.1024   277.8914    1.0096      699.9784
        β[2]   -0.0383    0.0089     0.0003    0.0004   558.2894    1.0000     1406.2704
        β[3]   -2.9652    0.5603     0.0177    0.0242   417.3633    1.0013     1051.2930
        β[4]    1.2305    0.4641     0.0147    0.0214   312.8441    1.0115      788.0203

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           σ    1.8921    2.2422    2.4544    2.6939    3.2290
        β[1]   26.4166   29.1803   30.7420   32.1294   34.8860
        β[2]   -0.0552   -0.0442   -0.0380   -0.0323   -0.0214
        β[3]   -4.0735   -3.3125   -2.9765   -2.6029   -1.8782
        β[4]    0.3809    0.8875    1.2182    1.5413    2.1436
```
"""
function fit(
    formula::FormulaTerm
    , data::DataFrame
    , modelClass::LinearRegression
    , prior::Prior_Gauss
    , alpha_prior_mean::Float64
    , beta_prior_mean::Vector{Float64}
    , sim_size::Int64 = 1000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)
        α0 = alpha_prior_mean
        β0 = beta_prior_mean
        β_prior_mean = vcat(α0, β0)

        #priors
        a0 = 0.1
        b0 = 0.1

        Ip = 1*Matrix(I,p,p)

        #S = cov(X)+Ip
        #v ~ InverseGamma(h, h)
        σ ~ InverseGamma(a0, b0)
        #α ~ Normal(α0, v * σ)
        #β ~ MvNormal(β0, S)
        β ~ MvNormal(β_prior_mean, σ)

        #likelihood
        #y ~ MvNormal(α .+ X * β, σ)
        y ~ MvNormal(X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end


"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Gauss, alpha_prior_mean::Float64, alpha_prior_sd::Float64, beta_prior_mean::Vector{Float64}, beta_prior_sd::Vector{Float64}, sim_size::Int64 = 1000)
```

Fit a Bayesian Linear Regression model on the input data with a Gaussian prior with user specific prior mean and sd for α and β. 

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> CRRao.set_rng(StableRNG(123));
julia> df = dataset("datasets", "mtcars");                                                                                                
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_Gauss(),30.0,10.0,[0.0,-3.0,1.0],[0.1,1.0,1.0],1000)
┌ Info: Found initial step size
└   ϵ = 0.000390625
Chains MCMC chain (1000×17×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 2.1 seconds
Compute duration  = 2.1 seconds
parameters        = σ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           σ    2.5902    0.3556     0.0112    0.0173   479.5282    1.0029      227.9126
        β[1]   31.5741    3.0940     0.0978    0.1654   438.4853    1.0016      208.4056
        β[2]   -0.0371    0.0088     0.0003    0.0003   728.7433    1.0017      346.3609
        β[3]   -3.1311    0.5910     0.0187    0.0253   537.6704    1.0019      255.5468
        β[4]    1.0910    0.5777     0.0183    0.0303   461.2719    1.0021      219.2357

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           σ    1.9892    2.3336    2.5579    2.8106    3.3548
        β[1]   24.9976   29.6654   31.4881   33.5860   37.6309
        β[2]   -0.0546   -0.0430   -0.0373   -0.0311   -0.0200
        β[3]   -4.2471   -3.5287   -3.1438   -2.7626   -1.9238
        β[4]   -0.0285    0.7312    1.0926    1.4948    2.1519

```
"""
function fit(
    formula::FormulaTerm
    , data::DataFrame
    , modelClass::LinearRegression
    , prior::Prior_Gauss
    , alpha_prior_mean::Float64
    , alpha_prior_sd::Float64
    , beta_prior_mean::Vector{Float64}
    , beta_prior_sd::Vector{Float64}
    , sim_size::Int64 = 1000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)
        α0 = alpha_prior_mean
        σ_α0 = alpha_prior_sd
        β0 = beta_prior_mean
        σ_β0 = beta_prior_sd

        β_prior_mean = vcat(α0, β0)
        β_prior_sd = vcat(σ_α0, σ_β0)

        S = Matrix(Diagonal(β_prior_sd))
        S = S*S

        #priors
        a0 = 0.1
        b0 = 0.1

        σ ~ InverseGamma(a0, b0)
        #α ~ Normal(α0, σ_α0)
        β ~ MvNormal(β_prior_mean, S)

        #likelihood
        #y ~ MvNormal(α .+ X * β, σ)
        y ~ MvNormal(X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end