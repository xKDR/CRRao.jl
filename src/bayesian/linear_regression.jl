function linear_reg(formula::FormulaTerm, data::DataFrame, turingModel::Function, sim_size::Int64)
    formula = apply_schema(formula, schema(formula, data))
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
           α   28.6866    5.4205     0.0542    0.1106   2431.5304    1.0001       79.8690
        β[1]   -0.0395    0.0106     0.0001    0.0002   4057.7267    0.9999      133.2849
        β[2]   -2.7056    0.9635     0.0096    0.0176   2897.6230    1.0001       95.1788
        β[3]    1.5912    0.9825     0.0098    0.0198   2538.0548    1.0001       83.3680

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           v    2.5200    4.5014    6.0397    8.2382   16.6220
           σ    2.0519    2.4030    2.6297    2.8942    3.5482
           α   17.6034   25.2623   28.9229   32.3360   38.7343
        β[1]   -0.0612   -0.0464   -0.0393   -0.0325   -0.0191
        β[2]   -4.5163   -3.3443   -2.7385   -2.1041   -0.7211
        β[3]   -0.2205    0.9158    1.5520    2.2028    3.6202
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
        α ~ Normal(0, v * σ)
        β ~ filldist(Normal(0, v * σ), p)

        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
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
           α   29.0523    5.2589     0.0526    0.1032   3144.5864    1.0004      110.1277
        β[1]   -0.0398    0.0106     0.0001    0.0002   4429.6471    1.0005      155.1323
        β[2]   -2.7161    0.9506     0.0095    0.0182   3299.1828    1.0009      115.5419
        β[3]    1.5129    0.9530     0.0095    0.0180   3383.7096    1.0002      118.5021

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           v    1.2438    2.3788    3.4110    5.1138   11.7423
           σ    2.0692    2.4016    2.6226    2.8897    3.5602
           α   17.8056   25.8001   29.2866   32.5385   38.8889
        β[1]   -0.0614   -0.0466   -0.0395   -0.0326   -0.0194
        β[2]   -4.5559   -3.3384   -2.7407   -2.1204   -0.7254
        β[3]   -0.2790    0.8794    1.4691    2.1092    3.5245
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
        α ~ Laplace(0, σ * v)
        β ~ filldist(Laplace(0, σ * v), p)

        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
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
           α   30.2926    4.6666     0.0330    0.0590   5600.5552    1.0013      164.2247
        β[1]   -0.0394    0.0100     0.0001    0.0001   7985.0944    1.0009      234.1464
        β[2]   -2.8393    0.8638     0.0061    0.0106   6031.2854    1.0012      176.8550
        β[3]    1.2738    0.8524     0.0060    0.0107   5814.5026    1.0014      170.4983

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           σ    2.0266    2.3485    2.5547    2.7908    3.3512
           α   20.8140   27.3265   30.3854   33.4168   39.1369
        β[1]   -0.0595   -0.0458   -0.0393   -0.0328   -0.0197
        β[2]   -4.5172   -3.4069   -2.8485   -2.2786   -1.1244
        β[3]   -0.3576    0.7039    1.2568    1.8199    3.0201
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
        α ~ TDist(1) * σ
        β ~ filldist(TDist(1) * σ, p)

        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
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
Chains MCMC chain (10000×18×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 41.09 seconds
Compute duration  = 41.09 seconds
parameters        = ν, σ, α, β[1], β[2], β[3]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           ν    1.0538    0.5576     0.0056    0.0143   1340.7091    0.9999       32.6318
           σ    2.6251    0.3559     0.0036    0.0043   6374.0312    0.9999      155.1388
           α   30.1859    4.7935     0.0479    0.0605   5361.7257    1.0006      130.5001
        β[1]   -0.0396    0.0103     0.0001    0.0001   5835.9959    1.0003      142.0434
        β[2]   -2.8099    0.8772     0.0088    0.0114   5301.0033    1.0010      129.0221
        β[3]    1.2856    0.8699     0.0087    0.0106   5752.1640    1.0003      140.0030

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           ν    0.3670    0.6600    0.9301    1.2961    2.4821
           σ    2.0327    2.3758    2.5885    2.8442    3.4393
           α   20.4816   27.0685   30.2787   33.4481   39.3462
        β[1]   -0.0599   -0.0464   -0.0396   -0.0326   -0.0198
        β[2]   -4.4924   -3.3902   -2.8250   -2.2351   -1.0257
        β[3]   -0.3642    0.7021    1.2642    1.8397    3.0849
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
        α ~ TDist(ν) * σ
        β ~ filldist(TDist(ν) * σ, p)

        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
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
julia> df = dataset("datasets", "mtcars");                                                                                                 25 rows omitted
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_HorseShoe())
Chains MCMC chain (1000×21×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 8.36 seconds
Compute duration  = 8.36 seconds
parameters        = τ, λ[1], λ[2], λ[3], σ, α, β[1], β[2], β[3]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           τ    6.9501    3.5908     0.1136    0.1893   315.9554    1.0005       37.7937
        λ[1]    0.2466    0.5920     0.0187    0.0210   658.8988    1.0010       78.8156
        λ[2]    0.6329    0.8646     0.0273    0.0285   692.1768    0.9992       82.7963
        λ[3]    0.6035    1.2252     0.0387    0.0527   490.4417    1.0017       58.6653
           σ    2.7096    0.3897     0.0123    0.0155   467.8857    1.0084       55.9672
           α   29.3528    5.3345     0.1687    0.3294   253.2201    1.0010       30.2895
        β[1]   -0.0389    0.0108     0.0003    0.0005   377.8089    0.9994       45.1924
        β[2]   -2.7502    0.9513     0.0301    0.0585   251.1050    1.0014       30.0365
        β[3]    1.4304    0.9676     0.0306    0.0564   253.3073    1.0001       30.2999

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           τ    2.8713    4.4991    6.0756    8.2895   16.6520
        λ[1]    0.0012    0.0094    0.0427    0.2089    1.7233
        λ[2]    0.0731    0.1972    0.4045    0.7690    2.4549
        λ[3]    0.0148    0.1111    0.2750    0.6325    3.0972
           σ    2.0768    2.4456    2.6632    2.9294    3.6832
           α   18.7766   25.6249   29.3846   33.1996   39.1392
        β[1]   -0.0580   -0.0461   -0.0394   -0.0322   -0.0154
        β[2]   -4.6896   -3.3985   -2.7725   -2.1005   -0.9716
        β[3]   -0.2594    0.7460    1.4037    2.0909    3.3493
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
        α ~ Normal(0, τ * σ)
        β0 = repeat([0], p)  ## prior mean
        β ~ MvNormal(β0, λ * τ *σ)
        
        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
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
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_Gauss(),0.0,[0.0,-3.0,1.0],1000)
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
    , h::Float64 = 0.1
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)
        α0 = alpha_prior_mean
        β0 = beta_prior_mean

        #priors
        a0 = 0.1
        b0 = 0.1

        Ip = 1*Matrix(I,p,p)

        S = cov(X)+Ip

        v ~ InverseGamma(h, h)
        σ ~ InverseGamma(a0, b0)
        α ~ Normal(α0, v * σ)
        β ~ MvNormal(β0, S)

        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
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

        S = Matrix(Diagonal(σ_β0))
        S = S*S

        #priors
        a0 = 0.1
        b0 = 0.1

        σ ~ InverseGamma(a0, b0)
        α ~ Normal(α0, σ_α0)
        β ~ MvNormal(β0, S)

        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end