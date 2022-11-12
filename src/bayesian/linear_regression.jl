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

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
  
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_Ridge())
┌ Info: Found initial step size
└   ϵ = 0.00078125
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 3.8 seconds
Compute duration  = 3.8 seconds
parameters        = v, σ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           v    6.7326    4.0697     0.1287    0.2091   385.1846    1.0072      101.3377
           σ    2.6887    0.3769     0.0119    0.0173   454.6314    0.9992      119.6084
        β[1]   28.5712    5.4865     0.1735    0.2940   275.1500    0.9999       72.3888
        β[2]   -0.0395    0.0101     0.0003    0.0005   449.6762    0.9994      118.3047
        β[3]   -2.7071    0.9612     0.0304    0.0495   304.8328    1.0005       80.1981
        β[4]    1.6235    0.9894     0.0313    0.0525   293.2379    0.9998       77.1476

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           v    2.4199    4.3092    5.8397    8.0100   15.9390
           σ    2.0880    2.4184    2.6358    2.9308    3.5183
        β[1]   17.0694   25.0878   28.6635   32.2368   39.1438
        β[2]   -0.0594   -0.0462   -0.0398   -0.0327   -0.0198
        β[3]   -4.5435   -3.3350   -2.6938   -2.1350   -0.7247
        β[4]   -0.2647    0.9636    1.5983    2.2412    3.6841
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
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_Laplace())
┌ Info: Found initial step size
└   ϵ = 0.00078125
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 2.81 seconds
Compute duration  = 2.81 seconds
parameters        = v, σ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           v    4.0802    2.9685     0.0939    0.1488   388.0064    1.0061      138.0806
           σ    2.6879    0.3859     0.0122    0.0187   334.4383    1.0153      119.0172
        β[1]   28.6972    5.2832     0.1671    0.3359   182.9859    1.0255       65.1195
        β[2]   -0.0400    0.0107     0.0003    0.0005   391.2879    1.0077      139.2484
        β[3]   -2.6644    0.9818     0.0310    0.0543   249.1040    1.0171       88.6491
        β[4]    1.5659    0.9686     0.0306    0.0613   186.8354    1.0267       66.4895

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           v    1.3030    2.3492    3.2065    4.8058   12.2525
           σ    2.0493    2.4028    2.6405    2.9231    3.5532
        β[1]   17.7583   25.3061   28.8668   32.2456   38.1808
        β[2]   -0.0615   -0.0469   -0.0398   -0.0329   -0.0187
        β[3]   -4.4721   -3.3004   -2.7042   -2.0441   -0.7107
        β[4]   -0.1806    0.8682    1.5224    2.1637    3.6193
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
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
  
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_Cauchy(), 1000)
┌ Info: Found initial step size
└   ϵ = 0.000390625
Chains MCMC chain (1000×17×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 0.75 seconds
Compute duration  = 0.75 seconds
parameters        = σ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           σ    2.5922    0.3538     0.0112    0.0173   425.5703    0.9993      570.4696
        β[1]   30.0880    4.7389     0.1499    0.2468   229.5288    1.0065      307.6794
        β[2]   -0.0395    0.0100     0.0003    0.0005   329.9440    1.0007      442.2842
        β[3]   -2.8197    0.8453     0.0267    0.0475   218.5494    1.0049      292.9617
        β[4]    1.3148    0.8630     0.0273    0.0428   245.2960    1.0057      328.8150

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           σ    2.0287    2.3224    2.5539    2.8095    3.3997
        β[1]   20.5220   26.8878   30.1467   33.4241   38.9193
        β[2]   -0.0600   -0.0454   -0.0400   -0.0334   -0.0192
        β[3]   -4.4784   -3.3625   -2.8345   -2.2815   -1.1017
        β[4]   -0.2688    0.6897    1.3047    1.8932    3.0914
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
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 2.44140625e-5
Chains MCMC chain (1000×18×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 3.08 seconds
Compute duration  = 3.08 seconds
parameters        = ν, σ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           ν    1.0756    0.5583     0.0177    0.0260   513.4511    0.9990      166.7049
           σ    2.6164    0.3540     0.0112    0.0178   417.8954    0.9990      135.6803
        β[1]   29.9375    4.8157     0.1523    0.2756   247.5134    1.0049       80.3615
        β[2]   -0.0396    0.0096     0.0003    0.0004   416.4579    0.9996      135.2136
        β[3]   -2.7843    0.8477     0.0268    0.0424   271.7779    1.0003       88.2396
        β[4]    1.3307    0.8752     0.0277    0.0477   275.8558    1.0047       89.5636

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           ν    0.3741    0.6856    0.9406    1.3410    2.4627
           σ    2.0262    2.3547    2.5815    2.8333    3.4249
        β[1]   19.2904   27.0648   30.1711   33.0837   38.8812
        β[2]   -0.0587   -0.0458   -0.0393   -0.0328   -0.0202
        β[3]   -4.3684   -3.3394   -2.8206   -2.2711   -1.0594
        β[4]   -0.2602    0.7464    1.3014    1.8909    3.1216
        
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
julia> df = dataset("datasets", "mtcars");                                                                                                 
julia> CRRao.set_rng(StableRNG(123));
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_HorseShoe())
┌ Info: Found initial step size
└   ϵ = 0.00078125
Chains MCMC chain (1000×22×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 1.25 seconds
Compute duration  = 1.25 seconds
parameters        = τ, λ[1], λ[2], λ[3], λ[4], σ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean        std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64    Float64    Float64   Float64    Float64   Float64       Float64 

           τ    1.5261     1.6534     0.0523    0.0828   436.8230    0.9997      350.8618
        λ[1]   24.2059   136.8466     4.3275    6.3344   451.4820    0.9999      362.6362
        λ[2]    0.3444     0.5997     0.0190    0.0200   781.1539    0.9993      627.4329
        λ[3]    2.1643     4.5499     0.1439    0.2182   389.8004    1.0002      313.0927
        λ[4]    1.1324     2.6245     0.0830    0.1221   493.6815    1.0004      396.5314
           σ    2.6141     0.3517     0.0111    0.0165   460.6283    0.9993      369.9826
        β[1]   31.8252     4.8143     0.1522    0.3050   179.8377    1.0103      144.4479
        β[2]   -0.0372     0.0110     0.0003    0.0006   294.1383    1.0042      236.2557
        β[3]   -3.0585     0.9423     0.0298    0.0595   178.2695    1.0099      143.1883
        β[4]    0.9625     0.8550     0.0270    0.0523   198.7195    1.0077      159.6141

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           τ    0.2477    0.6709    1.0774    1.8008    5.6212
        λ[1]    1.6288    5.3511    9.8538   20.7166   83.5487
        λ[2]    0.0059    0.0418    0.1257    0.3737    2.2231
        λ[3]    0.2183    0.7248    1.2997    2.4004    8.0168
        λ[4]    0.0437    0.2932    0.6094    1.1282    4.8634
           σ    2.0417    2.3731    2.5859    2.8043    3.3936
        β[1]   20.9216   29.1543   32.3072   35.1725   39.4603
        β[2]   -0.0597   -0.0442   -0.0369   -0.0300   -0.0171
        β[3]   -4.7741   -3.6626   -3.1250   -2.5222   -1.0155
        β[4]   -0.3640    0.3357    0.8594    1.4728    2.8541

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
julia> df = dataset("datasets", "mtcars");
julia> CRRao.set_rng(StableRNG(123));
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_Gauss(),30.0,[0.0,-3.0,1.0],1000)
┌ Info: Found initial step size
└   ϵ = 0.000390625
Chains MCMC chain (1000×17×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 0.44 seconds
Compute duration  = 0.44 seconds
parameters        = σ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           σ    2.4817    0.3419     0.0108    0.0164   442.9220    1.0038     1011.2375
        β[1]   30.6898    2.2222     0.0703    0.1024   277.8914    1.0096      634.4553
        β[2]   -0.0383    0.0089     0.0003    0.0004   558.2894    1.0000     1274.6332
        β[3]   -2.9652    0.5603     0.0177    0.0242   417.3633    1.0013      952.8843
        β[4]    1.2305    0.4641     0.0147    0.0214   312.8441    1.0115      714.2558

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
julia> df = dataset("datasets", "mtcars");  
julia> CRRao.set_rng(StableRNG(123));                                                                                             
julia> container = fit(@formula(MPG ~ HP + WT + Gear), df, LinearRegression(), Prior_Gauss(),30.0,10.0,[0.0,-3.0,1.0],[0.1,1.0,1.0],1000)
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

           σ    2.5902    0.3556     0.0112    0.0173   479.5282    1.0029     1207.8796
        β[1]   31.5741    3.0940     0.0978    0.1654   438.4853    1.0016     1104.4971
        β[2]   -0.0371    0.0088     0.0003    0.0003   728.7433    1.0017     1835.6254
        β[3]   -3.1311    0.5910     0.0187    0.0253   537.6704    1.0019     1354.3334
        β[4]    1.0910    0.5777     0.0183    0.0303   461.2719    1.0021     1161.8939

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