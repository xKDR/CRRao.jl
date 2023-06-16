function poisson_reg(formula::FormulaTerm, data::DataFrame, turingModel::Function, sim_size::Int64)
    formula = apply_schema(formula, schema(formula, data),RegressionModel)
    y, X = modelcols(formula, data)

    if sim_size < 500
        @warn "Simulation size should generally be atleast 500."
    end
    chain = sample(CRRao_rng, turingModel(X, y), NUTS(), sim_size)
    return BayesianRegression(:PoissonRegression, chain, formula)
end

"""
```julia
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Ridge, h::Float64 = 0.1, sim_size::Int64 = 1000)
```

Fit a Bayesian Poisson Regression model on the input data with a Ridge prior.

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
  
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), Ridge())
┌ Info: Found initial step size
└   ϵ = 0.025
Chains MCMC chain (1000×19×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 1.82 seconds
Compute duration  = 1.82 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5], β[6]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    1.3067    0.4614     0.0146    0.0187   619.3195    0.9991      339.9119
        β[1]   -1.7954    0.2642     0.0084    0.0116   469.7099    0.9999      257.7990
        β[2]    0.1387    0.0647     0.0020    0.0026   654.1981    0.9993      359.0549
        β[3]    1.1327    0.0538     0.0017    0.0017   653.9282    0.9992      358.9068
        β[4]   -0.3230    0.2179     0.0069    0.0075   743.3958    0.9994      408.0108
        β[5]    1.6961    0.0979     0.0031    0.0036   756.8072    1.0002      415.3717
        β[6]    0.4041    0.1715     0.0054    0.0068   571.0010    0.9999      313.3924

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.7431    1.0176    1.2106    1.5135    2.5469
        β[1]   -2.3068   -1.9807   -1.7939   -1.6064   -1.2936
        β[2]    0.0182    0.0939    0.1375    0.1795    0.2709
        β[3]    1.0273    1.0957    1.1309    1.1687    1.2394
        β[4]   -0.7491   -0.4674   -0.3303   -0.1709    0.1006
        β[5]    1.5116    1.6290    1.6943    1.7634    1.8955
        β[6]    0.0665    0.2867    0.4031    0.5242    0.7363

julia> predict(container,sanction)
78-element Vector{Float64}:
    17.245248953660397
    13.311751426697569
    0.7896838153702627
    0.7896838153702627
    0.7896838153702627
    0.7896838153702627
    2.12053910182601
    0.7896838153702627
    ⋮
    3.1972595300758626
    5.726972543900231
    2.7850411549518532
    0.7896838153702627
    23.451157821412178
    1.0317165509592108
    5.726972543900231
        
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::PoissonRegression,
    prior::Ridge,
    h::Float64 = 0.1,
    sim_size::Int64 = 1000
)
    @model PoissonRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        #α ~ Normal(0, λ)
        β ~ filldist(Normal(0, λ), p)

        ## link
        #z = α .+ X * β
        z =  X * β
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
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::Laplace, h::Float64 = 0.1, sim_size::Int64 = 1000)
```

Fit a Bayesian Poisson Regression model on the input data with a Laplace prior.

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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), Laplace())
┌ Info: Found initial step size
└   ϵ = 0.025
Chains MCMC chain (1000×19×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 1.83 seconds
Compute duration  = 1.83 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5], β[6]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    1.0888    0.5555     0.0176    0.0299   479.8254    1.0001      262.4865
        β[1]   -1.7821    0.2695     0.0085    0.0134   462.2693    0.9996      252.8826
        β[2]    0.1346    0.0613     0.0019    0.0025   677.6548    0.9992      370.7083
        β[3]    1.1312    0.0573     0.0018    0.0025   539.8120    0.9993      295.3020
        β[4]   -0.3032    0.2243     0.0071    0.0080   665.6348    1.0004      364.1328
        β[5]    1.6986    0.0987     0.0031    0.0036   701.9177    0.9991      383.9812
        β[6]    0.3837    0.1743     0.0055    0.0069   605.4188    1.0001      331.1919

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.4487    0.7271    0.9567    1.2873    2.4804
        β[1]   -2.3044   -1.9714   -1.7767   -1.5968   -1.2912
        β[2]    0.0216    0.0939    0.1302    0.1750    0.2612
        β[3]    1.0107    1.0931    1.1321    1.1708    1.2404
        β[4]   -0.7412   -0.4491   -0.2980   -0.1489    0.1149
        β[5]    1.5122    1.6303    1.6944    1.7690    1.8982
        β[6]    0.0486    0.2672    0.3897    0.4973    0.7221
julia> predict(container,sanction)
78-element Vector{Float64}:
    17.600829854957034
    13.353816665957497
    0.7892315194893381
    0.7892315194893381
    0.7892315194893381
    0.7892315194893381
    2.1250050292583618
    0.7892315194893381
    ⋮
    3.149014396126593
    5.759286021571133
    2.7567571295421613
    0.7892315194893381
    23.288584913402246
    1.0177004055294072
    5.759286021571133
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::PoissonRegression,
    prior::Laplace,
    h::Float64 = 0.1,
    sim_size::Int64 = 1000
)
    @model PoissonRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        #α ~ Distributions.Laplace(0, λ)
        β ~ filldist(Distributions.Laplace(0, λ), p)

        ## link
        #z = α .+ X * β
        z =  X * β
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
fit(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Cauchy, h::Float64 = 1.0, sim_size::Int64 = 1000)
```

Fit a Bayesian Poisson Regression model on the input data with a Cauchy prior.

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
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), Cauchy())
┌ Info: Found initial step size
└   ϵ = 0.025
Chains MCMC chain (1000×19×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 1.82 seconds
Compute duration  = 1.82 seconds
parameters        = λ, β[1], β[2], β[3], β[4], β[5], β[6]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    0.8433    0.4191     0.0133    0.0170   651.9481    1.0030      357.2318
        β[1]   -1.7880    0.2531     0.0080    0.0117   440.1548    1.0012      241.1807
        β[2]    0.1374    0.0640     0.0020    0.0027   596.1778    0.9992      326.6728
        β[3]    1.1299    0.0558     0.0018    0.0018   747.2077    1.0009      409.4289
        β[4]   -0.2965    0.2204     0.0070    0.0106   526.3280    1.0032      288.3989
        β[5]    1.7036    0.0973     0.0031    0.0043   600.4246    0.9992      328.9998
        β[6]    0.3928    0.1705     0.0054    0.0068   754.6819    1.0010      413.5243

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.3101    0.5456    0.7598    1.0488    1.8316
        β[1]   -2.3023   -1.9535   -1.7797   -1.6129   -1.3056
        β[2]    0.0282    0.0920    0.1355    0.1794    0.2670
        β[3]    1.0131    1.0925    1.1303    1.1680    1.2368
        β[4]   -0.7701   -0.4356   -0.2757   -0.1448    0.1045
        β[5]    1.5159    1.6388    1.7020    1.7689    1.9046
        β[6]    0.0738    0.2632    0.3881    0.5176    0.7230
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::PoissonRegression,
    prior::Cauchy,
    h::Float64 = 1.0,
    sim_size::Int64 = 1000
)
    @model PoissonRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        #α ~ Distributions.TDist(1) * λ
        β ~ filldist(Distributions.TDist(1) * λ, p)

        ## link
        #z = α .+ X * β
        z = X * β
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
fit(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression, prior::TDist, h::Float64 = 2.0, sim_size::Int64 = 1000)
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
julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), TDist())
┌ Info: Found initial step size
└   ϵ = 0.025
Chains MCMC chain (1000×20×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 0.68 seconds
Compute duration  = 0.68 seconds
parameters        = λ, ν, β[1], β[2], β[3], β[4], β[5], β[6]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           λ    0.9539    0.3968     0.0125    0.0138   724.4593    0.9991     1071.6854
           ν    2.7018    3.4303     0.1085    0.1503   473.2383    1.0014      700.0567
        β[1]   -1.7913    0.2504     0.0079    0.0130   301.7822    0.9992      446.4233
        β[2]    0.1368    0.0650     0.0021    0.0030   512.6922    0.9998      758.4204
        β[3]    1.1322    0.0536     0.0017    0.0026   349.4069    0.9995      516.8741
        β[4]   -0.3148    0.2317     0.0073    0.0080   824.9216    0.9993     1220.2983
        β[5]    1.7001    0.1017     0.0032    0.0039   661.7511    1.0054      978.9218
        β[6]    0.3909    0.1681     0.0053    0.0068   459.0907    0.9994      679.1283

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.3999    0.6705    0.8799    1.1535    1.8898
           ν    0.5494    1.1003    1.7737    2.9372    9.8194
        β[1]   -2.2817   -1.9561   -1.7941   -1.6262   -1.2766
        β[2]    0.0151    0.0891    0.1383    0.1795    0.2614
        β[3]    1.0322    1.0932    1.1310    1.1688    1.2362
        β[4]   -0.7680   -0.4694   -0.3020   -0.1519    0.1128
        β[5]    1.5030    1.6252    1.7037    1.7682    1.8936
        β[6]    0.0701    0.2710    0.3911    0.5023    0.7236
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::PoissonRegression,
    prior::TDist,
    h::Float64 = 2.0,
    sim_size::Int64 = 1000
)
    @model PoissonRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        ν ~ InverseGamma(h, h)
        #α ~ Distributions.TDist(ν) * λ
        β ~ filldist(Distributions.TDist(ν) * λ, p)

        ## link
        #z = α .+ X * β
        z = X * β
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
fit(formula::FormulaTerm,data::DataFrame,modelClass::PoissonRegression,prior::HorseShoe,sim_size::Int64 = 1000)
```

Fit a Bayesian Poisson Regression model on the input data with a Horse Shoe prior.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsModels
julia> sanction = dataset("Zelig", "sanction");
julia> CRRao.set_rng(StableRNG(123))
julia> container = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), HorseShoe())
┌ Info: Found initial step size
└   ϵ = 0.025
Chains MCMC chain (1000×25×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 4.1 seconds
Compute duration  = 4.1 seconds
parameters        = τ, λ[1], λ[2], λ[3], λ[4], λ[5], λ[6], β[1], β[2], β[3], β[4], β[5], β[6]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           τ    1.0968    0.7202     0.0228    0.0309   495.7869    1.0052      120.9531
        λ[1]    3.1027    3.8858     0.1229    0.1974   409.1763    1.0058       99.8234
        λ[2]    0.7001    1.3791     0.0436    0.0493   705.5283    0.9990      172.1220
        λ[3]    2.3944    3.2504     0.1028    0.1437   501.7931    1.0042      122.4184
        λ[4]    0.9352    1.7652     0.0558    0.0791   582.3219    1.0034      142.0644
        λ[5]    3.3768    6.5014     0.2056    0.2544   610.2202    1.0001      148.8705
        λ[6]    1.2451    2.2218     0.0703    0.1224   350.1233    1.0004       85.4168
        β[1]   -1.7490    0.2761     0.0087    0.0128   431.8956    0.9998      105.3661
        β[2]    0.1184    0.0706     0.0022    0.0034   468.3049    0.9991      114.2486
        β[3]    1.1297    0.0571     0.0018    0.0025   564.1507    1.0001      137.6313
        β[4]   -0.2202    0.2117     0.0067    0.0063   733.3563    0.9997      178.9110
        β[5]    1.7059    0.1021     0.0032    0.0035   909.1702    0.9994      221.8029
        β[6]    0.3723    0.1751     0.0055    0.0063   559.1851    0.9990      136.4199

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           τ    0.2712    0.6245    0.8979    1.3762    2.9053
        λ[1]    0.5269    1.2258    2.0465    3.3300   14.0447
        λ[2]    0.0259    0.1688    0.3584    0.7633    3.2174
        λ[3]    0.3468    0.8696    1.4059    2.5462   12.1721
        λ[4]    0.0374    0.2091    0.5065    0.9706    4.2828
        λ[5]    0.4982    1.2214    1.9925    3.4033   14.6181
        λ[6]    0.1048    0.3658    0.6567    1.2087    7.2850
        β[1]   -2.3007   -1.9417   -1.7414   -1.5683   -1.2303
        β[2]   -0.0040    0.0678    0.1152    0.1684    0.2634
        β[3]    1.0235    1.0905    1.1299    1.1665    1.2456
        β[4]   -0.6741   -0.3592   -0.2024   -0.0517    0.1380
        β[5]    1.4992    1.6387    1.7051    1.7718    1.9102
        β[6]    0.0305    0.2485    0.3763    0.4911    0.7002
julia> using StatsPlots
julia> plot(container.chain)
```
"""
function fit(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::PoissonRegression,
    prior::HorseShoe,
    sim_size::Int64 = 1000
)
    @model PoissonRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        
        halfcauchy = Truncated(Distributions.TDist(1), 0, Inf)
        
        τ ~ halfcauchy    ## Global Shrinkage
        λ ~ filldist(halfcauchy, p) ## Local Shrinkage
        #α ~ Normal(0, τ )
        β0 = repeat([0], p)  ## prior mean
        β ~ MvNormal(β0, λ * τ )

        ## link
        #z = α .+ X * β
        z =  X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Poisson(mu[i])
        end
    end

    return poisson_reg(formula, data, PoissonRegression, sim_size)
end