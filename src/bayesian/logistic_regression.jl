function logistic_reg(formula::FormulaTerm, data::DataFrame, turingModel::Function, sim_size::Int64)
    formula = apply_schema(formula, schema(formula, data))
    y, X = modelcols(formula, data)

    if sim_size < 500
        @warn "Simulation size should generally be atleast 500."
    end
    chain = sample(CRRao_rng, turingModel(X, y), NUTS(), sim_size)
    return BayesianRegression(:LogisticRegression, chain, formula)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Ridge, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```

Fit a Bayesian Logistic Regression model on the input data with a Ridge prior with the provided `Link` function.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs
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
julia> container_logit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit(), Prior_Ridge(),)
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 153.07 seconds
Compute duration  = 153.07 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.1338    0.0730     0.0007    0.0010   4607.5094    1.0006       30.0999
        β[1]    0.0051    0.0022     0.0000    0.0000   9318.7672    1.0000       60.8775
        β[2]   -0.0303    0.0956     0.0010    0.0013   5817.9985    1.0002       38.0078
        β[3]    0.1590    0.0252     0.0003    0.0003   6219.4220    0.9999       40.6302
        β[4]    0.0338    0.0116     0.0001    0.0002   5886.4337    1.0002       38.4548

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0551    0.0878    0.1151    0.1571    0.3184
        β[1]    0.0008    0.0036    0.0051    0.0065    0.0095
        β[2]   -0.2316   -0.0896   -0.0259    0.0310    0.1509
        β[3]    0.1108    0.1416    0.1588    0.1755    0.2094
        β[4]    0.0107    0.0260    0.0338    0.0417    0.0563
julia> container_probit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Probit(), Prior_Ridge())
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 177.96 seconds
Compute duration  = 177.96 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.0867    0.0533     0.0005    0.0008   3682.1660    0.9999       20.6909
        β[1]    0.0033    0.0013     0.0000    0.0000   9024.2636    0.9999       50.7092
        β[2]   -0.0162    0.0577     0.0006    0.0007   5712.7441    1.0000       32.1011
        β[3]    0.0902    0.0137     0.0001    0.0002   6239.2706    1.0004       35.0598
        β[4]    0.0220    0.0068     0.0001    0.0001   6004.4582    0.9999       33.7403

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0354    0.0556    0.0733    0.1018    0.2139
        β[1]    0.0007    0.0024    0.0033    0.0041    0.0058
        β[2]   -0.1353   -0.0525   -0.0157    0.0218    0.0958
        β[3]    0.0634    0.0808    0.0901    0.0992    0.1179
        β[4]    0.0086    0.0174    0.0221    0.0265    0.0354
julia> container_cloglog = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cloglog(), Prior_Ridge())
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 94.56 seconds
Compute duration  = 94.56 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse       ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64   Float64   Float64       Float64 

           λ    0.4868    0.0003     0.0000    0.0000   20.6437    1.8077        0.2183
        β[1]   -0.1684    0.0026     0.0000    0.0003   20.2642    2.5746        0.2143
        β[2]    0.4824    0.0008     0.0000    0.0001   20.4744    2.2619        0.2165
        β[3]    0.9618    0.0058     0.0001    0.0006   20.2614    2.5797        0.2143
        β[4]   -0.3887    0.0004     0.0000    0.0000   21.1046    1.7123        0.2232

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.4861    0.4865    0.4869    0.4870    0.4872
        β[1]   -0.1725   -0.1706   -0.1687   -0.1661   -0.1642
        β[2]    0.4813    0.4817    0.4823    0.4831    0.4840
        β[3]    0.9521    0.9566    0.9626    0.9664    0.9707
        β[4]   -0.3892   -0.3890   -0.3888   -0.3882   -0.3878
julia> container_cauchit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cauchit(), Prior_Ridge())
┌ Info: Found initial step size
└   ϵ = 0.003125
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 153.65 seconds
Compute duration  = 153.65 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.1737    0.1033     0.0010    0.0015   3731.6701    0.9999       24.2865
        β[1]    0.0033    0.0024     0.0000    0.0000   8722.9987    1.0000       56.7711
        β[2]   -0.0598    0.1193     0.0012    0.0017   5364.6587    1.0008       34.9143
        β[3]    0.2191    0.0365     0.0004    0.0005   5826.8422    0.9999       37.9223
        β[4]    0.0204    0.0128     0.0001    0.0002   5304.7531    0.9999       34.5245

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0702    0.1130    0.1483    0.2052    0.4066
        β[1]   -0.0013    0.0016    0.0032    0.0049    0.0083
        β[2]   -0.3183   -0.1325   -0.0507    0.0204    0.1532
        β[3]    0.1488    0.1942    0.2187    0.2429    0.2919
        β[4]   -0.0046    0.0117    0.0203    0.0292    0.0459
```
"""
function fitmodel(
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
        α ~ Normal(0, λ)
        β ~ filldist(Normal(0, λ), p)

        z = α .+ X * β

        ## Link Function

        prob = Link.link.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Laplace, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```

Fit a Bayesian Logistic Regression model on the input data with a Laplace prior with the provided `Link` function.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs
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
julia> container_logit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit(), Prior_Laplace())
┌ Info: Found initial step size
└   ϵ = 0.003125
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 143.44 seconds
Compute duration  = 143.44 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.1178    0.0826     0.0008    0.0012   4670.5185    0.9999       32.5619
        β[1]    0.0051    0.0022     0.0000    0.0000   9160.7544    1.0001       63.8669
        β[2]   -0.0228    0.0890     0.0009    0.0013   4963.2154    1.0002       34.6025
        β[3]    0.1628    0.0254     0.0003    0.0004   5795.2458    1.0000       40.4033
        β[4]    0.0321    0.0118     0.0001    0.0002   5366.5589    1.0006       37.4146

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0380    0.0677    0.0958    0.1410    0.3341
        β[1]    0.0007    0.0036    0.0051    0.0066    0.0095
        β[2]   -0.2299   -0.0690   -0.0133    0.0273    0.1522
        β[3]    0.1145    0.1454    0.1624    0.1796    0.2133
        β[4]    0.0090    0.0240    0.0323    0.0400    0.0549
julia> container_probit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Probit(), Prior_Laplace())
┌ Info: Found initial step size
└   ϵ = 0.00078125
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 171.43 seconds
Compute duration  = 171.43 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.0821    0.0551     0.0006    0.0008   4512.1853    1.0003       26.3206
        β[1]    0.0033    0.0013     0.0000    0.0000   8915.4805    0.9999       52.0059
        β[2]   -0.0138    0.0553     0.0006    0.0008   5240.6484    1.0000       30.5698
        β[3]    0.0916    0.0141     0.0001    0.0002   6402.4324    1.0001       37.3468
        β[4]    0.0212    0.0070     0.0001    0.0001   5508.5643    1.0000       32.1326

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0275    0.0477    0.0667    0.0988    0.2270
        β[1]    0.0008    0.0024    0.0033    0.0042    0.0059
        β[2]   -0.1346   -0.0444   -0.0088    0.0192    0.0907
        β[3]    0.0641    0.0820    0.0913    0.1011    0.1195
        β[4]    0.0074    0.0165    0.0213    0.0260    0.0349
julia> container_cloglog = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cloglog(), Prior_Laplace())
┌ Info: Found initial step size
└   ϵ = 0.0015625
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 194.12 seconds
Compute duration  = 194.12 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.0731    0.0509     0.0005    0.0007   4981.1484    1.0002       25.6608
        β[1]    0.0008    0.0012     0.0000    0.0000   9615.6483    1.0003       49.5358
        β[2]   -0.0266    0.0521     0.0005    0.0007   4812.7260    1.0001       24.7932
        β[3]    0.0759    0.0114     0.0001    0.0002   5448.6076    0.9999       28.0690
        β[4]    0.0069    0.0060     0.0001    0.0001   4591.7360    0.9999       23.6547

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0242    0.0421    0.0600    0.0873    0.2035
        β[1]   -0.0015    0.0000    0.0008    0.0016    0.0031
        β[2]   -0.1478   -0.0559   -0.0199    0.0063    0.0647
        β[3]    0.0538    0.0682    0.0760    0.0836    0.0983
        β[4]   -0.0045    0.0027    0.0068    0.0111    0.0188
julia> container_cauchit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cauchit(), Prior_Laplace())
┌ Info: Found initial step size
└   ϵ = 0.0330078125
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 151.32 seconds
Compute duration  = 151.32 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.1426    0.1158     0.0012    0.0019   4085.6946    1.0002       27.0013
        β[1]    0.0032    0.0024     0.0000    0.0000   6944.6444    1.0001       45.8953
        β[2]   -0.0474    0.1145     0.0011    0.0016   4859.5428    1.0001       32.1154
        β[3]    0.2237    0.0363     0.0004    0.0005   4613.9690    0.9999       30.4925
        β[4]    0.0185    0.0126     0.0001    0.0002   4725.1966    0.9999       31.2275

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0446    0.0790    0.1127    0.1686    0.4193
        β[1]   -0.0013    0.0016    0.0032    0.0047    0.0081
        β[2]   -0.3124   -0.1050   -0.0278    0.0204    0.1493
        β[3]    0.1551    0.1986    0.2228    0.2472    0.2979
        β[4]   -0.0055    0.0097    0.0182    0.0269    0.0439
```
"""
function fitmodel(
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
        α ~ Normal(0, λ)
        β ~ filldist(Laplace(0, λ), p)

        z = α .+ X * β

        ## Link Function

        prob = Link.link.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Cauchy, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 1000)
```

Fit a Bayesian Logistic Regression model on the input data with a Cauchy prior with the provided `Link` function.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs
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
julia> container_logit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit(), Prior_Cauchy())
┌ Info: Found initial step size
└   ϵ = 0.003125
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 152.61 seconds
Compute duration  = 152.61 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.0792    0.0857     0.0009    0.0012   4663.0871    0.9999       30.5560
        β[1]    0.0052    0.0022     0.0000    0.0000   7200.5641    0.9999       47.1834
        β[2]   -0.0205    0.0797     0.0008    0.0013   4355.2582    0.9999       28.5389
        β[3]    0.1653    0.0256     0.0003    0.0003   4895.3528    1.0004       32.0780
        β[4]    0.0306    0.0117     0.0001    0.0002   3982.8457    1.0001       26.0985

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0092    0.0297    0.0545    0.0981    0.2908
        β[1]    0.0010    0.0038    0.0052    0.0067    0.0094
        β[2]   -0.2124   -0.0533   -0.0088    0.0194    0.1293
        β[3]    0.1153    0.1481    0.1652    0.1822    0.2164
        β[4]    0.0080    0.0227    0.0304    0.0384    0.0537
julia> container_probit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Probit(), Prior_Cauchy())
┌ Info: Found initial step size
└   ϵ = 0.00078125
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 188.49 seconds
Compute duration  = 188.49 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.0484    0.0512     0.0005    0.0007   5054.3994    1.0000       26.8155
        β[1]    0.0034    0.0013     0.0000    0.0000   8376.0026    0.9999       44.4379
        β[2]   -0.0101    0.0470     0.0005    0.0007   3497.1991    1.0000       18.5540
        β[3]    0.0927    0.0142     0.0001    0.0002   5007.2301    1.0000       26.5652
        β[4]    0.0202    0.0070     0.0001    0.0001   4277.4390    0.9999       22.6934

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0057    0.0185    0.0335    0.0599    0.1824
        β[1]    0.0009    0.0025    0.0034    0.0042    0.0059
        β[2]   -0.1236   -0.0297   -0.0045    0.0135    0.0783
        β[3]    0.0649    0.0830    0.0927    0.1021    0.1207
        β[4]    0.0068    0.0155    0.0202    0.0249    0.0343
julia> container_cloglog = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cloglog(), Prior_Cauchy())
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 121.33 seconds
Compute duration  = 121.33 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse       ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64   Float64   Float64       Float64 

           λ    0.2562    0.0001     0.0000    0.0000   24.5965    1.0444        0.2027
        β[1]    0.0457    0.0000     0.0000    0.0000   30.9386    1.3997        0.2550
        β[2]    0.3084    0.0008     0.0000    0.0001   20.7337    1.6095        0.1709
        β[3]    0.0931    0.0023     0.0000    0.0002   20.2943    2.5468        0.1673
        β[4]   -1.3797    0.0072     0.0001    0.0007   20.2381    2.7801        0.1668

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.2561    0.2562    0.2562    0.2563    0.2564
        β[1]    0.0456    0.0457    0.0457    0.0457    0.0457
        β[2]    0.3069    0.3077    0.3083    0.3089    0.3100
        β[3]    0.0893    0.0912    0.0933    0.0948    0.0971
        β[4]   -1.3910   -1.3863   -1.3800   -1.3733   -1.3681
julia> container_cauchit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cauchit(), Prior_Cauchy())
┌ Info: Found initial step size
└   ϵ = 0.05
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 169.7 seconds
Compute duration  = 169.7 seconds
parameters        = λ, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           λ    0.0759    0.0909     0.0009    0.0014   3905.2459    0.9999       23.0129
        β[1]    0.0033    0.0023     0.0000    0.0000   5465.8674    1.0000       32.2094
        β[2]   -0.0318    0.0969     0.0010    0.0015   4113.9104    0.9999       24.2425
        β[3]    0.2285    0.0364     0.0004    0.0007   2716.3177    1.0006       16.0068
        β[4]    0.0158    0.0124     0.0001    0.0002   3167.8247    1.0000       18.6674

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.0050    0.0211    0.0451    0.0970    0.3174
        β[1]   -0.0011    0.0018    0.0033    0.0048    0.0078
        β[2]   -0.2849   -0.0621   -0.0083    0.0151    0.1277
        β[3]    0.1586    0.2033    0.2276    0.2529    0.3004
        β[4]   -0.0067    0.0069    0.0153    0.0239    0.0415
```
"""
function fitmodel(
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
        α ~ TDist(1) * λ
        β ~ filldist(TDist(1) * λ, p)

        z = α .+ X * β

        ## Link Function

        prob = Link.link.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_TDist, h::Float64 = 1.0, level::Float64 = 0.95, sim_size::Int64 = 1000)
```

Fit a Bayesian Logistic Regression model on the input data with a T-Dist prior with the provided `Link` function.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs
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
julia> container_logit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 0.003125
Chains MCMC chain (10000×18×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 191.64 seconds
Compute duration  = 191.64 seconds
parameters        = λ, ν, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean        std   naive_se      mcse          ess      rhat   ess_per_sec 
      Symbol   Float64    Float64    Float64   Float64      Float64   Float64       Float64 

           λ    0.3043     0.1678     0.0017    0.0022    5712.7538    1.0000       29.8104
           ν   27.2612   549.4979     5.4950    7.5617    5110.0078    1.0002       26.6652
        β[1]    0.0052     0.0023     0.0000    0.0000   11260.6563    1.0004       58.7607
        β[2]   -0.0589     0.1247     0.0012    0.0012    8832.9112    1.0000       46.0921
        β[3]    0.1667     0.0255     0.0003    0.0003    8308.4832    1.0000       43.3555
        β[4]    0.0333     0.0123     0.0001    0.0001    7630.2720    0.9999       39.8165

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.1211    0.1958    0.2627    0.3635    0.7244
           ν    0.5077    1.4402    2.9883    7.2135   79.1950
        β[1]    0.0008    0.0037    0.0052    0.0067    0.0097
        β[2]   -0.3138   -0.1399   -0.0546    0.0244    0.1796
        β[3]    0.1177    0.1491    0.1665    0.1841    0.2177
        β[4]    0.0094    0.0251    0.0334    0.0414    0.0575
julia> container_probit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Probit(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 0.00078125
Chains MCMC chain (10000×18×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 262.37 seconds
Compute duration  = 262.37 seconds
parameters        = λ, ν, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean        std   naive_se      mcse          ess      rhat   ess_per_sec 
      Symbol   Float64    Float64    Float64   Float64      Float64   Float64       Float64 

           λ    0.2694     0.1551     0.0016    0.0020    5091.0519    1.0002       19.4040
           ν   21.1549   329.1679     3.2917    4.5051    5189.9802    1.0000       19.7811
        β[1]    0.0034     0.0013     0.0000    0.0000   12584.4962    0.9999       47.9645
        β[2]   -0.0331     0.0779     0.0008    0.0009    6959.6501    0.9999       26.5260
        β[3]    0.0936     0.0138     0.0001    0.0002    6043.0747    1.0012       23.0326
        β[4]    0.0218     0.0072     0.0001    0.0001    5938.2251    1.0010       22.6329

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.1104    0.1761    0.2314    0.3177    0.6395
           ν    0.4766    1.3862    2.9031    6.9434   79.8040
        β[1]    0.0008    0.0025    0.0034    0.0043    0.0060
        β[2]   -0.1901   -0.0841   -0.0325    0.0184    0.1189
        β[3]    0.0671    0.0843    0.0935    0.1027    0.1210
        β[4]    0.0077    0.0169    0.0217    0.0267    0.0361
julia> container_cloglog = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cloglog(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 0.0015625
Chains MCMC chain (10000×18×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 241.88 seconds
Compute duration  = 241.88 seconds
parameters        = λ, ν, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean        std   naive_se      mcse          ess      rhat   ess_per_sec 
      Symbol   Float64    Float64    Float64   Float64      Float64   Float64       Float64 

           λ    0.2705     0.1490     0.0015    0.0018    6373.8367    1.0001       26.3518
           ν   25.1429   513.9686     5.1397    7.9414    4083.5546    1.0000       16.8829
        β[1]    0.0010     0.0012     0.0000    0.0000   11899.8637    0.9999       49.1984
        β[2]   -0.0562     0.0693     0.0007    0.0009    6611.6159    0.9999       27.3348
        β[3]    0.0774     0.0115     0.0001    0.0001    6350.5188    0.9999       26.2554
        β[4]    0.0081     0.0066     0.0001    0.0001    5974.2918    1.0000       24.6999

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.1079    0.1732    0.2318    0.3236    0.6667
           ν    0.4629    1.3747    2.7811    6.9236   88.1070
        β[1]   -0.0014    0.0001    0.0010    0.0018    0.0034
        β[2]   -0.1960   -0.1021   -0.0563   -0.0089    0.0781
        β[3]    0.0549    0.0696    0.0775    0.0851    0.0996
        β[4]   -0.0050    0.0036    0.0081    0.0126    0.0209
julia> container_cauchit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cauchit(), Prior_TDist())
┌ Info: Found initial step size
└   ϵ = 0.009375000000000001
Chains MCMC chain (10000×18×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 224.46 seconds
Compute duration  = 224.46 seconds
parameters        = λ, ν, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters      mean        std   naive_se      mcse          ess      rhat   ess_per_sec 
      Symbol   Float64    Float64    Float64   Float64      Float64   Float64       Float64 

           λ    0.3293     0.1847     0.0018    0.0022    6475.4472    1.0000       28.8487
           ν   16.2524   148.5657     1.4857    2.0404    5495.7033    1.0003       24.4839
        β[1]    0.0036     0.0025     0.0000    0.0000   10948.8161    0.9999       48.7780
        β[2]   -0.1076     0.1519     0.0015    0.0016    7435.6058    1.0000       33.1263
        β[3]    0.2321     0.0361     0.0004    0.0004    7391.3297    0.9999       32.9291
        β[4]    0.0198     0.0133     0.0001    0.0002    6891.4114    1.0000       30.7019

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           λ    0.1305    0.2117    0.2828    0.3905    0.7982
           ν    0.5166    1.4512    2.9337    6.9187   79.9496
        β[1]   -0.0011    0.0019    0.0036    0.0052    0.0087
        β[2]   -0.4347   -0.2049   -0.0983   -0.0031    0.1670
        β[3]    0.1629    0.2075    0.2313    0.2560    0.3057
        β[4]   -0.0058    0.0106    0.0196    0.0287    0.0461
```
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::CRRaoLink,
    prior::Prior_TDist,
    h::Float64 = 1.0,
    level::Float64 = 0.95,
    sim_size::Int64 = 1000
)
    @model LogisticRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        ν ~ InverseGamma(h, h)
        α ~ TDist(ν) * λ
        β ~ filldist(TDist(ν) * λ, p)

        z = α .+ X * β

        ## Link Function

        prob = Link.link.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Uniform, h::Float64 = 0.01, level::Float64 = 0.95, sim_size::Int64 = 1000)
```

Fit a Bayesian Logistic Regression model on the input data with a Uniform prior with the provided `Link` function.

# Example
```julia-repl
julia> using CRRao, RDatasets, StableRNGs
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
julia> container_logit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit(), Prior_Uniform())
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 189.43 seconds
Compute duration  = 189.43 seconds
parameters        = v, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters                                                                                                             ⋯
      Symbol                                                                                                             ⋯

           v   675917680092296823408089342391941239673271191525001008413719254637203390096714276526402253793438558348179 ⋯
        β[1]                                                                                                             ⋯
        β[2]                                                                                                             ⋯
        β[3]                                                                                                             ⋯
        β[4]                                                                                                             ⋯
                                                                                                         7 columns omitted

Quantiles
  parameters      2.5%                25.0%                                  50.0%                                       ⋯
      Symbol   Float64              Float64                                Float64                                       ⋯

           v   10.0068   6671382021570.9727   2875917206819862279706875265024.0000   10765098578457618185304163237787764 ⋯
        β[1]    0.0025               0.0052                                 0.0066                                       ⋯
        β[2]   -0.2792              -0.2792                                -0.2792                                       ⋯
        β[3]    0.1295               0.1615                                 0.1791                                       ⋯
        β[4]    0.0180               0.0327                                 0.0399                                       ⋯
                                                                                                         2 columns omitted
julia> container_probit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Probit(), Prior_Uniform())
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 271.82 seconds
Compute duration  = 271.82 seconds
parameters        = v, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters                                                                                                             ⋯
      Symbol                                                                                                             ⋯

           v   315982558180625088687517999180569534700873081964494331646387645413807923241067841219325555242963795532234 ⋯
        β[1]                                                                                                             ⋯
        β[2]                                                                                                             ⋯
        β[3]                                                                                                             ⋯
        β[4]                                                                                                             ⋯
                                                                                                         7 columns omitted

Quantiles
  parameters      2.5%               25.0%                                 50.0%                                         ⋯
      Symbol   Float64             Float64                               Float64                                         ⋯

           v    2.9888   500027163480.9627   114267416620826600088306450432.0000   8731986448381257740408362721640235131 ⋯
        β[1]    0.0008              0.0025                                0.0033                                         ⋯
        β[2]   -0.2059             -0.0982                               -0.0400                                         ⋯
        β[3]    0.0668              0.0850                                0.0941                                         ⋯
        β[4]    0.0078              0.0169                                0.0220                                         ⋯
                                                                                                         2 columns omitted
julia> container_cloglog = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cloglog(), Prior_Uniform())
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 221.98 seconds
Compute duration  = 221.98 seconds
parameters        = v, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters                                                                                                             ⋯
      Symbol                                                                                                             ⋯

           v   997108327601492157485369149438354840181147942886188650973018774473864498882782024293920677223471145239174 ⋯
        β[1]                                                                                                             ⋯
        β[2]                                                                                                             ⋯
        β[3]                                                                                                             ⋯
        β[4]                                                                                                             ⋯
                                                                                                         7 columns omitted

Quantiles
  parameters      2.5%                25.0%                                50.0%                                         ⋯
      Symbol   Float64              Float64                              Float64                                         ⋯

           v    2.9344   1617934162465.7087   79417083014024744675909304320.0000   6335481071385452850562280409696628675 ⋯
        β[1]   -0.0013               0.0002                               0.0010                                         ⋯
        β[2]   -0.2056              -0.1137                              -0.0639                                         ⋯
        β[3]    0.0555               0.0705                               0.0779                                         ⋯
        β[4]   -0.0042               0.0036                               0.0079                                         ⋯
                                                                                                         2 columns omitted
julia> container_cauchit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cauchit(), Prior_Uniform())
Chains MCMC chain (10000×17×1 Array{Float64, 3}):

Iterations        = 1001:1:11000
Number of chains  = 1
Samples per chain = 10000
Wall duration     = 250.81 seconds
Compute duration  = 250.81 seconds
parameters        = v, β[1], β[2], β[3], β[4]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
  parameters                                                                                                             ⋯
      Symbol                                                                                                             ⋯

           v   145105825023746239211260804740935487372396420958652923206621612953582232010367942920611301369886999826663 ⋯
        β[1]                                                                                                             ⋯
        β[2]                                                                                                             ⋯
        β[3]                                                                                                             ⋯
        β[4]                                                                                                             ⋯
                                                                                                         7 columns omitted

Quantiles
  parameters      2.5%               25.0%                                50.0%                                          ⋯
      Symbol   Float64             Float64                              Float64                                          ⋯

           v   10.8874   371822401390.3905   15665245298723267168052445184.0000   97091064597143721776615147479481412134 ⋯
        β[1]   -0.0009              0.0022                               0.0039                                          ⋯
        β[2]   -0.5461             -0.2897                              -0.1591                                          ⋯
        β[3]    0.1679              0.2138                               0.2371                                          ⋯
        β[4]   -0.0047              0.0122                               0.0212                                          ⋯
                                                                                                         2 columns omitted
```
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::CRRaoLink,
    prior::Prior_Uniform,
    h::Float64 = 0.01,
    level::Float64 = 0.95,
    sim_size::Int64 = 1000
)
    @model LogisticRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        v ~ InverseGamma(h, h)
        α ~ TDist(1)
        β ~ filldist(Uniform(-v, v), p)

        z = α .+ X * β

        ## Link Function

        prob = Link.link.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, LogisticRegression, sim_size)
end
