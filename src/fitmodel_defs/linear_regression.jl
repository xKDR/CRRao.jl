# Include linear_reg definitions
include("regression_models/LinearRegression.jl")

"""
# Fit OLS Linear Regression with fitmodel

   + To fit linear regression with likelihood method, provide the following
   three information in `fitmodel`


   1.  `formula` Provide the equation. Eg. `@formula(y~x1+x2+...)`
   2.  `data` Provide training data as `DataFrame``
   3.  `modelClass`  : LinearRegression()

   ```Julia

   Julia> using RDatasets, StatsModels, StatsPlots

   Julia> df = dataset("datasets", "mtcars");

   Julia> model = @fitmodel(MPG ~ HP + WT+Gear,df,LinearRegression());

   Julia> model.fit

   ────────────────────────────────────────────────────────────────────────────
                     Coef.  Std. Error      t  Pr(>|t|)   Lower 95%   Upper 95%
   ────────────────────────────────────────────────────────────────────────────
   (Intercept)  32.0137     4.63226      6.91    <1e-06  22.5249     41.5024
   HP           -0.0367861  0.00989146  -3.72    0.0009  -0.0570478  -0.0165243
   WT           -3.19781    0.846546    -3.78    0.0008  -4.93188    -1.46374
   Gear          1.01998    0.851408     1.20    0.2410  -0.72405     2.76401
   ────────────────────────────────────────────────────────────────────────────

   ## Fitted model returns following informations

   Julia> model.sigma
   2.5741691724978977

   Julia> model.LogLike
   -73.52638935960971

   Julia> model.AIC
   157.05277871921942

   Julia> model.BIC
   164.38145823321804

   Julia> model.R_sqr
   0.8352309600685555

   Julia> model.Adjusted_R_sqr
   0.8175771343616149

   Julia> model.fittedResponse
   32-element Vector{Float64}:
   23.66884995233871
   22.85340824320635
   25.25355614074087
   20.746171762311327
   17.635570543830113
   ...

   Julia> model.residuals
   32-element Vector{Float64}:
   -2.668849952338711
   -1.8534082432063492
   -2.4535561407408686
   0.6538282376886713
   ...

   Julia> model.Cooks_distance
   32-element Vector{Float64}:
   0.013342034282302845
   0.00688728266731234
   0.015495847517058972
   ...

   Julia> plot(model.Cooks_distance)
   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression)
   ans = linear_reg(formula,data);
   ans
end

"""
# Fit Bayesian Linear Regression with Ridge Prior with fitmodel

   # Prior
      v ~ InverseGamma(h,h)
      σ ~ InverseGamma(a0,b0)
      α|σ ~ Normal(0,v*σ)
      β|σ ~ Normal(0,v*σ)

      a0=b0=0.1
      h=0.01 (Default is 0.01, user can choose other values)

   # Data model /likelihood
      y ~ MvNormal(α .+ X * β, σ)


   Julia> model = @fitmodel(MPG ~ HP + WT+Gear,df,LinearRegression(),Prior_Ridge());

   ┌ Info: Found initial step size
   └   ϵ = 0.003125

   Julia> model.summaries
   Summary Statistics
   parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
         Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

            v    6.8184    3.8850     0.0388    0.0680   3316.5165    1.0003      266.6868
            σ    2.6816    0.3879     0.0039    0.0064   3526.9739    1.0003      283.6100
            α   28.4108    5.5547     0.0555    0.1075   2390.7583    1.0009      192.2450
         β[1]   -0.0402    0.0107     0.0001    0.0002   3503.1094    1.0008      281.6910
         β[2]   -2.6507    0.9780     0.0098    0.0189   2486.6245    1.0008      199.9537
         β[3]    1.6426    1.0086     0.0101    0.0189   2522.2853    1.0011      202.8213

   ## All rhat values are close to 1; indicates convergence of Markov Chain.
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Ridge, h::Float64=0.01, sim_size::Int64=10000)
   # Defining the Turing model
   @model LinearRegression(X, y) = begin
      p = size(X, 2);
  
      #priors
      a0=0.1
      b0=0.1
      
      v ~ InverseGamma(h,h)
      σ ~ InverseGamma(a0,b0)
      α ~ Normal(0,v*σ)
      β ~ filldist(Normal(0,v*σ), p)
  
      #likelihood
      y ~ MvNormal(α .+ X * β, σ)
   end

   return linear_reg(formula, data, LinearRegression, sim_size);
end

"""
# Fit Bayesian Linear Regression with Laplace Prior with fitmodel

   # Prior
      v ~ InverseGamma(h,h)
      σ ~ InverseGamma(a0,b0)
      α|σ ~ Laplace(0,v*σ)
      β|σ ~ Laplace(0,v*σ)

      a0=b0=0.1
      h=0.01 (Default is 0.01, user can choose other values)

   # Data model /likelihood
      y ~ MvNormal(α .+ X * β, σ)

   # h is 0.01 as Default
   # sim_size is 10000

   Julia> model = @fitmodel(MPG ~ HP + WT+Gear,df,LinearRegression(),Prior_Laplace());

   Alternative:

   Julia> model = @fitmodel(MPG ~ HP + WT+Gear,df,LinearRegression(),Prior_Laplace(),0.01,10000);

   ┌ Warning: The current proposal will be rejected due to numerical error(s).
   │   isfinite.((θ, r, ℓπ, ℓκ)) = (true, false, false, false)
   └ @ AdvancedHMC ~/.julia/packages/AdvancedHMC/w90s5/src/hamiltonian.jl:47
   ┌ Info: Found initial step size
   └   ϵ = 0.0015625

   Julia> model.summaries
   Summary Statistics
   parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
         Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

            v    4.3206    3.3165     0.0332    0.0567   3115.4478    1.0008      354.1489
            σ    2.6659    0.3791     0.0038    0.0063   4303.1480    0.9999      489.1609
            α   29.1723    5.0562     0.0506    0.0843   2889.0590    1.0000      328.4141
         β[1]   -0.0395    0.0104     0.0001    0.0002   4278.0975    0.9999      486.3132
         β[2]   -2.7379    0.9098     0.0091    0.0156   2979.1114    0.9999      338.6508
         β[3]    1.4906    0.9176     0.0092    0.0141   3140.5300    1.0000      357.0001

   ## All rhat values are close to 1; indicates convergence of Markov Chain.
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Laplace, h::Float64=0.01, sim_size::Int64=10000)
   # Defining the Turing model
   @model LinearRegression(X, y) = begin
      p=size(X, 2);
  
      #priors
      a0=0.1
      b0=0.1
      v ~ InverseGamma(h,h)
      σ ~ InverseGamma(a0,b0)
      α ~ Laplace(0,σ*v)
      β ~ filldist(Laplace(0,σ*v), p)
  
      #likelihood
      y ~ MvNormal(α .+ X * β, σ);
   end

   return linear_reg(formula, data, LinearRegression, sim_size);
end

"""
# Fit Bayesian Linear Regression with Cauchy Prior with fitmodel

   #priors
      σ ~ Truncated(TDist(1),0,Inf)
      α|σ ~ σ*TDist(1)
      β|σ ~ σ*TDist(1)

   1. t-distribution with 1 df follows Cauchy distribution
   2. On σ we elicite Half-Cauchy distribution (see Gelman 2006)
      
   # Data model /likelihood
      y ~ MvNormal(α .+ X * β, σ)

   # sim_size is 10000 (default). In the following example we considered 20000.


   Julia> model = @fitmodel(MPG ~ HP + WT+Gear,df,LinearRegression(),Prior_Cauchy(),20000);

   ┌ Info: Found initial step size
   └   ϵ = 0.00078125

   Julia> model.summaries
   Summary Statistics
   parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
         Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

            σ    2.5919    0.3439     0.0024    0.0034   9909.9764    1.0000      771.7449
            α   30.2828    4.6736     0.0330    0.0677   4769.9124    1.0001      371.4596
         β[1]   -0.0395    0.0101     0.0001    0.0001   7610.6711    1.0000      592.6852
         β[2]   -2.8297    0.8653     0.0061    0.0119   5136.0610    1.0000      399.9736
         β[3]    1.2719    0.8446     0.0060    0.0117   5086.2652    1.0001      396.0957

   # All rhat values are close to 1; indicates convergence of Markov Chain.
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Cauchy, h::Float64=0.01, sim_size::Int64=10000)
   # Defining the Turing model
   @model LinearRegression(X, y) = begin
      p=size(X, 2);
  
      #priors
      σ ~ Truncated(TDist(1),0,Inf)
      α ~ TDist(1)*σ
      β ~ filldist(TDist(1)*σ, p)
  
      #likelihood
      y ~ MvNormal(α .+ X * β, σ);
   end

   return linear_reg(formula, data, LinearRegression, sim_size);
end

"""
# Fit Bayesian Linear Regression with t(ν) distributed Prior with fitmodel

   #priors
      ν ~ InverseGamma(h,h)
      σ ~ InverseGamma(a0,b0)
      α|σ ~ σ*TDist(ν)
      β|σ ~ σ*TDist(ν)

   1. t-distribution with ν df follows Cauchy distribution
   2. On σ we elicite InverseGamma(a0,b0) distribution (see Gelman 2006)
      
   # Data model /likelihood
      y ~ MvNormal(α .+ X * β, σ)

   # sim_size is 10000 (default).

   Julia> model = @fitmodel(MPG ~ HP + WT+Gear,df,LinearRegression(),Prior_TDist());

   ┌ Info: Found initial step size
   └   ϵ = 0.0001953125

   Julia> model.summaries
   Summary Statistics
   parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
         Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

            ν    1.0513    0.5599     0.0056    0.0082   5053.9417    1.0000      366.6794
            σ    2.6200    0.3526     0.0035    0.0050   5835.7064    0.9999      423.3989
            α   30.2533    4.7311     0.0473    0.0918   2619.3110    1.0006      190.0393
         β[1]   -0.0394    0.0100     0.0001    0.0001   4267.6480    1.0002      309.6313
         β[2]   -2.8329    0.8630     0.0086    0.0155   2854.0611    0.9999      207.0711
         β[3]    1.2791    0.8614     0.0086    0.0165   2659.5009    1.0005      192.9552

   # All rhat values are close to 1; indicates convergence of Markov Chain.
   # We estimate the posterior estimates of degrees of freedom is 1.0513.
   So posterior distribution of β is close to Cauchy.
   Hence it is better to estimate the posterior median. Instead of posterior mean.

   Julia> model.quantiles
   Quantiles
   parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
         Symbol   Float64   Float64   Float64   Float64   Float64 

            ν    0.3849    0.6656    0.9170    1.2905    2.4945
            σ    2.0373    2.3724    2.5847    2.8248    3.4096
            α   20.5143   27.2162   30.2498   33.3969   39.3762
         β[1]   -0.0591   -0.0460   -0.0394   -0.0328   -0.0199
         β[2]   -4.5112   -3.4040   -2.8377   -2.2691   -1.0928
         β[3]   -0.3783    0.7007    1.2674    1.8364    3.0627


   ## Check if trace-plots are stationary !
   plot(model.chain)

"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_TDist, h::Float64=0.01, sim_size::Int64=10000)
   # Defining the Turing model
   @model LinearRegression(X, y) = begin
      p=size(X, 2);
  
      #priors
      a0=0.1
      b0=0.1
      ν ~ InverseGamma(h,h)
      σ ~ InverseGamma(a0,b0)
      α ~ TDist(ν)*σ
      β ~ filldist(TDist(ν)*σ, p)
  
      #likelihood
      y ~ MvNormal(α .+ X * β, σ);s
   end

   return linear_reg(formula, data, LinearRegression, sim_size);
end

"""
# Fit Bayesian Linear Regression with Uniform Prior
# distributed Prior with fitmodel

   #priors
   v=1/h;
   σ ~ Uniform(0,v)
   α ~ Uniform(-v*σ,v*σ)
   β ~ filldist(Uniform(-v,v), predictors)
      
   # Data model /likelihood
      y ~ MvNormal(α .+ X * β, σ)

   # sim_size is 10000 (default).

   Julia> model = @fitmodel(MPG ~ HP + WT+Gear,df
                           ,LinearRegression()
                           ,Prior_Uniform());

   ## Check if trace-plots are stationary !
   Julia> plot(model.chain)

"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Uniform, h::Float64=0.01, sim_size::Int64=10000)
   # Defining the Turing model
   @model LinearRegression(X, y) = begin
      p=size(X, 2);
  
      #priors
      v=1/h;
      σ ~ Uniform(0,v)
      α ~ Uniform(-v*σ,v*σ)
      β ~ filldist(Uniform(-v,v), predictors)
      
      #likelihood
      y ~ MvNormal(α .+ X * β, σ);
   end

   return linear_reg(formula, data, LinearRegression, sim_size);
end