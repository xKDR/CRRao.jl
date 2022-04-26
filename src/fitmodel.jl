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
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Ridge,h::Float64=0.01,sim_size::Int64=10000)
   ans = linear_reg(formula,data,Prior_Ridge(),h,sim_size);
   ans
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
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Laplace,h::Float64=0.01,sim_size::Int64=10000)
   ans = linear_reg(formula,data,Prior_Laplace(),h,sim_size);
   ans
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
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Cauchy,sim_size::Int64=10000)
   ans = linear_reg(formula,data,Prior_Cauchy(),sim_size);
   ans
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
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_TDist,h::Float64=2.0,sim_size::Int64=10000)
   ans = linear_reg(formula,data,Prior_TDist(),h,sim_size);
   ans
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
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Uniform,h::Float64=0.01,sim_size::Int64=10000)
   ans = linear_reg(formula,data,Prior_TDist(),h,sim_size);
   ans
end


"""
# Logistic Regression with Logit link : GLM with Scoring Method
    
    ## Link Function
    z = X*β

    prob = Logit.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
   ```    
   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                  ,turnout,LogisticRegression(),Logit());

   Julia> model.fit

   ────────────────────────────────────────────────────────────────────────────
                     Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
   ────────────────────────────────────────────────────────────────────────────
   (Intercept)  -3.03426    0.325927    -9.31    <1e-19  -3.67307    -2.39546
   Age           0.0283543  0.00346034   8.19    <1e-15   0.0215722   0.0351365
   Race: white   0.250798   0.146457     1.71    0.0868  -0.0362521   0.537847
   Income        0.177112   0.0271516    6.52    <1e-10   0.123896    0.230328
   Educate       0.175634   0.0203308    8.64    <1e-17   0.135786    0.215481
   ────────────────────────────────────────────────────────────────────────────

   Julia> model.LogLike
   -1011.9906318515576

   Julia> model.AIC
   2033.9812637031152

   Julia> model.BIC
   2061.985776000826
   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Logit)
   ans = logistic_reg(formula,data,"LogitLink");
   ans
end


"""
# Logistic Regression with Probit link : GLM with Scoring Method
    
    ## Link Function
    z = X*β

    prob = Probit.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
   
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                  ,turnout,LogisticRegression(),Probit());
   Julia> model.fit

   ────────────────────────────────────────────────────────────────────────────
                     Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
   ────────────────────────────────────────────────────────────────────────────
   (Intercept)  -1.76141    0.188556    -9.34    <1e-20  -2.13097    -1.39185
   Age           0.0164973  0.00199897   8.25    <1e-15   0.0125794   0.0204152
   Race: white   0.162856   0.0876885    1.86    0.0633  -0.0090108   0.334722
   Income        0.0963117  0.0149675    6.43    <1e-09   0.066976    0.125647
   Educate       0.10417    0.0116713    8.93    <1e-18   0.0812949   0.127046
   ────────────────────────────────────────────────────────────────────────────
   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Probit)
   ans = logistic_reg(formula,data,"ProbitLink");
   ans
end

"""
# Logistic Regression with CloglogLink link : GLM with Scoring Method
    
    ## Link Function
    z = X*β

    prob = CloglogLink.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
   
   ```Julia
   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                  ,turnout,LogisticRegression(),Cloglog());
   Julia> model.fit

   ─────────────────────────────────────────────────────────────────────────────
                     Coef.  Std. Error       z  Pr(>|z|)   Lower 95%   Upper 95%
   ─────────────────────────────────────────────────────────────────────────────
   (Intercept)  -1.94617    0.184123    -10.57    <1e-25  -2.30704    -1.58529
   Age           0.0147857  0.00184088    8.03    <1e-15   0.0111776   0.0183937
   Race: white   0.185139   0.087101      2.13    0.0335   0.014424    0.355854
   Income        0.0768268  0.0126411     6.08    <1e-08   0.0520506   0.101603
   Educate       0.0983976  0.0108857     9.04    <1e-18   0.077062    0.119733
   ─────────────────────────────────────────────────────────────────────────────
   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Cloglog)
   ans = logistic_reg(formula,data,"CloglogLink");
   ans
end

"""
# Logistic Regression with Cauchit link : GLM with Scoring Method
    
    ## Link Function
    z = X*β

    prob = Cauchit.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
    ```Julia
      Julia> turnout = dataset("Zelig", "turnout")

      Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                     ,turnout,LogisticRegression(),Cauchit());
      Julia> model.fit

      ────────────────────────────────────────────────────────────────────────────
                        Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
      ────────────────────────────────────────────────────────────────────────────
      (Intercept)  -3.16889    0.384429    -8.24    <1e-15  -3.92235    -2.41542
      Age           0.0304105  0.00413473   7.35    <1e-12   0.0223066   0.0385144
      Race: white   0.181839   0.144766     1.26    0.2091  -0.101898    0.465576
      Income        0.235267   0.038152     6.17    <1e-09   0.16049     0.310043
      Educate       0.169276   0.0240098    7.05    <1e-11   0.122217    0.216334
      ────────────────────────────────────────────────────────────────────────────
   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Cauchit)
   ans = logistic_reg(formula,data,"CauchitLink");
   ans
end



"""
# Logistic Regression with Ridge prior using HMC method in Turing
    
    ## priors
    λ ~ InverseGamma(h,h)
    β ~ Normal(0,λ)  

    ## Link Function
    z = X*β

    prob = Logit.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                  ,turnout,LogisticRegression(),Cauchit());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Logit,PriorMod::Prior_Ridge, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Logit(),Prior_Ridge(),h,sim_size)
   ans      
end

"""
# Logistic Regression with Ridge prior using HMC method in Turing
    
    ## priors
    λ ~ InverseGamma(h,h)
    β ~ Normal(0,λ)

    ## Link Function
    z = X*β

    prob = Probit.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                  ,turnout,LogisticRegression()
                  ,Probit(),Prior_Ridge());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Probit,PriorMod::Prior_Ridge, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Probit(),Prior_Ridge(),h,sim_size)
   ans      
end

"""
# Logistic Regression with Ridge prior using HMC method in Turing
    
    ## priors
    λ ~ InverseGamma(h,h)
    β ~ Normal(0,λ)  

    ## Link Function
    z = X*β

    prob = Cloglog.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Cloglog()
                           ,Prior_Ridge());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Cloglog,PriorMod::Prior_Ridge, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Cloglog(),Prior_Ridge(),h,sim_size)
   ans      
end

"""
# Logistic Regression with Ridge prior using HMC method in Turing
    
    ## priors
    λ ~ InverseGamma(h,h)
    β ~ Normal(0,λ)  

    ## Link Function
    z = X*β

    prob = Cauchit.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Cauchit()
                           ,Prior_Ridge());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Cauchit,PriorMod::Prior_Ridge, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Cauchit(),Prior_Ridge(),h,sim_size)
   ans      
end

"""
# Logistic Regression with Laplace prior using HMC method in Turing
    
    ## priors
    λ ~ InverseGamma(h,h)
    β ~ Laplace(0,λ)

    ## Link Function
    z = X*β

    prob = Logit.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Logit()
                           ,Prior_Laplace());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit, PriorMod::Prior_Laplace, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Logit(),Prior_Laplace(),h,sim_size)
   ans
end

"""
# Logistic Regression with Laplace prior using HMC method in Turing
    
    ## priors
    λ ~ InverseGamma(h,h)
    β ~ Laplace(0,λ)

    ## Link Function
    z = X*β

    prob = Probit.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Probit()
                           ,Prior_Laplace());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit, PriorMod::Prior_Laplace, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Probit(),Prior_Laplace(),h,sim_size)
   ans
end

"""
# Logistic Regression with Laplace prior using HMC method in Turing
    
    ## priors
    λ ~ InverseGamma(h,h)
    β ~ Laplace(0,λ)

    ## Link Function
    z = X*β

    prob = Cloglog.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Cloglog()
                           ,Prior_Laplace());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog, PriorMod::Prior_Laplace, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Cloglog(),Prior_Laplace(),h,sim_size)
   ans
end

"""
# Logistic Regression with Laplace prior using HMC method in Turing
    
    ## priors
    λ ~ InverseGamma(h,h)
    β ~ Laplace(0,λ)

    ## Link Function
    z = X*β

    prob = Cauchit.(z)

    #likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Cauchit()
                           ,Prior_Laplace());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit, PriorMod::Prior_Laplace, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Cauchit(),Prior_Laplace(),h,sim_size)
   ans
end



"""
# Logistic Regression with Cauchy prior using HMC method in Turing
    
   ## priors
    
    λ   ~ Truncated(TDist(1),0,Inf) # Half-Cauchy prior
    β|λ ~ λ*TDist(1)  # Cauchy prior

    ## Link Function
    z = X*β

    prob = Logit.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Logit()
                           ,Prior_Cauchy());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit, PriorMod::Prior_Cauchy, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Logit(),Prior_Cauchy(),h,sim_size)
   ans
end

"""
# Logistic Regression with Cauchy prior using HMC method in Turing
    
   ## priors
    
    λ   ~ Truncated(TDist(1),0,Inf) # Half-Cauchy prior
    β|λ ~ λ*TDist(1)  # Cauchy prior

    ## Link Function
    z = X*β

    prob = Probit.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Probit()
                           ,Prior_Cauchy());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit, PriorMod::Prior_Cauchy, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Probit(),Prior_Cauchy(),h,sim_size)
   ans
end

"""
# Logistic Regression with Cauchy prior using HMC method in Turing
    
   ## priors
    
    λ   ~ Truncated(TDist(1),0,Inf) # Half-Cauchy prior
    β|λ ~ λ*TDist(1)  # Cauchy prior

    ## Link Function
    z = X*β

    prob = Cloglog.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Cloglog()
                           ,Prior_Cauchy());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog, PriorMod::Prior_Cauchy, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Cloglog(),Prior_Cauchy(),h,sim_size)
   ans
end

"""
# Logistic Regression with Cauchy prior using HMC method in Turing
    
   ## priors
    
    λ   ~ Truncated(TDist(1),0,Inf) # Half-Cauchy prior
    β|λ ~ λ*TDist(1)  # Cauchy prior

    ## Link Function
    z = X*β

    prob = Cauchit.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Cauchit()
                           ,Prior_Cauchy());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit, PriorMod::Prior_Cauchy, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Cauchit(),Prior_Cauchy(),h,sim_size)
   ans
end


"""
# Logistic Regression with T-Dist prior using HMC method in Turing
    
   ## priors
    
    λ ~ InverseGamma(h,h)
    ν ~ InverseGamma(h,h)
    β|λ,ν ~ λ*TDist(ν)  

    ## Link Function
    z = X*β

    prob = Logit.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Logit()
                           ,Prior_TDist());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit, PriorMod::Prior_TDist, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Logit(),Prior_TDist(),h,sim_size)
   ans
end

"""
# Logistic Regression with T-Dist prior using HMC method in Turing
    
   ## priors
    
    λ ~ InverseGamma(h,h)
    ν ~ InverseGamma(h,h)
    β|λ,ν ~ λ*TDist(ν)  

    ## Link Function
    z = X*β

    prob = Probit.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Probit()
                           ,Prior_TDist());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit, PriorMod::Prior_TDist, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Probit(),Prior_TDist(),h,sim_size)
   ans
end

"""
# Logistic Regression with T-Dist prior using HMC method in Turing
    
   ## priors
    
    λ ~ InverseGamma(h,h)
    ν ~ InverseGamma(h,h)
    β|λ,ν ~ λ*TDist(ν)  

    ## Link Function
    z = X*β

    prob = Cloglog.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Cloglog()
                           ,Prior_TDist());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog, PriorMod::Prior_TDist, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Cloglog(),Prior_TDist(),h,sim_size)
   ans
end

"""
# Logistic Regression with T-Dist prior using HMC method in Turing
    
   ## priors
    
    λ ~ InverseGamma(h,h)
    ν ~ InverseGamma(h,h)
    β|λ,ν ~ λ*TDist(ν)  

    ## Link Function
    z = X*β

    prob = Cauchit.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Cauchit()
                           ,Prior_TDist());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit, PriorMod::Prior_TDist, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Cauchit(),Prior_TDist(),h,sim_size)
   ans
end

"""
# Logistic Regression with Uniform prior using HMC method in Turing
    
   ## priors
    v ~ InverseGamma(h,h)
    β ~ filldist(Uniform(-v,v), p) 
   
    ## Link Function
    z = X*β

    prob = Cauchit.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Logit()
                           ,Prior_Uniform());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit, PriorMod::Prior_Uniform, h::Real=0.01, level::Real=0.95, sim_size::Int64=10000)
  ans = logistic_reg(formula,data,Logit(),Prior_Uniform(),h,sim_size)
  ans     
end

"""
# Logistic Regression with Uniform prior using HMC method in Turing
    
   ## priors
    v ~ InverseGamma(h,h)
    β ~ filldist(Uniform(-v,v), p) 
   
    ## Link Function
    z = X*β

    prob = Probit.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Probit()
                           ,Prior_Uniform());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit, PriorMod::Prior_Uniform, h::Real=0.01, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Probit(),Prior_Uniform(),h,sim_size)
   ans     
end

"""
# Logistic Regression with Uniform prior using HMC method in Turing
    
   ## priors
    v ~ InverseGamma(h,h)
    β ~ filldist(Uniform(-v,v), p) 
   
    ## Link Function
    z = X*β

    prob = Cloglog.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Cloglog()
                           ,Prior_Uniform());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog, PriorMod::Prior_Uniform, h::Real=0.01, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Cloglog(),Prior_Uniform(),h,sim_size)
   ans     
end


"""
# Logistic Regression with Uniform prior using HMC method in Turing
    
   ## priors
    v ~ InverseGamma(h,h)
    β ~ filldist(Uniform(-v,v), p) 
   
    ## Link Function
    z = X*β

    prob = Cauchit.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Bernoulli(prob[i])
    end
    
   ```Julia

   Julia> turnout = dataset("Zelig", "turnout")

   Julia> model = @fitmodel(Vote ~ Age + Race +Income + Educate
                           ,turnout
                           ,LogisticRegression()
                           ,Cauchit()
                           ,Prior_Uniform());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit, PriorMod::Prior_Uniform, h::Real=0.01, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_reg(formula,data,Cauchit(),Prior_Uniform(),h,sim_size)
   ans     
end


"""
# Poisson Regression - Likelihood Method
    
    ## Link Function
    z = X*β

    μ = exp.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Poisson(μ[i])
    end
    
   ```Julia

   Julia> sanction = dataset("Zelig", "sanction");

   Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                           , sanction
                           , PoissonRegression());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression)
   ans = Poisson_Reg(formula,data)
   ans     
end


"""
# Poisson Regression with Ridge Prior using HMC with Turing
    
    ## priors
    λ ~ InverseGamma(h,h)
    α|λ ~ λ*Normal(0,λ)
    β|λ ~ λ*Normal(0,λ)

    ## Link Function
    z = X*β

    μ = exp.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Poisson(μ[i])
    end
    
   ```Julia

   Julia> sanction = dataset("Zelig", "sanction");

   Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                           , sanction
                           , PoissonRegression()
                           , Prior_Ridge());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression,PriorMod::Prior_Ridge,h::Float64=0.1,sim_size::Int64=10000)
   ans = Poisson_Reg(formula,data,Prior_Ridge(),h,sim_size)
   ans
end

  
"""
# Poisson Regression with Laplace Prior using HMC with Turing
    
    ## priors
    λ ~ InverseGamma(h,h)
    α|λ ~ λ * Laplace(0,λ)
    β|λ ~ λ * Laplace(0,λ)

    ## Link Function
    z = X*β

    μ = exp.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Poisson(μ[i])
    end
    
   ```Julia

   Julia> sanction = dataset("Zelig", "sanction");

   Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                           , sanction
                           , PoissonRegression()
                           , Prior_Laplace());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression,PriorMod::Prior_Laplace,h::Float64=0.1,sim_size::Int64=10000)
   ans = Poisson_Reg(formula,data,Prior_Laplace(),h,sim_size)
   ans
end

"""
# Poisson Regression with Cauchy Prior using HMC with Turing
    
    ## priors
    λ~InverseGamma(h,h)
    α ~ TDist(1)*λ
    β ~ filldist(TDist(1)*λ, p)  
  
    ## Link Function
    z = X*β

    μ = exp.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Poisson(μ[i])
    end
    
   ```Julia

   Julia> sanction = dataset("Zelig", "sanction");

   Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                           , sanction
                           , PoissonRegression()
                           , Prior_Cauchy());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression,PriorMod::Prior_Cauchy,h::Float64=1.0,sim_size::Int64=10000)
   ans = Poisson_Reg(formula,data,Prior_Cauchy(),h,sim_size)
   ans
end

"""
# Poisson Regression with TDist Prior using HMC with Turing
    
    ## priors
    λ~InverseGamma(h,h)
    α|ν ~ TDist(ν)*λ
    β|ν ~ TDist(ν)*λ 
  
    ## Link Function
    z = X*β

    μ = exp.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Poisson(μ[i])
    end
    
   ```Julia

   Julia> sanction = dataset("Zelig", "sanction");

   Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                           , sanction
                           , PoissonRegression()
                           , Prior_TDist());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression,PriorMod::Prior_TDist,h::Float64=2.0,sim_size::Int64=10000)
   ans = Poisson_Reg(formula,data,Prior_TDist(),h,sim_size)
   ans
end


"""
# Poisson Regression with Uniform Prior using HMC with Turing
    
    ## priors
    λ~InverseGamma(h,h)
    α ~ Uniform(-λ,λ)
    β ~ Uniform(-λ,λ)
  
    ## Link Function
    z = X*β

    μ = exp.(z)

    ## likelihood
    for i = 1:n
        y[i] ~ Poisson(μ[i])
    end
    
   ```Julia

   Julia> sanction = dataset("Zelig", "sanction");

   Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                           , sanction
                           , PoissonRegression()
                           , Prior_Uniform());

   ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression,PriorMod::Prior_Uniform,h::Float64=1.0,sim_size::Int64=10000)
   ans = Poisson_Reg(formula,data,Prior_Uniform(),h,sim_size)
   ans
end



"""
## NegBinomial Regression with Ridge Prior

    #priors
    λ ~ InverseGamma(h,h)
    α|λ ~ Normal(0,λ)
    β|λ ~ Normal(0,λ)
  
    ## link
    z = α .+ X * β
    mu = exp.(z)

    #likelihood
    for i = 1:n
      y[i] ~ NegativeBinomial2(mu[i],λ)
    end

    h::Float64 = 0.1 (default)
    sim_size::Int64 = 10000 (default)
    ```Julia
    Julia> using RDatasets
    Julia> sanction = dataset("Zelig", "sanction");
    Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                            , sanction
                            , NegBinomRegression()
                            , Prior_Ridge());

    ┌ Info: Found initial step size
    └   ϵ = 0.025

    Julia> model.summaries
    Summary Statistics
    parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
        Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

            λ    2.0350    0.4392     0.0044    0.0052   7933.2864    1.0000      639.1111
            α   -1.0802    0.5194     0.0052    0.0082   3565.0359    1.0007      287.2018
            β[1]   -0.0053    0.1637     0.0016    0.0022   5160.0061    1.0003      415.6937
            β[2]    1.0617    0.1320     0.0013    0.0017   5259.8195    1.0001      423.7348
            β[3]   -0.1775    0.5511     0.0055    0.0057   7967.5892    0.9999      641.8746
            β[4]    1.2770    0.3160     0.0032    0.0038   7812.4049    0.9999      629.3728
            β[5]    0.1518    0.2839     0.0028    0.0036   6062.4239    1.0006      488.3931

    ```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression,PriorMod::Prior_Ridge,h::Float64=0.1,sim_size::Int64=10000)
    ans = NegBinom_Reg(formula,data,Prior_Ridge(),h,sim_size)
    ans
 end


 """
 ## NegBinomial Regression with Laplace Prior
 
     #priors
     λ ~ InverseGamma(h,h)
     α|λ ~ Laplace(0,λ)
     β|λ ~ Laplace(0,λ)
   
     ## link
     z = α .+ X * β
     mu = exp.(z)
 
     #likelihood
     for i = 1:n
       y[i] ~ NegativeBinomial2(mu[i],λ)
     end
 
     h::Float64 = 0.1 (default)
     sim_size::Int64 = 10000 (default)

    ```Julia

    Julia> using RDatasets
    Julia> sanction = dataset("Zelig", "sanction");
    Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                            , sanction
                            , NegBinomRegression()
                            , Prior_Laplace());

    ```
 
 """
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression,PriorMod::Prior_Laplace,h::Float64=0.1,sim_size::Int64=10000)
    ans = NegBinom_Reg(formula,data,Prior_Laplace(),h,sim_size)
    ans
end


"""
 ## NegBinomial Regression with Cauchy Prior
 
     #priors
     λ~InverseGamma(h,h)
     α|λ ~ λ*TDist(1)
     β|λ ~ λ*TDist(1)

     ## link
     z = α .+ X * β
     mu = exp.(z)
 
     #likelihood
     for i = 1:n
       y[i] ~ NegativeBinomial2(mu[i],λ)
     end
 
     h::Float64 = 1.0 (default)
     sim_size::Int64 = 10000 (default)

    ```Julia

    Julia> using RDatasets
    Julia> sanction = dataset("Zelig", "sanction");
    Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                            , sanction
                            , NegBinomRegression()
                            , Prior_Cauchy());

    ```
 
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression,PriorMod::Prior_Cauchy,h::Float64=1.0,sim_size::Int64=10000)
    ans = NegBinom_Reg(formula,data,Prior_Cauchy(),h,sim_size)
    ans
end

"""
 ## NegBinomial Regression with TDist Prior
 
     #priors
     λ ~ InverseGamma(h,h)
     ν ~ InverseGamma(h,h)
     α|λ,μ ~ λ*TDist(ν)
     β|λ,μ ~ λ*TDist(ν)

     ## link
     z = α .+ X * β
     mu = exp.(z)
 
     #likelihood
     for i = 1:n
       y[i] ~ NegativeBinomial2(mu[i],λ)
     end
 
     h::Float64 = 1.0 (default)
     sim_size::Int64 = 10000 (default)

    ```Julia

    Julia> using RDatasets
    Julia> sanction = dataset("Zelig", "sanction");
    Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                            , sanction
                            , NegBinomRegression()
                            , Prior_TDist());

    ```
 
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression,PriorMod::Prior_TDist,h::Float64=1.0,sim_size::Int64=10000)
    ans = NegBinom_Reg(formula,data,Prior_TDist(),h,sim_size)
    ans
end

"""
## NegBinomial Regression with Uniform Prior

    Ibrahim and Laud (JASA, 1990) showed that the uniform flat priors 
    for GLM's can lead to improper posterior distributions thus making 
    them undesirable. In such cases, the Markov Chain struggles to converge. 
    Even if it converges, results are unreliable.
 
     #priors
     λ~InverseGamma(h,h)
     α|λ ~ Uniform(-λ,λ)
     β|λ ~ Uniform(-λ,λ)
    

     ## link
     z = α .+ X * β
     mu = exp.(z)
 
     #likelihood
     for i = 1:n
       y[i] ~ NegativeBinomial2(mu[i],λ)
     end
 
     h::Float64 = 1.0 (default)
     sim_size::Int64 = 10000 (default)

    ```Julia

    Julia> using RDatasets
    Julia> sanction = dataset("Zelig", "sanction");
    Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                            , sanction
                            , NegBinomRegression()
                            , Prior_Uniform());

    ```
 
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression,PriorMod::Prior_Uniform,h::Float64=0.1,sim_size::Int64=10000)
    ans = NegBinom_Reg(formula,data,Prior_Uniform(),h,sim_size)
    ans
end

"""
## NegBinomial Regression - Likelihood method

     ## link
     z = α .+ X * β
     mu = exp.(z)
 
     #likelihood
     for i = 1:n
       y[i] ~ NegativeBinomial2(mu[i],λ)
     end
 
    ```Julia

    Julia> using RDatasets
    Julia> sanction = dataset("Zelig", "sanction");
    Julia> model = @fitmodel(Num ~ Target + Coop + NCost
                            , sanction
                            , NegBinomRegression());

    ```
 
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression)
    ans = NegBinom_Reg(formula,data,"LogLink")
    ans
end

"""
```julia
   @fitmodel(formula, args...)
```

Macro for calling `fitmodel` without using `@formula` to create the formula. 

# Example
```julia
using CRRao, RDatasets
sanction = dataset("Zelig", "sanction")
model = @fitmodel(Num ~ Target + Coop + NCost, sanction, NegBinomRegression())
```
"""
macro fitmodel(formula, args...)
   quote
      fitmodel(@formula($formula), $(map(esc, args)...))
   end
end
