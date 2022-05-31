# Include logistic_reg definitions
include("regression_models/LogisticRegression.jl")

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