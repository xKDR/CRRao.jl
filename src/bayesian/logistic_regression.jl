# Include logistic_reg definitions
include("regression_models/LogisticRegression.jl")

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