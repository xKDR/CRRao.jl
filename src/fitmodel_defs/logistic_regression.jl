# Include logistic_reg definitions
include("regression_models/LogisticRegression.jl")

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link)
   formula = apply_schema(formula, schema(formula, data))
   model = glm(formula, data, Binomial(), Link)
   return FrequentistRegression(:LogisticRegression, model)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit)
```

Fit a Logistic Regression model on the input data using the Logit link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.

# Arguments
- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset. 
- `modelClass`: Object representing the type of regression, which is Logistic Regression in this case.
- `Link`: A type to specify the link function to be used, which is Logit in this case.

```julia-repl
julia> using CRRao, RDatasets, StatsBase

julia> turnout = dataset("Zelig", "turnout");

julia> container = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit());

julia> coeftable(container.model)
────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  -3.03426    0.325927    -9.31    <1e-19  -3.67307    -2.39546
Age           0.0283543  0.00346034   8.19    <1e-15   0.0215722   0.0351365
Race: white   0.250798   0.146457     1.71    0.0868  -0.0362521   0.537847
Income        0.177112   0.0271516    6.52    <1e-10   0.123896    0.230328
Educate       0.175634   0.0203308    8.64    <1e-17   0.135786    0.215481
────────────────────────────────────────────────────────────────────────────

julia> loglikelihood(container.model)
-1011.9906318515575

julia> aic(container.model)
2033.981263703115

julia> bic(container.model)
2061.9857760008254
```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit)
   return logistic_reg(formula, data, LogitLink())
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit)
```

Fit a Logistic Regression model on the input data using the Probit link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit)
   return logistic_reg(formula, data, ProbitLink())
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog)
```

Fit a Logistic Regression model on the input data using the Cloglog link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.
```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog)
   return logistic_reg(formula, data, CloglogLink())
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit)
```

Fit a Logistic Regression model on the input data using the Cauchit link. Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.
```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Cauchit)
   return logistic_reg(formula, data, CauchitLink())
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