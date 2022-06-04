# Include Poisson_Reg definitions
include("regression_models/PoissonRegression.jl")

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