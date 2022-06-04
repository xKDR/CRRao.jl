# Include Poisson_Reg definitions
include("regression_models/PoissonRegression.jl")

function poisson_reg(formula::FormulaTerm, data::DataFrame, Link::GLM.Link)
    formula = apply_schema(formula, schema(formula, data))
    model = glm(formula, data, Poisson(), Link)
    return FrequentistRegression(:PoissonRegression, model)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression)
```

Fit a Poisson Regression model on the input data (with the default link function being the Log link). Uses the `glm` method from the [GLM](https://github.com/JuliaStats/GLM.jl) package under the hood.

# Arguments
- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset. 
- `modelClass`: Object representing the type of regression, which is Poisson Regression in this case.xc
# Example

```julia-repl
julia> using CRRao, RDatasets, StatsBase

julia> sanction = dataset("Zelig", "sanction");

julia> container = @fitmodel(Num ~ Target + Coop + NCost, sanction, PoissonRegression());
```
"""
function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression)
    return poisson_reg(formula, data, LogLink())
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