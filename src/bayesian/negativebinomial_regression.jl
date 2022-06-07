# Include NegBinom_Reg definitions
include("regression_models/NegBinomialRegression.jl")

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