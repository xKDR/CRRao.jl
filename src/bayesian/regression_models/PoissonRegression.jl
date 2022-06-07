function Poisson_Reg_predicts(obj,newdata::DataFrame)
  
  formula = obj.formula;
  fm_frame=ModelFrame(formula,newdata);
  X=modelmatrix(fm_frame);
  beta = obj.beta
  z = X*beta;
  μ = exp.(z) ;  
  μ
end
  
  
## Poisson Regression with Ridge Prior
function Poisson_Reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Ridge,h::Float64,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    @model PoissonReg(X, y) = begin
      p = size(X, 2);
      n = size(X, 1);
      #priors
      λ~InverseGamma(h,h)
      α ~ Normal(0,λ)
      β ~ filldist(Normal(0,λ), p)
    
      ## link
      z = α .+ X * β
      mu = exp.(z)
  
      #likelihood
      for i = 1:n
        y[i] ~ Poisson(mu[i])
      end
    end
    PoissonReg_model=PoissonReg(X,y);
    chain = sample(CRRao_rng, PoissonReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
end
  
## Poisson Regression with Laplace Prior
function Poisson_Reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Laplace,h::Float64,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    @model PoissonReg(X, y) = begin
      p = size(X, 2);
      n = size(X, 1);
      #priors
      λ~InverseGamma(h,h)
      α ~ Laplace(0,λ)
      β ~ filldist(Laplace(0,λ), p)
    
      ## link
      z = α .+ X * β
      mu = exp.(z)
  
      #likelihood
      for i = 1:n
        y[i] ~ Poisson(mu[i])
      end
    end
    PoissonReg_model=PoissonReg(X,y);
    chain = sample(CRRao_rng, PoissonReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
end
  
## Poisson Regression with Cauchy Prior
function Poisson_Reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Cauchy,h::Float64,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    @model PoissonReg(X, y) = begin
      p = size(X, 2);
      n = size(X, 1);
      #priors
      λ~InverseGamma(h,h)
      α ~ TDist(1)*λ
      β ~ filldist(TDist(1)*λ, p)  
  
      ## link
      z = α .+ X * β
      mu = exp.(z)
  
      #likelihood
      for i = 1:n
        y[i] ~ Poisson(mu[i])
      end
    end
    PoissonReg_model=PoissonReg(X,y);
    chain = sample(CRRao_rng, PoissonReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
end
  
## Poisson Regression with T-Distributed Prior
function Poisson_Reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_TDist,h::Float64,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    @model PoissonReg(X, y) = begin
      p = size(X, 2);
      n = size(X, 1);
      #priors
      λ~InverseGamma(h,h)
      ν~InverseGamma(h,h)
      α ~ TDist(ν)*λ
      β ~ filldist(TDist(ν)*λ, p)  
  
      ## link
      z = α .+ X * β
      mu = exp.(z)
  
      #likelihood
      for i = 1:n
        y[i] ~ Poisson(mu[i])
      end
    end
    PoissonReg_model=PoissonReg(X,y);
    chain = sample(CRRao_rng, PoissonReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
end
  
## Poisson Regression with Uniform Prior
function Poisson_Reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Uniform,h::Float64,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    @model PoissonReg(X, y) = begin
      p = size(X, 2);
      n = size(X, 1);
      #priors
      λ~InverseGamma(h,h)
      α ~ Uniform(-λ,λ)
      β ~ filldist(Uniform(-λ,λ), p) 
      ## link
      z = α .+ X * β
      mu = exp.(z)
  
      #likelihood
      for i = 1:n
        y[i] ~ Poisson(mu[i])
      end
    end
    PoissonReg_model=PoissonReg(X,y);
    chain = sample(CRRao_rng, PoissonReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
end
  
  