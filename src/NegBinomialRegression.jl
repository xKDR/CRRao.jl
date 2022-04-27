## Negative Binomial regression

struct analysis_Count_NegativeBinom_NIP_Optim
    formula::FormulaTerm
    modelClass::String
    LikelihoodMod::String
    PriorMod::String
    Link::String
    ComputeMethod::String
    fit
    beta
    LogLike::Float64
    AIC::Float64
    BIC::Float64
    #R_sqr::Float64
    #Adjusted_R_sqr::Float64
    #sigma::Float64
    #Cooks_distance
end
  
  
function NegBinom_Reg(formula::FormulaTerm,data,Link::String="LogLink")
  CRRao.seed !=  nothing && Random.seed!(CRRao.seed)
  modelClass = "CountReg";
  LikelihoodMod="NegativeBinomial";
  PriorMod="NIP";
  ComputeMethod="Optimization";
  formula = apply_schema(formula, schema(formula, data));
  y, X = modelcols(formula, data);
  fm_frame=ModelFrame(formula,data);
  X=modelmatrix(fm_frame);
  p = size(X, 2);
  n = size(X, 1);
  
  if (Link=="LogLink")
      res = glm(formula,data,NegativeBinomial(), LogLink());
  end
  
  fit = coeftable(res)
  beta_hat = coef(res)
  logLike = GLM.loglikelihood(res)
  LogPost = logLike
  npar = p;
  AIC = 2*npar - 2*logLike
  BIC = log(n)*npar - 2*logLike
  
  ans = analysis_Count_NegativeBinom_NIP_Optim(formula,modelClass,LikelihoodMod
          ,PriorMod,Link,ComputeMethod,fit,beta_hat
          ,logLike,AIC,BIC)
  ans
  
end
  

function NegBinom_Reg_predicts(obj,newdata::DataFrame)
  CRRao.seed !=  nothing && Random.seed!(CRRao.seed)
  formula = obj.formula;
  fm_frame=ModelFrame(formula,newdata);
  X=modelmatrix(fm_frame);
  beta = obj.beta
  z = X*beta;

  if(obj.Link=="LogLink")
    p = exp.(z) ;
      
  else
      println("This link function is not part of NegativeBinomial regression family.")
      println("-------------------------------------------------------------")
  end 
  p
end
  

## Negative Binomial Regression with Ridge Prior
function NegBinom_Reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Ridge,h::Float64,sim_size::Int64)
  CRRao.seed !=  nothing && Random.seed!(CRRao.seed)  
  formula = apply_schema(formula, schema(formula, data));
  y, X = modelcols(formula, data);
  @model NegBinomReg(X, y) = begin
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
      y[i] ~ NegativeBinomial2(mu[i],λ)
    end
  end
  NegBinomReg_model=NegBinomReg(X,y);
  chain = sample(NegBinomReg_model, NUTS(), sim_size);
  summaries, quantiles = describe(chain);
  ans = MCMC_chain(chain,summaries,quantiles)
  ans
end
  
  
  
## Negative Binomial Regression with Laplace Prior
function NegBinom_Reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Laplace,h::Float64,sim_size::Int64)
  CRRao.seed !=  nothing && Random.seed!(CRRao.seed)
  formula = apply_schema(formula, schema(formula, data));
  y, X = modelcols(formula, data);
  @model NegBinomReg(X, y) = begin
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
      y[i] ~ NegativeBinomial2(mu[i],λ)
    end
  end
  NegBinomReg_model=NegBinomReg(X,y);
  chain = sample(NegBinomReg_model, NUTS(), sim_size);
  summaries, quantiles = describe(chain);
  ans = MCMC_chain(chain,summaries,quantiles)
  ans
end
  
## Negative Binomial Regression with Cauchy Prior
function NegBinom_Reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Cauchy,h::Float64,sim_size::Int64)
    CRRao.seed !=  nothing && Random.seed!(CRRao.seed)
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    @model NegBinomReg(X, y) = begin
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
        y[i] ~ NegativeBinomial2(mu[i],λ)
      end
    end
    NegBinomReg_model=NegBinomReg(X,y);
    chain = sample(NegBinomReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
end
  
  
## Negative Binomial Regression with TDist Prior
function NegBinom_Reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_TDist,h::Float64,sim_size::Int64)
    CRRao.seed !=  nothing && Random.seed!(CRRao.seed)
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    @model NegBinomReg(X, y) = begin
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
        y[i] ~ NegativeBinomial2(mu[i],λ)
      end
    end
    NegBinomReg_model=NegBinomReg(X,y);
    chain = sample(NegBinomReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
end
  
  
  
## Negative Binomial Regression with Uniform Prior
function NegBinom_Reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Uniform,h::Float64,sim_size::Int64)
    CRRao.seed !=  nothing && Random.seed!(CRRao.seed)
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    @model NegBinomReg(X, y) = begin
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
        y[i] ~ NegativeBinomial2(mu[i],λ)
      end
    end
    NegBinomReg_model=NegBinomReg(X,y);
    chain = sample(NegBinomReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
end  