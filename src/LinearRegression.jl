struct analysis_lm_Gauss_NIP_Optim
    formula::FormulaTerm
    modelClass::String
    LikelihoodMod::String
    PriorMod::String
    Link::String
    ComputeMethod::String
    res
    fit
    LogLike::Float64
    AIC::Float64
    BIC::Float64
    R_sqr::Float64
    Adjusted_R_sqr::Float64
    sigma::Float64
    fittedResponse
    residuals
    Cooks_distance
end
  
function linear_reg(formula::FormulaTerm,data::DataFrame)
      
      modelClass = "LinearReg";
      LikelihoodMod="Gauss";
      PriorMod="NIP";
      Link = "Identity";
      ComputeMethod="Optimization";
      formla = formula;
      formula = apply_schema(formula, schema(formula, data));
      y, X = modelcols(formula, data);
      fm_frame=ModelFrame(formula,data);
      X=modelmatrix(fm_frame);
      p = size(X, 2);
      n = size(X, 1);
      
      res = lm(formula,data);
      fit = coeftable(res)
      R_sqr = GLM.r2(res);
      Adjusted_R_sqr = GLM.adjr2(res);
      
      logLike = GLM.loglikelihood(res)
      npar =(p+1);
      AIC = 2*npar - 2*logLike
      BIC = log(n)*npar - 2*logLike
      
      sse = GLM.deviance(res)
      degrees_of_freedon = GLM.dof_residual(res)
      mse = sse/degrees_of_freedon
      sigma = sqrt(mse)
      
      fittedResponse = predict(res);
      resid = residuals(res);
  
      cooks_distance = GLM.StatsBase.cooksdistance(res); 
      ans = analysis_lm_Gauss_NIP_Optim(formla,modelClass,LikelihoodMod,PriorMod
      ,Link,ComputeMethod,res,fit, logLike, AIC, BIC, R_sqr
      ,Adjusted_R_sqr,sigma, fittedResponse,resid,cooks_distance)
  
      return ans
        
end
  
function linear_reg_predicts(obj,newdata::DataFrame)
    
    formula = obj.formula;
    fm_frame=ModelFrame(formula,newdata);
    X=modelmatrix(fm_frame);
    beta = coef(obj.res)
    y_pred = X*beta;
    y_pred
end

function linear_reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Ridge,h::Float64,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
      
    @model LinReg(X, y) = begin
      p=size(X, 2);
  
      #priors
      a0=0.1
      b0=0.1
      
      v ~ InverseGamma(h,h)
      σ ~ InverseGamma(a0,b0)
      α ~ Normal(0,v*σ)
      β ~ filldist(Normal(0,v*σ), p)
  
      #likelihood
      y ~ MvNormal(α .+ X * β, σ);
    end;
    LinReg_model=LinReg(X,y);
    chain = sample(CRRao_rng, LinReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
end
  
function linear_reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Laplace,h::Float64,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
      
    @model LinReg(X, y) = begin
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
    end;
    LinReg_model=LinReg(X,y);
    chain = sample(CRRao_rng, LinReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
  
end
  
function linear_reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Cauchy,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
      
    @model LinReg(X, y) = begin
      p=size(X, 2);
  
      #priors
      σ ~ Truncated(TDist(1),0,Inf)
      α ~ TDist(1)*σ
      β ~ filldist(TDist(1)*σ, p)
  
      #likelihood
      y ~ MvNormal(α .+ X * β, σ);
    end;
    LinReg_model=LinReg(X,y);
    chain = sample(CRRao_rng, LinReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
  
end
  
function linear_reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_TDist,h::Float64,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
      
    @model LinReg(X, y) = begin
      p=size(X, 2);
  
      #priors
      a0=0.1
      b0=0.1
      ν ~ InverseGamma(h,h)
      σ ~ InverseGamma(a0,b0)
      α ~ TDist(ν)*σ
      β ~ filldist(TDist(ν)*σ, p)
  
      #likelihood
      y ~ MvNormal(α .+ X * β, σ);
    end;
    LinReg_model=LinReg(X,y);
    chain = sample(CRRao_rng, LinReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
  
end
  
function linear_reg(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Uniform,h::Float64,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
      
    @model LinReg(X, y) = begin
      p=size(X, 2);
  
      #priors
      v=1/h;
      σ ~ Uniform(0,v)
      α ~ Uniform(-v*σ,v*σ)
      β ~ filldist(Uniform(-v,v), predictors)
      
      #likelihood
      y ~ MvNormal(α .+ X * β, σ);
    end;
    LinReg_model=LinReg(X,y);
    chain = sample(CRRao_rng, LinReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    
    ans = MCMC_chain(chain,summaries,quantiles)
    ans
  
end
    