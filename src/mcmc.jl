struct analysis_lm_Gauss_MCMC
    chain::Chains
    summaries
    quantiles
  end

struct analysis_logistic_Binom_MCMC
  chain::Chains
  summaries
  quantiles
end



function Linear_Reg_Ridge_prior(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Ridge,h::Float64,sim_size::Int64)
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
  chain = sample(LinReg_model, NUTS(), sim_size);
  summaries, quantiles = describe(chain);
  
  ans = analysis_lm_Gauss_MCMC(chain,summaries,quantiles)
  ans
end

function Linear_Reg_Laplace_prior(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Laplace,h::Float64,sim_size::Int64)
  formula = apply_schema(formula, schema(formula, data));
  y, X = modelcols(formula, data);
    
  @model LinReg(X, y) = begin
    p=size(X, 2);

    #priors
    a0=0.1
    b0=0.1
    v ~ InverseGamma(h,h)
    σ ~ InverseGamma(a0,b0)
    α ~ Laplace(0,v)
    β ~ filldist(Laplace(0,v), p)

    #likelihood
    y ~ MvNormal(α .+ X * β, σ);
  end;
  LinReg_model=LinReg(X,y);
  chain = sample(LinReg_model, NUTS(), sim_size);
  summaries, quantiles = describe(chain);
  
  ans = analysis_lm_Gauss_MCMC(chain,summaries,quantiles)
  ans

end


function Linear_Reg_Cauchy_prior(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Cauchy,h::Float64,sim_size::Int64)
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
  chain = sample(LinReg_model, NUTS(), sim_size);
  summaries, quantiles = describe(chain);
  
  ans = analysis_lm_Gauss_MCMC(chain,summaries,quantiles)
  ans

end



function Linear_Reg_TDist_prior(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_TDist,h::Float64,sim_size::Int64)
  formula = apply_schema(formula, schema(formula, data));
  y, X = modelcols(formula, data);
    
  @model LinReg(X, y) = begin
    p=size(X, 2);

    #priors
    a0=0.1
    b0=0.1
    v ~ InverseGamma(h,h)
    σ ~ InverseGamma(a0,b0)
    α ~ TDist(v)*σ
    β ~ filldist(TDist(v)*σ, p)

    #likelihood
    y ~ MvNormal(α .+ X * β, σ);
  end;
  LinReg_model=LinReg(X,y);
  chain = sample(LinReg_model, NUTS(), sim_size);
  summaries, quantiles = describe(chain);
  
  ans = analysis_lm_Gauss_MCMC(chain,summaries,quantiles)
  ans

end


function Linear_Reg_Uniform_prior(formula::FormulaTerm,data::DataFrame,PriorMod::Prior_Uniform,h::Float64,sim_size::Int64)
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
  chain = sample(LinReg_model, NUTS(), sim_size);
  summaries, quantiles = describe(chain);
  
  ans = analysis_lm_Gauss_MCMC(chain,summaries,quantiles)
  ans

end




function lm_Gauss_MCMC(formula::FormulaTerm,data,PriorMod::String="NIP",h::Float64=0.01,sim_size::Int64=10000)
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    
    @model LinReg(X, y; predictors=size(X, 2)) = begin
      
      if(PriorMod=="NIP")
        #priors
        v=1/h;
        α ~ Uniform(-v,v)
        β ~ filldist(Uniform(-v,v), predictors)
        σ ~ Uniform(0,v)
      elseif (PriorMod=="Ridge")
        #priors
        a0=0.1
        b0=0.1
        
        v ~ InverseGamma(h,h)
        σ ~ InverseGamma(a0,b0)
        α ~ Normal(0,v*σ)
        β ~ filldist(Normal(0,v*σ), predictors)
      elseif (PriorMod=="Laplace")
        a0=0.1
        b0=0.1
        v ~ InverseGamma(h,h)
        σ ~ InverseGamma(a0,b0)
        α ~ Laplace(0,v)
        β ~ filldist(Laplace(0,v), predictors)
      elseif (PriorMod=="TDist")
        a0=0.1
        b0=0.1
        v ~ InverseGamma(h,h)
        σ ~ InverseGamma(a0,b0)
        α ~ TDist(v)*σ
        β ~ filldist(TDist(v)*σ, predictors)
      elseif (PriorMod=="Cauchy")
        σ ~ Truncated(TDist(1),0,Inf)
        α ~ TDist(1)*σ
        β ~ filldist(TDist(1)*σ, predictors)
      elseif (PriorMod=="Stack&Slab")
        σ ~ Truncated(TDist(1),0,Inf)
        d=MixtureModel(Normal, [(0.0, 0.1), (0.0, 1.0)], [0.5, 0.5])
        α ~ d*σ
        β ~ filldist(d*σ, predictors)
      end
      #likelihood
      y ~ MvNormal(α .+ X * β, σ)
    end;
    
    LinReg_model=LinReg(X,y);
    chain = sample(LinReg_model, NUTS(), sim_size);
    summaries, quantiles = describe(chain);
    
    ans = analysis_lm_Gauss_MCMC(chain,summaries,quantiles)
    ans
  end
  
  
## logistic regression
  
  
  
  function logistic_Binom_Ridge_Internal(formula::FormulaTerm, data::DataFrame, link_function, PriorMod::Prior_Ridge, h::Float64=0.5,sim_size::Int64=10000)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    
    @model LogisticReg(X, y) = begin
      p = size(X, 2);
      n = size(X, 1);
      #priors
      λ~InverseGamma(h,h)
      β ~ filldist(Normal(0,λ), p)  
    
      z = X*β
      
      ## Link Function

      prob = link_function.(z)

      #likelihood
      for i = 1:n
          y[i] ~ Bernoulli(prob[i])
      end
      
    end;
  
  
    LogisticReg_model=LogisticReg(X,y);
    
    chain = sample(LogisticReg_model, NUTS(), sim_size);

    summaries, quantiles = describe(chain);
    
    ans = analysis_logistic_Binom_MCMC(chain,summaries,quantiles)
    
    return ans
    
  end

  function logistic_Binom_Ridge(formula::FormulaTerm, data::DataFrame, Link::Logit, PriorMod::Prior_Ridge, h::Float64=0.5,sim_size::Int64=10000)
    logistic_Binom_Ridge_Internal(formula, data, Logit_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Ridge(formula::FormulaTerm, data::DataFrame, Link::Probit, PriorMod::Prior_Ridge, h::Float64=0.5,sim_size::Int64=10000)
    logistic_Binom_Ridge_Internal(formula, data, Probit_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Ridge(formula::FormulaTerm, data::DataFrame, Link::Cloglog, PriorMod::Prior_Ridge, h::Float64=0.5,sim_size::Int64=10000)
    logistic_Binom_Ridge_Internal(formula, data, Cloglog_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Ridge(formula::FormulaTerm, data::DataFrame, Link::Cauchit, PriorMod::Prior_Ridge, h::Float64=0.5,sim_size::Int64=10000)
    logistic_Binom_Ridge_Internal(formula, data, Cauchit_Link, PriorMod, h, sim_size)
  end


  function logistic_Binom_Laplace_Internal(formula::FormulaTerm,data::DataFrame, link_function,PriorMod::Prior_Laplace,h::Float64=0.5,sim_size::Int64=10000)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    
    @model LogisticReg(X, y) = begin
      p = size(X, 2);
      n = size(X, 1);
      #priors
      λ~InverseGamma(h,h)
      β ~ filldist(Laplace(0,λ), p)  
    
      z = X*β
      
      ## Link Function

      prob = Logit_Link.(z)

      #likelihood
      for i = 1:n
          y[i] ~ Bernoulli(prob[i])
      end
      
    end;
  
  
    LogisticReg_model=LogisticReg(X,y);
    
    chain = sample(LogisticReg_model, NUTS(), sim_size);

    summaries, quantiles = describe(chain);
    
    ans = analysis_logistic_Binom_MCMC(chain,summaries,quantiles)
    
    return ans
    
  end

  function logistic_Binom_Laplace(formula::FormulaTerm, data::DataFrame, Link::Logit, PriorMod::Prior_Laplace, h::Float64=0.5,sim_size::Int64=10000)
    logistic_Binom_Laplace_Internal(formula, data, Logit_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Laplace(formula::FormulaTerm, data::DataFrame, Link::Probit, PriorMod::Prior_Laplace, h::Float64=0.5,sim_size::Int64=10000)
    logistic_Binom_Laplace_Internal(formula, data, Probit_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Laplace(formula::FormulaTerm, data::DataFrame, Link::Cloglog, PriorMod::Prior_Laplace, h::Float64=0.5,sim_size::Int64=10000)
    logistic_Binom_Laplace_Internal(formula, data, Cloglog_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Laplace(formula::FormulaTerm, data::DataFrame, Link::Cauchit, PriorMod::Prior_Laplace, h::Float64=0.5,sim_size::Int64=10000)
    logistic_Binom_Laplace_Internal(formula, data, Cauchit_Link, PriorMod, h, sim_size)
  end


  function logistic_Binom_Cauchy_Internal(formula::FormulaTerm,data,link_function,PriorMod::Prior_Cauchy,h::Float64,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    
    @model LogisticReg(X, y) = begin
      p = size(X, 2);
      n = size(X, 1);
      #priors
      λ~ Truncated(TDist(1),0,Inf)
      β ~ filldist(TDist(1)*λ, p)  

      z = X*β
      
      ## Link Function

      prob = Logit_Link.(z)

      #likelihood
      for i = 1:n
          y[i] ~ Bernoulli(prob[i])
      end
      
    end;
  
  
    LogisticReg_model=LogisticReg(X,y);
    
    chain = sample(LogisticReg_model, NUTS(), sim_size);

    summaries, quantiles = describe(chain);
    
    ans = analysis_logistic_Binom_MCMC(chain,summaries,quantiles)
    
    return ans
    
  end

  function logistic_Binom_Cauchy(formula::FormulaTerm, data::DataFrame, Link::Logit, PriorMod::Prior_Cauchy, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_Cauchy_Internal(formula, data, Logit_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Cauchy(formula::FormulaTerm, data::DataFrame, Link::Probit, PriorMod::Prior_Cauchy, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_Cauchy_Internal(formula, data, Probit_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Cauchy(formula::FormulaTerm, data::DataFrame, Link::Cloglog, PriorMod::Prior_Cauchy, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_Cauchy_Internal(formula, data, Cloglog_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Cauchy(formula::FormulaTerm, data::DataFrame, Link::Cauchit, PriorMod::Prior_Cauchy, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_Cauchy_Internal(formula, data, Cauchit_Link, PriorMod, h, sim_size)
  end


  function logistic_Binom_TDist_Internal(formula::FormulaTerm,data,link_function,PriorMod::Prior_TDist,h::Float64=0.5,sim_size::Int64=10000)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    
    @model LogisticReg(X, y) = begin
      p = size(X, 2);
      n = size(X, 1);
      #priors
      ν ~ InverseGamma(h,h)
      β ~ filldist(TDist(ν), p)  

      z = X*β
      
      ## Link Function

      prob = Logit_Link.(z)

      #likelihood
      for i = 1:n
          y[i] ~ Bernoulli(prob[i])
      end
      
    end;
  
  
    LogisticReg_model=LogisticReg(X,y);
    
    chain = sample(LogisticReg_model, NUTS(), sim_size);

    summaries, quantiles = describe(chain);
    
    ans = analysis_logistic_Binom_MCMC(chain,summaries,quantiles)
    
    return ans
    
  end

  function logistic_Binom_TDist(formula::FormulaTerm, data::DataFrame, Link::Logit, PriorMod::Prior_TDist, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_TDist_Internal(formula, data, Logit_Link, PriorMod, h, sim_size)
  end


  function logistic_Binom_TDist(formula::FormulaTerm, data::DataFrame, Link::Probit, PriorMod::Prior_TDist, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_TDist_Internal(formula, data, Probit_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_TDist(formula::FormulaTerm, data::DataFrame, Link::Cloglog, PriorMod::Prior_TDist, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_TDist_Internal(formula, data, Cloglog_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_TDist(formula::FormulaTerm, data::DataFrame, Link::Cauchit, PriorMod::Prior_TDist, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_TDist_Internal(formula, data, Cauchit_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Uniform_Internal(formula::FormulaTerm,data,link_function,PriorMod::Prior_Uniform,h::Float64=0.5,sim_size::Int64=10000)
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    
    @model LogisticReg(X, y) = begin
      p = size(X, 2);
      n = size(X, 1);
      #priors
      v ~ InverseGamma(h,h)
      β ~ filldist(Uniform(-v,v), p) 
      
      z = X*β
      
      ## Link Function

      prob = Logit_Link.(z)

      #likelihood
      for i = 1:n
          y[i] ~ Bernoulli(prob[i])
      end
      
    end;
  
  
    LogisticReg_model=LogisticReg(X,y);
    
    chain = sample(LogisticReg_model, NUTS(), sim_size);

    summaries, quantiles = describe(chain);
    
    ans = analysis_logistic_Binom_MCMC(chain,summaries,quantiles)
    
    return ans
    
  end

  function logistic_Binom_Uniform(formula::FormulaTerm, data::DataFrame, Link::Logit, PriorMod::Prior_Uniform, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_Uniform_Internal(formula, data, Logit_Link, PriorMod, h, sim_size)
  end


  function logistic_Binom_Uniform(formula::FormulaTerm, data::DataFrame, Link::Probit, PriorMod::Prior_Uniform, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_Uniform_Internal(formula, data, Probit_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Uniform(formula::FormulaTerm, data::DataFrame, Link::Cauchit, PriorMod::Prior_Uniform, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_Uniform_Internal(formula, data, Cauchit_Link, PriorMod, h, sim_size)
  end

  function logistic_Binom_Uniform(formula::FormulaTerm, data::DataFrame, Link::Cloglog, PriorMod::Prior_Uniform, h::Float64=1.0,sim_size::Int64=10000)
    logistic_Binom_Uniform_Internal(formula, data, Cloglog_Link, PriorMod, h, sim_size)
  end


  function logistic_Binom_MCMC(formula::FormulaTerm,data,PriorMod::String="NIP",Link::String="LogitLink",h::Float64=0.5,sim_size::Int64=10000)
  
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    
    @model LogisticReg(X, y,Link) = begin
        p = size(X, 2);
        n = size(X, 1);
        #priors
        if(PriorMod=="NIP")
          v=1/abs(h)
          #β ~ filldist(TDist(1)*v, p) 
          β ~ filldist(Uniform(-v,v), p) 
        elseif (PriorMod=="Ridge")
          λ~InverseGamma(h,h)
          β ~ filldist(Normal(0,λ), p)  
        elseif (PriorMod=="Laplace")
          λ~InverseGamma(h,h)
          β ~ filldist(Laplace(0,λ), p)  
        elseif (PriorMod=="TDist")
          ν~InverseGamma(h,h)
          β ~ filldist(TDist(ν), p)  
        elseif (PriorMod=="Cauchy")
          λ~ Truncated(TDist(1),0,Inf)
          β ~ filldist(TDist(1)*λ, p)  
        elseif (PriorMod=="Stack&Slab")
          σ ~ Truncated(TDist(1),0,Inf)
          d=MixtureModel(Normal, [(0.0, 0.01), (0.0, 1.0)], [0.5, 0.5])
          α ~ d*σ
          β ~ filldist(d*σ, p)
        end
        
        z = X*β
        
        ## Link Function
        if(Link=="ProbitLink")
           prob = Probit_Link.(z)
        elseif(Link=="LogitLink")
           prob = Logit_Link.(z)
        elseif(Link=="CauchitLink")
           prob = Cauchit_Link.(z)
        elseif(Link=="CloglogLink")
           prob = Cloglog_Link.(z)
        end
        
        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
        
    end;
    
    
    LogisticReg_model=LogisticReg(X,y,Link);
    
    chain = sample(LogisticReg_model, NUTS(), sim_size);
  
    summaries, quantiles = describe(chain);
    
    ans = analysis_logistic_Binom_MCMC(chain,summaries,quantiles)
    
    if (PriorMod=="NIP")
      
    end
    return ans
  end
  
  ## Count Regression
  
  struct analysis_count_reg_MCMC
    chain::Chains
    summaries
    quantiles
  end
  
  function Count_Reg_MCMC(formula::FormulaTerm,data,LikelihoodMod::String="Poisson",PriorMod::String="NIP",h::Float64=0.5,sim_size::Int64=10000)
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    
    @model CountReg(X, y,Link) = begin
      p = size(X, 2);
      n = size(X, 1);
      #priors
      if (PriorMod=="NIP")
          
        v=1/abs(h)
        λ~InverseGamma(h,h)
        α ~ Uniform(-v,v)
        #β ~ filldist(TDist(1)*v, p) 
        β ~ filldist(Uniform(-v,v), p) 
      elseif (PriorMod=="Ridge")
        λ~InverseGamma(h,h)
        α ~ Normal(0,λ)
        β ~ filldist(Normal(0,λ), p)  
      elseif (PriorMod=="Laplace")
        λ~InverseGamma(h,h)
        α ~ Laplace(0,λ)
        β ~ filldist(Laplace(0,λ), p)  
      elseif (PriorMod=="TDist")
        λ~InverseGamma(h,h)
        ν~InverseGamma(h,h)
        α ~ TDist(ν)
        β ~ filldist(TDist(ν), p)  
      elseif (PriorMod=="Cauchy")
        #λ~Truncated(TDist(1),0,Inf)
        λ~InverseGamma(h,h)
        α ~ TDist(1)*λ
        β ~ filldist(TDist(1)*λ, p)  
      elseif (PriorMod=="Stack&Slab")
        σ ~ Truncated(TDist(1),0,Inf)
        d=MixtureModel(Normal, [(0.0, 0.01), (0.0, 1.0)], [0.5, 0.5])
        α ~ d*σ
        β ~ filldist(d*σ, p)
      end
      
      z = α .+ X * β
      mu = exp.(z)
  
      #likelihood
      if(LikelihoodMod=="Poisson")
        #y ~ arraydist(LazyArray(@~ LogPoisson.(α .+ X * β)))
        for i = 1:n
          y[i] ~ Poisson(mu[i])
        end
      elseif (LikelihoodMod=="NegativeBinomial")
        #y ~ arraydist(LazyArray(@~ NegativeBinomial2.(exp.(α .+ X * β), λ)))
        for i = 1:n
          y[i] ~ NegativeBinomial2(mu[i],λ)
        end
      end
    end
    CountReg_model=CountReg(X,y,Link);
    
    chain = sample(CountReg_model, NUTS(), sim_size);
  
    summaries, quantiles = describe(chain);
    
    ans = analysis_count_reg_MCMC(chain,summaries,quantiles)
    if(PriorMod=="NIP")
      println("Ibrahim and Laud (JASA, 1990) showed that 
          the uniform flat priors for GLM's can lead to improper
          posterior distributions thus making them undesirable.")
          println("In such cases, the Markov Chain struggles to converge. 
          Even if it converges, results are completely unreliable.")
    end
    ans
  end
  