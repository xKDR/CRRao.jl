function logistic_reg_predicts(obj,newdata::DataFrame)
    
    formula = obj.formula;
    fm_frame=ModelFrame(formula,newdata);
    X=modelmatrix(fm_frame);
    beta = obj.beta
    z = X*beta;
  
    if (obj.Link=="LogitLink")
      p = exp.(z) ./ (1 .+ exp.(z));
        
    elseif (obj.Link=="ProbitLink")
      p = Probit_Link.(z);
        
    elseif (obj.Link=="CauchitLink")
      p = Cauchit_Link.(z);
  
    elseif (obj.Link=="CloglogLink")
      p = Cloglog_Link.(z);
  
    else
        println("This link function is not part of logistic regression family.")
        println("-------------------------------------------------------------")
    end 
    p
end
  
## logistic regression with Ridge Prior

function logistic_reg_internal(formula::FormulaTerm, data::DataFrame, link_function, PriorMod::Prior_Ridge, h::Float64=0.5,sim_size::Int64=10000)
    
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

    chain = sample(CRRao_rng, LogisticReg_model, NUTS(), sim_size);

    summaries, quantiles = describe(chain);

    ans = MCMC_chain(chain,summaries,quantiles)

    ans

end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Logit, PriorMod::Prior_Ridge, h::Float64=0.5,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Logit_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Probit, PriorMod::Prior_Ridge, h::Float64=0.5,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Probit_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Cloglog, PriorMod::Prior_Ridge, h::Float64=0.5,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Cloglog_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Cauchit, PriorMod::Prior_Ridge, h::Float64=0.5,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Cauchit_Link, PriorMod, h, sim_size)
end

## logistic regression with Laplace Prior

function logistic_reg_internal(formula::FormulaTerm,data::DataFrame, link_function,PriorMod::Prior_Laplace,h::Float64=0.5,sim_size::Int64=10000)
    
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

    chain = sample(CRRao_rng, LogisticReg_model, NUTS(), sim_size);

    summaries, quantiles = describe(chain);

    ans = MCMC_chain(chain,summaries,quantiles)

    return ans

end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Logit, PriorMod::Prior_Laplace, h::Float64=0.5,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Logit_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Probit, PriorMod::Prior_Laplace, h::Float64=0.5,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Probit_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Cloglog, PriorMod::Prior_Laplace, h::Float64=0.5,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Cloglog_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Cauchit, PriorMod::Prior_Laplace, h::Float64=0.5,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Cauchit_Link, PriorMod, h, sim_size)
end

## logistic regression with Cauchy Prior

function logistic_reg_internal(formula::FormulaTerm,data,link_function,PriorMod::Prior_Cauchy,h::Float64,sim_size::Int64)
    
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

    chain = sample(CRRao_rng, LogisticReg_model, NUTS(), sim_size);

    summaries, quantiles = describe(chain);

    ans = MCMC_chain(chain,summaries,quantiles)

    return ans

end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Logit, PriorMod::Prior_Cauchy, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Logit_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Probit, PriorMod::Prior_Cauchy, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Probit_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Cloglog, PriorMod::Prior_Cauchy, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Cloglog_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Cauchit, PriorMod::Prior_Cauchy, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Cauchit_Link, PriorMod, h, sim_size)
end

## logistic regression with TDist Prior

function logistic_reg_internal(formula::FormulaTerm,data,link_function,PriorMod::Prior_TDist,h::Float64,sim_size::Int64)
    
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);

    @model LogisticReg(X, y) = begin
        p = size(X, 2);
        n = size(X, 1);
        #priors
        λ ~ InverseGamma(h,h)
        ν ~ InverseGamma(h,h)
        β ~ filldist(TDist(ν)*λ, p)  

        z = X*β
        
        ## Link Function

        prob = Logit_Link.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
        
    end;


    LogisticReg_model=LogisticReg(X,y);

    chain = sample(CRRao_rng, LogisticReg_model, NUTS(), sim_size);

    summaries, quantiles = describe(chain);

    ans = MCMC_chain(chain,summaries,quantiles)

    return ans

end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Logit, PriorMod::Prior_TDist, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Logit_Link, PriorMod, h, sim_size)
end


function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Probit, PriorMod::Prior_TDist, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Probit_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Cloglog, PriorMod::Prior_TDist, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Cloglog_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Cauchit, PriorMod::Prior_TDist, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Cauchit_Link, PriorMod, h, sim_size)
end

## logistic regression with Uniform Prior

function logistic_reg_internal(formula::FormulaTerm,data,link_function,PriorMod::Prior_Uniform,h::Float64=0.5,sim_size::Int64=10000)
    
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

    chain = sample(CRRao_rng, LogisticReg_model, NUTS(), sim_size);

    summaries, quantiles = describe(chain);

    ans = MCMC_chain(chain,summaries,quantiles)

    return ans

end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Logit, PriorMod::Prior_Uniform, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Logit_Link, PriorMod, h, sim_size)
end


function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Probit, PriorMod::Prior_Uniform, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Probit_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Cauchit, PriorMod::Prior_Uniform, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Cauchit_Link, PriorMod, h, sim_size)
end

function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::Cloglog, PriorMod::Prior_Uniform, h::Float64=1.0,sim_size::Int64=10000)
    logistic_reg_internal(formula, data, Cloglog_Link, PriorMod, h, sim_size)
end
