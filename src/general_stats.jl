struct sanitised_arguement
   modelClass::String
   LikelihoodMod::String
   PriorMod::String
   Link::String
   ComputeMethod::String
end

function arguement_sanitisation(modelClass::String="LinearReg", LikelihoodMod::String="Gauss",PriorMod::String="NIP",Link::String="Identity",ComputeMethod::String="Optimization")
    
    modelClass_dumbo=lowercase(modelClass)
    LikelihoodMod_dumbo=lowercase(LikelihoodMod)
    PriorMod_dumbo = lowercase(PriorMod)
    Link_dumbo =lowercase(Link)
    ComputeMethod_dumbo =lowercase(ComputeMethod)
    
    ## bag of words in memory
    modelClass_reg_set =["linearreg","lm","linearmodel","regression","reg","regresion","linreg"]
    modelClass_LogisticReg_set =["logisticreg","logistic","logisticregression","logistic_regression","logistic_reg","logistic_model","logisticmodel"
    ,"logitmod", "logistic.model","logitreg","logit_mod","logit.mod"
    ,"logistic.reg","logit_reg","logit.reg","logitmodl","logit.modl"]
    
    LikelihoodMod_Gauss_set=["gauss","normal","norm","gaus"]
    LikelihoodMod_Binom_set=["binom","binomial","bin","bernoulli"]
    
    PriorMod_NIP_set=["nul","null","nip","flat","flatprior","flat_prior","flat.prior","flat_gauss","uniform"]
    PriorMod_Ridge_set=["ridge","gauss","ridge_prior","ridge.prior","rige", "rige_prior","rige.prior","ridgeprior"]
    PriorMod_Lasso_set=["lasso","laplace","lasso_prior","lasso.prior","lassopenalty", "lasso_penalty","lasso.penalty","laplace_prior","laplace.prior","laplaceprior"]
    
    Link_identity_set = ["identity","id"]
    Link_probit_set = ["probit","probitlink","probitlink()","probit()"]
    Link_logit_set = ["logit","logitlink","logitlink()","lozit","lojit","logit_link"]
    Link_cauchy_set = ["cauchitlink","cauchitlink()","cauchylink","cauchylink()","cauchy_link"]
    Link_cloglog_set = ["clogloglink","clogloglink()"]
    
    ComputeMethod_optim_set = ["optim","optimization","optimisation","opts"]
    ComputeMethod_mcmc_set = ["mcmc","mc","hmc","mcmc()"]
    ComputeMethod_bootstrap_set = ["bootstrap","boot","boot()","bootstrap()","but","voot","noot","vootstrap","nootstrap"]
    
    ## Check snity for model class
    
    if(modelClass_dumbo in modelClass_reg_set)
    
       modelClass="LinearReg"
       
    elseif (modelClass_dumbo in modelClass_LogisticReg_set)
    
       modelClass="LogisticReg"
       
    end
    
    ## Check snity for likelihood model
    
    if(LikelihoodMod_dumbo in LikelihoodMod_Gauss_set)
    
       LikelihoodMod="Gauss"
       
    elseif (LikelihoodMod_dumbo in LikelihoodMod_Binom_set)
    
       LikelihoodMod="Binomial"
       
    end
    
    ## Check sanity for prior
    
    if (PriorMod_dumbo in PriorMod_NIP_set)
    
       PriorMod="NIP"
       
    elseif (PriorMod_dumbo in PriorMod_Ridge_set)
    
       PriorMod = "Ridge"

    elseif (PriorMod_dumbo in PriorMod_Lasso_set)
      PriorMod = "Laplace"
    end
    
    ## Check snity for link function
    
    if(Link_dumbo in Link_identity_set)
    
       Link="Identity"
       
    elseif (Link_dumbo in Link_probit_set)
    
       Link="ProbitLink"
       
    elseif (Link_dumbo in  Link_logit_set)
       Link="LogitLink"
    
    elseif (Link_dumbo in  Link_cauchy_set)
       Link="CauchitLink"
    
    elseif (Link_dumbo in  Link_cloglog_set)
       Link="CloglogLink"
       
    end
    
    ## Check snity for computational method
    
    if(ComputeMethod_dumbo in ComputeMethod_optim_set)
    
       ComputeMethod="Optimization"
       
    elseif (ComputeMethod_dumbo in ComputeMethod_mcmc_set)
    
       ComputeMethod="MCMC"
    
    elseif (ComputeMethod_dumbo in ComputeMethod_bootstrap_set)
    
       ComputeMethod="Bootstrap"
    end
    
    ##
    
    ans = sanitised_arguement(modelClass,LikelihoodMod,PriorMod,Link,ComputeMethod)
    ans
    
end

## prior and likelihood for linear regression
function LinearReg_neg_LogLike(β,σ,y,X)
   n = size(X, 1);
   nll = n/2*log(2π)+n*log(σ) + ((y-X*β)'*(y-X*β))*(1/(2*σ^2))
   nll
end

function LinReg_Gauss_er_neg_LogLike(θ::Vector,y,X)
   n = size(X, 1);
   p = size(X, 2);
   β = θ[1:p];
   σ = exp(θ[p+1]);
   nll = n/2*log(2π)+n*log(σ) + ((y-X*β)'*(y-X*β))*(1/(2*σ^2))
   nll
end

function neg_log_Ridge_prior(θ::Vector,h)
   p1 = length(θ);
   p = p1-1;
   β = θ[1:p];
   σ = exp(θ[p+1]);
   
   d1 = filldist(Normal(0,σ), p);
   d2 = InverseGamma(h,h);
   neg_log_prior = -logpdf(d1, β)-logpdf(d2,σ);
   neg_log_prior
   
 end


function neg_log_Laplace_prior(θ::Vector,h)
   p1 = length(θ);
   p = p1-1;
   β = θ[1:p];
   σ = exp(θ[p+1]);
   
   d1 = filldist(Laplace(0,σ), p);
   d2 = InverseGamma(h,h);
   neg_log_prior = -logpdf(d1, β)-logpdf(d2,σ);
   neg_log_prior
   #neg_log_prior = sum(abs.(β))*h+1/σ
   #neg_log_prior
 end
 
 function neg_log_Cauchy_prior(θ::Vector,h)
   p1 = length(θ);
   p = p1-1;
   β = θ[1:p];
   σ = exp(θ[p+1]);
   
   d1 = filldist(TDist(1)*σ, p);
   #d2 = Truncated(TDist(1),0.0,Inf);
   d2 = InverseGamma(h,h);
   neg_log_prior = -logpdf(d1, β)-logpdf(d2,σ);
   neg_log_prior
 end

 function neg_log_T_prior(θ)
   p = size(X, 2);
   β = θ[1:p];
   σ = exp(θ[p+1]);
   ν = exp(θ[p+2]);
   
   d1 = filldist(TDist(3)*σ, p);
   d2 = InverseGamma(0.1,0.1);
   d3 = InverseGamma(1,1);
   
   neg_log_prior = -logpdf(d1, β)-logpdf(d2,σ)-logpdf(d3,ν);
   neg_log_prior
 end

 function neg_log_StacknSlab_prior(θ)
   p = size(X, 2);
   β = θ[1:p];
   σ = exp(θ[p+1]);
   
   d0=MixtureModel(Normal, [(0.0, 0.1), (0.0, 1.0)], [0.5, 0.5])
   
   d1 = filldist(d0*σ, p);
   d2 = InverseGamma(0.1,0.1);
   
   neg_log_prior = -logpdf(d1, β)-logpdf(d2,σ);
   neg_log_prior
 end

## prior and likelihood for logistic regression

function LogisticReg_neg_LogLike(beta,y, X)
  z = X*beta
  p = exp.(z) ./ (1 .+ exp.(z))

  nll = y.*log.(p) + (-(y.-1)).*log.(-(p.-1))
  nll = - sum(nll)
  return(nll)
end


function Logit_Link(z::Real)
   1 / (1 + exp(-z))
end

function Cauchit_Link(z::Real)
   0.5+atan(z)/π
end

function Probit_Link(z::Real)
   d = Distributions.Normal(0,1)
   cdf(d,z)
end

function Cloglog_Link(z::Real)
  1-exp(-exp(z))
end

function NegativeBinomial2(μ, ϕ)
   p = 1 / (1 + μ / ϕ)
   r = ϕ

   return NegativeBinomial(r, p)
end
