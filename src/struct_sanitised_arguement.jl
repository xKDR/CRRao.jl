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
 
struct analysis_lm_Gauss_Gen_Optim
    formula::FormulaTerm
    modelClass::String
    LikelihoodMod::String
    PriorMod::String
    Link::String
    ComputeMethod::String
    fit::DataFrame
    CoefPostMode
    CoefPostSd
    CoefPostLower
    CoefPostUpper
    SigmaPostMode
    SigmaPostLower
    SigmaPostUpper
    LogLike::Float64
    AIC::Float64
    BIC::Float64
  end

  
function lm_Gauss_Ridge_Optim(formula::FormulaTerm,data::DataFrame,h::Float64=0.0001,level::Real=0.95)
    modelClass = "LinearReg";
    LikelihoodMod="Gauss";
    PriorMod="RidgePrior";
    Link = "Identity";
    ComputeMethod="Optimization";
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    n = size(X, 1);
    p = size(X, 2);
  
    beta_pred = coefnames(formula)[2];
    beta_nms = Vector{String}()
    push!(beta_nms, "Intercept")
    for i in 1:length(beta_pred)
        push!(beta_nms, beta_pred[i]) 
    end
  
    
    beta_hat = inv(X'X)X'y;
    
    IdMat =Matrix(1.0I, p, p) 
    Lambda0 =h*IdMat
    a0=1
    b0=1
    
    Lambda1 = X'X+Lambda0
    cov_beta = inv(Lambda1)
    beta_bar= cov_beta*X'y
    a1 = a0+(n/2)
    b1 = b0+0.5*(y'y-beta_bar'*Lambda1*beta_bar)
    sigma_sqr_bar = b1/(a1-1)
    sigma_bar = sqrt(sigma_sqr_bar)
    
    sd_bata_bar = sigma_bar*sqrt.(cov_beta[diagind(cov_beta)])
    ci = sd_bata_bar*quantile(Normal(0,1), (1-level)/2)
    Lower = beta_bar+ci
    Upper = beta_bar-ci
    
    sigma_sqr_bar_lower, sigma_sqr_bar_upper = quantile.(InverseGamma(a1,b1),[0.025,0.975])
    sigma_bar_lower = sqrt(sigma_sqr_bar_lower)
    sigma_bar_upper = sqrt(sigma_sqr_bar_upper)
    
    fit = DataFrame(hcat(beta_nms,beta_bar,sd_bata_bar,Lower,Upper));
  rename!(fit,:x1 =>:"Predictors",:x2 =>:"Posterior Mode",:x3 =>:"Posterior Sd",:x4 =>:"Lower Bound",:x5 =>:"Upper Bound");
  
    
    loglike = LinearReg_neg_LogLike(beta_bar,sigma_bar,y,X);
    loglike= -1*loglike;
    npar =(p+1);
    AIC = 2*npar - 2*loglike
    BIC = log(n)*npar - 2*loglike
    
    ans = analysis_lm_Gauss_Gen_Optim(formula,modelClass,LikelihoodMod,PriorMod
    ,Link,ComputeMethod,fit,beta_bar,sd_bata_bar,Lower,Upper
    ,sigma_bar,sigma_bar_lower,sigma_bar_upper
    ,loglike,AIC,BIC)
  
    ans
end
function lm_Gauss_Ridge_Optim_predicts(obj,newdata::DataFrame)
    formula = obj.formula;
    fm_frame=ModelFrame(formula,newdata);
    X=modelmatrix(fm_frame);
    beta = obj.CoefPostMode
    y_pred = X*beta;
    y_pred  
end
function lm_Gauss_General_Optim(formula::FormulaTerm,data::DataFrame,modelClass::String="LinearReg",LikelihoodMod::String="Gauss",PriorMod::String="NIP",Link::String="Identity",ComputeMethod::String="Optimization",h::Float64=0.01,level::Real=0.95)

    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    n = size(X, 1);
    p = size(X, 2);
  
    beta_pred = coefnames(formula)[2];
    theta_nms = Vector{String}()
    push!(theta_nms, "Intercept")
    for i in 1:length(beta_pred)
        push!(theta_nms, beta_pred[i]) 
    end
    push!(theta_nms,"sigma_hat")
    
    
    beta_hat = inv(X'X)X'y
    y_hat = X*beta_hat
    sigma_hat = sqrt((y-y_hat)'*(y-y_hat)/(n-p))
  
    initial_values = append!(beta_hat,log(sigma_hat));
  
    
    function LinearReg_neg_LogPost(θ::Vector,y,X,h)
      n = size(X, 1);
      p = size(X, 2);
      β = θ[1:p];
      σ = exp(θ[p+1]);
      if(LikelihoodMod=="Gauss")
        neg_log_like = n/2*log(2π) + n*log(σ) + ((y-X*β)'*(y-X*β))*(1/(2*σ^2))
      elseif (LikelihoodMod=="Laplace")
        neg_log_like = n/2*log(2π)+n*log(σ) + ((y-X*β)'*(y-X*β))*(1/(2*σ^2))
      end
   
      if (PriorMod=="Laplace")
        neg_log_prior = neg_log_Laplace_prior(θ,h);
      elseif (PriorMod=="Cauchy")
        neg_log_prior = neg_log_Cauchy_prior(θ);
      elseif (PriorMod=="Stack&Slab")
        neg_log_prior = neg_log_StacknSlab_prior(θ);
      end
   
      neg_log_post = neg_log_like + neg_log_prior;
      neg_log_post
    end
    #d1 = DifferentiableFunction(LinearReg_neg_LogPost);
    func = TwiceDifferentiable(vars -> LinearReg_neg_LogPost(vars,y,X,h)
                                      ,initial_values
                                      ; autodiff=:forward);
  
    res = optimize(func,initial_values,Newton()
                      ,Optim.Options(x_tol=1e-32
                        ,f_tol =1e-32
                        ,g_tol = 1e-64
                        ,f_calls_limit=10
                        ,g_calls_limit=10
                        ,h_calls_limit=10
                        ,iterations = 10000));
    theta_hat = Optim.minimizer(res)
    theta_hat = round.(theta_hat,digits=5)
  
    beta_bar = theta_hat[1:p];
    
    sigma_hat=exp(theta_hat[p+1])
    
  
    numerical_hessian = hessian!(func,theta_hat)
    var_cov_matrix = inv(numerical_hessian)
    post_var = diag(var_cov_matrix)
    post_sd = real.(sqrt.(Complex.(post_var)))
    #post_sd = sqrt.(post_var);
    post_sd_beta = post_sd[1:p]
  
    ci = post_sd_beta*quantile(Normal(0,1), (1-level)/2);
    Lower = beta_bar+ci;
    Upper = beta_bar-ci;
    
  
    fit = DataFrame(hcat(theta_nms[1:p],beta_bar,post_sd_beta,Lower,Upper));
    rename!(fit,:x1 =>:"Predictors",:x2 =>:"Posterior Mode",:x3 =>:"Posterior Sd"
    ,:x4 =>:"Lower Bound",:x5 =>:"Upper Bound");
  
    
  
    post_var_sigma = exp(theta_hat[p+1])^2*post_var[p+1]
    post_sd_sigma = sqrt(post_var_sigma)
    ci = post_sd_sigma*quantile(Normal(0,1), (1-level)/2);
    sigma_hat_lower=sigma_hat+ci
    sigma_hat_upper=sigma_hat-ci
    
    loglike = Optim.minimum(res)
    loglike= -1*loglike;
    npar =(p+1);
    AIC = 2*npar - 2*loglike
    BIC = log(n)*npar - 2*loglike
    
    ans = analysis_lm_Gauss_Gen_Optim(formula,modelClass,LikelihoodMod,PriorMod
    ,Link,ComputeMethod,fit,beta_bar,post_sd_beta,Lower,Upper
    ,sigma_hat,sigma_hat_lower,sigma_hat_upper,loglike,AIC,BIC)
    
    ans
end

struct analysis_GLM_Gen_Optim
    formula::FormulaTerm
    modelClass::String
    LikelihoodMod::String
    PriorMod::String
    Link::String
    ComputeMethod::String
    fit::DataFrame
    CoefPostMode
    CoefPostSd
    CoefPostLower
    CoefPostUpper
    LambdaPostMode
    LambdaPostLower
    LambdaPostUpper
    LogLike::Float64
    AIC::Float64
    BIC::Float64
end

function logistic_Binom_Gen_Optim(formula::FormulaTerm,data,PriorMod::String="Ridge",Link::String="LogitLink",h::Real=0.01,level::Real=0.95)
    modelClass = "LogisticReg";
    LikelihoodMod="Binomial";
    ComputeMethod="Optimization";
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    p = size(X, 2);
    n = size(X, 1);

    beta_pred = coefnames(formula)[2];
    theta_nms = Vector{String}()
    push!(theta_nms, "Intercept")
    for i in 1:length(beta_pred)
        push!(theta_nms, beta_pred[i]) 
    end
    push!(theta_nms,"lambda_hat")

    if (Link=="LogitLink")
        res = glm(formula,data,Binomial(), LogitLink());
        
    elseif (Link=="ProbitLink")
        res = glm(formula,data,Binomial(), ProbitLink());
        
    elseif (Link=="CauchitLink")
        res = glm(formula,data,Binomial(), CauchitLink());

    elseif (Link=="CloglogLink")
        res = glm(formula,data,Binomial(), CloglogLink());

    end

    println(res)

    beta_hat = coef(res);

    initial_values = append!(beta_hat,log(h));

    function LogisticReg_neg_LogPost(θ::Vector,y,X,h)
        n = size(X, 1);
        p = size(X, 2);
        β = θ[1:p];
        
        z = X*β;

        if (Link=="LogitLink")
        prob = Logit_Link.(z);
            
        elseif (Link=="ProbitLink")
        prob = Probit_Link.(z);
            
        elseif (Link=="CauchitLink")
        prob = Cauchit_Link.(z);

        elseif (Link=="CloglogLink")
        prob = Cloglog_Link.(z);
        end

        neg_log_like = y.*log.(prob) + (-(y.-1)).*log.(-(prob.-1));
        neg_log_like = - sum(neg_log_like);

        if (PriorMod=="Laplace")
        neg_log_prior = neg_log_Laplace_prior(θ,h);
        elseif (PriorMod=="Cauchy")
        neg_log_prior = neg_log_Cauchy_prior(θ);
        elseif (PriorMod=="Ridge")
        neg_log_prior = neg_log_Ridge_prior(θ,h);
        elseif (PriorMod=="Stack&Slab")
        neg_log_prior = neg_log_StacknSlab_prior(θ);
        end

        neg_log_post = neg_log_like + neg_log_prior;
        neg_log_post
    end

    func = TwiceDifferentiable(vars -> LogisticReg_neg_LogPost(vars,y,X,h)
                                        ,initial_values
                                        ; autodiff=:forward);

    res = optimize(func,initial_values,Newton()
                        ,Optim.Options(x_tol=1e-32
                        ,f_tol =1e-32
                        ,g_tol = 1e-64
                        ,f_calls_limit=10
                        ,g_calls_limit=10
                        ,h_calls_limit=10
                        ,iterations = 10000));
                        theta_hat = Optim.minimizer(res)
                        theta_hat = round.(theta_hat,digits=5)
                        
    beta_bar = theta_hat[1:p];

    lambda_hat=exp(theta_hat[p+1]);


    numerical_hessian = hessian!(func,theta_hat)
    var_cov_matrix = inv(numerical_hessian)
    println(var_cov_matrix);
    post_var = diag(var_cov_matrix)
    post_sd = real.(sqrt.(Complex.(post_var)))
    #post_sd = sqrt.(post_var);
    post_sd_beta = post_sd[1:p]

    ci = post_sd_beta*quantile(Normal(0,1), (1-level)/2);
    Lower = beta_bar+ci;
    Upper = beta_bar-ci;

    beta_bar = round.(beta_bar,digits=5);
    post_sd_beta = round.(post_sd_beta,digits=5);
    Lower = round.(Lower,digits=5);
    Upper = round.(Upper,digits=5);

    fit = DataFrame(hcat(theta_nms[1:p],beta_bar,post_sd_beta,Lower,Upper));
    rename!(fit,:x1 =>:"Predictors",:x2 =>:"Posterior Mode",:x3 =>:"Posterior Sd"
    ,:x4 =>:"Lower Bound",:x5 =>:"Upper Bound");


    post_var_lambda = exp(theta_hat[p+1])^2*post_var[p+1]
    post_sd_lambda = real(sqrt(Complex(post_var_lambda)))
    ci = post_sd_lambda*quantile(Normal(0,1), (1-level)/2);
    lambda_hat_lower=lambda_hat+ci
    lambda_hat_upper=lambda_hat-ci
    if (lambda_hat_lower<0.0) 
        lambda_hat_lower=0.0;
    end
    #lambda_hat_lower=nothing
    #lambda_hat_upper=nothing


    loglike = Optim.minimum(res)
    loglike= -1*loglike;
    npar =(p+1);
    AIC = 2*npar - 2*loglike
    BIC = log(n)*npar - 2*loglike

    ans = analysis_GLM_Gen_Optim(formula,modelClass,LikelihoodMod,PriorMod
    ,Link,ComputeMethod,fit,beta_bar,post_sd_beta,Lower,Upper
    ,lambda_hat,lambda_hat_lower,lambda_hat_upper,loglike,AIC,BIC)

    ans
end

function Count_Poisson_Gen_Optim(formula::FormulaTerm,data,PriorMod::String="Ridge",Link::String="LogLink",h::Real=0.01,level::Real=0.95)
    modelClass = "CountReg";
    LikelihoodMod="Poisson";
    ComputeMethod="Optimization";
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    p = size(X, 2);
    n = size(X, 1);
  
    beta_pred = coefnames(formula)[2];
    theta_nms = Vector{String}()
    push!(theta_nms, "Intercept")
    for i in 1:length(beta_pred)
        push!(theta_nms, beta_pred[i]) 
    end
    push!(theta_nms,"lambda_hat")
    
    res = glm(formula,data,Poisson(), LogLink());
    beta_hat = coef(res);
    initial_values = append!(beta_hat,log(h));
  
    function PoissonReg_neg_LogPost(θ::Vector,y,X,h)
      n = size(X, 1);
      p = size(X, 2);
      β = θ[1:p];
      
      z = X*β;
      μ = exp.(z);
      d = Poisson.(μ);
      neg_log_like = logpdf.(d,y);
      neg_log_like = - sum(neg_log_like);
  
      if (PriorMod=="Laplace")
        neg_log_prior = neg_log_Laplace_prior(θ,h);
      elseif (PriorMod=="Cauchy")
        neg_log_prior = neg_log_Cauchy_prior(θ,h);
      elseif (PriorMod=="Ridge")
        neg_log_prior = neg_log_Ridge_prior(θ,h);
      elseif (PriorMod=="Stack&Slab")
        neg_log_prior = neg_log_StacknSlab_prior(θ);
      end
   
      neg_log_post = neg_log_like + neg_log_prior;
      neg_log_post
    end
  
    func = TwiceDifferentiable(vars -> PoissonReg_neg_LogPost(vars,y,X,h)
                                      ,initial_values
                                      ; autodiff=:forward);
  
    res = optimize(func,initial_values,Newton()
                      ,Optim.Options(x_tol=1e-32
                        ,f_tol =1e-32
                        ,g_tol = 1e-64
                        ,f_calls_limit=10
                        ,g_calls_limit=10
                        ,h_calls_limit=10
                        ,iterations = 10000));
                        theta_hat = Optim.minimizer(res)
                        theta_hat = round.(theta_hat,digits=5)
                      
    beta_bar = theta_hat[1:p];
    
    lambda_hat=exp(theta_hat[p+1]);
    
    numerical_hessian = hessian!(func,theta_hat)
    var_cov_matrix = inv(numerical_hessian)
    post_var = diag(var_cov_matrix)
    post_sd = real.(sqrt.(Complex.(post_var)))
    #post_sd = sqrt.(post_var);
    post_sd_beta = post_sd[1:p]
  
    ci = post_sd_beta*quantile(Normal(0,1), (1-level)/2);
    Lower = beta_bar+ci;
    Upper = beta_bar-ci;
  
    beta_bar = round.(beta_bar,digits=5);
    post_sd_beta = round.(post_sd_beta,digits=5);
    Lower = round.(Lower,digits=5);
    Upper = round.(Upper,digits=5);
  
    fit = DataFrame(hcat(theta_nms[1:p],beta_bar,post_sd_beta,Lower,Upper));
    rename!(fit,:x1 =>:"Predictors",:x2 =>:"Posterior Mode",:x3 =>:"Posterior Sd"
    ,:x4 =>:"Lower Bound",:x5 =>:"Upper Bound");
  
    post_var_lambda = exp(theta_hat[p+1])^2*post_var[p+1]
    post_sd_lambda = real(sqrt(Complex(post_var_lambda)))
    ci = post_sd_lambda*quantile(Normal(0,1), (1-level)/2);
    lambda_hat_lower=lambda_hat+ci
    lambda_hat_upper=lambda_hat-ci
    if (lambda_hat_lower<0.0) 
      lambda_hat_lower=0.0;
    end
    #lambda_hat_lower=nothing
    #lambda_hat_upper=nothing
  
    loglike = Optim.minimum(res);
    loglike= -1*loglike;
    npar =(p+1);
    AIC = 2*npar - 2*loglike;
    BIC = log(n)*npar - 2*loglike;
    
    ans = analysis_GLM_Gen_Optim(formula,modelClass,LikelihoodMod,PriorMod
          ,Link,ComputeMethod,fit,beta_bar,post_sd_beta,Lower,Upper
          ,lambda_hat,lambda_hat_lower,lambda_hat_upper,loglike,AIC,BIC);
    
    ans
end

function Count_NegBin_Gen_Optim(formula::FormulaTerm,data,PriorMod::String="Ridge",Link::String="LogLink",h::Real=0.01,level::Real=0.95)
    modelClass = "CountReg";
    LikelihoodMod="NegativeBinomial";
    ComputeMethod="Optimization";
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    p = size(X, 2);
    n = size(X, 1);
  
    beta_pred = coefnames(formula)[2];
    theta_nms = Vector{String}()
    push!(theta_nms, "Intercept")
    for i in 1:length(beta_pred)
        push!(theta_nms, beta_pred[i]) 
    end
    push!(theta_nms,"lambda_hat")
    
    res = glm(formula,data,Poisson(), LogLink());
    beta_hat = coef(res);
    initial_values = append!(beta_hat,log(h));
  
    function NegBinReg_neg_LogPost(θ::Vector,y,X,h)
      n = size(X, 1);
      p = size(X, 2);
      β = θ[1:p];
      λ = exp(θ[p+1]);
      
      z = X*β;
      μ = exp.(z);
      d = NegativeBinomial2.(μ,λ);
      neg_log_like = logpdf.(d,y);
      neg_log_like = - sum(neg_log_like);
  
      if (PriorMod=="Laplace")
        neg_log_prior = neg_log_Laplace_prior(θ,h);
      elseif (PriorMod=="Cauchy")
        neg_log_prior = neg_log_Cauchy_prior(θ,h);
      elseif (PriorMod=="Ridge")
        neg_log_prior = neg_log_Ridge_prior(θ,h);
      elseif (PriorMod=="Stack&Slab")
        neg_log_prior = neg_log_StacknSlab_prior(θ);
      end
   
      neg_log_post = neg_log_like + neg_log_prior;
      neg_log_post
    end
  
    func = TwiceDifferentiable(vars -> NegBinReg_neg_LogPost(vars,y,X,h)
                                      ,initial_values
                                      ; autodiff=:forward);
  
    res = optimize(func,initial_values,Newton()
                      ,Optim.Options(x_tol=1e-32
                        ,f_tol =1e-32
                        ,g_tol = 1e-64
                        ,f_calls_limit=10
                        ,g_calls_limit=10
                        ,h_calls_limit=10
                        ,iterations = 10000));
                        theta_hat = Optim.minimizer(res)
                        theta_hat = round.(theta_hat,digits=5)
                      
    beta_bar = theta_hat[1:p];
    
    lambda_hat=exp(theta_hat[p+1]);
    
    numerical_hessian = hessian!(func,theta_hat)
    var_cov_matrix = inv(numerical_hessian)
    post_var = diag(var_cov_matrix)
    post_sd = real.(sqrt.(Complex.(post_var)))
    #post_sd = sqrt.(post_var);
    post_sd_beta = post_sd[1:p]
  
    ci = post_sd_beta*quantile(Normal(0,1), (1-level)/2);
    Lower = beta_bar+ci;
    Upper = beta_bar-ci;
  
    beta_bar = round.(beta_bar,digits=5);
    post_sd_beta = round.(post_sd_beta,digits=5);
    Lower = round.(Lower,digits=5);
    Upper = round.(Upper,digits=5);
  
    fit = DataFrame(hcat(theta_nms[1:p],beta_bar,post_sd_beta,Lower,Upper));
    rename!(fit,:x1 =>:"Predictors",:x2 =>:"Posterior Mode",:x3 =>:"Posterior Sd"
    ,:x4 =>:"Lower Bound",:x5 =>:"Upper Bound");
  
    post_var_lambda = exp(theta_hat[p+1])^2*post_var[p+1]
    post_sd_lambda = real(sqrt(Complex(post_var_lambda)))
    ci = post_sd_lambda*quantile(Normal(0,1), (1-level)/2);
    lambda_hat_lower=lambda_hat+ci
    lambda_hat_upper=lambda_hat-ci
    if (lambda_hat_lower<0.0) 
      lambda_hat_lower=0.0;
    end
    #lambda_hat_lower=nothing
    #lambda_hat_upper=nothing
  
    loglike = Optim.minimum(res);
    loglike= -1*loglike;
    npar =(p+1);
    AIC = 2*npar - 2*loglike;
    BIC = log(n)*npar - 2*loglike;
    
    ans = analysis_GLM_Gen_Optim(formula,modelClass,LikelihoodMod,PriorMod
          ,Link,ComputeMethod,fit,beta_bar,post_sd_beta,Lower,Upper
          ,lambda_hat,lambda_hat_lower,lambda_hat_upper,loglike,AIC,BIC);
    
    ans
end
  
  
  
  