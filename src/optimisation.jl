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
  LogPost::Float64
  AIC::Float64
  BIC::Float64
  R_sqr::Float64
  Adjusted_R_sqr::Float64
  sigma::Float64
  fittedResponse
  residuals
  Cooks_distance
end

function lm_Gauss_NIP_Optim(formula::FormulaTerm,data::DataFrame)
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
    ,Link,ComputeMethod,res,fit, logLike, logLike, AIC, BIC, R_sqr
    ,Adjusted_R_sqr,sigma, fittedResponse,resid,cooks_distance)

    return ans
      
end

function lm_Gauss_NIP_Optim_predicts(obj,newdata::DataFrame)
  formula = obj.formula;
  fm_frame=ModelFrame(formula,newdata);
  X=modelmatrix(fm_frame);
  beta = coef(obj.res)
  y_pred = X*beta;
  y_pred
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

## logistic regression

struct analysis_logistic_Binom_NIP_Optim
  formula::FormulaTerm
  modelClass::String
  LikelihoodMod::String
  PriorMod::String
  Link::String
  ComputeMethod::String
  fit
  beta
  LogLike::Float64
  LogPost::Float64
  AIC::Float64
  BIC::Float64
  #R_sqr::Float64
  #Adjusted_R_sqr::Float64
  #sigma::Float64
  #Cooks_distance
end

function logistic_Binom_NIP_Optim(formula::FormulaTerm,data,Link::String="LogitLink")
    modelClass = "LogisticReg";
    LikelihoodMod="Binomial";
    PriorMod="NIP";
    ComputeMethod="Optimization";
    formula = apply_schema(formula, schema(formula, data));
    y, X = modelcols(formula, data);
    fm_frame=ModelFrame(formula,data);
    X=modelmatrix(fm_frame);
    p = size(X, 2);
    n = size(X, 1);
    
    if (Link=="LogitLink")
        res = glm(formula,data,Binomial(), LogitLink());
        
    elseif (Link=="ProbitLink")
        res = glm(formula,data,Binomial(), ProbitLink());
        
    elseif (Link=="CauchitLink")
        res = glm(formula,data,Binomial(), CauchitLink());
    
    elseif (Link=="CloglogLink")
        res = glm(formula,data,Binomial(), CloglogLink());
    
    else
        println("This link function is not part of logistic regression family.")
        println("-------------------------------------------------------------")
    end
    
    fit = coeftable(res)
    beta_hat = coef(res)
    logLike = GLM.loglikelihood(res)
    LogPost = logLike
    npar = p;
    AIC = 2*npar - 2*logLike
    BIC = log(n)*npar - 2*logLike
    
    ans = analysis_logistic_Binom_NIP_Optim(formula,modelClass,LikelihoodMod
            ,PriorMod,Link,ComputeMethod,fit,beta_hat
            ,logLike,LogPost,AIC,BIC)
    ans

end

function logistic_Binom_NIP_Optim_predicts(obj,newdata::DataFrame)
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

## Poisson regression


struct analysis_Poisson_Regression
  formula::FormulaTerm
  modelClass::String
  LikelihoodMod::String
  PriorMod::String
  Link::String
  ComputeMethod::String
  fit
  beta
  LogLike::Float64
  LogPost::Float64
  AIC::Float64
  BIC::Float64
  #R_sqr::Float64
  #Adjusted_R_sqr::Float64
  #sigma::Float64
  #Cooks_distance
end


function Count_Poisson_NIP_Optim(formula::FormulaTerm,data,Link::String="LogLink")
  modelClass = "CountReg";
  LikelihoodMod="Poisson";
  PriorMod="NIP";
  ComputeMethod="Optimization";
  formula = apply_schema(formula, schema(formula, data));
  y, X = modelcols(formula, data);
  fm_frame=ModelFrame(formula,data);
  X=modelmatrix(fm_frame);
  p = size(X, 2);
  n = size(X, 1);
  
  if(Link=="LogLink")
      res = glm(formula,data,Poisson(), LogLink());
  else
      println("This link function is not part of Poisson regression family.")
      println("-------------------------------------------------------------")
  end
  
  fit = coeftable(res)
  beta_hat = coef(res)
  logLike = GLM.loglikelihood(res)
  LogPost = logLike
  npar = p;
  AIC = 2*npar - 2*logLike
  BIC = log(n)*npar - 2*logLike
  
  ans = analysis_Count_Poisson_NIP_Optim(formula,modelClass,LikelihoodMod
          ,PriorMod,Link,ComputeMethod,fit,beta_hat
          ,logLike,LogPost,AIC,BIC)
  ans

end

function Count_Poisson_NIP_Optim_predicts(obj,newdata::DataFrame)
  formula = obj.formula;
  fm_frame=ModelFrame(formula,newdata);
  X=modelmatrix(fm_frame);
  beta = obj.beta
  z = X*beta;

  if(obj.Link=="LogLink")
    p = exp.(z) ;
      
  else
      println("This link function is not part of Poisson regression family.")
      println("-------------------------------------------------------------")
  end 
  p
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
  LogPost::Float64
  AIC::Float64
  BIC::Float64
  #R_sqr::Float64
  #Adjusted_R_sqr::Float64
  #sigma::Float64
  #Cooks_distance
end



function Count_NegativeBinom_NIP_Optim(formula::FormulaTerm,data,Link::String="LogLink")
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
  
  if(Link=="LogLink")
      res = glm(formula,data,NegativeBinomial(), LogLink());
  else
      println("This link function is not part of Poisson regression family.")
      println("-------------------------------------------------------------")
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
          ,logLike,LogPost,AIC,BIC)
  ans

end


function Count_NegBinom_NIP_Optim_predicts(obj,newdata::DataFrame)
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
