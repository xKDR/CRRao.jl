
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
   
   ans = MCMC_chain(chain,summaries,quantiles)
   ans
end

function fitmodel(formula::FormulaTerm, data::DataFrame; modelClass::String="LinearReg", LikelihoodMod::String="Gauss", PriorMod::String="NIP", Link::String="Identity", ComputeMethod::String="Optimization", h::Real=0.01, level::Real=0.95, sim_size::Int64=10000)
    ## arguement sanitisation
    sant_args = arguement_sanitisation(modelClass, LikelihoodMod,PriorMod,Link,ComputeMethod)
    modelClass = sant_args.modelClass
    LikelihoodMod = sant_args.LikelihoodMod
    PriorMod = sant_args.PriorMod
    Link = sant_args.Link
    ComputeMethod = sant_args.ComputeMethod
    
    ##-----------------------
    
    if (modelClass=="LinearReg")
    
       #println("Linear Regression Model class is under development")
       
       if (LikelihoodMod=="Gauss")
          
          if (PriorMod=="NIP")
             if (ComputeMethod=="Optimization")
                 
                 #ans = lm_Gauss_NIP_Optim(formula,data)
                 #println("---------------------------------------")
                 #println("Linear Regression with OLS method from GLM package is implemented.")
                 #ans
                 
             elseif (ComputeMethod=="MCMC")
                 
                 #ansBayes = lm_Gauss_MCMC(formula,data,PriorMod,h,sim_size);
                 #println("---------------------------------------")
                 #println("Linear Regression with non informative uniform prior using the Hamiltonian Monter Carlo is being implemented.")
                 #println("Particularly we implemented the NUTS algorithm from Turing.")
                 #ansBayes
                 
             end
          elseif  (PriorMod=="Ridge") 
             if (ComputeMethod=="Optimization")
             
                 #ans = lm_Gauss_Ridge_Optim(formula,data,h,level)
                 #println("---------------------------------------")
                 #println("Linear Regression with Ridge prior is implemented using Optimization. The Ridge regression handles multicillinearity.")
                 
                 #ans
                 
             elseif (ComputeMethod=="MCMC")
             
                 #ans = lm_Gauss_MCMC(formula,data,PriorMod,h,sim_size)
                 #println("Linear Regression with Ridge prior is implemented using MCMC. The Ridge regression handles multicillinearity.")
                 #ans
             end
          elseif (PriorMod=="Laplace")    
             if (ComputeMethod=="Optimization")
 
                #ans =lm_Gauss_General_Optim(formula,data,modelClass
                #                     ,LikelihoodMod,PriorMod,Link
                #                     ,ComputeMethod,h,level)
                #println("Linear Regression with Laplace/Lasso prior is implemented using general purpose optimisation.")
                #println("The Lasso regression handles sparsity.")
                #ans
                
             elseif (ComputeMethod=="MCMC")
                #ans = lm_Gauss_MCMC(formula,data,PriorMod,h,sim_size)
                #println("Linear Regression with Laplace/Lasso prior is implemented using MCMC.")
                #println("The Lasso regression handles sparsity")
                #ans
             end
          elseif (PriorMod=="Cauchy")
             if (ComputeMethod=="Optimization")
 
                #ans =lm_Gauss_General_Optim(formula,data,modelClass
                #                     ,LikelihoodMod,PriorMod,Link
                #                     ,ComputeMethod,h,level)
                #println("Linear Regression with Cauchy prior is implemented using general purpose optimisation package Optim.")
                #println("The regression with Cauchy prior provide robust inference.")
                #ans
                
             elseif (ComputeMethod=="MCMC")
                #ans = lm_Gauss_MCMC(formula,data,PriorMod,h,sim_size)
                #println("Linear Regression with Cauchy prior is implemented using HMC algo with Turing.")
                #println("The regression with Cauchy prior provide robust inference.")
                #ans
             end
          end
         
       elseif (LikelihoodMod=="Laplace")
       
           #dumbo = LikelihoodMod*" distributed error model is under development."
           #println(dumbo)
          
         
       elseif (LikelihoodMod=="t")
       
         #dumbo = LikelihoodMod*" t-distributed error model is under development."
         #println(dumbo)
         
       else
         #dumbo = LikelihoodMod*" model is not in the list of development. You can make a request us to develop this model class at sourish@cmi.ac.in"
         #println(dumbo)
       end
     
     
    
    elseif (modelClass=="CountReg")
       println("Count Regression Model class is under development")
       if (LikelihoodMod=="Poisson")
          
          if (PriorMod=="NIP")
             if (ComputeMethod=="Optimization")
                 #ans = Count_Poisson_NIP_Optim(formula,data,Link)
                 #println("Poisson Regression from GLM package is implemented.")
                 #ans
                 
             elseif (ComputeMethod=="MCMC")
                #println("Poisson Regression with MCMC is under development.")
                 #ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
                 #println("Poisson Regression with uniform flat prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
                 #ans
             end
          elseif (PriorMod=="Ridge")
             if (ComputeMethod=="Optimization")
                #ans = Count_Poisson_NIP_Optim(formula,data,Link)
                #println("Poisson Regression from GLM package is implemented.")
                #ans
                #ans = Count_Poisson_Gen_Optim(formula,data,PriorMod,Link,h,level)
                #println("Poisson Regression with Ridge prior (L2-penalty) on coefficients is implemented.")
                #ans
             elseif (ComputeMethod=="MCMC")
               #println("Poisson Regression with MCMC is under development.")
                #ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
                #println("Poisson Regression with Ridge prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
                #ans
             elseif (ComputeMethod=="Bootstrap")
                #println("Bootstrap method is under development.")
            end
          elseif (PriorMod=="Laplace")
             if (ComputeMethod=="Optimization")
                #ans = Count_Poisson_Gen_Optim(formula,data,PriorMod,Link,h,level)
                #println("Poisson Regression with Laplace prior (L1-penalty) on coefficients is implemented.")
                #ans
 
             elseif (ComputeMethod=="MCMC")
                #ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
                #println("Poisson Regression with Laplace prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
                #ans
             elseif (ComputeMethod=="Bootstrap")
                #println("Bootstrap method is under development.")
            end
          elseif (PriorMod=="Cauchy")
             if (ComputeMethod=="Optimization")
                #ans = Count_Poisson_Gen_Optim(formula,data,PriorMod,Link,h,level)
                #println("Poisson Regression with Cauchy prior on coefficients is implemented.")
                #ans
                
             elseif (ComputeMethod=="MCMC")
                #ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
                #println("Poisson Regression with Cauchy prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
                #ans
             elseif (ComputeMethod=="Bootstrap")
                #println("Bootstrap method is under development.")
            end
          end
       elseif (LikelihoodMod=="NegativeBinomial")
          if (PriorMod=="NIP")
             if (ComputeMethod=="Optimization")
                 #ans = Count_NegativeBinom_NIP_Optim(formula,data,Link)
                 #println("NegativeBinomial Regression from GLM package is implemented.")
                 #ans
                 
             elseif (ComputeMethod=="MCMC")
                println("NegativeBinomial Regression with MCMC is under development.")
                ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
                println("NegativeBinomial Regression with Flat uniform prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
                ans
             elseif (ComputeMethod=="Bootstrap")
                println("Bootstrap method is under development.")
             end
          elseif (PriorMod=="Ridge")
             if (ComputeMethod=="Optimization")
                ans = Count_NegBin_Gen_Optim(formula,data,PriorMod,Link,h,level);
                println("NegativeBinomial Regression with Ridge penalty is implemented using Optim package.")
                
                ans
                
                
            elseif (ComputeMethod=="MCMC")
               ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
               println("NegativeBinomial Regression with Ridge prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
               ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap method is under development.")
            end
          elseif (PriorMod=="Laplace")
             if (ComputeMethod=="Optimization")
                
                ans = Count_NegBin_Gen_Optim(formula,data,PriorMod,Link,h,level);
                println("NegativeBinomial Regression with Laplace prior is implemented using Optim package.");
                ans
                
                
            elseif (ComputeMethod=="MCMC")
               ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
               println("NegativeBinomial Regression with Laplace prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
               ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap method is under development.")
            end
          elseif (PriorMod=="Cauchy")
             if (ComputeMethod=="Optimization")
                
                ans = Count_NegBin_Gen_Optim(formula,data,PriorMod,Link,h,level);
                println("NegativeBinomial Regression with Cauchy prior is implemented using Optim package.");
                ans
                
                
            elseif (ComputeMethod=="MCMC")
               ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
               println("NegativeBinomial Regression with Laplace prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
               ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap method is under development.")
            end
          end
       end 
    else
       dumbo = modelClass*" Model class is not in the list of development. You can make a request us to develop this likelihood model at xyz@abc.com"
       println(dumbo)
    end
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
   
   ans = MCMC_chain(chain,summaries,quantiles)
   
   if (PriorMod=="NIP")
     
   end
   return ans
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
   
   ans = MCMC_chain(chain,summaries,quantiles)
   if (PriorMod=="NIP")
     println("Ibrahim and Laud (JASA, 1990) showed that 
         the uniform flat priors for GLM's can lead to improper
         posterior distributions thus making them undesirable.")
         println("In such cases, the Markov Chain struggles to converge. 
         Even if it converges, results are completely unreliable.")
   end
   ans
end
 