"""
CRRao function implements the Statistical models. The implementation 
   of Statistical models become straightforward for most Julia users 
   with the help of this package. This is going to be wrapper package;
   leveraging the strength of wonderful Julia packages that already exists, 
   such as StatsBase, 
   StatsModels, Distributions,GLM, Turing, DataFrames,
   LinearAlgebra, etc.

   **Note**: You can read more about **Prof C.R. Rao** [Click Here](https://en.wikipedia.org/wiki/C._R._Rao)

CRRao needs six inputs to fit (or train) a model.

### Input:

1. **formula** (explain relationship between the variables)
  `y~x1+x2+...`

2. **data** : it will contain y and X, typically DataFrames (**Notes**: Currently it only works on DataFrames)

3. **model class**: provide name of the model class. For example: lm, logistic, poisson, ARIMA, etc.
  Note: Detail of the model class is provided below.

4. **distribution**
+ data model or likelihood model
+ prior (if NULL - non-informative flat prior is default. In that case the estimates will be MLE.)

5. **link**
  Provide appropriate link function

6. **fitting methodology** :

+ Optimization (Package Dependence: Optim, GLM)
+ MCMC (Package Dependence: Turing, Soss)
+ Bootstrap

### Syntax

```Julia
CRRao(formula::FormulaTerm,data,modelClass="LinearReg",LikelihoodMod="Gauss",PriorMod="NIP",Link::String="Identity"
       ,ComputeMethod::String="Optimization",h::Float64=0.01,level::Real=0.95,sim_size::Int64=10000)

```

+  `formula` Provide the equation
+  `data` Provide DataFrame
+  `modelClass` Provide the name of the class of model. Default is "LinearReg". Other classes are "LogisticReg" for logistic regression, "CountReg" for count regression etc.
+  `LikelihoodMod` Provide the likelihood model (aka., data model). Default is "Gauss" for Gaussian distribution.
+  `PriorMod` Provide the prior model or prior distributions on parameters. Default is NIP, i.e., "Non-Informative Prior" aka. "Flat Prior"
+  `Link` Provide the link function. Default is "Identity".
+  `ComputeMethod` Provide the computational methods. Default is "Optimization". Other methods are MCMC and Bootstrap. The MCMC method implements the NUTS algorithm of Hamiltonian Monte Carlo class. The Bootstrap method is under development.
+  `h` Provide the value of `h` for Ridge prior, where `h>0`. If `h` is close to 0, the the Ridge prior contain no-information. It will be non-informative flat prior. But if you want your Ridge prior to have effect and correct for multicollinearity, then you should try some positive value for `h`. Effectively it assigns `beta ~ N(0 , 1/h)` as prior on unknown coefficients.
+  `level` assign level for confidence interval
+  `sim_size` Provide value for simulation size for MCMC method.



"""
function CRRao(formula::FormulaTerm, data::DataFrame, modelClass::String="LinearReg", LikelihoodMod::String="Gauss", PriorMod::String="NIP", Link::String="Identity", ComputeMethod::String="Optimization", h::Real=0.01, level::Real=0.95, sim_size::Int64=10000)
   ## arguement sanitisation
   sant_args = arguement_sanitisation(modelClass, LikelihoodMod,PriorMod,Link,ComputeMethod)
   modelClass = sant_args.modelClass
   LikelihoodMod = sant_args.LikelihoodMod
   PriorMod = sant_args.PriorMod
   Link = sant_args.Link
   ComputeMethod = sant_args.ComputeMethod
   
   ##-----------------------
   
   if(modelClass=="LinearReg")
   
      #println("Linear Regression Model class is under development")
      
      if (LikelihoodMod=="Gauss")
         
         if (PriorMod=="NIP")
            if (ComputeMethod=="Optimization")
                
                ans = lm_Gauss_NIP_Optim(formula,data)
                println("---------------------------------------")
                println("Linear Regression with OLS method from GLM package is implemented.")
                ans
                
            elseif (ComputeMethod=="MCMC")
                
                ansBayes = lm_Gauss_MCMC(formula,data,PriorMod,h,sim_size);
                println("---------------------------------------")
                println("Linear Regression with non informative uniform prior using the Hamiltonian Monter Carlo is being implemented.")
                println("Particularly we implemented the NUTS algorithm from Turing.")
                ansBayes
                
            elseif (ComputeMethod=="Bootstrap")
                println("Bootstrap methood for Linear Regression model is under development") 
            end
         elseif  (PriorMod=="Ridge") 
            if (ComputeMethod=="Optimization")
            
                ans = lm_Gauss_Ridge_Optim(formula,data,h,level)
                println("---------------------------------------")
                println("Linear Regression with Ridge prior is implemented using Optimization. The Ridge regression handles multicillinearity.")
                
                ans
                
            elseif (ComputeMethod=="MCMC")
            
                ans = lm_Gauss_MCMC(formula,data,PriorMod,h,sim_size)
                println("Linear Regression with Ridge prior is implemented using MCMC. The Ridge regression handles multicillinearity.")
                ans
            elseif (ComputeMethod=="Bootstrap")
                  println("Bootstrap methood for Linear Regression model is under development")  
            end
         elseif (PriorMod=="Laplace")    
            if (ComputeMethod=="Optimization")

               ans =lm_Gauss_General_Optim(formula,data,modelClass
                                    ,LikelihoodMod,PriorMod,Link
                                    ,ComputeMethod,h,level)
               println("Linear Regression with Laplace/Lasso prior is implemented using general purpose optimisation.")
               println("The Lasso regression handles sparsity.")
               ans
               
            elseif (ComputeMethod=="MCMC")
               ans = lm_Gauss_MCMC(formula,data,PriorMod,h,sim_size)
               println("Linear Regression with Laplace/Lasso prior is implemented using MCMC.")
               println("The Lasso regression handles sparsity")
               ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap methood for Linear Regression model is under development") 
            end
         elseif (PriorMod=="Cauchy")
            if (ComputeMethod=="Optimization")

               ans =lm_Gauss_General_Optim(formula,data,modelClass
                                    ,LikelihoodMod,PriorMod,Link
                                    ,ComputeMethod,h,level)
               println("Linear Regression with Cauchy prior is implemented using general purpose optimisation package Optim.")
               println("The regression with Cauchy prior provide robust inference.")
               ans
               
            elseif (ComputeMethod=="MCMC")
               ans = lm_Gauss_MCMC(formula,data,PriorMod,h,sim_size)
               println("Linear Regression with Cauchy prior is implemented using HMC algo with Turing.")
               println("The regression with Cauchy prior provide robust inference.")
               ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap methood is under development") 
            end
         end
        
      elseif (LikelihoodMod=="Laplace")
      
          dumbo = LikelihoodMod*" distributed error model is under development."
          println(dumbo)
         
        
      elseif (LikelihoodMod=="t")
      
        dumbo = LikelihoodMod*" t-distributed error model is under development."
        println(dumbo)
        
      else
        dumbo = LikelihoodMod*" model is not in the list of development. You can make a request us to develop this model class at sourish@cmi.ac.in"
        println(dumbo)
      end
    
    
   elseif (modelClass=="LogisticReg")
      println("Logistic Regression Model class is under development")
      
      if (LikelihoodMod=="Binomial")
         
         if (PriorMod=="NIP")
            if (ComputeMethod=="Optimization")
                ans = logistic_Binom_NIP_Optim(formula,data,Link)
                println("Logistic Regression from GLM package is implemented.")
                ans
                
            elseif (ComputeMethod=="MCMC")
                ans = logistic_Binom_MCMC(formula,data,PriorMod,Link,h,sim_size)
                println("Logistic Regression with uniform prior using the Hamiltonian 
                        Monter Carlo is being implemented. Particularly implemented the 
                        NUTS algorithm from Turing.")
                println("Ibrahim and Laud (JASA, 1990) showed that 
                        the uniform flat priors for GLM's can lead to improper
                        posterior distributions thus making them undesirable.")
                println("In such cases, the Markov Chain struggles to converge. 
                        Even if it converges, results are unreliable.")
                
                ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap methood is under development")
            end
         elseif (PriorMod=="Ridge")
            if (ComputeMethod=="Optimization")
               println("Logistic Regression with Ridge Prior is under development.")
               ans = logistic_Binom_Gen_Optim(formula,data,PriorMod,Link,h,level)
               ans

            elseif (ComputeMethod=="MCMC")
               ans = logistic_Binom_MCMC(formula,data,PriorMod,Link,h,sim_size)
                println("Logistic Regression with flat prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
               ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap methood is under development")
            end
         elseif (PriorMod=="Laplace")
            if (ComputeMethod=="Optimization")
               println("Logistic Regression with Laplace Prior is under development.")
               ans = logistic_Binom_Gen_Optim(formula,data,PriorMod,Link,h,level)
               ans
            elseif (ComputeMethod=="MCMC")
               ans = logistic_Binom_MCMC(formula,data,PriorMod,Link,h,sim_size)
                println("Logistic Regression with Laplace prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
               ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap methood is under development")
            end
         elseif (PriorMod=="Cauchy")
            if (ComputeMethod=="Optimization")
               println("Logistic Regression with Cauchy Prior is under development.")
               ans = logistic_Binom_Gen_Optim(formula,data,PriorMod,Link,h,level)
               ans
            elseif (ComputeMethod=="MCMC")
               ans = logistic_Binom_MCMC(formula,data,PriorMod,Link,h,sim_size)
                println("Logistic Regression with Cauchy prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
               ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap methood is under development")
            end
         end
      end
   elseif (modelClass=="CountReg")
      println("Count Regression Model class is under development")
      if (LikelihoodMod=="Poisson")
         
         if (PriorMod=="NIP")
            if (ComputeMethod=="Optimization")
                ans = Count_Poisson_NIP_Optim(formula,data,Link)
                println("Poisson Regression from GLM package is implemented.")
                ans
                
            elseif (ComputeMethod=="MCMC")
               println("Poisson Regression with MCMC is under development.")
                ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
                println("Poisson Regression with uniform flat prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
                ans
            end
         elseif (PriorMod=="Ridge")
            if (ComputeMethod=="Optimization")
               #ans = Count_Poisson_NIP_Optim(formula,data,Link)
               #println("Poisson Regression from GLM package is implemented.")
               #ans
               ans = Count_Poisson_Gen_Optim(formula,data,PriorMod,Link,h,level)
               println("Poisson Regression with Ridge prior (L2-penalty) on coefficients is implemented.")
               ans
            elseif (ComputeMethod=="MCMC")
              println("Poisson Regression with MCMC is under development.")
               ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
               println("Poisson Regression with Ridge prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
               ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap method is under development.")
           end
         elseif (PriorMod=="Laplace")
            if (ComputeMethod=="Optimization")
               ans = Count_Poisson_Gen_Optim(formula,data,PriorMod,Link,h,level)
               println("Poisson Regression with Laplace prior (L1-penalty) on coefficients is implemented.")
               ans

            elseif (ComputeMethod=="MCMC")
               ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
               println("Poisson Regression with Laplace prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
               ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap method is under development.")
           end
         elseif (PriorMod=="Cauchy")
            if (ComputeMethod=="Optimization")
               ans = Count_Poisson_Gen_Optim(formula,data,PriorMod,Link,h,level)
               println("Poisson Regression with Cauchy prior on coefficients is implemented.")
               ans
               
            elseif (ComputeMethod=="MCMC")
               ans = Count_Reg_MCMC(formula,data,LikelihoodMod,PriorMod,h,sim_size)
               println("Poisson Regression with Cauchy prior using the Hamiltonian Monter Carlo is being implemented. Particularly implemented the NUTS algorithm from Turing.")
               ans
            elseif (ComputeMethod=="Bootstrap")
               println("Bootstrap method is under development.")
           end
         end
      elseif (LikelihoodMod=="NegativeBinomial")
         if (PriorMod=="NIP")
            if (ComputeMethod=="Optimization")
                ans = Count_NegativeBinom_NIP_Optim(formula,data,Link)
                println("NegativeBinomial Regression from GLM package is implemented.")
                ans
                
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
