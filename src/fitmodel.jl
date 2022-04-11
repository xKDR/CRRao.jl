"""
CRRao is a Julia package that implements the Statistical models. The implementation 
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

3. **model class**: provide name of the model class. For example: LinearRegression
    , LogisticRegression, PoissonRegression, ARIMA, etc.
  Note: Detail of the model class is provided below.

4. **prior distribution**
+ provide prior distribution if you want to implement Bayesian models. 
+ Leave it blank if you want to implement classical likelihood models.

5. **link**
  Provide appropriate link function for GLM models



### Syntax


+  `formula` Provide the equation
+  `data` Provide DataFrame
+  `modelClass` Provide the name of the class of model. Default is "LinearReg". Other classes are "LogisticReg" for logistic regression, "CountReg" for count regression etc.
+  `PriorMod` Provide the prior model or prior distributions on parameters. 
+  `Link` Provide the link function for GLM 
+  `h` Provide the value of `h` for Ridge prior, where `h>0`. If `h` is close to 0, the the Ridge prior contain no-information. It will be non-informative flat prior. But if you want your Ridge prior to have effect and correct for multicollinearity, then you should try some positive value for `h`. Effectively it assigns `beta ~ N(0 , 1/h)` as prior on unknown coefficients.
+  `level` assign level for confidence interval
+  `sim_size` Provide value for simulation size for MCMC method.

```Julia

Julia> using RDatasets, StatsModels, StatsPlots, NLSolversBase

 
## Example: Linear Regression

Julia> df = dataset("datasets", "mtcars");

Julia> model = fitmodel(@formula(MPG ~ HP + WT+Gear),df,LinearRegression());

Julia> model.fit

────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      t  Pr(>|t|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  32.0137     4.63226      6.91    <1e-06  22.5249     41.5024
HP           -0.0367861  0.00989146  -3.72    0.0009  -0.0570478  -0.0165243
WT           -3.19781    0.846546    -3.78    0.0008  -4.93188    -1.46374
Gear          1.01998    0.851408     1.20    0.2410  -0.72405     2.76401
────────────────────────────────────────────────────────────────────────────

Julia> model.sigma
2.5741691724978977

Julia> model.LogLike
-73.52638935960971

Julia> model.AIC
157.05277871921942

Julia> model.BIC
164.38145823321804

Julia> model.R_sqr
0.8352309600685555

Julia> model.Adjusted_R_sqr
0.8175771343616149

Julia> model.fittedResponse
32-element Vector{Float64}:
 23.66884995233871
 22.85340824320635
 25.25355614074087
 20.746171762311327
 17.635570543830113
 ...

Julia> model.residuals
32-element Vector{Float64}:
 -2.668849952338711
 -1.8534082432063492
 -2.4535561407408686
  0.6538282376886713
  ...

Julia> model.Cooks_distance
32-element Vector{Float64}:
 0.013342034282302845
 0.00688728266731234
 0.015495847517058972
 ...

Julia> plot(model.Cooks_distance)

## Example: Bayesian Linear Regression with Ridge Prior


Julia> model = fitmodel(@formula(MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Ridge());

┌ Info: Found initial step size
└   ϵ = 0.003125

Julia> model.summaries
Summary Statistics
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64     Float64   Float64       Float64 

           v    6.8184    3.8850     0.0388    0.0680   3316.5165    1.0003      266.6868
           σ    2.6816    0.3879     0.0039    0.0064   3526.9739    1.0003      283.6100
           α   28.4108    5.5547     0.0555    0.1075   2390.7583    1.0009      192.2450
        β[1]   -0.0402    0.0107     0.0001    0.0002   3503.1094    1.0008      281.6910
        β[2]   -2.6507    0.9780     0.0098    0.0189   2486.6245    1.0008      199.9537
        β[3]    1.6426    1.0086     0.0101    0.0189   2522.2853    1.0011      202.8213

## All rhat values are close to 1; indicates convergence of Markov Chain.


### LogisticReg - GLM with Scoring Method

Julia> turnout = dataset("Zelig", "turnout")

Julia> model = fitmodel(@formula(Vote ~ Age + Race +Income + Educate)
                ,turnout,LogisticRegression(),Logit());

Julia> model.fit

────────────────────────────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)   Lower 95%   Upper 95%
────────────────────────────────────────────────────────────────────────────
(Intercept)  -3.03426    0.325927    -9.31    <1e-19  -3.67307    -2.39546
Age           0.0283543  0.00346034   8.19    <1e-15   0.0215722   0.0351365
Race: white   0.250798   0.146457     1.71    0.0868  -0.0362521   0.537847
Income        0.177112   0.0271516    6.52    <1e-10   0.123896    0.230328
Educate       0.175634   0.0203308    8.64    <1e-17   0.135786    0.215481
────────────────────────────────────────────────────────────────────────────

Julia> model.LogLike
-1011.9906318515576

Julia> model.AIC
2033.9812637031152


```

"""
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression)
   ## Linear Regression - Classical

   ans = lm_Gauss_NIP_Optim(formula,data);
   ans
end

## Linear Regression - Ridge Prior
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Ridge,h::Float64=0.01,sim_size::Int64=10000)
   ans = Linear_Reg_Ridge_prior(formula,data,Prior_Ridge(),h,sim_size);
   ans
end

## Linear Regression - Laplace Prior
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Laplace,h::Float64=0.01,sim_size::Int64=10000)
   ans = Linear_Reg_Laplace_prior(formula,data,Prior_Laplace(),h,sim_size);
   ans
end

## Linear Regression - Cauchy Prior
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Cauchy,h::Float64=0.01,sim_size::Int64=10000)
   ans = Linear_Reg_Cauchy_prior(formula,data,Prior_Cauchy(),h,sim_size);
   ans
end

## Linear Regression - TDist Prior
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_TDist,h::Float64=1.0,sim_size::Int64=10000)
   ans = Linear_Reg_TDist_prior(formula,data,Prior_TDist(),h,sim_size);
   ans
end

## Linear Regression - Uniform Prior
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LinearRegression,PriorMod::Prior_Uniform,h::Float64=0.01,sim_size::Int64=10000)
   ans = Linear_Reg_Uniform_prior(formula,data,Prior_TDist(),h,sim_size);
   ans
end


# Logistic Regression -- Classical GLM
function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Logit)
   ans = logistic_Binom_NIP_Optim(formula,data,"LogitLink");
   ans
end

function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Probit)
   ans = logistic_Binom_NIP_Optim(formula,data,"ProbitLink");
   ans
end


function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Cloglog)
   ans = logistic_Binom_NIP_Optim(formula,data,"CloglogLink");
   ans
end


function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Cauchit)
   ans = logistic_Binom_NIP_Optim(formula,data,"CauchitLink");
   ans
end


## logistic regression -- Ridge prior

function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Logit,PriorMod::Prior_Ridge, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Ridge(formula,data,Logit(),Prior_Ridge(),h,sim_size)
   ans      
end

function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Probit,PriorMod::Prior_Ridge, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Ridge(formula,data,Probit(),Prior_Ridge(),h,sim_size)
   ans      
end


function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Cloglog,PriorMod::Prior_Ridge, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Ridge(formula,data,Cloglog(),Prior_Ridge(),h,sim_size)
   ans      
end


function fitmodel(formula::FormulaTerm, data::DataFrame,modelClass::LogisticRegression,Link::Cauchit,PriorMod::Prior_Ridge, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Ridge(formula,data,Cauchit(),Prior_Ridge(),h,sim_size)
   ans      
end

## logistic regression -- Laplace prior

function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit, PriorMod::Prior_Laplace, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Laplace(formula,data,Logit(),Prior_Laplace(),h,sim_size)
   ans
end

function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit, PriorMod::Prior_Laplace, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Laplace(formula,data,Probit(),Prior_Laplace(),h,sim_size)
   ans
end

function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog, PriorMod::Prior_Laplace, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Laplace(formula,data,Cloglog(),Prior_Laplace(),h,sim_size)
   ans
end


function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit, PriorMod::Prior_Laplace, h::Real=0.1, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Laplace(formula,data,Cauchit(),Prior_Laplace(),h,sim_size)
   ans
end

## logistic regression -- Cauchy prior


function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit, PriorMod::Prior_Cauchy, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Cauchy(formula,data,Logit(),Prior_Cauchy(),h,sim_size)
   ans
end


function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit, PriorMod::Prior_Cauchy, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Cauchy(formula,data,Probit(),Prior_Cauchy(),h,sim_size)
   ans
end


function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog, PriorMod::Prior_Cauchy, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Cauchy(formula,data,Cloglog(),Prior_Cauchy(),h,sim_size)
   ans
end

function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit, PriorMod::Prior_Cauchy, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Cauchy(formula,data,Cauchit(),Prior_Cauchy(),h,sim_size)
   ans
end


## logistic regression -- T-Dist prior


function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit, PriorMod::Prior_TDist, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_TDist(formula,data,Logit(),Prior_TDist(),h,sim_size)
   ans
end


function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit, PriorMod::Prior_TDist, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_TDist(formula,data,Probit(),Prior_TDist(),h,sim_size)
   ans
end

function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog, PriorMod::Prior_TDist, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_TDist(formula,data,Cloglog(),Prior_TDist(),h,sim_size)
   ans
end


function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit, PriorMod::Prior_TDist, h::Real=1.0, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_TDist(formula,data,Cauchit(),Prior_TDist(),h,sim_size)
   ans
end

## logistic regression -- Uniform prior

function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Logit, PriorMod::Prior_Uniform, h::Real=0.01, level::Real=0.95, sim_size::Int64=10000)
  ans = logistic_Binom_Uniform(formula,data,Logit(),Prior_Uniform(),h,sim_size)
  ans     
end


function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Probit, PriorMod::Prior_Uniform, h::Real=0.01, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Uniform(formula,data,Probit(),Prior_Uniform(),h,sim_size)
   ans     
end

function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cloglog, PriorMod::Prior_Uniform, h::Real=0.01, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Uniform(formula,data,Cloglog(),Prior_Uniform(),h,sim_size)
   ans     
end

function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::Cauchit, PriorMod::Prior_Uniform, h::Real=0.01, level::Real=0.95, sim_size::Int64=10000)
   ans = logistic_Binom_Uniform(formula,data,Cauchit(),Prior_Uniform(),h,sim_size)
   ans     
end

