module CRRao

using DataFrames, GLM, Turing, StatsModels, StatsBase
using StatsBase, Distributions, LinearAlgebra
using Optim, NLSolversBase, Random

"""
```julia
LinearRegression
```

Type representing the Linear Regression model class.
```math
y =\\alpha +  X \\beta+ \\varepsilon,
```
where 
```math
\\varepsilon \\sim N(0,\\sigma^2),
```
+ ``y`` is the response vector of size ``n``, 
+ ``X`` is the matrix of predictor variable of size ``n \\times p``,
+ ``n`` is the sample size, and ``p`` is the number of predictors, 
+ ``\\alpha`` is the intercept of the model,  
+ ``\\beta`` is the regression coefficients of the model, and
+ ``\\sigma`` is the standard deviation of the noise ``\\varepsilon``.
"""
struct LinearRegression end

"""
```julia
LogisticRegression
```

Type representing the Logistic Regression model class.
```math
y_i \\sim Bernoulli(p_i), 
```
where ``i=1,2,\\cdots,n, 0 < p_i < 1``, 
+ ``\\mathbb{E}(y_i)=p_i``,
+ ``\\mathbb{P}(y_i=1) = p_i `` and ``\\mathbb{P}(y_i=0) = 1-p_i ``, such that 
```math
\\mathbb{E}(y_i)= p_i =g(\\alpha +\\mathbf{x}_i^T\\beta),
```
+ ``g(.)`` is the link-function,
+ ``y_i`` is the ``i^{th}`` element of the response vector ``y``,
+ ``\\mathbf{x}_i=(x_{i1},x_{i2},\\cdots,x_{in})`` is the ``i^{th}`` row of the design matix of size ``n \\times p``,
+ ``\\alpha`` is the intercept of the model, and
+ ``\\beta`` is the regression coefficients of the model.
"""
struct LogisticRegression end

"""
```julia
NegBinomRegression
```

Type representing the Negative Binomial Regression model class.
```math
y_i \\sim NegativeBinomial(\\mu_i,\\phi), i=1,2,\\cdots,n
```
where

```math
\\mu_i = \\exp(\\alpha +\\mathbf{x}_i^T\\beta),
```
+ ``y_i`` is the ``i^{th}`` element of the response vector ``y``,
+ ``\\mathbf{x}=(x_{i1},x_{i2},\\cdots,x_{in})`` is the ``i^{th}`` row of the design matix of size ``n \\times p``,
+ ``\\alpha`` is the intercept of the model, and
+ ``\\beta`` is the regression coefficients of the model.
"""
struct NegBinomRegression end

"""
```julia
PoissonRegression
```

Type representing the Poisson Regression model class.
```math
y_i \\sim Poisson(\\lambda_i), i=1,2,\\cdots,n
```
where

```math
\\lambda_i = \\exp(\\alpha +\\mathbf{x}_i^T\\beta),
```
+ ``y_i`` is the ``i^{th}`` element of the response vector ``y``,
+ ``\\mathbf{x}=(x_{i1},x_{i2},\\cdots,x_{in})`` is the ``i^{th}`` row of the design matix of size ``n \\times p``,
+ ``\\alpha`` is the intercept of the model, and
+ ``\\beta`` is the regression coefficients of the model.

"""
struct PoissonRegression end


struct Prior_Gauss end

"""
```julia
Prior_Ridge
```
Type representing the Ridge Prior.

*Prior model*
```math
v \\sim InverseGamma(h,h),
```
```math
\\sigma \\sim InverseGamma(a_0,b_0),
```
```math
\\alpha | \\sigma,v \\sim Normal(0,v*\\sigma),
```
```math
\\beta | \\sigma,v \\sim Normal_p(0,v*\\sigma),
```
*Likelihood or data model*

```math
\\mu_i= \\alpha + \\mathbf{x}_i^T\\beta
```
```math
y_i \\sim D(\\mu_i,\\sigma),
```
**Note**: ``D()`` is appropriate distribution of ``y_i`` based on the `modelClass`, where 

+ ``\\mathbf{E}(y_i)=g(\\mu_i)``, and 
+ ``Var(y_i)=\\sigma^2``.
"""
struct Prior_Ridge end

"""
```julia
Prior_Laplace
```
Type representing the Laplace Prior.

*Prior model*
```math
v \\sim InverseGamma(h,h),
```
```math
\\sigma \\sim InverseGamma(a_0,b_0),
```
```math
\\alpha | \\sigma,v \\sim Laplace(0,v*\\sigma),
```
```math
\\beta | \\sigma,v \\sim Laplace(0,v*\\sigma),
```
*Likelihood or data model*
```math
\\mu_i= \\alpha + \\mathbf{x}_i^T\\beta
```
```math
y_i \\sim D(\\mu_i,\\sigma),
```
**Note**: ``D()`` is appropriate distribution of ``y_i`` based on the `modelClass`, where 

+ ``\\mathbf{E}(y_i)=g(\\mu_i)``, and 
+ ``Var(y_i)=\\sigma^2``.
"""
struct Prior_Laplace end

"""
```julia
Prior_Cauchy
```
Type representing the Cauchy Prior.

*Prior model*

```math
\\sigma \\sim Half-Cauchy(0,1),
```
```math
\\alpha | \\sigma \\sim  Cauchy(0,\\sigma),
```
```math
\\beta | \\sigma \\sim Cauchy(0,v*\\sigma),
```
*Likelihood or data model*
```math
\\mu_i= \\alpha + \\mathbf{x}_i^T\\beta
```
```math
y_i \\sim D(\\mu_i,\\sigma),
```
**Note**: ``D()`` is appropriate distribution of ``y_i`` based on the `modelClass`, where 

+ ``\\mathbf{E}(y_i)=g(\\mu_i)``, and 
+ ``Var(y_i)=\\sigma^2``.
"""
struct Prior_Cauchy end

"""
```julia
Prior_TDist
```
Type representing the T-Distributed Prior.

*Prior model*

```math
v \\sim InverseGamma(h,h),
```
```math
\\sigma \\sim InverseGamma(a_0,b_0),
```
```math
\\alpha | \\sigma,v \\sim \\sigma t(v),
```
```math
\\beta | \\sigma,v \\sim \\sigma t(v),
```
*Likelihood or data model*
```math
\\mu_i= \\alpha + \\mathbf{x}_i^T\\beta
```
```math
y_i \\sim D(\\mu_i,\\sigma),
```
**Note**: ``D()`` is appropriate distribution of ``y_i`` based on the `modelClass`, where 

+ ``\\mathbf{E}(y_i)=g(\\mu_i)``, and 
+ ``Var(y_i)=\\sigma^2``. 
+ The ``t(v)`` is ``t`` distribution with ``v`` degrees of freedom.
"""
struct Prior_TDist end

"""
```julia
Prior_Uniform
```
Type representing the Uniform Prior.

*Prior model*

```math
v = 1/h,
```
```math
\\sigma \\sim Uniform(0,v),
```
```math
\\alpha | \\sigma,v \\sim Uniform(-v \\sigma,v \\sigma),
```
```math
\\beta | \\sigma,v \\sim Uniform(-v ,v),
```
*Likelihood or data model*
```math
\\mu_i= \\alpha + \\mathbf{x}_i^T\\beta
```
```math
y_i \\sim D(\\mu_i,\\sigma),
```
**Note**: ``D()`` is appropriate distribution of ``y_i`` based on the `modelClass`, where 

+ ``\\mathbf{E}(y_i)=g(\\mu_i)``, and 
+ ``Var(y_i)=\\sigma^2``.
"""
struct Prior_Uniform end

"""
```julia
Prior_HorseShoe
```
Type representing the HorseShoe Prior.

*Prior model*

```math
\\tau \\sim HalfCauchy(0,1),
```
```math
\\lambda_j \\sim HalfCauchy(0,1), j=1,2,\\cdots,p
```
```math
\\sigma \\sim HalfCauchy(0,1),
```
```math
\\alpha | \\sigma,\\tau \\sim N(0,\\tau *\\sigma),
```
```math
\\beta_j | \\sigma,\\lambda_j ,\\tau \\sim Normal(0,\\lambda_j *\\tau *\\sigma),
```
*Likelihood or data model*
```math
\\mu_i= \\alpha + \\mathbf{x}_i^T\\beta
```
```math
y_i \\sim D(\\mu_i,\\sigma), i=1,2,\\cdots,n
```
**Note**:  ``D()`` is appropriate distribution of ``y_i`` based on the `modelClass`, where 

+ ``\\mathbf{E}(y_i)=g(\\mu_i)``, 
+ ``Var(y_i)=\\sigma^2``, and 
+ ``\\beta``=(``\\beta_1,\\beta_2,\\cdots,\\beta_p``)
"""
struct Prior_HorseShoe end

"""
```julia
CRRaoLink
```

Abstract type representing link functions which are used to dispatch to appropriate calls.
"""
abstract type CRRaoLink end

struct Identity <: CRRaoLink
    link_function::Function
end

Identity() = Identity(Identity_Link)

"""
```julia
Logit <: CRRaoLink
```

A type representing the Logit link function, which is defined by the formula

```math
z\\mapsto \\dfrac{1}{1 + \\exp(-z)}
```
"""
struct Logit <: CRRaoLink
    link_function::Function
end

Logit() = Logit(Logit_Link)

"""
```julia
Probit <: CRRaoLink
```

A type representing the Probit link function, which is defined by the formula

```math
z\\mapsto \\mathbb{P}[Z\\le z]
```

where ``Z\\sim \\text{Normal}(0, 1)``.
"""
struct Probit <: CRRaoLink
    link_function::Function
end

Probit() = Probit(Probit_Link)

"""
```julia
Cloglog <: CRRaoLink
```

A type representing the Cloglog link function, which is defined by the formula 

```math
z\\mapsto 1 - \\exp(-\\exp(z))
```
"""
struct Cloglog <: CRRaoLink
    link_function::Function
end

Cloglog() = Cloglog(Cloglog_Link)

"""
```julia
Cauchit <: CRRaoLink
```

A type representing the Cauchit link function, which is defined by the formula

```math
z\\mapsto \\dfrac{1}{2} + \\dfrac{\\text{atan}(z)}{\\pi}
```
"""
struct Cauchit <: CRRaoLink
    link_function::Function
end

Cauchit() = Cauchit(Cauchit_Link)

export LinearRegression, LogisticRegression, PoissonRegression, NegBinomRegression
export Prior_Ridge, Prior_Laplace, Prior_Cauchy, Prior_TDist, Prior_Uniform, Prior_HorseShoe, Prior_Gauss
export CRRaoLink, Logit, Probit, Cloglog, Cauchit, fit
export coeftable, r2, adjr2, loglikelihood, aic, bic, sigma, predict, residuals, cooksdistance
export FrequentistRegression, BayesianRegression

include("random_number_generator.jl")
include("general_stats.jl")
include("fitmodel.jl")

end
