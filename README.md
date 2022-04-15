# CRRao

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://xKDR.github.io/CRRao.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://xKDR.github.io/CRRao.jl/dev)
[![Build Status](https://travis-ci.com/xKDR/CRRao.jl.svg?branch=main)](https://travis-ci.com/xKDR/CRRao.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/xKDR/CRRao.jl?svg=true)](https://ci.appveyor.com/project/xKDR/CRRao-jl)
[![Build Status](https://api.cirrus-ci.com/github/xKDR/CRRao.jl.svg)](https://cirrus-ci.com/github/xKDR/CRRao.jl)
[![Coverage](https://codecov.io/gh/xKDR/CRRao.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xKDR/CRRao.jl)
[![Coverage](https://coveralls.io/repos/github/xKDR/CRRao.jl/badge.svg?branch=main)](https://coveralls.io/github/xKDR/CRRao.jl?branch=main)

## To install: 
```Julia
 add "https://github.com/xKDR/CRRao.jl.git"
```

# CRRao: Julia Statistical Modeling Package for All

CRRao is consistent framework for statistical models. There is value in having a consistent API for a wide variety of statistical models. The CRRao package offers this design, and at present has four models. We will build more in coming days, and we hope other authors of models will also build new models in this framework.

The current version includes the following four models: 
(1) Linear Regression, 
(2) Logistic Regression, 
(3) Poisson Regression, and 
(4) Negative Binomial Regression. 

For all four models, we implemented both likelihood and variety of Bayesian models using Turing.jl package of Julia. 

Currently the Bayesian versions of these four models can handle variety of Prior distribution class, such as 
(1) Ridge Prior, 
(2) Laplace prior, 
(3) Cauchy Prior, 
(4) T-Distributed prior, and 
(5) Uniform flat prior. 

For Logistic Regression it can handles four link functions: (1) Logit Link, (2) Probit Link, (3) Cloglog Link and (4) Cauchy Link.

Soon we will publish a developer doc so that more people can contribute to it.

CRRao leverage the strength of wonderful Julia packages that already exists, such as 
   1. GLM,  2. StatsModels, 3. Turing,  4. Soss, 5. DataFrames, 6. StatsBase, 7. Distributions, 8. LinearAlgebra

+ We are at the very early stage of the development.
+ **Note**: You can read more about **Prof C.R. Rao** [Click Here](https://en.wikipedia.org/wiki/C._R._Rao)

