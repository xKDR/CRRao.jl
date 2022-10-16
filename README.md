# CRRao

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://xKDR.github.io/CRRao.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://xKDR.github.io/CRRao.jl/dev)
![Build Status](https://github.com/xKDR/CRRao.jl/actions/workflows/ci.yml/badge.svg)
![Build Status](https://github.com/xKDR/CRRao.jl/actions/workflows/documentation.yml/badge.svg)
[![Coverage](https://codecov.io/gh/xKDR/CRRao.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xKDR/CRRao.jl)
[![Milestones](https://img.shields.io/badge/-milestones-brightgreen)](https://github.com/xKDR/CRRao.jl/milestones)

## To install: 
```Julia
 add "https://github.com/xKDR/CRRao.jl.git"
```

# CRRao: A single API for diverse statistical models

Many statistical models can be estimated in Julia, and the diversity of the model ecosystem is steadily improving. Drawing inspiration from the [Zelig](http://docs.zeligproject.org/index.html) package in the R world, the CRRao package gives a simple and consistent API to the end user. The end-user then faces the fixed cost of getting a hang of this, once, and after that a wide array of models and associated capabilities become available with a consistent syntax. We hope others developing statistical models will build within this framework. 

Here's an example of estimating the linear regression

MPG = β0 + β1 HP + β2 WT + β3 Gear + ϵ

```Julia

   using CRRao, RDatasets, StatsModels
   df = dataset("datasets", "mtcars")
   model = fit(@formula(MPG ~ HP + WT+Gear), df, LinearRegression())
   model.fit

   ────────────────────────────────────────────────────────────────────────────
                     Coef.  Std. Error      t  Pr(>|t|)   Lower 95%   Upper 95%
   ────────────────────────────────────────────────────────────────────────────
   (Intercept)  32.0137     4.63226      6.91    <1e-06  22.5249     41.5024
   HP           -0.0367861  0.00989146  -3.72    0.0009  -0.0570478  -0.0165243
   WT           -3.19781    0.846546    -3.78    0.0008  -4.93188    -1.46374
   Gear          1.01998    0.851408     1.20    0.2410  -0.72405     2.76401
   ────────────────────────────────────────────────────────────────────────────

   ```

This calls the generic function fit(), where you supply a formula, a dataset, and pick the model.

# Present capabilities

We have implemented four regression models:
1. Linear
2. Logistic (with four link functions) 
3. Poisson 
4. Negative binomial

In all cases, we have traditional frequentist models and Bayesian versions with four kinds of priors :

1. Ridge
2. Laplace
3. Cauchy
4. T-Distributed

All these models are built out of foundations in the Julia package ecosystem, such as GLM.jl and Turing.jl. Here in CRRao.jl, we are not building additional models; we are only building the scaffolding for the consistent API to a diverse array of models.

# Help us build this

Please use CRRao and tell us what is not good about it.

We have exploited Julia capabilities to make it convenient to build additional functionality within CRRao, and for multiple developers to build new models.

We want to build out CRRao into a simple and consistent approach to the statistical modelling workflow. Please help us plan and build this.

As a developer, you can begin contributing by adding the features requested in the [milestones](https://github.com/xKDR/CRRao.jl/milestones) section of the repository. 

# Performance gains

The efficiency gains of the Julia language and the package ecosystem accrue to the end-user of CRRao. (CRRao is just a thin layer, and the heavy lifting is all done by the underlying packages). Here is some measurement of the above model, done through four alternative systems.

**R**
```{r}
> attach(datasets::mtcars)
> library(microbenchmark)
> microbenchmark(lm(mpg~hp+wt))
```

```{r}
Unit: microseconds
              expr     min      lq     mean   median      uq      max neval
 lm(mpg ~ hp + wt) 290.534 311.209 380.1334 325.9485 395.288 2223.736   100
```
**Julia**
```julia
using RDatasets, CRRao, BenchmarkTools, StatsModels
df = dataset("datasets", "mtcars")
@benchmark fit(@formula(MPG ~ HP + WT), df, LinearRegression())
```

```julia
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):   90.092 μs …  34.761 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     127.941 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   160.215 μs ± 559.192 μs  ┊ GC (mean ± σ):  4.54% ± 3.30%
```

To summarise the performance across four alternatives:
--------------------------------------------------
Language   |   Package/Function |    Mean time taken
-----------| -------------------|------------------
`Python`   |  `statsmodes`/`ols`|  2106.6 μs
`Python`   |  `sklearn`/`fit`   |   559.9 μs
`R`        |  `stats`/`lm`      |   380.13 μs
`Julia`    |  `CRRao`/`fit`     |    160.22 μs
-----------|--------------------|------------------

where we emphasise that the performance of fit() here is a tiny overhead on top of the implementation of the linear regression in GLM.jl.

# Support

We gratefully acknowledge the JuliaLab at MIT for financial support for this project.