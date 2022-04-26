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

# CRRao: Julia Statistical Modeling Package for All

CRRao is a consistent framework for statistical models. There is value in having a consistent API for a wide variety of statistical models. The CRRao package offers this design, and at present has four models. We will build more in coming days, and we hope other authors of models will also build new models in this framework.

```Julia

   Julia> using CRRao, RDatasets, StatsModels

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

   ```

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

## **Why one should use CRRao in Julia over lm in R?**

Julia works much faster in terms of time and memory management. Hence it is suitable for handling really large data. We demonstrate that just to implement the OLS regression Julia is approximately 2.5x times faster than `R`.

We took `mtcars` data and fit a simple linear regression using `lm` in `R` and benchmarked the process using the microbenchmark library. 

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

We fit the exact same model using the `fitmodel` API of `CRRao` in `Julia` and benchmarked the process using the BenchmarkTools package.

```julia
using RDatasets, CRRao, BenchmarkTools, StatsModels
df = dataset("datasets", "mtcars");
@benchmark fitmodel(@formula(MPG ~ HP + WT), df, LinearRegression())
```

```julia
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):   90.092 μs …  34.761 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     127.941 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   160.215 μs ± 559.192 μs  ┊ GC (mean ± σ):  4.54% ± 3.30%

        ▄▇█▆▄▃▂                                                 ▂
  █▆▆▅▆██████████▆▆▅▆▆▅▅▅▅▅▅▄▄▅▅▆▅▅▅▄▅▄▄▆▃▅▄▄▄▁▄▄▄▁▃▁▄▃▃▁▄▁▁▅▄▃ █
  90.1 μs       Histogram: log(frequency) by time        365 μs <

 Memory estimate: 83.77 KiB, allocs estimate: 723.
```


Clearly we see a gain of 254% or 2.54 times gain in time if you use the CRRao in `Julia` instead of `lm` in `R`.


In another independent machine when we compared the same statistical model on the same `mtcars` data over `Julia`, `R` and `Python`; for 100,000 iterations, we have the following results:

--------------------------------------------------
Language   |   Package/Function |    Time Taken
-----------| -------------------|------------------
`Python`   |  `statsmodes`/`ols`|  210.66 seconds
`Python`.  |  `sklearn`/`fit`.  |   55.99 seconds
`R`        |  `stats`/`lm`      |   21.02 seconds
`Julia`    |  `CRRao`/`fitmodel`|    8.05 seconds
-----------|--------------------|------------------
