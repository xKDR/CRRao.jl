# Package Guide

## Installation

To install the package, type `]` in the Julia REPL to enter the `Pkg` mode and run

```
pkg> add https://github.com/xKDR/CRRao.jl
```

## Tutorial: Frequentist Linear Regression

The goal of the CRRao package is to try to unify calling variety of statistical models under the same API. Note that this is different from what something like [`StatsAPI.jl`](https://github.com/JuliaStats/StatsAPI.jl) is doing; instead of introducing namespaces for development of packages, CRRao tries to *call* those packages with a uniform API. A very similar package comes from the R world: the [Zelig Project](https://zeligproject.org/).

To see how this API works, we will go over an example in which we'll train a linear regression model with the usual ordinary least squares method (which falls under the category of the frequentist viewpoint of statistics). For our example, we will be working with the `mtcars` dataset.

We first import the required packages.

```@repl ols_linear_regression
using CRRao, RDatasets, StatsPlots, Plots
```

Then we import the dataset.

```@repl ols_linear_regression
mtcars = dataset("datasets", "mtcars")
```

This dataset has 11 columns (barring the index). We want to train a linear regression model to predict `MPG` of a car from the information contained in the attributes `HP`, `WT` and `Gear`. We can represent this as a formula term of type [`StatsModels.formula`](https://juliastats.org/StatsModels.jl/stable/api/#StatsModels.@formula). The formula term will look like

```julia
MPG ~ HP + WT + Gear
```

More information about such terms can be found in the corresponding [docs](https://juliastats.org/StatsModels.jl/stable/api/#StatsModels.@formula). 

Next, we train a linear regression model.

```@repl ols_linear_regression
model = @fitmodel(MPG ~ HP + WT + Gear, mtcars, LinearRegression())
```

As we can see from the output, a table of coefficients has been printed for us. We can now infer other details of the model from the various getter functions that apply to frequentist models. So one can do the following.

```@repl ols_linear_regression
coeftable(model)
sigma(model)
aic(model)
bic(model)
r2(model)
adjr2(model)
residuals(model)
```

We can also get the predicted response of the model, along with other measures like the vector of Cook's distances.

```@repl ols_linear_regression
predict(model)
residuals(model)
cooksdistance(model)
```

And here is the plot of the vector of Cook's distances.
```@example ols_linear_regression
plot(cooksdistance(model))
```

To understand more about these functions and in general how frequentist models work in CRRao, please visit the section of the API reference on [Frequentist Regression Models](@ref).

## Tutorial: Bayesian Logistic Regression