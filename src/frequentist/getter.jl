"""
```julia
coef(container::FrequentistRegression)
```

Estimated coefficients of the model. Extends the `coef` method from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).

# Example

```julia
using CRRao, RDatasets, StatsModels

# Get the dataset
mtcars = dataset("datasets", "mtcars")

# Train the model
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())

# Get table of coefficients
coef(container)
```
"""
function coef(container::FrequentistRegression)
    return StatsBase.coef(container.model)
end

"""
```julia
coeftable(container::FrequentistRegression)
```

Table of coefficients and other statistics of the model. Extends the `coeftable` method from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).

# Example

```julia
using CRRao, RDatasets, StatsModels

# Get the dataset
mtcars = dataset("datasets", "mtcars")

# Train the model
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())

# Get table of coefficients
coeftable(container)
```
"""
function coeftable(container::FrequentistRegression)
    return StatsBase.coeftable(container.model)
end

"""
```julia
r2(container::FrequentistRegression)
```

Coeffient of determination. Extends the `r2` method from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).

# Example

```julia
using CRRao, RDatasets, StatsModels

# Get the dataset
mtcars = dataset("datasets", "mtcars")

# Train the model
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())

# Get r2
r2(container)
```
"""
function r2(container::FrequentistRegression)
    return StatsBase.r2(container.model)
end

"""
```julia
adjr2(container::FrequentistRegression)
```

Adjusted coeffient of determination. Extends the `adjr2` method from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).

# Example

```julia
using CRRao, RDatasets, StatsModels

# Get the dataset
mtcars = dataset("datasets", "mtcars")

# Train the model
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())

# Get adjr2
adjr2(container)
```
"""
function adjr2(container::FrequentistRegression)
    return StatsBase.adjr2(container.model)
end

"""
```julia
loglikelihood(container::FrequentistRegression)
```

Log-likelihood of the model. Extends the `loglikelihood` method from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).

# Example

```julia
using CRRao, RDatasets, StatsModels

# Get the dataset
mtcars = dataset("datasets", "mtcars")

# Train the model
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())

# Get loglikelihood
adjr2(container)
```
"""
function loglikelihood(container::FrequentistRegression)
    return StatsBase.loglikelihood(container.model)
end

"""
```julia
aic(container::FrequentistRegression)
```

Akaike's Information Criterion. Extends the `aic` method from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).

# Example

```julia
using CRRao, RDatasets, StatsModels

# Get the dataset
mtcars = dataset("datasets", "mtcars")

# Train the model
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())

# Get aic
aic(container)
```
"""
function aic(container::FrequentistRegression)
    return StatsBase.aic(container.model)
end

"""
```julia
bic(container::FrequentistRegression)
```

Bayesian Information Criterion. Extends the `bic` method from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).

# Example

```julia
using CRRao, RDatasets, StatsModels

# Get the dataset
mtcars = dataset("datasets", "mtcars")

# Train the model
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())

# Get bic
bic(container)
```
"""
function bic(container::FrequentistRegression)
    return StatsBase.bic(container.model)
end

"""
```julia
sigma(container::FrequentistRegression)
```

The `sigma` computes the residual standard error from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).

# Example

```julia
using CRRao, RDatasets, StatsModels

# Get the dataset
mtcars = dataset("datasets", "mtcars")

# Train the model
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())

# Get sigma
sigma(container)
```
"""
function sigma(container::FrequentistRegression)
    return sqrt(StatsBase.deviance(container.model) / StatsBase.dof_residual(container.model))
end

"""
```julia
predict(container::FrequentistRegression)
```

Predicted response of the model. Extends the `predict` method from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).

# Example

```julia
using CRRao, RDatasets, StatsModels

# Get the dataset
mtcars = dataset("datasets", "mtcars")

# Train the model
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())

# Get predicted response
predict(container)
```
"""
function predict(container::FrequentistRegression)
    return StatsBase.predict(container.model)
end

function predict(container::FrequentistRegression{:LinearRegression}, newdata::DataFrame)
    fm_frame = ModelFrame(container.formula, newdata)
    return modelmatrix(fm_frame) * StatsBase.coef(container.model)
end

function predict(container::FrequentistRegression{:LogisticRegression}, newdata::DataFrame)
    fm_frame = ModelFrame(container.formula, newdata)
    z = modelmatrix(fm_frame) * StatsBase.coef(container.model)

    if (container.link == GLM.LogitLink)
        return exp.(z) ./ (1 .+ exp.(z))
    elseif (container.link == GLM.ProbitLink)
        return Probit_Link.(z)
    elseif (container.link == GLM.CauchitLink)
        return Cauchit_Link.(z)
    elseif (container.link == GLM.Cloglog)
        return Cloglog_Link.(z)
    end
end

function predict(container::FrequentistRegression{:NegativeBinomialRegression}, newdata::DataFrame)
    fm_frame = ModelFrame(container.formula, newdata)
    z = modelmatrix(fm_frame) * StatsBase.coef(container.model)

    if (container.link == GLM.LogLink)
        return exp.(z)
    end
end

function predict(container::FrequentistRegression{:PoissonRegression}, newdata::DataFrame)
    fm_frame = ModelFrame(container.formula, newdata)
    return exp.(modelmatrix(fm_frame) * StatsBase.coef(container.model))
end

"""
```julia
residuals(container::FrequentistRegression)
```

Residuals of the model. Extends the `residuals` method from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).

# Example

```julia
using CRRao, RDatasets, StatsModels

# Get the dataset
mtcars = dataset("datasets", "mtcars")

# Train the model
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())

# Get residuals
residuals(container)
```
"""
function residuals(container::FrequentistRegression)
    return StatsBase.residuals(container.model)
end

"""
```julia
cooksdistance(container::FrequentistRegression)
```

Compute Cook's distance for each observation in a linear model. Extends the `cooksdistance` method from [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl).

# Example

```julia
using CRRao, RDatasets, StatsModels

# Get the dataset
mtcars = dataset("datasets", "mtcars")

# Train the model
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())

# Get vector of Cook's distances
cooksdistance(container)
```
"""
function cooksdistance(container::FrequentistRegression)
    return StatsBase.cooksdistance(container.model)
end
