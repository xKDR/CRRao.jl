function linear_reg(formula::FormulaTerm, data::DataFrame, turingModel::Function, sim_size::Int64)
    formula = apply_schema(formula, schema(formula, data))
    y, X = modelcols(formula, data)

    if sim_size < 500
        @warn "Simulation size should generally be atleast 500."
    end
    chain = sample(CRRao_rng, turingModel(X, y), NUTS(), sim_size)
    return BayesianRegression(:LinearRegression, chain, formula)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Ridge, h::Float64 = 0.01, sim_size::Int64 = 10000)
```

Fit a Bayesian Linear Regression model on the input data with a Ridge prior.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Linear Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Ridge prior.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> df = dataset("datasets", "mtcars");

julia> container = @fitmodel(MPG ~ HP + WT + Gear, df, LinearRegression(), Prior_Ridge());
```
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LinearRegression,
    prior::Prior_Ridge,
    h::Float64 = 0.01,
    sim_size::Int64 = 10000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)

        #priors
        a0 = 0.1
        b0 = 0.1

        v ~ InverseGamma(h, h)
        σ ~ InverseGamma(a0, b0)
        α ~ Normal(0, v * σ)
        β ~ filldist(Normal(0, v * σ), p)

        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Laplace, h::Float64 = 0.01, sim_size::Int64 = 10000)
```

Fit a Bayesian Linear Regression model on the input data with a Laplace prior.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Linear Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Laplace prior.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> df = dataset("datasets", "mtcars");

julia> container = @fitmodel(MPG ~ HP + WT + Gear, df, LinearRegression(), Prior_Laplace());
```
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LinearRegression,
    prior::Prior_Laplace,
    h::Float64 = 0.01,
    sim_size::Int64 = 10000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)

        #priors
        a0 = 0.1
        b0 = 0.1
        v ~ InverseGamma(h, h)
        σ ~ InverseGamma(a0, b0)
        α ~ Laplace(0, σ * v)
        β ~ filldist(Laplace(0, σ * v), p)

        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Cauchy, sim_size::Int64 = 10000)
```

Fit a Bayesian Linear Regression model on the input data with a Cauchy prior.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Linear Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Cauchy prior.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> df = dataset("datasets", "mtcars");

julia> container = @fitmodel(MPG ~ HP + WT + Gear, df, LinearRegression(), Prior_Cauchy(), 20000);
```
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LinearRegression,
    prior::Prior_Cauchy,
    sim_size::Int64 = 10000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)

        #priors
        σ ~ Truncated(TDist(1), 0, Inf)
        α ~ TDist(1) * σ
        β ~ filldist(TDist(1) * σ, p)

        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_TDist, h::Float64 = 2.0, sim_size::Int64 = 10000)
```

Fit a Bayesian Linear Regression model on the input data with a t(ν) distributed prior.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Linear Regression in our case.
- `prior`: A type representing the prior. In this case, it is the TDist prior.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsPlots

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> df = dataset("datasets", "mtcars");

julia> container = @fitmodel(MPG ~ HP + WT + Gear, df, LinearRegression(), Prior_TDist());

julia> plot(container.chain)
```
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LinearRegression,
    prior::Prior_TDist,
    h::Float64 = 2.0,
    sim_size::Int64 = 10000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)

        #priors
        a0 = 0.1
        b0 = 0.1
        ν ~ InverseGamma(h, h)
        σ ~ InverseGamma(a0, b0)
        α ~ TDist(ν) * σ
        β ~ filldist(TDist(ν) * σ, p)

        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LinearRegression, prior::Prior_Uniform, h::Float64 = 0.01, sim_size::Int64 = 10000)
```

Fit a Bayesian Linear Regression model on the input data with a Uniform prior.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Linear Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Uniform prior.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs, StatsPlots

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> df = dataset("datasets", "mtcars");

julia> container = @fitmodel(MPG ~ HP + WT + Gear, df, LinearRegression(), Prior_Uniform());

julia> plot(container.chain)
```
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LinearRegression,
    prior::Prior_Uniform,
    h::Float64 = 0.01,
    sim_size::Int64 = 10000
)
    @model LinearRegression(X, y) = begin
        p = size(X, 2)

        #priors
        v = 1 / h
        σ ~ Uniform(0, v)
        α ~ Uniform(-v * σ, v * σ)
        β ~ filldist(Uniform(-v, v), predictors)

        #likelihood
        y ~ MvNormal(α .+ X * β, σ)
    end

    return linear_reg(formula, data, LinearRegression, sim_size)
end
