function negativebinomial_reg(
    formula::FormulaTerm,
    data::DataFrame,
    turingModel::Function,
    sim_size::Int64
)
    formula = apply_schema(formula, schema(formula, data))
    y, X = modelcols(formula, data)

    chain = sample(CRRao_rng, turingModel(X, y), NUTS(), sim_size)
    return BayesianRegression{:NegativeBinomialRegression}(chain)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Ridge, h::Float64 = 0.1, sim_size::Int64 = 10000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a Ridge prior.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Negative Binomial Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Ridge prior.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> sanction = dataset("Zelig", "sanction");

julia> container = @fitmodel(Num ~ Target + Coop + NCost, sanction, NegBinomRegression(), Prior_Ridge());
```
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    prior::Prior_Ridge,
    h::Float64 = 0.1,
    sim_size::Int64 = 10000
)
    @model NegativeBinomialRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)

        #priors
        λ ~ InverseGamma(h, h)
        α ~ Normal(0, λ)
        β ~ filldist(Normal(0, λ), p)

        ## link
        z = α .+ X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ NegativeBinomial2(mu[i], λ)
        end
    end

    return negativebinomial_reg(formula, data, NegativeBinomialRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Laplace, h::Float64 = 0.01, sim_size::Int64 = 10000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a Laplace prior.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Negative Binomial Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Laplace prior.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> sanction = dataset("Zelig", "sanction");

julia> container = @fitmodel(Num ~ Target + Coop + NCost, sanction, NegBinomRegression(), Prior_Laplace());
``` 
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    prior::Prior_Laplace,
    h::Float64 = 0.1,
    sim_size::Int64 = 10000
)
    @model NegativeBinomialRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)

        #priors
        λ ~ InverseGamma(h, h)
        α ~ Laplace(0, λ)
        β ~ filldist(Laplace(0, λ), p)

        ## link
        z = α .+ X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ NegativeBinomial2(mu[i], λ)
        end
    end

    return negativebinomial_reg(formula, data, NegativeBinomialRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Cauchy, h::Float64 = 1.0, sim_size::Int64 = 10000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a Cauchy prior.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Negative Binomial Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Cauchy prior.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> sanction = dataset("Zelig", "sanction");

julia> container = @fitmodel(Num ~ Target + Coop + NCost, sanction, NegBinomRegression(), Prior_Cauchy());
```
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    prior::Prior_Cauchy,
    h::Float64 = 1.0,
    sim_size::Int64 = 10000
)
    @model NegativeBinomialRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        α ~ TDist(1) * λ
        β ~ filldist(TDist(1) * λ, p)

        ## link
        z = α .+ X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ NegativeBinomial2(mu[i], λ)
        end
    end

    return negativebinomial_reg(formula, data, NegativeBinomialRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_TDist, h::Float64 = 1.0, sim_size::Int64 = 10000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a t(ν) distributed prior.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Negative Binomial Regression in our case.
- `prior`: A type representing the prior. In this case, it is the TDist prior.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> sanction = dataset("Zelig", "sanction");

julia> container = @fitmodel(Num ~ Target + Coop + NCost, sanction, NegBinomRegression(), Prior_TDist());
```
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    prior::Prior_TDist,
    h::Float64 = 1.0,
    sim_size::Int64 = 10000
)
    @model NegativeBinomialRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        ν ~ InverseGamma(h, h)
        α ~ TDist(ν) * λ
        β ~ filldist(TDist(ν) * λ, p)

        ## link
        z = α .+ X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ NegativeBinomial2(mu[i], λ)
        end
    end

    return negativebinomial_reg(formula, data, NegativeBinomialRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Uniform, h::Float64 = 0.1, sim_size::Int64 = 10000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a Uniform prior. Ibrahim and Laud (JASA, 1990) showed that the uniform flat priors for GLM's can lead to improper posterior distributions thus making them undesirable. In such cases, the Markov Chain struggles to converge. Even if it converges, results are unreliable.

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Negative Binomial Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Uniform prior.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> sanction = dataset("Zelig", "sanction");

julia> container = @fitmodel(Num ~ Target + Coop + NCost, sanction, NegBinomRegression(), Prior_Uniform());
``` 
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::NegBinomRegression,
    PriorMod::Prior_Uniform,
    h::Float64 = 0.1,
    sim_size::Int64 = 10000
)
    @model NegativeBinomialRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)

        #priors
        λ ~ InverseGamma(h, h)
        α ~ Uniform(-λ, λ)
        β ~ filldist(Uniform(-λ, λ), p)

        ## link
        z = α .+ X * β
        mu = exp.(z)

        #likelihood
        for i = 1:n
            y[i] ~ NegativeBinomial2(mu[i], λ)
        end
    end

    return negativebinomial_reg(formula, data, NegativeBinomialRegression, sim_size)
end
