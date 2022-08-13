function negativebinomial_reg(
    formula::FormulaTerm,
    data::DataFrame,
    turingModel::Function,
    sim_size::Int64
)
    formula = apply_schema(formula, schema(formula, data))
    y, X = modelcols(formula, data)

    if sim_size < 500
        @warn "Simulation size should generally be atleast 500."
    end
    chain = sample(CRRao_rng, turingModel(X, y), NUTS(), sim_size)
    return BayesianRegression(:NegativeBinomialRegression, chain, formula)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::NegBinomRegression, prior::Prior_Ridge, h::Float64 = 0.1, sim_size::Int64 = 10000)
```

Fit a Bayesian Negative Binomial Regression model on the input data with a Ridge prior.
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

Fit a Bayesian Negative Binomial Regression model on the input data with a Uniform prior. Ibrahim and Laud (JASA, 1990) showed that the uniform flat priors for GLMs can lead to improper posterior distributions thus making them undesirable. In such cases, the Markov Chain struggles to converge. Even if it converges, results are unreliable.
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
