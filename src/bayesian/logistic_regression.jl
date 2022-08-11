function logistic_reg(formula::FormulaTerm, data::DataFrame, turingModel::Function, sim_size::Int64)
    formula = apply_schema(formula, schema(formula, data))
    y, X = modelcols(formula, data)

    if sim_size < 500
        @warn "Simulation size should generally be atleast 500."
    end
    chain = sample(CRRao_rng, turingModel(X, y), NUTS(), sim_size)
    return BayesianRegression(:LogisticRegression, chain, formula)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Ridge, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 10000)
```

Fit a Bayesian Logistic Regression model on the input data with a Ridge prior with the provided `Link` function.
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::CRRaoLink,
    prior::Prior_Ridge,
    h::Float64 = 0.1,
    level::Float64 = 0.95,
    sim_size::Int64 = 10000
)
    @model LogisticRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        β ~ filldist(Normal(0, λ), p)

        z = X * β

        ## Link Function

        prob = Link.link.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Laplace, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 10000)
```

Fit a Bayesian Logistic Regression model on the input data with a Laplace prior with the provided `Link` function.
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::CRRaoLink,
    prior::Prior_Laplace,
    h::Float64 = 0.1,
    level::Float64 = 0.95,
    sim_size::Int64 = 10000
)
    @model LogisticRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        β ~ filldist(Laplace(0, λ), p)

        z = X * β

        ## Link Function

        prob = Link.link.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Cauchy, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 10000)
```

Fit a Bayesian Logistic Regression model on the input data with a Cauchy prior with the provided `Link` function.
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::CRRaoLink,
    prior::Prior_Cauchy,
    h::Float64 = 0.1,
    level::Float64 = 0.95,
    sim_size::Int64 = 10000
)
    @model LogisticRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ Truncated(TDist(1), 0, Inf)
        β ~ filldist(TDist(1) * λ, p)

        z = X * β

        ## Link Function

        prob = Link.link.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_TDist, h::Float64 = 1.0, level::Float64 = 0.95, sim_size::Int64 = 10000)
```

Fit a Bayesian Logistic Regression model on the input data with a T-Dist prior with the provided `Link` function.
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::CRRaoLink,
    prior::Prior_TDist,
    h::Float64 = 1.0,
    level::Float64 = 0.95,
    sim_size::Int64 = 10000
)
    @model LogisticRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        λ ~ InverseGamma(h, h)
        ν ~ InverseGamma(h, h)
        β ~ filldist(TDist(ν) * λ, p)

        z = X * β

        ## Link Function

        prob = Link.link.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Uniform, h::Float64 = 0.01, level::Float64 = 0.95, sim_size::Int64 = 10000)
```

Fit a Bayesian Logistic Regression model on the input data with a Uniform prior with the provided `Link` function.
"""
function fitmodel(
    formula::FormulaTerm,
    data::DataFrame,
    modelClass::LogisticRegression,
    Link::CRRaoLink,
    prior::Prior_Uniform,
    h::Float64 = 0.01,
    level::Float64 = 0.95,
    sim_size::Int64 = 10000
)
    @model LogisticRegression(X, y) = begin
        p = size(X, 2)
        n = size(X, 1)
        #priors
        v ~ InverseGamma(h, h)
        β ~ filldist(Uniform(-v, v), p)

        z = X * β

        ## Link Function

        prob = Link.link.(z)

        #likelihood
        for i = 1:n
            y[i] ~ Bernoulli(prob[i])
        end
    end

    return logistic_reg(formula, data, LogisticRegression, sim_size)
end
