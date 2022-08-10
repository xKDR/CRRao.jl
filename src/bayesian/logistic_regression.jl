function logistic_reg(formula::FormulaTerm, data::DataFrame, Link::CRRaoLink, turingModel::Function, sim_size::Int64)
    formula = apply_schema(formula, schema(formula, data))
    y, X = modelcols(formula, data)

    if sim_size < 500
        @warn "Simulation size should generally be atleast 500."
    end
    chain = sample(CRRao_rng, turingModel(X, y), NUTS(), sim_size)
    return BayesianRegression(:LogisticRegression, chain, formula, Link)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Ridge, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 10000)
```

Fit a Bayesian Logistic Regression model on the input data with a Ridge prior. 

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Logistic Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Ridge prior.
- `Link`: A type representing the link function to be used. Possible values are `Logit()`, `Probit()`, `Cloglog()` and `Cauchit()`.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> turnout = dataset("Zelig", "turnout");

julia> container_logit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit(), Prior_Ridge());

julia> container_probit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Probit(), Prior_Ridge());

julia> container_cloglog = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cloglog(), Prior_Ridge());

julia> container_cauchit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cauchit(), Prior_Ridge());
```
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

    return logistic_reg(formula, data, Link, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Laplace, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 10000)
```

Fit a Bayesian Logistic Regression model on the input data with a Laplace prior. 

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Logistic Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Laplace prior.
- `Link`: A type representing the link function to be used. Possible values are `Logit()`, `Probit()`, `Cloglog()` and `Cauchit()`.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> turnout = dataset("Zelig", "turnout");

julia> container_logit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit(), Prior_Laplace());

julia> container_probit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Probit(), Prior_Laplace());

julia> container_cloglog = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cloglog(), Prior_Laplace());

julia> container_cauchit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cauchit(), Prior_Laplace());
```
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

    return logistic_reg(formula, data, Link, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Cauchy, h::Float64 = 0.1, level::Float64 = 0.95, sim_size::Int64 = 10000)
```

Fit a Bayesian Logistic Regression model on the input data with a Cauchy prior. 

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Logistic Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Cauchy prior.
- `Link`: A type representing the link function to be used. Possible values are `Logit()`, `Probit()`, `Cloglog()` and `Cauchit()`.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> turnout = dataset("Zelig", "turnout");

julia> container_logit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit(), Prior_Cauchy());

julia> container_probit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Probit(), Prior_Cauchy());

julia> container_cloglog = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cloglog(), Prior_Cauchy());

julia> container_cauchit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cauchit(), Prior_Cauchy());
```
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

    return logistic_reg(formula, data, Link, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_TDist, h::Float64 = 1.0, level::Float64 = 0.95, sim_size::Int64 = 10000)
```

Fit a Bayesian Logistic Regression model on the input data with a T-Dist prior. 

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Logistic Regression in our case.
- `prior`: A type representing the prior. In this case, it is the T-Dist prior.
- `Link`: A type representing the link function to be used. Possible values are `Logit()`, `Probit()`, `Cloglog()` and `Cauchit()`.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> turnout = dataset("Zelig", "turnout");

julia> container_logit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit(), Prior_TDist());

julia> container_probit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Probit(), Prior_TDist());

julia> container_cloglog = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cloglog(), Prior_TDist());

julia> container_cauchit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cauchit(), Prior_TDist());
```
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

    return logistic_reg(formula, data, Link, LogisticRegression, sim_size)
end

"""
```julia
fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::LogisticRegression, Link::CRRaoLink, prior::Prior_Uniform, h::Float64 = 0.01, level::Float64 = 0.95, sim_size::Int64 = 10000)
```

Fit a Bayesian Logistic Regression model on the input data with a Uniform prior. 

# Arguments

- `formula`: A formula term representing dependencies between the columns in the dataset.
- `data`: The dataset.
- `modelClass`: Object representing the type of regression, which is Logistic Regression in our case.
- `prior`: A type representing the prior. In this case, it is the Uniform prior.
- `Link`: A type representing the link function to be used. Possible values are `Logit()`, `Probit()`, `Cloglog()` and `Cauchit()`.
- `h`: A parameter used in setting the priors.
- `sim_size`: The number of samples to be drawn during inference.

# Example

```julia-repl
julia> using CRRao, RDatasets, StableRNGs

julia> CRRao.set_rng(StableRNG(123))
StableRNGs.LehmerRNG(state=0x000000000000000000000000000000f7)

julia> turnout = dataset("Zelig", "turnout");

julia> container_logit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Logit(), Prior_Uniform());

julia> container_probit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Probit(), Prior_Uniform());

julia> container_cloglog = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cloglog(), Prior_Uniform());

julia> container_cauchit = @fitmodel(Vote ~ Age + Race + Income + Educate, turnout, LogisticRegression(), Cauchit(), Prior_Uniform());
```
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

    return logistic_reg(formula, data, Link, LogisticRegression, sim_size)
end
