struct MCMC_chain
    chain::Chains
    summaries
    quantiles
end

function Logit_Link(z::Real)
    1 / (1 + exp(-z))
end

function Cauchit_Link(z::Real)
    0.5 + atan(z) / π
end

function Probit_Link(z::Real)
    d = Distributions.Normal(0, 1)
    cdf(d, z)
end

function Cloglog_Link(z::Real)
    1 - exp(-exp(z))
end

function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    r = ϕ

    return NegativeBinomial(r, p)
end
