function coeftable(container::FrequentistRegression)
    return StatsBase.coeftable(container.model)
end

function r2(container::FrequentistRegression)
    return StatsBase.r2(container.model)
end

function adjr2(container::FrequentistRegression)
    return StatsBase.adjr2(container.model)
end

function loglikelihood(container::FrequentistRegression)
    return StatsBase.loglikelihood(container.model)
end

function aic(container::FrequentistRegression)
    return StatsBase.aic(container.model)
end

function bic(container::FrequentistRegression)
    return StatsBase.bic(container.model)
end

function sigma(container::FrequentistRegression)
    return sqrt(StatsBase.deviance(container.model) / StatsBase.dof_residual(container.model))
end

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

function residuals(container::FrequentistRegression)
    return StatsBase.residuals(container.model)
end

function cooksdistance(container::FrequentistRegression)
    return StatsBase.cooksdistance(container.model)
end
