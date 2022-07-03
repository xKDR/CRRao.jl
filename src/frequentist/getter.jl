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
    # container.ndims[2] is the number of parameters
    return (2 * container.ndims[2] - 2 * loglikelihood(container))
end

function bic(container::FrequentistRegression)
    # container.ndims[1] is the number of data points
    # container.ndims[2] is the number of parameters
    return (log(container.ndims[1]) * container.ndims[2] - 2 * loglikelihood(container))
end

function sigma(container::FrequentistRegression)
    return sqrt(StatsBase.deviance(container.model) / StatsBase.dof_residual(container.model))
end

function predict(container::FrequentistRegression)
    return StatsBase.predict(container.model)
end

function residuals(container::FrequentistRegression)
    return StatsBase.residuals(container.model)
end

function cooksdistance(container::FrequentistRegression)
    return StatsBase.cooksdistance(container.model)
end
