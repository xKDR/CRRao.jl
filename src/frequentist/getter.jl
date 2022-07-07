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

function predict(container::FrequentistRegression{:LinearRegression}, newdata::DataFrame)
    formula = container.formula
    fm_frame = ModelFrame(formula, newdata)
    X = modelmatrix(fm_frame)
    beta = coef(container.res)
    y_pred = X * beta
    y_pred
end

function predict(container::FrequentistRegression{:LogisticRegression}, newdata::DataFrame)
    formula = container.formula
    fm_frame = ModelFrame(formula, newdata)
    X = modelmatrix(fm_frame)
    beta = container.beta
    z = X * beta

    if (container.Link == "LogitLink")
        p = exp.(z) ./ (1 .+ exp.(z))

    elseif (container.Link == "ProbitLink")
        p = Probit_Link.(z)

    elseif (container.Link == "CauchitLink")
        p = Cauchit_Link.(z)

    elseif (container.Link == "CloglogLink")
        p = Cloglog_Link.(z)

    else
        println("This link function is not part of logistic regression family.")
        println("-------------------------------------------------------------")
    end
    p
end

function predict(container::FrequentistRegression{:PoissonRegression}, newdata::DataFrame)
    formula = container.formula
    fm_frame = ModelFrame(formula, newdata)
    X = modelmatrix(fm_frame)
    beta = container.beta
    z = X * beta
    μ = exp.(z)
    μ
end

function predict(container::FrequentistRegression{:NegativeBinomialRegression}, newdata::DataFrame)
    formula = container.formula
    fm_frame = ModelFrame(formula, newdata)
    X = modelmatrix(fm_frame)
    beta = container.beta
    z = X * beta

    if (container.Link == "LogLink")
        p = exp.(z)

    else
        println("This link function is not part of NegativeBinomial regression family.")
        println("-------------------------------------------------------------")
    end
    p
end

function residuals(container::FrequentistRegression)
    return StatsBase.residuals(container.model)
end

function cooksdistance(container::FrequentistRegression)
    return StatsBase.cooksdistance(container.model)
end
