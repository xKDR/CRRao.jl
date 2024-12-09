function predict(container::BayesianRegression{:LinearRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)
    W = container.samples[:, prediction_chain_start:end]
    predictions = X * W
    return vec(mean(predictions, dims=2))
end

function predict(container::BayesianRegression{:LogisticRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)
    W = container.samples[:, prediction_chain_start:end]
    z =  X * W
    return vec(mean(container.link.link_function.(z), dims=2))
end

function predict(container::BayesianRegression{:NegativeBinomialRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)
    W = container.samples[:, prediction_chain_start:end]
    z =  X * W
    return vec(mean(exp.(z), dims=2))
end

function predict(container::BayesianRegression{:PoissonRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)
    W = container.samples[:, prediction_chain_start:end]
    z =  X * W
    return vec(mean(exp.(z), dims=2))
end