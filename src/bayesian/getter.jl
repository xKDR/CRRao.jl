function predict(container::BayesianRegression{:LinearRegression}, newdata::DataFrame)
    X = modelmatrix(container.formula, newdata)
    W = container.samples
    predictions = X * W
    return vec(mean(predictions, dims=2))
end

function predict(container::BayesianRegression{:LogisticRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)
    W = container.samples[:, prediction_chain_start:end]
    z =  X * W
    return vec(mean(container.link.link_function.(z), dims=2))
end

function predict(container::BayesianRegressionVI{:LogisticRegression}, newdata::DataFrame, number_of_samples::Int64 = 1000)
    X = modelmatrix(container.formula, newdata)
    
    W = rand(CRRao_rng, container.dist, number_of_samples)
    W = W[union(container.symbol_to_range[:β]...), :]
    z = X * W 
    return vec(mean(container.link.link_function.(z), dims=2))
end

function predict(container::BayesianRegressionMCMC{:NegativeBinomialRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
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