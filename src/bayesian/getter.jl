function predict(container::BayesianRegression{:LinearRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)

    params = get_params(container.chain[prediction_chain_start:end,:,:])
    W = params.β
    if isa(W, Tuple)
        W = reduce(hcat, W)
    end
    predictions = params.α' .+ X * W'
    return vec(mean(predictions, dims=2))
end

function predict(container::BayesianRegression{:LogisticRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)

    params = get_params(container.chain[prediction_chain_start:end,:,:])
    W = params.β
    if isa(W, Tuple)
        W = reduce(hcat, W)
    end
    z = X * W'
    return vec(mean(container.link.link.(z), dims=2))
end

function predict(container::BayesianRegression{:NegativeBinomialRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)

    params = get_params(container.chain[prediction_chain_start:end,:,:])
    W = params.β
    if isa(W, Tuple)
        W = reduce(hcat, W)
    end
    z = params.α' .+ X * W'
    return vec(mean(exp.(z), dims=2))
end

function predict(container::BayesianRegression{:PoissonRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)

    params = get_params(container.chain[prediction_chain_start:end,:,:])
    W = params.β
    if isa(W, Tuple)
        W = reduce(hcat, W)
    end
    z = params.α' .+ X * W'
    return vec(mean(exp.(z), dims=2))
end