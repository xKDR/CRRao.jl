function predict(container::BayesianRegressionMCMC{:LinearRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)

    params = get_params(container.chain[prediction_chain_start:end,:,:])
    W = params.β
    if isa(W, Tuple)
        W = reduce(hcat, W)
    end
    #predictions = params.α' .+ X * W'
    predictions = X * W'
    return vec(mean(predictions, dims=2))
end

function predict(container::BayesianRegressionVI{:LinearRegression}, newdata::DataFrame, number_of_samples::Int64 = 1000)
    X = modelmatrix(container.formula, newdata)

    W = rand(CRRao_rng, container.dist, number_of_samples)
    W = W[union(container.symbol_to_range[:β]...), :]
    predictions = X * W
    return vec(mean(predictions, dims=2))
end

function predict(container::BayesianRegressionMCMC{:LogisticRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)

    params = get_params(container.chain[prediction_chain_start:end,:,:])
    W = params.β
    if isa(W, Tuple)
        W = reduce(hcat, W)
    end
    #z = params.α' .+ X * W'
    z =  X * W'
    return vec(mean(container.link.link_function.(z), dims=2))
end

function predict(container::BayesianRegressionMCMC{:NegativeBinomialRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)

    params = get_params(container.chain[prediction_chain_start:end,:,:])
    W = params.β
    if isa(W, Tuple)
        W = reduce(hcat, W)
    end
    #z = params.α' .+ X * W'
    z =  X * W'
    return vec(mean(exp.(z), dims=2))
end

function predict(container::BayesianRegressionMCMC{:PoissonRegression}, newdata::DataFrame, prediction_chain_start::Int64 = 200)
    X = modelmatrix(container.formula, newdata)

    params = get_params(container.chain[prediction_chain_start:end,:,:])
    W = params.β
    if isa(W, Tuple)
        W = reduce(hcat, W)
    end
    #z = params.α' .+ X * W'
    z =  X * W'
    return vec(mean(exp.(z), dims=2))
end