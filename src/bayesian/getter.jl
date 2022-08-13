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