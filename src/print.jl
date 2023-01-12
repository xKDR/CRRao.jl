# Frequentist models
function details(io::IO, modelclass::String, likelihood::String, link::String, method::String)
    println(io, "Model Class: ", modelclass)
    println(io, "Likelihood Mode: ", likelihood)
    println(io, "Link Function: ", link)
    println(io, "Computing Method: ", method)
end

function Base.show(io::IO, container::FrequentistRegression{:LinearRegression})
    details(io, "Linear Regression", "Gauss", "Identity", "Optimization")
    println(io, coeftable(container))
end

function Base.show(io::IO, container::FrequentistRegression{:LogisticRegression})
    details(io, "Logistic Regression", "Binomial", "Identity", "Optimization")
    println(io, coeftable(container))
end

function Base.show(io::IO, container::FrequentistRegression{:NegativeBinomialRegression})
    details(io, "Count Regression", "Negative Binomial", "Log", "Optimization")
    println(io, coeftable(container))
end

function Base.show(io::IO, container::FrequentistRegression{:PoissonRegression})
    details(io, "Poisson Regression", "Poison", "Log", "Optimization")
    println(io, coeftable(container))
end

# Bayesian Models
function Base.show(io::IO, container::BayesianRegression)
    println(io, "Formula: ", container.formula)
    println(io, "Link: ", container.link)
    print(io, "Chain: ")
    show(io, MIME("text/plain"), container.chain)
end
