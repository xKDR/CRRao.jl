# Frequentist models
function details(modelclass::String, likelihood::String, link::String, method::String)
    println("Model Class: ", modelclass)
    println("Likelihood Mode: ", likelihood)
    println("Link Function: ", link)
    println("Computing Method: ", method)
end

function Base.show(io::IO, container::FrequentistRegression{:LinearRegression})
    details("Linear Regression", "Gauss", "Identity", "Optimization")
    print(coeftable(container))
end

function Base.show(io::IO, container::FrequentistRegression{:LogisticRegression})
    details("Logistic Regression", "Binomial", "Identity", "Optimization")
    print(coeftable(container))
end

function Base.show(io::IO, container::FrequentistRegression{:NegativeBinomialRegression})
    details("Count Regression", "Negative Binomial", "Log", "Optimization")
    print(coeftable(container))
end

function Base.show(io::IO, container::FrequentistRegression{:PoissonRegression})
    details("Poisson Regression", "Poison", "Log", "Optimization")
    print(coeftable(container))
end

# Bayesian Models
function Base.show(io::IO, container::BayesianRegression)
    show(io, MIME("text/plain"), container.chain)
end
