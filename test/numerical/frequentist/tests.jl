@testset "Linear Regression" begin
    include("LinearRegression.jl")
end

@testset "Logistic Regression" begin
    include("LogisticRegression.jl")
end

@testset "Poisson Regression" begin
    include("PoissonRegression.jl")
end

@testset "Negative Binomial Regression" begin
    include("NegBinomialRegression.jl")
end