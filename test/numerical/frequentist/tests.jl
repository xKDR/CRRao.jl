function compare_models(crrao_model, glm_model, df)
    @test isapprox(coeftable(crrao_model).cols, coeftable(glm_model).cols)
    @test isapprox(predict(crrao_model, df), predict(glm_model, df))
end

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