using CRRao, Test, StableRNGs, Logging, RDatasets, StatsModels, GLM

Logging.disable_logging(Logging.Warn)

CRRao.setprogress!(false)
CRRao.set_rng(StableRNG(123))

@testset "CRRao.jl" begin
    @testset "Basic Tests" begin
        @testset "Linear Regression" begin
            include("basic/LinearRegression.jl")
        end

        @testset "Logistic Regression" begin
            include("basic/LogisticRegression.jl")
        end

        @testset "Poisson Regression" begin
            include("basic/PoissonRegression.jl")
        end

        @testset "Negative Binomial Regression" begin
            include("basic/NegBinomialRegression.jl")
        end
    end

    @testset "Numerical Tests" begin
        @testset "Frequentist" begin
            include("numerical/frequentist/tests.jl")
        end
    end
end
