using Test
using DataFrames

@testset "KMeans Clustering Tests" begin
    df = DataFrame(A = rand(100), B = rand(100), C = rand(100))
    K = 3

    # Test fit function without variable selection
    @testset "Fit without variable selection" begin
        result = fit(df, KMeansClustering(), K)

        @test length(result.assignments) == nrow(df)
        @test size(result.centers, 1) == 2  # Centroids should have dimensions matching the input features
        @test size(result.centers, 2) == K  # There should be K centroids
    end

    # Test fit function with variable selection
    @testset "Fit with variable selection" begin
        selected_vars = [:A]  # Select only one column for clustering
        result = fit(selected_vars, df, KMeansClustering(), K)

        @test length(result.assignments) == nrow(df)
        @test size(result.centers, 1) == length(selected_vars)  # Centroids should have dimensions matching the input features
        @test size(result.centers, 2) == K  
    end
end
