using Clustering, DataFrames

struct KMeansClustering end

"""
    fit(df::DataFrame, modelClass::KMeansClustering, K::Int64; max_iters::Int=100, tol::Float64=1e-4)


# Arguments
- `df`: A DataFrame where each row is an observation and each column is a feature.
- `K`: The number of clusters to form.
- `max_iters`: (Optional) Maximum number of iterations for the K-means algorithm. Default is 100.
- `tol`: (Optional) Tolerance for convergence. Default is 1e-4.

# Returns
- 'KmeansResult' object that contains the following fields providing details about the clustering outcome:

  - centers: This is a matrix where each column is the centroid of a cluster. The number of columns is equal to the number of clusters k, and the number of rows is equal to the number of features in the dataset.
  - assignments: An array indicating the cluster assignment for each observation in the dataset. The length of this array is equal to the number of observations, and each element is an integer representing the cluster index to which the observation has been assigned.
  - costs: An array of the costs associated with each observation, typically representing the squared distance from each observation to its assigned cluster center.
  - counts: An array indicating the number of observations assigned to each cluster.
  - totalcost: The total cost of the clustering solution, which is the sum of all individual observation costs. This can be interpreted as a measure of the clustering quality, with lower values indicating a better fit.
  - converged: A boolean value indicating whether the algorithm has converged. The algorithm is considered to have converged if the centroids do not change significantly in the last iteration or if it reaches the maximum number of iterations.
  - iterations: The number of iterations the algorithm ran before stopping. This could be due to convergence or reaching the maximum number of allowed iterations.


  

Perform K-means clustering on the entire DataFrame using `K` clusters.


"""
function fit(
    df::DataFrame,
    modelClass::KMeansClustering,
    K::Int64;
    max_iters::Int = 100,
    tol::Float64 = 1e-4
)
    # Ensure the DataFrame is converted to a Float64 matrix for Clustering.jl
    data = Matrix{Float64}(df)

    # Perform K-means clustering. Since Clustering.jl expects data in columns, so we transpose.
    result = kmeans(data', K; maxiter = max_iters, tol = tol)

    return (result)
end

"""
    fit(VarName, df::DataFrame, modelClass:: KMeansClustering, K::Int64; max_iters::Int=100, tol::Float64=1e-4)

    # Arguments
    - `df`: A DataFrame where each row is an observation and each column is a feature.
    - `K`: The number of clusters to form.
    - `max_iters`: (Optional) Maximum number of iterations for the K-means algorithm. Default is 100.
    - `tol`: (Optional) Tolerance for convergence. Default is 1e-4.
    
    # Returns
    - 'KmeansResult' object that contains the following fields providing details about the clustering outcome:
    
      - centers: This is a matrix where each column is the centroid of a cluster. The number of columns is equal to the number of clusters k, and the number of rows is equal to the number of features in the dataset.
      - assignments: An array indicating the cluster assignment for each observation in the dataset. The length of this array is equal to the number of observations, and each element is an integer representing the cluster index to which the observation has been assigned.
      - costs: An array of the costs associated with each observation, typically representing the squared distance from each observation to its assigned cluster center.
      - counts: An array indicating the number of observations assigned to each cluster.
      - totalcost: The total cost of the clustering solution, which is the sum of all individual observation costs. This can be interpreted as a measure of the clustering quality, with lower values indicating a better fit.
      - converged: A boolean value indicating whether the algorithm has converged. The algorithm is considered to have converged if the centroids do not change significantly in the last iteration or if it reaches the maximum number of iterations.
      - iterations: The number of iterations the algorithm ran before stopping. This could be due to convergence or reaching the maximum number of allowed iterations.
    
    

Perform K-means clustering on selected variables from the DataFrame using `K` clusters.
"""
function fit(
    VarName,
    df::DataFrame,
    modelClass::KMeansClustering,
    K::Int64;
    max_iters::Int = 100,
    tol::Float64 = 1e-4
)
    # Select specified variables from DataFrame
    selected_data = select(df, VarName)

    # Convert the selected DataFrame columns to a Float64 matrix
    data = Matrix{Float64}(selected_data)

    # Perform K-means clustering
    result = kmeans(data', K; maxiter = max_iters, tol = tol)

    return (result)
end
