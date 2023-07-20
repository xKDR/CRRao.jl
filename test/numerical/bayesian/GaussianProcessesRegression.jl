mtcars = dataset("datasets", "mtcars")

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ 0 + HP), mtcars, GaussianProcessesRegression(), [:MPG, :HP], MeanZero(), SE(0.0, 0.0), Euclidean())

@test get_params(model.chain)[2:3] ≈ [5.464908573213355, 3.3936838718120708]
@test noise_variance(model.chain) ≈ 9.667961411202336
@test model.chain.target ≈ -89.74473360543863