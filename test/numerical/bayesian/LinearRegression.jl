mtcars = dataset("datasets", "mtcars")

tests = [
    (Prior_Ridge(), 20.080877893580514),
    (Prior_Laplace(), 20.070783434589128),
    (Prior_Cauchy(), 20.019759144845644),
    (Prior_TDist(), 20.08147561106022),
    (Prior_HorseShoe(), 20.042984550677183),
]

for (prior, test_mean) in tests
    # MCMC
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), prior)
    mcmc_prediction = predict(model, mtcars)
    @test mean(mcmc_prediction) - 2 * std(mcmc_prediction) <= test_mean && test_mean <= mean(mcmc_prediction) + 2 * std(mcmc_prediction)

    # VI
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), prior, VI())
    vi_prediction = predict(model, mtcars)
    @test mean(vi_prediction) - 2 * std(vi_prediction) <= test_mean && test_mean <= mean(vi_prediction) + 2 * std(vi_prediction)

    @test maximum(abs.(mcmc_prediction - vi_prediction)) <= 5.0
end

gauss_test = 20.0796026428345

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Prior_Gauss(), 30.0, [0.0,-3.0,1.0])
mcmc_prediction = predict(model, mtcars)
@test mean(mcmc_prediction) - 2 * std(mcmc_prediction) <= gauss_test && gauss_test <= mean(mcmc_prediction) + 2 * std(mcmc_prediction)

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Prior_Gauss(), 30.0, [0.0,-3.0,1.0], VI())
vi_prediction = predict(model, mtcars)
@test mean(vi_prediction) - 2 * std(vi_prediction) <= gauss_test && gauss_test <= mean(vi_prediction) + 2 * std(vi_prediction)

@test maximum(abs.(mcmc_prediction - vi_prediction)) <= 5.0