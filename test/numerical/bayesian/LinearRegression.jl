mtcars = dataset("datasets", "mtcars")

tests = [
    (Ridge(), 20.080877893580514),
    (Laplace(), 20.070783434589128),
    (Cauchy(), 20.019759144845644),
    (TDist(), 20.08147561106022),
    (HorseShoe(), 20.042984550677183),
]

for (prior, test_mean) in tests
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), prior)

    @test mean(predict(model, mtcars)) ≈ test_mean
end

gauss_test = 20.0796026428345

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Gauss(), 30.0, [0.0,-3.0,1.0], 1000)

@test mean(predict(model, mtcars)) ≈ gauss_test