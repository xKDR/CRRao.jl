mtcars = dataset("datasets", "mtcars")

tests = [
    (Prior_Ridge(), 20.080877893580514),
    (Prior_Laplace(), 20.070783434589128),
    (Prior_Cauchy(), 20.019759144845644),
    (Prior_TDist(), 20.08147561106022),
    (Prior_HorseShoe(), 20.042984550677183),
]

for (prior, test_mean) in tests
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), prior)

    @test mean(predict(model, mtcars)) ≈ test_mean
end

gauss_test = 20.0796026428345

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Prior_Gauss(), 30.0, [0.0,-3.0,1.0], 1000)

@test mean(predict(model, mtcars)) ≈ gauss_test