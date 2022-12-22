mtcars = dataset("datasets", "mtcars")

tests = [
    (Prior_Ridge(), 20.094329014886135),
    (Prior_Laplace(), 20.070803242904567),
    (Prior_Cauchy(), 20.089299279637608),
    (Prior_TDist(), 20.07675037883895),
    (Prior_HorseShoe(), 20.08147629414915),
]

for (prior, test_mean) in tests
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), prior)

    @test mean(predict(model, mtcars)) ≈ test_mean
end

gauss_test = 20.077725307634

CRRao.set_rng(StableRNG(123))
model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Prior_Gauss(), 30.0, [0.0,-3.0,1.0], 1000)

@test mean(predict(model, mtcars)) ≈ gauss_test