mtcars = dataset("datasets", "mtcars")

priors = [
    Ridge(),
    Laplace(),
    Cauchy(),
    TDist(),
]

model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())
@test sizeof(model) > 0

for prior in priors
    model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), prior)
    @test sizeof(model) > 0
end