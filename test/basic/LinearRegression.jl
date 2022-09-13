mtcars = dataset("datasets", "mtcars")

priors = [
    Prior_Ridge(),
    Prior_Laplace(),
    Prior_Cauchy(),
    Prior_TDist(),
    Prior_Uniform(),
]

model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())
@test sizeof(model) > 0

for prior in priors
    model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), prior)
    @test sizeof(model) > 0
end