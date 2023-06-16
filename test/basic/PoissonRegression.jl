sanction = dataset("Zelig", "sanction")

priors = [
    Ridge(),
    Laplace(),
    Cauchy(),
    TDist(),
]

model = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression())
@test sizeof(model) > 0

for prior in priors
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), prior)
    @test sizeof(model) > 0
end