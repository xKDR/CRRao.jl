sanction = dataset("Zelig", "sanction")

priors = [
    Prior_Ridge(),
    Prior_Laplace(),
    Prior_Cauchy(),
    Prior_TDist(),
]

CRRao.set_rng(StableRNG(123))
model = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression())
@test sizeof(model) > 0

for prior in priors
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), prior)
    @test sizeof(model) > 0
end