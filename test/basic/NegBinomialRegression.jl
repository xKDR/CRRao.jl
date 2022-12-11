sanction = dataset("Zelig", "sanction")

priors = [
    Prior_Ridge(),
    Prior_Laplace(),
    Prior_Cauchy(),
    Prior_TDist(),
]

CRRao.set_rng(StableRNG(123))

@testset "Frequentist Binomial Regression" begin
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression())
    @test sizeof(model) > 0
    @test coef(model) ≈ [ -1.1451746163020158, 0.00862527221332845, 1.0639661414178674, -0.23510970207997078,
                          1.3076725069242925, 0.18345340856394354]
    @test aic(model) ≈ 344.8795490118083
    @test bic(model) ≈ 361.3765107986354
end

for prior in priors
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), prior)
    @test sizeof(model) > 0
end