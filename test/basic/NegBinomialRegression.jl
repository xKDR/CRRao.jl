sanction = dataset("Zelig", "sanction")

priors = [
    Prior_Ridge(),
    Prior_Laplace(),
    Prior_Cauchy(),
    Prior_TDist(),
]

CRRao.set_rng(StableRNG(123))

@testset "Frequentist NegativeBinomial Regression" begin
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression())
    @test sizeof(model) > 0
    @test coef(model) ≈ [-1.1450754826987306, 0.008593046159347997, 
                         1.0639242654967949, -0.23501846904848583,
                         1.307960123972135, 0.18348159198531477]
    @test aic(model) ≈ 344.8795490118083
    @test bic(model) ≈ 361.3765107986354
end

for prior in priors
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), prior)
    @test sizeof(model) > 0
end