sanction = dataset("Zelig", "sanction")

tests = [
    (Prior_Ridge(), 7.163048138457556),
    (Prior_Laplace(), 7.164837449702468),
    (Prior_Cauchy(), 7.166326185314563),
    (Prior_TDist(), 7.167147727917408),
    (Prior_HorseShoe(), 7.158818008027834),
]

for (prior, test_mean) in tests
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), prior)

    @test mean(predict(model, sanction)) ≈ test_mean
end