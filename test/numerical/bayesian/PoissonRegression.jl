sanction = dataset("Zelig", "sanction")

tests = [
    (Ridge(), 7.163048138457556),
    (Laplace(), 7.164837449702468),
    (Cauchy(), 7.166326185314563),
    (TDist(), 7.167147727917408),
    (HorseShoe(), 7.158818008027834),
]

for (prior, test_mean) in tests
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), prior)

    @test mean(predict(model, sanction)) ≈ test_mean
end