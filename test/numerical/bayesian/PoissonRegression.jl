sanction = dataset("Zelig", "sanction")

tests = [
    (Prior_Ridge(), 7.156569426585206),
    (Prior_Laplace(), 7.147034162448096),
    (Prior_Cauchy(), 7.160021974618625),
    (Prior_TDist(), 7.144672898872307),
    (Prior_HorseShoe(), 7.139133430699899),
]

for (prior, test_mean) in tests
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), prior)

    @test mean(predict(model, sanction)) ≈ test_mean
end