sanction = dataset("Zelig", "sanction")

tests = [
    (Prior_Ridge(), 7.177578002644547),
    (Prior_Laplace(), 7.1454141602741785),
    (Prior_Cauchy(), 7.148699646242317),
    (Prior_TDist(), 7.165968828611132),
    (Prior_HorseShoe(), 7.144190707091213),
]

for (prior, test_mean) in tests
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, PoissonRegression(), prior)

    @test mean(predict(model, sanction)) ≈ test_mean
end