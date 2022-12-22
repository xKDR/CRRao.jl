sanction = dataset("Zelig", "sanction")

tests = [
    (Prior_Ridge(), 6.89333113986786),
    (Prior_Laplace(), 6.868506051646364),
    (Prior_Cauchy(), 6.871750107984425),
    (Prior_TDist(), 6.871687824045264),
    (Prior_HorseShoe(), 6.512395375168992),
]

for (prior, test_mean) in tests
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), prior)

    @test mean(predict(model, sanction)) ≈ test_mean
end