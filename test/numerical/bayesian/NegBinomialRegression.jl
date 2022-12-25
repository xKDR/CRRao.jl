sanction = dataset("Zelig", "sanction")

tests = [
    (Prior_Ridge(), 6.8753100988051274),
    (Prior_Laplace(), 6.908332048475347),
    (Prior_Cauchy(), 6.9829255933233645),
    (Prior_TDist(), 6.915515248823249),
    (Prior_HorseShoe(), 6.703023191644206),
]

for (prior, test_mean) in tests
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), prior)

    @test mean(predict(model, sanction)) ≈ test_mean
end