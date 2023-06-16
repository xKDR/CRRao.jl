sanction = dataset("Zelig", "sanction")

tests = [
    (Ridge(), 6.999865486088317),
    (Laplace(), 6.886529206600885),
    (Cauchy(), 6.900001819752649),
    (TDist(), 6.876415480722939),
    (HorseShoe(), 6.902138507950901),
]

for (prior, test_mean) in tests
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), prior)

    @test mean(predict(model, sanction)) ≈ test_mean
end