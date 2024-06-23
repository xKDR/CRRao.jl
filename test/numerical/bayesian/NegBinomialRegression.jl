sanction = dataset("Zelig", "sanction")

tests = [
    (Prior_Ridge(), 6.999865486088317),
    (Prior_Laplace(), 6.886529206600885),
    (Prior_Cauchy(), 6.900001819752649),
    (Prior_TDist(), 6.876415480722939),
    (Prior_HorseShoe(), 6.902138507950901),
]

for (prior, test_mean) in tests
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression(), prior)
    prediction = predict(model, sanction)
    @test mean(prediction) - 2 * std(prediction) <= test_mean && test_mean <= mean(prediction) + 2 * std(prediction)
end