sanction = dataset("Zelig", "sanction")

tests = [
    (@formula(Num ~ Target + Coop + NCost), 344.8795500751855, 361.37651186201265),
    (@formula(Num ~ 0 + Target + Coop + NCost), 344.87955007518536, 361.37651186201253),
    (@formula(Num ~ Target + Target^2 + Coop + Coop * Target), 362.262357097934, 376.40261005807156),
    (@formula(Num ~ log(Target) + log(Coop)), 367.1949690199781, 376.6218043267365)
]

for (test_formula, test_aic, test_bic) in tests
    crrao_model = fit(test_formula, sanction, NegBinomRegression())
    glm_model = negbin(test_formula, sanction, LogLink())
    compare_models(crrao_model, glm_model, sanction)
    @test isapprox(aic(crrao_model), test_aic)
    @test isapprox(bic(crrao_model), test_bic)
end