sanction = dataset("Zelig", "sanction")

tests = [
    (@formula(Num ~ Target + Coop + NCost), 580.673868966946884, 594.814121927084443),
    (@formula(Num ~ 0 + Target + Coop + NCost), 580.673868966946770, 594.814121927084329),
    (@formula(Num ~ Target + Target^2 + Coop + Coop * Target), 871.418230125844502, 883.201774259292506),
    (@formula(Num ~ log(Target) + log(Coop)), 944.326386272138961, 951.396512752207741)
]

for (test_formula, test_aic, test_bic) in tests
    crrao_model = fit(test_formula, sanction, PoissonRegression(), wts=ones(size(sanction)[1]))
    glm_model = glm(test_formula, sanction, Poisson(), LogLink())
    compare_models(crrao_model, glm_model, sanction)
    @test isapprox(aic(crrao_model), test_aic)
    @test isapprox(bic(crrao_model), test_bic)
end