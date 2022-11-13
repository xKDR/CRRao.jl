sanction = dataset("Zelig", "sanction")

crrao_model = fit(@formula(Num ~ Target + Coop + NCost), sanction, NegBinomRegression())
glm_model = glm(@formula(Num ~ Target + Coop + NCost), sanction, NegativeBinomial(), LogLink())
@test isapprox(coeftable(crrao_model).cols, coeftable(glm_model).cols)

crrao_model = fit(@formula(Num ~ 0 + Target + Coop + NCost), sanction, NegBinomRegression())
glm_model = glm(@formula(Num ~ 0 + Target + Coop + NCost), sanction, NegativeBinomial(), LogLink())
@test isapprox(coeftable(crrao_model).cols, coeftable(glm_model).cols)