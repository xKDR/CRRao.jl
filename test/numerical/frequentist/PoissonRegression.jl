sanction = dataset("Zelig", "sanction")

formulae = [
    @formula(Num ~ Target + Coop + NCost),
    @formula(Num ~ 0 + Target + Coop + NCost),
    @formula(Num ~ Target + Target^2 + Coop + Coop * Target),
    @formula(Num ~ log(Target) + log(Coop))
]

for f in formulae
    crrao_model = fit(f, sanction, PoissonRegression())
    glm_model = glm(f, sanction, Poisson(), LogLink())
    compare_models(crrao_model, glm_model, sanction)
end