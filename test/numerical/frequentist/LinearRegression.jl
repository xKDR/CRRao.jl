mtcars = dataset("datasets", "mtcars")

formulae = [
    @formula(MPG ~ HP + WT + Gear),
    @formula(MPG ~ 0 + HP + WT + Gear),
    @formula(MPG ~ HP + HP^2 + WT + WT * HP),
    @formula(log(MPG) ~ log(HP) + log(WT))
]

for f in formulae
    crrao_model = fit(f, mtcars, LinearRegression())
    glm_model = lm(f, mtcars)
    compare_models(crrao_model, glm_model, mtcars)
end