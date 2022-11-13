mtcars = dataset("datasets", "mtcars")

crrao_model = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression())
glm_model = lm(@formula(MPG ~ HP + WT + Gear), mtcars)
@test isapprox(coeftable(crrao_model).cols, coeftable(glm_model).cols)

crrao_model = fit(@formula(MPG ~ 0 + HP + WT + Gear), mtcars, LinearRegression())
glm_model = lm(@formula(MPG ~ 0 + HP + WT + Gear), mtcars)
@test isapprox(coeftable(crrao_model).cols, coeftable(glm_model).cols)