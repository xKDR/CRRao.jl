mtcars = dataset("datasets", "mtcars")

wts = ones(size(mtcars)[1])

tests = [
    (@formula(MPG ~ HP + WT + Gear), 157.052778719219475, 164.381458233218098),
    (@formula(MPG ~ 0 + HP + WT + Gear), 186.905400588726366, 192.768344199925281),
    (@formula(MPG ~ HP + HP^2 + WT + WT * HP), 145.922971789355415, 154.717387206153774),
    (@formula(log(MPG) ~ log(HP) + log(WT)), -48.353276320549490, -42.490332709350582)
]

for (test_formula, test_aic, test_bic) in tests
    crrao_model = fit(test_formula, mtcars, LinearRegression(); wts)
    glm_model = lm(test_formula, mtcars)
    compare_models(crrao_model, glm_model, mtcars)
    @test isapprox(aic(crrao_model), test_aic)
    @test isapprox(bic(crrao_model), test_bic)
end