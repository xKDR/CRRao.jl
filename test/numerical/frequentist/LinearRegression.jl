mtcars = dataset("datasets", "mtcars")

wts = ones(size(mtcars)[1])

tests = [
    (@formula(MPG ~ HP + WT + Gear), 157.052778719219475, 164.381458233218098, 0.46203539153321815),
    (@formula(MPG ~ 0 + HP + WT + Gear), 186.905400588726366, 192.768344199925281, nothing),
    (@formula(MPG ~ HP + HP^2 + WT + WT * HP), 145.922971789355415, 154.717387206153774, 0.3036259931929549),
    (@formula(log(MPG) ~ log(HP) + log(WT)), -48.353276320549490, -42.490332709350582, 0.0018795270720472204)
]

for (test_formula, test_aic, test_bic, test_pvalue) in tests
    crrao_model = fit(test_formula, mtcars, LinearRegression(); wts)
    glm_model = lm(test_formula, mtcars)
    compare_models(crrao_model, glm_model, mtcars)
    @test isapprox(aic(crrao_model), test_aic)
    @test isapprox(bic(crrao_model), test_bic)
    if test_pvalue !== nothing
        bp_test = BPTest(crrao_model, mtcars)
        @test isapprox(pvalue(bp_test), test_pvalue)
    end
end

CRRao.set_rng(StableRNG(123))
container = fit(@formula(MPG ~ HP + WT + Gear), mtcars, LinearRegression(), Boot_Residual())
@test isapprox(container.Coef, [32.13093064426382, -0.036497081866302794, -3.2257634531003574, 1.0001169309164597])