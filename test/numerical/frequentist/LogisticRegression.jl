turnout = dataset("Zelig", "turnout")

wts = ones(size(turnout)[1])

tests = [
    (Logit(), LogitLink(), [
        (@formula(Vote ~ Age + Race + Income + Educate), 2033.981263703107970, 2061.985776000818532),
        (@formula(Vote ~ 0 + Age + Race + Income + Educate), 2033.981263703107970, 2061.985776000818532),
        (@formula(Vote ~ Age + Age^2 + Race + Income * Educate), 2023.429129040224325, 2062.635446257018884),
        (@formula(Vote ~ log(Age) + Educate), 2069.570124645007581, 2086.372832023633691)
    ]),
    (Probit(), ProbitLink(), [
        (@formula(Vote ~ Age + Race + Income + Educate), 2034.196513858159960, 2062.201026155870295),
        (@formula(Vote ~ 0 + Age + Race + Income + Educate), 2034.196513858159960, 2062.201026155870295),
        (@formula(Vote ~ Age + Age^2 + Race + Income * Educate), 2023.024253890667069, 2062.230571107461856),
        (@formula(Vote ~ log(Age) + Educate), 2068.082640703962625, 2084.885348082588735)
    ]),
    (Cloglog(), CloglogLink(), [
        (@formula(Vote ~ Age + Race + Income + Educate), 2036.690120606579512, 2064.694632904289847),
        (@formula(Vote ~ 0 + Age + Race + Income + Educate), 2036.690120606579512, 2064.694632904289847),
        (@formula(Vote ~ log(Age) + Educate), 2067.047234348701750, 2083.849941727327860)
    ]),
    (Cauchit(), CauchitLink(), [
        (@formula(Vote ~ Age + Race + Income + Educate), 2050.941937867712113, 2078.946450165422448),
        (@formula(Vote ~ 0 + Age + Race + Income + Educate), 2050.941937867712113, 2078.946450165422448),
        (@formula(Vote ~ Age + Age^2 + Race + Income * Educate), 2036.876992804688371, 2076.083310021483157),
        (@formula(Vote ~ log(Age) + Educate), 2085.365660144347203, 2102.168367522973313)
    ])
]

for (crrao_link, glm_link, formulae_and_values) in tests
    for (test_formula, test_aic, test_bic) in formulae_and_values
        crrao_model = fit(test_formula, turnout, LogisticRegression(), crrao_link; wts)
        glm_model = glm(test_formula, turnout, Binomial(), glm_link)
        compare_models(crrao_model, glm_model, turnout)
        @test isapprox(aic(crrao_model), test_aic)
        @test isapprox(bic(crrao_model), test_bic)
    end
end