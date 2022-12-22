turnout = dataset("Zelig", "turnout")[1:100,:] # Take a subset of rows to reduce input size

tests = [
    (
        Prior_Ridge(),
        (
            (Logit(), 0.7726301574939893),
            (Probit(), 0.7718991855077517),
            (Cloglog(), 0.7729607868936296),
            (Cauchit(), 0.773088648259103)
        )
    ),
    (
        Prior_Laplace(),
        (
            (Logit(), 0.7721450629705013),
            (Probit(), 0.7700924638261658),
            (Cloglog(), 0.7725984571763792),
            (Cauchit(), 0.7733539997964879)
        )
    ),
    (
        Prior_Cauchy(),
        (
            (Logit(), 0.7490077946647627),
            (Probit(), 0.7666419321169409),
            (Cloglog(), 0.7630424969124491),
            (Cauchit(), 0.7751462774369108)
        )
    ),
    (
        Prior_TDist(),
        (
            (Logit(), 0.5859376296718818),
            (Probit(), 0.7612744071932722),
            (Cloglog(), 0.7584442886274094),
            (Cauchit(), 0.7715325526207547)
        )
    ),
    (
        Prior_HorseShoe(),
        (
            (Logit(), 0.38795793121702976),
            (Probit(), 0.4088010293870976),
            (Cloglog(), 0.7662231188565767),
            (Cauchit(), 0.7685459396568979)
        )
    ),
]

for (prior, prior_testcases) in tests
    for (link, test_mean) in prior_testcases
        CRRao.set_rng(StableRNG(123))
        model = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), link, prior)

        @test mean(predict(model, turnout)) ≈ test_mean
    end
end