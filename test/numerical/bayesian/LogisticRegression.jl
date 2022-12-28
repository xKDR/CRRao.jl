turnout = dataset("Zelig", "turnout")[1:100,:] # Take a subset of rows to reduce input size

tests = [
    (
        Prior_Ridge(),
        (
            (Logit(), 0.7690822208626806),
            (Probit(), 0.7685999218881091),
            (Cloglog(), 0.7751111243871245),
            (Cauchit(), 0.7730511118602764)
        )
    ),
    (
        Prior_Laplace(),
        (
            (Logit(), 0.7718593681922629),
            (Probit(), 0.7695587585010469),
            (Cloglog(), 0.7714870967902365),
            (Cauchit(), 0.7714839338283468)
        )
    ),
    (
        Prior_Cauchy(),
        (
            (Logit(), 0.7678814727043146),
            (Probit(), 0.764699194194744),
            (Cloglog(), 0.7642369367775604),
            (Cauchit(), 0.7692152829967064)
        )
    ),
    (
        Prior_TDist(),
        (
            (Logit(), 0.588835403024102),
            (Probit(), 0.7635642627091132),
            (Cloglog(), 0.7609943137312546),
            (Cauchit(), 0.772095066757767)
        )
    ),
    (
        Prior_HorseShoe(),
        (
            (Logit(), 0.38683395333332327),
            (Probit(), 0.38253233489484173),
            (Cloglog(), 0.7667553778881738),
            (Cauchit(), 0.7706755564626601)
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