turnout = dataset("Zelig", "turnout")[1:100,:] # Take a subset of rows to reduce input size

links = [Logit(), Probit(), Cloglog(), Cauchit()]

priors = [
    Prior_Ridge(),
    Prior_Laplace(),
    Prior_Cauchy(),
    Prior_TDist(),
]

for link in links
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), link)
    @test sizeof(model) > 0
end

uniform_prior_tests = [
    (Logit(),125), 
    (Probit(),123), 
    (Cloglog(),125), 
    (Cauchit(),123)
]

for (link,seed) in uniform_prior_tests
    CRRao.set_rng(StableRNG(seed))
    model = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), link, Prior_Uniform())
    @test sizeof(model) > 0
end

for prior in priors
    for link in links
        CRRao.set_rng(StableRNG(123))
        model = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), link, prior)
        @test sizeof(model) > 0
    end
end