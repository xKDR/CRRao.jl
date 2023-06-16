turnout = dataset("Zelig", "turnout")[1:100,:] # Take a subset of rows to reduce input size

links = [Logit(), Probit(), Cloglog(), Cauchit()]

priors = [
    Ridge(),
    Laplace(),
    Cauchy(),
    TDist(),
]

for link in links
    CRRao.set_rng(StableRNG(123))
    model = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), link)
    @test sizeof(model) > 0
end

for prior in priors
    for link in links
        CRRao.set_rng(StableRNG(123))
        model = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), link, prior)
        @test sizeof(model) > 0
    end
end