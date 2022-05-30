turnout = dataset("Zelig", "turnout")[1:100,:] # Take a subset of rows to reduce input size

links = [Logit(), Probit(), Cloglog(), Cauchit()]

priors = [
    Prior_Ridge(),
    Prior_Laplace(),
    Prior_Cauchy(),
    Prior_TDist(),
    Prior_Uniform(),
]

CRRao.set_rng(StableRNG(55))

for link in links
    model = @fitmodel((Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), link)
    @test sizeof(model) > 0
end

for prior in priors
    for link in links
        model = @fitmodel((Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), link, prior)
        @test sizeof(model) > 0
    end
end