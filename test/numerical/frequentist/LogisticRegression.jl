turnout = dataset("Zelig", "turnout")[1:100,:] # Take a subset of rows to reduce input size

links = [
    (Logit(), LogitLink()),
    (Probit(), ProbitLink()),
    (Cloglog(), CloglogLink()),
    (Cauchit(), CauchitLink())
]

for (crrao_link, glm_link) in links
    crrao_model = fit(@formula(Vote ~ Age + Race + Income + Educate), turnout, LogisticRegression(), crrao_link)
    glm_model = glm(@formula(Vote ~ Age + Race + Income + Educate), turnout, Binomial(), glm_link)
    @test isapprox(coeftable(crrao_model).cols, coeftable(glm_model).cols)
end

for (crrao_link, glm_link) in links
    crrao_model = fit(@formula(Vote ~ 0 + Age + Race + Income + Educate), turnout, LogisticRegression(), crrao_link)
    glm_model = glm(@formula(Vote ~ 0 + Age + Race + Income + Educate), turnout, Binomial(), glm_link)
    @test isapprox(coeftable(crrao_model).cols, coeftable(glm_model).cols)
end