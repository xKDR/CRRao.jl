turnout = dataset("Zelig", "turnout")[1:100,:] # Take a subset of rows to reduce input size

formulae = [
    @formula(Vote ~ Age + Race + Income + Educate),
    @formula(Vote ~ 0 + Age + Race + Income + Educate),
    @formula(Vote ~ Age + Age^2 + Race + Income * Educate),
    @formula(Vote ~ log(Age) + log(Educate))
]

links = [
    (Logit(), LogitLink()),
    (Probit(), ProbitLink()),
    (Cloglog(), CloglogLink()),
    (Cauchit(), CauchitLink())
]

for f in formulae
    for (crrao_link, glm_link) in links
        crrao_model = fit(f, turnout, LogisticRegression(), crrao_link)
        glm_model = glm(f, turnout, Binomial(), glm_link)
        compare_models(crrao_model, glm_model, turnout)
    end
end