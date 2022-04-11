## Poisson Regression - Classical

function fitmodel(formula::FormulaTerm, data::DataFrame, modelClass::PoissonRegression)
    ans = Poisson_Regression_fit(formula,data)
    ans     
end
