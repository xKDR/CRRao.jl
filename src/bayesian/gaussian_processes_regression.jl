
function fit(formula, data::DataFrame, modelClass::GaussianProcessesRegression,IndexVar, mean, kern::Kernel, 
    DistanceClass::Euclidean)

 formula = apply_schema(formula, schema(formula, data), RegressionModel)
 select!(data, IndexVar)
 y, X = modelcols(formula, data)
 logObsNoise = -1.0
 gp = GP(X', y, mean, kern, logObsNoise)
 optimize!(gp)
 return BayesianRegression(:GaussianProcessesRegression, gp, formula)

end



function fit(formula, data::DataFrame, modelClass::GaussianProcessesRegression,IndexVar,
    DistanceClass::Euclidean)

formula = apply_schema(formula, schema(formula, data), RegressionModel)
select!(data, IndexVar)
y, X = modelcols(formula, data)
logObsNoise = -1.0
mean= MeanZero()
kern=SE(0.0,0.0)
gp = GP(X', y, mean, kern, logObsNoise)
optimize!(gp)
return BayesianRegression(:GaussianProcessesRegression, gp, formula)

end
