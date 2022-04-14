## Example 1: Linear Regression

df = dataset("datasets", "mtcars");


### Example 1.1
## formula = MPG ~ HP + WT+Gear
## data = df
## modelClass = "LinearReg"
## LikelihoodMod = "Gauss"
## PriorMod = "NULL"
## Link = "Identity"
## ComputeMethod = "Optimization"

## Ayush : m1_1 = @fitmodel(mpg ~ hp + wt+Gears, df);

m1_1 = fitmodel(@formula(MPG ~ HP + WT+Gear),df);
m1_1 = fitmodel(@formula(MPG ~ HP + WT+Gear),df,"reg","gauss","NULL","identity","opts");
m1_1 = fitmodel(@formula(MPG ~ HP + WT+Gear),df,"LinearReg","Gauss","Null","Identity","Optimization");

m1_1.fit
m1_1.sigma
m1_1.LogLike
m1_1.LogPost
m1_1.AIC
m1_1.BIC
m1_1.R_sqr
m1_1.Adjusted_R_sqr

m1_1.fittedResponse
m1_1.residuals
m1_1.Cooks_distance

plot(m1_1.Cooks_distance)



### Example 1.2
## "LinearReg","Gauss","Flat","Identity","MCMC"

m1_2 = fitmodel(@formula(MPG ~ HP + WT+Gear),df,"LinearReg","Gauss","Flat","Identity","MCMC");
m1_2.summaries
plot(m1_2.chain)
m1_2.quantiles

### Example 1.3
## "LinearReg","Gauss","Null","Identity","Optimization"
m1_3 = fitmodel(@formula(MPG ~ HP + WT*Gear),df,"LinearReg","Gauss","Null","Identity","Optimization");

m1_3.fit
m1_3.sigma
m1_3.AIC
m1_3.BIC
m1_3.sigma

### Example 1.4
## "LinearReg","Gauss","Flat","Identity","MCMC"
m1_4 = fitmodel(@formula(MPG ~ HP + WT*Gear),df,"LinearReg","Gauss","Flat","Identity","MCMC");
m1_4.summaries
plot(m1_4.chain)

m1_4.quantiles


### Example 1.5
## "LinearReg","Gauss","RidgePrior","Identity","Optimization"
m1_5 = fitmodel(@formula(MPG ~ HP + WT*Gear),df,"LinearReg","Gauss","RidgePrior","Identity","Optimization",0.1,0.95);
m1_5.fit
m1_5.SigmaPostMode
[m1_5.SigmaPostLower,m1_5.SigmaPostUpper]

### Example 1.6
## "LinearReg","Gauss","RidgePrior","Identity","MCMC"
m1_6=fitmodel(@formula(MPG ~ HP + WT*Gear),df,"LinearReg","Gauss","RidgePrior","Identity","MCMC",0.1,0.95,10000);
m1_6.summaries
m1_6.quantiles

### Example 1.7
## "LinearReg","Gauss","Lasso","Identity","Optimization"
m1_7 = fitmodel(@formula(MPG ~ HP + WT*Gear),df,"LinearReg","Gauss","Laplace","Identity","Optimization",0.1,0.95);

m1_7 = fitmodel(@formula(MPG ~ HP + WT*Gear),df,"LinearReg","Gauss","Lasso_Prior","Identity","Optimization",0.1,0.95);

m1_7.fit
m1_7.SigmaPostMode
m1_7.SigmaPostLower
m1_7.SigmaPostUpper
m1_7.LogLike
m1_7.AIC
m1_7.BIC

### Example 1.8
## "LinearReg","Gauss","Lasso","Identity","MCMC"

m1_8 = fitmodel(@formula(MPG ~ HP + WT*Gear),df,"LinearReg","Gauss","Lasso_Prior","Identity","MCMC",0.1,0.95);
m1_8.summaries
m1_8.quantiles
plot(m1_8.chain)

### Example 1.9
## "LinearReg","Gauss","Cauchy","Identity","Optimization"
m1_9 = fitmodel(@formula(MPG ~ HP + WT*Gear),df,"LinearReg","Gauss","Cauchy","Identity","Optimization");
m1_9.fit
m1_9.SigmaPostMode
m1_9.SigmaPostLower
m1_9.SigmaPostUpper
m1_9.LogLike
m1_9.AIC
m1_9.BIC

### Example 1.10
## "LinearReg","Gauss","Cauchy","Identity","MCMC"
m1_10 = CRRao(@formula(MPG ~ HP + WT*Gear),df,"LinearReg","Gauss","Cauchy","Identity","MCMC");
m1_10.summaries
m1_10.quantiles
plot(m1_9.chain)

