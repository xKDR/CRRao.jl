using RDatasets, StatsModels, StatsPlots, NLSolversBase

include("CRRao.jl")
using .CRRao

## Example 3: Poisson Regression

sanction = dataset("Zelig", "sanction");
sanction
 
m3_1 <- fitmodel(@formula(Num ~ Target + Coop + NCost), sanction,PoissonRegression())

CRRao.Poisson_Regression_fit(@formula(Num ~ Target + Coop + NCost), sanction)

## Example 1: Linear Regression

df = dataset("datasets", "mtcars");


### Example 1.1
## formula = MPG ~ HP + WT+Gear
## data = df

m1_1 = fitmodel(@formula(MPG ~ HP + WT+Gear),df,LinearRegression());

m1_1.fit
m1_1.sigma
m1_1.LogLike

m1_1.AIC
m1_1.BIC
m1_1.R_sqr
m1_1.Adjusted_R_sqr

m1_1.fittedResponse
m1_1.residuals
m1_1.Cooks_distance

plot(m1_1.Cooks_distance)

## Linear Regression - Ridge Prior

m1_2 = fitmodel(@formula(MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Ridge());
m1_2.summaries
plot(m1_2.chain)
m1_2.quantiles

## Linear Regression - Laplace Prior

m1_3 = fitmodel(@formula(MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Laplace());
m1_3.summaries
plot(m1_3.chain)
m1_3.quantiles



## Linear Regression - Cauchy Prior

m1_4 = fitmodel(@formula(MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_Cauchy());
m1_4.summaries
plot(m1_4.chain)
m1_4.quantiles

## Linear Regression - T-Distributed Prior

m1_5 = fitmodel(@formula(MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_TDist());
m1_5.summaries
plot(m1_5.chain)
m1_5.quantiles

## Linear Regression - Uniform Prior

m1_6 = fitmodel(@formula(MPG ~ HP + WT+Gear),df,LinearRegression(),Prior_TDist());
m1_6.summaries
plot(m1_6.chain)
m1_6.quantiles


### LogisticReg - Classical method
turnout = dataset("Zelig", "turnout")


m2_1 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate)
                ,turnout,LogisticRegression(),Logit());
m2_1.fit
m2_1.modelClass
m2_1.LogLike
m2_1.AIC
m2_1.BIC


m2_2 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate)
                ,turnout,LogisticRegression(),Probit());
m2_2.fit
m2_2.BIC


m2_3 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate)
                ,turnout,LogisticRegression(),Cloglog());
m2_3.fit
m2_3.BIC


m2_4 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate)
                ,turnout,LogisticRegression(),Cauchit());
m2_4.fit
m2_4.BIC


### LogisticReg - with Ridge Prior


m2_5 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Logit(),Prior_Ridge())
m2_5.summaries
plot(m2_5.chain)
m2_5.quantiles


m2_6 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Probit(),Prior_Ridge(),1.0)

m2_6.summaries
plot(m2_6.chain)
m2_6.quantiles


m2_7 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Cloglog(),Prior_Ridge(),1.0)

m2_7.summaries
plot(m2_7.chain)
m2_7.quantiles


m2_8 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Cauchit(),Prior_Ridge(),1.0)

m2_8.summaries
plot(m2_8.chain)
m2_8.quantiles

### LogisticReg - with Laplace Prior


m2_9 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Logit(),Prior_Laplace())

m2_9.summaries
plot(m2_9.chain)
m2_9.quantiles

m2_10 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Probit(),Prior_Laplace())

m2_10.summaries
plot(m2_10.chain)
m2_10.quantiles


m2_11 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Cloglog(),Prior_Laplace(),1.0)

m2_11.summaries
plot(m2_11.chain)
m2_11.quantiles


m2_12 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Cauchit(),Prior_Laplace(),1.0)

m2_12.summaries
plot(m2_12.chain)
m2_12.quantiles

### LogisticReg - with Cauchy Prior

m2_13 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Logit(),Prior_Cauchy(),1.0)

m2_13.summaries
plot(m2_13.chain)
m2_13.quantiles


m2_14 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Probit(),Prior_Cauchy(),2.0)
m2_14.summaries
plot(m2_14.chain)
m2_14.quantiles
                


m2_15 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Cloglog(),Prior_Cauchy(),1.0)
m2_15.summaries
plot(m2_15.chain)
m2_15.quantiles
                

m2_16 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Cauchit(),Prior_Cauchy(),1.0)
m2_16.summaries
plot(m2_16.chain)
m2_16.quantiles
 
### LogisticReg - with T-Dist Prior

m2_17 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Logit(),Prior_TDist(),1.0)
m2_17.summaries
plot(m2_17.chain)
m2_17.quantiles

m2_18 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Probit(),Prior_TDist(),1.0)
m2_18.summaries
plot(m2_18.chain)
m2_18.quantiles


m2_19 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Cloglog(),Prior_TDist(),1.0)
m2_19.summaries
plot(m2_19.chain)
m2_19.quantiles

m2_20 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Cauchit(),Prior_TDist(),1.0)
m2_20.summaries
plot(m2_20.chain)
m2_20.quantiles


### LogisticReg - with Uniform Prior


m2_21 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Logit(),Prior_Uniform(),1.0)
m2_21.summaries
plot(m2_21.chain)
m2_21.quantiles


m2_22 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Probit(),Prior_Uniform(),1.0)
m2_22.summaries
plot(m2_22.chain)
m2_22.quantiles

m2_23 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Cloglog(),Prior_Uniform(),1.0)
m2_23.summaries
plot(m2_23.chain)
m2_23.quantiles
                
m2_24 = fitmodel(@formula(Vote ~ Age + Race +Income + Educate),turnout
                ,LogisticRegression(),Cauchit(),Prior_Uniform(),1.0)
m2_24.summaries
plot(m2_24.chain)
m2_24.quantiles

