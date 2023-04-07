using StateSpaceModels, CRRao, DataFrames

# load the data
data = DataFrame(x = [1.3, 2.5, 3.1, 4.7, 5.2, 6.8])

# define the state space model
ssm = SSModel(data.x, ARIMA(p=1, d=1, q=1))

# perform Kalman filtering and smoothing
kalman_result = kalman(ssm)

# get estimated states and observations
est_states = states(kalman_result)
est_obs = observations(kalman_result)

