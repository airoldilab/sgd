source('functions.R')


#model: the name of the model
#       Or a list of functions: sample.x(niters, np, ...), 
#                               sample.theta(np, ...) 
#                               sample.y(mu, ...)
#                               family object
#method: a string or a list of strings
#learning.rate: a string or a list of strings
#nreps: repetitions in a experiment
#niters: number of iterations in one experiment
#...: x.control: a list of arguments to be passed to sample.x
#     theta.control: a list of arguments to be passed to sample.theta
#     y.control: a list of arguments to be passed to sample.x

result = empirical.variance('poisson', list('implicit', 'sgd'), list('p-dim'),
                   np=2, nreps=10, niters=50000, plot=T)

par(mfrow=c(2, 2))
plot(result$mean.estimates$implicit[['p-dim']][1, ], xlab='iter', ylab='estimate')
plot(result$mean.estimates$implicit[['p-dim']][5, ], xlab='iter', ylab='estimate')
plot(result$mean.estimates$implicit[['p-dim']][10, ], xlab='iter', ylab='estimate')
plot(result$mean.estimates$implicit[['p-dim']][15, ], xlab='iter', ylab='estimate')
par(mfrow=c(1,1))


result$estimates$implicit[['p-dim']][1, ,50000]
dim(result$mean.estimates$sgd[['p-dim']])

model = list(sample.x=sample.x.uniform, sample.theta=sample.theta.const, 
             sample.y=sample.y.poisson, family = poisson())
result = empirical.variance(model, list('implicit','sgd'), list('uni-dim', 'p-dim'),
                            np=2, nreps=10, niters=50000, plot=T, theta.control=list(theta=c(log(2), log(4))))
par(mfrow=c(2, 1))
plot(result$mean.estimates$implicit[['p-dim']][1, ], xlab='iter', ylab='estimate')
plot(result$mean.estimates$implicit[['p-dim']][2, ], xlab='iter', ylab='estimate')
par(mfrow=c(1, 1))
