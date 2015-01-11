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

result = empirical.variance('poisson', list('implicit'), list('uni-dim', 'p-dim'),
                   np=20, nreps=1, niters=10000, plot=T)
plot(result$mean.estimates$implicit[['p-dim']][17,])
result$estimates$implicit[['p-dim']][1, ,50000]
dim(result$mean.estimates$sgd[['p-dim']])

model = list(sample.x=sample.x.uniform, sample.theta=sample.theta.const, 
             sample.y=sample.y.poisson, family = poisson())
result = empirical.variance(model, list('implicit'), list('p-dim'),
                            np=2, nreps=1, niters=10000, plot=F, theta.control=list(theta=c(log(2), log(4))))
plot(result$mean.estimates$implicit[['p-dim']][2,])
