source('functions.R')


#model: the name of the model 'normal':mvtnorm X, normal Y :optional: y.control=list(std=1), theta.control=list(theta=?)
#                             'poisson': poisson X, poisson Y :optional: x.control(lambda=?),theta.control=list(theta=?) 
#                             Note: theta, lambda must be an array with np elements
#       Or a list of functions: sample.x(niters, np, ...), 
#                               sample.theta(np, ...) 
#                               sample.y(mu, ...)
#                               family object
#method: a string or a list of strings: implicit, sgd, asgd
#learning.rate: a string or a list of strings: uni-dim, p-dim, adagrad
#   Note: experiment bypasses safe checks in the pacakge. wrong method or learning.rate lead to R crush.
#nreps: repetitions in a experiment
#niters: number of iterations in one experiment
#...: x.control: a list of arguments to be passed to sample.x
#     theta.control: a list of arguments to be passed to sample.theta
#     y.control: a list of arguments to be passed to sample.x

# Run the default normal model.
result = empirical.variance('normal', list('implicit', 'sgd'), list('uni-dim' ,'p-dim', 'adagrad'),
                   np=20, nreps=10, niters=10000, plot=T, theta.control = list(theta=seq(1,20, length.out = 20)))

# Plot the estimates.
# TODO: make this into a function
par(mfrow=c(2, 3))
plot(result$mean.estimates$implicit[['uni-dim']][3, ], xlab='iter', ylab='estimate', type='l')
title('implicit uni-dim')
plot(result$mean.estimates$implicit[['p-dim']][3, ], xlab='iter', ylab='estimate', type='l')
title('implicit p-dim')
plot(result$mean.estimates$implicit[['adagrad']][3, ], xlab='iter', ylab='estimate', type='l')
title('implicit adagrad')
plot(result$mean.estimates$sgd[['uni-dim']][3, ], xlab='iter', ylab='estimate', type='l', ylim=c(-10,10))
title('sgd uni-dim')
plot(result$mean.estimates$sgd[['p-dim']][3, ], xlab='iter', ylab='estimate', type='l')
title('sgd p-dim')
plot(result$mean.estimates$sgd[['adagrad']][3, ], xlab='iter', ylab='estimate', type='l')
title('sgd adagrad')
par(mfrow=c(1,1))

# This model is identical to the one in icml2014.
# It demos how to create a new model.
# for customized sample functions, extra parameters can be passed
# in by x.control, y.control, theta.control. each should be a list
model = list(sample.x=sample.x.uniform, sample.theta=sample.theta.const, 
             sample.y=sample.y.poisson, family = poisson())
result = empirical.variance(model, list('implicit','sgd'), list('uni-dim', 'p-dim'),
                            np=2, nreps=10, niters=10000, plot=T, theta.control=list(theta=c(log(2), log(4))))
par(mfrow=c(2, 1))
plot(result$mean.estimates$implicit[['uni-dim']][2, ], xlab='iter', ylab='estimate', type='l')
plot(result$mean.estimates$implicit[['p-dim']][2, ], xlab='iter', ylab='estimate', type='l')
par(mfrow=c(1, 1))
