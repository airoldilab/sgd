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

result = empirical.variance('normal', list('sgd', 'implicit'), list('uni-dim', 'px-dim'),
                   np=20, nreps=1, niters=50000, plot=T)
plot(result$mean.estimates$sgd[['px-dim']][3,])
