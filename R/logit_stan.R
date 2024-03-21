library(cmdstanr)
library(stein.thinning)
source('./svgd.R')

data(logit)

set.seed(12345)

chain_len = 100000

mod<-cmdstan_model("./logit.stan", compile_model_methods=TRUE,force_recompile = TRUE)

chain<-mod$sample(data=list(Y=logit[,1],X=logit[,2:5]), chains=1, seed=12345, iter_sampling=chain_len, iter_warmup=10000)

chain$init_model_methods()

chain_steps = matrix(data=NA,nrow=chain_len,ncol=5)
chain_steps[,1]=chain$draws('y_intercept')
chain_steps[,2]=chain$draws('beta[1]')
chain_steps[,3]=chain$draws('beta[2]')
chain_steps[,4]=chain$draws('beta[3]')
chain_steps[,5]=chain$draws('beta[4]')

grad_func <- function(x0) {
  output_matrix = matrix(data=NA, nrow=nrow(x0),ncol=ncol(x0))
  
  for (ii in 1:nrow(x0)) {
    output_matrix[ii,] = chain$grad_log_prob(x0[ii,])
  }
  
  return(output_matrix)
}

chain_grads <- grad_func(chain_steps)
stein_points <- chain_steps[thin(chain_steps,chain_grads,50),]

theta = chain_steps[sample(1:nrow(chain_steps),50),]
gd_points <- svgd(theta,grad_func,n_iter=10000)

# Ground 'Truth'
print(chain$summary())

# 'Random Samples from Chain Output'
print(colMeans(theta))
# 'Shift those points using SVGD'
print(colMeans(gd_points))

# 'Test Against Stein Thinning Points'
print(colMeans(stein_points))
