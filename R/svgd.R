library(spatstat.geom)

svgd_kernel <- function(theta, h = -1) {
  # Squared pairwise distance
  sq_dist = pairdist(theta)**2
  
  # Median trick for h < 0
  if (h < 0) {
    h = median(sq_dist)
    h = sqrt(.5*h/log(nrow(theta)+1))
  }
  
  # Calculate radial basis function kernel for theta
  Kxy = exp(-sq_dist / h**2 / 2)
  
  dxkxy = -Kxy%*%theta
  sumkxy = colSums(Kxy)
  
  for (ii in 1:ncol(theta)) {
    dxkxy[,ii] = dxkxy[,ii] + theta[,ii] * sumkxy
  }
  dxkxy = dxkxy / (h**2)
  
  return(list(Kxy,dxkxy))
}

# x0 is a nxp matrix of points, lnprob is a function which returns the gradient log probabilities for your desired probability space,
# stepsize is how far each iteration moves each point
svgd <- function(x0, lnprob, n_iter = 1000, stepsize = 1e-3, alpha = 0.9, debug=FALSE) {
  fudge_factor = 1e-6
  historical_grad = 0
  
  theta = x0
  
  for (ii in 1:n_iter) {
    if (debug) { cat("Iteration: ", ii, "\n") }
    
    # calculating the kernel matrix from gradient function
    grad_log_prob = lnprob(theta)
    
    kxy = svgd_kernel(theta, h = -1)[[1]]
    dxkxy = svgd_kernel(theta, h = -1)[[2]]
    grad_theta = (kxy %*% grad_log_prob + dxkxy) / nrow(x0)
    
    # Update historical gradients
    if (ii == 1) {
      historical_grad = historical_grad + grad_theta ** 2
    } else {
      historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
    }
    
    # Perform step based on gradient step size calculation
    adj_grad = grad_theta / (fudge_factor + sqrt(historical_grad))
    theta = theta + stepsize * adj_grad
  }
  
  return(theta)
}
