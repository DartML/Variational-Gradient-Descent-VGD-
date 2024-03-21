data {
  //y value prediction
  //this is all the output y values we have from the data
  array[100] int<lower=0,upper=1> Y;
  
  //predictor matrix (x1,x2,x3,x4) by data (100), so 100x4
  //this is all the independent variable data we have as a matrix
  matrix[100,4] X;
}

parameters {
  //this is the y-intercept we predict; without this, the data isn't correct
  real y_intercept;
  
  //this is the vector of our predictor paramaters; so Beta_i corrosponds to x_i
  vector[4] beta;
}

model {
  // this is defining the prior for our variables ; the geyer document specified them to be ~ normal (0,2) so that's what I did
  y_intercept ~ normal(0,2);
  beta ~ normal(0,2);
  
  //we do bernoulli_logit because we have a logistic regression and we only have 0 or 1 outcomes. If we had more, we could do categorical_logit. Then. since it is logitic, we know it is of the form y_int+X*betas
  Y ~ bernoulli_logit(y_intercept + X * beta);
}

