

# Logistic Function
logistic_function <- function(beta, X) {
  return(1 / (1 + exp(-X %*% beta)))
}

# Cost Function
cost_function <- function(beta, X, y) {
  p <- logistic_function(beta, X)
  cost <- -sum(y * log(p) + (1 - y) * log(1 - p))
  return(cost)
}

# # Optimizing the Cost Function
# logistic_regression <- function(X, y) {
#   # Initial guess for beta
#   initial_beta <- rep(0, ncol(X))
# 
#   # Use optim() to minimize the cost function
#   model <- optim(par = initial_beta, fn = cost_function, X = X, y = y)
# 
#   return(model$par)
# }

gradient_function <- function(beta, X, y) {
  p <- logistic_function(beta, X)
  return(t(X) %*% (p - y))
}

# Gradient Descent for Logistic Regression
gradient_descent <- function(X, y, learning_rate = 0.01, max_iterations = 1000, tolerance = 1e-6) {
  beta <- rep(0, ncol(X))
  
  for (iteration in 1:max_iterations) {
    grad <- gradient_function(beta, X, y)
    beta <- beta - learning_rate * grad  # Update rule
    
    # Check for convergence
    if (norm(grad, type = "2") < tolerance) {
      break
    }
  }
}
  
  
  
# Example using our created functions
  data(mtcars)

  # Prepare the data
  X <- mtcars$mpg
  y <- mtcars$am

  # Convert X to a matrix and add an intercept
  X_matrix <- cbind(1, X)

  beta_estimates <- logistic_regression(X_matrix, y)

  # Print the estimated coefficients
  beta_estimates
  
  
  
  
  
  

# Logistic Function (Calculating p_i)
logistic_function <- function(beta, X) {
  # pi = 1 / (1 + exp(-xi^T * beta))
  return(1 / (1 + exp(-X %*% beta)))
}

# Cost Function (Calculating the sum in the argmin equation)
cost_function <- function(beta, X, y) {
  p <- logistic_function(beta, X)
  # -sum(yi * log(pi) + (1 - yi) * log(1 - pi))
  return(-sum(y * log(p) + (1 - y) * log(1 - p)))
}

# Gradient of the Cost Function
gradient_function <- function(beta, X, y) {
  p <- logistic_function(beta, X)
  # Gradient computation
  return(t(X) %*% (p - y))
}

# Gradient Descent (Implementing argmin)
gradient_descent <- function(X, y, learning_rate = 0.01, max_iterations = 10000, tolerance = 1e-6) {
  beta <- rep(0, ncol(X))
  
  for (iteration in 1:max_iterations) {
    grad <- gradient_function(beta, X, y)
    beta <- beta - learning_rate * grad  # Update rule
    
    # Check for convergence
    if (norm(grad, type = "2") < tolerance) {
      break
    }
  }
  
  return(beta)
}

# Example Usage
# Using the mtcars dataset
data(mtcars)
X <- as.matrix(cbind(1, mtcars$mpg))  # Include intercept
y <- mtcars$am

# Run Gradient Descent to find beta_hat
beta_hat <- gradient_descent(X, y)
beta_hat



#Actual Logistic Regression in R for comparison
  data_glm <- data.frame(y = y, X = X)

  # Fit Logistic Regression model using glm()
  model_glm <- glm(y ~ X, data = data_glm, family = binomial(link = "logit"))

  # Display the summary of the model
  summary(model_glm)

  # To get the coefficients
  coefficients(model_glm)



