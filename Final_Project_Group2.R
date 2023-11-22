

# Logistic Function
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
  p <- 1 / (1 + exp(-X %*% beta))
  return(t(X) %*% (p - y))
}

# Gradient Descent for Logistic Regression
gradient_descent <- function(X, y, learning_rate = 0.01, max_iterations = 1000, tolerance = 1e-6) {
  beta <- solve(t(X) %*% X) %*% t(X) %*% y
  previous_cost <- Inf
  
  for (iteration in 1:max_iterations) {t
    current_cost <- cost_function(beta, X, y)

    if (abs(previous_cost - current_cost) < tolerance) {
      break
    }
    previous_cost <- current_cost
    
    # Update beta based on the gradient
    grad <- gradient_function(beta, X, y)
    beta <- beta - learning_rate * grad
  }
  
  return(beta)
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

#Actual Logistic Regression in R for comparison
  data_glm <- data.frame(y = y, X = X)

  # Fit Logistic Regression model using glm()
  model_glm <- glm(y ~ X, data = data_glm, family = binomial(link = "logit"))

  # Display the summary of the model
  summary(model_glm)

  # To get the coefficients
  coefficients(model_glm)



