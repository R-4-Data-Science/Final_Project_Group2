

# Logistic Function
# Purpose: Computes logistic function for given predictors X and coefficients beta.
# How it Works: Calculates p_i using formula p_i = 1 / (1 + exp(-X^T * beta)), returning predicted probabilities for each observation.
logistic_function <- function(beta, X) {
  # pi = 1 / (1 + exp(-xi^T * beta))
  return(1 / (1 + exp(-X %*% beta)))
}

# Cost Function (Calculating the sum in the argmin equation)
# Purpose: Calculates the cost for the current set of coefficients beta.
# How it Works: Implements logisticfunction as the negative sum of log-likelihood across all observations.
cost_function <- function(beta, X, y) {
  p <- logistic_function(beta, X)
  # -sum(yi * log(pi) + (1 - yi) * log(1 - pi))
  cost <- -sum(y * log(p + 1e-9) + (1 - y) * log(1 - p + 1e-9))  # Adding a small constant to avoid log(0)
  return(cost)
}


# Gradient of the Cost Function
# Purpose: Computes the gradient of the cost function with respect to beta.
# How it Works: Used in gradient descent to determine direction to adjust coefficients beta. Gradient calculated as X^T * (p - y).
gradient_function <- function(beta, X, y) {
  p <- 1 / (1 + exp(-X %*% beta))
  return(t(X) %*% (p - y))
}

# Logistic Regression
# This is a function presented to the user, it takes an X Vector as predictor and y values of 0 or 1 as output variables.
# The learning rate, max iterations, and tolerance can be adjusted by the user.
# Purpose: Optimizes coefficients beta to minimize the cost function.
# How it Works: Initializes beta using least-squares formula (X^T * X)^-1 * X^T * y. 
# Iteratively updates beta in the direction of negative gradient of the cost function,
# scaled by learning rate. Checks for convergence in each iteration. Returns optimized coefficients beta.
logistic_regression <- function(X, y, learning_rate = 0.01, max_iterations = 10000, tolerance = 1e-6) {
  # Initial coefficient estimation using least squares
  beta <- solve(t(X) %*% X) %*% t(X) %*% y
  previous_cost <- Inf
  
  # Iterative optimization process
  for (iteration in 1:max_iterations) {
    current_cost <- cost_function(beta, X, y)
    if (!is.na(current_cost) && !is.infinite(current_cost) && abs(previous_cost - current_cost) < tolerance) {
      break
    }
    previous_cost <- current_cost
    grad <- gradient_function(beta, X, y)
    beta <- beta - learning_rate * grad
  }
  
  # Predict and binarize predictions
  predicted_probs <- logistic_function(beta, X)
  predictions <- ifelse(predicted_probs > 0.5, 1, 0)
  
  # Confusion matrix components
  TP <- sum(predictions == 1 & y == 1)
  TN <- sum(predictions == 0 & y == 0)
  FP <- sum(predictions == 1 & y == 0)
  FN <- sum(predictions == 0 & y == 1)
  
  # Calculate metrics
  prevalence <- sum(y) / length(y)
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  false_discovery_rate <- FP / (TP + FP)
  diagnostic_odds_ratio <- (TP / FP) / (FN / TN)
  
  # Return a list containing optimized coefficients, confusion matrix, and metrics
  return(list(
    OptimizedCoefficients = beta,
    ConfusionMatrix = matrix(c(TP, FP, FN, TN), nrow = 2, byrow = TRUE),
    Prevalence = prevalence,
    Accuracy = accuracy,
    Sensitivity = sensitivity,
    Specificity = specificity,
    FalseDiscoveryRate = false_discovery_rate,
    DiagnosticOddsRatio = diagnostic_odds_ratio
  ))
}

  
plot_logistic_curve <- function(X, y) {
  # Add an intercept to X
  X_matrix <- cbind(1, X)
  
  # Compute beta estimates using your logistic regression function
  logistic_regression_result <- logistic_regression(X_matrix, y)
  beta_estimates <- logistic_regression_result$OptimizedCoefficients
  
  # Calculate predicted probabilities using the logistic function and beta estimates
  predicted_probabilities <- logistic_function(beta_estimates, X_matrix)
  
  # Create a data frame for plotting
  plot_data <- data.frame(X = X, y = y, predicted = predicted_probabilities)
  
  # Sorting the data for a smooth curve in plot
  plot_data <- plot_data[order(plot_data$X), ]
  
  # Plot the original data points
  plot(y ~ X, data = plot_data, col = "red", xlab = "Predictor", ylab = "Binary Response", 
       main = "Fitted Logistic Curve and Original Data Points",
       pch = 19)  # Points for actual responses
  
  # Add the logistic regression line
  lines(plot_data$X, plot_data$predicted, type = "l", col = "blue", lwd = 2)
}


plot_logistic_curve(mtcars$wt, mtcars$am)

#gpt link for creating cutsom logistic regression and plotting functions: https://chat.openai.com/share/2827b3e8-a2c6-4c1d-ab39-a42235cb1deb
#gpt link for additional editing, and the addition of the matrix and other metrics to the logistic regression function
# https://chat.openai.com/share/39e19490-5083-4928-a5ef-46666b1759e4
  

  
  # EXAMPLE TO BE USED
  data(mtcars)
  
  # Prepare the data
  X <- mtcars$wt
  y <- mtcars$am
  
  # Convert X to a matrix and add an intercept
  X_matrix <- cbind(1, X)
  
  beta_estimates <- logistic_regression(X_matrix, y)
  beta_estimates
  
  # Print the estimated coefficients
  beta_estimates
  plot_logistic_curve(mtcars$wt, mtcars$am)
  
  
  
  
  
  
  #ACTUAL REGRESSION FOR COMPARISON OF BETA
  data_glm <- data.frame(y = y, X = X)

  # Fit Logistic Regression model using glm()
  model_glm <- glm(y ~ X, data = data_glm, family = binomial(link = "logit"))

  # Display the summary of the model
  summary(model_glm)

  # To get the coefficients
  coefficients(model_glm)

  
# Bootstrap Function
# This is a function presented to the user and it takes the significance level, alpha, the number of bootstraps, B, 
# an X Vector as predictor and y values of 0 or 1 as output variables. 
# Purpose: To compute a more accurate confidence interval for the small data set, beta_estimates. 
# How it works: This function creates a new larger sample, boot_mean, using values from the original data set, beta_estimates.
# This larger sample is made by creating n sized samples, beta_estimates_star, and using the mean of each n sized sample. 
# The n sized samples are made by selecting numbers randomly from the original data set with replacement.
# The function then uses the quantile function to create the new confidence interval using the alpha provided by the user.
Bootstrap_function <- function(alpha, B = 20, X, y) {
  # n is the length of the vector beta_estimates that we computed earlier, but I repeated the calculation here.
  X_matrix <- cbind(1, X)
  logistic_regression_result <- logistic_regression(X_matrix, y)
  beta_estimates <- logistic_regression_result$OptimizedCoefficients
  n <- length(beta_estimates)
  boot_mean <- rep(NA, B)
  for (i in 1:B) {
    beta_estimates_star <- beta_estimates[sample(1:n, replace = TRUE)]
    boot_mean[i] <- mean(beta_estimates_star)
  }
  confidence_int <- quantile(boot_mean, c((alpha/2), 1-(alpha/2)))
  return(confidence_int)
}

# Example Bootstrap Function
data(mtcars)
# Prepare the data
X <- mtcars$wt
y <- mtcars$am
Bootstrap_function(alpha = 0.5, B = 20, X = X, y = y)

# Cut-off Value Function
# This is a function presented to the user and it takes an X Vector as predictor, y values of 0 or 1 as output variables,
# coefficients beta, and the metric value. The metric value needs to be either "prevalence", "accuracy", "sensitivity",
# "specificity", "false discovery rate", or "diagnostic odds ratio".
# Purpose: To create a plot that shows how the value of the metric chosen changes as the cut-off value changes.
# How it works: This function creates a vector for the metric chosen that has all of the values for that metric for each 
# cut-off value from 0.1 to 0.9 with steps of 0.1. It then plots this vector against the cut-off values.
cutoff_value_function <- function(X, y, beta, metric) {
 
   # Predict and binarize predictions
  predicted_probs <- logistic_function(beta, X)
  
  # create empty vectors for each metric
  prevalence_vector <- rep(NA, 9)
  accuracy_vector <- rep(NA, 9)
  sensitivity_vector <- rep(NA, 9)
  specificity_vector <- rep(NA, 9)
  FDR_vector <- rep(NA, 9)
  DOR_vector <- rep(NA, 9)
  
  for (i in seq(0.1, 0.9, by = 0.1)) {
    predictions <- ifelse(predicted_probs > i, 1, 0)
    
    # Confusion matrix components
    TP <- sum(predictions == 1 & y == 1)
    TN <- sum(predictions == 0 & y == 0)
    FP <- sum(predictions == 1 & y == 0)
    FN <- sum(predictions == 0 & y == 1)
    
    if(metric == "prevalence") {
    # Calculate prevalence
    prevalence_star <- sum(y) / length(y)
    prevalence_vector[i] <- prevalence_star
    plot(x = seq(0.1, 0.9, by = 0.1), y = prevalence_vector, main = "Prevalence Plot", xlab = "Cut-off Values", ylab = "Prevalence", pch = 16, col = "blue")
  
  } else if(metric == "accuracy") {
    # Calculate accuracy
    accuracy_star <- (TP + TN) / (TP + TN + FP + FN)
    accuracy_vector[i] <- accuracy_star
    plot(x = seq(0.1, 0.9, by = 0.1), y = accuracy_vector, main = "Accuracy Plot", xlab = "Cut-off Values", ylab = "Accuracy", pch = 16, col = "blue")
  
  } else if(metric == "sensitivity") {
    # Calculate sensitivity
    sensitivity_star <- TP / (TP + FN)
    sensitivity_vector[i] <- sensitivity_star
    plot(x = seq(0.1, 0.9, by = 0.1), y = sensitivity_vector, main = "Sensitivity Plot", xlab = "Cut-off Values", ylab = "Sensitivity", pch = 16, col = "blue")
  
  } else if(metric == "specificity") {
    # Calculate specificity
    specificity_star <- TN / (TN + FP)
    specificity_vector[i] <- specificity_star
    plot(x = seq(0.1, 0.9, by = 0.1), y = specificity_vector, main = "Specificity Plot", xlab = "Cut-off Values", ylab = "Specificity", pch = 16, col = "blue")
  
  } else if(metric == "false discovery rate") {
    # Calculate false discovery rate
    FDR_star <- FP / (TP + FP)
    FDR_vector[i] <- FDR_star
    plot(x = seq(0.1, 0.9, by = 0.1), y = FDR_vector, main = "False Discovery Rate Plot", xlab = "Cut-off Values", ylab = "False Discovery Rate", pch = 16, col = "blue")
  
  } else if(metric == "diagnostic odds ratio") {
    # Calculate diagnostic odds ratio
    DOR_star <- FP / (TP + FP)
    DOR_vector[i] <- DOR_star
    plot(x = seq(0.1, 0.9, by = 0.1), y = DOR_vector, main = "Diagnostic Odds Ratio Plot", xlab = "Cut-off Values", ylab = "Diagnostic Odds Ratio", pch = 16, col = "blue")
  }
}}

#chatgpt link for creating plots: https://chat.openai.com/share/b77ecd7f-58ef-4d44-9551-e0bb87087b39 