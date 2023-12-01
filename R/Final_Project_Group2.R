#' @title Logistic Function
#'
#' @description Computes logistic function for given predictors X and coefficients beta.
#' @param beta A \code{vector} of coefficients that represent the relationship
#' between the predictor variable and the log-odds of the outcome.
#' @param X A \code{matrix} containing the predictor variables. Each row corresponds
#' to an observation, and each column corresponds to a predictor variable.
#' @return A \code{vector} containing the estimated probability of the dependent
#' variable being a \code{1} or a \code{0}.
#' @author Cameron Tice
#' @export
#'
#' @examples
#' X <- matrix(c(1, 2, 3, 4, 1, 0), nrow = 3, byrow = TRUE)
#' beta <- c(0.5, -0.2)
#' predicted_probabilities <- logistic_regression(beta, X)

logistic_function <- function(beta, X) {
  # pi = 1 / (1 + exp(-xi^T * beta))
  return(1 / (1 + exp(-X %*% beta)))
}

#' @title Cost Function for Logistic Regression
#'
#' @description Computes the cost for logistic regression given coefficients,
#' predictor variables, and observed outcomes using the logistic function and
#' the cross-entropy loss formula.
#' @param beta A \code{vector} of coefficients for the logistic regression model.
#' @param X A \code{matrix} containing the predictor variables. Each row corresponds
#' to an observation, and each column corresponds to a predictor variable.
#' @param y A \code{vector} representing the observed binary outcomes (0 or 1)
#' for each observation.
#' @return A \code{numeric} value the total cost or loss calculated using the
#' cross-entropy loss formula. The return value indicates how well the modeled
#' coefficients fit the observed data. A better fit is represented by a lower value.
#' @details The function calculates the predicted probabilities using the logistic function
#' and then computes the cost using the formula:
#' \code{-sum(y * log(p + 1e-9) + (1 - y) * log(1 - p + 1e-9))}.
#' The small constant (1e-9) is added to probabilities in the log function to avoid log(0).
#' @author Cameron Tice
#' @export
#'
#' @examples
#' beta <- c(0.5, -0.25)
#' X <- matrix(c(1, 2, 3, 2, 1, 0), nrow = 3, byrow = TRUE)
#' y <- c(1, 0, 1)
#' cost_model <- cost_function(beta, X, y)


cost_function <- function(beta, X, y) {
  p <- logistic_function(beta, X)
  cost <- -sum(y * log(p + 1e-9) + (1 - y) * log(1 - p + 1e-9))
  return(cost)
}

#' @title Gradient Function for Logistic Regression
#'
#' @description Computes the gradient of the cost function for logistic regression.
#' This function calculates the difference between the predicted probabilities and
#' the actual outcomes, scaled by the predictor variables.
#' @param beta A \code{vector} of coefficients for the logistic regression model.
#' @param X A \code{matrix} containing the predictor variables. Each row corresponds
#' to an observation, and each column corresponds to a predictor variable.
#' @param y A \code{vector} representing the observed binary outcomes (0 or 1)
#' for each observation.
#' @return A \code{matrix} that represents the gradient of the cost function with
#' respect to the coefficients.
#' @details Calculates the gradient as the product of the transposed matrix of
#' predictor variables and the vector of residuals. The gradient indicates the
#' direction and rate of fastest increase of the cost function, and is used to
#' update coefficients in the optimization process.
#' @author Cameron Tice
#' @export
#'
#' @examples
#' beta <- c(0.5, -0.25)
#' X <- matrix(c(1, 2, 3, 2, 1, 0), nrow = 3, byrow = TRUE)
#' y <- c(1, 0, 1)
#' gradient_model <- gradient_function(beta, X, y)


gradient_function <- function(beta, X, y) {
  p <- 1 / (1 + exp(-X %*% beta))
  return(t(X) %*% (p - y))
}

#' @title Logistic Regression
#'
#' @description Fits a logistic regression model to the given data. This function
#' uses gradient descent to optimize the logistic regression coefficients and
#' computes several metrics of its performance.
#' @param X A \code{matrix} containing the predictor variables.
#' @param y A \code{vector} representing the observed binary outcomes (0 or 1)
#' for each observation.
#' @param learning_rate The learning rate for the gradient descent algorithm.
#' Default is 0.01.
#' @param max_iterations The maximum number of iterations for the gradient descent
#' algorithm. Default is 10,000.
#' @param tolerance The tolerance for the difference in cost between iterations
#' to determine convergence. Default is 1e-6.
#'
#' @return A \code{list} containing the optimized coefficients, confusion matrix,
#' and performance metrics, including prevalence, accuracy, sensitivity, specificity,
#' false discovery rate, and diagnostic odds ratio.
#'
#' @details Initializes beta using the least-squares formula:
#' \code{(X^T * X)^-1 * X^T * y}. Iteratively updates beta in the direction of
#' negative gradient of the cost function, scaled by the learning rate. Checks
#' for convergence in each iteration. Returns optimized coefficients beta.
#' @author Cameron Tice
#' @export
#'
#' @examples
#' data(mtcars)
#' model <- logistic_regression(X = cbind(mtcars$mpg, mtcars$wt), y = mtcars$am)

logistic_regression <- function(X, y, learning_rate = 0.01, max_iterations = 10000, tolerance = 1e-6) {
  X <- as.matrix(X)
  X <- cbind(1, X)
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

#' @title Plot Logistic Regression Curve
#'
#' @description Plots the logistic regression curve along with the original binary
#' response data points. The function fits a logistic regression model and then
#' uses the estimated coefficients to predict probabilities, which are plotted
#' as a curve.
#'
#' @param X A \code{vector} or a single-column \code{matrix} representing the
#' predictor variables.
#' @param y A \code{vector} representing the binary response variable.
#' @return A plot with the original data points and the fitted logistic regression
#' curve. Points are plotted in red and the logistic curve is plotted in blue.
#' @importFrom graphics plot lines
#' @author Cameron Tice
#' @export
#'
#' @examples
#' data(mtcars)
#' plot_logistic_curve(mtcars$wt, mtcars$am)

plot_logistic_curve <- function(X, y) {
  # Add an intercept to X
  X_matrix <- cbind(1, X)

  # Compute beta estimates using your logistic regression function
  logistic_regression_result <- logistic_regression(X, y)
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

# gpt link for creating cutsom logistic regression and plotting functions:
# https://chat.openai.com/share/2827b3e8-a2c6-4c1d-ab39-a42235cb1deb
# gpt link for additional editing, and the addition of the matrix and other metrics
# to the logistic regression function:
# https://chat.openai.com/share/39e19490-5083-4928-a5ef-46666b1759e4


#' @title Bootstrap Confidence Interval for Logistic Regression Coefficients
#'
#' @description Computes bootstrap confidence intervals for the coefficients
#' estimated by logistic regression. The function uses resampling with replacement
#' to estimate the variability of the coefficient estimates.
#'
#' @param alpha The significance level used to calculate the confidence interval.
#' Typically set to 0.05 for a 95% confidence interval.
#' @param B The number of bootstrap samples to generate. Default is 20.
#' @param X A \code{vector} or \code{matrix} representing the predictor variables.
#' @param y A \code{vector} representing the binary response variable.
#'
#' @return Returns a \code{vector} containing the lower and upper bounds of the
#' confidence interval for the mean of the bootstrapped coefficient estimates.
#' @details This function takes the significance level, alpha, the number of
#' boostraps, beta, predictors, and observed variables (0 or 1) and computes a
#' more accurate confidence interval for the small data set, \code{beta_estimates}.
#' It creates a new larger sample, \code{boot_mean}, using values from the original
#' data set, \code{beta_estimates}. This larger sample is made creating \code{n}
#' sized samples, \code{beta_estimates_star}, and using the mean of each \code{n}
#' sized sample. The \code{n} sized samples are made by selecting numbers randomly
#' from the original data set with replacement. The function then uses the quantile
#' function to create the new confidence interval using the alpha provided by the
#' user.
#' @author Shannon Clark
#' @export
#'
#' @examples
#' data(mtcars)
#' X <- mtcars$mpg
#' y <- mtcars$am
#' Bootstrap_function(alpha = 0.05, B = 20, X = X, y = y)

Bootstrap_function <- function(alpha, B = 20, X, y) {
  logistic_regression_result <- logistic_regression(X, y)
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

#' @title Determine Optimal Cutoff Value for Logistic Regression Predictions
#'
#' @description Evaluates logistic regression predictions at different cutoff values
#' for a specified performance metric. The function provides a visual plot of the
#' performance metric across various cutoff thresholds.
#'
#' @param X A \code{vector} or \code{matrix} representing the predictor variables.
#' @param y A \code{vector} representing the binary response variable.
#' @param metric A \code{string} specifying the performance metric to be evaluated.
#' Possible metrics include "accuracy", "sensitivity", "specificity",
#' "false discovery rate", "diagnostic odds ratio", and "prevalence".
#'
#' @return Generates a plot showing the performance of the specified metric
#' at different cutoff values ranging from 0.1 to 0.9. The plot helps in determining
#' the optimal cutoff value for making predictions.
#' @details Creates a \code{vector} for the metric chosen that has all of the values
#' for that metric for each cut-off value from 0.1 to 0.9 with increments of 0.1.
#' It plots this \code{vector} against the cut-off values.#'
#' @importFrom graphics plot
#' @author Shannon Clark
#' @export
#'
#' @examples
#' data(mtcars)
#' cutoff_value_function(X = mtcars$wt, y = mtcars$am, metric = "sensitivity")

cutoff_value_function <- function(X, y, metric) {

  logistic_regression_result <- logistic_regression(X, y)
  beta_vector <- logistic_regression_result$OptimizedCoefficients

  # Now use beta_vector in your logistic function
  predicted_probs <- logistic_function(beta_vector, cbind(1, X))

  # Initialize vectors for metrics
  metric_values <- rep(NA, 9)
  cutoffs <- seq(0.1, 0.9, by = 0.1)

  for (i in 1:length(cutoffs)) {
    cutoff <- cutoffs[i]
    predictions <- ifelse(predicted_probs > cutoff, 1, 0)

    # Calculate metrics based on predictions
    TP <- sum(predictions == 1 & y == 1)
    TN <- sum(predictions == 0 & y == 0)
    FP <- sum(predictions == 1 & y == 0)
    FN <- sum(predictions == 0 & y == 1)

    if (metric == "accuracy") {
      metric_values[i] <- (TP + TN) / (TP + TN + FP + FN)
    } else if (metric == "sensitivity") {
      metric_values[i] <- TP / (TP + FN)
    } else if (metric == "specificity") {
      metric_values[i] <- TN / (TN + FP)
    } else if (metric == "false discovery rate") {
      metric_values[i] <- FP / (TP + FP)
    } else if (metric == "diagnostic odds ratio") {
      metric_values[i] <- (TP / FP) / (FN / TN)
    } else if (metric == "prevalence") {
      metric_values[i] <- sum(y) / length(y)
    }
  }

  # Plot the specified metric
  plot(cutoffs, metric_values, type = "b", xlab = "Cut-off Values", ylab = metric,
       main = paste(metric, "plot"), pch = 16, col = "blue")
}

#chatgpt link for creating plots: https://chat.openai.com/share/b77ecd7f-58ef-4d44-9551-e0bb87087b39
#chatgpt link for editing of cutoff_value_function and general modifications: https://chat.openai.com/share/9d2dc05f-76d9-40b2-9944-4d8830bd95aa
