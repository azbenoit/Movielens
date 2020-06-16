# Note: This part of the code was not written by me, it was given as a jumping off point

# To load the edx and verification datasets for the first time from scratch
if(!exists("edx") & !exists("verification")){
  ################################
  # Create edx set, validation set
  ################################
  
  # Note: this process could take a couple of minutes
  
  if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
  if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
  
  # MovieLens 10M dataset:
  # https://grouplens.org/datasets/movielens/10m/
  # http://files.grouplens.org/datasets/movielens/ml-10m.zip
  
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  
  ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                   col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                             title = as.character(title),
                                             genres = as.character(genres))
  
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # Validation set will be 10% of MovieLens data
  set.seed(1, sample.kind="Rounding")
  # if using R 3.5 or earlier, use `set.seed(1)` instead
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  # Make sure userId and movieId in validation set are also in edx set
  validation <- temp %>% 
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")
  
  # Add rows removed from validation set back into edx set
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)
  
  rm(dl, ratings, movies, test_index, temp, movielens, removed)
}


# Start of my code: (Alix Benoit)
# ---------------------------GOAL RMSE < 0.86490 ----------------------------------

if(!require(recosystem)) install.packages("recosystem")
options(digits = 7)
# Load libraries
library(tidyverse)
library(caret)

# Splitting edx set into test and training set (note: test set is different than validation set)
i <- createDataPartition(edx$rating, times = 1, p = .1, list = F)
train <- edx[-i,]
test <- edx[i,]
# Making sure all the movies and users in the test set are also in the train set
test <- test %>% semi_join(train, by = "movieId") %>% 
  semi_join(train, by = "userId")

# Writing a function to compute the Root Mean Squared Error of a model (RMSE)
RMSE <- function(actual_rating, predicted_rating){
  sqrt(mean((actual_rating - predicted_rating)^2))
}

#Using a guessing model as a baseline (Y_u,i = mu + epsilon_u,i)
mu <- mean(train$rating)
rmse_guess <- RMSE(test$rating, mu)



#Model using movie effects (b_i), and user effects (b_u) (Y_u,i = mu + b_i + b_u):

# Movie effect: can be found by taking the residual from the avg rating for each movie
# Better than avg movies will have positive residuals, while worse than avg movies will have negative residuals
b_i <- train %>% group_by(movieId) %>% 
  summarise(b_i = mean(rating - mu))

# In order to test the model this effect can then be incorporated into the train and test data frame as such: 
train <- train  %>% left_join(b_i, by = "movieId")
test <- test  %>% left_join(b_i, by = "movieId")
# Movie Id RMSE
RMSE(test$rating, mu + test$b_i)

# User effect can be found similarily
b_u <- train %>% group_by(userId) %>% 
  summarise(b_u = mean(rating - mu - b_i))
train <- train  %>% left_join(b_u, by = "userId")
test <- test  %>% left_join(b_u, by = "userId")
# User Id RMSE
RMSE(test$rating, mu + test$b_u)
# User + Movie RMSE
RMSE(test$rating, mu + test$b_u + test$b_i)


# Regularizing the movie and user effect:

# Finding best lambda
# Define functions to find rmse given specific lambdas:

# Used to find optimal lambda_i
reg_RMSE_b_i <- function(l_i){
  b_i_reg <- train %>% group_by(movieId) %>% 
    summarise(b_i_reg = sum(rating - mu)/(n() + l_i))
  test_r <- test %>% left_join(b_i_reg, by = "movieId")
  data.frame(rmse =  RMSE(test_r$rating, mu + test_r$b_i_reg), l_i = l_i)
}

# Used to find optimal lambda_u, using lambda_i
reg_RMSE <- function(l_i, l_u){
  b_i_reg <- train %>% group_by(movieId) %>% 
    summarise(b_i_reg = sum(rating - mu)/(n() + l_i))
  test_r <- test %>% left_join(b_i_reg, by = "movieId")
  train_r <- train %>% left_join(b_i_reg, by = "movieId")
  b_u_reg <- train_r %>% group_by(userId) %>% 
    summarise(b_u_reg = sum(rating - mu - b_i_reg)/(n() + l_u))
  test_r <- test_r %>% left_join(b_u_reg, by = "userId")
  data.frame(rmse =  RMSE(test_r$rating, mu + test_r$b_i_reg + test_r$b_u_reg), l_i = l_i , l_u = l_u)
}

# Test out lambdas 1-20 for lambda_i
lambdas <- 1:20
reg_rmses_b_i <- map_df(lambdas, function(x) reg_RMSE_b_i(x))
plot(reg_rmses_b_i$l_i, reg_rmses_b_i$rmse) #plot results
reg_rmses_b_i[which.min(reg_rmses_b_i$rmse),] #Best result (l = 2)

# Narrow down l_i 
lambdas <- seq(1,3,.1)
reg_rmses_b_i <- map_df(lambdas, function(x) reg_RMSE_b_i(x))
plot(reg_rmses_b_i$l_i, reg_rmses_b_i$rmse) #plot results
reg_rmses_b_i[which.min(reg_rmses_b_i$rmse),] #Best result (l_i = 2.3)
l_i <- reg_rmses_b_i[which.min(reg_rmses_b_i$rmse),]$l_i

# Test out lambdas 1-20 for lambda_u
lambdas <- 1:20
reg_rmses <-  map_df(lambdas, function(x) reg_RMSE(l_i, x))
plot(reg_rmses$l_u, reg_rmses$rmse) #plot results
reg_rmses[which.min(reg_rmses$rmse),] #Best result (l_u = 5)

# Narrow down l_u
lambdas <- seq(4,6, by = .1)
reg_rmses <-  map_df(lambdas, function(x) reg_RMSE(l_i, x))
plot(reg_rmses$l_u, reg_rmses$rmse) #plot results
reg_rmses[which.min(reg_rmses$rmse),] #Best result (l_u = 4.7) (rmse = 0.8654892)
l_u <- reg_rmses[which.min(reg_rmses$rmse),]$l_u

reg_RMSE(l_i, l_u)

# Adding the reg effects into the datasets
b_i_reg <- train %>% group_by(movieId) %>% 
  summarise(b_i_reg = sum(rating - mu)/(n() + l_i))
test <- test %>% left_join(b_i_reg, by = "movieId")
train <- train %>% left_join(b_i_reg, by = "movieId")
b_u_reg <- train %>% group_by(userId) %>% 
  summarise(b_u_reg = sum(rating - mu - b_i_reg)/(n() + l_u))
train <- train %>% left_join(b_u_reg, by = "userId") %>% 
  mutate(rating_minus_effects = rating - b_i_reg - b_u_reg - mu)
test <- test %>% left_join(b_u_reg, by = "userId")

# Matrix factorization using Stochastic gradient Descent (SGD) through "recosystem" package
library(recosystem)


# setting up training and testing data
train_data <- data_memory(user_index = train$userId, item_index = train$movieId,
                          rating = train$rating_minus_effects, index1 = T)
test_data <- data_memory(user_index = test$userId, item_index = test$movieId, index1 = T)
r <- Reco()
class(r)[1]

# Tuning parameters 1 by 1 to save time
opts <- r$tune(train_data = train_data, 
               opts = list(dim = 20, costp_l1 = 0,
                           costp_l2 = c(.01,.1), costq_l1 = 0, costq_l2 = .2,
                           lrate = .1, verbose = T, nfold = 5))
# Best parameters: 
# dim(seq(5,35,5), seq(32.5,45,2.5)) = 45
# costp_l1 (c(0,.01,.05,.1,.15,.2)) = 0
# costp_l2 (c(0,.01,.05,.1,.15,.2)) = 0.1 (.01?)
# costq_l1 (c(0,.01,.05,.1,.15)) = 0
# costq_l2 (c(0,.01,.05,.1,.15,.2, .25,.3)) = .2
# lrate = c(.01, .05, .1, .15, .2) = .1
r$train(train_data = train_data, opts = c(niter = 40, dim = 45, costp_l1 = 0, costp_l2 = 0.01,
                                          costq_l1 = 0, costq_l2 = .2, lrate = .1))
y_hat_sgd <- r$predict(test_data, out_pred = out_memory()) +mu + test$b_i_reg + test$b_u_reg

RMSE(test$rating, y_hat_sgd) # 0.7904563  


# ------------------- Implementing final model on validation set -----------------


# Model: Y_u,i = mu + b_i + b_u + sum(p_u,m * q_i_m) + epsilon_u,i
# Note: all effects are regularized
# Note: sum(p_u,m * q_i_m) represents the matrix factorization

# Training model on full edx dataset:
mu <- mean(edx$rating)

# Movie and user effects
reg_b_i <- edx %>% group_by(movieId) %>% 
  summarise(reg_b_i = sum(rating - mu)/(n() + l_i))
edx <- edx %>% left_join(reg_b_i, by = "movieId")
reg_b_u <- edx %>% group_by(userId) %>% 
  summarise(reg_b_u = sum(rating - mu - reg_b_i)/(n() + l_u))
edx <- edx %>% left_join(reg_b_u, by = "userId") %>% 
  mutate(rating_minus_effects = rating - reg_b_i - reg_b_u - mu)

# Matrix factorization
edx_data <- data_memory(user_index = edx$userId, item_index = edx$movieId,
                        rating = edx$rating_minus_effects, index1 = T)

r_final <- Reco()
r_final$train(train_data = edx_data, opts = c(niter = 40, dim = 45, costp_l1 = 0, costp_l2 = 0.01,
                                                costq_l1 = 0, costq_l2 = .2, lrate = .1))

# Testing on validation set: (Hasn't been used in any part of the training)

# Adding b_i and b_u effects to columns (note: this does not affect the dataset in 
# any way, it simply adds a column for the corresponding b_i and b_u effects)
validation <- validation %>% left_join(reg_b_i, by = "movieId") %>% 
  left_join(reg_b_u, by = "userId")

# Note: rating is left null, therefore it is not used, only movie_Id and user_Id are used
validation_data <- data_memory(user_index = validation$userId, 
                               item_index = validation$movieId,
                               rating = NULL, index1 = T)

final_predictions <-  mu + validation$reg_b_i + validation$reg_b_u + 
  r_final$predict(validation_data, out_pred = out_memory())

rmse_final <- RMSE(validation$rating, final_predictions)

rmse_final #0.7878149!

(1 - rmse_final/rmse_target) * 100
