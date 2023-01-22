
# --------------------------------------------------------------
#                         Introduction
# --------------------------------------------------------------

# A recommendation system is a tool that suggests content to users based on their
# past preferences and ratings (Irizarry 2019).
# These systems are commonly used by companies to successfully 
# promote themselves on a market: the more useful a platform is,
# the more time a user spends on the platform, enjoys using it and shares it with
# friends, the better it is for the business. Therefore, elaborating an efficient model
# that could efficiently predict a user's choice is an important and useful skill and task to perform.
# 
# In this project, I will elaborate user's movies ratings prediction model:
# a model that would let us know, what would a user likely to rate a movie before
# this user has ever watched it.
# 
# The accuracy of this model will be evaluated by RMSE.


# RMSE is a loss function, which is a technique of establishing 
# the accuracy of a model. 

# 
# The whole analysis will consist of several parts:

#   1. The inital HarvardX code
#   2. My code for exploring the data
#   3. Building the model with train set
#   4. Finding the optimal lambda value and building the regularization regression 
#   5. Applying the regularization to the final_holdout_test and finding RMSE
  


#-------------------------------------------------------------
#                        HarvardX code
#-------------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]


# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)




# --------------------------------------------------------------
#                          My analysis
# --------------------------------------------------------------


  
# loading the libraries
library(Hmisc) # data analysis 
library(stats)
library(ggplot2)
options(scipen=999) # disabling scintific notation
options(warn=-1) # suppressing warnings


# STEP 1: Data preparation

# splitting data into train and test

set.seed(1)
index <- createDataPartition(edx$rating, p=0.8, times = 1, list = FALSE)
train_set <- edx %>% slice(index)
test_set <- edx %>% slice(-index)


# STEP 2: Exploring the data


head(train_set)

summary(train_set)

str(train_set)

dim(train_set)


describe(train_set)

# This function shows apart from the data we have already seen with summary(),
# information on missing values. In this case they are equal to 0.
# I can additionally check to make sure the dataset has no missing values by
# running this code:

any(is.na(train_set))

 
# To further explore the data, I can create some visualizations: 

train_set %>% ggplot() + geom_histogram(aes(rating), binwidth = 0.45, fill = '#0F3D3E', color = '#F1F1F1') +
  ggtitle('Most common movie ratings') +
  theme(plot.background = element_rect(fill = '#E2DCC8'),
        panel.background = element_rect(fill = '#E2DCC8'),
        plot.title = element_text(size = 18, hjust = 0.5))

# what this plot tells us is that 3,4 and 5 are the most common ratings, 
# with 4 being the most common.


# checking all the unique rating values
sort(unique(train_set$rating))


# checking the number of unique
sum(unique(train_set$userId))
sum(unique(train_set$movieId))


train_set  %>% group_by(userId) %>% summarise (n = n()) %>% ggplot() +
  geom_col(aes(userId, n), fill = "#A10035") + ylab('Number of ratings') +
  xlab("User") + ggtitle("Number of ratings per user") +
  theme(plot.title = element_text(size = 20, hjust = 0.5),
        plot.background = element_rect(fill = "#E2DCC8"),
        panel.background = element_rect(fill = "#E2DCC8"))


# converting timestamp into yy-m-d format:
train_set$timestamp <- as.POSIXct(train_set$timestamp, origin = '1970-01-01')

train_set$month <- sapply(train_set$timestamp, function(x){format(x, '%m')})


head(train_set)

# Now I can use this information to see in which month on average users tend to rate movies more or less often.
train_set %>% group_by(month) %>% summarise (n = n()) %>%
  ggplot() + geom_col(aes(month, n), fill = '#74959A',color = "#354259") +
  ggtitle("Number of ratings by month") + xlab('Month') + ylab('Number of ratings') +
  theme(plot.title = element_text(size = 20, hjust = 0.5),
        plot.background = element_rect(fill = "#E2DCC8"),
        panel.background = element_rect(fill = "#E2DCC8"))

# What this graph shows is that November happens to be the month when there's on average the biggest amount of ratings.
# If I don't need this column later I can also remove it like this:

train_set <- subset(train_set, select = - c(month))


# STEP 3: building the model

# Structure:

# Model 1 (Naive rmse)
# Model 2 (introducing bias)
# Model 3 (adding the user bias)
# Regularization
# Result and conclusion 


mu <- mean(train_set$rating)
mu

naive_rmse <- sqrt(mean((train_set$rating - mu)^2))
naive_rmse


# Building the second model

mov_avg <- train_set%>% group_by(movieId) %>% summarise(bias = mean(rating - mu))

mov_avg %>% ggplot() + geom_histogram(aes(bias), bins = 15, binwidth = .4, color = "#FEC260") +
  theme(plot.background = element_rect(fill = "#E2DCC8"),
        panel.background = element_rect(fill = "#E2DCC8"))


# Now using the mean I'm building  a predictor
r_hat <- mu + train_set %>%
  left_join(mov_avg, by = 'movieId') %>% pull(bias)

# And I can build the second model, using the bias:

biased_rmse <- sqrt(mean((r_hat - train_set$rating)^2))

table_rmse <- tibble(naive_rmse, biased_rmse)
table_rmse
  
# The model has improved a bit, but not significantly.
# In the third model I will add the user bias.

user_b <- edx %>%
  left_join(mov_avg, by = 'movieId') %>%
  group_by(userId) %>% summarise(user_bias = mean(rating - mu - bias))


nrow(user_b)

pred_ratings <- train_set %>%
  left_join(mov_avg, by = 'movieId') %>%
  left_join(user_b, by = 'userId') %>%
  mutate(prediction = mu + bias + user_bias) %>%
  pull(prediction)



user_b_rmse <- sqrt(mean((pred_ratings - train_set$rating)^2))


table_rmse <- tibble(naive_rmse, biased_rmse, user_b_rmse)
table_rmse


# Regularization



# first I'll create a vector of possible lambda values
# then it will be fed to the model to see which lambda value
# brings the RMSE for each of the model I created down.

lambdas <- seq(0, 8, 0.2)


 m4 <- train_set %>% 
  group_by(movieId) %>% 
  summarise(s = sum(rating - mu), quant = n())

rmse <- sapply(lambdas, function(l){
  
    pred_ratings <- train_set %>% 
    left_join(m4, by='movieId') %>% 
    mutate(bias = s / (quant + l)) %>%
    mutate(pred = mu + bias) %>%
    pull(pred)
    
   return(sqrt(mean((pred_ratings - train_set$rating)^2)))
})

qplot(lambdas, rmse) 



# creating a function that will add each value of lambda from the vector
# to an existing model.

rmses <- sapply(lambdas, function(l) {
  
  mu <- mean(train_set$rating)
  
  bias <- train_set %>%
    group_by(movieId) %>% summarise(b = sum(rating - mu) / (n() + l))
  
  user_bias <- train_set %>%
    left_join(bias, by = "movieId") %>%
    group_by(userId) %>% summarise(ub = sum(rating - b - mu) / (n() + l))
  
  pred_ratings <- train_set %>%
    left_join(bias, by = "movieId") %>%
    left_join(user_bias, by = "userId") %>%
    mutate(pred = mu + b + ub) %>%
    pull(pred)
  
  return(sqrt(mean((pred_ratings - train_set$rating)^2)))
})


# plotting the result 
qplot(lambdas,rmses)


# I can find the value of the optimal lambda like this:

lambda <- lambdas[which.min(rmses)] 
lambda


# applying lamba value to the train_set and defining RMSE

bias <- train_set %>% 
  group_by(movieId) %>% 
  summarise(b = sum(rating - mu)/(n()+lambda), n = n()) 



user_bias <- train_set %>% 
  left_join(bias, by="movieId") %>%
  group_by(userId) %>%
  summarise(ub = sum(rating - b - mu)/(n()+lambda), n = n()) 



# applying the result to the test set
train_pred_ratings <- train_set %>% 
  left_join(bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(prediction = mu + b + ub) %>%
  pull(prediction)

final_train_rmse <- sqrt(mean((train_pred_ratings - train_set$rating)^2))
final_train_rmse



# applying my lambda value to the test_set
bias <- test_set %>% 
  group_by(movieId) %>% 
  summarise(b = sum(rating - mu)/(n()+lambda), n = n()) 



user_bias <- test_set %>% 
  left_join(bias, by="movieId") %>%
  group_by(userId) %>%
  summarise(ub = sum(rating - b - mu)/(n()+lambda), n = n()) 



# applying the result to the test set
test_pred_ratings <- test_set %>% 
  left_join(bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(prediction = mu + b + ub) %>%
  pull(prediction)


# calculating the RMSE
reg_rmse <- sqrt(mean((test_pred_ratings - test_set$rating)^2))
reg_rmse


# testing the algorithm on the final_holdout_test
bias <- final_holdout_test %>% 
  group_by(movieId) %>% 
  summarise(b = sum(rating - mu)/(n()+lambda), n = n()) 


user_bias <- final_holdout_test %>% 
  left_join(bias, by="movieId") %>%
  group_by(userId) %>%
  summarise(ub = sum(rating - b - mu)/(n()+lambda), n = n()) 



final_ratings <- final_holdout_test %>% 
  left_join(bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(prediction = mu + b + ub) %>%
  pull(prediction)

RMSE <- sqrt(mean((final_ratings - final_holdout_test$rating)^2))
RMSE


# Conclusion
# 
# 
# In this analysis I explored the movielens dataset and built a 
# regularization regression model that included the optimal
# lambda value. The final task of the analysis, which is finding the
# RMSE value of the final_holdout_test, was fulfilled, with RMSE = 0.825615.
# On my preparation stage for this analysis, I was able to read through lots
# of interesting references to the topic of recommendations systems. Some helped me
# build the current model, others showed that there are even more efficient ways
# of doing it with even better results, such as NI and KNI. Which is an inspiring
# way to make next steps to further improve my knowledge of machine learning.


# References

#Irizarry, R A 2019, 'Introduction to Data Science', CRC Press, Boca Raton.

#Qu, Y, Bai, T, Zhang, W, Nie, J & Tang, J 2019, <i>An End-to-End Neighborhood-based Interaction Model for Knowledge-enhanced Recommendation</i>, viewed 21 January 2023, <https://arxiv.org/pdf/1908.04032v2.pdf>.






