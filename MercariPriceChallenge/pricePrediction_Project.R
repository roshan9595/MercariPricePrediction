rm(list=ls())


library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(Matrix)
library(tidyverse)
library(lightgbm)
library(quanteda)
library(stringr)
library(tictoc)
library(glue)
library(xgboost)
#set current working directory
setwd("/home/roshan/MercariPriceChallenge")


data_cols <- cols(
  train_id = col_integer(),
  name = col_character(),
  item_condition_id = col_integer(),
  category_name = col_character(),
  brand_name = col_character(),
  price = col_double(),
  shipping = col_integer(),
  item_description = col_character()
)


data <- read_tsv("train.tsv", col_types = data_cols)

data <- data %>%
  filter(price != 0)

data$price_log  <- log(data$price + 1)


indexes = sample(1:nrow(data), size=0.3*nrow(data))

# Split data
df_test = data[indexes,]


df_train = data[-indexes,]

## Handling categories

tic("Splitting categories")
temp <- as_tibble(str_split(df_train$category_name, "/", n = 3, simplify = TRUE))
names(temp) <- paste0("category", 1:3)
df_train <- bind_cols(df_train, temp)

temp <- as_tibble(str_split(df_test$category_name, "/", n = 3, simplify = TRUE))
names(temp) <- paste0("category", 1:3)
df_test <- bind_cols(df_test, temp)
toc()

# Cleaning and some new features
tic("Data preprocessing")
df_train <- df_train %>% 
  mutate(item_description = if_else(is.na(item_description) | 
                                      str_to_lower(item_description) == "no description yet", 
                                    "nodescription", 
                                    item_description),
         desc_length = if_else(item_description == "nodescription", 0L, str_length(item_description)),
         na_brand = is.na(brand_name),
         brand_name = if_else(brand_name == "Céline", "Celine", brand_name)
  )

df_test <- df_test %>% 
  mutate(item_description = if_else(is.na(item_description) | 
                                      str_to_lower(item_description) == "no description yet", 
                                    "nodescription", 
                                    item_description),
         desc_length = if_else(item_description == "nodescription", 0L, str_length(item_description)),
         na_brand = is.na(brand_name),
         brand_name = if_else(brand_name == "Céline", "Celine", brand_name)
  )



## Handling missing values

df_train$category1[is.na(df_train$category1)] <- "missing" 
df_train$category2[is.na(df_train$category2)] <- "missing" 
df_train$category3[is.na(df_train$category3)] <- "missing" 

df_test$category1[is.na(df_test$category1)] <- "missing" 
df_test$category2[is.na(df_test$category2)] <- "missing" 
df_test$category3[is.na(df_test$category3)] <- "missing" 


df_train$brand_name[is.na(df_train$brand_name)] <- "missing"
df_test$brand_name[is.na(df_test$brand_name)] <- "missing"


df_train$category_name[is.na(df_train$category_name)] <- "missing"
df_test$category_name[is.na(df_test$category_name)] <- "missing"

toc()

log_trainprices  <- df_train$price_log
df_train$price_log <- NULL
log_testprices  <- df_test$price_log
df_test$price_log <- NULL


all <-  rbind(df_train, df_test)

tic("Descriptions dtm")
descriptions <- corpus(char_tolower(all$item_description))

description_tokens <- tokens(
  tokens_remove(tokens(descriptions,   
                       remove_numbers = FALSE, 
                       remove_punct = TRUE,
                       remove_symbols = TRUE, 
                       remove_separators = TRUE), 
                stopwords("english")), 
  ngrams = 1:2
)


description_dtm <- dfm(
  description_tokens
)
toc()

rmseEval=function(yTrain,yPred) {
  mseEval=sum((yTrain - yPred)^2)/length(yTrain)
  return(sqrt(mseEval)) }

tic("Descriptions tf-idf")
description_dtm_trimmed <- dfm_trim(description_dtm, min_count = 600)
description_tf_matrix <- dfm_tfidf(description_dtm_trimmed)
toc()

description_tf_matrix

tic("Names dtm")
names <- corpus(char_tolower(all$name))

names_tokens <- tokens(
  tokens_remove(tokens(names,    
                       remove_numbers = TRUE, 
                       remove_punct = TRUE,
                       remove_symbols = TRUE, 
                       remove_separators = TRUE), 
                stopwords("english")), 
  ngrams = 1
  
)


names_dtm <- dfm(
  names_tokens
)
toc()

trimmed_names_dfm <- dfm_trim(names_dtm, min_count = 30)

tic("Preparing data for modelling")
sparse_matrix <- sparse.model.matrix(
  ~item_condition_id + 
    shipping + 
    na_brand + 
    category1 + 
    category2 + 
    category3 + 
    desc_length + 
    brand_name,
  data = all)

## Fix for cbind dfm and sparse matrix
class(description_tf_matrix) <- class(sparse_matrix)
class(trimmed_names_dfm) <- class(sparse_matrix)

aaa <- cbind(
  sparse_matrix, # basic features
  description_tf_matrix,  # description
  trimmed_names_dfm # name
)

rownames(aaa) <- NULL

glue("Number of features: {dim(aaa)[2]}")

sparse_train <- aaa[seq_len(nrow(df_train)), ]
sparse_test  <- aaa[seq(from = (nrow(df_train) + 1), to = nrow(aaa)), ]

dtrain <- lgb.Dataset(sparse_train, label=log_trainprices)
toc()
#XGBoost
boost <- xgboost(data = sparse_train, label = log_trainprices, 
                 eta = 0.1,
                 max_depth = 16, 
                 nround=2000, 
                 subsample = 0.7,
                 colsample_bytree = 0.5,
                 seed = 333,
                 eval_metric = "rmse",
                 objective = "reg:linear",
                 nthread = 4)
log_answer <- predict(boost, sparse_test)
print("XGBoost prediction R"+rmseEval(log_testprices,log_answer))
predicted <- exp(log_answer) - 1

results <- data.frame(
  test_id = as.integer(seq_len(nrow(df_test)) - 1),
  price = predicted
)
write_csv(results, "xgboostPrediction.csv")


#LGBM
nrounds <- 8000
param <- list(
  objective = "regression",
  metric = "rmse"
)

set.seed(333)
tic("Modelling")

model <- lgb.train(
  params = param,
  data = dtrain,
  nrounds = nrounds,
  learning_rate = 1,
  subsample = 0.7,
  max_depth = 4,
  eval_freq = 50,
  verbose = -1,
  nthread = 4
)
toc()

tic("Predicting results")
log_predicted <- predict(model, sparse_test)
print("LGBM prediction R"+rmseEval(log_testprices,log_predicted))
predicted <- exp(log_predicted) - 1

results <- data.frame(
  test_id = as.integer(seq_len(nrow(df_test)) - 1),
  price = predicted
)
write_csv(results, "LGBMprediction.csv")

toc()