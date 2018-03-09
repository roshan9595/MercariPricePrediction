rm(list=ls())

library(Matrix)
library(tidyverse)
library(lightgbm)
library(quanteda)
library(stringr)
library(tictoc)
library(glue)

library(data.table) # Loading data
library(ggplot2) # Data visualization
library(treemapify) # Treemap visualization
library(gridExtra) # Create multiplot
library(dplyr) # data manipulation
library(tidyr) # data manipulation
library(tibble) # data wrangling
library(stringr) # String processing
library(repr)
library(stringi) # String processing

#set current working directory
setwd("/home/roshan/MercariPriceChallenge")


train_cols <- cols(
  train_id = col_integer(),
  name = col_character(),
  item_condition_id = col_integer(),
  category_name = col_character(),
  brand_name = col_character(),
  price = col_double(),
  shipping = col_integer(),
  item_description = col_character()
)

test_cols <- cols(
  test_id = col_integer(),
  name = col_character(),
  item_condition_id = col_integer(),
  category_name = col_character(),
  brand_name = col_character(),
  shipping = col_integer(),
  item_description = col_character()
)

train <- read_tsv("train.tsv", col_types = train_cols)

train = train %>% mutate(log_price = log(price+1))

head(train, 3)

summary(train)

options(repr.plot.width=7, repr.plot.height=7)

p1 = train %>% ggplot(aes(x=log_price)) +
  geom_histogram(bins=30) +
  ggtitle('Distributon of Log1p Price')

p2 = train %>% ggplot(aes(x=price)) +
  geom_histogram(bins=30) +
  xlim(0,300) +
  ggtitle('Distributon of Price')

p3 = train %>% ggplot(aes(x=item_condition_id)) +
  geom_bar() +
  ggtitle('Distribution of Item Conditions') +
  theme(legend.position="none")

p4 = train %>% ggplot(aes(x=shipping)) +
  geom_bar(width=0.5) +
  ggtitle('Distribution of Shipping Info') +
  theme(legend.position="none")

suppressWarnings(grid.arrange(p1, p2, p3, p4, ncol=2))

train = data.frame(train, str_split_fixed(train$category_name, '/', 4)) %>%
  mutate(cat1=X1, cat2=X2, cat3=X3, cat4=X4) %>% select(-X1, -X2, -X3, -X4)

train %>% summarise(Num_Cat1 = length(unique(cat1)), Num_Cat2 = length(unique(cat2)),
                    Num_Cat3 = length(unique(cat3)), Num_Cat4 = length(unique(cat4)))

options(repr.plot.width=7, repr.plot.height=7)

train %>%
  group_by(cat1, cat2) %>%
  count() %>%
  ungroup() %>%
  ggplot(aes(area=n, fill=cat1, label=cat2, subgroup=cat1)) +
  geom_treemap() +
  geom_treemap_subgroup_text(grow = T, alpha = 0.5, colour =
                               "black", fontface = "italic", min.size = 0) +
  geom_treemap_text(colour = "white", place = "topleft", reflow = T) +
  theme(legend.position = "null") +
  ggtitle("1st and 2nd Hierarchical Category Levels")

options(repr.plot.width=7, repr.plot.height=7)

train %>% filter(cat1=='Women') %>% 
  group_by(cat2, cat3) %>%
  count() %>%
  ungroup() %>%
  ggplot(aes(area=n, fill=cat2, label=cat3, subgroup=cat2)) +
  geom_treemap() +
  geom_treemap_subgroup_text(grow = T, alpha = 0.5, colour =
                               "black", fontface = "italic", min.size = 0) +
  geom_treemap_text(colour = "white", place = "topleft", reflow = T) +
  theme(legend.position = "null") +
  ggtitle("2nd and 3rd Hierarchical Category Levels Under Woman")

options(repr.plot.width=7, repr.plot.height=3.5)

train = train %>% mutate(has_brand=(brand_name!=''))
train %>%
  ggplot(aes(x=cat1, fill=has_brand)) +
  geom_bar(position='fill') +
  theme(axis.text.x=element_text(angle=15, hjust=1, size=8)) +
  xlab('1st Categories') +
  ylab('Proportion') +
  ggtitle('Items With and Without Brands')


options(repr.plot.width=7, repr.plot.height=3.5)

p1 = train %>% mutate(len_of_des = str_length(item_description)) %>%
  ggplot(aes(x=len_of_des)) +
  geom_histogram(bins=50) +
  ggtitle('Distribution of Length of Descriptions') +
  xlab('Length of Item Description') +
  theme(plot.title = element_text(size=10))

p2 = train %>% mutate(num_token_des = str_count(item_description, '\\S+')) %>% 
  ggplot(aes(x=num_token_des)) +
  geom_histogram(bins=50) +
  ggtitle('Distribution of # of Tokens of Descriptions') +
  xlab('Number of Tokens') +
  theme(plot.title = element_text(size=10))

grid.arrange(p1, p2, ncol=2)

options(repr.plot.width=7, repr.plot.height=3.5)

train = train %>% mutate(num_token_name = str_count(name, '\\S+'))
train %>%
  ggplot(aes(x=num_token_name)) +
  geom_bar(width=0.7) +
  ggtitle('Distribution of # of Tokens of Names') +
  xlab('Number of Words')



options(repr.plot.width=7, repr.plot.height=3.5)

p1 = train %>%
  ggplot(aes(x=item_condition_id, y=log_price, fill=item_condition_id)) +
  geom_boxplot(outlier.size=0.1) +
  ggtitle('Boxplot of Log Price versus Condition') +
  theme(legend.position="none", plot.title = element_text(size=10))

p2 = train %>%
  ggplot(aes(x=shipping, y=log_price, fill=shipping)) +
  geom_boxplot(width=0.5, outlier.size=0.1) +
  ggtitle('Boxplot of Log Price versus Shipping') +
  theme(legend.position="none", plot.title = element_text(size=10))

grid.arrange(p1, p2, ncol=2)

options(repr.plot.width=7, repr.plot.height=3.5)

train %>%
  ggplot(aes(x=cat1, y=log_price, fill=has_brand)) +
  geom_boxplot(outlier.size=0.1) +
  ggtitle('Boxplot of Log Price versus 1st Category') +
  xlab('1st Category') +
  theme(axis.text.x=element_text(angle=15, hjust=1))

options(repr.plot.width=7, repr.plot.height=7)

train %>% mutate(len_of_des = str_length(item_description)) %>%
  group_by(len_of_des) %>%
  summarise(mean_log_price = mean(log_price)) %>% 
  ggplot(aes(x=len_of_des, y=mean_log_price)) +
  geom_point(size=0.5) +
  geom_smooth(method = "loess", color = "red", size=0.5) +
  ggtitle('Mean Log Price versus Length of Description')

options(repr.plot.width=7, repr.plot.height=7)

train %>% mutate(num_token_des = str_count(item_description, '\\S+')) %>%
  group_by(num_token_des) %>%
  summarise(mean_log_price = mean(log_price)) %>% 
  ggplot(aes(x=num_token_des, y=mean_log_price)) +
  geom_point(size=0.5) +
  geom_smooth(method = "loess", color = "red", size=0.5) +
  ggtitle('Mean Log Price versus # of Tokens of Description')

ggplot(data = train, aes(x = as.factor(item_condition_id), y = log(price + 1))) + 
  geom_boxplot(fill = 'cyan2', color = 'darkgrey')

train %>%
  ggplot(aes(x = log(price+1), fill = factor(shipping))) + 
  geom_density(adjust = 2, alpha = 0.6) + 
  labs(x = 'Log price', y = '', title = 'Distribution of price by shipping')

dcorpus <- corpus(train$item_description)
dfm1 <- dfm(
  dcorpus, 
  ngrams = 1, 
  remove = c("rm", stopwords("english")),
  remove_punct = TRUE,
  remove_numbers = TRUE,
  stem = TRUE)

set.seed(100)
textplot_wordcloud(dfm1, min.freq = 3e4, random.order = FALSE,
                   rot.per = .25, 
                   colors = RColorBrewer::brewer.pal(8,"Dark2"))


