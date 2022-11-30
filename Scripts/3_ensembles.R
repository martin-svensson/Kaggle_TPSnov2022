# ====================================================================================================== #
# Description
#
#   Producing and testing a bunch of ensemble models
#
# Change log:
#   Ver   Date        Comment
#   1.0   30/11/22    Initial version
#
# ====================================================================================================== #
# ------------------------------------------------------------------------------------------------------ #
# LIBRARIES
# ------------------------------------------------------------------------------------------------------ #

library(data.table)
library(tidyverse)
library(magrittr)
library(tidymodels)

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

load("./Output/1_mat_train.RData")
load("./Output/1_index.RData")

df_train_labels <- 
  fread("./Data/train_labels.csv")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

log_loss <- function(y_true, y_pred) {
  
  -1 * mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
  
}

ll_row <- nrow(mat_train)
id_col <- 1

# -- Predictions to exclude

# predictions with 0 or 1
bad <- 
  map_int(
    2:ncol(mat_train),
    ~ ((min(mat_train[index$training , .x]) < 1/10000) | (max(mat_train[index$training , .x]) > 1 - 1/10000))
  )

idx_bad <- 
  which(bad == 1) + 1

# log loss for training data (see EDA) - exclude predictions high log loss
ll_training <- 
  map_dbl(
    2:ncol(mat_train),
    ~ log_loss(
        df_train_labels[index$training, label],
        mat_train[index$training, .x]
      )
  )

excl_threshold <- 
  ll_training %>% 
  quantile(0.05, na.rm = TRUE)

idx_incl <- 
  which(ll_training < excl_threshold) + 1



# ----  Models ------------------------------------------------------------------

df_train <- 
  mat_train[
    c(index$training, index$validation, index$test), 
    setdiff(idx_incl, idx_bad)
  ] %>% as.data.table()

df_train$label <-
  df_train_labels[ , label] %>% 
  as.factor()

# -- Recipes

recipe_raw <- 
  recipe(
    label ~ .,
    df_train
  ) 

# -- Models

mod_xgb <- 
  boost_tree(
    mode = "classification"
  ) %>% 
  set_engine("xgboost")

# -- Workflows

wflow_xgb <- 
  workflow() %>% 
  add_model(mod_xgb) %>% 
  add_recipe(recipe_raw) 
  


# ---- Validation --------------------------------------------------------------

# -- xgb

fit_xgb <- 
  fit(
    wflow_xgb, 
    df_train[index$training, ]
  )

val_xgb <- 
  predict(
    fit_xgb,
    df_train[index$validation, ],
    type = "prob"
  )

log_loss(
  df_train[index$validation, label] %>% 
    as.character() %>% 
    as.integer(),
  val_xgb$.pred_1
)
# thats so bad


# simple mean

log_loss(
  df_train[index$validation, label] %>% 
    as.character() %>% 
    as.integer(),
  df_train[index$validation, -"label"] %>% 
    rowMeans()
)
#thats even worse .. 



# ---- Submissions -------------------------------------------------------------

df_test <- 
  mat_train[
    -c(index$training, index$validation, index$test, ll_row), 
    setdiff(idx_incl, idx_bad)
  ] %>% as.data.table()

fit_xgb_final <- 
  fit(
    wflow_xgb, 
    df_train
  )

submission <- 
  data.table(
    "id" = mat_train[-c(index$training, index$validation, index$test, ll_row), 1]
  )
  
submission$pred <-  
  predict(
    fit_xgb_final,
    df_test,
    type = "prob"
  ) %>% 
  pull(.pred_1)


# ---- EXPORT ------------------------------------------------------------------

subm_time <- 
  format(
    Sys.time(),
    "%d-%b-%Y_%H-%M"
  )

submission %>% 
  fwrite(
    file = paste0("./Output/submissions/subm_", subm_time, ".csv")
  )
