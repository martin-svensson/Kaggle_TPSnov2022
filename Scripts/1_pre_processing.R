# ====================================================================================================== #
# Description
#
#   Pre processing data  
#
# Change log:
#   Ver   Date        Comment
#   1.0   29/11/22    Initial version
#
# ====================================================================================================== #
# ------------------------------------------------------------------------------------------------------ #
# LIBRARIES
# ------------------------------------------------------------------------------------------------------ #

library(data.table)
library(tidyverse)
library(magrittr)

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

df_train_labels <- 
  fread("./Data/train_labels.csv")
  
df_train <- 
  list.files("./Data/submission_files") %>% 
  map(
    ~ fread(paste0("./Data/submission_files/", .x))
  ) %>% 
  set_names(
    gsub("\\.csv", "", list.files("./Data/submission_files"))
  )

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

# Store data in a matrix: data is all of the same type and a matrix is much more efficient
mat_train <- 
  matrix(
    nrow = nrow(df_train[[1]]) + 1, # first row is log loss
    ncol = length(df_train) + 1 # first columns is id
  )
  
# Last row is log loss
mat_train[ , 1] <- 
  c(df_train[[1]]$id, nrow(df_train[[1]])) 

for (i in 1:length(df_train)) {
  
  mat_train[ , i + 1] <- 
    c(df_train[[i]]$pred, 
      as.double(names(df_train[i])))
  
}



# -- Resampling index (tidymodels needs a data.frame, so we'll do it manually)

set.seed(486)

index <- 
  list(
    "test" = 
      sample(
        1:nrow(df_train_labels), 
        size = nrow(df_train_labels) * 0.2
      )
  )

# No CV this time due to time restrictions
index[["validation"]] <- 
  sample(
    setdiff(1:nrow(df_train_labels), index$test), 
    size = nrow(df_train_labels) * 0.2
  )

index[["training"]] <- 
  setdiff(
    1:nrow(df_train_labels), 
    c(index$test, index$validation)
  )

# ==== EXPORT ------------------------------------------------------------------------------------------ 

save(
  mat_train,
  file = "./Output/1_mat_train.RData"
)

save(
  index,
  file = "./Output/1_index.RData"
)