## Logistic Regression in `training.py`

### `train_test_split` function

This is the function to do the train_test_split for before train data into our Logistic Regression Model. We are not doing a random split here because the data we are having is a time-series model, with a proper cutoff by time, we can avoid data leakage that may caused by random split.


### `create_preprocessor` function

This is the function for training set preprocessing.

1. Imputer / Imputation
2. Scaling - for numerical
3. One-hot Encoding - for categorical