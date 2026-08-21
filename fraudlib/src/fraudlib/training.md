## Logistic Regression in `training.py`

### `train_test_split` function

This is the function to do the train_test_split for before train data into our Logistic Regression Model. We are not doing a random split here because the data we are having is a time-series model, with a proper cutoff by time, we can avoid data leakage that may caused by random split.


### `create_preprocessor` function

This is the function for training set preprocessing.

1. Imputer / Imputation
2. Scaling - for numerical
3. One-hot Encoding - for categorical

In our script, we have 2 types of columns - numericals and categoricals. If a numerical column has a missing value, we will fill the *NaN* with **median**. If a categorical column is having a missing value, we will fill the *NaN* with **the most frequent category**.

After the imputation, we then put numerical columns into `StandardScaler()`, it will rescale numerical variables so that we can get approximately *mean = 0* and *std = 1*.

$\text{scaled value} = (value - mean) / std$

Without doing the scaling columns like amount can have very different influence from columns like age. As amount can up-to a very high value £1,000,000, while age value usually within 100.


For categorical columns, we put them into `OneHotEncoder()`, it will convert the values into binary columns and each column will get its own column. The reason why we can't just use the numbers like 1,2,3 to assign each column rather than 0 and 1 is because the model might interpret 3 > 2 > 1 as if there's some numerical relationships between the columns but there's not.

In summary, `StandardScaler()` put numerical variables on a comparable scale, while `OneHotEncoder` convert categories into numerical columns without inventing an ordering.


### `train_model` function

This is a function where our Logistic Regression model been trained. In the function, we have assigned Logistic Regression model to detect fraud within our transaction dataset, which is the training set output from `train_test_split` function we've created above. We have assigned *class_weight='balanced'* here as we are doing the fraud detection and this will tell our model "don't treat every training example equally and give more importance to minority class.". The *max_iter=1000* says the model can make up to 1,000 optimisation iterations, in which this is to give the model enough opportunity to converge within 1,000 times.

`pipeline` is where the raw data feed into the `proprocessor()` then our Logistic Regression model then this model is trained for **fraud detection**. Pipeline is important for preventing data leakage with the exact order you would like the process to be before putting into the model.

We are here doing the `.fit()` with selected variables to train our models. The model will learn things/rules like the *median values, means, std, categories* in our training set (exclude the answer column). Then `.transform()` (the apply what the model learned part), which is hiding in the `Pipeline()` step.

**Architecture Breakdown**
```
X_train
   │
   ↓
preprocessor.fit()
   │
   │ learns:
   │ - median
   │ - mean
   │ - standard deviation
   │ - categories
   │
   ↓
preprocessor.transform()
   │
   ↓
transformed X_train
   │
   ↓
LogisticRegression.fit()
   │
   ↓
trained model
```


### `evaluate_model` function
```
X_test
   │
   ↓
preprocessor.transform(X_test)
   │
   ↓
transformed X_test
   │
   ↓
LogisticRegression.predict()
   │
   ↓
y_pred
```


### Summary

```
                    TRAINING
                       │
                       ↓
                    X_train
                       │
                       ↓
            preprocessor.fit(X_train)
                       │
              learns parameters
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
     median          mean/std      categories
        │              │              │
        └──────────────┼──────────────┘
                       ↓
          preprocessor.transform(X_train)
                       │
                       ↓
              transformed X_train
                       │
                       ↓
            LogisticRegression.fit()
                       ↑
                       │
                    y_train
                       │
                       ↓
                 TRAINED MODEL
                       ↓
                    TESTING
                       │
                       ↓
                     X_test
                       │
                       ↓
                ┌───────────────┐
                │   PIPELINE    │
                │               │
                │   transform   │
                │      ↓        │
                │    predict    │
                └──────┬────────┘
                       │
                       ↓
                   prediction
```