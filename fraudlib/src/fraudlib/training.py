import pandas as pd
import fraudlib.features as features
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

TARGET_LABEL = 'is_fraud'

NUMERIC_FEATURES = ['amount', 'amount_mean_7d', 'amount_std_7d', 'amount_max_7d', 'amount_min_7d', 'amount_count_7d']
CATEGORICAL_FEATURES = ['channel']


def train_test_split(df):
    """
    Split the data into train and test based on time.
    """
    cutoff = df['ts'].min() + pd.Timedelta(days=60)

    ## Day 1 - 60
    train_df = df[df['ts'] <= cutoff]

    ## Day 61 - 90
    test_df = df[df['ts'] > cutoff]

    ## Random Split might cheat the model because we are forcasting time-series data, therefore, we should split the data based on time to avoid data leakage.

    return train_df, test_df


def create_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first'), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def train_model(train_df, test_df):

    """This is a logistic regression model for fraud detection."""
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)

    pipeline = Pipeline(steps=[
        ('preprocessor', create_preprocessor()),
        ('classifier', lr)
    ])

    pipeline.fit(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df[TARGET_LABEL])
    return pipeline

def evaluate_model(pipeline, test_df):
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET_LABEL]

    y_pred = pipeline.predict(X_test)

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))


