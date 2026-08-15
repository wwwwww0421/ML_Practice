import fraudlib.features as features
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np


def train_test_split(df):
    """
    Split the data into train and test based on time.
    """
    ## Day 1 - 60
    train_df = df[df['ts'] <= np.percentile(df.ts, 90/100*60)]

    ## Day 61 - 90
    test_df = df[df['ts'] > np.percentile(df.ts, 90/100*60)]

    ## Random Split might cheat the model because we are forcasting time-series data, therefore, we should split the data based on time to avoid data leakage.

    return train_df, test_df


def create_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['amount']),
            ('cat', OneHotEncoder(drop='first'), ['is_fraud'])
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

    pipeline.fit(train_df[['amount', 'is_fraud']], train_df['is_fraud'])
    return pipeline

def evaluate_model(pipeline, test_df):
    X_test = test_df[['amount', 'is_fraud']]
    y_test = test_df['is_fraud']

    y_pred = pipeline.predict(X_test)

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))


