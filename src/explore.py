#!/usr/bin/env python

import pandas as pd

df_train = pd.read_csv('./input/train.csv')
df_test = pd.read_csv('./input/test.csv')

top_users = df_train['user_id'].value_counts().head(20)

mean_deal_prob = df_train.groupby('user_id')['deal_probability'].mean()

cat_columns = [
    'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2',
    'param_3', 'user_type'
]


def encode_categories(df_train, df_test):
    for col in cat_columns:
        train_vals = df_train[col].dropna().unique()
        test_vals = df_test[col].dropna().unique()
        vals = set(train_vals).union(set(test_vals))
        print(f"Number of categories [{col}]:  {len(vals)}")

        df_train[col] = pd.Categorical(df_train[col], categories=vals)
        df_test[col] = pd.Categorical(df_test[col], categories=vals)

    return df_train, df_test


df_train, df_test = encode_categories(df_train, df_test)
df_enc = pd.get_dummies(df_train[cat_columns])
