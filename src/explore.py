#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('./input/train.csv')

sns.distplot(df_train['deal_probability'])

plt.figure()
sns.distplot(df_train.loc[pd.isnull(df_train['price']), 'deal_probability'])

sns.distplot(np.log(df_train['price'].dropna() + 0.001), kde=False)
plt.figure()
sns.distplot(np.log1p(df_train['price'].dropna()), kde=False)
sns.heatmap(data=df_train[['price', 'deal_probability']])
