#!/usr/bin/env python3

import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Note: Calling context: Project root directory!

sns.set(style="whitegrid")

din = pd.read_csv('data/concrete-query-counts.csv')
df = din.rename(index=str, columns={
    "no": "Repository",
    "select": "SELECT",
    "ask": "ASK",
    "describe": "DESCRIBE",
    "construct": "CONSTRUCT",
    })

x = df[['Repository', 'SELECT', 'ASK', 'DESCRIBE', 'CONSTRUCT']]
y = x.set_index('Repository')
z = y.groupby('Repository').mean()

z.plot.bar(stacked=True,
        color=('#ffffff', '#0b459d', '#757575', '#aaaaaa'),
        edgecolor='#0b459d')

plt.ylabel('Number of Queries')

plt.savefig('querykind.pdf')
