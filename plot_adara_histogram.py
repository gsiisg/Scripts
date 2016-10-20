# Geoffrey So 9/30/2016

import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt

# data mining test 1

df1=pd.read_csv('data_mining_test_1.csv')

print('any data null ', df1.isnull().values.any())

# plot histogram

counts = Counter(df1.columns.values)
plot_df = pd.DataFrame.from_dict(counts, orient='index')
plot_df.plot(kind='bar')

for column_name in df1.columns.values:
    plt.figure();
    df1[column_name][df1['Click']>0].plot.hist(title=column_name)

campaign_counts = Counter(df1['A'][df1['Click']>0].values)
df = pd.DataFrame.from_dict(campaign_counts, orient='index')
df.plot(kind='bar')

    
# data mining test 2

df2=pd.read_csv('data_mining_test_2.csv')

print('any data null ', df2.isnull().values.any())

# plot histogram

counts = Counter(df2.columns.values)
plot_df = pd.DataFrame.from_dict(counts, orient='index')
plot_df.plot(kind='bar')

for column_name in df2.columns.values[1:]:
    plt.figure();
    df2[column_name].plot.hist(title=column_name)
