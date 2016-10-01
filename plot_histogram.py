# Geoffrey So 9/30/2016

import numpy as np
import pandas as pd
from collections import Counter

# read data
'''
        Name            Data Type       Meas.   Description
        ----            ---------       -----   -----------
        Sex             nominal                 M, F, and I (infant)
        Length          continuous      mm      Longest shell measurement
        Diameter        continuous      mm      perpendicular to length
        Height          continuous      mm      with meat in shell
        Whole weight    continuous      grams   whole abalone
        Shucked weight  continuous      grams   weight of meat
        Viscera weight  continuous      grams   gut weight (after bleeding)
        Shell weight    continuous      grams   after being dried
        Rings           integer                 +1.5 gives the age in years
'''
names = ['sex','length','diameter','height','whole_weight','shucked_weight',
         'viscera_weight','shell_weight','rings']
df=pd.read_csv('abalone.data',header=None,names=names)

print('any data null ', df.isnull().values.any())

# plot histogram

# sex
counts = Counter(df[column_name].values)
plot_df = pd.DataFrame.from_dict(counts, orient='index')
plot_df.plot(kind='bar')

for column_name in names[1:]:
    plt.figure();
    df[column_name].plot.hist(title=column_name)

