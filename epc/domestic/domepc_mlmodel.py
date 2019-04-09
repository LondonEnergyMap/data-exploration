import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# filepath = 'https://media.githubusercontent.com/media/LondonEnergyMap/cleandata/master/epc/domestic/epcshort_postcodefull.csv'
filepath = 'epcshort_postcodefull.csv'
df_all = pd.read_csv(filepath)

# select only entries with less than 10 rooms
n = 10
df = df_all[(df_all.nrooms <= n)]

# restrict upper and lower limit for floor area
tfa_upper = 50*n
tfa_lower = 20
df = df[(df.tfa <= tfa_upper) & (df.tfa >= tfa_lower)]

# create new column of building age based on wall description and transcation type
df['wall_firstword'] = df.wall.str.split().str.get(0)
wall_mapping = {'Cavity': 2, 'System': 3, 'Timber': 3}
df['age'] = df.wall_firstword.map(wall_mapping)
df.age.fillna(1, inplace=True)
df.loc[df.transact_type == 'new dwelling', 'age'] = 4

# create new column for number of exposed sides based on property type and form
prop_mapping = {'House': 0, 'Flat': -2, 'Bungalow': 0.5, 'Maisonette': -2,
                'Park home': 0}
built_mapping = {'Detached': 0, 'Mid-Terrace': -2, 'Semi-Detached': -1,
                 'Enclosed Mid-Terrace': -2.5, 'Enclosed End-Terrace': -1.5,
                 '': 0}


df['propmap'] = df.prop_type.map(prop_mapping)
df['builtmap'] = df.builtform.map(built_mapping)
df['exposedsides'] = 6 + df.propmap + df.builtmap

# select entries only with certain number of meters, and select mains gas only
m = 6
dfml = df[(df.gasmeters <= m) & (df.elecmeters <= m) & (df.mainsgas == 'Y')]

