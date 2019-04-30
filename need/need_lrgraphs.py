import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# filepath to CLEANED need data
filepath = 'https://media.githubusercontent.com/media/LondonEnergyMap/cleandata/master/need/need_ldn.csv'

# read csv into dataframe
df_all = pd.read_csv(filepath)

# select data in year 2012
y = 2012

df = df_all[df_all.year == 2012]

# ---------------------
# make regression graphs for main parameters

cols = ['imd_eng', 'age', 'proptype', 'floorarea_band', 'epc_band', 'gcons', 'econs']

xaxisnames = ['IMD', 'age', 'property type', 'floor area band', 'EPC band']

# temporary dataframe for plotting graphs
t = df[cols]
temp = t.copy()

# shift age and proptype to range 1-6 rather than 101-106
temp['age'] = temp['age'] - 100
temp['proptype'] = temp['proptype'] - 100

# maximum consumption limits (from cleaning data)
maxgcons = 50000
maxecons = 25000

cols_len = len(temp.columns)

# omit the last 2 columns
k = 2

# number of subplots to include according to columns length
nplot = cols_len - k

# ---------------------
# create figure with a number of subplots for gas
f, axes = plt.subplots(1, nplot, figsize=(3*nplot, 3))

ymin = -5000

for i in range(nplot):
    xmax = len(temp[cols[i]].unique()) + 1
    xmin = temp[cols[i]].min() - 1

    sns.regplot(data=temp, x=cols[i], y='gcons', scatter_kws={'alpha': 0.1}, ax=axes[i])

    axes[i].set_xlim([xmin, xmax])
    axes[i].set_ylim([ymin, maxgcons-ymin])
    axes[i].set_ylabel('annual gas consumption (kWh)')
    axes[i].set_xlabel(xaxisnames[i])
    if i != 0:
        axes[i].get_yaxis().set_ticks([])
        axes[i].set_ylabel('')


plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.2)

plt.savefig('need2012gas_regplots.png')

# ---------------------
# create figure with a number of subplots for electricity
f, axes = plt.subplots(1, nplot, figsize=(3*nplot, 3))

ymin = -2500

for i in range(nplot):
    xmax = len(temp[cols[i]].unique()) + 1
    xmin = temp[cols[i]].min() - 1

    sns.regplot(data=temp, x=cols[i], y='econs', scatter_kws={'alpha': 0.1}, ax=axes[i])

    axes[i].set_xlim([xmin, xmax])
    axes[i].set_ylim([ymin, maxecons-ymin])
    axes[i].set_ylabel('annual electricity consumption (kWh)')
    axes[i].set_xlabel(xaxisnames[i])
    if i != 0:
        axes[i].get_yaxis().set_ticks([])
        axes[i].set_ylabel('')


plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.2)

plt.savefig('need2012elec_regplots.png')

# ---------------------
# create figure with a number of BOXPLOTS for gas
f, axes = plt.subplots(1, nplot, figsize=(3*nplot, 3))

ymin = 0

for i in range(nplot):
    xmax = len(temp[cols[i]].unique())
    xmin = temp[cols[i]].min() - 2

    sns.boxplot(data=temp, x=cols[i], y='gcons', ax=axes[i])

    axes[i].set_xlim(xmin, xmax)
    axes[i].set_ylim([ymin, maxgcons-ymin])
    axes[i].set_ylabel('annual gas consumption (kWh)')
    axes[i].set_xlabel(xaxisnames[i])
    if i != 0:
        axes[i].get_yaxis().set_ticks([])
        axes[i].set_ylabel('')

plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.2)

plt.savefig('need2012gas_boxplots.png')

# ---------------------
# create figure with a number of BOXPLOTS for electricity
f, axes = plt.subplots(1, nplot, figsize=(3*nplot, 3))

ymin = 0

for i in range(nplot):
    xmax = len(temp[cols[i]].unique())
    xmin = temp[cols[i]].min() - 2

    sns.boxplot(data=temp, x=cols[i], y='econs', ax=axes[i])

    axes[i].set_xlim(xmin, xmax)
    axes[i].set_ylim([ymin, maxecons-ymin])
    axes[i].set_ylabel('annual electricity consumption (kWh)')
    axes[i].set_xlabel(xaxisnames[i])
    if i != 0:
        axes[i].get_yaxis().set_ticks([])
        axes[i].set_ylabel('')

plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.2)

plt.savefig('need2012elec_boxplots.png')

# ---------------------

cols2 = ['mainheatfuel', 'walls', 'cwi', 'loftins', 'boiler', 'loftdepth', 'gcons', 'econs']

xaxisnames2 = ['gas / other', 'cavity wall / other', 'wall insulation', 'loft insulation', 'new boiler', 'loft depth']

# temporary dataframe for plotting graphs
t = df[cols2]
temp = t.copy()

# fill NaN as 0
temp = temp.fillna(value=0)

dfloftdepth = df[df.loftdepth != 99]

nplot = len(xaxisnames2)

# ---------------------
# create figure with a number of subplots for gas
f, axes = plt.subplots(1, nplot, figsize=(3*nplot, 3))

ymin = -5000


for i in range(nplot):
    xmax = temp[cols2[i]].max() + 1
    xmin = temp[cols2[i]].min() - 1

    if cols2[i] == 'loftdepth':
        sns.regplot(data=dfloftdepth, x=cols2[i], y='gcons', scatter_kws={'alpha': 0.1}, ax=axes[i])
        xmax = dfloftdepth[cols2[i]].max() + 1
    else:
        sns.regplot(data=temp, x=cols2[i], y='gcons', scatter_kws={'alpha': 0.1}, ax=axes[i])

    axes[i].set_xlim([xmin, xmax])
    axes[i].set_ylim([ymin, maxgcons-ymin])
    axes[i].set_ylabel('annual gas consumption (kWh)')
    axes[i].set_xlabel(xaxisnames2[i])
    if i != 0:
        axes[i].get_yaxis().set_ticks([])
        axes[i].set_ylabel('')

plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.2)

plt.savefig('need2012gas_regplots2.png')

# ---------------------
# create figure with a number of subplots for electricity
f, axes = plt.subplots(1, nplot, figsize=(3*nplot, 3))

ymin = -2500


for i in range(nplot):
    xmax = temp[cols2[i]].max() + 1
    xmin = temp[cols2[i]].min() - 1

    if cols2[i] == 'loftdepth':
        sns.regplot(data=dfloftdepth, x=cols2[i], y='econs', scatter_kws={'alpha': 0.1}, ax=axes[i])
        xmax = dfloftdepth[cols2[i]].max() + 1
        xmin = dfloftdepth[cols2[i]].min() - 1
    else:
        sns.regplot(data=temp, x=cols2[i], y='econs', scatter_kws={'alpha': 0.1}, ax=axes[i])

    axes[i].set_xlim([xmin, xmax])
    axes[i].set_ylim([ymin, maxecons-ymin])
    axes[i].set_ylabel('annual electricity consumption (kWh)')
    axes[i].set_xlabel(xaxisnames2[i])
    if i != 0:
        axes[i].get_yaxis().set_ticks([])
        axes[i].set_ylabel('')

plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.2)

plt.savefig('need2012elec_regplots2.png')

# ---------------------
# create figure with a number of BOXPLOTS for gas
f, axes = plt.subplots(1, nplot, figsize=(3*nplot, 3))

ymin = 0


for i in range(nplot):
    xmax = temp[cols2[i]].unique().max() + 1
    xmin = temp[cols2[i]].unique().min() - 1
    axes[i].set_xlim(xmin, xmax)

    if cols2[i] == 'loftdepth':
        sns.boxplot(data=dfloftdepth, x=cols2[i], y='gcons', ax=axes[i])
        xmax = dfloftdepth[cols2[i]].max()
        xmin = dfloftdepth[cols2[i]].min() - 2
        axes[i].set_xlim(xmin, xmax)
    else:
        sns.boxplot(data=temp, x=cols2[i], y='gcons', ax=axes[i])

    axes[i].set_ylim([ymin, maxgcons-ymin])
    axes[i].set_ylabel('annual gas consumption (kWh)')
    axes[i].set_xlabel(xaxisnames2[i])
    if i != 0:
        axes[i].get_yaxis().set_ticks([])
        axes[i].set_ylabel('')

plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.2)

plt.savefig('need2012gas_boxplots2.png')

# ---------------------
# create figure with a number of BOXPLOTS for electricity
f, axes = plt.subplots(1, nplot, figsize=(3*nplot, 3))

ymin = 0

for i in range(nplot):
    xmax = temp[cols2[i]].unique().max() + 1
    xmin = temp[cols2[i]].unique().min() - 1
    axes[i].set_xlim(xmin, xmax)

    if cols2[i] == 'loftdepth':
        sns.boxplot(data=dfloftdepth, x=cols2[i], y='econs', ax=axes[i])
        xmax = dfloftdepth[cols2[i]].max()
        xmin = dfloftdepth[cols2[i]].min() - 2
        axes[i].set_xlim(xmin, xmax)
    else:
        sns.boxplot(data=temp, x=cols2[i], y='econs', ax=axes[i])

    axes[i].set_ylim([ymin, maxecons-ymin])
    axes[i].set_ylabel('annual electricity consumption (kWh)')
    axes[i].set_xlabel(xaxisnames2[i])
    if i != 0:
        axes[i].get_yaxis().set_ticks([])
        axes[i].set_ylabel('')

plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9, wspace=0.2, hspace=0.2)

plt.savefig('need2012elec_boxplots2.png')
