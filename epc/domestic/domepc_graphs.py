import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import time


def main():

    # start runtime clock
    start_time = time.time()

    filepath = 'epc_postcodefull.csv'   # using local file for faster speed    
    # filepath = 'https://raw.githubusercontent.com/LondonEnergyMap/cleandata/master/epc/domestic/epc_postcodesample.csv'
    # filepath = 'https://media.githubusercontent.com/media/LondonEnergyMap/cleandata/master/epc/domestic/epc_postcodefull.csv'

    df_all = pd.read_csv(filepath, low_memory=False)
    print('read in data %s sec' % round(time.time() - start_time, 2))
    # -----------------------------------
    # clean data
    # -----------------------------------
    # select only exact poscode matches of gas and electricity meter estimates
    d = df_all[(df_all.gasmatch.str.contains('exact')) &
               (df_all.elecmatch.str.contains('exact'))]

    df = d.copy()

    # clean builtform column
    df['builtform'] = df.builtform.str.replace('NO DATA!', '', n=-1, regex=True)

    # create new column that combines property type and built form
    df['form'] = df.builtform + ' ' + df.prop_type

    # change datatypes of columns
    df = changetypes(df)
    print('changed datatypes %s sec' % round(time.time() - start_time, 2))

    # order the buildings in categorical lists
    df = orderlist(df)
    print('finished cleaning data %s sec' % round(time.time() - start_time, 2))

    # create new total energy consumption columns from estimated epc data and postcode estimates
    df['tcons'] = df.curr_encons*df.tfa
    df['tcons_pcode'] = df.gasmid + df.elecmid
    print('created new total consumption columns %s sec' % round(time.time() - start_time, 2))

    # -----------------------------------
    # Generate different graphs
    # -----------------------------------

    # create comparison boxplots for epc and meter estimates for different house types
    temp = create_dfepcmeter(df)
    print('created new structured dataframe for comparison boxplot %s sec' % round(time.time() - start_time, 2))

    # create figure to show outliers by not adjusting axis, then proper figure
    plt.figure()
    g = sns.boxplot(data=temp, x='prop_type', y='tcons', hue='cons_type')
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.legend(loc='upper center')
    plt.savefig('proptype_outliers.png')
    g.set_ylim(0, 100000)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('proptype_epcmeters.png')

    # create figure for builtform
    plt.figure()
    g = sns.boxplot(data=temp, x='builtform', y='tcons', hue='cons_type')
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set_ylim(0, 100000)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('builtform_epcmeters.png')

    # create figure for builtform
    plt.figure(figsize=(15, 5))
    g = sns.boxplot(data=temp, x='form', y='tcons', hue='cons_type')
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set_ylim(0, 100000)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('form_epcmeters.png')

    print('output all 3 comparison boxplots %s sec' % round(time.time() - start_time, 2))

    # -----------------------------------

    temp = create_dfnmeter(df, 10)
    print('created new structured dataframe for comparison boxplot %s sec' % round(time.time() - start_time, 2))

    # create figure for builtform
    plt.figure(figsize=(15, 5))
    g = sns.boxplot(data=temp, x='form', y='tcons', hue='nmeters')
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set_ylim(0, 100000)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('form_epcmeters.png')

    print('output nmeters comparison boxplots %s sec' % round(time.time() - start_time, 2))

    # -----------------------------------

    # create 2 subplots for postcode matched meters distribution
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    meters = ['gasmeters', 'elecmeters']
    # set x axis limit as maximum of the 2 columns
    xl = df[meters].values.max()

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    xpos = 0.55
    ypos = 0.95

    for i in range(2):
        # draw the distribution
        sns.kdeplot(df[meters[i]], shade=True, legend=False, ax=ax[i])
        ax[i].set_title(df[meters[i]].name)
        ax[i].set_xlabel('number of meters')
        ax[i].set_xlim(0, xl)
        ax[i].set_ylim(0, 0.025)

        # place a text box in upper left in axes coords
        x = df[meters[i]]
        mu = x.mean()
        median = np.median(x)
        sigma = x.std()
        maxx = x.max()
        minn = x.min()
        textstr = '\n'.join((
            r'$\mu=%.2f$' % (mu, ),
            r'$\mathrm{median}=%.2f$' % (median, ),
            r'$\sigma=%.2f$' % (sigma, ),
            r'$\max=%.f$' % (maxx, ),
            r'$\min=%.f$' % (minn, )))
        ax[i].text(xpos, ypos, textstr, transform=ax[i].transAxes,
                   fontsize=14, verticalalignment='top', bbox=props)

    plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9,
                        wspace=0.3, hspace=0.2)
    plt.savefig('meters.png')
    print('output meters graph %s sec' % round(time.time() - start_time, 2))

    # -----------------------------------

    # check difference between mean and median in gas and electricity consumptions
    df['gasdif'] = df.gasavg - df.gasmid
    df['elecdif'] = df.elecavg - df.elecmid

    meter_dist = ['gasdif', 'elecdif']
    meter_disttitles = ['gas', 'electricity']

    for i in range(2):
        sns.boxplot(y=df[meter_dist[i]], ax=ax[i])
        ax[i].set_ylim(-2000, 2000)
        ax[i].set_ylabel('Consumption (average - median)')
        ax[i].set_title(meter_disttitles[i])

    plt.subplots_adjust(left=0.07, bottom=0.2, right=0.95, top=0.9,
                        wspace=0.4, hspace=0.2)
    plt.savefig('meters_skew.png')
    print('output meters_skew graph %s sec' % round(time.time() - start_time, 2))


# function to change datatypes to reduce dataframe size
def changetypes(df):
    # change to category
    cat = [
        'curr_enr',
        'poten_enr',
        'prop_type',
        'builtform',
        'transact_type',
        'glaze_type',
        'glaze_area',
        'hotwtr_eff',
        'floor_eff',
        'window_eff',
        'wall_eff',
        'heat2_eff',
        'roof_eff',
        'heat_eff',
        'control_eff',
        'light_eff',
        'mainfuel',
        'mechvent',
        'gasmatch',
        'elecmatch']

    for i in range(len(cat)):
        df[cat[i]] = df[cat[i]].astype('category')

    # change to float32
    floatlist = [
        'curr_co2',
        'curr_co2perarea',
        'poten_co2',
        'tfa',
        'fheight',
        'gascons',
        'gasmeters',
        'gasavg',
        'gasmid',
        'eleccons',
        'elecmeters',
        'elecavg',
        'elecmid']

    for i in range(len(floatlist)):
        df[floatlist[i]] = df[floatlist[i]].astype('float32')

    # change to int32
    intlist = [
        'flattop_cnt',
        'glaze_percent',
        'nextension',
        'nrooms',
        'nheatedrooms',
        'led_percent',
        'nfireplace',
        'windt',
        'pv']
    for i in range(len(intlist)):
        df[intlist[i]] = df[intlist[i]].astype('int32')

    # change to int8 (boolean) after mapping Y/N to 1/0 and Nan to -1
    int8list = [
        'mainsgas',
        'flattop',
        'solarwtr']

    for i in range(len(int8list)):
        df[int8list[i]] = df[int8list[i]].map({'Y': 1, 'N': 0})
        df[int8list[i]] = df[int8list[i]].fillna(-1)
        df[int8list[i]] = df[int8list[i]].astype('int8')

    return df


def orderlist(df):
    # sort builtform list in order of energy use
    bflist = [
        'Detached',
        'Semi-Detached',
        'End-Terrace',
        'Enclosed End-Terrace',
        'Mid-Terrace',
        'Enclosed Mid-Terrace',
        '']
    df['builtform'] = pd.Categorical(df['builtform'], bflist)

    # sort property type list in order of energy use
    ptlist = [
        'House',
        'Bungalow',
        'Maisonette',
        'Flat']
    df['prop_type'] = pd.Categorical(df['prop_type'], ptlist)

    # sort forms in order of energy use
    sortforms = [
             ' House',
             'Detached House',
             'Semi-Detached House',
             'End-Terrace House',
             'Enclosed End-Terrace House',
             'Mid-Terrace House',
             'Enclosed Mid-Terrace House',
             ' Bungalow',
             'Detached Bungalow',
             'Semi-Detached Bungalow',
             'End-Terrace Bungalow',
             'Enclosed End-Terrace Bungalow',
             'Mid-Terrace Bungalow',
             'Enclosed Mid-Terrace Bungalow',
             ' Maisonette',
             'Detached Maisonette',
             'Semi-Detached Maisonette',
             'End-Terrace Maisonette',
             'Enclosed End-Terrace Maisonette',
             'Mid-Terrace Maisonette',
             'Enclosed Mid-Terrace Maisonette',
             ' Flat',
             'Detached Flat',
             'Semi-Detached Flat',
             'End-Terrace Flat',
             'Enclosed End-Terrace Flat',
             'Mid-Terrace Flat',
             'Enclosed Mid-Terrace Flat',
             'Detached Park home']
    df['form'] = pd.Categorical(df['form'], sortforms)

    return df


def create_dfepcmeter(df):
    # create new structured dataframe for comparison boxplots
    new1 = df[['builtform', 'prop_type', 'form', 'tfa', 'tcons']]
    n1 = new1.copy()
    n1['cons_type'] = 'EPC'

    new2 = df[['builtform', 'prop_type', 'form', 'tfa', 'tcons_pcode']]
    n2 = new2.copy()
    n2.rename(columns={'tcons_pcode': 'tcons'}, inplace=True)
    n2['cons_type'] = 'meter'

    df_epcmeter = n1.append(n2)
    return df_epcmeter


def create_dfnmeter(df, n):
    # create new structured dataframe for comparison boxplots
    df1 = df[(df.gasmeters > n) & (df.elecmeters > n)]
    new1 = df1[['builtform', 'prop_type', 'form', 'tfa', 'tcons_pcode']]
    n1 = new1.copy()
    n1.rename(columns={'tcons_pcode': 'tcons'}, inplace=True)
    n1['nmeters'] = 'less than' + str(n)

    new2 = df[['builtform', 'prop_type', 'form', 'tfa', 'tcons_pcode']]
    n2 = new2.copy()
    n2.rename(columns={'tcons_pcode': 'tcons'}, inplace=True)
    n2['nmeters'] = 'original'

    df_nmeter = n1.append(n2)
    return df_nmeter

if __name__ == "__main__":
    main()
