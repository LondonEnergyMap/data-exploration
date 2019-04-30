import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def main():

    # filepath to CLEANED need data
    filepath = 'https://media.githubusercontent.com/media/LondonEnergyMap/cleandata/master/need/need_ldn.csv'

    # read csv into dataframe
    df_all = pd.read_csv(filepath)

    interventions = ['cwi', 'loftins', 'boiler']
    titles = ['Cavity wall insulation', 'Loft insulation', 'New boiler']
    leglabels = ['before', 'during', 'after']

    xyear = [x+'_year' for x in interventions]
    
    # ------------
    for i in range(len(interventions)):
        temp_inter = df_all[['hid', interventions[i], xyear[i], 'gcons', 'econs', 'year']]
        temp = temp_inter[temp_inter[interventions[i]]==1]
        temp['xhue'] = temp.apply(colorgen, xyear=xyear[i],  axis=1) 

        d = {x: i for i, x in enumerate(set(temp.hid))}
        temp['hid2'] = [d[x] for x in temp['hid']]
        l = len(temp.hid2.unique())

        # analysis of effectiveness of the intervention
        temp_eff = temp.groupby(['hid2', 'xhue']).mean()
        temp_eff = temp_eff[['gcons']]
        t = temp_eff.unstack()
        t.columns = t.columns.droplevel(level=1)
        t.columns = leglabels
        t = t.dropna()
        t['dif'] = t.after - t.before
        t['eff'] = np.where(t.dif<0, '1', '0')
        nhouses = len(t.index)
        t['eff'] = pd.to_numeric(t.eff)
        eff_percent = t.eff.sum()/nhouses
        print(eff_percent)
        
        # plot and save scatter graphs for different interventions
        plt.figure(figsize=(25, 5))
        g = sns.scatterplot(data=temp, x='hid2', y='gcons', hue='xhue')
        plt.xlim([-1,l+1])
        plt.xlabel('Household ID')
        plt.ylabel('annual gas consumption (kWh)')
        plt.title(titles[i] + 'effectiveness = %0.1f%%' % (eff_percent*100))

        legend = g.legend()
        handles, labels = g.get_legend_handles_labels()
        g.legend(handles=handles[1:], labels=leglabels, title='Period relative to intervention')

        plt.savefig(interventions[i]+'.png')
        
        # zoom into 100 houses only and save
        plt.xlim([-1,101])
        plt.savefig(interventions[i]+'_zoom100.png')

        # create distribution graphs of gas consumption based on effectiveness
        t_eff = t[t.eff==1]
        t_neff = t[t.eff==0]
        
        plt.figure()
        sns.kdeplot(t_eff.before, shade=True)
        sns.kdeplot(t_neff.before, shade=True)
        plt.legend(['effective', 'not effective'])
        plt.xlabel('annual gas consumption (kWh)')
        plt.title(titles[i] + ' - consumption before intervention')
        plt.savefig(interventions[i]+'_before.png')

        plt.figure()
        sns.kdeplot(t_eff.before, shade=True, color='r')
        sns.kdeplot(t_eff.after, shade=True, color='g')
        plt.xlabel('annual gas consumption (kWh)')
        plt.title(titles[i] + ' - before and after effective intervention')
        plt.savefig(interventions[i]+'_eff.png')

        plt.figure()
        sns.kdeplot(t_neff.before, shade=True, color='r')
        sns.kdeplot(t_neff.after, shade=True, color='g')
        plt.xlabel('annual gas consumption (kWh)')
        plt.title(titles[i] + ' - before and after NON-effective intervention')
        plt.savefig(interventions[i]+'_neff.png')

        plt.figure()
        sns.kdeplot(t_eff.before, shade=True)
        sns.kdeplot(t_eff.after, shade=True)
        sns.kdeplot(t_neff.before, shade=True)
        sns.kdeplot(t_neff.after, shade=True)
        plt.legend(['effective before', 'effective after', 'not effective before', 'not effective after'])
        plt.xlabel('annual gas consumption (kWh)')
        plt.title(titles[i] + ' - before and after intervention')
        plt.savefig(interventions[i]+'_both.png')

def colorgen(df, xyear):
    c = 100 # error value
    if df.year > df[str(xyear)]:
        c = 1
    if df.year < df[str(xyear)]:
        c = -1
    if df.year == df[str(xyear)]:
        c = 0
    return c

# call main body of code
if __name__=='__main__':
    main()
