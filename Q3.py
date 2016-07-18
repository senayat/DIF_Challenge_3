import pandas as pd
import numpy as np
from collections import Counter
import pylab 
import matplotlib.pyplot as plt

data = pd.read_csv('D:\DIF_Challenge\Challenge3\Crimes_-_2001_to_present.csv')

#####################################
#find the type of crime with the highest frequency over 2001-2016
Type=data['Primary Type'].values.astype('S')
df = pd.DataFrame.from_dict(Counter(Type), orient='index')
ax=df.plot(kind='bar',fontsize=4,legend=False,title='histogram plot of primary crime types in chicago (2001-present)')
fig = ax.get_figure()
fig.savefig('type_hist.png')# theft crime is the highest frequency
#####################################
#find the highest frequency of location desciption of theft
locdesc=data['Location Description'].values.astype('S')
locdesc_theft=locdesc[np.where(Type=='THEFT')]
df_theftType=pd.DataFrame.from_dict(Counter(locdesc_theft), orient='index')
ax_theft=df_theftType.plot(kind='bar',fontsize=4,legend=False,title='histogram plot of location of theft crime in chicago (2001-present)')
fig2 = ax_theft.get_figure()
fig2.savefig('theft_type_hist.png')# street 
#####################################
# find the districts with the highest rate of theft in the street
dist=data['District'].values.astype(int)
dist=np.delete(dist,np.where(dist<0))
loc_theft_street=dist[np.where(locdesc_theft=='STREET')]
df_theft_street=pd.DataFrame.from_dict(Counter(loc_theft_street), orient='index')
street_district=df_theft_street.plot(kind='bar',fontsize=10,legend=False,title='histogram plot of districts of theft in street in chicago (2001-present)')
fig3 = street_district.get_figure()
fig3.savefig('street_district.png')
print "2 districts with the most theft crime in the street:"
dist[loc_theft_street.argsort()[-2:][::-1]]# district 12 and 15
#####################################
#find the trend theft in street for distrcits 12 and 15 over years (since 2001)
year=data['Year'].values.astype(int)
year_theft_street_dis12=year[np.where(dist==12)]
district12_eachyear=dist[np.where(dist==12)]
district12_eachyear=[len(district12_eachyear[np.where(year_theft_street_dis12==x)])for x in np.unique(year_theft_street_dis12)]
year_theft_street_dis15=year[np.where(dist==15)]
district15_eachyear=dist[np.where(dist==15)]
district15_eachyear=[len(district15_eachyear[np.where(year_theft_street_dis15==x)])for x in np.unique(year_theft_street_dis15)]
pylab.plot(np.unique(year_theft_street_dis12), district12_eachyear, 'o-b', label='district12')
pylab.plot(np.unique(year_theft_street_dis15), district15_eachyear, 'o-r', label='district15')
pylab.title('yearly trend of street theft at districts with most frequecy since 2001')
pylab.xlim(2000,2016)
pylab.legend(loc='upper right')
pylab.show()# decreasing vastly
#####################################
# find the crime type with highest occurance rate in the last 3 years
year_last3_id=[np.where(year==x) for x in range(2014,2017)]
type_last3=[Type[year_last3_id[x]] for x in range(3)]
counterAll=[Counter(type_last3[x]) for x in range(3)]
keysAll=[counterAll[x].keys() for x in range(3)]
mean={}
for k in keysAll[0]:
   mean[k]= np.mean([x[k] for x in counterAll])
std={}
for k in keysAll[0]:   
  std[k]=np.std([x[k] for x in counterAll])  

mostImportantKeys=sorted(mean, key=mean.__getitem__, reverse=True)[0:10]
y_pos = np.arange(len(mostImportantKeys))
performance = [mean[k] for k in mostImportantKeys]
error=[std[k] for k in mostImportantKeys]
plt.barh(y_pos, performance, xerr=error, align='center',alpha=0.4)
plt.yticks(y_pos, mostImportantKeys)
plt.title('Crime types with the 10 highest frequencies in last 3 years')
plt.show()
#####################################
#find the trend of highest frequencies over last 3 years
y={}
for k in range(3):
    y[mostImportantKeys[k]]=[counterAll[x][mostImportantKeys[k]] for x in range(3)]
    

pylab.plot(range(1,4),y[mostImportantKeys[0]], 'o-b', label=mostImportantKeys[0])
pylab.plot(range(1,4),y[mostImportantKeys[1]], 'o-r', label=mostImportantKeys[1])
pylab.plot(range(1,4),y[mostImportantKeys[2]], 'o-g', label=mostImportantKeys[2])
pylab.xticks(range(1,4),range(2013,2017))
pylab.legend(loc='upper right')
pylab.title('Trend of 3 crimes with highest rate in the last 3 years')
pylab.show()