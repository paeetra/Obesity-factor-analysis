#!/usr/bin/env python
# coding: utf-8

# #Data loading

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='openpyxl')


# In[4]:


df_population_total=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\population_2019_total.xlsx", sheet_name='Sheet 1', header=1, skiprows=10)


# In[5]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[6]:


df_population_total=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\population_2019_total.xlsx", sheet_name='Sheet 1', header=1, skiprows=9)


# In[7]:


df_population_total


# In[8]:


df_population_total.rename(columns={"GEO (Labels)": "GEO labels", "Unnamed: 1": "Population_total", "Unnamed: 2": "Flag_t"},inplace=True)


# In[9]:


df_population_total


# In[10]:


df_population_male=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\population_m.xlsx", sheet_name='Sheet 1', header=1, skiprows=9)


# In[11]:


df_population_male.rename(columns={"GEO (Labels)": "GEO labels", "Unnamed: 1": "Population_male", "Unnamed: 2": "Flag_m"}, inplace=True)


# In[12]:


df_population_male


# In[13]:


df_population_female=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\population_f.xlsx", sheet_name='Sheet 1', header=1, skiprows=9)
df_population_female.rename(columns={"GEO (Labels)": "GEO labels", "Unnamed: 1": "Population_female", "Unnamed: 2": "Flag_f"}, inplace=True)


# In[14]:


df_population_female


# In[15]:


df_bmi_total=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\bmi_total.xlsx", sheet_name='Sheet 1', header=1, skiprows=11)


# In[16]:


df_bmi_total.rename(columns={"GEO (Labels)": "GEO labels", "Unnamed: 1": "Underweight_total", "Unnamed: 3": "Normal_total", "Unnamed: 5": "Obese_total"}, inplace=True)


# In[17]:


df_bmi_total


# In[18]:


df_bmi_male=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\bmi_m.xlsx", sheet_name='Sheet 1', header=1, skiprows=11)


# In[19]:


df_bmi_male.rename(columns={"GEO (Labels)": "GEO labels", "Unnamed: 1": "Underweight_m", "Unnamed: 3": "Normal_m", "Unnamed: 5": "Obese_m"}, inplace=True)


# In[20]:


df_bmi_male


# In[21]:


df_bmi_female=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\bmi_f.xlsx", sheet_name='Sheet 1', header=1, skiprows=11)


# In[22]:


df_bmi_female.rename(columns={"GEO (Labels)": "GEO labels", "Unnamed: 1": "Underweight_f", "Unnamed: 3": "Normal_f", "Unnamed: 5": "Obese_f"},inplace=True)


# In[23]:


df_bmi_female


# In[24]:


population_t_m=pd.merge(df_population_total,df_population_male,how='inner', on='GEO labels')


# In[25]:


population_t_m


# In[26]:


population_merge=pd.merge(population_t_m,df_population_female,how='left', on='GEO labels')


# In[27]:


population_merge


# In[28]:


population_merge


# In[29]:


#deleting table rows that will not be used in the analysis because there is no data related to them in other tables 
#deleting table rows that contain explanations for special characters


# In[30]:


population_merge.drop([0,1,2,3,4,10,34,35,36,37,60,61,62,63,64,65,66,67], inplace=True) 


# In[31]:


population_merge


# In[32]:


population_merge.reset_index(drop=True, inplace=True)


# In[33]:


population_merge


# In[34]:


bmi_t_m=pd.merge(df_bmi_total,df_bmi_male,how='left', on='GEO labels')


# In[35]:


bmi_t_m


# In[36]:


bmi_merge=pd.merge(bmi_t_m,df_bmi_female,how='left', on='GEO labels')


# In[37]:


bmi_merge


# In[38]:


population_bmi=pd.merge(population_merge,bmi_merge, how='right', on= 'GEO labels')


# In[39]:


population_bmi


# In[40]:


#deleting excess rows that will not be used in analysis


# In[41]:


population_bmi.drop([0,1,34,35,36,37,38], inplace=True)


# In[42]:


#dataset index reset


# In[43]:


population_bmi.reset_index(drop=True, inplace=True)


# In[44]:


population_bmi


# In[45]:


df_daily_fruit_veg=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\daily_fruit_and_veg.xlsx", sheet_name='Sheet 1', header=1, skiprows=10)


# In[46]:


df_daily_fruit_veg


# In[47]:


df_daily_fruit_veg.rename(columns={"N_PORTION (Labels)": "GEO labels"}, inplace=True)


# In[48]:


df_daily_fruit_veg


# In[49]:


df_daily_fruit_veg.drop([0,1,2,35,36,37], inplace=True)


# In[50]:


df_daily_fruit_veg.reset_index(drop=True, inplace=True)


# In[51]:


df_daily_fruit_veg


# In[52]:


dataset=pd.merge(population_bmi,df_daily_fruit_veg, how='right', on= 'GEO labels')


# In[53]:


dataset


# In[54]:


df_sweet_drinks=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\sweet_drinks_total.xlsx", sheet_name='Sheet 1', header=1, skiprows=10)


# In[55]:


df_sweet_drinks


# In[56]:


df_sweet_drinks.rename(columns={"FREQUENC (Labels)": "GEO labels"}, inplace=True)


# In[57]:


df_sweet_drinks.drop([0,1,32,33,34], inplace=True)


# In[58]:


df_sweet_drinks


# In[59]:


df_sweet_drinks.reset_index(drop=True, inplace=True)


# In[60]:


df_sweet_drinks


# In[61]:


dataset


# In[62]:


dataset=pd.merge(dataset,df_sweet_drinks, how='outer', on= 'GEO labels')


# In[63]:


dataset


# In[64]:


dataset


# In[65]:


df_physical_activites=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\non_work_related_physical_activites_total.xlsx", sheet_name='Sheet 1', header=1, skiprows=10)


# In[66]:


df_physical_activites


# In[67]:


df_physical_activites.rename(columns={"PHYSACT (Labels)": "GEO labels"}, inplace=True)


# In[68]:


df_physical_activites.drop([35,36,37,38,39], inplace=True)


# In[69]:


df_physical_activites.reset_index(drop=True, inplace=True)


# In[70]:


df_physical_activites


# In[71]:


df_physical_activites.drop([0,1,2], inplace=True)
df_physical_activites.reset_index(drop=True, inplace=True)
df_physical_activites


# In[72]:


cols_to_drop = df_physical_activites.columns[df_physical_activites.columns.str.contains('Unnamed')]
df_physical_activites.drop(cols_to_drop, axis=1, inplace=True)


# In[73]:


dataset=pd.merge(dataset,df_physical_activites, how='outer', on= 'GEO labels')


# In[74]:


dataset


# In[75]:


df_he_0=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\HE_total_0min.xlsx", sheet_name='Sheet 1', header=1, skiprows=11)


# In[76]:


df_he_0


# In[77]:


df_he_0.rename(columns={"GEO (Labels)": "GEO labels", "Unnamed: 1": "HE_0_MIN", "Unnamed: 2": "Flag_HE_0"}, inplace=True)


# In[78]:


df_he_0


# In[79]:


df_he_0.drop([0,1,35,36,37,38], inplace=True)


# In[80]:


df_he_0


# In[81]:


df_he_0.reset_index(drop=True, inplace=True)


# In[82]:


dataset=pd.merge(dataset,df_he_0, how='inner', on= 'GEO labels')


# In[83]:


dataset


# In[84]:


df_he_149=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\HE_total_1_149.xlsx", sheet_name='Sheet 1', header=1, skiprows=11)
df_he_149


# In[85]:


df_he_149.rename(columns={"GEO (Labels)": "GEO labels", "Unnamed: 1": "HE_1_149", "Unnamed: 2": "Flag_HE_1_49"}, inplace=True)
df_he_149


# In[86]:


df_he_149.drop([0,1,34,35,36], inplace=True)



# In[87]:


df_he_149.reset_index(drop=True, inplace=True)
dataset=pd.merge(dataset,df_he_149, how='inner', on= 'GEO labels')
dataset


# In[88]:


df_he_150=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\HE_total_over_150.xlsx", sheet_name='Sheet 1', header=1, skiprows=11)
df_he_150


# In[89]:


df_he_150.rename(columns={"GEO (Labels)": "GEO labels", "Unnamed: 1": "HE_150_MIN", "Unnamed: 2": "Flag_HE_150"}, inplace=True)

df_he_150.drop([0,1,34,35,36], inplace=True)

df_he_150.reset_index(drop=True, inplace=True)

dataset=pd.merge(dataset,df_he_150, how='inner', on= 'GEO labels')
dataset


# In[90]:


df_he_300=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\HE_total_over_300.xlsx", sheet_name='Sheet 1', header=1, skiprows=11)
df_he_300


# In[91]:


df_he_300.rename(columns={"GEO (Labels)": "GEO labels", "Unnamed: 1": "HE_300_MIN", "Unnamed: 2": "Flag_HE_300"}, inplace=True)

df_he_300.drop([0,1,34,35,36,37,38], inplace=True)

df_he_300.reset_index(drop=True, inplace=True)

dataset=pd.merge(dataset,df_he_300, how='inner', on= 'GEO labels')
dataset


# In[92]:


df_daily_cigarettes=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\daily_cigarettes.xlsx", sheet_name='Sheet 1', header=1, skiprows=10)


# In[93]:


df_daily_cigarettes


# In[94]:


df_daily_cigarettes.rename(columns={"SMOKING (Labels)": "GEO labels","Unnamed: 4": "flag<20", "Unnamed: 6": "flag>20"}, inplace=True)


# In[95]:


df_daily_cigarettes


# In[96]:


df_daily_cigarettes


# In[97]:


dataset=pd.merge(dataset,df_daily_cigarettes, how='outer', on= 'GEO labels')


# In[98]:


dataset


# In[99]:


dataset.rename(columns={"Total": "Total smokers"}, inplace=True)
dataset


# In[100]:


cols_to_drop = dataset.columns[dataset.columns.str.contains('Unnamed')]
dataset.drop(cols_to_drop, axis=1, inplace=True)


# In[101]:


df_alcohol=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\alcohol_consumtion_total.xlsx", sheet_name='Sheet 1', header=1, skiprows=10)


# In[102]:


df_alcohol


# In[103]:


df_alcohol.rename(columns={"FREQUENC (Labels)": "GEO labels",}, inplace=True)


# In[104]:


cols_to_drop = df_alcohol.columns[df_alcohol.columns.str.contains('Unnamed')]
df_alcohol.drop(cols_to_drop, axis=1, inplace=True)


# In[105]:


df_alcohol.drop([0,1,2,35,37,38,39], inplace=True)


# In[106]:


df_alcohol


# In[107]:


df_alcohol.reset_index(drop=True, inplace=True)


# In[108]:


df_alcohol


# In[109]:


dataset=pd.merge(dataset,df_alcohol, how='right', on= 'GEO labels')


# In[110]:


dataset


# In[111]:


pd.set_option('display.max_columns', None)


# In[112]:


dataset


# In[113]:


dataset.drop([32], inplace=True)


# In[114]:


dataset.reset_index(drop=True, inplace=True)


# In[115]:


dataset


# In[116]:


flag_column = dataset['Flag_t'].copy()
flag_column.replace(np.nan, '/', inplace=True)
dataset['Flag_t'] = flag_column
dataset


# In[117]:


flag_column = dataset['Flag_m'].copy()
flag_column.replace(np.nan, '/', inplace=True)
dataset['Flag_m'] = flag_column

flag_column = dataset['Flag_f'].copy()
flag_column.replace(np.nan, '/', inplace=True)
dataset['Flag_f'] = flag_column


# In[118]:


dataset


# In[119]:


#removing United Kingdom because it doesn't have any data
dataset.drop([29], inplace=True)
dataset.reset_index(drop=True, inplace=True)


# In[120]:


dataset


# In[121]:


cols_to_drop = dataset.columns[dataset.columns.str.contains('Unnamed')]
dataset.drop(cols_to_drop, axis=1, inplace=True)


# In[122]:


dataset.replace(":", np.nan, inplace=True)


# In[123]:


dataset


# In[124]:


#removing Ireland because it's missing BMI data


# In[125]:


dataset.drop([6], inplace=True)
dataset


# In[126]:


#dodavanje podataka o uvjetima stanovanja (gradovi, manji gradovi i predgrađa, ruralni krajevi)


# In[127]:


df_cities=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\cities.xlsx", sheet_name='Sheet 1', header=1, skiprows=9)


# In[128]:


df_cities


# In[129]:


df_cities.rename(columns={"TIME": "GEO labels","2019":"Living in cities","Unnamed: 2":"Cities_flag"}, inplace=True)


# In[130]:


df_cities


# In[131]:


df_cities.drop([0,1,2,3,4,5,6,7,8,43,44,45,46,47,48], inplace=True)


# In[132]:


df_cities


# In[133]:


df_cities.reset_index(drop=True, inplace=True)


# In[134]:


df_cities


# In[135]:


df_town_sub=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\town_sub.xlsx", sheet_name='Sheet 1', header=1, skiprows=9)


# In[136]:


df_town_sub


# In[137]:


df_town_sub.rename(columns={"TIME": "GEO labels","2019":"Living_in_towns_sub","Unnamed: 2":"Towns_flag"}, inplace=True)
df_town_sub.drop([0,1,2,3,4,5,6,7,8,43,44,45,46,47,48], inplace=True)
df_town_sub.reset_index(drop=True, inplace=True)


# In[138]:


df_town_sub


# In[139]:


df_rural=pd.read_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Original datasets\rural.xlsx", sheet_name='Sheet 1', header=1, skiprows=9)


# In[140]:


df_rural


# In[141]:


df_rural.rename(columns={"TIME": "GEO labels","2019":"Living_in_rural_areas","Unnamed: 2":"Rural_flag"}, inplace=True)
df_rural.drop([0,1,2,3,4,5,6,7,8,43,44,45,46,47,48], inplace=True)
df_rural.reset_index(drop=True, inplace=True)


# In[142]:


df_rural


# In[143]:


df_rural.drop([34], inplace=True)


# In[144]:


df_rural


# In[145]:


dataset


# In[146]:


df_cities=pd.merge(df_cities,df_town_sub, how='inner', on= 'GEO labels')
df_cities=pd.merge(df_cities,df_rural, how='inner', on= 'GEO labels')


# In[147]:


df_cities


# In[148]:


dataset=pd.merge(dataset,df_cities, how='outer', on= 'GEO labels')


# In[149]:


dataset


# In[150]:


dataset.drop([30,31,32,33,34], inplace=True)


# In[151]:


dataset


# In[152]:


dataset.drop([24],inplace=True)
df_rural.reset_index(drop=True, inplace=True)


# In[153]:


dataset


# In[154]:


dataset.columns


# In[155]:


#grouping data by economic regions


# In[156]:


western = ['Ireland', 'France', 'Netherlands', 'Belgium', 'Luxembourg', 'Monaco']
northern =['Iceland', 'Denmark', 'Norway', 'Sweden', 'Finland', 'Lithuania', 'Latvia', 'Estonia']
central =['Germany', 'Austria', 'Switzerland', 'Liechtenstein', 'Slovenia', 'Hungary', 'Czechia', 'Slovakia', 'Poland', 'Croatia']
southern =['Portugal', 'Spain', 'Italy', 'Greece', 'Albania', 'Malta']
southeastern =['Romania', 'Bulgaria', 'Macedonia', 'Bosnia and Herzegovina', 'Serbia', 'Montenegro', 'Kosovo','Türkiye','Cyprus']


# In[157]:


region_mapping = {country: 'Western' for country in western}
region_mapping.update({country: 'Northern' for country in northern})
region_mapping.update({country: 'Central' for country in central})
region_mapping.update({country: 'Southern' for country in southern})
region_mapping.update({country: 'Southeastern' for country in southeastern})
dataset['Region'] = dataset['GEO labels'].map(region_mapping)
dataset.insert(1, 'Region', dataset.pop('Region'))
dataset


# In[158]:


df_western=dataset[dataset['GEO labels'].isin(western)]


# In[159]:


df_western[['GEO labels','Region','Population_total','Obese_total','Obese_m','Obese_f']]


# In[160]:


df_northern=dataset[dataset['GEO labels'].isin(northern)]
df_central=dataset[dataset['GEO labels'].isin(central)]
df_southern=dataset[dataset['GEO labels'].isin(southern)]
df_southeastern=dataset[dataset['GEO labels'].isin(southeastern)]


# In[161]:


df_northern


# In[162]:


df_central[['GEO labels','Region','Population_total','Obese_total','Obese_m','Obese_f']]


# In[163]:


df_southern[['GEO labels','Region','Population_total','Obese_total','Obese_m','Obese_f']]


# In[164]:


d=['Flag_HE_0','Flag_HE_1_49', 'Flag_HE_150','Flag_HE_300', 'flag<20', 'flag>20','Cities_flag','Towns_flag','Rural_flag']

for x in d:
    flag_column = dataset[x].copy()
    flag_column.replace(np.nan, '/', inplace=True)
    dataset[x] = flag_column


# In[165]:


dataset


# In[166]:


#filling in missing data with the median of the region in which they are located


# In[167]:


dataset.replace(":", np.nan, inplace=True)


# In[168]:


#for Finland, the median is needed for 'Aerobic sports', 'HE_0_MIN', 'HE_1_149', 'HE_150_MIN', 'HE_300_MIN', 'Every day', 'Every week', 'Every month', 'Less than once a month', 'Never or not in the last 12 months' 
#since there are too many values to estimate, and it would affect the results, Finland is removed


# In[169]:


types = dataset['Muscle-strengthening'].apply(type)
print(types)


# In[170]:


print(dataset['Muscle-strengthening'])


# In[171]:


#Latvia - 'Muscle-strengthening'
latvia_ms_median = df_northern['Muscle-strengthening'].median()
print(latvia_ms_median)


# In[172]:


#Netherlands - "Sugar sweetened drinks"
niz_1_median = df_western['At least once a day'].median()
print(niz_1_median)
niz_1_3_median = df_western['From 1 to 3 times a week'].median()
print(niz_1_3_median)
niz_4_6_median = df_western['From 4 to 6 times a week'].median()
print(niz_4_6_median)
niz_never_median = df_western['Never or occasionally'].median()
print(niz_never_median)


# In[173]:


#Turkey - "Level of urbansation"
turk_city_median = df_southeastern['Living in cities'].median()
turk_town_median = df_southeastern['Living_in_towns_sub'].median()
turk_rural_median = df_southeastern['Living_in_rural_areas'].median()


# In[174]:


dataset['Muscle-strengthening'].replace({np.nan: latvia_ms_median}, inplace=True)


# In[175]:


dataset['At least once a day'].replace({np.nan: niz_1_median}, inplace=True)


# In[176]:


dataset['From 1 to 3 times a week'].replace({np.nan: niz_1_3_median}, inplace=True)


# In[177]:


dataset['From 4 to 6 times a week'].replace({np.nan: niz_4_6_median}, inplace=True)


# In[178]:


dataset['Never or occasionally'].replace({np.nan: niz_never_median}, inplace=True)


# In[179]:


dataset['Living in cities'].replace({np.nan: turk_city_median}, inplace=True)


# In[180]:


dataset['Living_in_towns_sub'].replace({np.nan: turk_town_median}, inplace=True)


# In[181]:


dataset['Living_in_rural_areas'].replace({np.nan: turk_rural_median}, inplace=True)


# In[182]:


dataset


# In[183]:


#the same values are missing in data frames for specific regions so the data is updated there as well 

df_northern=dataset[dataset['GEO labels'].isin(northern)]
df_central=dataset[dataset['GEO labels'].isin(central)]
df_southern=dataset[dataset['GEO labels'].isin(southern)]
df_southeastern=dataset[dataset['GEO labels'].isin(southeastern)]


# dataset['Muscle-strengthening'].replace({np.nan: latvia_ms_median}, inplace=True)
# dataset['At least once a day'].replace({np.nan: niz_1_median}, inplace=True)
# dataset['From 1 to 3 times a week'].replace({np.nan: niz_1_3_median}, inplace=True)
# dataset['From 4 to 6 times a week'].replace({np.nan: niz_4_6_median}, inplace=True)
# dataset['Never or occasionally'].replace({np.nan: niz_never_median}, inplace=True)
# dataset['Living in cities'].replace({np.nan: turk_city}, inplace=True)
# dataset['Living_in_towns_sub'].replace({np.nan: turk_town}, inplace=True)
# dataset['Living_in_rural_areas'].replace({np.nan: turk_rural}, inplace=True)

# In[184]:


dataset


# In[185]:


#PZ2_Data_Manipulation


# In[186]:


#dataframe dimensions
shape=dataset.shape


# In[187]:


shape


# In[188]:


#print the first 5 rows
dataset.head()


# In[189]:


#print the last 5 rows
dataset.tail()


# In[190]:


dataset


# In[191]:


#calculating the median of obese people in European countries


# In[192]:


obesity_median = dataset['Obese_total'].median()


# In[193]:


obesity_median


# In[194]:


#calculating the median of obese men


# In[195]:


obesity_median_m = dataset['Obese_m'].median()
print(obesity_median_m)


# In[196]:


#calculating the median of obese women


# In[197]:


obesity_median_f = dataset['Obese_f'].median()
print(obesity_median_f)


# In[198]:


q1_total = dataset['Obese_total'].quantile(0.25)
q3_total = dataset['Obese_total'].quantile(0.75)
iq3=q3_total - q1_total
print(q1_total)
print(q3_total)
print(iq3)


# In[199]:


q1_m = dataset['Obese_m'].quantile(0.25)
q3_m = dataset['Obese_m'].quantile(0.75)
iq3=q3_m - q1_m
print(q1_m)
print(q3_m)
print(iq3)


# In[200]:


q1_f = dataset['Obese_f'].quantile(0.25)
q3_f = dataset['Obese_f'].quantile(0.75)
iq3=q3_f - q1_f
print(q1_f)
print(q3_f)
print(iq3)


# In[201]:


t_stat, p_value = stats.ttest_ind(dataset['Obese_m'], dataset['Obese_f'])
print("t-test: ",t_stat)
print("p-value: ",p_value)


# In[202]:


#t-test for comparing obesity mean between men and women in Europe


# In[203]:


t_stat, p_value = stats.ttest_ind(dataset['Obese_m'], dataset['Obese_f'])
print("t-test: ",t_stat)
print("p-value: ",p_value)


# In[204]:


#countries that have a percentage of obese people higher than the median 


# In[205]:


obese_filtered=dataset[dataset['Obese_total'] > obesity_median]


# In[206]:


obese_filtered


# In[207]:


max_obese= obese_filtered[['GEO labels','Population_total', 'Obese_total']].sort_values(by='Obese_total',ascending=False)
max_obese


# In[208]:


max_HE_150= dataset[['GEO labels','Population_total', 'HE_150_MIN']].sort_values(by='HE_150_MIN',ascending=False)
max_HE_150


# In[209]:


max_HE_300= dataset[['GEO labels','Population_total', 'HE_300_MIN']].sort_values(by='HE_300_MIN',ascending=False)
max_HE_300


# In[210]:


max_obese.head()


# In[211]:


#obesity median by region


# In[212]:


n_obesity_median = df_northern['Obese_total'].median()
central_obesity_median = df_central['Obese_total'].median()
w_obesity_median = df_western['Obese_total'].median()
s_obesity_median = df_southern['Obese_total'].median() 
se_obesity_median = df_southeastern['Obese_total'].median()
print("% obese population in Northern Europe: "+ str(n_obesity_median))
print("% obese population in Western Europe: "+ str(w_obesity_median))
print("% obese population in Central Europe: "+ str(central_obesity_median))
print("% obese population in Southern Europe: "+ str(s_obesity_median))
print("% obese population in Southeastern Europe: "+ str(se_obesity_median))


# In[213]:


n_obesity_median = df_northern['Obese_f'].median()
central_obesity_median = df_central['Obese_f'].median()
s_obesity_median = df_southern['Obese_f'].median() 
se_obesity_median = df_southeastern['Obese_f'].median()
print(n_obesity_median)
print(central_obesity_median )
print(s_obesity_median)
print(se_obesity_median)


# In[214]:


n_obesity_median = df_northern['Obese_m'].median()
central_obesity_median = df_central['Obese_m'].median()
s_obesity_median = df_southern['Obese_m'].median() 
se_obesity_median = df_southeastern['Obese_m'].median()
print(n_obesity_median)
print(central_obesity_median )
print(s_obesity_median)
print(se_obesity_median)


# In[215]:


from scipy.stats import f_oneway


# In[216]:


stat,pvalue= f_oneway(df_northern['Obese_total'],df_central['Obese_total'], df_southern['Obese_total'],df_southeastern['Obese_total'])
print("stat: ",stat)
print("pvalue: ", pvalue)


# In[217]:


dataset


# In[218]:


#comparison of alcohol consumption on a daily basis (by region)


# In[219]:


n_alcohol_median = df_northern['Every day'].median() #svaki dan konzumiraju alkohol
print("% consuming alcohol on a daily basis in Northern Europe: "+ str(n_alcohol_median))
central_alcohol_median = df_central['Every day'].median() 
print("% consuming alcohol on a daily basisn in Central Europe: "+ str(central_alcohol_median))
s_alcohol_median = df_southern['Every day'].median() 
print("% consuming alcohol on a daily basis in Southern Europe: "+ str(s_alcohol_median))
se_alcohol_median = df_southeastern['Every day'].median()
print("% consuming alcohol on a daily basis in Southeastern Europe: "+ str(se_alcohol_median))


# In[220]:


#comparion of percentage of people who cycle to get to and from place (by region)
n_alcohol_median = df_northern['Cycling to get to and from place'].median() 
print("% cycle to and from places in Northern Europe: "+ str(n_alcohol_median))
central_alcohol_median = df_central['Cycling to get to and from place'].median() 
print("% cycle to and from places in Central Europe: "+ str(central_alcohol_median))
s_alcohol_median = df_southern['Cycling to get to and from place'].median() 
print("% cycle to and from places in Southern Europe: "+ str(s_alcohol_median))
se_alcohol_median = df_southeastern['Cycling to get to and from place'].median()
print("% cycle to and from places in Southeastern Europe: "+ str(se_alcohol_median))


# In[221]:


#countries in ascending order by the percetange of people that consume alcohol on a daily basis


# In[222]:


dataset[['GEO labels', 'Every day']].sort_values(by='Every day', ascending=False).rename(columns={'GEO labels': 'GEO labels', 'Every day': 'Daily Consumption'})


# In[223]:


#inferencial statistics and statistical analysis


# In[224]:


import scipy.stats as stats


# In[225]:


#Pearson correlation factor


# In[226]:


dataset


# In[227]:


#obesity and walking

corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['Walking to get to and from place'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[228]:


dataset.to_excel(r"C:\Users\petra\Desktop\faks\treca\ZAVRSNI\Eurostat_data\Obesity_data_new.xlsx", index=False)


# # KORELACIJA

# In[229]:


#correlation "Daily fruit and veg"


# In[230]:


#0 portions a day
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['0 portions'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[231]:


#From 1 to 4 portions
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['From 1 to 4 portions'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[232]:


#5 portions or more
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['5 portions or more'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[233]:


#correlation "sugar softened sweet drinks"


# In[234]:


#At least once a day
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['At least once a day'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[235]:


#From 1 to 3 times a week
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['From 1 to 3 times a week'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[236]:


ax=sns.regplot(data=dataset,x='Obese_total', y='From 1 to 3 times a week', 
scatter_kws={'s':70,'color':'lightblue','edgecolor' : 'darkblue'},line_kws={'color':'#54A3CA'})
ax.set_xlabel('Obese_total',fontsize=10)
ax.set_ylabel('From 1 to 3 times a week',fontsize=10)
ax.set_title(f'Korelacija Obese_total & From 1 to 3 times a week ',fontsize=12,pad=15)
    
plt.tight_layout()
plt.savefig("C:/Users/petra/Desktop/faks/treca/ZAVRSNI/obese_1_3.png")
plt.show()


# In[237]:


plt.scatter(dataset["Obese_total"],dataset["From 1 to 3 times a week"])
plt.xlabel('Postotak pretile populacije')
plt.ylabel('Konzumacija zaslađenih pića 1-3 puta tjedno')
plt.title('Korelacija postotka pretilosti i konzumacije zaslađenih pića')


# In[238]:


#From 4 to 6 times a week
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['From 4 to 6 times a week'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[239]:


#Never or occasionally
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['Never or occasionally'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[240]:


#corelation "Non work related physical activity"


# In[241]:


#Walking to get to and from place
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['Walking to get to and from place'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[242]:


#Cycling to get to and from place
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['Cycling to get to and from place'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[243]:


#Aerobic sports
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['Aerobic sports'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[244]:


#Muscle-strengthening
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['Muscle-strengthening'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[245]:


#correlation HE activities (0 min)
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['HE_0_MIN'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[246]:


#correlation HE activities (1 - 149 min)
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['HE_1_149'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[247]:


#correlation HE activities (>150 min)
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['HE_150_MIN'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[248]:


#correlation HE activities (>300) min)
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['HE_300_MIN'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[249]:


dataset


# In[250]:


#correlation "Daily smokers of cigarettes"


# In[251]:


#Less than 20 cigarettes per day
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['Less than 20 cigarettes per day'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[252]:


#20 or more cigarettes per day
corr, p_value = stats.pearsonr(dataset['Obese_total'], dataset['20 or more cigarettes per day'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[253]:


#correlation "Alcohol consumption"


# In[254]:


#correlation "Every day"
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['Every day'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[255]:


#correlation "Every week"
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['Every week'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[256]:


#correlation"Every month"
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['Every month'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[257]:


#correlation "Less than once a month"
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['Less than once a month'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[258]:


#correlation "Never or not in the last 12 months"
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['Never or not in the last 12 months'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[259]:


#correlation "Degree of urbanisation"


# In[260]:


#correlation "Living in cities"
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['Living in cities'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[261]:


ax=sns.regplot(data=dataset,x='Obese_total', y='Living in cities', 
scatter_kws={'s':70,'color':'lightblue','edgecolor' : 'darkblue'},line_kws={'color':'#54A3CA'})
ax.set_xlabel('Obese_total',fontsize=10)
ax.set_ylabel('Living in cities',fontsize=10)
ax.set_title(f'Korelacija Obese_total & Living in cities',fontsize=12,pad=15)
    
plt.tight_layout()
plt.savefig("C:/Users/petra/Desktop/faks/treca/ZAVRSNI/obese_cities.png")
plt.show()


# In[262]:


#correlation "Living in towns and suburbs" 
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['Living_in_towns_sub'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[263]:


ax=sns.regplot(data=dataset,x='Obese_total', y='Living_in_towns_sub', 
scatter_kws={'s':70,'color':'lightblue','edgecolor' : 'darkblue'},line_kws={'color':'#54A3CA'})
ax.set_xlabel('Obese_total',fontsize=10)
ax.set_ylabel('Living in cities',fontsize=10)
ax.set_title(f'Korelacija Obese_total & Living_in_towns_sub ')
    
plt.tight_layout()
plt.savefig("C:/Users/petra/Desktop/faks/treca/ZAVRSNI/obese_towns_sub.png")
plt.show()


# In[264]:


#correlation "Living_in_rural_areas" 
corr, p_value = stats.pearsonr(dataset['Obese_total'],dataset['Living_in_rural_areas'])
print("Correlation:", corr)
print("P-value:", p_value)


# In[265]:


dataset


# In[266]:


# food habits and physical activitiy


# In[267]:


correlation_food_exercise=dataset[['0 portions','From 1 to 4 portions', '5 portions or more', 'At least once a day','From 1 to 3 times a week', 'From 4 to 6 times a week','Never or occasionally', 'Walking to get to and from place','Cycling to get to and from place', 'Aerobic sports','Muscle-strengthening', 'HE_0_MIN','HE_1_149','HE_150_MIN','HE_300_MIN']].corr()
correlation_food_exercise


# In[268]:


dataset.columns


# In[269]:


def align_index(d1,d2):
    d1_clean=d1.dropna(axis=0,how='any')
    d2_clean=d2.dropna(axis=0,how='any')
    common_index =  d1_clean.index.intersection(d2_clean.index)
    d1_aligned = d1_clean[common_index]
    d2_aligned = d2_clean[common_index]
    return d1_aligned,d2_aligned


# In[270]:


#checking statistical siginificance of correlation with "From 1 to 3 times a week"


# In[271]:


data1,data2= align_index(dataset['From 1 to 3 times a week'],dataset['0 portions'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['From 1 to 3 times a week'],dataset['From 1 to 4 portions'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['From 1 to 3 times a week'],dataset['5 portions or more'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['From 1 to 3 times a week'],dataset['From 4 to 6 times a week'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['From 1 to 3 times a week'],dataset['HE_0_MIN'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['From 1 to 3 times a week'],dataset['HE_150_MIN'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['From 1 to 3 times a week'],dataset['Aerobic sports'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['From 1 to 3 times a week'],dataset['Muscle-strengthening'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['From 1 to 3 times a week'],dataset['Walking to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['From 1 to 3 times a week'],dataset['Cycling to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")


# In[272]:


sns.heatmap(correlation_food_exercise,cmap="coolwarm")
plt.tight_layout()
plt.savefig("seaborn_plot.jpg")
plt.savefig("C:/Users/petra/Desktop/faks/treca/UAVP/food_lifestyle.png")


# In[273]:


#food habits & smoking & alcohol consumption


# In[274]:


food_alcohol_cig=dataset[['From 1 to 4 portions', '5 portions or more', 'At least once a day','From 1 to 3 times a week', 'From 4 to 6 times a week','Never or occasionally','Less than 20 cigarettes per day','20 or more cigarettes per day', 'Every day', 'Every week','Every month', 'Less than once a month','Never or not in the last 12 months']]
food_alcohol_cig.corr()


# In[275]:


sns.heatmap(food_alcohol_cig.corr(),cmap="coolwarm")
plt.tight_layout()
plt.savefig("seaborn_plot.jpg")
plt.savefig("C:/Users/petra/Desktop/faks/treca/UAVP/food_alcohol_cig.png")
plt.show()


# In[276]:


#physical acitivity & smoking & alcohol consumption


# In[277]:


lifestyle_alcohol_cig=dataset[[ 'Walking to get to and from place',
       'Cycling to get to and from place', 'Aerobic sports',
       'Muscle-strengthening', 'HE_0_MIN','HE_1_149','HE_150_MIN','HE_300_MIN','Less than 20 cigarettes per day',
       '20 or more cigarettes per day','Every day', 'Every week',
       'Every month', 'Less than once a month',
       'Never or not in the last 12 months']].corr()
lifestyle_alcohol_cig


# In[278]:


sns.heatmap(lifestyle_alcohol_cig,cmap="coolwarm")
plt.tight_layout()
plt.savefig("seaborn_plot.jpg")
plt.savefig("C:/Users/petra/Desktop/faks/treca/UAVP/lifestyle_alcohol_cig.png")
plt.show()


# In[279]:


food_habits_urbanisation=dataset[['From 1 to 4 portions', '5 portions or more', 'At least once a day',
       'From 1 to 3 times a week', 'From 4 to 6 times a week',
       'Never or occasionally', 'Living in cities',
       'Living_in_towns_sub', 'Living_in_rural_areas']]
fha=food_habits_urbanisation.dropna()


# In[280]:


print(fha.dtypes)


# In[281]:


print(fha['Living in cities'].unique())
print(fha['Living_in_towns_sub'].unique())
print(fha['Living_in_rural_areas'].unique())


# In[282]:


dataset['Living in cities'] = pd.to_numeric(dataset['Living in cities'], errors='coerce')
dataset['Living_in_towns_sub'] = pd.to_numeric(dataset['Living_in_towns_sub'], errors='coerce')
dataset['Living_in_rural_areas'] = pd.to_numeric(dataset['Living_in_rural_areas'], errors='coerce')


# In[283]:


#prehrambene navike i urbanizacija


# In[284]:


food_habits_urbanisation=dataset[['From 1 to 4 portions', '5 portions or more', 'At least once a day',
       'From 1 to 3 times a week', 'From 4 to 6 times a week',
       'Never or occasionally', 'Living in cities',
       'Living_in_towns_sub', 'Living_in_rural_areas']]
fha=food_habits_urbanisation.dropna()
fha.corr()


# In[285]:


sns.heatmap(fha.corr(),cmap="coolwarm")
plt.tight_layout()
plt.savefig("seaborn_plot.jpg")
plt.savefig("C:/Users/petra/Desktop/faks/treca/UAVP/food_habits_urbanisation.png")
plt.show()


# In[286]:


physical_activity_urbanisation=dataset[['Walking to get to and from place',
       'Cycling to get to and from place', 'Aerobic sports',
       'Muscle-strengthening', 'HE_0_MIN', 'HE_1_149',
       'HE_150_MIN',  'HE_300_MIN', 'Living in cities',
       'Living_in_towns_sub', 'Living_in_rural_areas']].corr()
physical_activity_urbanisation


# In[287]:


sns.heatmap(physical_activity_urbanisation,cmap="coolwarm")
plt.tight_layout()
plt.savefig("seaborn_plot.jpg")
plt.savefig("C:/Users/petra/Desktop/faks/treca/UAVP/physical_activity_urbanisation.png")
plt.show()


# In[288]:


alcohol_cigarettes_urbanisation=dataset[['Total smokers', 'Less than 20 cigarettes per day',
       '20 or more cigarettes per day', 'Every day',
       'Every week', 'Every month', 'Less than once a month',
       'Never or not in the last 12 months', 'Living in cities',
       'Living_in_towns_sub', 'Living_in_rural_areas']].corr()
alcohol_cigarettes_urbanisation


# In[289]:


sns.heatmap(alcohol_cigarettes_urbanisation,cmap="coolwarm")
plt.tight_layout()
plt.savefig("seaborn_plot.jpg")
plt.savefig("C:/Users/petra/Desktop/faks/treca/UAVP/alcohol_cigarettes_urbanisation.png")
plt.show()


# In[290]:


#checking statistical siginificance of correlation with "Living_in_towns_sub"
data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['HE_150_MIN'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['HE_0_MIN'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['HE_300_MIN'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['Aerobic sports'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['Muscle-strengthening'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['Walking to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['Cycling to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['At least once a day'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['From 1 to 4 portions'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['Every week'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['Every month'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['Every day'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['Never or occasionally'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['Less than 20 cigarettes per day'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['20 or more cigarettes per day'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['Total smokers'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")


# In[291]:


#checking statistical siginificance of correlation with "Living in cities"
data1,data2= align_index(dataset['Living in cities'],dataset['Walking to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living in cities'],dataset['Cycling to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living in cities'],dataset['HE_0_MIN'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living in cities'],dataset['HE_1_149'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living in cities'],dataset['0 portions'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living in cities'],dataset['Every day'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living in cities'],dataset['Every week'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")


# In[292]:


import statsmodels.api as sm


# In[293]:


# regression on statisticly significant correlations and BMI


# In[294]:


X = dataset['From 1 to 3 times a week']
y = dataset['Obese_total']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[295]:


X = dataset['Living_in_towns_sub']
y = dataset['Obese_total']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[296]:


X = dataset['Living in cities']
y = dataset['Obese_total']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[297]:


dataset.columns


# In[298]:


#regression of statistically significant correlations


# In[299]:


factors= [ '0 portions', 'From 1 to 4 portions', '5 portions or more','From 4 to 6 times a week',
            'At least once a day', 'Muscle-strengthening', 'Aerobic sports', 'HE_0_MIN','HE_1_149',
            'HE_150_MIN' ,'20 or more cigarettes per day', 'Cycling to get to and from place', 
            'Every week', 'Every day','Never or occasionally']

def reg_function(f1,data,factors):
    for f in factors:
        X = dataset[f1]
        y = dataset[f]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        print(model.summary())
        print("\n\n***........................................................................***\n\n")


# In[300]:


reg_function('From 1 to 3 times a week',dataset,factors)


# In[301]:


reg_function('Living_in_towns_sub',dataset,factors)


# In[302]:


reg_function('Living in cities',dataset,factors)


# In[303]:


## CORRELATION BETWEEN DIFFERENT OBESITY FACTORS ## (checking for statistical significance)


# In[304]:


#food habits
print("-----prehrambene medusobno")

data1,data2= align_index(dataset['0 portions'],dataset['From 4 to 6 times a week'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['5 portions or more'],dataset['From 4 to 6 times a week'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")


#food habits and physical activity
print("-----prehrambene i fizicka aktivnost")

data1,data2= align_index(dataset['Never or occasionally'],dataset['Muscle-strengthening'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['From 4 to 6 times a week'],dataset['HE_1_149'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['5 portions or more'],dataset['Cycling to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['5 portions or more'],dataset['HE_0_MIN'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['5 portions or more'],dataset['HE_300_MIN'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

#different types of physical activity
print("-----razliciti oblici fizicke aktivnosti")

data1,data2= align_index(dataset['HE_300_MIN'],dataset['Cycling to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['HE_300_MIN'],dataset['Aerobic sports'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['HE_300_MIN'],dataset['Muscle-strengthening'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['HE_150_MIN'],dataset['Cycling to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['HE_150_MIN'],dataset['Aerobic sports'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['HE_150_MIN'],dataset['Muscle-strengthening'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Aerobic sports'],dataset['Muscle-strengthening'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

#food habits & alcohol & cigarettes
print("-----prehrambene navike i alkohol i cigarete")

data1,data2= align_index(dataset['5 portions or more'],dataset['Every week'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['5 portions or more'],dataset['20 or more cigarettes per day'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['From 4 to 6 times a week'],dataset['20 or more cigarettes per day'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

#urbanisation
print("-----utjecaj urbanizacije")

data1,data2= align_index(dataset['At least once a day'],dataset['Living_in_towns_sub'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_rural_areas'],dataset['Walking to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_rural_areas'],dataset['Walking to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")

data1,data2= align_index(dataset['Living_in_towns_sub'],dataset['Cycling to get to and from place'])
corr, p_value = stats.pearsonr(data1, data2)
print(f"corr: {corr},  p: {p_value}")


# In[305]:


dataset.columns


# In[306]:


food_habits = dataset[['From 4 to 6 times a week', '0 portions', '5 portions or more']].corr()
food_habits


# In[307]:


food_habits = dataset[['From 4 to 6 times a week', '0 portions', '5 portions or more']].corr()
ax= sns.heatmap(food_habits,cmap="Blues",linewidths=0.5, linecolor='white',cbar_kws={'label': 'Jačina korelacije', 'orientation': 'vertical'},annot=True)
plt.title('Korelacija različitih prehrambenih navika', fontsize=13, color='black',pad=20)
ax.set_xticklabels(food_habits.columns, rotation=0,fontsize=8)
ax.set_yticklabels(food_habits.columns, rotation=0,ha='right',fontsize=8) 
plt.tight_layout()
plt.savefig("C:/Users/petra/Desktop/faks/treca/ZAVRSNI/food_habits.png")


# In[308]:


pa_habits = dataset[[ 'HE_150_MIN','HE_300_MIN','Cycling to get to and from place','Walking to get to and from place','Aerobic sports','Muscle-strengthening']].corr()

plt.figure(figsize=(20, 14))

ax =sns.heatmap(pa_habits,cmap="Blues",linewidths=0.5, linecolor='white',cbar_kws={'label': 'Jačina korelacije', 'orientation': 'vertical'},annot=True,annot_kws={'size': 25})
plt.title('Korelacija različitih oblika fizičkih aktivnosti', fontsize=35, color='black',pad=20,loc='center')

ax.set_xticklabels(pa_habits.columns, rotation=45,ha='right',fontsize=25)
ax.set_yticklabels(pa_habits.columns, rotation=0,ha='right',fontsize=25) 
plt.tight_layout()

#plt.savefig("C:/Users/petra/Desktop/faks/treca/ZAVRSNI/pa_habits.png")
plt.show()


# In[309]:


factors = ['Cycling to get to and from place','HE_300_MIN',
        'From 4 to 6 times a week','HE_0_MIN']

for f in factors:
    ax=sns.regplot(data=dataset,x='5 portions or more', y=f,
        scatter_kws={'s':70,'color':'lightblue','edgecolor' : 'darkblue'},line_kws={'color':'#54A3CA'})
        #ili orangered ili darkorange
    ax.set_xlabel('5 portions or more',fontsize=10)
    ax.set_ylabel(f,fontsize=10)
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_title(f'Korelacija 5 portions or more  \n & {f}')
    
    plt.tight_layout()
    
    filename = f"C:/Users/petra/Desktop/faks/treca/ZAVRSNI/{f.replace(' ', '-')}.png"
    plt.savefig(filename)
    
    plt.show()


# In[310]:


food_pa_habits = dataset[['Never or occasionally',
    'Muscle-strengthening',
    'From 4 to 6 times a week',
    'HE_1_149',
    '5 portions or more',
    'Cycling to get to and from place',
    'HE_0_MIN',
    'HE_300_MIN']].corr()

ax =sns.heatmap(food_pa_habits,cmap="coolwarm",linewidths=0.5, vmin=-1, vmax=1, linecolor='white',cbar_kws={'label': 'Jačina korelacije', 'orientation': 'vertical'},annot=True)
plt.title('Korelacija prehrane i \n fizičkih aktivnosti', fontsize=13, color='black',pad=20)
plt.figure(figsize=(12, 10))

ax.set_xticklabels(food_pa_habits.columns, rotation=60,ha='right',fontsize=10)
ax.set_yticklabels(food_pa_habits.columns, rotation=0,ha='right',fontsize=10) 
plt.tight_layout()
plt.savefig("pa_habits.png")
plt.savefig("C:/Users/petra/Desktop/faks/treca/ZAVRSNI/food_pa_habits.png")


# In[311]:


## regression ## 


# In[312]:


X = dataset['Cycling to get to and from place']
y = dataset['HE_300_MIN']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[313]:


X = dataset['Aerobic sports']
y = dataset['HE_300_MIN']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[314]:


X = dataset['Muscle-strengthening']
y = dataset['HE_300_MIN']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[315]:


X = dataset['Muscle-strengthening']
y = dataset['HE_150_MIN']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[316]:


X = dataset['Aerobic sports']
y = dataset['HE_150_MIN']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[317]:


X = dataset['Cycling to get to and from place']
y = dataset['HE_150_MIN']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[318]:


X = dataset['Aerobic sports']
y = dataset['Muscle-strengthening']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[319]:


X = dataset['HE_150_MIN']
y = dataset['HE_300_MIN']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())


# In[320]:


############### REGIONS ################


# In[321]:


df_northern


# In[322]:


df_southern


# In[323]:


df_central


# In[324]:


df_western


# In[325]:


df_southern


# In[326]:


df_southeastern


# In[327]:


region_mean= dataset.groupby('Region').mean()
region_mean


# In[328]:


region_std = dataset.groupby('Region').std()
region_std


# In[329]:


region_medians = dataset.groupby('Region').median()
region_medians


# In[330]:


dataset


# In[ ]:




