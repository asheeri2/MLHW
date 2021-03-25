#!/usr/bin/env python
# coding: utf-8

# # Author - Ashleigh Sheerin
# 

# In[16]:


import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs


# In[5]:


df = pd.read_csv("/Users/AshleighSheerin/titanic.csv")
df.head()


# In[6]:


df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
df


# In[8]:


column = df.columns.tolist()
column = [column[-1]] + column[:-1]
column = [column[-1]] + column[:-1]
column = [column[-1]] + column[:-1]
column = [column[-1]] + column[:-1]
column = [column[-1]] + column[:-1]
column = [column[-1]] + column[:-1]
df = df[column]
df.iloc[:5, :]


# In[9]:


df.Sex[df.Sex == 'female'] = 0
df.Sex[df.Sex == 'male'] = 1
df["Sex"] = df["Sex"].astype(str).astype(int)
df.iloc[:5,:]


# In[12]:


df.dtypes


# In[45]:


counts = df['Survived'].value_counts()
w = .35
plt.bar(x = [0, .4], height = counts, width = w)
plt.xticks([0, .4], ("Perished", "Survived"))


# In[32]:


average = df.mean(axis = 0)
mean = df[["Survived", "Pclass", "Sex" , "Age", "SibSp", "Parch", "Fare"]].mean()
average


# In[35]:


group = df.groupby(df['Survived']).mean()
groupT = group.T
groupT["All Passenger Avg" ] = mean
groupT


# In[37]:


plt.plot(groupT['All Passenger Avg'], label='All Passenger Avg')
plt.plot(groupT[0], label='Perished')
plt.plot(groupT[1], label='Survived')
plt.title("Features Averages")
plt.xlabel("Features")
plt.ylabel("Averages")
plt.legend()
plt.show()


# In[51]:


df1 = groupT.iloc[0]
df1 = pd.DataFrame(df1)
df1 = df1.T
dfg2 = groupT.iloc[1]
dfg2 = pd.DataFrame(dfg2)
dfg2 = dfg2.T
df3 = groupT.iloc[2]
df3 = pd.DataFrame(df3)
df3 = df3.T
df4 = groupT.iloc[3]
df4 = pd.DataFrame(df4)
df4 = df4.T
df5 = groupT.iloc[4]
df5 = pd.DataFrame(df5)
df5 = df5.T
df6 = groupT.iloc[5]
df6 = pd.DataFrame(df6)
df6 = df6.T


# In[63]:


df_list = [df1, dfg2, df3, df4, df5, df6]
rows =2
cols =3
fig, ax = plt.subplots (rows, cols, figsize=(12,5))

df1.plot.bar(ax=ax[0,0], title = "Ticket Class - 1st, 2nd, 3rd", width=4, legend=None, xlabel=None)
dfg2.plot.bar(ax=ax[0,1], title = "Sex - 0 female, 1 male", width=4, legend=None)
df3.plot.bar(ax=ax[0,2], title = "Age", width=4, legend=None)
df4.plot.bar(ax=ax[1,0], title = "# Siblings Abroad", width=4, legend=None)
df5.plot.bar(ax=ax[1,1], title = "# Children Abroad", width=4, legend=None)
df6.plot.bar(ax=ax[1,2], title = "Fare Price", width=4, legend=None)

fig.legend()

plt.suptitle('Feature Averages, As A Whole')
plt.show


# In[65]:


corr = dfg2.corr()
corr = corr.drop(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'], axis =1)
corr =corr.drop(index = "Survived")
corr


# In[68]:


scat = dfg2.drop(["Pclass", "Sex", "SibSp", "Parch"], axis =1)
scat
purple = scat.where(df["Survived"] == 0)
purple = purple.dropna()
purple
yellow = scat.where(df["Survived"] == 1)
yellow = yellow.dropna()
yellow


# In[69]:


ax1 = yellow.plot(kind= "scatter", x="Age", y = "Fare", color = "yellow")
purple.plot(kind= "scatter", x= "Age", y = "Fare", color= "purple", ax=ax1)
plot.show()


# In[ ]:




