#!/usr/bin/env python
# coding: utf-8

# # 911 Calls EDA

# We will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 
# 

# ## Data and Setup

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df= pd.read_csv('911.csv')


# In[5]:


df.info()


# In[6]:


df.head()


# ## Basic Questions

# ** What are the top 5 zip codes from which 911 calls were made **

# In[8]:


df['zip'].value_counts().head(5)


# ** What are the top 5 townships (twp) for 911 calls? **

# In[9]:


df['twp'].value_counts().head(5)


# ** Take a look at the 'title' column, how many unique title codes are there? **

# In[10]:


df['title'].nunique()


# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Using .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[11]:


df['Reasons']=df['title'].apply(lambda x: x.split(':')[0])


# ** What is the most common Reason for a 911 call based off of this new column? **

# In[12]:


df['Reasons'].value_counts()


# ** Now using seaborn to create a countplot of 911 calls by Reason. **

# In[14]:


sns.countplot(x='Reasons',data=df)


# ___
# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[18]:


type(df['timeStamp'].iloc[0])


# ** You should have seen that these timestamps are still strings. Using [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **

# In[19]:


df['timeStamp']= pd.to_datetime(df['timeStamp'])


# ** You can now grab specific attributes from a Datetime object by calling them. For example:**
# 
#     time = df['timeStamp'].iloc[0]
#     time.hour
# 
# **You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.**

# In[20]:


df['hour']= df['timeStamp'].apply(lambda time: time.hour)
df['month']= df['timeStamp'].apply(lambda time: time.month)
df['dayofweek']= df['timeStamp'].apply(lambda time: time.dayofweek)


# ** Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: **
# 
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[22]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['dayofweek']= df['dayofweek'].map(dmap)


# In[144]:





# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[24]:


sns.countplot(x='dayofweek', data=df, hue='Reasons')
plt.legend(loc='best')


# **Now do the same for Month:**

# In[25]:


sns.countplot(x='month',data=df, hue='Reasons', palette='viridis')


# **Did you notice something strange about the Plot?**
# 
# _____
# 
# ** You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas... **

# ** Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame. **

# In[27]:


by_month= df.groupby('month').count()
by_month.head()


# ** Now create a simple plot off of the dataframe indicating the count of calls per month. **

# In[28]:


by_month['twp'].plot()


# ** Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **

# In[30]:


sns.lmplot(x='month',y='twp', data=by_month.reset_index())


# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# In[31]:


df['date']= df['timeStamp'].apply(lambda t: t.date())


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[35]:


df.groupby('date').count()['twp'].plot()
plt.tight_layout()


# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[37]:


df[df['Reasons']=='Traffic'].groupby('date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[38]:


df[df['Reasons']=='Fire'].groupby('date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[39]:


df[df['Reasons']=='EMS'].groupby('date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# ____
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**

# In[41]:


dayHour = df.groupby(by=['dayofweek','hour']).count()['Reasons'].unstack()
dayHour.head()


# ** Now create a HeatMap using this new DataFrame. **

# In[43]:


plt.plot(figsize=(12,3))
sns.heatmap(dayHour, cmap='viridis')


# ** Now create a clustermap using this DataFrame. **

# In[45]:


sns.clustermap(dayHour, cmap='viridis')


# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[46]:


dayMonth = df.groupby(by=['dayofweek','month']).count()['Reasons'].unstack()
dayMonth.head()


# In[49]:


plt.plot(figsize=(12,3))
sns.heatmap(dayMonth)


# In[50]:


sns.clustermap(dayMonth)


# In[ ]:




