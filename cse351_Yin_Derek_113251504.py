#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set(rc = {'figure.figsize':(18,18)})
df = pd.read_csv(r'C:\Users\derek\Documents\CSE351\AB_NYC_2019.csv')
# First step is to clean the data of any duplicate rows.
df.drop_duplicates()
# Second step is to check total number of null cells in this dataframe.
# Total number of null cells is 20141
# We can also check exactly which rows contain null values. 


# The first step when cleaning this data is to drop all duplicate rows. In the case of this dataset, there are no duplicate rows, so we can then move on to other cleaning techniques. 

# In[2]:


# first create missing indicator for features with missing data
for col in df.columns:
    missing = df[col].isnull()
    num_missing = np.sum(missing)
    
    if num_missing > 0:  
        print('created missing indicator for: {}'.format(col))
        df['{}_ismissing'.format(col)] = missing


# then based on the indicator, plot the histogram of missing values
ismissing_cols = [col for col in df.columns if 'ismissing' in col]
df['num_missing'] = df[ismissing_cols].sum(axis=1)

df['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')


# Here I have created a simple bar chart to better visualize which columns contain invalid values / have no values. From this bar chart I now know that the 'reviews per month' column have missing values.

# In[3]:


df['reviews_per_month'] = df['reviews_per_month'].fillna( 0 )
# Replacing the null values in "reviews_per_month" with 0 reduced total null cells from 20141 to 10089.   
print(df.isnull().sum().sum())
df.head(30)


# Replacing the null values in "reviews_per_month" with 0 reduced total null cells from 20141 to 10089.

# In[4]:


# The other columns ('name', 'host_name,' 'last_review') that contain null values cannot be imputed. 
# This concludes the data cleaning portion.
# Now, let's take a look at the number of neighborhoods in this dataset. 
print(len(pd.unique(df['neighbourhood'])))
# There are 221 unique neighborhoods in this dataset. 


# There are 221 unique neighbourhoods in this dataset. 

# In[5]:


#This block of code prints a table that shows the different neighbourhoods. 
frequency = df['neighbourhood'].value_counts()
print("Frequency of value in column 'neighbourhood' :")
print(frequency)


# Some neighbourhoods only appear once in the dataset. We need to filter these neighbourhoods out before doing our next calculations.

# In[6]:


# df['neighbourhood'].value_counts()+
df2 = df
frequency = df2['neighbourhood'].value_counts()
for cell in df2['neighbourhood'].unique():
    if frequency[cell] < 5:
        df2 = df2[df2['neighbourhood'] != cell]

df2.groupby('neighbourhood')['price'].mean().sort_values(ascending = False)
#Filtering out all neighbourhoods with less than 5 listings, the below table shows the top 5 and bottom 5 priced neighbourhoods.
#for i in range(0,221):
#    if frequency[i] < 5:
        
#df2.drop(df2[(frequency[df2.neighbourhood] < 5)])
# frequency[]
# df.groupby('neighbourhood')['price'].mean().sort_values(ascending = False)
# grouped.filter(lambda x: x['neighbourhood'].count() >= 5)
# grouped['price'].mean().sort_values(ascending = False)

# df.groupby('neighbourhood').filter(lambda x: x.value_counts() >=5)['price'].mean().sort_values(ascending = False)


# As shown in the table above, the Top 5 neighbourhoods in regards to mean price are Tribeca, Sea Gate, Riverdale, Battery Park City, and Flatiron District. The Bottom 5 neighbourhoods are Bull's Head, Hunts Point, Tremont, Soundview, and Bronxdale. 

# In[7]:


# Now that we have the top 5 and bottom 5 neighbourhoods based on price, we can plot the overall price variation among neighbourhood groups.
df2.groupby('neighbourhood_group')['price'].mean().plot.bar(ylabel = 'Average price')
# As you can see from this graph, Manhattan has by far the largest mean price for an Airbnb.


# When grouping by neighbourhood group and creating a simple bar graph, one can see that Manhattan has by far the highest mean price for AirBnbs.

# In[8]:


# Now, heatmap and correlation coefficient of all interesting data sets.
# As you can see below, there are no strong correlations between any of the data points. 
dataplot=sns.heatmap(df[['price', 'minimum_nights', 'number_of_reviews', 'availability_365', 'calculated_host_listings_count', 'latitude', 'longitude']].corr(method='pearson'), vmin = -1)


# This heatmap shows the Pearson correlation of specific variables. Observe how there are no strong correlations whatsover in the heatmap. 

# In[9]:


sns.scatterplot(x = 'longitude', y = 'latitude', data = df, hue = 'neighbourhood_group')
# The below scatterplot shows a color-coded grid of AirBnb listings in the NYC area based on borough.


# By creating a color-coded scatter plot of all AirBnb locations in the dataset, one can see the distribution of AirBnbs in each borough.

# In[10]:


df3 = df[df['price'] < 1000]
sns.scatterplot(x = 'longitude', y = 'latitude', data = df3, hue = 'price', palette = 'OrRd', size = 'price', hue_norm = (0, 999))
# Below is the scatterplot of all AirBnb locations, color coded to price to identify where the priciest locations are. 
# The priciest locations are in Manhattan.


# This scatterplot again plots the locations of all AirBnbs in the dataset, but instead of coloring by borough, it is colored according to the price of the room. Observing the graph, Manhattan has the densest clusters of highly priced rooms compared to other boroughs.

# In[11]:


# Now, we create a word cloud based on the words found in the Airbnb listings. 
from wordcloud import WordCloud
import matplotlib as mpl
words = df['name'].values
wordcloud = WordCloud().generate(str(words))
mpl.pyplot.imshow(wordcloud)
# As you can see from this word cloud, the most common words found in an Airbnb name include 'Clean' and 'quiet', which are definitely desirable attributes for a property to have.


# This word cloud contains the most frequently used words in the description of each AirBnb. By far the most frequent words include Clean and Quiet.

# In[12]:


df.groupby('neighbourhood_group')['calculated_host_listings_count'].mean().plot.bar(ylabel = '# Host listings')
# From this bar plot, we can see that the average # host listings is highest in Manhattan and around the same for the Bronx, Brooklyn, and Staten Island.


# From this bar plot, we can see that the average # host listings is highest in Manhattan and around the same for the Bronx, Brooklyn, and Staten Island.

# In[13]:


dataplot=sns.heatmap(df[['price', 'number_of_reviews', 'availability_365', 'calculated_host_listings_count']].corr(method='pearson'), vmin = -1).set(title = "Correlation heatmap")

# This heatmap shows there is no strong correlation between calculated host listings count and any other variable. 


# This correlation coefficient heatmap again shows the weak correlation among all variables. 

# In[14]:



sns.boxplot(x='room_type', y = 'availability_365', data = df).set(title = "Types of Rooms and Availability")
# This boxplot shows how the availability of a room is dependent on the type of room. As you can see, shared rooms have a higher median and 3rd quartile of availability.


# This boxplot is used to show the dependence of the availability of a room on the type of room. Shared rooms have a higher median and 3rd quartile of available days.

# In[15]:


df['room_type'].value_counts().plot.pie(title = 'Types of Rooms')
# This pie char shows the distribution of types of rooms. Shared rooms are by far the least frequent listing. 


# This pie chart shows the distrbution of frequency of types of rooms. Shared rooms are by far the least frequent listings. 
