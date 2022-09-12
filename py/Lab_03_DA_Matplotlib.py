#!/usr/bin/env python
# coding: utf-8

# # Matplotlib

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")


# In[5]:


def create_random_data(N=1000):
    data1 = np.random.normal(0, 0.1, N)
    data2 = np.random.normal(1, 0.4, N) + np.linspace(0, 1, N)
    data3 = 2 + (np.random.random(N) * np.linspace(1, 5, N))
    data4 = np.random.normal(3, 0.2, N) + 0.3 * np.sin(np.linspace(0, 20, N))

    data = np.vstack([data1, data2, data3, data4])
    data = data.transpose()

    cols = ["data1", "data2", "data3", "data4"]

    df = pd.DataFrame(data, columns=cols)
    return df


# In[6]:


df = create_random_data()


# In[7]:


df.plot(title="Line plot")


# In[8]:


df.plot(style=".", title="Scatter Plot")


# In[9]:


df.plot(kind="hist", bins=50, title="Histogram", alpha=0.6)


# In[10]:


df.plot(
    kind="hist",
    bins=100,
    title="Cumulative distributions",
    density=True,
    cumulative=True,
    alpha=0.4,
)


# In[11]:


df.plot(kind="box", title="Boxplot")


# In[12]:


fig, ax = plt.subplots(2, 2, figsize=(16, 12))

df.plot(ax=ax[0][0], title="Line plot")

df.plot(ax=ax[0][1], style="o", title="Scatter plot")

df.plot(ax=ax[1][0], kind="hist", bins=50, title="Histogram")

df.plot(ax=ax[1][1], kind="box", title="Boxplot")


# In[13]:


categories = df["data1"] > 0.01


# In[14]:


counts = categories.value_counts()
counts


# In[15]:


counts.plot(
    kind="pie",
    figsize=(5, 5),
    explode=[0, 0.15],
    labels=["<= 0.1", "> 0.1"],
    autopct="%1.1f%%",
    shadow=True,
    startangle=90,
    fontsize=16,
)


# ## Exercises:
#
# ### Exercise 1
# - load the dataset: ../data/international-airline-passengers.csv
# - inspect it using the .info() and .head() commands
# - use the function pd.to_datetime() to change the column type of 'Month' to a datatime type
# - set the index of df to be a datetime index using the column 'Month' and the df.set_index() method
# - choose the appropriate plot and display the data
# - choose appropriate scale
# - label the axes
# - discuss with your neighbor

# In[ ]:


# ### Exercise 2
# - load the dataset: ../data/weight-height.csv
# - inspect it
# - plot it using a scatter plot with Weight as a function of Height
# - plot the male and female populations with 2 different colors on a new scatter plot
# - remember to label the axes
# - discuss
#

# In[ ]:


# ### Exercise 3
# - plot the histogram of the heights for males and for females on the same plot
# - use alpha to control transparency in the plot comand
# - plot a vertical line at the mean of each population using plt.axvline()
#

# In[ ]:
