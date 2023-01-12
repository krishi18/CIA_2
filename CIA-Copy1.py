#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Generating 1000 points and building Linear Regression Model

# In[32]:


rp1=[]
for i in range(500):
    rp1.append(random.randint(0,300))
rp1


# In[33]:


rp2=[]
for i in range(500):
    rp2.append(random.randint(0,300))
rp2


# In[46]:


df1=pd.DataFrame()
df1['x']=rp1
df1['y']=rp2
df1


# In[34]:


plt.scatter(rp1,rp2, c ="blue")


# In[47]:


x = df1["x"].values
y= df1["y"].values
mean_x = np.mean(x)
mean_y = np.mean(y)
print(mean_x,mean_y)


# In[48]:


m = len(x)
# calculating m & c
numer = 0
denom = 0
for i in range(m):
  numer += (x[i] - mean_x) * (y[i] - mean_y)
  denom += (x[i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)
print(m,c)


# In[52]:


x_max = np.max(x) + 100
x_min = np.min(x) - 100


# In[66]:


X = np.linspace(x_min, x_max, 1000)
Y = (m*x)+c


# In[73]:


#plotting line 
plt.plot(X, Y, color='red', label='Linear Regression')
#plot the data points
plt.scatter(rp1,rp2, color='blue', label='Data Points')
plt.legend()
plt.show()


# In[74]:


mean_pred=np.mean(Y)
mean_pred


# In[75]:


diff=0
for i in range(len(y)):  
    diff += (y[i]-mean_pred)**2


# In[86]:


mse=diff/len(y)
mse


# In[81]:


rmse=math.sqrt(diff/len(y))
rmse


# In[82]:


diff1=0
for i in range(len(y)):  
    diff1 += np.abs(y[i]-mean_pred)


# In[83]:


mae=diff1/len(y)
mae


# ## Gradient Descend (without sklearn)

# In[ ]:


m = 0
c = 0

L = 0.00001 
epochs = 10000  

n = float(len(x)) 

# Performing Gradient Descent 
for i in range(epochs): 
    y_pred = m*x + c  
    D_m = (-2/n) * sum(x * (y - mean_pred)) 
    D_c = (-2/n) * sum(y - mean_pred) 
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
y_pred = m*x + c   
print ("Slope2 : ",m)
print("Intercept2 : ",c)


# ## Using sklearn

# In[140]:


x=df1['x']
y=df1['y']
from sklearn import linear_model,metrics,model_selection
model=linear_model.LinearRegression(normalize=True)
x=np.expand_dims(x,1)
model.fit(x,y)


# In[141]:


slope=model.coef_
slope


# In[142]:


intercept=model.intercept_
intercept


# In[143]:


y1=slope*x + intercept


# In[145]:


plt.scatter(x,y)
plt.plot(x,y1,c='red')
plt.show()


# In[ ]:




