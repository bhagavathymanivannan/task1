#!/usr/bin/env python
# coding: utf-8

# In[1]:



#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


# In[2]:


url="http://bit.ly/w-data"
data=pd.read_csv(url)
print("data imported sucessfully")


# In[3]:


data.head(10)


# In[4]:


data.shape


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data.Hours,data.Scores)
plt.title('HOURS VS MARKS',size=20)
plt.xlabel('hours studied',size=15)
plt.ylabel('marks scored',size=15)


# In[6]:


data.plot(kind='box')
plt.show()


# In[92]:


data.corr()


# In[93]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# In[94]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[95]:


line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# # making predictions

# In[96]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[97]:


#comparing the actual and predicted
df = pd.DataFrame({'Actual Marks': y_test, 'Predicted Marks': y_pred}) 
df


# In[98]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(x=y_pred,y=y_test)
plt.title('ACTUAL MARKS VS PREDICTED MARKS ',size=20)


# # Evaluating the model

# In[75]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# # ----------------------------------------------END----------------------------------------------------------

# In[ ]:




