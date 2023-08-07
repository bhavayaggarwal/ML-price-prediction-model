#!/usr/bin/env python
# coding: utf-8

# In[157]:


import pandas as pd


# In[158]:


housing = pd.read_csv("housingdata.csv")


# In[159]:


housing.head()


# In[160]:


housing['CHAS']=housing['CHAS'].apply(lambda x: 0 if x<1 else 1)


# In[161]:


housing['CHAS'].value_counts()


# In[162]:


from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)
housing['MEDV'] = knn_imputer.fit_transform(housing[['MEDV']])


# In[163]:


housing.info()


# In[164]:


housing.info()


# In[165]:


housing.describe()


# In[166]:


housing['CRIM'][1]



# In[167]:


housing['CRIM'][1]


# In[168]:


housing.info()


# In[169]:


import matplotlib as plt



# In[170]:


housing.hist(column=["MEDV"],bins=50,figsize=(20,20),grid=True,color="purple" , alpha=1)


# ## Train test Splitting
# 

# In[171]:


import numpy as np
def split_train_test(data,test_ratio):
    np.random.seed(42)

    shuffled=np.random.permutation(len(data))
    print(shuffled)
    test_set_size=int(len(data) * test_ratio)
    test_indicies=shuffled[:test_set_size]
    train_indicies=shuffled[test_set_size:]
    return data.iloc[train_indicies],data.iloc[test_indicies]


# In[172]:


train_set,test_set=split_train_test(housing,0.2)


# In[173]:


print(f"Rows in train set is: {len(train_set)}\n rows in test set: {len(test_set)}\n")


# In[174]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["CHAS"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[175]:


strat_train_set['CHAS'].value_counts()



# In[176]:


strat_test_set['CHAS'].value_counts()


# In[177]:


#we have to take a copy now by 
housing=strat_train_set.copy()
housing_test=strat_test_set.copy()
#but as the set is already same so for edu purpose im not taking it


# ## LOOKING FOR CORRELATIONS
# 

# In[178]:


corr_matrix=housing.corr()


# In[179]:


corr_matrix['MEDV'].sort_values()


# In[180]:


from pandas.plotting import scatter_matrix
attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes])


# In[181]:


corr_mat=housing.corr()
corr_mat['MEDV']


# In[182]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.9)


# In[183]:


housing.plot(kind="scatter",x="CRIM",y="MEDV")


# In[184]:


housing["RM"].info()


# ##MISSING ATTRIBUTES

# In[185]:


housing["RM"].info()


# ### NOW WE See that 497 are the non null values in RM

# In[186]:


# seprsting featues and label (axis=1 means along column)
housing=strat_train_set.drop("MEDV",axis =1)
housing_labels=strat_train_set["MEDV"].copy()


# ## MISSING ATTRIBUTES

# WE have options
# 1. Get rid of the missing data points
# 2. Get rid of the whole attribute
# 3. Set value to mean,median or zero

# In[187]:


# 1 st option
a=housing.dropna(subset=["RM"])
a.shape


# In[188]:


housing.drop("RM",axis=1) #option 2



# In[189]:


#option 3
median=housing["RM"].median()
housing["RM"].fillna(median)


# In[190]:


housing.info()  ## before imputer
# as we can see we havent applied any changes to the orginal data


# In[191]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[192]:


imputer.statistics_


# In[193]:


X=imputer.transform(housing)
housing_tr=pd.DataFrame(X,columns=housing.columns)


# In[194]:


housing_tr.describe()


# In[195]:


housing=housing_tr.copy()


# In[196]:


housing.describe()


# # SCIKIT LEARN DESCRIBE

# ## Primarily 3 types of objects
# #### 1. Estimators- estimates some parameter based on dataset. eg - imputer
# #### It has fit method and transform method.
# #### 2. Transforms- tranforms method takes input and returs output based on the learnings from fit(). It also has a convience function fit_tranfrom() which fits and then transforms. like we did in 146th code block.
# #### 3. Predictors- Linear regression model is its example. fit() and predict() are two common functions.It also gives score functions that evalute your predictions.

# ## Feature Scaling (ALL FEATURES IN SAME RANGE)
# #### primarily 2 types of feature scaling method 
# 1. Min - Max Scaling (Normalization) = (value-min)/(max-min) -> goes to 0 to 1
#    Sklearn provides a class called MinMaxScaler for this
# 2. Standardization- (value-mean)/stadard deviation -> variance-1 and mean is mean only
#    Sklearn provided a class called Standard Scalar for this
#    Unlike normalization, standardization is less affected by outliers because it uses the    mean and standard deviation, which are more robust statistics.

# # CREATING PIPELINE

# In[197]:


from sklearn.pipeline import Pipeline


# In[198]:


from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([("imputer",SimpleImputer(strategy="median")),('std_scalar',StandardScaler())])
    


# In[199]:


housing_num_tr=my_pipeline.fit_transform(housing_tr)


# In[200]:


housing_num_tr.shape
## its now array as we are now using predictors 


# ## Selecting a desired model for the problem

# In[201]:


# from sklearn.linear_model import LinearRegression
# model=LinearRegression() #(ye bekar tha)
# from sklearn.tree import DecisionTreeRegressor 
#overfit kar diya bhai
# model=DecisionTreeRegressor()
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels) ## model train kar diya 


# In[202]:


some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]


# In[203]:


prepared_data=my_pipeline.transform(some_data) ## data ko model me dalne ke liye prepare kar diya
model.predict(prepared_data)


# In[204]:


list(some_labels)


# ## Evaluating the model

# In[205]:


from sklearn.metrics import mean_squared_error


# 

# In[206]:


housing_prediction=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_prediction)
rmse=np.sqrt(mse)



# In[207]:


rmse


# ## ye toh overfit ho gya hai bhai

# In[208]:


#using better evalution technique- Cross Validation
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[209]:


rmse_scores #( ye error hai total)


# In[210]:


##still good
def print_scores(scores):
    print("scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())


# In[211]:


print_scores(rmse_scores)


#  ## saving the model 

# In[212]:


from joblib import dump,load
dump(model,'Housemodel.joblib')


# ## testing the model on test data

# In[213]:


X_test=strat_test_set.drop("MEDV",axis=1)


# In[214]:


Y_test=strat_test_set["MEDV"].copy()


# In[215]:


X_test_prepared=my_pipeline.transform(X_test)


# In[216]:


final_predictions=model.predict(X_test_prepared)


# In[217]:


final_mse=mean_squared_error(Y_test,final_predictions)


# In[218]:


final_rmse=np.sqrt(final_mse)


# In[219]:


final_rmse


# In[221]:


import matplotlib.pyplot as plt
plt.hist(Y_test, bins=10, alpha=0.5, label='Y_test')
plt.hist(final_predictions, bins=10, alpha=0.5, label='Predictions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram Comparison')
plt.legend()
plt.grid(True)  # Optional: Add grid lines
plt.show()


# In[222]:


print(list(final_predictions),list(Y_test))


# ## finding the final accuracy

# In[223]:


final_rmse=np.sqrt(final_mse)


# In[224]:


final_rmse


# In[225]:


prepared_data[0]


# In[226]:


corr_mat["MEDV"]


# In[228]:


housing["LSTAT"].describe()


# In[ ]:




