#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost as xgb
import pandas as pd
import numpy as np
from geopy.distance import geodesic 
import math
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_excel("alakv-6.xlsx")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df['wallmaterial'] = df['wallmaterial'].str.replace('\d+', '')


# In[6]:


df.wallmaterial.unique()


# In[7]:


df['wallmaterial'] = df['wallmaterial'].replace('', np.nan)


# In[8]:


df.wallmaterial.unique()


# In[9]:


df.wallmaterial = df.wallmaterial.fillna('monolith')


# In[10]:


df.wallmaterial.unique()


# In[11]:


df.wallmaterial.describe(include='all')


# In[12]:


df["wallmaterial"].value_counts()


# In[13]:


df.info()


# In[ ]:





# In[ ]:





# In[14]:


df["floorsTotal"].value_counts()


# In[15]:


df.floorNumber.describe(include='all')


# In[16]:


df.floorNumber = df.floorNumber.fillna('4')


# In[17]:


df.floorsTotal.describe(include='all')


# In[18]:


df.floorsTotal = df.floorsTotal.fillna('7')


# In[19]:


df["state"].value_counts()


# In[20]:


df.state.describe(include='all')


# In[21]:


df.info()


# In[22]:


df["totalArea"].value_counts()


# In[23]:


df.totalArea = df.totalArea.fillna('60')


# In[24]:


df.totalArea.describe(include='all')


# In[25]:


df.info()


# In[ ]:





# In[26]:


df['floorNumber'] = df['floorNumber'].astype(float)
df['floorsTotal'] = df['floorsTotal'].astype(float)


# In[27]:


df.info()


# In[28]:


df['latitude'].describe()


# In[29]:


df['longitude'].describe()


# In[ ]:





# In[30]:


df['price'].describe()


# In[31]:


df.year.describe()


# In[32]:


df = df.dropna()


# In[33]:


df.info()


# In[34]:


def get_azimuth(latitude, longitude):
 
    rad = 6372795

    llat1 = city_center_coordinates[0]
    llong1 = city_center_coordinates[1]
    llat2 = latitude
    llong2 = longitude

    lat1 = llat1*math.pi/180.
    lat2 = llat2*math.pi/180.
    long1 = llong1*math.pi/180.
    long2 = llong2*math.pi/180.

    cl1 = math.cos(lat1)
    cl2 = math.cos(lat2)
    sl1 = math.sin(lat1)
    sl2 = math.sin(lat2)
    delta = long2 - long1
    cdelta = math.cos(delta)
    sdelta = math.sin(delta)

    y = math.sqrt(math.pow(cl2*sdelta,2)+math.pow(cl1*sl2-sl1*cl2*cdelta,2))
    x = sl1*sl2+cl1*cl2*cdelta
    ad = math.atan2(y,x)

    x = (cl1*sl2) - (sl1*cl2*cdelta)
    y = sdelta*cl2
    z = math.degrees(math.atan(-y/x))

    if (x < 0):
        z = z+180.

    z2 = (z+180.) % 360. - 180.
    z2 = - math.radians(z2)
    anglerad2 = z2 - ((2*math.pi)*math.floor((z2/(2*math.pi))) )
    angledeg = (anglerad2*180.)/math.pi
    
    return round(angledeg, 2)


# In[35]:


df['priceMetr'] = df['price']/df['totalArea']


# In[36]:


city_center_coordinates = [43.238293, 76.945465]
df['distance'] = list(map(lambda x, y: geodesic(city_center_coordinates, [x, y]).meters, df['latitude'], df['longitude']))
df['azimuth'] = list(map(lambda x, y: get_azimuth(x, y), df['latitude'], df['longitude']))


# In[37]:


df.info()


# In[38]:


df["distance"].max()


# In[39]:


df.head(3)


# In[40]:


categorical_columns = df.columns[df.dtypes == 'object']
labelencoder = LabelEncoder()
for column in categorical_columns:
    df[column] = labelencoder.fit_transform(df[column])
    print(dict(enumerate(labelencoder.classes_)))


# In[41]:


import seaborn as sns


# In[42]:


df.head()


# In[43]:


sns.countplot('wallmaterial',data=df)
plt.show()


# In[44]:


plt.figure(figsize=(100,40))
sns.countplot('year',data=df)
plt.show()


# In[ ]:





# In[45]:


sns.countplot('state',data=df)
plt.show()


# In[46]:


plt.figure(figsize=(100,40))
sns.countplot('year',data=df)
plt.show()


# In[98]:


sns.regplot(x=df["totalArea"], y=df["price"], data=df);


# In[47]:


sns.heatmap(df.corr())


# In[48]:


y = df['priceMetr']


# In[49]:


features = [
            'wallmaterial', 
            'floorNumber', 
            'floorsTotal',
            'state',
            'totalArea',
            'year',
            'distance',
            'azimuth'
           ]


# In[50]:


X = df[features]


# In[51]:


models = pd.DataFrame(columns=["Model","Avg MeanAE", "Median AE", "MSE","RMSE"])


# In[52]:


train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)


# In[53]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Calculates median absolute error in %
def median_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

def mean_squared_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    differences = np.subtract(y_true, y_pred)
    squared_differences = np.square(differences)
    return squared_differences.mean()

def root_mean_squared_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    differences = np.subtract(y_true, y_pred)
    squared_differences = np.square(differences)
    from math import sqrt
    rmse_score = sqrt(squared_differences.mean())
    return rmse_score

def print_metrics(prediction, val_y):
    val_mae = mean_absolute_percentage_error(val_y, prediction)
    median_AE = median_absolute_percentage_error(val_y, prediction)
    mse_score = mean_squared_error(val_y, prediction)
    rmse_score_res = root_mean_squared_error(val_y, prediction)
    return val_mae, median_AE, mse_score, rmse_score_res


# In[54]:


rf_model = RandomForestRegressor(n_estimators=2000, 
                                 n_jobs=-1,  
                                 bootstrap=False,
                                 criterion='mse',
                                 max_features=3,
                                 random_state=1,
                                 max_depth=55,
                                 min_samples_split=2
                                 )


# In[55]:


rf_model.fit(train_X, train_y)


# In[56]:


rf_prediction = rf_model.predict(val_X).round(0)


# In[57]:


val_mae, median_AE, mse_score, rmse_score_res = print_metrics(rf_prediction, val_y)

print('Avg mean absolute error:', round(val_mae,2),"%")
print('Median Absolute error:', round(median_AE,2),"%")
print('Mean squared error: ',mse_score)
print('Root mean squared error: ', rmse_score_res)

new_row = {"Model": "RandomForestRegressor", "Avg MeanAE": round(val_mae,2), "Median AE": round(median_AE,2), "MSE": mse_score, "RMSE": rmse_score_res}
models = models.append(new_row, ignore_index=True)


# In[58]:


xgb_model = xgb.XGBRegressor(objective ='reg:gamma', 
                             learning_rate = 0.01,
                             max_depth = 45, 
                             n_estimators = 2000,
                             nthread = -1,
                             eval_metric = 'gamma-nloglik', 
                             )


# In[59]:


xgb_model.fit(train_X, train_y)


# In[60]:


xgb_prediction = xgb_model.predict(val_X).round(0)


# In[61]:


val_mae, median_AE, mse_score, rmse_score_res = print_metrics(xgb_prediction, val_y)

print('Avg mean absolute error:', round(val_mae,2),"%")
print('Median Absolute error:', round(median_AE,2),"%")
print('Mean squared error: ',mse_score)
print('Root mean squared error: ', rmse_score_res)

new_row = {"Model": "XGBRegressor", "Avg MeanAE": round(val_mae,2), "Median AE": round(median_AE,2), "MSE": mse_score, "RMSE": rmse_score_res}
models = models.append(new_row, ignore_index=True)


# In[62]:


from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()


# In[63]:


lr_model.fit(train_X, train_y)


# In[64]:


lr_prediction = lr_model.predict(val_X).round(0)


# In[65]:


val_mae, median_AE, mse_score, rmse_score_res = print_metrics(lr_prediction, val_y)

print('Avg mean absolute error:', round(val_mae,2),"%")
print('Median Absolute error:', round(median_AE,2),"%")
print('Mean squared error: ',mse_score)
print('Root mean squared error: ', rmse_score_res)

new_row = {"Model": "LinearRegression", "Avg MeanAE": round(val_mae,2), "Median AE": round(median_AE,2), "MSE": mse_score, "RMSE": rmse_score_res}
models = models.append(new_row, ignore_index=True)


# In[66]:


from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()


# In[67]:


dt_model.fit(train_X, train_y)


# In[68]:


dt_prediction = dt_model.predict(val_X).round(0)


# In[69]:


val_mae, median_AE, mse_score, rmse_score_res = print_metrics(dt_prediction, val_y)

print('Avg mean absolute error:', round(val_mae,2),"%")
print('Median Absolute error:', round(median_AE,2),"%")
print('Mean squared error: ',mse_score)
print('Root mean squared error: ', rmse_score_res)

new_row = {"Model": "DecisionTreeRegressor", "Avg MeanAE": round(val_mae,2), "Median AE": round(median_AE,2), "MSE": mse_score, "RMSE": rmse_score_res}
models = models.append(new_row, ignore_index=True)


# In[70]:


from sklearn.linear_model import Ridge
ridge = Ridge()


# In[71]:


ridge.fit(train_X, train_y)


# In[72]:


ridge_prediction = ridge.predict(val_X).round(0)


# In[73]:


val_mae, median_AE, mse_score, rmse_score_res = print_metrics(ridge_prediction, val_y)

print('Avg mean absolute error:', round(val_mae,2),"%")
print('Median Absolute error:', round(median_AE,2),"%")
print('Mean squared error: ',mse_score)
print('Root mean squared error: ', rmse_score_res)

new_row = {"Model": "Ridge", "Avg MeanAE": round(val_mae,2), "Median AE": round(median_AE,2), "MSE": mse_score, "RMSE": rmse_score_res}
models = models.append(new_row, ignore_index=True)


# In[74]:


from sklearn.linear_model import Lasso
lasso = Lasso()


# In[75]:


lasso.fit(train_X, train_y)


# In[76]:


lasso_prediction = lasso.predict(val_X).round(0)


# In[77]:


val_mae, median_AE, mse_score, rmse_score_res = print_metrics(lasso_prediction, val_y)

print('Avg mean absolute error:', round(val_mae,2),"%")
print('Median Absolute error:', round(median_AE,2),"%")
print('Mean squared error: ',mse_score)
print('Root mean squared error: ', rmse_score_res)

new_row = {"Model": "Lasso", "Avg MeanAE": round(val_mae,2), "Median AE": round(median_AE,2), "MSE": mse_score, "RMSE": rmse_score_res}
models = models.append(new_row, ignore_index=True)


# In[78]:


# Creating a Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow


# In[79]:


from keras import backend as K

def root_mean_squared_error_keras(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


# In[80]:


model = Sequential()
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop',loss=root_mean_squared_error_keras)


# In[81]:


model.fit(x=train_X,y=train_y,
          validation_data=(val_X,val_y),
          batch_size=128,epochs=400)


# In[82]:


neuron_prediction = model.predict(val_X).round(0)


# In[83]:


val_mae, median_AE, mse_score, rmse_score_res = print_metrics(neuron_prediction, val_y)

print('Avg mean absolute error:', round(val_mae,2),"%")
print('Median Absolute error:', round(median_AE,2),"%")
print('Mean squared error: ',mse_score)
print('Root mean squared error: ', rmse_score_res)

new_row = {"Model": "Neural Network", "Avg MeanAE": round(val_mae,2), "Median AE": round(median_AE,2), "MSE": mse_score, "RMSE": rmse_score_res}
models = models.append(new_row, ignore_index=True)


# In[84]:


models.sort_values(by="RMSE")


# In[85]:


importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


print("Ranking of important features:")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

plt.figure()
plt.title("Importance ranking")
plt.bar(range(X.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[88]:


flat = pd.DataFrame({
                     'wallmaterial':[1], 
                     'floorNumber':[5],
                     'floorsTotal':[6],
                    'state':[2],
                     'totalArea':[45],
                    'year':[2014],
                     'latitude':[43.198874],
                     'longitude':[76.868674]
                     })


# In[89]:


flat['distance'] = list(map(lambda x, y: geodesic(city_center_coordinates, [x, y]).meters, flat['latitude'], flat['longitude']))
flat['azimuth'] = list(map(lambda x, y: get_azimuth(x, y), flat['latitude'], flat['longitude']))
flat['distance'] = flat['distance'].round(0)
flat['azimuth'] = flat['azimuth'].round(0)


# In[90]:


flat = flat.drop('latitude', axis=1)
flat = flat.drop('longitude', axis=1)


# In[ ]:





# In[91]:


rf_prediction_flat = rf_model.predict(flat).round(0)


# In[92]:


flat


# In[93]:


price = rf_prediction_flat *flat['totalArea'][0]


# In[94]:


print(f'Predicted prices of apartments: {int(price[0].round(-3))} tenge')


# In[95]:


import joblib


# joblib.dump(rf_model, 'rf_model.pkl')

# In[ ]:




