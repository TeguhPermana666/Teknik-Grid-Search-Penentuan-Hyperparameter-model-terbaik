import pandas as pd
#teknik grid untuk menemukan sebuah paramter pada model yang terbaik

#load data
file_path='Dasar\Salary_Data.csv'
data=pd.read_csv(file_path)
#examine data
print(data.describe())
print(data.dtypes)
print(data)

import numpy as np
#memisahkan fitur dengan label
X=data['YearsExperience']
y=data['Salary']
print(X)
print(y)
#split data->ga usah karena columns iterasi nya cuma 2 doang

#mengubah bentuk atribut
print(X.shape)
X=X[:,np.newaxis]
print(X.shape)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
model=SVR()
parameter={
    'kernel': ['rbf'],
    'C':[1000,10000,100000],
    'gamma':[0.5,0.05,0.005]
}
grid_search=GridSearchCV(model,parameter)
grid_search=grid_search.fit(X,y)

print(grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
print(grid_search.best_index_)

new_model=SVR(C=100000,gamma=0.005,kernel='rbf')
new_model=new_model.fit(X, y)
predicts=new_model.predict(X)
print(predicts)
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X,predicts)
plt.legend()
plt.show() 