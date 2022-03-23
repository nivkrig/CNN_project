###file starts with importing the data and libraries
```
import matplotlib
import math
import pandas as pd
import numpy as np
```
```
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits import mplot3d
%matplotlib inline
```
```
dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')

dataset.head()
```
```
from openpyxl import load_workbook

print(dataset[['DEATH_EVENT']])
dataset.info()
dataset.describe()
dataset['age'].plot.hist(bins=30)
```

```
import math
def sigmoid(in_z):
    in_z = np.longdouble(in_z)
    out_z = 1/(1 + np.exp(-in_z))
    return out_z
```
####initilalzie the variables that I will use for the mathematical calculation 
```
theta0 = []
theta1 = []
theta2 = []
cost = []
dxcost = 1
one_array = np.ones(299)
Features = ['time','ejection_fraction']
x = dataset[Features]
x = np.column_stack((one_array, x))

y = dataset["DEATH_EVENT"]
run = 0
params = np.array([1, .05, .05])

np.seterr(divide = 'ignore') 
np.log(0)
```
#### I initialize the theta valuables again because they were not reseting in value as I ran the code
#### the second loop results in inreasonable values aswell as a warning for the np.exp() function
```
theta0 = []
theta1 = []
theta2 = []
cost = []
run = 0
params = np.array([1, .05, .05])
sigma = np.array([-.1, .005, .007])
while sigma.any() != 0:
    z = x.dot(params)
    print("z:", z)
    g_z = np.array(sigmoid(z))
    print(g_z)
    I_xy = np.zeros(np.shape(y))
    for i in range(len(y)):
        gz = g_z[i]
        gz2 = g_z[i]
        if g_z[i] == 0:
            gz = 1
        if g_z[i] == 1:
            gz2= 0
        I_xy[i] = (y[i]*np.log(gz)) + ((1-y[i])*np.log(1-gz2))
    gradarry = np.zeros((len(y), len(g_z)))
    
    cost.append(-1*(sum(I_xy)/len(y))) 
    for j in range(len(g_z)):
        ech_g = np.full(np.shape(y), g_z[j])
        gradarry[:,[j]] = ech_g[j] - y[j]
    
    sigma = np.sum(np.transpose(x).dot(gradarry), axis = 1)
    
    sigma = sigma/299
    
    print("dx cost", sigma)
    params = params -  sigma
    print("param", params)
    theta0.append(params[0]) 
    theta1.append(params[1]) 
    theta2.append(params[2]) 
    run = run + 1
    if run > 11:
        break
print(run)
```


###Sklean version
```
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
```
```
Features = ['time','ejection_fraction']
x = dataset[Features]
y = dataset["DEATH_EVENT"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=2)
```
```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
```
```
accuracy_list = []

log_reg = LogisticRegression()
history = log_reg.fit(x_train, y_train)
log_reg_pred = log_reg.predict(x_test)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
accuracy_list.append(100*log_reg_acc)
```
```
print("Accuracy of Logistic Regression is : ", "{:.2f}%".format(100* log_reg_acc))
```
####obtain theta
```
log_reg.intercept_[0]
w1, w2 = log_reg.coef_.T
```
####print to be used for ploting and observation 
```
print(w1)
print(w2)
```
