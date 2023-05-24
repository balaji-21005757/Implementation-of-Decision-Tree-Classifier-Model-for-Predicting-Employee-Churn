# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: K.Balaji
RegisterNumber: 212221230011
*/
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation",
"number_project","average_montly_hours","time_spend_company",
"Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### 1. data.head()
![image](https://github.com/balaji-21005757/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94372294/8693d193-7a3f-4663-bf23-7257e2f2b9c5)
### 2. data.info()
![image](https://github.com/balaji-21005757/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94372294/42c585f0-7552-46c3-9925-3ab300c9dba1)
### 3. isnull() and sum()
![image](https://github.com/balaji-21005757/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94372294/878ae1fd-64f4-4575-abbd-1efa0e5cc626)
### 4. data value counts()
![image](https://github.com/balaji-21005757/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94372294/038d4e16-df3d-47bd-8adb-6b6e3d723adc)
### 5. data.head() for salary
![image](https://github.com/balaji-21005757/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94372294/869c8d5c-95c7-4b3f-83d9-e15e55dbc277)
### 6. x.head()
![image](https://github.com/balaji-21005757/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94372294/58414ee5-9b2e-432c-a42b-16e882200b6c)
### 7. accuracy value
![image](https://github.com/balaji-21005757/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94372294/172596c6-8760-4a2c-9e22-be48eba84633)
### 8. data prediction
![image](https://github.com/balaji-21005757/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94372294/7a6a6fc5-54b9-4d96-8c9a-8d5d94afe58f)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
