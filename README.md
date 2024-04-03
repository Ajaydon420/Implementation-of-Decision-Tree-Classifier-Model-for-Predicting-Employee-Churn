# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Ajay K 
RegisterNumber:  212222080005
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:
![decision tree classifier model](sam.png)
![image](https://github.com/Ajaydon420/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161410969/c1b02c4f-5695-4f58-b66f-a8d12604a788)

NULL AND COUNT;

![318890102-e26eeddb-0b03-42c5-955b-860c919822b0](https://github.com/Ajaydon420/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161410969/c3011a82-4dbe-4b71-9cb9-db8625278b40)

![318890102-e26eeddb-0b03-42c5-955b-860c919822b0](https://github.com/Ajaydon420/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161410969/840b72cf-8efb-4a00-a7ff-8771fc67c7da)
ACCURACY SCORE

![318890396-e67d3205-ec86-4b99-956b-00b9fb923498](https://github.com/Ajaydon420/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161410969/0de3e392-0a49-4e68-9ac2-3f4e9cb2b330)

DECISION TREE CLASSIFIER MODEL:

![318890578-7c93ff98-b7b4-455b-aa42-d1c38c6391f2](https://github.com/Ajaydon420/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/161410969/e2c73c4b-c23b-4b2e-8c01-5769495e7a6d)



## Result:

Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
