# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## AlgorithmGet the data and use label encoder to change all the values to numeric.
1. Drop the unwanted values,Check for NULL values, Duplicate values.     
2. Classify the training data and the test data.                
3. Calculate the accuracy score, confusion matrix and classification report.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Dhiyaneshwar P
RegisterNumber: 212222110009
*/
```
```
import pandas as pd
data = pd.read_csv('/content/Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:, :-1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver ="liblinear")
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy Score:",accuracy)
print("\nConfusion Matrix:\n",confusion)
print("\nClassification Report:\n",cr)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```
## Output:
![image](https://github.com/Dhiyanesh24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118362288/3336a4e2-b14b-41a7-aac3-0b2bd47ba14c)

![image](https://github.com/Dhiyanesh24/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118362288/8d99e3ad-eea6-4d2c-b4f7-d39d00abe433)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
