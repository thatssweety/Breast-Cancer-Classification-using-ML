# [Project 3](https://github.com/thatssweety/Breast-Cancer-Classification-using-ML) Breast Cancer Classification using ML
![shutterstock_701352787](https://user-images.githubusercontent.com/81384066/184434470-fae02f84-0c6a-4213-a490-2cf255c3fa4a.jpg)

## Dataset Used :

### [Breast Cancer Wisconsin (Diagnostic) Data Set](https://github.com/thatssweety/Breast-Cancer-Classification-using-ML/blob/main/Breast%20cancer%20dataset.zip)

## Tech Stack used :
1. Numpy
2. Pandas
3. Sklearn
# Code

## Import Libraries and sklearn Dataset

```python

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
breast_cancer_dataset=sklearn.datasets.load_breast_cancer()
data_frame=pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names) 
data_frame['label']=breast_cancer_dataset.target
data_frame.head()

```
<img width="1010" alt="Screenshot (446)" src="https://user-images.githubusercontent.com/81384066/184438361-5833df1f-aafc-45af-8966-6d04214e57e2.png">

## Checking missing values

```python

data_frame.isnull().sum()

```
<img width="421" alt="Screenshot (448)" src="https://user-images.githubusercontent.com/81384066/184438540-97690241-f888-456c-9ea3-15fdae903e36.png">

## For checking the Distribution of Target Variable
```python
data_frame['label'].value_counts()  #0 for malignant #1 for benign
data_frame.groupby('label').mean()
```


## Seperating features and target
```python
X=data_frame.drop(columns=['label'],axis=1)
Y=data_frame['label']
```
## Splitting into training and test split
```python
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
```
## Model Training - Logistic Regression
```python
model=LogisticRegression()
model.fit(X_train,Y_train)
```
## Model evaluation (training and test data)
```python
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print("Accuracy on training data ",training_data_accuracy)
_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print("Accuracy on test data ",test_data_accuracy)
```
<img width="411" alt="Screenshot (449)" src="https://user-images.githubusercontent.com/81384066/184438755-0079ab34-f29b-4835-8ac3-78f1141d98ce.png">

## Buidling a predictive system
```python
input_array=(20.29,14.34,135.1,1297,0.1003,0.1328,0.198,0.1043,0.1809,0.05883,0.7572,0.7813,5.438,94.44,0.01149,0.02461,0.05688,0.01885,0.01756,0.005115,22.54,16.67,152.2,1575,0.1374,0.205,0.4,0.1625,0.2364,0.07678);
inputtonumpy=np.asarray(input_array)
input_reshaped=inputtonumpy.reshape(1,-1)
prediction =model.predict(input_reshaped)
print(prediction)
if(prediction[0]==1):
  print("Breast cancer is benign")
else:
  print("Breast cancer is malignant")
   
```
<img width="295" alt="Screenshot (450)" src="https://user-images.githubusercontent.com/81384066/184438939-c4863c75-9066-48c6-aa44-a0ef4ebfec0a.png">
<br>
## [Complete Code Here](https://github.com/thatssweety/Breast-Cancer-Classification-using-ML)
        
             
                  
## [DataSet Here](https://github.com/thatssweety/Breast-Cancer-Classification-using-ML/blob/main/Breast%20cancer%20dataset.zip)
## [Github](https://github.com/thatssweety/)


