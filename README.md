# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  
df=pd.read_csv("Data_To_Transform.csv")  
df  
df.skew()  

#FUNCTION TRANSFORMATION:  

#Log Transformation  
np.log(df["Highly Positive Skew"])  

#Reciprocal Transformation  
np.reciprocal(df["Moderate Positive Skew"])  

#Square Root Transformation  
np.sqrt(df["Highly Positive Skew"])  

#Square Transformation  
np.square(df["Highly Negative Skew"])  

#POWER TRANSFORMATION:  

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])  
df  

df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])  
df  

df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])  
df  

df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])  
df  

#QUANTILE TRANSFORMATION:  

from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal')  

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])  
sm.qqplot(df['Moderate Negative Skew'],line='45')  
plt.show()
sm.qqplot(df['Moderate Negative Skew_1'],line='45')  
plt.show()  

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])  
sm.qqplot(df['Highly Negative Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Negative Skew_1'],line='45')  
plt.show()  

df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])  
sm.qqplot(df['Moderate Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Moderate Positive Skew_1'],line='45')  
plt.show() 

df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])  
sm.qqplot(df['Highly Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Positive Skew_1'],line='45')  
plt.show()  

df.skew()  
df 
```

# OUPUT
## Reading the given data set:
![output1](https://user-images.githubusercontent.com/93427923/168481051-54c08436-3354-44d6-91f1-0f3da648b9fc.png)

## Analyzing skewness of data:

![output2](https://user-images.githubusercontent.com/93427923/168481091-c996d52a-8390-4e5c-9ec1-8e5a912a5638.png)

## 1.FUNCTION TRANSFORMATION:
    
  ##  . Log Transformation:
![output3](https://user-images.githubusercontent.com/93427923/168481122-f25c3406-6138-44c8-90ea-8e50c613245b.png)

##  . Reciprocal Transformation:

![output4](https://user-images.githubusercontent.com/93427923/168481203-5818fab7-f32b-4488-b8bf-21b22095c354.png)

## Square Root Transformation:

![output5](https://user-images.githubusercontent.com/93427923/168481224-0e8362b3-b1b0-4866-b7cd-0866b532f37b.png)


## Square Transformation:

![output6](https://user-images.githubusercontent.com/93427923/168481227-ca4813e6-8800-475e-9acc-b282e100844a.png)

## 2.POWER TRANSFORMATION:
## Boxcox method:

![output7](https://user-images.githubusercontent.com/93427923/168481328-fb5e0d98-4171-4d04-ba9c-3f707f5e594c.png)

## Yeojohnson method:

![output8](https://user-images.githubusercontent.com/93427923/168481333-a581c4bd-0b7e-4f5c-be64-d0026c26787e.png)
![output9](https://user-images.githubusercontent.com/93427923/168481341-ed3430f3-1ff6-45af-a093-f3402481a816.png)
![output10](https://user-images.githubusercontent.com/93427923/168481346-1c94621b-ca64-4472-8f50-3da975238bba.png)

## 3.QUANTILE TRANSFORAMATION:
![output11](https://user-images.githubusercontent.com/93427923/168481356-c5437c8f-00de-4c76-be53-6aac962f90a1.png)
![output12](https://user-images.githubusercontent.com/93427923/168481360-89e5be91-195b-4f13-8500-3e7d82ad0357.png)
![output13](https://user-images.githubusercontent.com/93427923/168481366-57f86450-779b-4cd0-9576-7877105e8b22.png)
![output14](https://user-images.githubusercontent.com/93427923/168481368-60f54611-0177-4522-a408-083d96104703.png)

## Final Analysation of Skewness:
![output15](https://user-images.githubusercontent.com/93427923/168481391-84eea71c-b218-4647-abd8-d5e76e279f40.png)

# DATA SET-2:Titanic dataset.csv:

## CODE:

```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  

df=pd.read_csv("titanic_dataset.csv")  
df  

df.drop("Name",axis=1,inplace=True)  
df.drop("Cabin",axis=1,inplace=True)  
df.drop("Ticket",axis=1,inplace=True)  
df.isnull().sum()  

df["Age"]=df["Age"].fillna(df["Age"].median())  
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])  
df.info()  

from sklearn.preprocessing import OrdinalEncoder  
 
embark=["C","S","Q"]  
emb=OrdinalEncoder(categories=[embark])  
df["Embarked"]=emb.fit_transform(df[["Embarked"]])  

df  

#FUNCTION TRANSFORMATION:  
#Log Transformation  
np.log(df["Fare"])  

#ReciprocalTransformation  
np.reciprocal(df["Age"])  

#Squareroot Transformation:  
np.sqrt(df["Embarked"])  

#POWER TRANSFORMATION:  
df["Age _boxcox"], parameters=stats.boxcox(df["Age"])  
df  

df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])    
df    

df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])  
df  

df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])  
df  

df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])  
df  


#QUANTILE TRANSFORMATION  

from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)  


df["Age_1"]=qt.fit_transform(df[["Age"]])  
sm.qqplot(df['Age'],line='45')  
plt.show()  

sm.qqplot(df['Age_1'],line='45')  
plt.show()  

df["Fare_1"]=qt.fit_transform(df[["Fare"]])  
sm.qqplot(df["Fare"],line='45')  
plt.show()  

sm.qqplot(df['Fare_1'],line='45')  
plt.show()  

df.skew()  
df  
```

































