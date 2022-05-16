# EX-06-Feature-Transformation

# AIM
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
    
  ### Log Transformation:
![output3](https://user-images.githubusercontent.com/93427923/168481122-f25c3406-6138-44c8-90ea-8e50c613245b.png)

### Reciprocal Transformation:

![output4](https://user-images.githubusercontent.com/93427923/168481203-5818fab7-f32b-4488-b8bf-21b22095c354.png)

### Square Root Transformation:

![output5](https://user-images.githubusercontent.com/93427923/168481224-0e8362b3-b1b0-4866-b7cd-0866b532f37b.png)


### Square Transformation:

![output6](https://user-images.githubusercontent.com/93427923/168481227-ca4813e6-8800-475e-9acc-b282e100844a.png)

## 2.POWER TRANSFORMATION:
### Boxcox method:

![output7](https://user-images.githubusercontent.com/93427923/168481328-fb5e0d98-4171-4d04-ba9c-3f707f5e594c.png)

### Yeojohnson method:

![output8](https://user-images.githubusercontent.com/93427923/168481333-a581c4bd-0b7e-4f5c-be64-d0026c26787e.png)
![output9](https://user-images.githubusercontent.com/93427923/168481341-ed3430f3-1ff6-45af-a093-f3402481a816.png)
![output10](https://user-images.githubusercontent.com/93427923/168481346-1c94621b-ca64-4472-8f50-3da975238bba.png)

## 3.QUANTILE TRANSFORAMATION:
![output11](https://user-images.githubusercontent.com/93427923/168481356-c5437c8f-00de-4c76-be53-6aac962f90a1.png)
![output12](https://user-images.githubusercontent.com/93427923/168481360-89e5be91-195b-4f13-8500-3e7d82ad0357.png)
![output13](https://user-images.githubusercontent.com/93427923/168481366-57f86450-779b-4cd0-9576-7877105e8b22.png)
![output14](https://user-images.githubusercontent.com/93427923/168481368-60f54611-0177-4522-a408-083d96104703.png)

### Final Analysation of Skewness:
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

# OUTPUT:
## Reading the given data set:

![pic1](https://user-images.githubusercontent.com/93427923/168523156-b3089aa9-5fdc-46b2-a65f-b9a98767d48d.png)

## Data Cleaning Process:

![pic2](https://user-images.githubusercontent.com/93427923/168523228-1efae061-4ba5-444a-9a39-b4d8ee0d3448.png)
![pic4](https://user-images.githubusercontent.com/93427923/168523236-520a9c70-2bd9-4630-9cad-f674e36b02ce.png)

## 1.FUNCTION TRANSFORMATION:
### Log Transformation:

![pic5](https://user-images.githubusercontent.com/93427923/168523351-0c811632-84b7-4ea4-a47d-4c1374d0d2f5.png)

### Reciprocal Transformation:
![pic6](https://user-images.githubusercontent.com/93427923/168523388-b63588e7-4ec8-4735-8eca-484bf2f8ffda.png)

### Square Root Transformation:

![pic7](https://user-images.githubusercontent.com/93427923/168523432-1ead8511-0c32-4b0f-9650-6ff67189bb73.png)

## 2.POWER TRANSFORMATION:

### Boxcox method:

![pic8](https://user-images.githubusercontent.com/93427923/168523478-9ebff601-67b3-49f7-b462-6fe1ac505d81.png)
![pic9](https://user-images.githubusercontent.com/93427923/168523557-f584ec6d-6488-462e-8851-5e764b627dbf.png)

### Yeojohnson method:

![pic10](https://user-images.githubusercontent.com/93427923/168523528-ee9aaf6f-f421-4aa3-950b-2ce4755b9fec.png)
![pic11](https://user-images.githubusercontent.com/93427923/168523692-b1ae9aa6-a659-4ff8-9d23-6b64050d39f7.png)
![pic12](https://user-images.githubusercontent.com/93427923/168523704-ed114f5e-eb87-4c2b-8175-f6d806240ff6.png)

## 3.QUANTILE TRANSFORAMATION:


![pic13](https://user-images.githubusercontent.com/93427923/168523751-e1e7b4d8-ca64-4d30-83c7-f06b54249075.png)
![pic14](https://user-images.githubusercontent.com/93427923/168523773-17f1f6c4-cc66-4270-8130-5ca415a3b6f4.png)

## Final Analysation of Skewness:

![pic15](https://user-images.githubusercontent.com/93427923/168523807-2e195f45-ab60-4cda-8ac1-5a7704624e6d.png)

# INFERENCE:
Log transormation is applicable only for right skewed data.
Boxcox transformation is applicable only for positive data and yeojohnson transformation for data that contain zero or negative values.
Reciprocal transformation is not defined for zeroes.
Quantile transforms are a technique for transforming numerical input or output variables to have a Gaussian or uniform probability distribution.
Square transformation is applicable for left skewed data.

# RESULT:
Hence, Feature transformation techniques is been performed on given dataset and saved into a file successfully.
