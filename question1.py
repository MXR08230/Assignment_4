import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
#Importing the Salary_Data csv file
df=pd.read_csv("./Salary_Data.csv")
print(df.head())

#Slicing the features using iloc
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values
X_Training, X_Testing, Y_Training, Y_Testing = train_test_split(X,Y, test_size=1/3,random_state = 0)

#Training and Predicting the model
regressor = LinearRegression()
regressor.fit(X_Training, Y_Training)
Y_Predict = regressor.predict(X_Testing)

#calculating mean_squared_error
print(mean_squared_error(Y_Testing,Y_Predict))

#Scatterplot for Training Data
plt.title('Training data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(X_Training, Y_Training)
plt.show()
#Scatterplot for Testing Data
plt.title('Testing data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.scatter(X_Testing, Y_Testing)
plt.show()
