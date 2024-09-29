from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error as me

import pandas as pd

data= pd.read_csv('train.csv')
train_data,test_data=train_test_split(data,test_size=.3,train_size=.7,random_state=42)
print(test_data.shape)
train_data.info()

# Get list of categorical variables
object_cols = train_data.select_dtypes(include="object").columns.tolist()
print("Categorical variables:", object_cols)
train_features=train_data.iloc[:,1:80]
train_out=train_data.iloc[:,-1]
train_out.info()
train_features=train_features[['LotArea','MasVnrArea','FullBath','HalfBath','TotRmsAbvGrd','GarageArea','PoolArea']]
train_features.info()
test_features=test_data[['LotArea','MasVnrArea','FullBath','HalfBath','TotRmsAbvGrd','GarageArea','PoolArea']]
test_features.info()
test_out=test_data.iloc[:,-1]
imputer = SimpleImputer(strategy='mean')
imputer.fit(train_features)
train_features = imputer.transform(train_features)
test_features = imputer.transform(test_features)


model=LinearRegression()
result=model.fit(train_features,train_out)
prediction=model.predict(test_features)

mse=me(test_out,prediction)

print(mse)
