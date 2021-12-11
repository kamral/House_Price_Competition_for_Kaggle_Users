import pandas as pd
import numpy as np
from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
train_data=pd.read_csv('train.csv', sep=',', index_col='Id')
test_data=pd.read_csv('test.csv', sep=',', index_col='Id')
print(train_data.head())
print(train_data.describe())
'''
Id   MSSubClass  LotFrontage        LotArea  ...       MiscVal       MoSold       YrSold      SalePrice
count  1460.000000  1460.000000  1201.000000    1460.000000  ...   1460.000000  1460.000000  1460.000000    1460.000000
mean    730.500000    56.897260    70.049958   10516.828082  ...     43.489041     6.321918  2007.815753  180921.195890
std     421.610009    42.300571    24.284752    9981.264932  ...    496.123024     2.703626     1.328095   79442.502883
min       1.000000    20.000000    21.000000    1300.000000  ...      0.000000     1.000000  2006.000000   34900.000000
25%     365.750000    20.000000    59.000000    7553.500000  ...      0.000000     5.000000  2007.000000  129975.000000
50%     730.500000    50.000000    69.000000    9478.500000  ...      0.000000     6.000000  2008.000000  163000.000000
75%    1095.250000    70.000000    80.000000   11601.500000  ...      0.000000     8.000000  2009.000000  214000.000000
max    1460.000000   190.000000   313.000000  215245.000000  ...  15500.000000    12.000000  2010.000000  755000.000000

'''
print(train_data.columns)
'''
Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice'],
      dtype='object')

'''

y=train_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X_train=train_data[features].copy()
print(X_train.head())

'''
 LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  TotRmsAbvGrd
0     8450       2003       856       854         2             3             8
1     9600       1976      1262         0         2             3             6
2    11250       2001       920       866         2             3             6
3     9550       1915       961       756         1             3             7
4    14260       2000      1145      1053         2             4             9

'''
X_test=test_data[features].copy()
print(X_test.head())
'''
LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  TotRmsAbvGrd
0    11622       1961       896         0         1             2             5
1    14267       1958      1329         0         1             3             6
2    13830       1997       928       701         2             3             6
3     9978       1998       926       678         2             3             7
4     5005       1992      1280         0         2             2             5
'''

train_X, val_X, train_y, val_y=train_test_split(X_train,y, train_size=0.8,
                                                test_size=0.2,
                                                random_state=0)
print(train_X.head())
'''
LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  TotRmsAbvGrd
618    11694       2007      1828         0         2             3             9
870     6600       1962       894         0         1             2             5
92     13360       1921       964         0         1             2             5
817    13265       2002      1689         0         2             3             7
302    13704       2001      1541         0         2             3             6
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import  DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
model_1=RandomForestRegressor(n_estimators=50, random_state=0)
model_2=RandomForestRegressor(n_estimators=100, random_state=0)
model_3=RandomForestRegressor(n_estimators=150, criterion='mae', random_state=0)
model_4=RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5=RandomForestRegressor(n_estimators=200, min_samples_split=7, random_state=0)
model_6=DecisionTreeClassifier(random_state=0)
model_7=DecisionTreeRegressor(random_state=0)
models=[model_1,model_2,model_3,model_4,model_5,model_6]

def score_model(model, X_t=train_X, X_v=val_X, y_t=train_y,y_v=val_y ):
    model.fit(X_t,y_t)
    preds=model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0,len(models)):
    mae=score_model(models[i])
    print('Model %d MAE: %d' %(i+1, mae))


cols_with_missing=[col for col in X_train.columns
                   if X_train[col].isnull().any()]

reduced_X_train=X_train.drop(cols_with_missing, axis=1)
print(reduced_X_train)
reduced_X_test=X_test.drop(cols_with_missing, axis=1)
print(reduced_X_test)
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_test))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
print(imputed_X_valid)
imputed_X_valid.columns = X_test.columns
print(imputed_X_valid)
#
#
# my_model=model_3
# my_model.fit(X,y)
# preds_test=my_model.predict(X_test)
# output=pd.DataFrame({
#     'Id':X_test.index,
#     'SalePrice':preds_test
# })

# output.to_csv('submission.csv',index=False)
