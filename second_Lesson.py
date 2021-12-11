import pandas as pd
train_data=pd.read_csv('train.csv', sep=',', index_col='Id')
X_train=train_data
# print(train_data)
'''
MSSubClass MSZoning  LotFrontage  LotArea Street Alley  ... MiscVal MoSold YrSold SaleType SaleCondition SalePrice
Id                                                          ...                                                       
1           60       RL         65.0     8450   Pave   NaN  ...       0      2   2008       WD        Normal    208500
2           20       RL         80.0     9600   Pave   NaN  ...       0      5   2007       WD        Normal    181500
3           60       RL         68.0    11250   Pave   NaN  ...       0      9   2008       WD        Normal    223500
4           70       RL         60.0     9550   Pave   NaN  ...       0      2   2006       WD       Abnorml    140000
5           60       RL         84.0    14260   Pave   NaN  ...       0     12   2008       WD        Normal    250000

'''
test_data=pd.read_csv('test.csv', sep=',', index_col='Id')
X_test=test_data
# print(test_data.head())
'''
MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  ...  Fence MiscFeature MiscVal MoSold YrSold SaleType SaleCondition
Id                                                                     ...                                                                
1461          20       RH         80.0    11622   Pave   NaN      Reg  ...  MnPrv         NaN       0      6   2010       WD        Normal
1462          20       RL         81.0    14267   Pave   NaN      IR1  ...    NaN        Gar2   12500      6   2010       WD        Normal
1463          60       RL         74.0    13830   Pave   NaN      IR1  ...  MnPrv         NaN       0      3   2010       WD        Normal
1464          60       RL         78.0     9978   Pave   NaN      IR1  ...    NaN         NaN       0      6   2010       WD        Normal
1465         120       RL         43.0     5005   Pave   NaN      IR1  ...    NaN         NaN       0      1   2010       WD        Normal
'''

import pandas as pd
from sklearn.model_selection import train_test_split



# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
print(train_data)

mising_val_count_by_column=train_data.isnull().sum()
print(mising_val_count_by_column[mising_val_count_by_column>0])

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

print(score_dataset())