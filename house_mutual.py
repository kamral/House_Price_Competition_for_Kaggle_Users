import pandas as pd
housing_df=pd.read_csv('train.csv')
print(housing_df)
print(housing_df.isnull().sum())
# везде где эти типы данных  этому конкретному списку
numeric_lst=['int64','int32','int64','float64', 'float32', 'float64']
# включить numeric_lst в основном. я просто собираюсь взять все эти значения
numeric_cols=list(housing_df.select_dtypes(include=numeric_lst).columns)
print(numeric_cols)
housing_df=housing_df[numeric_cols]
print(housing_df)
housing_df=housing_df.fillna(housing_df.mean())
y=housing_df['SalePrice']

X=housing_df.drop('SalePrice', axis=1)
print(X)

# # разделим обучения и тестировани
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=0, test_size=0.3)
mutual_information=mutual_info_regression(X_train,y_train)
print(mutual_information)
mutual_info=pd.Series(mutual_information)
print(mutual_info)
mutual_info.index=X_train.columns
print(mutual_info.sort_values(ascending=False).plot.bar(figsize=(15,5)))
