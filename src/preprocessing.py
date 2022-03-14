'''
Created on 9 de mar. de 2022

@author: Antonino
'''

import pandas as pd
from pandas.core.frame import DataFrame
from pandas.api.types import is_numeric_dtype


class Preprocessing(object):
    '''
    classdocs
    '''

    def __init__(self, dataset: DataFrame, classIndex: str):
        '''
        Constructor
        '''
        self.data = dataset
        self.classIndex = classIndex
    
    def has_missing(self, index: str) -> None:
        return self.data[index].isna().sum() > 0
        
    def forceInt(self, index: str) -> None:
        assert self.data[index].isna().sum() == 0
        self.data[index] = self.data[index].astype('int64')
        return
    
    def forceCategory(self, index: str) -> None:
        assert self.data[index].isna().sum() == 0
        self.data[index] = self.data[index].astype('category')
        return
    
    def fillMissingValuesMean(self, index: str) -> None:
        assert is_numeric_dtype(self.data[index])
        if self.has_missing(index):
            u = self.data[index].mean()
            self.data[index].fillna(u, inplace=True)
        return
    
    def fillMissingValuesMeanSupervised(self, index: str) -> None:
        assert is_numeric_dtype(self.data[index])
        if self.has_missing(index):
            for value in self.data[self.classIndex].unique():
                u = self.data.loc[self.data[self.classIndex] == value, index].mean()
                self.data.loc[(self.data[index].isna()) & (self.data[self.classIndex] == value), index] = u
        return
        
    def fillMissingValuesMode(self, index: str) -> None:
        if self.has_missing(index):
            u = self.data[index].mode()
            self.data[index].fillna(u, inplace=True)
        return

    def fillMissingValuesModeSupervised(self, index: str) -> None:
        if self.has_missing(index):
            for value in self.data[self.classIndex].unique():
                u = self.data.loc[self.data[self.classIndex] == value, index].mode()
                self.data.loc[(self.data[index].isna()) & (self.data[self.classIndex] == value), index] = u
        return
    
    def fillMissingValuesNewValue(self, index: str) -> None:
        assert self.has_missing(index)
        newvalue = self.data[index].min() if is_numeric_dtype(self.data[index]) else 'Unknown'
        self.data[index].fillna(newvalue, inplace=True)
        return
    
    def fillMissingValuesNewValueSupervised(self, index: str) -> None:
        assert self.has_missing(index)
        for value in self.data[self.classIndex].unique():
            newvalue = self.data[index].min() if is_numeric_dtype(self.data[index]) else "Unknown{}".format(value)
            self.data.loc[(self.data[index].isna()) & (self.data[self.classIndex] == value), index] = newvalue
        return
        
    def missingValues(self, method: str, supervised: bool):
        if method == 'majority':
            for att in list(self.data.select_dtypes(include='number')):
                self.fillMissingValuesMeanSupervised(att) if supervised else self.fillMissingValuesMean(att)
            for att in list(self.data.select_dtypes(exclude='number')):
                self.fillMissingValuesModeSupervised(att) if supervised else self.fillMissingValuesMode(att)
        elif method == 'estimate':
            pass
        elif method == 'new':
            for att in list(self.data):
                self.fillMissingValuesNewValue(att) if supervised else self.fillMissingValuesNewValueSupervised(att) 
        elif method == 'eliminate':
            pass
            
    def normalize(self):
        pass


data = pd.read_csv("../input/spaceship-titanic-train.csv")
# data.drop(labels=['PassengerId', 'Name'], axis="columns", inplace=True)

print(is_numeric_dtype(data['Name']))

pre = Preprocessing(data, 'Transported')

# print(data['Destination'].unique())
# print(data['Destination'].mode())

data = pd.read_csv("../input/spaceship-titanic-train.csv")
with pd.option_context('display.max_columns', 40):
    dupli = data[(data['Name'].duplicated(keep=False)) & (data['Name'].notna())].sort_values(by='Name')

# with pd.option_context('display.max_columns', 40):
#    print(data.head(25))

# feature selection
# Using Pearson Correlation
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 10))
cor = train.corr()
sns.heatmap(cor, annot=True)
plt.show()

# feature selection
feature_selection = SelectKBest(score_func=chi2, k=10)
selected = feature_selection.fit(X_train, y_train)
# X_train = selected.transform(X_train)

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

names = X_train.columns.tolist()

preprocessor = ColumnTransformer(
     transformers=[
    ('num', numeric_transformer, numericas_all)
    , ('cat', categorical_transformer, categoricas_all)
    ])

pipe_preprocessor = Pipeline([("preprocessor", preprocessor), ("pandarizer", FunctionTransformer(lambda x: pd.DataFrame(x, columns=names)))]).fit(X_train)
    
X_train_pipe = pipe_preprocessor.transform(X_train)
X_test_pipe = pipe_preprocessor.transform(X_test)

