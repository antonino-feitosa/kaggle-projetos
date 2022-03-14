'''
Created on 13 de mar. de 2022

@author: Antonino
'''

from numpy import nan, where, int64
from pandas import read_csv, option_context
from pandas.core.dtypes.missing import isna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute._base import SimpleImputer
from tensorflow.python.ops import inplace_ops
from pandas.core.reshape.merge import merge


class SpaceshipTitanicPreprocessing(BaseEstimator, TransformerMixin):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def fit_transform(self, X, y=None):
        return self.transform(X, y)
    
    def transform(self, data, y=None):
        
        data['PassengerGroup'] = data['PassengerId'].apply(lambda x: x.split('_')[0])
        data['PassengerNo'] = data['PassengerId'].apply(lambda x: x.split('_')[1])
        
        data['CabinDeck'] = data['Cabin'].apply(lambda x: x.split('/')[0] if not isna(x) else nan)
        data['CabinNum'] = data['Cabin'].apply(lambda x: x.split('/')[1] if not isna(x) else nan)
        data['CabinSide'] = data['Cabin'].apply(lambda x: x.split('/')[2] == 'P' if not isna(x) else nan)
        
        data['CabinDeck'].fillna(data['CabinDeck'].mode()[0], inplace=True)
        data['HomePlanet'].fillna(data['HomePlanet'].mode()[0], inplace=True)
        data['Destination'].fillna(data['Destination'].mode()[0], inplace=True)
        
        data['HomePlanet'] = data['HomePlanet'].astype('category')
        data['Destination'] = data['Destination'].astype('category')
        
        data['CabinDeck'].replace({'A':0, 'B':1, 'C': 2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7}, inplace=True)
        data['CabinDeck'] = data['CabinDeck'].astype(int64)
        
        bool_features = ['VIP', 'CryoSleep', 'CabinSide']
        mode_values = data[bool_features].mode(axis='columns')
        data[bool_features] = data[bool_features].fillna(mode_values)
        data[bool_features] = data[bool_features].astype(bool)
        
        spend_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        data[spend_features] = data[spend_features].fillna(data.groupby('CryoSleep')[spend_features].transform('median'))
        
        data['TotalSpend'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck']
        
        pct_spend_features = ['PCT_RoomService', 'PCT_FoodCourt', 'PCT_ShoppingMall', 'PCT_Spa', 'PCT_VRDeck']
        data[pct_spend_features] = data[spend_features].apply(lambda x: x / data['TotalSpend'])
        data[pct_spend_features] = data[pct_spend_features].fillna(0)  # clear pct/0
        
        data['Age'] = data['Age'].fillna(data.groupby('HomePlanet')['Age'].transform('median'))
        
        # The instances in the same group must be similar
        group_data = (data.groupby('PassengerGroup', as_index=False)
          .agg({'PassengerNo':'nunique',
                'VIP':lambda x: sum(x == True),
                'CryoSleep': lambda x: sum(x == True),
                'Cabin': 'nunique',
                'CabinDeck': 'nunique',
                'CabinNum': 'nunique',
                'CabinSide': 'nunique',
                'Age': 'median',
                'HomePlanet': 'nunique',
                'Destination': 'nunique',
                'RoomService': 'median',
                'FoodCourt': 'median',
                'ShoppingMall':'median',
                'Spa':'median',
                'VRDeck': 'median',
                'TotalSpend':'median',
                'PCT_RoomService': 'mean',
                'PCT_FoodCourt': 'mean',
                'PCT_ShoppingMall': 'mean',
                'PCT_Spa': 'mean',
                'PCT_VRDeck': 'mean'
                })
          .rename(columns={
                'PassengerNo':'GroupCount',
                'VIP': 'GroupCountVIP',
                'CryoSleep': 'GroupCountCryoSleep',
                'Cabin': 'GroupCountCabin',
                'CabinDeck': 'GroupCountCabinDeck',
                'CabinNum': 'GroupCountCabinNum',
                'CabinSide': 'GroupCountCabinSide',
                'Age': 'GroupMedianAge',
                'HomePlanet': 'GroupMeanCountHomePlanet',
                'Destination': 'GroupMeanCountDestination',
                'RoomService': 'GroupMedianRoomService',
                'FoodCourt': 'GroupMedianFoodCourt',
                'ShoppingMall': 'GroupMedianShoppingMall',
                'Spa':'GroupMedianSpa',
                'VRDeck': 'GroupMedianVRDeck',
                'TotalSpend':'GroupMedianTotalSpend',
                'PCT_RoomService': 'GroupMeanPCT_RoomService',
                'PCT_FoodCourt': 'GroupMeanPCT_FoodCourt',
                'PCT_ShoppingMall': 'GroupMeanPCT_ShoppingMall',
                'PCT_Spa': 'GroupMeanPCT_Spa',
                'PCT_VRDeck': 'GroupMeanPCT_VRDeck'
                })
         )

        data = merge(data, group_data, on='PassengerGroup', how='left')
        
        data.drop(labels=['PassengerId', 'PassengerGroup', 'PassengerNo', 'Cabin', 'CabinNum', 'Name'], axis='columns', inplace=True)
        return data


data = read_csv("../input/spaceship-titanic-train.csv")
# data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

'''
spend_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
with option_context('display.max_columns', 40):
    print(data.loc[data['CryoSleep'] == True, spend_features].describe())
'''

'''
ages = data.groupby('HomePlanet', as_index=False)
with option_context('display.max_columns', 40):
    print(ages['Age'].describe())
'''

data = SpaceshipTitanicPreprocessing().fit_transform(data)

'''
with option_context('display.max_columns', 40):
    print(data.describe(include='all'))
    print(data.head(20))

print(data.isna().sum())
print(data.dtypes)
'''

# sorted_mat = data.corr().unstack().sort_values()

# print(sorted_mat[0:30])

'''
mean_spend = data.loc[data['CryoSleep'] == False, spend_features].mean()

data['RoomService'].fillna(int(data['RoomService'].mean()), inplace=True)
data['FoodCourt'].fillna(int(data['FoodCourt'].mean()), inplace=True)
data['ShoppingMall'].fillna(int(data['ShoppingMall'].mean()), inplace=True)
data['Spa'].fillna(int(data['Spa'].mean()), inplace=True)
data['VRDeck'].fillna(int(data['VRDeck'].mean()), inplace=True)
data['TotalSpend'] = data['RoomService'] + data['FoodCourt'] + data['ShoppingMall'] + data['Spa'] + data['VRDeck']
'''
# with option_context('display.max_columns', 40):
#    print(data.loc[data['CryoSleep'] == False, spend_features].head(30))

'''
   
   
   
   
   
   
   
   
   
   
   
   
   
   '''
   
