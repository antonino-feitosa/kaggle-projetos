"""
Created on 13 de mar. de 2022

@author: Antonino
"""

from numpy import nan
from pandas import DataFrame
from pandas.core.dtypes.missing import isna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class SpaceshipTitanicPreprocessing(BaseEstimator, TransformerMixin):

    def __init__(self,
                 fill_missing=True,
                 fill_total_spend=True,
                 fill_percent_spend=True,
                 scaling=True,
                 one_hot_encoding=True):

        self.fill_missing = fill_missing
        self.fill_percent_spend = fill_percent_spend
        self.fill_total_spend = fill_total_spend
        self.scaling = scaling
        self.one_hot_encoding = one_hot_encoding
        self.__cabin_deck_mode = None
        self.__destination_mode = None
        self.__home_planet_mode = None
        self.__cabin_side_mode = None
        self.__vip_mode = None
        self.__cryo_sleep_mode = None
        self.__cryo_grouped_median = None
        self.__age_grouped_median = None
        self.__ohe_encoder = {'Destination': OneHotEncoder(), 'HomePlanet': OneHotEncoder()}
        self.__scaling_encoder = {
            'Age': MinMaxScaler(), 'CabinDeck': MinMaxScaler(), 'RoomService': MinMaxScaler(),
            'FoodCourt': MinMaxScaler(), 'ShoppingMall': MinMaxScaler(), 'Spa': MinMaxScaler(),
            'VRDeck': MinMaxScaler(), 'TotalSpend': MinMaxScaler()
        }

    def fit(self, x, y=None):
        deck = x['Cabin'].apply(lambda v: v.split('/')[0] if not isna(v) else nan)
        self.__cabin_deck_mode = deck.mode()[0]
        self.__home_planet_mode = x['HomePlanet'].mode()[0]
        self.__destination_mode = x['Destination'].mode()[0]

        side = x['Cabin'].apply(lambda v: v.split('/')[2] == 'P' if not isna(v) else nan)
        self.__cabin_side_mode = side.mode()[0]
        self.__vip_mode = x['VIP'].mode()[0]
        self.__cryo_sleep_mode = x['CryoSleep'].mode()[0]

        spend_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        self.__cryo_grouped_median = x.loc[x['CryoSleep'] == False, spend_features].median()

        self.__age_grouped_median = x.groupby('HomePlanet', as_index=False).agg({'Age': 'median'})
        medians = dict()
        for value, med in self.__age_grouped_median.values:
            medians[value] = med
        self.__age_grouped_median = medians

        if self.one_hot_encoding:
            for name in self.__ohe_encoder.keys():
                self.__ohe_encoder[name].fit(x[[name]])

        if self.scaling:
            self.__scaling_encoder['CabinDeck'].fit(DataFrame([0, 7], columns=['CabinDeck']).dropna())
            if self.fill_total_spend:
                total_spend = x['RoomService'] + x['FoodCourt'] + x['ShoppingMall'] + x['Spa'] + x['VRDeck']
                self.__scaling_encoder['TotalSpend'].fit(DataFrame(total_spend, columns=['TotalSpend']).dropna())
            for name in spend_features + ['Age']:
                self.__scaling_encoder[name].fit(x[[name]].dropna())

        return self

    def transform(self, x, y=None):
        x['CabinDeck'] = x['Cabin'].apply(lambda v: v.split('/')[0] if not isna(v) else nan)
        x['CabinNum'] = x['Cabin'].apply(lambda v: v.split('/')[1] if not isna(v) else nan)
        x['CabinSide'] = x['Cabin'].apply(lambda v: v.split('/')[2] == 'P' if not isna(v) else nan)

        if self.fill_missing:
            x['CabinDeck'].fillna(self.__cabin_deck_mode, inplace=True)
            x['HomePlanet'].fillna(self.__home_planet_mode, inplace=True)
            x['Destination'].fillna(self.__destination_mode, inplace=True)

        x['HomePlanet'] = x['HomePlanet'].astype('category')
        x['Destination'] = x['Destination'].astype('category')

        x['CabinDeck'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}, inplace=True)

        if self.fill_missing:
            x['CabinSide'].fillna(self.__cabin_side_mode, inplace=True)
            x['VIP'].fillna(self.__vip_mode, inplace=True)
            x['CryoSleep'].fillna(self.__cryo_sleep_mode, inplace=True)

        spend_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        if self.fill_missing:
            for name in spend_features:
                x[name] = x[name].where(x['CryoSleep'], 0)
                x[name].fillna(self.__cryo_grouped_median[name], inplace=True)

        if self.fill_total_spend:
            x['TotalSpend'] = x['RoomService'] + x['FoodCourt'] + x['ShoppingMall'] + x['Spa'] + x['VRDeck']

        if self.fill_percent_spend:
            total_spend = x['TotalSpend'] if self.fill_total_spend else \
                x['RoomService'] + x['FoodCourt'] + x['ShoppingMall'] + x['Spa'] + x['VRDeck']

            pct_spend_features = ['PCT_RoomService', 'PCT_FoodCourt', 'PCT_ShoppingMall', 'PCT_Spa', 'PCT_VRDeck']
            x[pct_spend_features] = x[spend_features].apply(lambda v: v / total_spend)
            if self.fill_missing:
                x[pct_spend_features] = x[pct_spend_features].fillna(0)  # clear pct/0

        if self.fill_missing:
            x['Age'] = x[['HomePlanet', 'Age']].apply(lambda v: self.__age_grouped_median[v[0]] if isna(v[1]) else v[1],
                                                      axis=1).to_numpy()

        if self.one_hot_encoding:
            for name in self.__ohe_encoder.keys():
                new_features = self.__ohe_encoder[name].get_feature_names_out()
                x[new_features] = self.__ohe_encoder[name].transform(x[[name]]).toarray()
            x.drop(labels=['HomePlanet', 'Destination'], axis='columns', inplace=True)

        if self.scaling:
            for name in self.__scaling_encoder.keys():
                enc = self.__scaling_encoder[name]
                x[name] = enc.transform(x[[name]])

        x.drop(labels=['PassengerId', 'Cabin', 'CabinNum', 'Name'], axis='columns', inplace=True)

        return x


'''

data = read_csv("../input/spaceship-titanic-train.csv")
data = data.sample(frac=1).reset_index(drop=True)

X = DataFrame(data[0:1000])
Xt = DataFrame(data[1000:1020])

PREPROCESSING = SpaceshipTitanicPreprocessing(
    fill_missing=True,
    fill_total_spend=True,  # +1 = 18
    fill_percent_spend=True,  # +5 = 23
    scaling=True,
    one_hot_encoding=True  # 17 features
)

# X = PREPROCESSING.fit(X).transform(X)
X = PREPROCESSING.fit_transform(X)
Xt = PREPROCESSING.transform(Xt)

print(Xt.isna().sum())

'''