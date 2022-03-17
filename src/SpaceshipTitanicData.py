"""
Created on 13 de mar. de 2022

@author: Antonino
"""

from numpy import nan
from pandas import DataFrame
from pandas.core.dtypes.missing import isna
from pandas.core.reshape.merge import merge
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class SpaceshipTitanicPreprocessing(BaseEstimator, TransformerMixin):

    def __init__(self,
                 fill_missing=True,
                 fill_total_spend=True,
                 fill_percent_spend=True,
                 fill_group=True,
                 scaling=True,
                 one_hot_encoding=True):

        self.fill_missing = fill_missing
        self.fill_group = fill_group
        self.fill_percent_spend = fill_percent_spend
        self.fill_total_spend = fill_total_spend
        self.scaling = scaling
        self.one_hot_encoding = one_hot_encoding
        self.__cabin_deck_mode = None
        self.__destination_mode = None
        self.__home_planet_mode = None
        self.__bool_features_mode_values = None
        self.__cryo_grouped_median = None
        self.__age_grouped_median = None
        self.__group_data = None
        self.__ohe_encoder = {'Destination': OneHotEncoder(), 'HomePlanet': OneHotEncoder()}
        self.__scaling_encoder = {
            'Age': MinMaxScaler(), 'CabinDeck': MinMaxScaler(), 'RoomService': MinMaxScaler(),
            'FoodCourt': MinMaxScaler(), 'ShoppingMall': MinMaxScaler(), 'Spa': MinMaxScaler(),
            'VRDeck': MinMaxScaler(), 'TotalSpend': MinMaxScaler()
        }
        if self.fill_group:
            self.__scaling_encoder.update({
                'GroupCount': MinMaxScaler(), 'GroupCountVIP': MinMaxScaler(), 'GroupCountCryoSleep': MinMaxScaler(),
                'GroupCountCabin': MinMaxScaler(), 'GroupCountCabinDeck': MinMaxScaler(),
                'GroupCountCabinNum': MinMaxScaler(), 'GroupCountCabinSide': MinMaxScaler(),
                'GroupMedianAge': MinMaxScaler(), 'GroupMeanCountHomePlanet': MinMaxScaler(),
                'GroupMeanCountDestination': MinMaxScaler(), 'GroupMedianRoomService': MinMaxScaler(),
                'GroupMedianFoodCourt': MinMaxScaler(), 'GroupMedianShoppingMall': MinMaxScaler(),
                'GroupMedianSpa': MinMaxScaler(), 'GroupMedianVRDeck': MinMaxScaler(),
                'GroupMedianTotalSpend': MinMaxScaler()
            })

    def __drop_unused_features(self, x: DataFrame):
        x.drop(labels=['PassengerId', 'PassengerGroup', 'PassengerNo', 'Cabin', 'CabinNum', 'Name'], axis='columns',
               inplace=True)

        if self.fill_group:
            if not self.fill_percent_spend:
                x.drop(labels=['GroupMeanPCT_RoomService', 'GroupMeanPCT_FoodCourt', 'GroupMeanPCT_ShoppingMall',
                               'GroupMeanPCT_Spa', 'GroupMeanPCT_VRDeck'], axis='columns', inplace=True)
            if not self.fill_total_spend:
                x.drop(labels=['GroupMedianTotalSpend'], axis='columns', inplace=True)

        if not self.fill_percent_spend:
            x.drop(labels=['PCT_RoomService', 'PCT_FoodCourt', 'PCT_ShoppingMall', 'PCT_Spa', 'PCT_VRDeck'],
                   axis='columns', inplace=True)
        if not self.fill_total_spend:
            x.drop(labels=['TotalSpend'], axis='columns', inplace=True)
        if self.__ohe_encoder:
            x.drop(labels=['HomePlanet', 'Destination'], axis='columns', inplace=True)

    def fit_transform(self, x, y=None, **fit_params):
        x['CabinDeck'] = x['Cabin'].apply(lambda v: v.split('/')[0] if not isna(v) else nan)
        x['CabinNum'] = x['Cabin'].apply(lambda v: v.split('/')[1] if not isna(v) else nan)
        x['CabinSide'] = x['Cabin'].apply(lambda v: v.split('/')[2] == 'P' if not isna(v) else nan)

        if self.fill_missing:
            self.__cabin_deck_mode = x['CabinDeck'].mode()[0]
            self.__home_planet_mode = x['HomePlanet'].mode()[0]
            self.__destination_mode = x['Destination'].mode()[0]
            x['CabinDeck'].fillna(self.__cabin_deck_mode, inplace=True)
            x['HomePlanet'].fillna(self.__home_planet_mode, inplace=True)
            x['Destination'].fillna(self.__destination_mode, inplace=True)

        x['HomePlanet'] = x['HomePlanet'].astype('category')
        x['Destination'] = x['Destination'].astype('category')

        x['CabinDeck'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7}, inplace=True)
        # data['CabinDeck'] = data['CabinDeck'].astype(int64)

        bool_features = ['VIP', 'CryoSleep', 'CabinSide']
        if self.fill_missing:
            self.__bool_features_mode_values = x[bool_features].mode(axis='columns')
            x[bool_features] = x[bool_features].fillna(self.__bool_features_mode_values)
            x[bool_features] = x[bool_features].astype(bool)

        spend_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        if self.fill_missing:
            self.__cryo_grouped_median = x.groupby('CryoSleep')[spend_features].transform('median')
            x[spend_features] = x[spend_features].fillna(self.__cryo_grouped_median)

        x['TotalSpend'] = x['RoomService'] + x['FoodCourt'] + x['ShoppingMall'] + x['Spa'] + x['VRDeck']

        pct_spend_features = ['PCT_RoomService', 'PCT_FoodCourt', 'PCT_ShoppingMall', 'PCT_Spa', 'PCT_VRDeck']
        x[pct_spend_features] = x[spend_features].apply(lambda v: v / x['TotalSpend'])
        if self.fill_missing:
            x[pct_spend_features] = x[pct_spend_features].fillna(0)  # clear pct/0

        if self.fill_missing:
            self.__age_grouped_median = x.groupby('HomePlanet')['Age'].transform('median')
            x['Age'] = x['Age'].fillna(self.__age_grouped_median)

        # Instances in the same group must be similar
        x['PassengerGroup'] = x['PassengerId'].apply(lambda v: v.split('_')[0])
        x['PassengerNo'] = x['PassengerId'].apply(lambda v: v.split('_')[1])
        self.__group_data = (x.groupby('PassengerGroup', as_index=False).agg({
            'PassengerNo': 'nunique',
            'VIP': lambda v: sum(v == True), 'CryoSleep': lambda v: sum(v == True),
            'Cabin': 'nunique', 'CabinDeck': 'nunique', 'CabinNum': 'nunique', 'CabinSide': 'nunique',
            'HomePlanet': 'nunique', 'Destination': 'nunique', 'Age': 'median', 'RoomService': 'median',
            'FoodCourt': 'median', 'ShoppingMall': 'median', 'Spa': 'median', 'VRDeck': 'median',
            'TotalSpend': 'median', 'PCT_RoomService': 'mean', 'PCT_FoodCourt': 'mean', 'PCT_ShoppingMall': 'mean',
            'PCT_Spa': 'mean', 'PCT_VRDeck': 'mean'
        }).rename(columns={
            'PassengerNo': 'GroupCount',
            'VIP': 'GroupCountVIP', 'CryoSleep': 'GroupCountCryoSleep', 'Cabin': 'GroupCountCabin',
            'CabinDeck': 'GroupCountCabinDeck', 'CabinNum': 'GroupCountCabinNum', 'CabinSide': 'GroupCountCabinSide',
            'Age': 'GroupMedianAge', 'HomePlanet': 'GroupMeanCountHomePlanet',
            'Destination': 'GroupMeanCountDestination', 'RoomService': 'GroupMedianRoomService',
            'FoodCourt': 'GroupMedianFoodCourt', 'ShoppingMall': 'GroupMedianShoppingMall', 'Spa': 'GroupMedianSpa',
            'VRDeck': 'GroupMedianVRDeck', 'TotalSpend': 'GroupMedianTotalSpend',
            'PCT_RoomService': 'GroupMeanPCT_RoomService', 'PCT_FoodCourt': 'GroupMeanPCT_FoodCourt',
            'PCT_ShoppingMall': 'GroupMeanPCT_ShoppingMall', 'PCT_Spa': 'GroupMeanPCT_Spa',
            'PCT_VRDeck': 'GroupMeanPCT_VRDeck'
        }))

        if self.fill_group:
            x = merge(x, self.__group_data, on='PassengerGroup', how='left')

        if self.one_hot_encoding:
            for name in self.__ohe_encoder.keys():
                self.__ohe_encoder[name].fit(x[[name]])
                newfeatures = self.__ohe_encoder[name].get_feature_names_out()
                x[newfeatures] = self.__ohe_encoder[name].transform(x[[name]]).toarray()

        if self.scaling:
            for name in self.__scaling_encoder.keys():
                enc = self.__scaling_encoder[name]
                x[[name]] = enc.fit_transform(x[[name]])

        self.__drop_unused_features(x)

        return x

    def fit(self, x, y=None):
        data = x
        self.fit_transform(data, y)
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

        bool_features = ['VIP', 'CryoSleep', 'CabinSide']
        if self.fill_missing:
            x[bool_features] = x[bool_features].fillna(self.__bool_features_mode_values)
            x[bool_features] = x[bool_features].astype(bool)

        spend_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        if self.fill_missing:
            x[spend_features] = x[spend_features].fillna(self.__cryo_grouped_median)

        x['TotalSpend'] = x['RoomService'] + x['FoodCourt'] + x['ShoppingMall'] + x['Spa'] + x['VRDeck']

        pct_spend_features = ['PCT_RoomService', 'PCT_FoodCourt', 'PCT_ShoppingMall', 'PCT_Spa', 'PCT_VRDeck']
        x[pct_spend_features] = x[spend_features].apply(lambda v: v / x['TotalSpend'])
        if self.fill_missing:
            x[pct_spend_features] = x[pct_spend_features].fillna(0)  # clear pct/0

        if self.fill_missing:
            x['Age'] = x['Age'].fillna(self.__age_grouped_median)

        # Instances in the same group must be similar
        x['PassengerGroup'] = x['PassengerId'].apply(lambda v: v.split('_')[0])
        x['PassengerNo'] = x['PassengerId'].apply(lambda v: v.split('_')[1])

        if self.fill_group:
            x = merge(x, self.__group_data, on='PassengerGroup', how='left')

        if self.one_hot_encoding:
            for name in self.__ohe_encoder:
                newfeatures = self.__ohe_encoder[name].get_feature_names_out()
                x[newfeatures] = self.__ohe_encoder[name].transform(x[[name]]).toarray()

        if self.scaling:
            for name in self.__scaling_encoder.keys():
                enc = self.__scaling_encoder[name]
                x[[name]] = enc.transform(x[[name]])

        self.__drop_unused_features(x)

        return x
