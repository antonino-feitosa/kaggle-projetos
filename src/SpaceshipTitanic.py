
import types

from lightgbm.sklearn import LGBMClassifier
from pandas import isna
from pandas.core.frame import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.decomposition._pca import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import ExtraTreesClassifier
from sklearn.ensemble._gb import GradientBoostingClassifier
from sklearn.ensemble._stacking import StackingClassifier
from sklearn.ensemble._weight_boosting import AdaBoostClassifier
from sklearn.feature_selection._from_model import SelectFromModel
from sklearn.feature_selection._mutual_info import mutual_info_classif
from sklearn.feature_selection._rfe import RFE
from sklearn.feature_selection._sequential import SequentialFeatureSelector
from sklearn.feature_selection._univariate_selection import SelectKBest, \
    f_classif, chi2
from sklearn.feature_selection._variance_threshold import VarianceThreshold
from sklearn.impute._base import SimpleImputer
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.manifold._locally_linear import LocallyLinearEmbedding
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._validation import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors._classification import KNeighborsClassifier
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._data import MinMaxScaler, StandardScaler
from sklearn.preprocessing._encoders import OneHotEncoder
from sklearn.svm._classes import SVC
from sklearn.tree._classes import DecisionTreeClassifier
from xgboost import XGBClassifier

import numpy as np
import pandas as pd
from sklearn.ensemble._bagging import BaggingClassifier
from sklearn.ensemble._voting import VotingClassifier

# from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.ensemble import HistGradientBoostingClassifier
# CatBoostClassifier
# Pycaret

# normalize RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# probabilities, total amount
# normalize age by max
# column for total spent

# extract group from id and merge with cabin
# cabin deck is ordinal
# missing value on age must consider the home-planet
'''
def create_group_features(df):
    
    
    Group level features
    - Number of passengers
    - Number of VIPs passengers
    - Number of passengers in cryosleep
    - Number of unique cabins
    - Number of unique decks
    - Number of unique sides
    - Mean age of passengers in the group
    - mean spend on various expense area
    - mean total spend
    - Number of unique home planets
    
    
    
    df = (df.groupby('PassengerGroup', as_index = False)
          .agg({'PassengerNo':'nunique',
                'VIP':lambda x: sum(x == True),
                'CryoSleep': lambda x: sum(x == True),
                'Cabin': 'nunique',
                'Deck': 'nunique',
                'Side': 'nunique',
                'Age': 'mean',
                'RoomService': 'mean',
                'FoodCourt': 'mean',
                'ShoppingMall':'mean',
                'Spa':'mean',
                'VRDeck': 'mean',
                'TotalSpend':'mean',
                'HomePlanet': 'nunique'})
          .rename(columns = {'PassengerNo':'Count'})
         )
    
    df['PctRoomService'] = df['RoomService']/df['TotalSpend']
    df['PctFoodCourt'] = df['FoodCourt']/df['TotalSpend']
    df['PctShoppingMall'] = df['ShoppingMall']/df['TotalSpend']
    df['PctSpa'] = df['Spa']/df['TotalSpend']
    df['PctVRDeck'] = df['VRDeck']/df['TotalSpend']
    
    fill_cols = ['PctRoomService', 'PctFoodCourt', 'PctShoppingMall', 'PctSpa', 'PctVRDeck']
    df[fill_cols] = df[fill_cols].fillna(0)
    
    df.columns = [f'Group{i}' if i not in ['PassengerGroup'] else i for i in df.columns]
    
    
    
    return df


def create_features(df):
    
    bool_type = ['VIP', 'CryoSleep']
    df[bool_type] = df[bool_type].astype(bool)
    
    df['PassengerGroup'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['PassengerNo'] = df['PassengerId'].apply(lambda x: x.split('_')[1])
    df.loc[df['Cabin'].isnull(), 'Cabin'] = 'None/None/None'
    
    fill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[fill_cols] = df[fill_cols].fillna(0)
    df['TotalSpend'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['PctRoomService'] = df['RoomService']/df['TotalSpend']
    df['PctFoodCourt'] = df['FoodCourt']/df['TotalSpend']
    df['PctShoppingMall'] = df['ShoppingMall']/df['TotalSpend']
    df['PctSpa'] = df['Spa']/df['TotalSpend']
    df['PctVRDeck'] = df['VRDeck']/df['TotalSpend']
    fill_cols = ['PctRoomService', 'PctFoodCourt', 'PctShoppingMall', 'PctSpa', 'PctVRDeck']
    df[fill_cols] = df[fill_cols].fillna(0)
    
    df['Age'] = df['Age'].fillna(df.groupby('HomePlanet')['Age'].transform('median'))
    df['CryoSleep'] = df['CryoSleep'].fillna(False)
    
    df['Deck'] = df['Cabin'].apply(lambda x: str(x).split('/')[0])
    df['Side'] = df['Cabin'].apply(lambda x: str(x).split('/')[2])
    
    df_group_features = create_group_features(df)    
    
    df = pd.merge(df, df_group_features, on = 'PassengerGroup', how = 'left')
    
    return df
'''

# use automl

# LGBMImputer??

# from flaml import AutoML
# automl = AutoML()
# undummify
# autogluon for hyper
'''
from autogluon.tabular import TabularDataset, TabularPredictor

label = 'Transported'  
eval_metric = 'accuracy'

predictor = TabularPredictor(label=label, eval_metric=eval_metric, verbosity=3).fit(
    train, presets='best_quality', time_limit=3600)
results = predictor.fit_summary()
'''


class IdRemoveTransformer(BaseEstimator, TransformerMixin):
    
    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def transform(self, X, y=None):
        X.drop(labels=['PassengerId', 'Name'], axis='columns', inplace=True)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return []


class CabinTransformer(BaseEstimator, TransformerMixin):
    
    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def transform(self, X, y=None):
        # .str.split('/',expand=True)
        X['CabinDeck'] = X['Cabin'].apply(lambda x: x.split('/')[0] if not isna(x) else np.nan)
        X['CabinNum'] = X['Cabin'].apply(lambda x: x.split('/')[1] if not isna(x) else np.nan)
        X['CabinNum'] = X['CabinNum'].astype(float)
        X['CabinSide'] = X['Cabin'].apply(lambda x: x.split('/')[2] == 'P' if not isna(x) else np.nan)
        X['CabinSide'] = X['CabinSide'].astype(bool)
        X.drop(labels=['Cabin'], axis='columns', inplace=True)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return ['CabinDeck', 'CabinNum', 'CabinSide']


def get_feature_names_out(self, input_features=None):
    return input_features

# features =[
# 0 'PassengerId',
# 1 'HomePlanet',
# 2 'CryoSleep',
# 3 'Cabin',
# 4 'Destination',
# 5 'Age',
# 6 'VIP',
# 7 'RoomService',
# 8 'FoodCourt',
# 9 'ShoppingMall',
# 10 'Spa',
# 11 'VRDeck',
# 12 'Name',
# 13 'Transported']


pass1 = ColumnTransformer(transformers=[
    ('split_cabin', CabinTransformer(), [3]),
    ('del_ids', IdRemoveTransformer(), [0, 12])], remainder='passthrough')

pass2 = SimpleImputer(strategy="most_frequent")
pass2.get_feature_names_out = types.MethodType(get_feature_names_out, pass2)

pass3 = ColumnTransformer(transformers=[
    ('one_hot_enco', OneHotEncoder(handle_unknown="ignore"), [0, 3, 5])], remainder='passthrough')

# pass4 = MinMaxScaler()
pass4 = StandardScaler()

preprocessing = Pipeline([
    ('split_cabin', pass1),
    ('imputer', pass2),
    ('one_hot_enco', pass3),
    ('scaling', pass4)
])

'''
data = pd.read_csv("../input/spaceship-titanic-train.csv")
features = data.columns
data = preprocessing.fit_transform(data)
out_names = preprocessing[:-1].get_feature_names_out(features)

data = DataFrame(data, columns=out_names)
with pd.option_context('display.max_columns', 40):
    print(data.describe())
'''

SEED = None
NUM_FEATURES = 24
ESTIMATOR = LGBMClassifier(random_state=SEED)
FEATURE_SELECTOR = RFE(ESTIMATOR, n_features_to_select=20)  # 81.3%

pipe = Pipeline([
        ('preprocessing', preprocessing),
        ("reduce_dim", FEATURE_SELECTOR),
        # ("reduce_dim", "passthrough"),
        ("classifier", ESTIMATOR),
    ])

FEATURE_SELECTOR_GRID = {
    'reduce_dim': [RFE(ESTIMATOR)],
    'reduce_dim__n_features_to_select': [18, 20, 22]
    }

META_ESTIMATORS = [
            DecisionTreeClassifier(max_depth=2),
            LogisticRegression(random_state=SEED, solver='newton-cg')
        ]

ENSEMBLE_ESTIMATORS = [
        [
            ('SVM', SVC()),
            ('NB', GaussianNB()),
            ('LDA', LinearDiscriminantAnalysis()),
            ('kNN1', KNeighborsClassifier(n_neighbors=3, weights='uniform')),
            ('kNN2', KNeighborsClassifier(n_neighbors=3, weights='distance')),
            ('kNN3', KNeighborsClassifier(n_neighbors=7, weights='uniform')),
            ('kNN4', KNeighborsClassifier(n_neighbors=7, weights='distance')),
            ('DT1', DecisionTreeClassifier(random_state=1, max_features=0.9, criterion='gini')),
            ('DT2', DecisionTreeClassifier(random_state=2, max_features=0.9, criterion='entropy')),
            ('DT3', DecisionTreeClassifier(random_state=3, max_features=0.7, criterion='gini')),
            ('DT4', DecisionTreeClassifier(random_state=4, max_features=0.7, criterion='entropy')),
            ('LR1', LogisticRegression(C=1.0, solver='newton-cg')),
            ('LR2', LogisticRegression(C=0.9, solver='newton-cg')),
            ('LR3', LogisticRegression(C=0.8, solver='newton-cg')),
            ('LR4', LogisticRegression(C=0.7, solver='newton-cg')),
            ('MLP1', MLPClassifier(early_stopping=True, random_state=1, hidden_layer_sizes=(10,), max_iter=500)),
            ('MLP2', MLPClassifier(early_stopping=True, random_state=2, hidden_layer_sizes=(25,), max_iter=500)),
            ('MLP3', MLPClassifier(early_stopping=True, random_state=3, hidden_layer_sizes=(50,), max_iter=500)),
            ('MLP4', MLPClassifier(early_stopping=True, random_state=4, hidden_layer_sizes=(100,), max_iter=500))
        ],
        [
            ('MLP01', MLPClassifier(early_stopping=True, random_state=1, hidden_layer_sizes=(10,), max_iter=500)),
            ('MLP02', MLPClassifier(early_stopping=True, random_state=2, hidden_layer_sizes=(50,), max_iter=500)),
            ('MLP03', MLPClassifier(early_stopping=True, random_state=3, hidden_layer_sizes=(100,), max_iter=500))
        ],
        [
            ('MLP01', MLPClassifier(early_stopping=True, random_state=1, hidden_layer_sizes=(10,), max_iter=500)),
            ('MLP02', MLPClassifier(early_stopping=True, random_state=2, hidden_layer_sizes=(50,), max_iter=500)),
            ('MLP03', MLPClassifier(early_stopping=True, random_state=3, hidden_layer_sizes=(100,), max_iter=500)),
            ('MLP04', MLPClassifier(early_stopping=True, random_state=4, hidden_layer_sizes=(10,), max_iter=500)),
            ('MLP05', MLPClassifier(early_stopping=True, random_state=5, hidden_layer_sizes=(50,), max_iter=500)),
            ('MLP06', MLPClassifier(early_stopping=True, random_state=6, hidden_layer_sizes=(100,), max_iter=500)),
            ('MLP07', MLPClassifier(early_stopping=True, random_state=7, hidden_layer_sizes=(10,), max_iter=500)),
            ('MLP08', MLPClassifier(early_stopping=True, random_state=8, hidden_layer_sizes=(50,), max_iter=500)),
            ('MLP09', MLPClassifier(early_stopping=True, random_state=9, hidden_layer_sizes=(100,), max_iter=500)),
            ('MLP10', MLPClassifier(early_stopping=True, random_state=10, hidden_layer_sizes=(10,), max_iter=500)),
            ('MLP11', MLPClassifier(early_stopping=True, random_state=11, hidden_layer_sizes=(50,), max_iter=500)),
            ('MLP12', MLPClassifier(early_stopping=True, random_state=12, hidden_layer_sizes=(100,), max_iter=500)),
            ('MLP13', MLPClassifier(early_stopping=True, random_state=13, hidden_layer_sizes=(10,), max_iter=500)),
            ('MLP14', MLPClassifier(early_stopping=True, random_state=14, hidden_layer_sizes=(50,), max_iter=500)),
            ('MLP15', MLPClassifier(early_stopping=True, random_state=15, hidden_layer_sizes=(100,), max_iter=500))
        ],
        [
            ('LGB', LGBMClassifier(random_state=SEED)),
            ('ETC', ExtraTreesClassifier(random_state=SEED)),
            ('RFC', RandomForestClassifier(random_state=SEED)),
            ('GBC', GradientBoostingClassifier(random_state=SEED)),
            ('XBG', XGBClassifier(objective="binary:logistic", eval_metric='logloss', use_label_encoder=False, random_state=SEED))
        ]
    ]

param_grid_features = [
    {
        'reduce_dim': ['passthrough']
    },
    {
        'reduce_dim': [SelectKBest()],
        'reduce_dim__k': [4, 8, 12, 16, 20],
        'reduce_dim__score_func': [f_classif, mutual_info_classif]
    },
    {
        'reduce_dim': [VarianceThreshold()],
        'reduce_dim__threshold': [0.0, 0.01, 0.1, 0.15, 0.2]
    },
    {
        'reduce_dim': [SelectFromModel(DecisionTreeClassifier())],
        'reduce_dim__max_features': [4, 8, 12, 16, 20],
    },
    {
        'reduce_dim': [SelectFromModel(RandomForestClassifier(random_state=SEED))],
        'reduce_dim__max_features': [4, 8, 12, 16, 20],
    },
    {
        'reduce_dim': [LocallyLinearEmbedding(random_state=SEED, eigen_solver='dense')],
        'reduce_dim__n_neighbors': [7, 9],
        'reduce_dim__method': ['standard', 'hessian', 'modified', 'ltsa']
    },
    {
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': [4, 8, 12, 16, 20]
    },
    {
        'reduce_dim': [RFE(ESTIMATOR)],
        'reduce_dim__n_features_to_select': [4, 8, 12, 16, 20]
    },
    {
        'reduce_dim': [SequentialFeatureSelector(ESTIMATOR)],
        'reduce_dim__n_features_to_select': [4],
        'reduce_dim__direction': ['forward']
    },
    {
        'reduce_dim': [SequentialFeatureSelector(ESTIMATOR)],
        'reduce_dim__n_features_to_select': [NUM_FEATURES - 4],
        'reduce_dim__direction': [ 'backward']
    }
]

param_grid_models = [
    {
        'classifier': [AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), random_state=SEED)],
        'classifier__base_estimator': META_ESTIMATORS,
        'classifier__n_estimators':[10, 50, 100, 150, 200],
        'classifier__learning_rate': [0.8, 1, 1.2],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [VotingClassifier(estimators=[('LDA', LinearDiscriminantAnalysis())])],
        'classifier__estimators': ENSEMBLE_ESTIMATORS,
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [StackingClassifier(estimators=[('LDA', LinearDiscriminantAnalysis())])],
        'classifier__estimators': ENSEMBLE_ESTIMATORS,
        'classifier__final_estimator': META_ESTIMATORS,
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [LinearDiscriminantAnalysis()],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [GaussianNB()],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [LogisticRegression(random_state=SEED, solver='newton-cg')],
        'classifier__C': [1.0, 0.9, 0.7, 0.5],
        ** FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [DecisionTreeClassifier()],
        'classifier__criterion': ['gini', 'entropy'],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 7, 11],
        'classifier__weights': ['uniform', 'distance'],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [RandomForestClassifier(random_state=SEED)],
        'classifier__n_estimators':[10, 50, 100, 150, 200],
        'classifier__criterion': ['gini', 'entropy'],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [ExtraTreesClassifier(random_state=SEED)],
        'classifier__n_estimators':[10, 50, 100, 150, 200],
        'classifier__criterion': ['gini', 'entropy'],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [GradientBoostingClassifier(random_state=SEED)],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__n_estimators':[10, 50, 100, 150, 200],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [LGBMClassifier(random_state=SEED)],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__n_estimators':[10, 50, 100, 150, 200],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [XGBClassifier(objective="binary:logistic", eval_metric='logloss', use_label_encoder=False, random_state=SEED)],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__n_estimators':[10, 50, 100, 150, 200],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [SVC()],
        'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [MLPClassifier(early_stopping=True, random_state=SEED)],
        'classifier__activation': ['tanh', 'logistic', 'relu'],
        'classifier__alpha':[ 0.01, 0.1, 0.15],
        'classifier__hidden_layer_sizes':[(10,), (NUM_FEATURES,), (100,)],  # , (50,)],
        'classifier__max_iter': [200, 500, 1000],
        **FEATURE_SELECTOR_GRID
    },
    {
        'classifier': [BaggingClassifier(random_state=SEED, n_jobs=3)],
        'classifier__base_estimator': META_ESTIMATORS,
        'classifier__n_estimators': [10, 25, 50],
        'classifier__max_samples': [1.0, 0.8],
        'classifier__max_features': [1.0, 0.8],
        'classifier__bootstrap': [True, False],
        'classifier__bootstrap_features': [True, False],
        **FEATURE_SELECTOR_GRID
    },
    
]

ESTIMATOR = StackingClassifier(estimators=[
            ('LGB', LGBMClassifier(random_state=SEED)),
            ('ETC', ExtraTreesClassifier(random_state=SEED)),
            ('RFC', RandomForestClassifier(random_state=SEED)),
            ('GBC', GradientBoostingClassifier(random_state=SEED)),
            ('XBG', XGBClassifier(objective="binary:logistic", eval_metric='logloss', use_label_encoder=False, random_state=SEED))
        ],
       final_estimator=LogisticRegression(random_state=SEED, solver='newton-cg')
   )
FEATURE_SELECTOR = RFE(LGBMClassifier(random_state=SEED))

ESTIMATOR_GRID = [{
        'classifier': [ESTIMATOR],
        'classifier__passthrough': [True, False],
        'reduce_dim': [FEATURE_SELECTOR],
        'reduce_dim__n_features_to_select': [15, 16, 17, 18, 19, 20, 21, 22, 23]
    },
    {
        'classifier': [ESTIMATOR],
        'classifier__passthrough': [True, False],
        'reduce_dim': ['passthrough']
    }
    ]

param_grid = ESTIMATOR_GRID

print(param_grid)

data = pd.read_csv("../input/spaceship-titanic-train.csv")
data = data.sample(frac=1, random_state=SEED).reset_index(drop=True)

X = data.drop(labels=['Transported'], axis='columns')
Y = data['Transported']

grid = GridSearchCV(pipe, param_grid=param_grid, scoring='accuracy', cv=5, error_score='raise', verbose=10, n_jobs=3)
grid.fit(X, Y)

DataFrame(grid.cv_results_).to_csv('../output/spaceship-titanic-gridsearch-model-result.csv', index=False)

print("Best Estimator: ", grid.best_estimator_)
print('Best parameter:', grid.best_params_)
print('Best parameter score (CV score=%0.3f):' % grid.best_score_)

model = grid.best_estimator_
y_pred = cross_val_predict(model, X, Y, cv=10)

print(confusion_matrix(Y, y_pred))
print(classification_report(Y, y_pred))
print("Accuracy: {:.4}%".format(100 * accuracy_score(Y, y_pred)))

Xt = pd.read_csv("../input/spaceship-titanic-test.csv")
passid = Xt['PassengerId']
model = grid.best_estimator_
model.fit(X, Y)
Yt = model.predict(Xt)

result = pd.DataFrame({'PassengerId': passid, 'Transported':Yt})
result.to_csv('../output/spaceship-titanic-result.csv', index=False)

'''
Best parameter: {'classifier': StackingClassifier(estimators=[('LGB', LGBMClassifier()),
                               ('ETC', ExtraTreesClassifier()),
                               ('RFC', RandomForestClassifier()),
                               ('GBC', GradientBoostingClassifier()),
                               ('XBG',
                                XGBClassifier(base_score=None, booster=None,
                                              colsample_bylevel=None,
                                              colsample_bynode=None,
                                              colsample_bytree=None,
                                              enable_categorical=False,
                                              eval_metric='logloss', gamma=None,
                                              gpu_id=None, import...
                                              min_child_weight=None,
                                              missing=nan,
                                              monotone_constraints=None,
                                              n_estimators=100, n_jobs=None,
                                              num_parallel_tree=None,
                                              predictor=None, random_state=None,
                                              reg_alpha=None, reg_lambda=None,
                                              scale_pos_weight=None,
                                              subsample=None, tree_method=None,
                                              use_label_encoder=False,
                                              validate_parameters=None,
                                              verbosity=None))],
                   final_estimator=LogisticRegression(solver='newton-cg')), 'classifier__passthrough': False, 'reduce_dim': RFE(estimator=LGBMClassifier(), n_features_to_select=19), 'reduce_dim__n_features_to_select': 19}
Best parameter score (CV score=0.812):
[[3444  871]
 [ 760 3618]]
              precision    recall  f1-score   support

       False       0.82      0.80      0.81      4315
        True       0.81      0.83      0.82      4378

    accuracy                           0.81      8693
   macro avg       0.81      0.81      0.81      8693
weighted avg       0.81      0.81      0.81      8693

Accuracy: 81.24%
'''
