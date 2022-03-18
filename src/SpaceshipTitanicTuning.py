from lightgbm import LGBMClassifier
from pandas import read_csv, DataFrame
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    BaggingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold, \
    SequentialFeatureSelector, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.SpaceshipTitanicData import SpaceshipTitanicPreprocessing

PREPROCESSOR = SpaceshipTitanicPreprocessing(scaling=True, one_hot_encoding=True)
PREPROCESSOR_GRID = {'preprocessing': PREPROCESSOR}
ESTIMATOR = LGBMClassifier()
FEATURE_SELECTOR = RFE(ESTIMATOR, n_features_to_select=20)
FEATURE_SELECTOR_GRID = {'reduce_dim': ['passthrough']}
NUM_FEATURES_TO_SELECT = [10, 20, 30]

param_grid_preprocessing = [{
    'preprocessing': [SpaceshipTitanicPreprocessing()],
    'preprocessing__fill_missing': [True, False],
    'preprocessing__fill_total_spend': [True, False],
    'preprocessing__fill_percent_spend': [True, False],
    'preprocessing__fill_group': [True, False],
    # 'preprocessing__scaling':[True, False],
    # 'preprocessing__one_hot_encoding':[True, False]
}]

param_grid_features = [{
    'reduce_dim': ['passthrough'],
    **PREPROCESSOR_GRID
}, {
    'reduce_dim': [SelectKBest()],
    'reduce_dim__k': NUM_FEATURES_TO_SELECT,
    'reduce_dim__score_func': [chi2, f_classif, mutual_info_classif],
    **PREPROCESSOR_GRID
}, {
    'reduce_dim': [VarianceThreshold()],
    'reduce_dim__threshold': [0.0, 0.01, 0.1, 0.15, 0.2],
    **PREPROCESSOR_GRID
}, {
    'reduce_dim': [LocallyLinearEmbedding(eigen_solver='dense')],
    'reduce_dim__n_neighbors': [7, 9],
    # 'reduce_dim__method': ['standard', 'hessian', 'modified', 'ltsa'],
    **PREPROCESSOR_GRID
}, {
    'reduce_dim': [PCA()],
    'reduce_dim__n_components': NUM_FEATURES_TO_SELECT,
    **PREPROCESSOR_GRID
}, {
    'reduce_dim': [RFE(ESTIMATOR)],
    'reduce_dim__n_features_to_select': NUM_FEATURES_TO_SELECT,
    **PREPROCESSOR_GRID
}, {
    'reduce_dim': [SequentialFeatureSelector(ESTIMATOR)],
    'reduce_dim__n_features_to_select': [NUM_FEATURES_TO_SELECT[-1]],
    'reduce_dim__direction': ['backward'],
    **PREPROCESSOR_GRID
}
]

META_ESTIMATORS = [
    DecisionTreeClassifier(max_depth=2),
    LogisticRegression(solver='newton-cg')
]

ENSEMBLE_ESTIMATORS = [[
    ('SVM', SVC()),
    ('NB', GaussianNB()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('kNN1', KNeighborsClassifier(n_neighbors=3, weights='uniform')),
    ('kNN2', KNeighborsClassifier(n_neighbors=3, weights='distance')),
    ('kNN3', KNeighborsClassifier(n_neighbors=7, weights='uniform')),
    ('kNN4', KNeighborsClassifier(n_neighbors=7, weights='distance')),
    ('DT1', DecisionTreeClassifier(random_state=1, max_features=0.8, criterion='gini')),
    ('DT2', DecisionTreeClassifier(random_state=2, max_features=0.8, criterion='entropy')),
    ('DT3', DecisionTreeClassifier(random_state=3, max_features=1.0, criterion='gini')),
    ('DT4', DecisionTreeClassifier(random_state=4, max_features=1.0, criterion='entropy')),
    ('LR1', LogisticRegression(C=1.0, solver='newton-cg')),
    ('LR2', LogisticRegression(C=0.9, solver='newton-cg')),
    ('LR3', LogisticRegression(C=0.8, solver='newton-cg')),
    ('LR4', LogisticRegression(C=0.7, solver='newton-cg')),
    ('MLP1', MLPClassifier(early_stopping=True, random_state=1, hidden_layer_sizes=(10,), max_iter=500)),
    ('MLP2', MLPClassifier(early_stopping=True, random_state=2, hidden_layer_sizes=(25,), max_iter=500)),
    ('MLP3', MLPClassifier(early_stopping=True, random_state=3, hidden_layer_sizes=(50,), max_iter=500)),
    ('MLP4', MLPClassifier(early_stopping=True, random_state=4, hidden_layer_sizes=(100,), max_iter=500))
], [
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
], [
    ('LGB', LGBMClassifier(random_state=0, n_estimators=50)),
    ('LGB', LGBMClassifier(random_state=1)),
    ('ETC', ExtraTreesClassifier(random_state=0, n_estimators=50)),
    ('ETC', ExtraTreesClassifier(random_state=1)),
    ('RFC', RandomForestClassifier(random_state=0, n_estimators=50)),
    ('RFC', RandomForestClassifier(random_state=1)),
    ('GBC', GradientBoostingClassifier(random_state=0, n_estimators=50)),
    ('GBC', GradientBoostingClassifier(random_state=1)),
    ('XBG', XGBClassifier(objective="binary:logistic", eval_metric='logloss', use_label_encoder=False, random_state=0,
                          n_estimators=50)),
    ('XBG', XGBClassifier(objective="binary:logistic", eval_metric='logloss', use_label_encoder=False, random_state=1))
]
]

param_grid_models = [{
    'classifier': [LinearDiscriminantAnalysis()],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [GaussianNB()],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [LogisticRegression(solver='newton-cg')],
    'classifier__C': [1.0, 0.9, 0.7, 0.5],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [DecisionTreeClassifier()],
    'classifier__criterion': ['gini', 'entropy'],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [KNeighborsClassifier()],
    'classifier__n_neighbors': [3, 7, 11],
    'classifier__weights': ['uniform', 'distance'],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [RandomForestClassifier()],
    'classifier__n_estimators': [10, 50, 100, 150, 200],
    'classifier__criterion': ['gini', 'entropy'],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [ExtraTreesClassifier()],
    'classifier__n_estimators': [10, 50, 100, 150, 200],
    'classifier__criterion': ['gini', 'entropy'],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [GradientBoostingClassifier()],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__n_estimators': [10, 50, 100, 150, 200],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [LGBMClassifier()],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__n_estimators': [10, 50, 100, 150, 200],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [XGBClassifier(objective="binary:logistic", eval_metric='logloss', use_label_encoder=False)],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__n_estimators': [10, 50, 100, 150, 200],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [SVC()],
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [MLPClassifier(early_stopping=True)],
    'classifier__activation': ['tanh', 'logistic', 'relu'],
    'classifier__alpha': [0.01, 0.1, 0.15],
    'classifier__hidden_layer_sizes': [(10,), (50,), (100,)],  # , (50,)],
    'classifier__max_iter': [200, 500, 1000],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [BaggingClassifier(n_jobs=3)],
    'classifier__base_estimator': META_ESTIMATORS,
    'classifier__n_estimators': [10, 25, 50],
    'classifier__max_samples': [1.0, 0.8],
    'classifier__max_features': [1.0, 0.8],
    'classifier__bootstrap': [True, False],
    'classifier__bootstrap_features': [True, False],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))],
    'classifier__base_estimator': META_ESTIMATORS,
    'classifier__n_estimators': [10, 50, 100, 150, 200],
    'classifier__learning_rate': [0.8, 1, 1.2],
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [VotingClassifier(estimators=[('LDA', LinearDiscriminantAnalysis())])],
    'classifier__estimators': ENSEMBLE_ESTIMATORS,
    **FEATURE_SELECTOR_GRID
}, {
    'classifier': [StackingClassifier(estimators=[('LDA', LinearDiscriminantAnalysis())])],
    'classifier__estimators': ENSEMBLE_ESTIMATORS,
    'classifier__final_estimator': META_ESTIMATORS,
    **FEATURE_SELECTOR_GRID
}
]

data = read_csv("../input/spaceship-titanic-train.csv")
# data = data.sample(frac=1).reset_index(drop=True)

PREPROCESSING = SpaceshipTitanicPreprocessing(
    fill_missing=True,
    fill_total_spend=True,  # +1 = 18
    fill_percent_spend=True,  # +5 = 23
    fill_group=True,  # +21 = 44
    scaling=True,
    one_hot_encoding=True  # 17 features
)

param_grid = param_grid_preprocessing

pipe = Pipeline([
    ('preprocessing', PREPROCESSOR),
    ("reduce_dim", FEATURE_SELECTOR),
    ("classifier", ESTIMATOR),
])

X = data.drop(labels=['Transported'], axis='columns')
Y = data['Transported']

grid = GridSearchCV(pipe, param_grid=param_grid, scoring='accuracy', cv=10, error_score='raise', verbose=10, n_jobs=3)
grid.fit(X, Y)

DataFrame(grid.cv_results_).to_csv('../output/spaceship-titanic-gridsearch-model-result.csv', index=False)

print("Best Estimator: ", grid.best_estimator_)
print('Best parameter:', grid.best_params_)
print('Best parameter score (CV score=%0.4f):' % grid.best_score_)

model = grid.best_estimator_
y_pred = cross_val_predict(model, X, Y, cv=10)

print(confusion_matrix(Y, y_pred))
print(classification_report(Y, y_pred))
print("Accuracy: {:.4}%".format(100 * accuracy_score(Y, y_pred)))
