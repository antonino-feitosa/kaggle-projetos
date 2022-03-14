
from pandas import read_csv
from pandas._config.config import option_context
from pandas.core.frame import DataFrame
from sklearn.compose._column_transformer import ColumnTransformer, \
    make_column_selector
from sklearn.preprocessing._data import MinMaxScaler
from sklearn.preprocessing._encoders import OneHotEncoder

from SpaceshipTitanicData import SpaceshipTitanicPreprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.model_selection._validation import cross_val_predict
from sklearn.metrics._classification import confusion_matrix, \
    classification_report, accuracy_score
from catboost.core import CatBoostClassifier
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier

data = read_csv("../input/spaceship-titanic-train.csv")
data = data.sample(frac=1).reset_index(drop=True)

X = data.drop(labels=['Transported'], axis='columns')
Y = data['Transported'].astype(bool)

pass1 = SpaceshipTitanicPreprocessing()

pass2 = ColumnTransformer(transformers=[
    ('one_hot_enco', OneHotEncoder(), make_column_selector(dtype_include="category")),
    ('scale', MinMaxScaler(), make_column_selector(dtype_exclude="category"))])

preprocessing = Pipeline([('feature_eng', pass1), ('encode_scale', pass2)])

model = XGBClassifier(use_label_encoder=False, n_estimators=10000)

pipe = Pipeline([
        ('preprocessing', preprocessing),
        ("reduce_dim", "passthrough"),
        ("classifier", model),
    ])

time_limit = 60  # for quick demonstration only, you should set this to longest time you are willing to wait (in seconds)
metric = 'roc_auc'  # specify your evaluation metric here
predictor = TabularPredictor(label, eval_metric=metric).fit(train_data, time_limit=time_limit, presets='best_quality')
predictor.leaderboard(test_data, silent=True)

y_pred = cross_val_predict(pipe, X, Y, cv=10)

print(confusion_matrix(Y, y_pred))
print(classification_report(Y, y_pred))
print("Accuracy: {:.4}%".format(100 * accuracy_score(Y, y_pred)))

'''
data = pass1.fit_transform(data)
features = list(data.columns)
print(features)
data = pass2.fit_transform(data)

data = DataFrame(data, columns=pass2.get_feature_names_out(features))
with option_context('display.max_columns', 40):
    print(data.describe())

'''
