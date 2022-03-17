from autogluon.tabular import TabularPredictor
from pandas import read_csv, DataFrame
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict

from SpaceshipTitanicData import SpaceshipTitanicPreprocessing

data = read_csv("../input/spaceship-titanic-train.csv")
data = data.sample(frac=1).reset_index(drop=True)

preprocessing = SpaceshipTitanicPreprocessing(
    fill_missing=True,
    fill_total_spend=True,
    fill_percent_spend=True,
    fill_group=True,
    scaling=True,
    one_hot_encoding=True
)
data = preprocessing.fit_transform(data)


model = TabularPredictor.load("AutogluonModels/ag-20220316_162606/")
print("HERE")
print(model.info())
exit()
'''

X = data.drop(labels=['Transported'], axis='columns')
Y = data['Transported']
y_pred = cross_val_predict(model, X, Y, cv=10)

print(confusion_matrix(Y, y_pred))
print(classification_report(Y, y_pred))
print("Accuracy: {:.4}%".format(100 * accuracy_score(Y, y_pred)))
'''

minutes = 0.5
predictor = TabularPredictor(label='Transported', eval_metric='accuracy').fit(
    data, time_limit=minutes * 60, presets='best_quality')

predictor.leaderboard(data, silent=True)
print(predictor.info())

test = read_csv("../input/spaceship-titanic-test.csv")
passid = test['PassengerId']
test = preprocessing.fit_transform(test)
y_pred = predictor.predict(test)

result = DataFrame({'PassengerId': passid, 'Transported': y_pred})
result.to_csv('../output/spaceship-titanic-result.csv', index=False)

"""
Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.0s of the 565.14s of remaining time.
	0.8557	 = Validation score   (accuracy)
	4.23s	 = Training   runtime
	0.02s	 = Validation runtime
AutoGluon training complete, total runtime = 3039.12s ... Best model: "WeightedEnsemble_L3"
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("AutogluonModels/ag-20220316_162606/")
"""
