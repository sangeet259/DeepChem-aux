import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import autosklearn.regression
import json
import pickle


import numpy as np
# Load the data
X = np.load("X_reg.npy")
Y = np.load("Y_reg.npy")

Y = np.log10(Y)



sc_y = RobustScaler().fit(Y.reshape(-1, 1))
Y = sc_y.transform(Y.reshape(-1, 1))
Y = Y.reshape(-1,)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# X, y = sklearn.datasets.load_boston(return_X_y=True)
feature_types = (['numerical'] * 101)# + ['categorical'] + (['numerical'] * 9)
# X_train, X_test, y_train, y_test = \
#     sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.regression.AutoSklearnRegressor(per_run_time_limit=30)
#     time_left_for_this_task=120,
#     per_run_time_limit=30,
#     tmp_folder='/tmp/autosklearn_regression_example_tmp',
#     output_folder='/tmp/autosklearn_regression_example_out',
# )

automl.fit(X_train, Y_train)#, dataset_name='molc',feat_type=feature_types)

print(automl.show_models())
predictions = automl.predict(X_test)
print("R2 score:", sklearn.metrics.r2_score(Y_test, predictions))


fig, ax = plt.subplots(figsize=(20, 10))
ax.scatter(Y_test,predictions, s= 10)
plt.xlabel('Target solubility classes')
plt.ylabel('Predicted solubility classes')
plt.title('Performance of final Model')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.45, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.savefig('auto.png',bbox_inches='tight')


finalres = list(zip(Y_test,predictions))

with open('auto_predict.json','w') as handle:
    json.dump(finalres,handle)

with open('auto_model.pkl','w') as handle:
    json.dump(automl,handle)