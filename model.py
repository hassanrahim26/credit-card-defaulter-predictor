import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('UCI_Credit_Card.csv')

df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30000 entries, 0 to 29999
Data columns (total 25 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   ID         30000 non-null  int64
 1   LIMIT_BAL  30000 non-null  int64
 2   SEX        30000 non-null  int64
 3   EDUCATION  30000 non-null  int64
 4   MARRIAGE   30000 non-null  int64
 5   AGE        30000 non-null  int64
 6   PAY_0      30000 non-null  int64
 7   PAY_2      30000 non-null  int64
 8   PAY_3      30000 non-null  int64
 9   PAY_4      30000 non-null  int64
 10  PAY_5      30000 non-null  int64
 11  PAY_6      30000 non-null  int64
 12  BILL_AMT1  30000 non-null  int64
 13  BILL_AMT2  30000 non-null  int64
 14  BILL_AMT3  30000 non-null  int64
 15  BILL_AMT4  30000 non-null  int64
 16  BILL_AMT5  30000 non-null  int64
 17  BILL_AMT6  30000 non-null  int64
 18  PAY_AMT1   30000 non-null  int64
 19  PAY_AMT2   30000 non-null  int64
 20  PAY_AMT3   30000 non-null  int64
 21  PAY_AMT4   30000 non-null  int64
 22  PAY_AMT5   30000 non-null  int64
 23  PAY_AMT6   30000 non-null  int64
 24  defaulted  30000 non-null  int64
dtypes: int64(25)
memory usage: 5.7 MB
"""

df.EDUCATION.value_counts()
"""
2    14030
1    10585
3     4917
5      280
4      123
6       51
0       14
Name: EDUCATION, dtype: int64
"""

# As per data dictionary there are 5 categories 1 to 5 for Education column and 5 and 6 are both showing as 'UNKNOWN'. There is no **0** category in the dictionary but present in dataset.
# - Hence Combining `0, 5, and 6` together as **'UNKNOWN'** category. Changing all `6 and 0` to `5`.
df['EDUCATION'].replace([0, 6], 5, inplace = True)

df.EDUCATION.value_counts()
"""
2    14030
1    10585
3     4917
5      345
4      123
Name: EDUCATION, dtype: int64
"""

rows = len(df.axes[0])
cols = len(df.axes[1])
print("Number of Rows:- ", rows)
print("Number of Cols:-", cols)
"""
Number of Rows:-  30000
Number of Cols:- 25
"""

print(df.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30000 entries, 0 to 29999
Data columns (total 25 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   ID         30000 non-null  int64
 1   LIMIT_BAL  30000 non-null  int64
 2   SEX        30000 non-null  int64
 3   EDUCATION  30000 non-null  int64
 4   MARRIAGE   30000 non-null  int64
 5   AGE        30000 non-null  int64
 6   PAY_0      30000 non-null  int64
 7   PAY_2      30000 non-null  int64
 8   PAY_3      30000 non-null  int64
 9   PAY_4      30000 non-null  int64
 10  PAY_5      30000 non-null  int64
 11  PAY_6      30000 non-null  int64
 12  BILL_AMT1  30000 non-null  int64
 13  BILL_AMT2  30000 non-null  int64
 14  BILL_AMT3  30000 non-null  int64
 15  BILL_AMT4  30000 non-null  int64
 16  BILL_AMT5  30000 non-null  int64
 17  BILL_AMT6  30000 non-null  int64
 18  PAY_AMT1   30000 non-null  int64
 19  PAY_AMT2   30000 non-null  int64
 20  PAY_AMT3   30000 non-null  int64
 21  PAY_AMT4   30000 non-null  int64
 22  PAY_AMT5   30000 non-null  int64
 23  PAY_AMT6   30000 non-null  int64
 24  defaulted  30000 non-null  int64
dtypes: int64(25)
memory usage: 5.7 MB
None
"""

df.SEX.value_counts()
"""
2    18112
1    11888
Name: SEX, dtype: int64
"""

df.MARRIAGE.value_counts()
"""
2    15964
1    13659
3      323
0       54
Name: MARRIAGE, dtype: int64
"""

# As per data dictionary there are 3 categories 1 to 3 for Marriage column but **0** category present in dataset.
# - Hence Combining `0` as **'Others'** category. Changing all `0` to `3`.
df['MARRIAGE'].replace(0, 3, inplace = True)

df.MARRIAGE.value_counts()
"""
2    15964
1    13659
3      377
Name: MARRIAGE, dtype: int64
"""

df.PAY_2.value_counts()
"""
 0    15730
-1     6050
 2     3927
-2     3782
 3      326
 4       99
 1       28
 5       25
 7       20
 6       12
 8        1
Name: PAY_2, dtype: int64
"""

df.PAY_0.value_counts()
"""
 0    14737
-1     5686
 1     3688
-2     2759
 2     2667
 3      322
 4       76
 5       26
 8       19
 6       11
 7        9
Name: PAY_0, dtype: int64
"""

# Dropping id column as it's no use
df.drop('ID',axis=1, inplace=True)

# Putting feature variable to X
X = df.drop('defaulted',axis=1)

# Putting response variable to y
y = df['defaulted']

# Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Running the random forest with default parameters.
rfc = RandomForestClassifier()

# fit
rfc.fit(X_train,y_train)
"""
RandomForestClassifier()
"""

# Making predictions
predictions = rfc.predict(X_test)

# Checking the report of our default model
print(classification_report(y_test,predictions))
"""
              precision    recall  f1-score   support

           0       0.84      0.94      0.89      7058
           1       0.64      0.36      0.46      1942

    accuracy                           0.82      9000
   macro avg       0.74      0.65      0.68      9000
weighted avg       0.80      0.82      0.80      9000
"""

print(confusion_matrix(y_test,predictions))
"""
[[6669  389]
 [1238  704]]
"""

print(accuracy_score(y_test,predictions))
"""
0.8192222222222222
"""

"""
Hyperparameter Tuning
The following hyperparameters are present in a random forest classifier. We will tune each parameters:-
> n_estimators
> criterion
> max_features
> max_depth
> min_samples_split
> min_samples_leaf
> min_weight_fraction_leaf
> max_leaf_nodes
> min_impurity_split
"""

# Now we'll try to find the optimum values for max_depth and understand how the value of max_depth impacts the overall accuracy of the ensemble.

# Specifying number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(2, 20, 5)}

# Instantiating the model
rf = RandomForestClassifier()

# Fitting tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                 return_train_score=True)
"""
GridSearchCV(cv=5, estimator=RandomForestClassifier(),
             param_grid={'max_depth': range(2, 20, 5)}, return_train_score=True,
             scoring='accuracy')
"""

rf.fit(X_train, y_train)

# Scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
"""
	mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_max_depth	params	split0_test_score	split1_test_score	split2_test_score	split3_test_score	...	mean_test_score	std_test_score	rank_test_score	split0_train_score	split1_train_score	split2_train_score	split3_train_score	split4_train_score	mean_train_score	std_train_score
0	0.916575	0.028496	0.039343	0.002959	2	{'max_depth': 2}	0.804762	0.804048	0.801190	0.795952	...	0.800762	0.003422	4	0.801488	0.799881	0.799286	0.802024	0.800476	0.800631	0.001007
1	2.286264	0.027437	0.056545	0.000696	7	{'max_depth': 7}	0.814762	0.820000	0.817381	0.815952	...	0.817095	0.001752	1	0.833274	0.831369	0.830833	0.832262	0.832202	0.831988	0.000836
2	3.502665	0.041415	0.082894	0.002275	12	{'max_depth': 12}	0.813571	0.819286	0.820476	0.813095	...	0.817000	0.003059	2	0.883155	0.883571	0.881845	0.885060	0.882857	0.883298	0.001049
3	4.351939	0.021023	0.106443	0.003200	17	{'max_depth': 17}	0.812143	0.818810	0.817857	0.810476	...	0.814762	0.003201	3	0.928750	0.929702	0.929940	0.927976	0.927560	0.928786	0.000931
"""

# Plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], scores["mean_train_score"], label="training accuracy")
plt.plot(scores["param_max_depth"], scores["mean_test_score"], label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# As as we increase the value of max_depth, both train and test scores increase till a point, but after that test score starts to decrease. 
# The ensemble tries to overfit as we increase the max_depth.
# Thus, controlling the depth of the constituent trees will help reduce overfitting in the forest.


# Now we'll to find the optimum values for n_estimators and understand how the value of n_estimators impacts the overall accuracy. 
# We'll specify an appropriately low value of max_depth, so that the trees do not overfit.

n_folds = 5

# parameters to build the model on
parameters = {'n_estimators': range(100, 1500, 400)}

# Instantiating the model (Here we are specifying a max_depth)
rf = RandomForestClassifier(max_depth=4)

rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                 return_train_score=True)
rf.fit(X_train, y_train)
"""
GridSearchCV(cv=5, estimator=RandomForestClassifier(max_depth=4),
             param_grid={'n_estimators': range(100, 1500, 400)},
             return_train_score=True, scoring='accuracy')
"""

# Scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
"""
	mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_n_estimators	params	split0_test_score	split1_test_score	split2_test_score	split3_test_score	...	mean_test_score	std_test_score	rank_test_score	split0_train_score	split1_train_score	split2_train_score	split3_train_score	split4_train_score	mean_train_score	std_train_score
0	1.511325	0.049755	0.045183	0.002718	100	{'n_estimators': 100}	0.811429	0.813810	0.811667	0.804524	...	0.810000	0.003205	2	0.810714	0.811905	0.811845	0.811786	0.812857	0.811821	0.000679
1	7.704338	0.064056	0.215283	0.007631	500	{'n_estimators': 500}	0.812857	0.813810	0.810000	0.805000	...	0.810000	0.003177	2	0.811548	0.810238	0.812440	0.812857	0.812738	0.811964	0.000977
2	13.618155	0.108421	0.386318	0.013737	900	{'n_estimators': 900}	0.811429	0.813571	0.811429	0.805714	...	0.810048	0.002786	1	0.811488	0.810714	0.811667	0.812857	0.812500	0.811845	0.000760
3	19.999926	0.122974	0.594780	0.057989	1300	{'n_estimators': 1300}	0.811429	0.813333	0.811190	0.805476	...	0.810000	0.002723	2	0.811131	0.810060	0.811786	0.813095	0.812679	0.811750	0.001088
"""

# plotting accuracies with n_estimators
plt.figure()
plt.plot(scores["param_n_estimators"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_n_estimators"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Now we'll check how the model performance varies with max_features, which is the maximum numbre of features considered for splitting at a node.
n_folds = 5

# parameters to build the model on
parameters = {'max_features': [4, 8, 14, 20, 24]}

# Instantiating the model
rf = RandomForestClassifier(max_depth=4)

# Fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                 return_train_score=True)
"""
GridSearchCV(cv=5, estimator=RandomForestClassifier(max_depth=4),
             param_grid={'max_features': [4, 8, 14, 20, 24]},
             return_train_score=True, scoring='accuracy')
"""

# Scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
"""
	mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_max_features	params	split0_test_score	split1_test_score	split2_test_score	split3_test_score	...	mean_test_score	std_test_score	rank_test_score	split0_train_score	split1_train_score	split2_train_score	split3_train_score	split4_train_score	mean_train_score	std_train_score
0	1.541519	0.059716	0.048376	0.005476	4	{'max_features': 4}	0.814286	0.812381	0.810952	0.805238	...	0.810238	0.003166	4	0.812976	0.810119	0.812560	0.812083	0.812143	0.811976	0.000983
1	2.601289	0.047781	0.044778	0.001845	8	{'max_features': 8}	0.817857	0.821190	0.821429	0.816429	...	0.819190	0.001920	1	0.821071	0.820357	0.820833	0.821548	0.821488	0.821060	0.000440
2	4.241854	0.063571	0.045549	0.002329	14	{'max_features': 14}	0.817619	0.821190	0.819762	0.817619	...	0.818714	0.001510	2	0.822143	0.820893	0.821607	0.822381	0.822321	0.821869	0.000559
3	5.891822	0.110236	0.045341	0.002587	20	{'max_features': 20}	0.818333	0.820476	0.820238	0.817143	...	0.818619	0.001501	3	0.822857	0.820893	0.822262	0.822679	0.822679	0.822274	0.000718
4	0.044063	0.001419	0.000000	0.000000	24	{'max_features': 24}	NaN	NaN	NaN	NaN	...	NaN	NaN	5	NaN	NaN	NaN	NaN	NaN	NaN	NaN
5 rows × 21 columns
"""

# Plotting accuracies with max_features
plt.figure()
plt.plot(scores["param_max_features"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_features"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_features")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# Apparently, the training and test scores both seem to increase as we increase max_features,
# and the model doesn't seem to overfit more with increasing max_features. Think about why that might be the case.

# Now we'll check how the model performance varies with min_samples_leaf
# The hyperparameter, min_samples_leaf is the minimum number of samples required to be at a leaf node:-
# If int, then consider min_samples_leaf as the minimum number.
# If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

n_folds = 5

# Parameters to build the model on
parameters = {'min_samples_leaf': range(100, 400, 50)}

# Instantiating the model
rf = RandomForestClassifier()

# Fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                 return_train_score=True)

rf.fit(X_train, y_train)
"""
GridSearchCV(cv=5, estimator=RandomForestClassifier(),
             param_grid={'min_samples_leaf': range(100, 400, 50)},
             return_train_score=True, scoring='accuracy')
"""

# Scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
"""
	mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_min_samples_leaf	params	split0_test_score	split1_test_score	split2_test_score	split3_test_score	...	mean_test_score	std_test_score	rank_test_score	split0_train_score	split1_train_score	split2_train_score	split3_train_score	split4_train_score	mean_train_score	std_train_score
0	3.077505	0.270715	0.075196	0.011725	100	{'min_samples_leaf': 100}	0.814286	0.819048	0.820000	0.811190	...	0.815333	0.003578	1	0.817738	0.818690	0.818690	0.817560	0.817976	0.818131	0.000476
1	2.607727	0.203855	0.069227	0.009062	150	{'min_samples_leaf': 150}	0.811905	0.814048	0.812857	0.807381	...	0.810667	0.002863	2	0.812500	0.811905	0.811190	0.814405	0.811012	0.812202	0.001222
2	2.337224	0.067343	0.064520	0.006625	200	{'min_samples_leaf': 200}	0.810952	0.810238	0.811190	0.805952	...	0.809238	0.002023	3	0.811786	0.808333	0.810417	0.811905	0.811310	0.810750	0.001317
3	2.171087	0.061381	0.059740	0.002682	250	{'min_samples_leaf': 250}	0.809524	0.812143	0.809524	0.803095	...	0.808286	0.003039	5	0.808512	0.808095	0.808333	0.810357	0.811429	0.809345	0.001315
4	2.609055	0.474512	0.078751	0.025844	300	{'min_samples_leaf': 300}	0.811429	0.811190	0.807857	0.804524	...	0.808333	0.002656	4	0.808155	0.808036	0.807202	0.809940	0.809881	0.808643	0.001086
5 rows × 21 columns
"""

# Plotting accuracies with min_samples_leaf
plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Now we'll check how the model performance varies with  min_samples_split.

n_folds = 5

# Parameters to build the model on
parameters = {'min_samples_split': range(200, 500, 50)}

# Instantiating the model
rf = RandomForestClassifier()

rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                 return_train_score=True)

rf.fit(X_train, y_train)
"""
GridSearchCV(cv=5, estimator=RandomForestClassifier(),
             param_grid={'min_samples_split': range(200, 500, 50)},
             return_train_score=True, scoring='accuracy')
"""

# Scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
"""
	mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_min_samples_split	params	split0_test_score	split1_test_score	split2_test_score	split3_test_score	...	mean_test_score	std_test_score	rank_test_score	split0_train_score	split1_train_score	split2_train_score	split3_train_score	split4_train_score	mean_train_score	std_train_score
0	4.417149	0.677480	0.081390	0.014138	200	{'min_samples_split': 200}	0.818095	0.821429	0.820714	0.816429	...	0.819095	0.001803	1	0.824345	0.822083	0.823512	0.824345	0.823631	0.823583	0.000827
1	3.880176	0.334471	0.085965	0.011881	250	{'min_samples_split': 250}	0.818333	0.821190	0.820952	0.815952	...	0.819048	0.001917	2	0.823155	0.822321	0.822500	0.822083	0.822976	0.822607	0.000401
2	3.519414	0.108144	0.076175	0.010984	300	{'min_samples_split': 300}	0.817381	0.821429	0.821429	0.816190	...	0.818571	0.002367	4	0.822083	0.821607	0.822440	0.821667	0.822024	0.821964	0.000304
3	3.380801	0.264644	0.068113	0.000631	350	{'min_samples_split': 350}	0.817143	0.821905	0.820238	0.816429	...	0.818714	0.002046	3	0.821012	0.821488	0.820417	0.821488	0.821488	0.821179	0.000423
4	3.482911	0.506514	0.072949	0.009837	400	{'min_samples_split': 400}	0.816905	0.821429	0.822381	0.815000	...	0.818476	0.002891	5	0.820536	0.819881	0.819702	0.820655	0.820119	0.820179	0.000367
5 rows × 21 columns
"""

# Plotting accuracies with min_samples_split
plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Grid Search to Find Optimal Hyperparameters
# We can now find the optimal hyperparameters using GridSearchCV.

# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [4,8,10],
    'min_samples_leaf': range(100, 400, 200),
    'min_samples_split': range(200, 500, 200),
    'n_estimators': [100,200, 300], 
    'max_features': [5, 10]
}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
"""
Fitting 3 folds for each of 72 candidates, totalling 216 fits
GridSearchCV(cv=3, estimator=RandomForestClassifier(), n_jobs=-1,
             param_grid={'max_depth': [4, 8, 10], 'max_features': [5, 10],
                         'min_samples_leaf': range(100, 400, 200),
                         'min_samples_split': range(200, 500, 200),
                         'n_estimators': [100, 200, 300]},
             verbose=1)
"""

# Printing the optimal accuracy score and hyperparameters
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
"""
We can get accuracy of 0.8183809523809525 using {'max_depth': 4, 'max_features': 10, 'min_samples_leaf': 100, 'min_samples_split': 200, 'n_estimators': 300}
"""

type(grid_search.best_params_)
"""
dict
"""

# Model with the best hyperparameters
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=4,
                             min_samples_leaf=100, 
                             min_samples_split=200,
                             max_features=10,
                             n_estimators=300)

rfc.fit(X_train,y_train)
"""
RandomForestClassifier(max_depth=4, max_features=10, min_samples_leaf=100,
                       min_samples_split=200, n_estimators=300)
"""

# Let's check the report of our default model
print(classification_report(y_test,predictions))
"""
              precision    recall  f1-score   support

           0       0.84      0.96      0.90      7058
           1       0.69      0.35      0.47      1942

    accuracy                           0.83      9000
   macro avg       0.77      0.66      0.68      9000
weighted avg       0.81      0.83      0.80      9000
"""

# Printing confusion matrix
print(confusion_matrix(y_test,predictions))
"""
[[6752  306]
 [1253  689]]
"""

print(accuracy_score(y_test,predictions))
"""
0.8267777777777777
"""

# Saving the model to disk
pickle.dump(rfc, open('model.pkl', 'wb'))
