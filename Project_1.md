## Salary Prediction for NHL Players 

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from datetime import datetime
import gc
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
In [14]: train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
test_x = pd.read_csv('test.csv', encoding = "ISO-8859-1")
test_y = pd.read_csv('test_salaries.csv')
In [16]: train.info()
In [17]: obj_cols = train.select_dtypes('object')
obj_cols.head()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 612 entries, 0 to 611
Columns: 154 entries, Salary to GS/G
dtypes: float64(73), int64(71), object(10)
memory usage: 736.4+ KB
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
In [18]: obj_cols = train.select_dtypes('object')
obj_cols.head()
In [19]: train.Born.head()
In [23]: for c in obj_cols.columns:
print('Obj Col: ', c, ' Number of Unqiue Values ->', len(obj_cols
[c].value_counts()))
Out[17]: Boorrnn Ciittyy Prr//Stt Cnnttrryy Naatt Haanndd LLaasstt Naamee FFiirrsstt Naamee Poossiittiioonn TTeeaam
0 97-01-30 Sainte-Marie QC CAN CAN L Chabot Thomas D OTT
1 93-12-21 Ottawa ON CAN CAN R Ceci Cody D OTT
2 88-04-16 St. Paul MN USA USA R Okposo Kyle RW BUF
3 92-01-07 Ottawa ON CAN CAN R Gudbranson Erik D VAN
4 94-03-29 Toronto ON CAN CAN R Wilson Tom RW WSH
Out[18]:
Born City Pr/St Cntry Nat Hand Last Name First Name Position Team
0 97-01-30 Sainte-Marie QC CAN CAN L Chabot Thomas D OTT
1 93-12-21 Ottawa ON CAN CAN R Ceci Cody D OTT
2 88-04-16 St. Paul MN USA USA R Okposo Kyle RW BUF
3 92-01-07 Ottawa ON CAN CAN R Gudbranson Erik D VAN
4 94-03-29 Toronto ON CAN CAN R Wilson Tom RW WSH
Out[19]: 0 97-01-30
1 93-12-21
2 88-04-16
3 92-01-07
4 94-03-29
Name: Born, dtype: object
Obj Col: Born Number of Unqiue Values -> 576
Obj Col: City Number of Unqiue Values -> 373
Obj C l P /S N b f U i V l 37
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
In [24]: fig, ax=plt.subplots(1,2,figsize=(18,10))
obj_cols['Cntry'].value_counts().sort_values().plot(kind='barh',ax=ax[0
])
ax[0].set_title("Counts of Hockey Players by Country");
obj_cols['Cntry'].value_counts().plot(kind='pie', autopct='%.2f', shado
w=True,ax=ax[1]);
ax[1].set_title("Distribution of Hockey Players by Country");
In [25]: fig, ax=plt.subplots(1,1,figsize=(12,8))
Obj Col: Pr/St Number of Unqiue Values -> 37
Obj Col: Cntry Number of Unqiue Values -> 18
Obj Col: Nat Number of Unqiue Values -> 16
Obj Col: Hand Number of Unqiue Values -> 2
Obj Col: Last Name Number of Unqiue Values -> 573
Obj Col: First Name Number of Unqiue Values -> 308
Obj Col: Position Number of Unqiue Values -> 18
Obj Col: Team Number of Unqiue Values -> 68
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
obj_cols['Team'].value_counts().plot(kind='bar',ax=ax);
plt.title('Counts of Team Values');
In [27]: sal_gtmil = train[train.Salary >= 1e7]
In [28]: sal_gtmil.head(10)
Out[28]:
Salary Born City Pr/St Cntry Nat Ht Wt DftYr DftRd ... PEND OPS DP
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
In [29]: import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
In [30]: def data_clean(x):
## Were going to change Born to date time
x['Born'] = pd.to_datetime(x.Born, yearfirst=True)
x['dowBorn'] = x.Born.dt.dayofweek
x["doyBorn"] = x.Born.dt.dayofyear
x["monBorn"] = x.Born.dt.month
x['yrBorn'] = x.Born.dt.year
Salary Born City Pr/St Cntry Nat Ht Wt DftYr DftRd ... PEND OPS DP
103 13800000
88-
11-
19
Buffalo NY USA USA 71 177 2007.0 1.0 ... 91.0 8.7 2
145 10900000
87-
08-
07
Cole
Harbour NS CAN CAN 71 200 2005.0 1.0 ... 110.0 10.5 1
208 11000000
89-
05-
13
Toronto ON CAN CAN 72 210 2007.0 2.0 ... 99.0 3.6 2
226 10000000
89-
08-
15
Kladno NaN CZE CZE 74 214 2007.0 1.0 ... 112.0 4.6 1
260 10000000
85-
09-
17
Moscow NaN RUS RUS 75 239 2004.0 1.0 ... 72.0 7.0 1
496 13800000
88-
04-
29
Winnipeg MB CAN CAN 74 201 2006.0 1.0 ... 78.0 4.7 1
542 12000000
85-
08-
14
Sicamous BC CAN CAN 76 232 2003.0 2.0 ... 115.0 4.1 6
7 rows × 154 columns
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
## Drop Pr/St due to NaNs from other countries and First Name
x.drop(['Pr/St','First Name'], axis=1, inplace=True)
ocols = ['City', 'Cntry', 'Nat', 'Last Name', 'Position', 'Team']
for oc in ocols:
temp = pd.get_dummies(x[oc])
x = x.join(temp, rsuffix=str('_'+oc))
x['Hand'] = pd.factorize(x.Hand)[0]
x.drop(ocols, axis=1, inplace=True)
x.drop(['Born'],axis=1,inplace=True)
return x
In [31]: try:
del train, x0, xc, test
except:
pass
In [34]: train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
In [35]: train.head()
Out[35]:
Salary Born City Pr/St Cntry Nat Ht Wt DftYr DftRd ... PEND OPS DPS P
0 925000
97-
01-
30
Sainte-
Marie QC CAN CAN 74 190 2015.0 1.0 ... 1.0 0.0 -0.2 -0
1 2250000
93-
12-
21
Ottawa ON CAN CAN 74 207 2012.0 1.0 ... 98.0 -0.2 3.4 3
2 8000000
88-
04-
16
St.
Paul MN USA USA 72 218 2006.0 1.0 ... 70.0 3.7 1.3 5
3 3500000
92-
01-
07
Ottawa ON CAN CAN 77 220 2010.0 1.0 ... 22.0 0.0 0.4 0
4 1750000
94-
03-
29
Toronto ON CAN CAN 76 217 2012.0 1.0 ... 68.0 -0.1 1.4 1
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
In [37]: test_x = pd.read_csv('test.csv', encoding = "ISO-8859-1")
##test_y = pd.read_csv('test_salaries.csv')
In [39]: test_x.head()
In [41]: full = train.merge(test_x, how='outer')
#print(train.shape, test.shape, full.shape)
In [42]: print(train.shape, test_x.shape, full.shape)
In [43]: y = np.log(full.Salary.dropna())
5 rows × 154 columns
Out[39]:
Born City Pr/St Cntry Nat Ht Wt DftYr DftRd Ovrl ... PEND OPS DPS PS
0
88-
11-
05
Ithaca NY USA USA 72 216 2003.0 1.0 13.0 ... 65 1.9 1.8 3.7
1
00-
02-
29
Prague NaN CZE CZE 72 195 2014.0 1.0 13.0 ... 10 0.3 0.3 0.6
2
92-
04-
24
St.
Louis MO USA USA 75 227 2007.0 6.0 161.0 ... 86 3.9 2.0 6.0
3
99-
07-
05
Piikkio NaN FIN FIN 72 182 2013.0 2.0 55.0 ... 40 2.3 1.1 3.4
4
96-
10-
27
Niagara
Falls NY USA USA 72 196 2011.0 2.0 36.0 ... 25 0.8 1.1 1.9
5 rows × 153 columns
(612, 154) (262, 153) (874, 154)
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
full0 = full.drop(['Salary'],axis=1)
In [44]: fig, ax=plt.subplots(1,1,figsize=(10,6))
y.plot(ax=ax);
plt.title("Ln Salary");
In [45]: obj_cols.columns
In [46]: full_c = data_clean(full0)
In [47]: print(full0.shape, full_c.shape)
Out[45]: Index(['Born', 'City', 'Pr/St', 'Cntry', 'Nat', 'Hand', 'Last Name',
'First Name', 'Position', 'Team'],
dtype='object')
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
In [48]: full_c.head()
In [49]: ss = StandardScaler()
In [50]: full_cs = ss.fit_transform(full_c)
In [51]: train_c = full_cs[:612]
test_c = full_cs[612:]
In [52]: print(train_c.shape, y.shape, test_c.shape)
In [53]: type(y)
In [54]: folds = 3
lgbm_params = {
"max_depth": -1,
"num_leaves": 1000,
(874, 155) (874, 1573)
Out[48]:
Ht Wt DftYr DftRd Ovrl Hand GP G A A1 ... S.J S.J/VAN STL STL/WSH T.B
0 74 190 2015.0 1.0 18.0 0 1 0 0 0 ... 0 0 0 0 0
1 74 207 2012.0 1.0 15.0 1 79 2 15 6 ... 0 0 0 0 0
2 72 218 2006.0 1.0 7.0 1 65 19 26 13 ... 0 0 0 0 0
3 77 220 2010.0 1.0 3.0 1 30 1 5 5 ... 0 0 0 0 0
4 76 217 2012.0 1.0 16.0 1 82 7 12 4 ... 0 0 0 0 0
5 rows × 1573 columns
(612, 1573) (612,) (262, 1573)
Out[53]: pandas.core.series.Series
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
"learning_rate": 0.01,
"n_estimators": 1000,
"objective":'regression',
'min_data_in_leaf':64,
'feature_fraction': 0.8,
'colsample_bytree':0.8,
"metric":['mae','mse'],
"boosting_type": "gbdt",
"n_jobs": -1,
"reg_lambda": 0.9,
"random_state": 123
}
preds = 0
for f in range(folds):
xt, xv, yt, yv = train_test_split(train_c, y.values, test_size=0.2,
random_state=((f+1)*123))
xtd = lgb.Dataset(xt, label=yt)
xvd = lgb.Dataset(xv, label=yv)
mod = lgb.train(params=lgbm_params, train_set=xtd,
num_boost_round=1000, valid_sets=xvd, valid_names=[
'valset'],
early_stopping_rounds=20, verbose_eval=20)
preds += mod.predict(test_c)
preds = preds/folds
C:\Users\shefe\Anaconda3\lib\site-packages\lightgbm\engine.py:148: User
Warning: Found `n_estimators` in params. Will use it instead of argumen
t
warnings.warn("Found `{}` in params. Will use it instead of argumen
t".format(alias))
Training until validation scores don't improve for 20 rounds
[20] valset's l1: 0.673578 valset's l2: 0.564863
[40] valset's l1: 0.598066 valset's l2: 0.453872
[60] valset's l1: 0.540044 valset's l2: 0.380566
[80] valset's l1: 0.492135 valset's l2: 0.330633
[100] valset's l1: 0.453573 valset's l2: 0.295675
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
[120] valset's l1: 0.425478 valset's l2: 0.271764
[140] valset's l1: 0.402047 valset's l2: 0.254423
[160] valset's l1: 0.387692 valset's l2: 0.241881
[180] valset's l1: 0.376087 valset's l2: 0.231166
[200] valset's l1: 0.367421 valset's l2: 0.223362
[220] valset's l1: 0.360755 valset's l2: 0.217925
[240] valset's l1: 0.355547 valset's l2: 0.214305
[260] valset's l1: 0.350838 valset's l2: 0.211131
[280] valset's l1: 0.347377 valset's l2: 0.208286
[300] valset's l1: 0.344892 valset's l2: 0.206366
[320] valset's l1: 0.342563 valset's l2: 0.204723
[340] valset's l1: 0.340609 valset's l2: 0.203764
[360] valset's l1: 0.339329 valset's l2: 0.203372
[380] valset's l1: 0.337826 valset's l2: 0.202722
[400] valset's l1: 0.336645 valset's l2: 0.202358
[420] valset's l1: 0.335847 valset's l2: 0.20191
Early stopping, best iteration is:
[415] valset's l1: 0.335947 valset's l2: 0.201834
Training until validation scores don't improve for 20 rounds
[20] valset's l1: 0.698804 valset's l2: 0.619348
[40] valset's l1: 0.633312 valset's l2: 0.520693
[60] valset's l1: 0.579445 valset's l2: 0.452746
[80] valset's l1: 0.536438 valset's l2: 0.404334
[100] valset's l1: 0.503174 valset's l2: 0.369588
[120] valset's l1: 0.481706 valset's l2: 0.349145
[140] valset's l1: 0.465743 valset's l2: 0.335575
[160] valset's l1: 0.451074 valset's l2: 0.323382
[180] valset's l1: 0.440976 valset's l2: 0.313053
[200] valset's l1: 0.433398 valset's l2: 0.306632
[220] valset's l1: 0.426894 valset's l2: 0.301786
[240] valset's l1: 0.420267 valset's l2: 0.297452
[260] valset's l1: 0.415042 valset's l2: 0.294957
[280] valset's l1: 0.409745 valset's l2: 0.292286
[300] valset's l1: 0.405328 valset's l2: 0.290555
[320] valset's l1: 0.40155 valset's l2: 0.28921
[340] valset's l1: 0.398574 valset's l2: 0.288649
Early stopping, best iteration is:
[334] valset's l1: 0.398907 valset's l2: 0.288303
Training until validation scores don't improve for 20 rounds
[20] valset's l1: 0.690878 valset's l2: 0.590683
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
In [57]: acts = pd.read_csv('test_salaries.csv', encoding="ISO-8859-1")
acts['preds'] = np.exp(preds)
acts.head()
In [58]: import matplotlib
[40] valset's l1: 0.634363 valset's l2: 0.502408
[60] valset's l1: 0.588503 valset's l2: 0.44036
[80] valset's l1: 0.551728 valset's l2: 0.394662
[100] valset's l1: 0.519885 valset's l2: 0.361632
[120] valset's l1: 0.494413 valset's l2: 0.331885
[140] valset's l1: 0.476429 valset's l2: 0.311994
[160] valset's l1: 0.4616 valset's l2: 0.297144
[180] valset's l1: 0.45009 valset's l2: 0.285993
[200] valset's l1: 0.441458 valset's l2: 0.277851
[220] valset's l1: 0.435554 valset's l2: 0.272794
[240] valset's l1: 0.429608 valset's l2: 0.267513
[260] valset's l1: 0.423502 valset's l2: 0.262572
[280] valset's l1: 0.418379 valset's l2: 0.258796
[300] valset's l1: 0.413119 valset's l2: 0.255104
[320] valset's l1: 0.408946 valset's l2: 0.252298
[340] valset's l1: 0.404917 valset's l2: 0.249567
[360] valset's l1: 0.401932 valset's l2: 0.247393
[380] valset's l1: 0.399283 valset's l2: 0.245806
[400] valset's l1: 0.398316 valset's l2: 0.245166
Early stopping, best iteration is:
[394] valset's l1: 0.397988 valset's l2: 0.244716
Out[57]:
Salary preds
0 7000000.0 3.813335e+06
1 925000.0 8.500037e+05
2 2000000.0 3.165594e+06
3 667500.0 1.066605e+06
4 600000.0 8.832996e+05
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
from sklearn.metrics import mean_absolute_error, mean_squared_error
In [59]: fig, ax=plt.subplots(1,1,figsize=(12,8))
acts.plot(ax=ax, style=['b-','r-']);
plt.title("Comparison of Preds and Actuals");
plt.ylabel('$');
ax.get_yaxis().set_major_formatter(
matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()
In [60]: mse = mean_squared_error(np.log(acts.Salary), np.log(acts.preds))
mae = mean_absolute_error(np.log(acts.Salary), np.log(acts.preds))
In [61]: print("Ln Level Mean Squared Error :", mse)
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
print("Ln Level Mean Absolute Error :", mae)
In [62]: fi_df = pd.DataFrame( 100*mod.feature_importance()/mod.feature_importan
ce().max(),
index=full_c.columns, #mod.feature_name(),
columns =['importance'])
In [63]: fig, ax=plt.subplots(1,1,figsize=(12,8))
fi_df.sort_values(by='importance',ascending=True).iloc[-20:].plot(kind=
'barh', color='C0', ax=ax)
plt.title("Normalized Feature Importances");
Ln Level Mean Squared Error : 0.32829220938672277
Ln Level Mean Absolute Error : 0.4263148368980229
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
In [64]: import statsmodels.api as sma
In [65]: top10 = fi_df.sort_values(by='importance',ascending=True).iloc[-10:].in
dex
top10
exog = pd.DataFrame(test_c, columns=full_c.columns)[list(top10)].fillna
(0)
In [66]: ols = sma.OLS(exog=exog, endog=acts.Salary)
ols_fit = ols.fit()
print(ols_fit.summary())
OLS Regression Results
=======================================================================
================
Dep. Variable: Salary R-squared (uncentered):
0.734
Model: OLS Adj. R-squared (uncentered):
0.723
Method: Least Squares F-statistic:
69.44
Date: Sat, 07 Aug 2021 Prob (F-statistic):
1.30e-66
Time: 23:41:30 Log-Likelihood:
-4143.6
No. Observations: 262 AIC:
8307.
Df Residuals: 252 BIC:
8343.
Df Model: 10
Covariance Type: nonrobust
=======================================================================
=======
coef std err t P>|t| [0.025
0.975]
-----------------------------------------------------------------------
-------
doyBorn 2.51e+05 1.17e+05 2.154 0.032 2.15e+04
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
4.8e+05
ixG 3.132e+05 1.79e+05 1.746 0.082 -4e+04
6.67e+05
BLK% -4.775e+04 1.23e+05 -0.388 0.699 -2.9e+05
1.95e+05
GS/G 4.906e+05 1.81e+05 2.713 0.007 1.34e+05
8.47e+05
DftRd -1.227e+05 7.7e+05 -0.159 0.874 -1.64e+06
1.39e+06
TOI/GP.1 1.197e+07 1.07e+07 1.118 0.264 -9.11e+06
3.3e+07
TOI/GP -1.104e+07 1.07e+07 -1.034 0.302 -3.21e+07
9.99e+06
yrBorn 3.513e+06 1.73e+05 20.314 0.000 3.17e+06
3.85e+06
Ovrl 5.344e+04 7.77e+05 0.069 0.945 -1.48e+06
1.58e+06
DftYr -4.159e+06 2.02e+05 -20.579 0.000 -4.56e+06 -
3.76e+06
=======================================================================
=======
Omnibus: 45.090 Durbin-Watson:
1.894
Prob(Omnibus): 0.000 Jarque-Bera (JB):
110.659
Skew: 0.798 Prob(JB):
9.35e-25
Kurtosis: 5.755 Cond. No.
240.
=======================================================================
=======
Notes:
[1] R² is computed without centering (uncentered) since the model does
not contain a constant.
[2] Standard Errors assume that the covariance matrix of the errors is
correctly specified.
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
In [1]: import matplotlib.pyplot as plt
y_pos = np.arange(len(w_results))
weight_variant_names = ["{ 'xgb': 0.33, 'rf': 0.33, 'svm' : 0.34}",
"{ 'xgb': 0.9, 'rf': 0.05, 'svm' : 0.05}",
"{ 'xgb': 0.8, 'rf': 0.1, 'svm' : 0.1}",
"{ 'xgb': 0.5, 'rf': 0.3, 'svm' : 0.2}",
"{ 'xgb': 0.3, 'rf': 0.2, 'svm' : 0.5}",
"{ 'xgb': 0.3, 'rf': 0.5, 'svm' : 0.2}"]
plt.bar(y_pos, w_results, align='center', alpha=0.5)
plt.xticks(y_pos, weight_variant_names, rotation=90)
plt.ylabel('rmse')
plt.ylim(1300000,1450000)
plt.title('RMSE of different ensemble model weights')
plt.show()
In [67]: ols_preds = ols_fit.predict()
In [68]: fig, ax=plt.subplots(1,1,figsize=(12,8))
acts.Salary.plot(ax=ax, color='C1');
ax.plot(ols_preds, color='C0');
plt.title("Comparison of StatsModels Preds and Actuals");
plt.ylabel('$');
-----------------------------------------------------------------------
----
NameError Traceback (most recent call l
ast)
<ipython-input-1-d480ce9800b9> in <module>
1 import matplotlib.pyplot as plt
2
----> 3 y_pos = np.arange(len(w_results))
4
5 weight_variant_names = ["{ 'xgb': 0.33, 'rf': 0.33, 'svm' : 0.3
4}",
NameError: name 'np' is not defined
Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD
plt.legend(['salary actual','ols preds']);
ax.get_yaxis().set_major_formatter(
matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()
In [ ]:
