import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

#read the raw data in excel
rawdata = pd.read_excel("Global_Superstore2.xlsx",parse_dates= ["Order Date", "Ship Date"]) #51290*24 dim
data = rawdata.drop(columns=["Row ID","Order ID","Customer ID","Customer Name","City","State","Country","Postal Code","Profit"],axis=1) #removing the other columns not interested for my use case
data.dropna(inplace=True) #if 0 drops rows with 0 and 1 drops rows with missing values
#creating features
od = data["Order Date"]
sd = data["Ship Date"]
data["Days_to_Ship_Product"] = abs(sd - od)  #first feature #creating the no.of.days to ship the product as it would also be an important factor for product demand
#2nd feature = Extracting the month of the order date - important for predicting demand monthwise
data['Order month'] = pd.DatetimeIndex(data['Order Date']).month
#3rd feature #we have sales column which is the total sales for all quantites.
data["Price"] = ((data["Sales"] / data["Quantity"]) * (data["Discount"]+1)) #this will give the price of the product for one quantity along with discount.
#print(dataclean)
data["Ship Mode"] = data["Ship Mode"].astype('category')
data["Ship Mode cat"] = data["Ship Mode"].cat.codes
data["Segment"] = data["Segment"].astype('category')
data["Segment cat"] = data["Segment"].cat.codes
data["Market"] = data["Market"].astype('category')
data["Market cat"] = data["Market"].cat.codes
data["Region"] = data["Region"].astype('category')
data["Region cat"] = data["Region"].cat.codes
data["Order Priority"] = data["Order Priority"].astype('category')
data["Order Priority cat"] = data["Order Priority"].cat.codes
data["Category"] = data["Category"].astype('category')
data["Category cat"] = data["Category"].cat.codes
data["Sub-Category"] = data["Sub-Category"].astype('category')
data["Sub-Category cat"] = data["Sub-Category"].cat.codes
data["Days_to_Ship_Product"] = data["Days_to_Ship_Product"].astype('str')
data["Product shipping Days"] = [str(i).split(" ")[0] if " " in i else "not days" for i in
                                            data["Days_to_Ship_Product"]]
#print(data) # now the data is cleaned with new features created.
#final df preprocessed and cleaned # removing sales column as it is correlated to price
finaldf = data.drop(columns=["Sales", "Ship Mode","Segment","Market","Region","Order Priority", "Product ID","Product Name","Category","Sub-Category","Days_to_Ship_Product"])
#print(finaldf)
finaldf.isna().sum()  # to check which column has null values.

#Using Pearson Correlation - to check features correlation
plt.figure(figsize=(12,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#Correlation with output variable
cor_target = abs(cor["Quantity"])
relevant_features = cor_target[cor_target>0.5] # this is just a cross check to see if there are highly correlated variables and to remove them and have only 1 of the two
#print(relevant_features)
#fitting linear reg model for features selected
#X = pd.DataFrame(np.c_[finaldf['Discount'], finaldf['Shipping Cost'],finaldf['Price'], finaldf['Ship Mode cat'],finaldf['Segment cat'],finaldf['Market cat'],finaldf['Order month'],finaldf['Region cat'], finaldf['Order Priority cat'], finaldf['Category cat'],finaldf['Sub-Category cat'],finaldf['Product shipping Days']],
                 #columns=['Discount','Shipping Cost','Price','Ship Mode cat','Segment cat','Market cat','Order month','Region cat', 'Order Priority cat','Category cat','Sub-Category cat','Product shipping Days'])
#target = np.array(data["Quantity"])
#print(target)

#modeling #XGboost
cleandf = finaldf.iloc[:,3:15]
#cleandf.to_excel(r'C:\Users\gltla\OneDrive - Bayer\Personal Data\Thesis\mlflow\cleandfxgb.xlsx')
cleandf['Product shipping Days'] = cleandf['Product shipping Days'].astype(str).astype(int)
print(cleandf)
y= finaldf.iloc[:,2]
print(y)
xtrain, xtest, ytrain, ytest = train_test_split(cleandf, y, test_size=0.20) #quantity is my y variable target
print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)
xgbr = xgb.XGBRegressor(booster='gbtree', colsample_bylevel=1, colsample_bytree=1, learning_rate=0.1,
                                     objective="reg:linear",max_depth = 5, alpha = 10, n_estimators = 300,random_state=42)
xgbr.fit(xtrain, ytrain)
score = xgbr.score(xtrain, ytrain)
print("Training score: ", score)
# - cross validataion
scores = cross_val_score(xgbr, xtrain, ytrain, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())
kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbr, xtrain, ytrain, cv=kfold)
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
ypred = xgbr.predict(xtest)  #prediction on testdata --------- ypred is quantity - its coming in decimal because model is not so accurate
print("Checking for types to plot")
print(type(xtest)) #df
print(type(ypred)) #array
print(type(ytest)) #series, dtype - int64
print(xtest)
print(ypred) #plot this against ytest---actual values #10258 records
print(ytest)
plt.figure()
plt.scatter(ytest,ypred)
plt.xlabel("Actual")
plt.ylabel("Predictions")
plt.title("Actual vs Predictions")
plt.show()

xtestbkp = xtest.copy(deep=True)
xtestbkp['ytest'] = ytest
xtestbkp['ypred'] = ypred
print(xtestbkp)

#xtestbkp.to_excel("Test_Df_with_predicions.xlsx")
#reverting back to the labels

#encoded_labels = [2, 0, 1]
#decoded_labels = le.inverse_transform(encoded_labels)
#print("Encoded labels =", encoded_labels)
#print("Decoded labels =", list(decoded_labels))

xtestdf = xtestbkp.drop(columns = ['Discount','Order month','Shipping Cost','Price','Ship Mode cat','Segment cat','Market cat','Order Priority cat','Sub-Category cat','Product shipping Days'],axis = 1)
#xtestdf.plot(x="Order month", y=["ytest", "ypred"],figsize=(12,10),legend=True)
#plt.show()
#predplotsre = xtestdf.groupby(by=['Region cat']).count()
#predplotsre.plot(x="Order month", y=["ytest", "ypred"],figsize=(12,10),legend=True)
#plt.ylabel('Quantity')
#plt.show()
xtestdf1 = xtestbkp.drop(columns = ['Discount','Order month','Shipping Cost','Price','Ship Mode cat','Segment cat','Market cat','Order Priority cat','Sub-Category cat','Product shipping Days','Region cat'],axis = 1)
predplotscat = xtestdf.groupby(by=['Region cat','Category cat']).count().unstack()
predplotscat.columns = predplotscat.columns.droplevel()
predplotscat.reset_index().plot.bar()
plt.ylabel('Quantity')
plt.show()
#for pr, group in xtestdf1.groupby("Category cat"):
    #group.plot(x="Order month",y=["ytest","ypred"],legend=True, figsize=(14, 10))
    #group["ypred"].plot(x="Order month", legend=True, figsize=(14, 10), label = pr)
#plt.legend(loc=(1.05, 0.0))
#plt.tight_layout()
#plt.xlabel('Order month')
#plt.ylabel('Quantity')
#plt.show()
(xtestdf.groupby(['Category cat','Region cat'])['ypred','ytest'].count().unstack('Category cat').plot.bar())
plt.show()
#xtestdf1.groupby(['Category cat'])['Category cat'].count().plot.bar()
#plt.show()
#overall best performing category


#checking for metrics
mse = mean_squared_error(ytest, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse ** (1 / 2.0)))
r2xg = metrics.r2_score(ytest,ypred)
print("R-Squared in XGBtree:",r2xg)
#building 3 fold cv model in xgboost - so change these hyper parms and checking for rmse and r2
data_dmatrix = xgb.DMatrix(data=cleandf,label=y)
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print(cv_results.head())
print((cv_results["test-rmse-mean"]).tail(1))
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
#feature importance plot
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
#actual vs predicted plot
x_ax = xtest['Order month'].values
plt.plot(x_ax, ytest, color="blue", label="original")
plt.plot(x_ax, ypred, color="red", label="predicted")
plt.legend()
plt.title("Product demand actual vs predicted plot")
plt.show()

#Randomforest model
# Instantiate model with 500 decision trees
rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
# Train the model on training data
rf.fit(xtrain, ytrain)
predictions = rf.predict(xtest) # Use the forest's predict method on the test data
mserf = mean_squared_error(ytest,predictions)
rmserf = np.sqrt(mserf)
r2 = metrics.r2_score(ytest,predictions)
print('Mean Absolute Error of RF :', metrics.mean_absolute_error(ytest, predictions))
print('MSE OF Random forest model:', mserf)
print('RMSE OF Random forest model:', rmserf)
print('R-Squared of RF model:',r2)
#feature importance plot #Extract the two most important features
target = np.array(data["Quantity"])
# Saving feature names for later use
feature_list = list(cleandf.columns) #cleandf = features has all data other than target
print(feature_list)
# Convert to numpy array
features = np.array(cleandf)
#variable importance
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
plt.style.use('fivethirtyeight') #set the style and plot importance
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
plt.show()


#decision tree
decisiontreereg = DecisionTreeRegressor(max_depth=10)
decisiontreereg.fit(xtrain, ytrain)
predictions = decisiontreereg.predict(xtest) # Use the forest's predict method on the test data
msedt = mean_squared_error(ytest,predictions)
rmsedt = np.sqrt(mserf)
r2dt = metrics.r2_score(ytest,predictions)
print('Mean Absolute Error of DT :', metrics.mean_absolute_error(ytest, predictions))
print("MSE OF DT model:", msedt)
print("RMSE OF DT model:", rmsedt)
print("R-Squared of DT model:",r2dt)











