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
from yellowbrick.regressor import PredictionError
from yellowbrick.features import Rank1D
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import pydot


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
#print(data) # now the data is cleaned with new features created.
data.describe()
data.isna().sum()  # to check which column has null values.
#label encoding to change string and categorical variables to numeric before checking for feature selection and modeeling
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
#final df preprocessed and cleaned # removing sales column as it is correlated to price
finaldf = data.drop(columns=["Sales", "Ship Mode","Segment","Market","Region","Order Priority", "Product ID","Product Name","Category","Sub-Category","Days_to_Ship_Product"])
print(finaldf)
#Using Pearson Correlation - to check features correlation
plt.figure(figsize=(12,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#Correlation with output variable
cor_target = abs(cor["Quantity"])
relevant_features = cor_target[cor_target>0.5] # this is just a cross check to see if there are highly correlated variables and to remove them and have only 1 of the two
#fitting linear reg model for features selected
X = pd.DataFrame(np.c_[finaldf['Discount'], finaldf['Shipping Cost'],finaldf['Price'], finaldf['Ship Mode cat'],finaldf['Segment cat'],finaldf['Market cat'],finaldf['Order month'],finaldf['Region cat'], finaldf['Order Priority cat'], finaldf['Category cat'],finaldf['Sub-Category cat'],finaldf['Product shipping Days']],
                 columns=['Discount','Shipping Cost','Price','Ship Mode cat','Segment cat','Market cat','Order month','Region cat', 'Order Priority cat','Category cat','Sub-Category cat','Product shipping Days'])
target = np.array(data["Quantity"])
print(target)
#train_x, test_x, train_y, test_y = train_test_split(X, target, test_size=0.3, random_state=10)
#print(train_x.shape, train_y.shape)
#print(test_x.shape, test_y.shape)
#model1=LinearRegression().fit(train_x,train_y)
#print("R2 SCORE:", model1.score(train_x,train_y))# gives R2 value
#pred_y = model1.predict(test_x)
#print("MSE:", mean_squared_error(test_y,pred_y))
#test_set_rmse = (np.sqrt(mean_squared_error(test_y, pred_y)))
#print("RMSE:", test_set_rmse)
#mae = mean_absolute_error(test_y, pred_y)
#print("MAE:", mae)
#evaluating model using a new package yellow brick - checking for parity plots - checking for fit
#lin_reg_visualizer = PredictionError(model1) #using yellowbrick package for parity plot and linear reg model fit
#lin_reg_visualizer.fit(train_x,train_y)
#lin_reg_visualizer.score(test_x, test_y)  # Evaluate the model on the test data
#lin_reg_visualizer.show()  # Finalize and render the figure

#XGboost
cleandf = finaldf.iloc[:,3:15]
#cleandf.to_excel(r'C:\Users\gltla\OneDrive - Bayer\Personal Data\Thesis\mlflow\cleandfxgb.xlsx')
cleandf['Product shipping Days'] = cleandf['Product shipping Days'].astype(str).astype(int)
print(cleandf.dtypes)
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
ypred = xgbr.predict(xtest) #prediction on testdata --------- ypred is quantity - its coming in decimal because model is not so accurate
print(ypred) #plot this against ytest---actual values
mse = mean_squared_error(ytest, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse ** (1 / 2.0)))
r2xg = metrics.r2_score(ytest,ypred)
print("R-Squared in XGBtree:",r2xg)

#feature importance plot
data_dmatrix = xgb.DMatrix(data=cleandf,label=y)
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
#actual vs predicted plot
x_ax = range(len(ytest))
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
print("MSE OF Random forest model:", mserf)
print("RMSE OF Random forest model:", rmserf)
print("R-Squared of RF model:",r2)
# plots using RF to get pred vs actual, first finding feature importnace and then plotting
# Extract the two most important features
target = np.array(data["Quantity"])
# Saving feature names for later use
feature_list = list(cleandf.columns) # cleandf = features has all data other than target
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
# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
#myorder = [1,3,8,4,0,11,2,7,6,10,5,9]
#feature_list = [feature_list[i] for i in myorder]
important_indices = [feature_list.index('Shipping Cost'), feature_list.index('Price')]
dt = xtrain.values
dt = dt.astype('float32')
train_size = int(len(dt))
train_dataset = dt[0:train_size,:]
print(type(train_dataset))
dtest = xtest.values
dtest = dtest.astype('float32')
test_size = int(len(dtest))
test_dataset = dt[0:test_size,:]
train_important = train_dataset[:, important_indices]
print(train_important)
test_important = test_dataset[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, ytrain)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - ytest)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / ytest))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
plt.show()
#get orderdate column and plot with quantity
dates = finaldf['Order month']
labels = finaldf['Quantity']
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
test_dates = xtest['Order month']
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60')
plt.legend()
# Graph labels
plt.xlabel('Month of the order'); plt.ylabel('Quantity of product'); plt.title('Actual and Predicted Values')
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











