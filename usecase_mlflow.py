import os
import warnings
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from mlflow import log_metric, log_param, log_artifacts
import mlflow.sklearn
import logging
from urllib.parse import urlparse
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mse, rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = ("Global_Superstore2.xlsx")
    try:
        rawdata = pd.read_excel(csv_url, parse_dates= ["Order Date", "Ship Date"])
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    data = rawdata.drop(columns=["Row ID", "Order ID", "Customer ID", "Customer Name", "City", "State", "Country", "Postal Code",
                 "Profit"], axis=1)  # removing the other columns not interested for my use case
    data.dropna(inplace=True)  # if 0 drops rows with 0 and 1 drops rows with missing values
    # creating features #1st feature
    od = data["Order Date"]
    sd = data["Ship Date"]
    data["Days_to_Ship_Product"] = abs(sd - od)  #creating the no.of.days to ship the product as it would also be an important factor for product demand
    # 2nd feature = Extracting the month of the order date - important for predicting demand monthwise
    data['Order month'] = pd.DatetimeIndex(data['Order Date']).month
    # 3rd feature #we have sales column which is the total sales for all quantites.
    data["Price"] = ((data["Sales"] / data["Quantity"]) * (
                data["Discount"] + 1))  # this will give the price of the product for one quantity along with discount.
    # print(data) # now the data is cleaned with new features created.
    data.describe()
    data.isna().sum()  # to check which column has null values.
    # label encoding to change string and categorical variables to numeric before checking for feature selection and modeeling
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
    # final df preprocessed and cleaned # removing sales column as it is correlated to price
    finaldf = data.drop(
        columns=["Sales", "Ship Mode", "Segment", "Market", "Region", "Order Priority", "Product ID", "Product Name",
                 "Category", "Sub-Category", "Days_to_Ship_Product"])
    # print(finaldf)
    # Using Pearson Correlation - to check features correlation
    fig = plt.figure(figsize=(12, 10))
    cor = data.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    fig.savefig("Pearson Correaltion-Rawdata.png")

    # modeling - first model: XGboost
    cleandf = finaldf.iloc[:, 3:15]
    # cleandf.to_excel(r'C:\Users\gltla\OneDrive - Bayer\Personal Data\Thesis\mlflow\cleandfxgb.xlsx')
    cleandf['Product shipping Days'] = cleandf['Product shipping Days'].astype(str).astype(int)
    #print(cleandf)
    y = finaldf.iloc[:, 2]
    #print(y)
    #train-test split
    xtrain, xtest, ytrain, ytest = train_test_split(cleandf, y, test_size=0.20)  # quantity is my y variable target
    mlflow.end_run()
    experiment_id = mlflow.create_experiment("Prediction Final15")
    with mlflow.start_run(experiment_id=experiment_id):
        xgbr = xgb.XGBRegressor(colsample_bylevel=1, colsample_bytree=1, learning_rate=0.1,
                                objective="reg:linear", max_depth=5, alpha=10, n_estimators=300, random_state=42)
        xgbr.fit(xtrain, ytrain)
        score = xgbr.score(xtrain, ytrain)
        print("Training score: ", score)
        ypred = xgbr.predict(xtest)  # prediction on testdata --------- ypred is quantity - its coming in decimal because model is not so accurate
        # checking for metrics
        (mse, rmse, mae, r2) = eval_metrics(ytest, ypred)
        print("  MSE: %s" % mse)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        params = {'colsample_bylevel' : 1, 'colsample_bytree': 1, 'learning_rate': 0.1,
                  'objective': 'reg:linear' ,'max_depth': 5, 'alpha': 10,'n_estimators' : 300,'random_state': 42}
        mlflow.log_params(params)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("Training Score", score)
        # building 3 fold cv model in xgboost - so change these hyper parms and checking for rmse and r2
        data_dmatrix = xgb.DMatrix(data=cleandf, label=y)
        params2 = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
                  'max_depth': 5, 'alpha': 10,'verbose': 0}
        cv_results = xgb.cv(dtrain=data_dmatrix, params=params2, nfold=3,
                            num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
        print(cv_results.mean())
        #cv_results.to_excel("Cross Validations using XGB.xlsx")
        mlflow.log_artifact("Cross Validations using XGB.xlsx")
        xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
        # feature importance plot
        xgb.plot_importance(xg_reg)
        plt.rcParams['figure.figsize'] = [5, 5]
        plt.savefig("Feature Importance using XGB.png")
        plt.tight_layout()
        mlflow.log_artifact("Feature Importance using XGB.png")
        #plots visualization of model
        print(xtest)
        xtestbkp = xtest.copy(deep=True)
        xtestbkp['ytest'] = ytest
        xtestbkp['ypred'] = ypred
        print(xtestbkp)
        # converting the column values to string
        xtestbkp['Category cat'] = xtestbkp['Category cat'].replace([0, 1, 2],['Office Supplies', 'Technology', 'Furniture'])
        xtestbkp['Region cat'] = xtestbkp['Region cat'].replace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                                                ['Central', 'Oceania', 'Central Asia', 'EMEA',
                                                                 'North Asia', 'South', 'West', 'East', 'Carribean',
                                                                 'North', 'Southeast Asia', 'Africa','Canada'])
        xtestbkp = xtestbkp.rename(columns={'Category cat': 'Category', 'Region cat': 'Region'})
        xtestbkp.set_index('Category')
        # subplots - plot test # use unstack() category wise # actual vs predicted plot
        fig, ax = plt.subplots(figsize=(15, 7))
        xtestbkp.groupby(['Order month', 'Category']).sum()['ypred'].unstack().plot(ax=ax)
        xtestbkp.groupby(['Order month', 'Category']).sum()['ytest'].unstack().plot(ax=ax, kind='bar')
        plt.title('Prediction of demand monthly for all Categories')
        plt.savefig('Category_grpsum_permonthfin.png')
        plt.ylabel('Quantity')
        plt.show()
        fig, ax = plt.subplots(figsize=(15, 7))
        xtestbkp[xtestbkp['Category'] == 'Office Supplies'].groupby(['Order month', 'Category']).sum()[
            'ypred'].unstack().plot(ax=ax,color='r')
        xtestbkp[xtestbkp['Category'] == 'Office Supplies'].groupby(['Order month', 'Category']).sum()[
            'ytest'].unstack().plot(ax=ax, kind='bar')
        plt.title('Prediction of demand monthly for one Category')
        plt.savefig('pred_category1_month.png')
        plt.ylabel('Quantity')
        plt.show()
        # unstack region wise
        fig, ax = plt.subplots(figsize=(15, 7))
        xtestbkp.groupby(['Order month', 'Region']).sum()['ypred'].unstack().plot(ax=ax)
        xtestbkp.groupby(['Order month', 'Region']).sum()['ytest'].unstack().plot(ax=ax, kind='bar')
        plt.title('Prediction of demand monthly for all region')
        plt.ylabel('Quantity')
        plt.show()
        fig, ax = plt.subplots(figsize=(15, 7))
        xtestbkp[xtestbkp['Region']=='EMEA'].groupby(['Order month', 'Region']).sum()['ypred'].unstack().plot(ax=ax,color='r')
        xtestbkp[xtestbkp['Region']=='EMEA'].groupby(['Order month', 'Region']).sum()['ytest'].unstack().plot(ax=ax, kind='bar')
        plt.title('Prediction of demand monthly for One region')
        plt.savefig('Region1_grpsum_permonth.png')
        plt.ylabel('Quantity')
        plt.show()
        # unstack order, region wise
        fig, ax = plt.subplots(figsize=(15, 7))
        xtestbkp[xtestbkp['Region']=='EMEA'].groupby(['Order month', 'Region', 'Category']).sum()['ypred'].unstack().plot(ax=ax)
        xtestbkp[xtestbkp['Region']=='EMEA'].groupby(['Order month', 'Region', 'Category']).sum()['ytest'].unstack().plot(ax=ax, kind='bar')
        plt.title('Prediction of demand grouped by Category for one region')
        plt.savefig('mon_cat_region1_sum.png')
        plt.ylabel('Quantity')
        plt.show()

        mlflow.log_artifact("Pearson Correaltion-Rawdata.png")
        mlflow.log_artifact("Category_grpsum_permonth.png")
        mlflow.log_artifact("mon_cat_region1_sum.png") #categorywise
        mlflow.log_artifact("Region1_grpsum_permonth.png")
        mlflow.log_artifact("zoomed_3groups_novdec.png")
        mlflow.log_artifact("zoomed_2_months_region wise.png")
        mlflow.log_artifact("zoomed_order_region_category.png")
        mlflow.log_artifact("zoomed_regionwise.png")
        mlflow.log_artifact("zoomed_3groups_aug,sep,oct.png")
        mlflow.set_tag("Model", "XGB Tree")
        mlflow.sklearn.log_model(xgbr, "model")

        # Log artifacts (output files) in mlflow
        mlflow.log_artifact("Global_Superstore2.xlsx")  # input
        mlflow.end_run()

    #experiment_id = mlflow.create_experiment(" Prediction Experiment2")
    with mlflow.start_run(experiment_id=experiment_id):
        # Randomforest model # Instantiate model with 500 decision trees
        rf = RandomForestRegressor(min_samples_leaf=1, min_samples_split=2, max_depth=10,n_estimators=500, random_state=42)
        # Train the model on training data
        rf.fit(xtrain, ytrain)
        scorerf = rf.score(xtrain, ytrain)
        print("Training score of RF: ", scorerf)
        params3={'max_depth': 10,
         'min_samples_leaf': 1,
         'min_samples_split': 2,
         'n_estimators': 500,
         'random_state': 42,
         'verbose': 0}
        mlflow.log_params(params3)
        predictions = rf.predict(xtest)  # Use the forest's predict method on the test data
        (mse, rmse, mae, r2) = eval_metrics(ytest, predictions)
        print("  MSE: %s" % mse)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("Training Score", scorerf)
        # Saving feature names for later use
        feature_list = list(cleandf.columns)
        #feature importance
        importances = list(rf.feature_importances_)
        plt.style.use('fivethirtyeight')  # set the style and plot importance
        x_values = list(range(len(importances)))       # list of x locations for plotting
        # Make a bar chart
        fig = plt.figure()
        plt.bar(x_values, importances, orientation='vertical')
        # Tick labels for x axis
        plt.xticks(x_values, feature_list, rotation='vertical')
        # Axis labels and title
        plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
        plt.tight_layout()
        fig.savefig("Variable Importance RF.png")
        # Axis labels and title #plt.ylabel('Importance');plt.xlabel('Variable');plt.title('Variable Importances')
        mlflow.log_artifact("variable_imp_rf.png")
        mlflow.set_tag("Model", "RandomForest model")
        mlflow.sklearn.log_model(rf, "model2")
        mlflow.end_run()


    with mlflow.start_run(experiment_id=experiment_id):
        # decision tree
        decisiontreereg = DecisionTreeRegressor(min_samples_split = 2, min_samples_leaf=1,criterion="mse",splitter="best",max_depth=10)
        mlflow.log_param('max_depth', 10)
        mlflow.log_param('min_samples_split',2)
        mlflow.log_param('min_samples_leaf', "best")
        mlflow.log_param('splitter', "mse")
        mlflow.log_param('criterion', 1)
        decisiontreereg.fit(xtrain, ytrain)
        scoredt = decisiontreereg.score(xtrain, ytrain)
        print("Training score of DT: ", scoredt)
        predictionsdt = decisiontreereg.predict(xtest)  # Use the forest's predict method on the test data
        (mse, rmse, mae, r2) = eval_metrics(ytest, predictionsdt)
        print("  MSE: %s" % mse)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        mlflow.log_metric("Training Score", scoredt)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)
        #plots
        xtestdtr = xtest.copy(deep=True)
        xtestdtr['ytest'] = ytest
        xtestdtr['predictionsdt'] = predictionsdt
        print(xtestdtr)
        # converting the column values to string
        xtestdtr['Category cat'] = xtestdtr['Category cat'].replace([0, 1, 2],['Office Supplies', 'Technology', 'Furniture'])
        #xtestbkp['Region cat']=xtestbkp['Region cat'].replace([0,1,2,3,4,5,6,7,8,9,10,11,12],['Central','Oceania','Central Asia','EMEA','North Asia','South','West','East','Carribean','North','Southeast Asia','Africa'])
        xtestdtr = xtestdtr.rename(columns={'Category cat': 'Category', 'Region cat': 'Region'})
        xtestdtr.set_index('Category')
        # subplots - plot test # use unstack() category wise # actual vs predicted plot
        fig, ax = plt.subplots(figsize=(15, 7))
        xtestdtr.groupby(['Order month', 'Category']).count()['predictionsdt'].unstack().plot(ax=ax)
        xtestdtr.groupby(['Order month', 'Category']).count()['ytest'].unstack().plot(ax=ax, kind='bar')
        plt.title('Prediction of demand using Decision Tree category wise')
        plt.ylabel('Quantity')
        plt.show()
        plt.savefig("Pred_actual_DT.png")
        mlflow.log_artifact("Pred_actual_DT.png")
        mlflow.set_tag("Model", "DecisionTree")
        mlflow.sklearn.log_model(decisiontreereg, "model3")

        mlflow.end_run()


        #mlflow tracking
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        #if tracking_url_type_store != "file":
            #mlflow.sklearn.log_model(xgbr, "model", registered_model_name="XGB Regression")
            #mlflow.sklearn.log_model(rf, "model2", registered_model_name="Randomforest Regression")
            #mlflow.sklearn.log_model(decisiontreereg, "model3", registered_model_name="Decision Tree Regression")
        #else:
            #mlflow.sklearn.log_model(xgbr, "model")
            #mlflow.sklearn.log_model(rf, "model2")
            #mlflow.sklearn.log_model(decisiontreereg, "model3")






