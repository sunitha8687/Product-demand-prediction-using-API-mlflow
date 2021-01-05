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
        rawdata = pd.read_excel(csv_url, parse_dates=["Order Date", "Ship Date"])
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    data = rawdata.drop(
        columns=["Row ID", "Order ID", "Customer ID", "Customer Name", "City", "State", "Country", "Postal Code",
                 "Profit"], axis=1)  # removing the other columns not interested for my use case
    data.dropna(inplace=True)  # if 0 drops rows with 0 and 1 drops rows with missing values
    # creating features #1st feature
    od = data["Order Date"]
    sd = data["Ship Date"]
    data["Days_to_Ship_Product"] = abs(
        sd - od)  # creating the no.of.days to ship the product as it would also be an important factor for product demand
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
    # print(cleandf)
    y = finaldf.iloc[:, 2]
    # print(y)
    # train-test split
    xtrain, xtest, ytrain, ytest = train_test_split(cleandf, y, test_size=0.20)  # quantity is my y variable target
    experiment_id = mlflow.create_experiment("Predicting Product Demand Experiment")
    with mlflow.start_run(experiment_id=experiment_id):
        xgbr = xgb.XGBRegressor(booster='gbtree', colsample_bylevel=1, colsample_bytree=1, learning_rate=0.1,
                                objective="reg:linear", max_depth=5, alpha=10, n_estimators=300, random_state=42)
        xgbr.fit(xtrain, ytrain)
        score = xgbr.score(xtrain, ytrain)
        print("Training score: ", score)
        ypred = xgbr.predict(
            xtest)  # prediction on testdata --------- ypred is quantity - its coming in decimal because model is not so accurate
        # checking for metrics
        (mse, rmse, mae, r2) = eval_metrics(ytest, ypred)
        print("  MSE: %s" % mse)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        params = {'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bytree': 1, 'learning_rate': 0.1,
                  'objective': 'reg:linear', 'max_depth': 5, 'alpha': 10, 'n_estimators': 300, 'random_state': 42}

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        # building 3 fold cv model in xgboost - so change these hyper parms and checking for rmse and r2
        data_dmatrix = xgb.DMatrix(data=cleandf, label=y)
        params2 = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
                   'max_depth': 5, 'alpha': 10}
        cv_results = xgb.cv(dtrain=data_dmatrix, params=params2, nfold=3,
                            num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
        cv_results.to_excel("Cross Validations using XGB.xlsx")
        xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
        # feature importance plot
        xgb.plot_importance(xg_reg)
        plt.rcParams['figure.figsize'] = [5, 5]
        plt.savefig("Feature Importance using XGB.png")
        plt.tight_layout()
        # actual vs predicted plot
        x_ax = range(len(ytest))
        Pred_ind = plt.figure(figsize=(12, 10))
        plt.plot(x_ax, ytest, color="blue", label="original")
        plt.plot(x_ax, ypred, color="red", label="predicted")
        plt.legend()
        plt.tight_layout()
        plt.title("Actual vs predicted plot")
        Pred_ind.savefig("Predictions vs Actual plot.png")
        mlflow.set_tag("Model1", "Experiment1")
        mlflow.sklearn.log_model(xgbr, "model")
        mlflow.end_run()

    with mlflow.start_run(experiment_id=experiment_id):
        # Randomforest model
        # Instantiate model with 500 decision trees
        rf = RandomForestRegressor(max_features='auto', min_samples_leaf=1, min_samples_split=2, max_depth=5,
                                   n_estimators=500, random_state=42)
        # Train the model on training data
        rf.fit(xtrain, ytrain)
        params3 = {'max_depth': 5,
                   'max_features': 'auto',
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
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        # Saving feature names for later use
        feature_list = list(cleandf.columns)
        # feature importance
        importances = list(rf.feature_importances_)
        plt.style.use('fivethirtyeight')  # set the style and plot importance
        x_values = list(range(len(importances)))  # list of x locations for plotting
        # Make a bar chart
        fig = plt.figure()
        plt.bar(x_values, importances, orientation='vertical')
        # Tick labels for x axis
        plt.xticks(x_values, feature_list, rotation='vertical')
        fig.savefig("Variable Importance RF.png")
        plt.show()
        # Axis labels and title #plt.ylabel('Importance');plt.xlabel('Variable');plt.title('Variable Importances')
        mlflow.set_tag("Model2", "Experiment2")
        mlflow.sklearn.log_model(rf, "model2")
        mlflow.end_run()

        # decision tree
        decisiontreereg = DecisionTreeRegressor(max_depth=10)
        decisiontreereg.fit(xtrain, ytrain)
        predictionsdt = decisiontreereg.predict(xtest)  # Use the forest's predict method on the test data
        (mse, rmse, mae, r2) = eval_metrics(ytest, predictionsdt)
        print("  MSE: %s" % mse)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # mlflow tracking
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(xgbr, "model", registered_model_name="XGB Regression")
            mlflow.sklearn.log_model(rf, "model2", registered_model_name="Randomforest Regression")
            mlflow.sklearn.log_model(decisiontreereg, "model3", registered_model_name="Decision Tree Regression")
        else:
            mlflow.sklearn.log_model(xgbr, "model")
            mlflow.sklearn.log_model(rf, "model2")
            mlflow.sklearn.log_model(rf, "model3")

        # Log artifacts (output files) in mlflow
        mlflow.log_artifact("Global_Superstore2.xlsx")  # input
        mlflow.log_artifact("Cross Validations using XGB.xlsx")
        mlflow.log_artifact("Pearson Correaltion-Rawdata.png")
        mlflow.log_artifact("Feature Importance using XGB.png")
        mlflow.log_artifact("Predictions vs Actual plot.png")
        # mlflow.log_artifact("Feature Importance using RF.png")

