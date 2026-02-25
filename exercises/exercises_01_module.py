from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import pandas as pd

def fit_and_eval(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    rmse = root_mean_squared_error(y_true=y_test, y_pred=y_pred)

    return {'mae': mae, 
            'mse': mse, 
            'rmse': rmse}