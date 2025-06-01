import datetime
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import plotly.express as px
import pandas as pd
import numpy as np

class EvaluteModel():
    def __init__(self, X_test: pd.DataFrame, y_test: pd.DataFrame, X_validation: pd.DataFrame, county_names, random_state=None) -> None:
        self.X_test = X_test
        self.y_test = y_test
        self.X_validation = X_validation
        self.county_names = county_names
        self.random_state = random_state

    def test(self, model, random_day=None, normalization=None):
        self.results = pd.DataFrame()
        self.resultsPerCounty = pd.DataFrame(columns=['County', 'MAE', 'RMSE'])

        self.y_pred = model.predict(self.X_test)

        # For handling trend issue in target
        if normalization is not None:
            self.y_pred = self.y_pred * normalization
            self.y_test['target'] = self.y_test['target'] * normalization

        self.__MAE()
        self.__RMSE()
        self.__metricsPerCounty()
        display(self.results)
        display(self.resultsPerCounty)

        self.__Plot(random_day)
        self.__FeatureImportance(model)
    
    def __metricsPerCounty(self):
        validationData = self.X_test[['county']].copy()
        validationData['predictions'] = self.y_pred.tolist()
        validationData['target'] = self.y_test

        data = validationData.groupby('county').mean().reset_index()

        for county in data.county.unique():
            countyData = data[data.county == county]

            mae = MAE(countyData['target'], countyData['predictions'])
            mse = MSE(countyData['target'], countyData['predictions'])

            self.resultsPerCounty.loc[len(self.resultsPerCounty.index)] = [self.county_names[county], mae, mse] 
            
    def __MAE(self):
        self.results['MAE'] = [MAE(self.y_pred, self.y_test)]

    def __RMSE(self):
        self.results['RMSE'] = [np.sqrt(MSE(self.y_pred, self.y_test))]

    def selector(self, column_name):
        # just need to be careful that "column_name" is not any other string in "hovertemplate" data
        f = lambda x: True if column_name in x['hovertemplate'] else False
        return f

    def __Plot(self, random_day = None):
        # Get random date for plot
        if random_day is not None:
            self.random_day = random_day
        else:
            self.random_day = self.X_validation['datetime_date'].sample(random_state=self.random_state).iloc[0]

        # Get week based on random date
        self.X_validation['target'] = self.y_test
        self.X_validation['predictions'] = self.y_pred.tolist()
        self.validationData = self.X_validation[
            (self.X_validation.datetime_date >= self.random_day - datetime.timedelta(days=3)) &
            (self.X_validation.datetime_date <= self.random_day + datetime.timedelta(days=3))
        ]

        if len(self.validationData) == 0:
            display(f'Warning: Test data in range {min(self.X_validation.datetime_date)} - {max(self.X_validation.datetime_date)}')

        validationData = self.validationData[['datetime', 'target', 'is_consumption', 'is_business', 'predictions']].copy()
        validationData = validationData.groupby(by=['datetime', 'is_consumption', 'is_business']).mean().reset_index()

        display(validationData)

        fig = px.line(
                    validationData.rename(columns={'target': 'Real', 'predictions': 'Prediction'}),
                    x='datetime',
                    y=['Real', 'Prediction'], 
                    facet_col='is_consumption', 
                    facet_row='is_business', 
                    title=f'XGBoost model test',
                    labels={
                          'is_business' : 'Business',
                          'datetime' : 'Date',
                          'value': 'Value'}, height=700, width=1000)
        
        # Add custom plot style
        fig.update_traces(patch={"line": {"dash": "dot"}}, selector=self.selector('Prediction'))
        fig.update_layout(legend_title_text='Variables')
        fig.update_xaxes(tickformat="%d-%m-%Y")
        fig.for_each_annotation(lambda a: a.update(
            text=a.text
            .replace('is_consumption=True', 'Consumption')
            .replace('is_consumption=False', 'Production')
            .replace('=True', '(Yes)')
            .replace('=False', '(No)')))
        fig.show()


    def __FeatureImportance(self, model):
        feature_importance= model.get_booster().get_score(importance_type="gain")

        keys = list(feature_importance.keys())
        values = list(feature_importance.values())

        data = pd.DataFrame({'Attribute' : keys, 'Weight' : values}).sort_values(by = 'Weight', ascending=False)[0:15]

        fig = px.bar(
            data,
            x='Attribute',
            y='Weight',
            height=250,
            width=1000,
            title='Attribute weight')
        fig.show()