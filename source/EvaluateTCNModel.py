from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

class EvaluateTCNModel():
    def __init__(self, X_test, y_test, y_test_metadata, county_names, random_state=None):
        """
        y_test_metadata: DataFrame with columns:
            'county', 'is_consumption', 'is_business', 'datetime', 'datetime_date'
        """
        self.X_test = X_test
        self.y_test = y_test.flatten() if isinstance(y_test, np.ndarray) else y_test.values.flatten()
        self.metadata = y_test_metadata.reset_index(drop=True)
        self.county_names = county_names
        self.random_state = random_state

    def test(self, model, random_day=None, normalization=None):
        self.results = pd.DataFrame()
        self.resultsPerCounty = pd.DataFrame(columns=['County', 'MAE', 'RMSE'])

        # Predict
        self.y_pred = model.predict(self.X_test).flatten()

        if normalization is not None:
            self.y_pred = self.y_pred * normalization
            self.y_test = self.y_test * normalization

        self.__MAE()
        self.__RMSE()
        self.__metricsPerCounty()
        display(self.results)
        display(self.resultsPerCounty)

        self.__Plot(random_day)

    def __metricsPerCounty(self):
        validationData = self.metadata.copy()
        validationData['predictions'] = self.y_pred
        validationData['target'] = self.y_test

        for county in validationData['county'].unique():
            countyData = validationData[validationData['county'] == county]

            mae = MAE(countyData['target'], countyData['predictions'])
            rmse = np.sqrt(MSE(countyData['target'], countyData['predictions']))

            self.resultsPerCounty.loc[len(self.resultsPerCounty)] = [
                self.county_names.get(county, str(county)),
                mae,
                rmse
            ]

    def __MAE(self):
        self.results['MAE'] = [MAE(self.y_pred, self.y_test)]

    def __RMSE(self):
        self.results['RMSE'] = [np.sqrt(MSE(self.y_pred, self.y_test))]

    def selector(self, column_name):
        return lambda x: column_name in x['hovertemplate']

    def __Plot(self, random_day=None):
        df = self.metadata.copy()
        df['target'] = self.y_test
        df['predictions'] = self.y_pred

        if random_day is not None:
            self.random_day = random_day
        else:
            self.random_day = df['datetime_date'].sample(random_state=self.random_state).iloc[0]

        validationData = df[
            (df['datetime_date'] >= self.random_day - datetime.timedelta(days=3)) &
            (df['datetime_date'] <= self.random_day + datetime.timedelta(days=3))
        ]

        if len(validationData) == 0:
            display(f'Warning: No test data in selected range {self.random_day - datetime.timedelta(days=3)} - {self.random_day + datetime.timedelta(days=3)}')
            return

        plot_data = validationData[['datetime', 'target', 'is_consumption', 'is_business', 'predictions']].copy()
        plot_data = plot_data.groupby(by=['datetime', 'is_consumption', 'is_business']).mean().reset_index()

        fig = px.line(
            plot_data.rename(columns={'target': 'Cel', 'predictions': 'Predykcja'}),
            x='datetime',
            y=['Cel', 'Predykcja'],
            facet_col='is_consumption',
            facet_row='is_business',
            title='Test działania modelu TCN',
            labels={
                'is_business': 'Biznes',
                'datetime': 'Data',
                'value': 'Wartość'
            },
            height=700, width=1000
        )

        fig.update_traces(patch={"line": {"dash": "dot"}}, selector=self.selector('Predykcja'))
        fig.update_layout(legend_title_text='Zmienne')
        fig.update_xaxes(tickformat="%d-%m-%Y")
        fig.for_each_annotation(lambda a: a.update(
            text=a.text
            .replace('is_consumption=True', 'Konsumpcja')
            .replace('is_consumption=False', 'Produkcja')
            .replace('=1.0', '(Tak)')
            .replace('=-1.0', '(Nie)')))
        fig.show()