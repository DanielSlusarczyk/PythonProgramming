import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Charts
import plotly.graph_objects as go

class CustomPlot:
    seriesPerPlot = 1

    # Layout
    title = 'Custom value'
    prefix = 'Value'

    def __init__(self, df: pd.DataFrame, custom: str):
        self.df = df
        self.customColumn = custom

        # Create fig to show
        self.fig = go.Figure()

        # Options for slider
        self.options = df[custom].unique()

    def Table(self):

        for option in self.options:

            # Data associated with specific option
            data = self.df[self.df[self.customColumn] == option]

            header=dict(values=data.columns, font=dict(size=15), align='left')
            cells=dict(values=data.T.values.tolist(), align='left')

            self.fig.add_trace(go.Table(visible=False, header=header, cells=cells))

        return self
        
    def Scatter(self, normalize=False):
        
        # One column - One serie
        columns = [col for col in self.df.columns if col not in ['datetime', self.customColumn]]
        self.seriesPerPlot = len(columns)

        if(normalize):
            scaler = MinMaxScaler()
            self.df.loc[:, ['normalized_' + col for col in columns]] = scaler.fit_transform(self.df[columns])

            self.fig.update_yaxes(range=[0, 1])
        else:
            min_value = min(self.df[columns])
            max_value = max(self.df[columns])

            self.fig.update_yaxes(range=[min_value, max_value])
        
        for option in self.options:

            # Data associated with specific option
            data = self.df[self.df[self.customColumn] == option]

            for series in columns:
                if normalize:
                    column = 'normalized_' + series
                else:
                    column = series

                self.fig.add_trace(go.Scatter(
                    x=data['datetime'], 
                    y=data[column], 
                    name=series, 
                    line={'dash': 'solid', 'width': 2},
                    customdata=np.array((data[series])).reshape(-1, 1),
                    hovertemplate= 
                    '<b>Date: </b> %{x}<br>' +
                    '<b>' + series + '</b>: %{customdata[0]:.2f}<br>' +
                    '<extra></extra>'))
        
        return self
    
    def Slider(self, title = None, prefix = None):
        if title != None:
            self.title = title

        if prefix != None:
            self.prefix = prefix

        return self

    def Show(self):
        self.__CreateSlider()

        self.fig.show()
    
    def __CreateSlider(self):
        # Prepare behaviour for each slider option
        steps = []
        for i in range(0, len(self.fig.data), self.seriesPerPlot):
            step = dict(
                method='update',
                args=[{'visible': [False] * len(self.fig.data)},
                    {'title': f'{self.title}: ' + str(self.options[int(i/self.seriesPerPlot)])}],
                label=str(self.options[int(i/self.seriesPerPlot)])
            )

            # Set visible all series connected with slider option
            for j in range(i, i + self.seriesPerPlot):
                step['args'][0]['visible'][j] = True

            steps.append(step)

        # Create slider
        sliders = [dict(
            currentvalue={'prefix': f'{self.prefix}: ', 'xanchor': 'center'},
            steps=steps,
        )]

        self.fig.update_layout(sliders=sliders)