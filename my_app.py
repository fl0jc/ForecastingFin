#my_app.py
#The Dash app that will display the predictions and actuals.

from flask import Flask
from dash import Dash, dcc, html
import plotly.graph_objs as go
import pandas as pd


combined_results = pd.read_csv("combined_results.csv")
future_results = pd.read_csv("future_results.csv")

server = Flask(__name__)

app = Dash(__name__, server=server)

app.layout = html.Div([
    dcc.Graph(
        id='predictions-plot',
        figure={
            'data': [
                go.Scatter(x=combined_results['Date'], y=combined_results['Train Predictions'], mode='lines', name='Train Predictions'),
                go.Scatter(x=combined_results['Date'], y=combined_results['Actuals'], mode='lines', name='Actuals'),
                go.Scatter(x=future_results['Date'], y=future_results['Predictions'], mode='lines', name='Future Predictions')
            ],
            'layout': go.Layout(
                title='Train Predictions, Actuals, and Future Predictions',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Value'},
                margin={'l': 60, 'r': 10, 't': 60, 'b': 60},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)