from flask import Flask

from dash import Dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

import pandas as pd
import numpy as np

from timeloop import Timeloop
from datetime import timedelta
from threading import Lock

from game import Game, generate_random_players

app = Dash(__name__)
tl = Timeloop()


app.layout = html.Div([
    html.H4('Rocket Positions Live Update'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Graph(id="graph"),
    dcc.Slider(min=0, max=0, step=1, id='time-slider', value=0,
               tooltip={"placement": "bottom", "always_visible": True})
])


@app.callback(Output('time-slider', 'max'),
              Input('interval-component', 'n_intervals'))
def update_slider(n):
    global df
    lock.acquire()
    max_time = int(df['time'].max())
    lock.release()
    return max_time


@app.callback(Output('graph', 'figure'),
              Input('time-slider', 'value'))
def update_s_state(frame):
    global df
    lock.acquire()
    fig = px.scatter(
        df[df['time'] == frame], x="x", y="y",
        color="player", hover_name="player", size_max=10,
        range_x=[-2.5, 2.5], range_y=[-2.5, 2.5],
        width=800, height=800)
    # fig.layout['sliders'][0]['active'] = int(len(fig.frames) - 1)
    lock.release()
    return fig


@tl.job(interval=timedelta(seconds=1))
def update_game_state():
    global game, df, time, done, resp, lock

    lock.acquire()
    for state_index, state in enumerate(resp["state"]):
        df = pd.concat([df, pd.DataFrame({
            'time': [time],
            'x': [state[0]],
            'y': [state[1]],
            'player': [resp["names"][state_index]]
        })])
        df.reset_index()
    lock.release()

    res = game.game_step()
    done = res['end']
    resp = game.get_state()

    print(res)

    time += 1


if __name__ == '__main__':
    players = generate_random_players(10)
    game = Game(players)

    df = pd.DataFrame({'time': [], 'player': [], 'x': [], 'y': []})
    lock = Lock()

    resp = game.get_state()
    time = 0
    done = False

    tl.start()

    app.run_server(debug=True)

    while not done and time < 100:
        pass

    tl.stop()
