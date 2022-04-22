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
    dcc.Graph(id="graph")
])


@app.callback(Output('graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_s_state(n):
    global df
    lock.acquire()
    fig = px.scatter(
        df, x="x", y="y", animation_frame="time",
        color="player", hover_name="player", size_max=10,
        range_x=[-2.5, 2.5], range_y=[-2.5, 2.5],
        width=800, height=800)
    fig.update_traces()
    lock.release()
    return fig

@tl.job(interval=timedelta(seconds=10))
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
    print(df)
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
