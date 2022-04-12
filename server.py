from flask_socketio import SocketIO, Namespace, emit
from flask import Flask, render_template, request
from threading import Lock
from dataclasses import dataclass
from typing import Dict

from game import Player, Game, player_id, player_info

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


class GameBackend(Namespace):
    def on_connect(self, auth):
        '''
        Authenticate with the player's id. If the player is registered, then add them to
        the player id dictionary
        '''
        if auth in player_id:
            player_info[request.sid] = player_id[auth]
            emit('connection_success', {'message': 'Connection Success'})
        else:
            emit('failure', {
                 'message': 'Authentification Failure: id does not exist'})

    def on_disconnect(self):
        print(f'Client Disconnected, sid: {request.sid}')

    def on_client_response(self, data):
        print(request.sid, data)
        emit('response', {'state': [1, 0, 0, 1]})

    def on_state_request(self, data):
        print(data)
        emit('response', {'state': [1, 0, 0, 1]})


if __name__ == '__main__':
    socketio.on_namespace(GameBackend('/control'))
    socketio.run(app)
