from multiprocessing.connection import wait
import socketio
import time

# standard Python
sio = socketio.Client()
id = '123'


@sio.on('response', namespace='/control')
def recieve_state(data):
    print(data)
    thrust = calculate_thrust(data['state'])
    sio.emit('client_response', {'thrust': thrust}, namespace='/control')


@sio.on('connection_success', namespace='/control')
def connection_success(data):
    print(data['message'])
    sio.sleep(1)
    sio.emit('state_request', {'auth': id}, namespace='/control')


@sio.on('failure', namespace='/control')
def failure(data):
    print(data['message'])
    sio.disconnect()

def calculate_thrust(state):
    sio.sleep(1)
    return [1, 1]


sio.connect('http://localhost:5000', namespaces=['/control'], auth=id)
