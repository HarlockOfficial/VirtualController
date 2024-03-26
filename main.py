"""
This file will firstly load the generators from the DatasetAugmentation
Will then load the classificator model
Will then connect to the WebsocketServer
And will lastly wait for keyboard input to generate data with the generator, feed them into the classificator and send the classification result to the WebsocketServer
"""
import datetime
import enum
import pickle

import sshkeyboard
import sys

import tensorflow as tf
import websocket

import random
import numpy as np

import EEGClassificator.utils

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
np.random.RandomState(42)

class KeyStatus(enum.Enum):
    UP = 0
    DOWN = 1

def setup_keyboard_input(right_generator, left_generator, feet_generator, classificator, connection):
    key = KeyStatus.UP
    def on_key(generator):
        while key == KeyStatus.DOWN:
            seed = tf.random.normal([1, 58, 65])
            data = generator(seed)['output_0']
            data = data.numpy()
            classification = classificator.predict(data)[0]
            classification = EEGClassificator.utils.from_categorical(classification.item())
            connection.send(classification)

    def press(k):
        print(f"{k} key pressed")
        nonlocal key
        if k in ['a', 'd', 'w', 'space']:
            key = KeyStatus.DOWN
        if k == 'a':
            on_left_key()
        elif k == 'd':
            on_right_key()
        elif k == 'w' or k == 'space':
            on_feet_key()

    def release(k):
        print(f"{k} key released")
        nonlocal key
        if k in ['a', 'd', 'w', 'space']:
            key = KeyStatus.UP

    on_right_key = lambda: on_key(right_generator)
    on_left_key = lambda: on_key(left_generator)
    on_feet_key = lambda: on_key(feet_generator)
    sshkeyboard.listen_keyboard(on_press=press, on_release=release, sequential=False, until='esc')


def connect_to_websocket_server(websocket_server_url: str):
    return websocket.create_connection(websocket_server_url)


def load_generators(path_to_generators):
    right_generator = tf.keras.layers.TFSMLayer(f"{path_to_generators}/generator_right_hand", call_endpoint='serving_default')
    left_generator = tf.keras.layers.TFSMLayer(f"{path_to_generators}/generator_left_hand", call_endpoint='serving_default')
    feet_generator = tf.keras.layers.TFSMLayer(f"{path_to_generators}/generator_feet", call_endpoint='serving_default')
    return right_generator, left_generator, feet_generator


def load_classificator(path_to_classificator):
    pipe = pickle.load(open(f"{path_to_classificator}", "rb"))
    return pipe


def main(path_to_generators, path_to_classificator, websocket_server_url):
    right_generator, left_generator, feet_generator = load_generators(path_to_generators)
    classificator = load_classificator(path_to_classificator)
    connection = connect_to_websocket_server(websocket_server_url)
    try:
        setup_keyboard_input(right_generator, left_generator, feet_generator, classificator, connection)
    except KeyboardInterrupt:
        pass
    connection.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: main.py <path_to_generators> <path_to_classificator> <websocket_server_url>")
        sys.exit(1)

    path_to_generators = sys.argv[1]
    path_to_classificator = sys.argv[2]
    websocket_server_url = sys.argv[3]
    main(path_to_generators, path_to_classificator, websocket_server_url)