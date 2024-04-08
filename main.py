"""
This file will firstly load the generators from the DatasetAugmentation
Will then load the classificator model
Will then connect to the WebsocketServer
And will lastly wait for keyboard input to generate data with the generator, feed them into the classificator and send the classification result to the WebsocketServer
"""
import enum
import os
import pickle
import random
import sys

import numpy as np
import sshkeyboard
import tensorflow as tf
import torch
import websocket

import EEGClassificator.utils

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)
np.random.RandomState(42)

class KeyStatus(enum.Enum):
    UP = 0
    DOWN = 1

def setup_keyboard_input(right_generator, left_generator, feet_generator, classificator, connection, device):
    key = KeyStatus.UP
    def on_key(generator):
        while key == KeyStatus.DOWN:
            seed = torch.rand([1, 1, 58, 65]).to(device).to(torch.float32)
            data = generator(seed)
            data = data.detach().cpu().numpy()
            data = np.squeeze(data, axis=1)
            classification = classificator.predict(data)[0]
            classification = EEGClassificator.utils.from_categorical(classification.item())
            connection.send(classification)

    def press(k):
        # did block code due to high multitasking
        # print(f"{k} key pressed")
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
        # did block code due to high multitasking
        # print(f"{k} key released")
        nonlocal key
        if k in ['a', 'd', 'w', 'space']:
            key = KeyStatus.UP

    on_right_key = lambda: on_key(right_generator)
    on_left_key = lambda: on_key(left_generator)
    on_feet_key = lambda: on_key(feet_generator)
    print("Press 'a' to generate data for the left hand, 'd' for the right hand and 'w' or 'space' for the feet")
    sshkeyboard.listen_keyboard(on_press=press, on_release=release, sequential=False, until='esc')


def connect_to_websocket_server(websocket_server_url: str):
    return websocket.create_connection(websocket_server_url)


def load_generators(path_to_generators, device=None):
    def is_keras_model(path):
        files = os.listdir(path)
        return sum(1 if (os.path.isdir(tmp_path) and 'saved_model.pb' in os.listdir(tmp_path)) else 0
            for tmp_path in [os.path.join(path, file) for file in files]) == 3

    def is_pkl_model(path):
        files = os.listdir(path)
        return sum(1 if file.endswith(".pkl") else 0 for file in files) == 3

    if is_keras_model(path_to_generators):
        tmp_right_generator = tf.keras.layers.TFSMLayer(f"{path_to_generators}/generator_right_hand", call_endpoint='serving_default')
        tmp_left_generator = tf.keras.layers.TFSMLayer(f"{path_to_generators}/generator_left_hand", call_endpoint='serving_default')
        tmp_feet_generator = tf.keras.layers.TFSMLayer(f"{path_to_generators}/generator_feet", call_endpoint='serving_default')
        right_generator = lambda x: tmp_right_generator(x)['output_0'].numpy()
        left_generator = lambda x: tmp_left_generator(x)['output_0'].numpy()
        feet_generator = lambda x: tmp_feet_generator(x)['output_0'].numpy()
    elif is_pkl_model(path_to_generators):
        tmp_right_generator = pickle.load(open(f"{path_to_generators}/generator_right_hand.pkl", "rb"))
        tmp_left_generator = pickle.load(open(f"{path_to_generators}/generator_left_hand.pkl", "rb"))
        tmp_feet_generator = pickle.load(open(f"{path_to_generators}/generator_feet.pkl", "rb"))
        if device is not None:
            tmp_right_generator = tmp_right_generator.to(device)
            tmp_left_generator = tmp_left_generator.to(device)
            tmp_feet_generator = tmp_feet_generator.to(device)
        # generators that use eegfusenet return 2 params, first is generated data, second is useless for us
        right_generator = lambda x: tmp_right_generator(x)[0]
        left_generator = lambda x: tmp_left_generator(x)[0]
        feet_generator = lambda x: tmp_feet_generator(x)[0]
    else:
        raise ValueError("Generators must be either a Keras model or a pickle file")
    return right_generator, left_generator, feet_generator


def load_classificator(path_to_classificator):
    pipe = pickle.load(open(f"{path_to_classificator}", "rb"))
    return pipe


def main(path_to_generators, path_to_classificator, websocket_server_url):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    right_generator, left_generator, feet_generator = load_generators(path_to_generators, device=device)
    classificator = load_classificator(path_to_classificator)
    connection = connect_to_websocket_server(websocket_server_url)
    try:
        setup_keyboard_input(right_generator, left_generator, feet_generator, classificator, connection, device)
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