"""
This file will firstly load the generators from the DatasetAugmentation
Will then load the classificator model
Will then connect to the WebsocketServer
And will lastly wait for keyboard input to generate data with the generator, feed them into the classificator and send the classification result to the WebsocketServer
"""
import datetime

import keyboard
import sys

import tensorflow as tf
import websocket

tf.random.set_seed(42)
# random.seed(42)
# np.random.seed(42)
# np.random.RandomState(42)

def setup_keyboard_input(right_generator, left_generator, feet_generator, classificator, connection):
    seed = tf.random.normal([1, 500, 50])
    def on_key(e: keyboard.KeyboardEvent, generator, label):
        with open('log.txt', 'a') as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {label} {e.event_type}\n")
        while e.event_type == keyboard.KEY_DOWN:
            data = generator.predict(seed)
            classification = classificator.predict(data)
            connection.send(classification)
        with open('log.txt', 'a') as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {label} {e.event_type}\n")

    on_right_key = lambda e: on_key(e, right_generator, 'right')
    on_left_key = lambda e: on_key(e, left_generator, 'left')
    on_feet_key = lambda e: on_key(e, feet_generator, 'feet')

    keyboard.hook_key('a', on_right_key)
    keyboard.hook_key('d', on_left_key)
    keyboard.hook_key('space', on_feet_key)


def connect_to_websocket_server(websocket_server_url: str):
    return websocket.create_connection(websocket_server_url)


def load_generators(path_to_generators):
    right_generator = tf.keras.models.load_model(f"{path_to_generators}/right_generator")
    left_generator = tf.keras.models.load_model(f"{path_to_generators}/left_generator")
    feet_generator = tf.keras.models.load_model(f"{path_to_generators}/feet_generator")
    return right_generator, left_generator, feet_generator


def load_classificator(path_to_classificator):
    return tf.keras.models.load_model(path_to_classificator)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: main.py <path_to_generators> <path_to_classificator> <websocket_server_url>")
        sys.exit(1)

    path_to_generators = sys.argv[1]
    path_to_classificator = sys.argv[2]
    websocket_server_url = sys.argv[3]
    right_generator, left_generator, feet_generator = load_generators(path_to_generators)
    classificator = load_classificator(path_to_classificator)
    connection = connect_to_websocket_server(websocket_server_url)
    setup_keyboard_input(right_generator, left_generator, feet_generator, classificator, connection)
    keyboard.wait('esc')
    keyboard.clear_all_hotkeys()
    connection.close()