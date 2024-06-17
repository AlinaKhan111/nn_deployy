import os
import pandas as pd
import pickle

from src.config import config


import os
import pandas as pd
import pickle
from src.config import config

def load_dataset(file_name):
    file_path = os.path.join(config.DATAPATH, file_name)
    data = pd.read_csv(file_path)
    return data

def save_model(theta0, theta):
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH, "two_input_xor_nn.pkl")
    with open(pkl_file_path, "wb") as file_handle:
        pickle.dump({"params": {"biases": theta0, "weights": theta}, "activations": config.f}, file_handle)
    print(f"Saved model '{pkl_file_path}' successfully.")

def load_model(file_name):
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH, file_name)
    if not os.path.exists(pkl_file_path):
        raise FileNotFoundError(f"Model file '{file_name}' not found in '{config.SAVED_MODEL_PATH}'.")
    with open(pkl_file_path, "rb") as file_handle:
        loaded_model = pickle.load(file_handle)
    return loaded_model
