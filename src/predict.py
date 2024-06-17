import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config import config
from src.preprocessing.data_management import load_model
import pipeline as pl
import os

def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):
    return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs, current_layer_neurons_weights)

def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):
    if current_layer_neurons_activation_function == "linear":
        return current_layer_neurons_weighted_sums
    elif current_layer_neurons_activation_function == "sigmoid":
        return 1 / (1 + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "tanh":
        return np.tanh(current_layer_neurons_weighted_sums)
    elif current_layer_neurons_activation_function == "relu":
        return np.maximum(current_layer_neurons_weighted_sums, 0)

def predict(X, model_file):
    model = load_model(model_file)
    pl.theta0 = [np.array(b, dtype=float) for b in model["params"]["biases"]]
    pl.theta = [np.array(w, dtype=float) for w in model["params"]["weights"]]

    predictions = []
    for x in X:
        h = [None] * config.NUM_LAYERS
        z = [None] * config.NUM_LAYERS

        h[0] = x.reshape(1, -1)

        for l in range(1, config.NUM_LAYERS):
            z[l] = layer_neurons_weighted_sum(h[l - 1], pl.theta0[l], pl.theta[l])
            h[l] = layer_neurons_output(z[l], config.f[l])

        predictions.append(h[-1][0, 0])

    return np.array(predictions).reshape(-1, 1)

def evaluate_model(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    # Example usage
    X_test = np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])
    Y_test = np.array([[0], [1], [1], [0]])

    model_file = "two_input_xor_nn.pkl"  # Adjust this to the actual path
    model_path = os.path.join(config.SAVED_MODEL_PATH, model_file)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
    else:
        Y_pred = predict(X_test, model_path)
        accuracy, precision, recall, f1 = evaluate_model(Y_test, Y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        print("Predictions:")
        for i in range(len(X_test)):
            print(f"Input: {X_test[i]}, Predicted Output: {Y_pred[i][0]}")
