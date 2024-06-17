import pandas as pd
import numpy as np
from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model
import pipeline as pl

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

def del_layer_neurons_outputs_wrt_weighted_sums(current_layer_neurons_activation_function, current_layer_neurons_weighted_sums):
    if current_layer_neurons_activation_function == "linear":
        return np.ones_like(current_layer_neurons_weighted_sums)
    elif current_layer_neurons_activation_function == "sigmoid":
        sigmoid = 1 / (1 + np.exp(-current_layer_neurons_weighted_sums))
        return sigmoid * (1 - sigmoid)
    elif current_layer_neurons_activation_function == "tanh":
        return 1 - np.tanh(current_layer_neurons_weighted_sums)**2
    elif current_layer_neurons_activation_function == "relu":
        return (current_layer_neurons_weighted_sums > 0).astype(float)

def del_layer_neurons_outputs_wrt_biases(current_layer_neurons_outputs_dels):
    return current_layer_neurons_outputs_dels

def del_layer_neurons_outputs_wrt_weights(previous_layer_neurons_outputs, current_layer_neurons_outputs_dels):
    return np.matmul(previous_layer_neurons_outputs.T, current_layer_neurons_outputs_dels)

def run_training(tol, epsilon, epochs):
    epoch_counter = 0
    mse = 1
    loss_per_epoch = [mse]

    training_data = load_dataset("train.csv")

    obj = pp.preprocess_data()
    obj.fit(training_data.iloc[:, 0:2], training_data.iloc[:, 2])
    X_train, Y_train = obj.transform(training_data.iloc[:, 0:2], training_data.iloc[:, 2])

    pl.initialize_parameters()

    while epoch_counter < epochs:
        mse = 0
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        Y_train_shuffled = Y_train[permutation]

        for i in range(0, X_train.shape[0], config.MINI_BATCH_SIZE):
            X_batch = X_train_shuffled[i:i + config.MINI_BATCH_SIZE]
            Y_batch = Y_train_shuffled[i:i + config.MINI_BATCH_SIZE]

            for j in range(X_batch.shape[0]):
                h = [None] * config.NUM_LAYERS
                z = [None] * config.NUM_LAYERS
                del_fl_by_del_z = [None] * config.NUM_LAYERS
                del_hl_by_del_theta0 = [None] * config.NUM_LAYERS
                del_hl_by_del_theta = [None] * config.NUM_LAYERS
                del_L_by_del_h = [None] * config.NUM_LAYERS
                del_L_by_del_theta0 = [None] * config.NUM_LAYERS
                del_L_by_del_theta = [None] * config.NUM_LAYERS

                h[0] = X_batch[j].reshape(1, X_batch.shape[1])

                for l in range(1, config.NUM_LAYERS):
                    z[l] = layer_neurons_weighted_sum(h[l - 1], pl.theta0[l], pl.theta[l])
                    h[l] = layer_neurons_output(z[l], config.f[l])
                    del_fl_by_del_z[l] = del_layer_neurons_outputs_wrt_weighted_sums(config.f[l], z[l])
                    del_hl_by_del_theta0[l] = del_layer_neurons_outputs_wrt_biases(del_fl_by_del_z[l])
                    del_hl_by_del_theta[l] = del_layer_neurons_outputs_wrt_weights(h[l - 1], del_fl_by_del_z[l])

                Y_batch[j] = Y_batch[j].reshape(Y_batch[j].shape[0], 1)
                L = (1 / 2) * (Y_batch[j][0] - h[config.NUM_LAYERS - 1][0, 0]) ** 2
                mse += L

                del_L_by_del_h[config.NUM_LAYERS - 1] = (h[config.NUM_LAYERS - 1] - Y_batch[j])
                for l in range(config.NUM_LAYERS - 2, 0, -1):
                    del_L_by_del_h[l] = np.matmul(del_L_by_del_h[l + 1], (del_fl_by_del_z[l + 1] * pl.theta[l + 1]).T)

                for l in range(1, config.NUM_LAYERS):
                    del_L_by_del_theta0[l] = del_L_by_del_h[l] * del_hl_by_del_theta0[l]
                    del_L_by_del_theta[l] = del_L_by_del_h[l] * del_hl_by_del_theta[l]

                    pl.theta0[l] -= (epsilon * del_L_by_del_theta0[l])
                    pl.theta[l] -= (epsilon * del_L_by_del_theta[l])

        mse /= X_train.shape[0]
        epoch_counter += 1
        loss_per_epoch.append(mse)

        print(f"Epoch #{epoch_counter}, Loss = {mse}")

        if abs(loss_per_epoch[epoch_counter] - loss_per_epoch[epoch_counter - 1]) < tol:
            break

    save_model(pl.theta0, pl.theta)

if __name__ == "__main__":
    run_training(tol=10 ** (-7), epsilon=10 ** (-2), epochs=10000)
