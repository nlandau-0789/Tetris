#include "utils/std.h"
#include "utils/print_epu16.h"
#include <stdlib.h>
#include <time.h>

struct nn {
    float *input;
    float ***weights, **biases, *output_weights;
    float output;
    int input_size, n_hidden_layers;
    int *hidden_layer_sizes;
};

typedef struct nn nn;

void init_nn(nn* network, int input_size, int output_size, int n_hidden_layers, int* hidden_layer_sizes) {
    network->input_size = input_size;
    network->n_hidden_layers = n_hidden_layers;
    network->hidden_layer_sizes = hidden_layer_sizes;

    // Allocate memory for inputs, outputs, weights, and biases
    network->input = (float*)malloc(input_size * sizeof(float));
    
    network->weights = (float***)malloc(n_hidden_layers * sizeof(float**));
    network->biases = (float**)malloc(n_hidden_layers * sizeof(float*));
    network->output_weights = (float*)malloc(hidden_layer_sizes[n_hidden_layers - 1] * sizeof(float));

    srand((unsigned int)time(NULL));

    for (int i = 0; i < n_hidden_layers; i++) {
        int layer_size = hidden_layer_sizes[i];
        network->weights[i] = (float**)malloc(layer_size * sizeof(float*));
        int last_layer_size = (i == 0) ? input_size : hidden_layer_sizes[i - 1];
        for (int j = 0; j < layer_size; j++) {
            network->weights[i][j] = (float*)malloc(last_layer_size * sizeof(float));
            // Initialize weights with random values
            for (int k = 0; k < last_layer_size; k++) {
                network->weights[i][j][k] = ((float)rand() / RAND_MAX) - 0.5f;
            }
        }
        network->biases[i] = (float*)malloc(layer_size * sizeof(float));
        // Initialize biases with random values
        for (int j = 0; j < layer_size; j++) {
            network->biases[i][j] = ((float)rand() / RAND_MAX) - 0.5f;
        }
    }

    // Output weights
    int last_layer_size = hidden_layer_sizes[n_hidden_layers - 1];
    for (int k = 0; k < last_layer_size; k++) {
        network->output_weights[k] = ((float)rand() / RAND_MAX) - 0.5f;
    }
}

void free_nn(nn* network) {
    for (int i = 0; i < network->n_hidden_layers; i++) {
        int layer_size = network->hidden_layer_sizes[i];
        for (int j = 0; j < layer_size; j++) {
            free(network->weights[i][j]);
        }
        free(network->weights[i]);
        free(network->biases[i]);
    }
    free(network->weights);
    free(network->biases);
    if (network->input) {
        free(network->input);
    }
    if (network->output_weights) {
        free(network->output_weights);
    }
}

void feed_forward(nn* network, float (*activation)(float)) {
    float *input = (float*)malloc(network->input_size * sizeof(float));
    for (int i = 0; i < network->input_size; i++) {
        input[i] = network->input[i];
    }

    for (int layer = 0; layer < network->n_hidden_layers; layer++) {
        int layer_size = network->hidden_layer_sizes[layer];
        float* layer_output = (float*)malloc(layer_size * sizeof(float));

        for (int j = 0; j < layer_size; j++) {
            float sum = 0.0f;
            int prev_layer_size = (layer == 0) ? network->input_size : network->hidden_layer_sizes[layer - 1];
            for (int k = 0; k < prev_layer_size; k++) {
                sum += network->weights[layer][j][k] * input[k];
            }
            sum += network->biases[layer][j];
            layer_output[j] = activation(sum);
        }
        free(input);
        input = layer_output;
    }

    network->output = 0.0f;
    int last_layer_size = network->hidden_layer_sizes[network->n_hidden_layers - 1];
    for (int j = 0; j < last_layer_size; j++) {
        network->output += network->output_weights[j] * input[j];
    }
    network->output = activation(network->output);

    free(input);
}

void backpropagate(nn* network, float *input, float target, float (*activation)(float), float (*activation_derivative)(float), float learning_rate) {
    int n_layers = network->n_hidden_layers;
    float **x = (float**)malloc(n_layers * sizeof(float*));
    for (int i = 0; i < n_layers; i++)
        x[i] = (float*)malloc(network->hidden_layer_sizes[i] * sizeof(float));
    
    float **xhat = (float**)malloc(n_layers * sizeof(float*));
    for (int i = 0; i < n_layers; i++)
        xhat[i] = (float*)malloc(network->hidden_layer_sizes[i] * sizeof(float));
    
    // Forward pass
    for (int i = 0; i < network->hidden_layer_sizes[0]; i++) {
        float sum = 0.0f;
        for (int k = 0; k < network->input_size; k++) {
            sum += network->weights[0][i][k] * input[k];
        }
        sum += network->biases[0][i];
        x[0][i] = sum;
    }

    for (int layer = 1; layer < n_layers; layer++) {
        int layer_size = network->hidden_layer_sizes[layer];
        int prev_layer_size = network->hidden_layer_sizes[layer - 1];
        for (int i = 0; i < layer_size; i++) {
            float sum = 0.0f;
            for (int k = 0; k < prev_layer_size; k++) {
                sum += network->weights[layer][i][k] * activation(x[layer - 1][k]);
            }
            sum += network->biases[layer][i];
            x[layer][i] = sum;
        }
    }

    // Output layer
    float y = 0.0f;
    int last_layer_size = network->hidden_layer_sizes[n_layers - 1];
    for (int i = 0; i < last_layer_size; i++) {
        y += network->output_weights[i] * activation(x[n_layers - 1][i]);
    }

    float error = 0.5f * (target - activation(y)) * (target - activation(y));
    // float delta = - learning_rate * error / activation_derivative(y) / (target - activation(y));

    // Update output weights
    for (int i = 0; i < last_layer_size; i++) {
        float weight = network->output_weights[i];
        network->output_weights[i] -= learning_rate * error / activation_derivative(y) / (target - activation(y)) / activation(x[n_layers - 1][i]);
        xhat[n_layers - 1][i] = x[n_layers - 1][i] - error / activation_derivative(y) / (target - activation(y)) / weight;
    }

    for (int layer = n_layers - 1; layer > 0; layer--) {
        int layer_size = network->hidden_layer_sizes[layer];
        int prev_layer_size = network->hidden_layer_sizes[layer - 1];

        float error = 0.0f;
        for (int j = 0; j < layer_size; j++) {
            error += 0.5f * (x[layer][j] - xhat[layer][j]) * (x[layer][j] - xhat[layer][j]);
        }

        for (int i = 0; i < prev_layer_size; i++) {
            float derrordx = 0.0f;
            float derrordw = 0.0f;
            float derrordb = 0.0f;
            for (int j = 0; j < layer_size; j++) {
                derrordx += network->weights[layer][j][i] * activation_derivative(x[layer][j]) * (activation(x[layer][j]) - activation(xhat[layer][j]));
                derrordw = x[layer - 1][i] * activation_derivative(x[layer][j]) * (activation(x[layer][j]) - activation(xhat[layer][j]));
                derrordb = activation_derivative(x[layer][j]) * (activation(x[layer][j]) - activation(xhat[layer][j]));

                network->weights[layer][j][i] -= learning_rate * error / derrordw;
                network->biases[layer][j] -= learning_rate * error / derrordb;
            }
            xhat[layer - 1][i] = x[layer - 1][i] - error / derrordx;
        }
    }

    // Update first layer weights and biases
    for (int i = 0; i < network->hidden_layer_sizes[0]; i++) {
        float derrordw = 0.0f;
        float derrordb = 0.0f;
        for (int j = 0; j < network->input_size; j++) {
            derrordw = input[j] * activation_derivative(x[0][i]) * (activation(x[0][i]) - activation(xhat[0][i]));
            derrordb = activation_derivative(x[0][i]) * (activation(x[0][i]) - activation(xhat[0][i]));
            network->weights[0][i][j] -= learning_rate * error / derrordw;
            network->biases[0][i] -= learning_rate * error / derrordb;
        }
    }

    for (int i = 0; i < n_layers; i++){
        free(x[i]);
        free(xhat[i]);
    }
    free(x);
    free(xhat);
}
