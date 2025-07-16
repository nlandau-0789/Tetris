#include "utils.h"

struct nn {
    float *input;
    float ***weights, **biases, *output_weights;
    float output;
    int input_size, n_hidden_layers;
    int *hidden_layer_sizes;
};

typedef struct nn nn;

void init_nn(nn* network, int input_size, int n_hidden_layers, int* hidden_layer_sizes) {
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
    
    float **dedx = (float**)malloc(n_layers * sizeof(float*));
    for (int i = 0; i < n_layers; i++)
        dedx[i] = (float*)malloc(network->hidden_layer_sizes[i] * sizeof(float));
    
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

    // Standard output error and delta
    float output_error = activation(y) - target;
    float delta = output_error * activation_derivative(y);

    // Update output weights and dedx for next layer
    for (int i = 0; i < last_layer_size; i++) {
        network->output_weights[i] -= learning_rate * delta * activation(x[n_layers - 1][i]);
        dedx[n_layers - 1][i] = delta * network->output_weights[i];
    }

    // Hidden layers backpropagation
    for (int layer = n_layers - 1; layer > 0; layer--) {
        int layer_size = network->hidden_layer_sizes[layer];
        int prev_layer_size = network->hidden_layer_sizes[layer - 1];

        for (int i = 0; i < prev_layer_size; i++) {
            float error = 0.0f;
            for (int j = 0; j < layer_size; j++) {
                error += dedx[layer][j] * network->weights[layer][j][i];
            }
            dedx[layer - 1][i] = error * activation_derivative(x[layer - 1][i]);
        }

        for (int j = 0; j < layer_size; j++) {
            for (int k = 0; k < prev_layer_size; k++) {
                network->weights[layer][j][k] -= learning_rate * dedx[layer][j] * activation(x[layer - 1][k]);
            }
            network->biases[layer][j] -= learning_rate * dedx[layer][j];
        }
    }

    // First layer update
    for (int i = 0; i < network->hidden_layer_sizes[0]; i++) {
        for (int j = 0; j < network->input_size; j++) {
            network->weights[0][i][j] -= learning_rate * dedx[0][i] * input[j];
        }
        network->biases[0][i] -= learning_rate * dedx[0][i];
    }

    for (int i = 0; i < n_layers; i++){
        free(x[i]);
        free(dedx[i]);
    }
    free(x);
    free(dedx);
}

float ReLU(float x) {
    return ((x > 0) ? 1.0f : 0.1f)*x;
}

float ReLU_derivative(float x) {
    return (x > 0) ? 1.0f : 0.1f;
}

float f(float x) {
    // make the prime counting function using a sieve
    int y = (int)round(x * 100);
    int sieve[y + 1];
    sieve[0] = 0;
    sieve[1] = 0;
    sieve[2] = 1;
    for (int i = 3; i <= y; i++) {
        sieve[i] = 1; // assume all numbers are prime
    }
    for (int i = 2; i * i <= y; i++) {
        if (sieve[i]) {
            for (int j = i * i; j <= y; j += i) {
                sieve[j] = 0; // mark multiples of i as non-prime
            }
        }
    }
    int count = 0;
    for (int i = 2; i <= y; i++) {
        if (sieve[i]) {
            count++;
        }
    }
    return (float)count;
}

// make a quick test of the neural network
int main() {
    nn network;
    int hidden_layer_sizes[] = {100, 100, 100};
    init_nn(&network, 1, 3, hidden_layer_sizes);
    
    // Train the network on f
    for (int j = 0; j < 1000; j++) {
        printf("epoch %d\n", j);
        for (int i = 0; i < 10000; i++) {
            float input = (float)(i) / 10000.0f; // Random input between 0 and 1
            network.input[0] = input;
            // printf("Input: %.2f, is prime %f\n", input, f(input));
            // feed_forward(&network, ReLU);
            
            float target = f(input);
            backpropagate(&network, network.input, target, ReLU, ReLU_derivative, 0.0001f);
            // float MSE = 0.0f;
            // for (int i = 0; i < 1000; i++) {
            //     float input = (float)(rand() % 100) / 100.0f; // Random input between 0 and 1
            //     network.input[0] = input;
            //     feed_forward(&network, ReLU);
            //     float target = f(input);
            //     float error = network.output - target;
            //     MSE += error * error / 2.0f;
            // }
            // printf("Mean Squared Error: %.16f\r", MSE / (float)i);
        }
    }

    // Test the network
    network.input[0] = 0;
    feed_forward(&network, ReLU);
    float prev = network.output;
    for (int i = 1; i < 100; i++) {
        // float input = (float)(rand() % 1000000) / 100.0f; // Random input between 0 and 1
        network.input[0] = (float)i / 100.0f;
        feed_forward(&network, ReLU);
        // float target = f(input);
        // float error = network.output - target;
        // MSE += error * error / 2.0f;
        // printf("Mean Squared Error: %.16f\r\b\n", MSE / (float)i);
        // if (network.output-prev >= 0.5f) {
        //     printf("%d is prime", i);
        // }
        printf("%f\n", network.output - f( (float)i / 100.0f));
        prev = network.output;
    }
    free_nn(&network);
}