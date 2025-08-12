#include "utils.h"

struct nn {
    float *input;
    float ***weights, **biases, **output_weights;
    float *output;
    int input_size, n_hidden_layers, output_size;
    int *hidden_layer_sizes;
};

typedef struct nn nn;

void init_nn(nn* network, int input_size, int n_hidden_layers, int* hidden_layer_sizes, int output_size, float start_range, int seed) {
    network->input_size = input_size;
    network->n_hidden_layers = n_hidden_layers;
    network->output_size = output_size;
    network->hidden_layer_sizes = (int*)malloc(n_hidden_layers * sizeof(int));
    network->output_weights = (float**)malloc(output_size * sizeof(float*));
    for (int i = 0; i < n_hidden_layers; i++) {
        network->hidden_layer_sizes[i] = hidden_layer_sizes[i];
    }
    int last_layer_size = (n_hidden_layers > 0)?hidden_layer_sizes[n_hidden_layers - 1]:input_size;
    // Allocate memory for inputs, outputs, weights, and biases
    network->input = (float*)malloc(input_size * sizeof(float));
    network->output = (float*)malloc(output_size * sizeof(float));
    network->weights = (float***)malloc(n_hidden_layers * sizeof(float**));
    network->biases = (float**)malloc(n_hidden_layers * sizeof(float*));
    for (int i = 0; i < output_size; i++){
        network->output_weights[i] = (float*)malloc(last_layer_size * sizeof(float));
    }

    if (seed) {
        srand(seed);
    } else {
        srand((unsigned int)time(NULL));
    }

    for (int i = 0; i < n_hidden_layers; i++) {
        int layer_size = hidden_layer_sizes[i];
        network->weights[i] = (float**)malloc(layer_size * sizeof(float*));
        int last_layer_size = (i == 0) ? input_size : hidden_layer_sizes[i - 1];
        for (int j = 0; j < layer_size; j++) {
            network->weights[i][j] = (float*)malloc(last_layer_size * sizeof(float));
            for (int k = 0; k < last_layer_size; k++) {
                network->weights[i][j][k] = ((float)rand() / RAND_MAX) * 2.0f * start_range - start_range;
            }
        }

        network->biases[i] = (float*)malloc(layer_size * sizeof(float));
        for (int j = 0; j < layer_size; j++) {
            // network->biases[i][j] = ((float)rand() / RAND_MAX) * 2.0f * start_range - start_range;
            network->biases[i][j] = 0;
        }
    }
    for (int i = 0; i < output_size; i++){
        for (int k = 0; k < last_layer_size; k++) {
            network->output_weights[i][k] = ((float)rand() / RAND_MAX) * 2.0f * start_range - start_range;
        }
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
    for (int i = 0; i < network->output_size; i++){
        free(network->output_weights[i]);
    }
    if (network->weights){
        free(network->weights);
    }
    if (network->biases){
        free(network->biases);
    }
    if (network->hidden_layer_sizes){
        free(network->hidden_layer_sizes);
    }
    if (network->output_weights){
        free(network->output_weights);
    }
    if (network->output){
        free(network->output);
    }
    // if (network->input) {
    //     free(network->input);
    // }
}

void feed_forward(nn* network, float (*activation)(float)) {
    float *input = (float*)malloc(network->input_size * sizeof(float));
    for (int i = 0; i < network->input_size; i++) {
        input[i] = network->input[i];
    }

    // printf("%d inputs, %d hidden layers\n", network->input_size, network->n_hidden_layers);
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


    int last_layer_size = (network->n_hidden_layers > 0)?network->hidden_layer_sizes[network->n_hidden_layers - 1]:network->input_size;
    for (int i = 0; i < network->output_size; i++){
        network->output[i] = 0;
        for (int j = 0; j < last_layer_size; j++) {
            network->output[i] += network->output_weights[i][j] * input[j];
            // printf("input[%d]: %f %f\n", j, input[j], network->output_weights[j]);
        }
    }
    // network->output = activation(network->output);

    free(input);
}

void backpropagate(nn* network, float *input, float *target, float (*activation)(float), float (*activation_derivative)(float), float learning_rate) {
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
    float *y = malloc(sizeof(float) * network->output_size);
    int last_layer_size = (network->n_hidden_layers > 0)?network->hidden_layer_sizes[network->n_hidden_layers - 1]:network->input_size;
    for (int i = 0; i < network->output_size; i++){
        y[i] = 0;
        for (int j = 0; j < last_layer_size; j++) {
            y[i] += network->output_weights[i][j] * activation(x[n_layers - 1][j]);
        }
    }

    // Standard output error and delta
    float *output_error = malloc(sizeof(float) * network->output_size);
    float *delta = malloc(sizeof(float) * network->output_size);
    for (int i = 0; i < network->output_size; i++){
        output_error[i] = y[i] - target[i];
        delta[i] = output_error[i];
    }
    
    // printf("Output: %f, Target: %f, Error: %f\n", y, target, output_error);

    // Update output weights and dedx for next layer
    for (int j = 0; j < last_layer_size; j++) {
        dedx[n_layers - 1][j] = 0;
        for (int i = 0; i < network->output_size; i++) {
            network->output_weights[i][j] -= learning_rate * delta[i] * activation(x[n_layers - 1][j]);
            dedx[n_layers - 1][j] += delta[i] * network->output_weights[i][j];
        }
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

void batched_backpropagate(nn* network, float **input, float **target, int batch_size, float (*activation)(float), float (*activation_derivative)(float), float learning_rate) {
    int n_layers = network->n_hidden_layers;
    int last_layer_size = (network->n_hidden_layers > 0)?network->hidden_layer_sizes[network->n_hidden_layers - 1]:network->input_size;


    // printf("Batched backpropagation with batch size %d\n", batch_size);
    // Allocate memory for gradient accumulation
    float ***grad_weights = (float***)malloc(n_layers * sizeof(float**));
    float **grad_output_weights = (float**)malloc(network->output_size * sizeof(float*));
    float **grad_biases = (float**)malloc(n_layers * sizeof(float*));

    for (int layer = 0; layer < n_layers; layer++) {
        grad_weights[layer] = (float**)malloc(network->hidden_layer_sizes[layer] * sizeof(float*));
        grad_biases[layer] = (float*)calloc(network->hidden_layer_sizes[layer], sizeof(float));
        for (int i = 0; i < network->hidden_layer_sizes[layer]; i++) {
            grad_weights[layer][i] = (float*)calloc((layer==0)?network->input_size:network->hidden_layer_sizes[layer - 1], sizeof(float));
        }
    }

    for (int o = 0; o < network->output_size; o++){
        grad_output_weights[o] = (float*)calloc(last_layer_size, sizeof(float));
    }
    // printf("Memory allocated for gradients\n");

    // Temporary arrays for forward and backward pass
    float **x = (float**)malloc(n_layers * sizeof(float*));
    float **dedx = (float**)malloc(n_layers * sizeof(float*));

    for (int i = 0; i < n_layers; i++) {
        x[i] = (float*)malloc(network->hidden_layer_sizes[i] * sizeof(float));
        dedx[i] = (float*)malloc(network->hidden_layer_sizes[i] * sizeof(float));
    }
    // printf("Temporary arrays allocated\n");
    
    // Process each sample in the batch
    float *y = malloc(network->output_size * sizeof(float));
    float *output_error = malloc(network->output_size * sizeof(float));
    float *delta = malloc(network->output_size * sizeof(float));
    for (int b = 0; b < batch_size; b++) {
        // Forward pass
        for (int o = 0; o < network->output_size; o++){
            y[o] = 0;
        }
        for (int layer = 0; layer < n_layers; layer++){
            for (int i = 0; i < network->hidden_layer_sizes[layer]; i++){
                dedx[layer][i] = 0;
            }
        }

        if (network->n_hidden_layers > 0){
            for (int i = 0; i < network->hidden_layer_sizes[0]; i++) {
                float sum = 0.0f;
                for (int k = 0; k < network->input_size; k++) {
                    sum += network->weights[0][i][k] * input[b][k];
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
            for (int o = 0; o < network->output_size; o++){
                for (int i = 0; i < last_layer_size; i++) {
                    y[o] += network->output_weights[o][i] * activation(x[n_layers - 1][i]);
                }
            }
        } else {
            for (int o = 0; o < network->output_size; o++){
                for (int i = 0; i < last_layer_size; i++) {
                    y[o] += network->output_weights[o][i] * activation(input[b][i]);
                }
            }
        }
        
        // Backward pass
        for (int i = 0; i < last_layer_size; i++) {
            dedx[n_layers - 1][i] = 0;
        }
        for (int o = 0; o < network->output_size; o++){
            output_error[o] = y[o] - target[b][o];
            delta[o] = output_error[o];
            
            for (int i = 0; i < last_layer_size; i++) {
                grad_output_weights[o][i] += delta[o] * activation(x[n_layers - 1][i]);
                dedx[n_layers - 1][i] += delta[o] * network->output_weights[o][i];
            }
        }
        

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
                    grad_weights[layer][j][k] += dedx[layer][j] * activation(x[layer - 1][k]);
                }
                grad_biases[layer][j] += dedx[layer][j];
            }
        }
        if (network->n_hidden_layers > 0){
            for (int i = 0; i < network->hidden_layer_sizes[0]; i++) {
                for (int j = 0; j < network->input_size; j++) {
                    grad_weights[0][i][j] += dedx[0][i] * input[b][j];
                }
                grad_biases[0][i] += dedx[0][i];
            }
        } 
    }

    free(y);
    free(output_error);
    free(delta);

    // Update weights and biases using the gradient
    for (int o = 0; o < network->output_size; o++){
        for (int i = 0; i < last_layer_size; i++) {
            network->output_weights[o][i] -= learning_rate * grad_output_weights[o][i];
        }
    }

    for (int layer = 0; layer < n_layers; layer++) {
        for (int j = 0; j < network->hidden_layer_sizes[layer]; j++) {
            int prev_layer_size = (layer == 0) ? network->input_size : network->hidden_layer_sizes[layer - 1];
            for (int k = 0; k < prev_layer_size; k++) {
                network->weights[layer][j][k] -= learning_rate * grad_weights[layer][j][k];
            }
            network->biases[layer][j] -= learning_rate * grad_biases[layer][j];
        }
    }
    // printf("Weights and biases updated\n");

    // Free allocated memory
    for (int layer = 0; layer < n_layers; layer++) {
        for (int i = 0; i < network->hidden_layer_sizes[layer]; i++) {
            free(grad_weights[layer][i]);
        }
        free(grad_weights[layer]);
        free(grad_biases[layer]);
    }
    for (int o = 0; o < network->output_size; o++){
        free(grad_output_weights[o]);
    }
    free(grad_weights);
    free(grad_biases);
    free(grad_output_weights);

    for (int i = 0; i < n_layers; i++) {
        free(x[i]);
        free(dedx[i]);
    }
    free(x);
    free(dedx);
    // printf("Memory freed\n");
}


// float ReLU(float x) {
//     return ((x > 0) ? 1.0f : 0.1f)*x;
// }

// float ReLU_derivative(float x) {
//     return (x > 0) ? 1.0f : 0.1f;
// }

float ReLU(float x) {
    return ((x > 0) ? 1.0f : 0.0f)*x;
}

float ReLU_derivative(float x) {
    return (x > 0) ? 1.0f : 0.0f;
}

void weight_avg_nn(nn *network, nn *other_network, float alpha) {
    int last_layer_size = (network->n_hidden_layers > 0)?network->hidden_layer_sizes[network->n_hidden_layers - 1]:network->input_size;
    // Average the weights and biases of two networks
    for (int i = 0; i < network->n_hidden_layers; i++) {
        int layer_size = network->hidden_layer_sizes[i];
        int prev_layer_size = (i == 0) ? network->input_size : network->hidden_layer_sizes[i - 1];
        for (int j = 0; j < layer_size; j++) {
            for (int k = 0; k < prev_layer_size; k++) {
                network->weights[i][j][k] = alpha * network->weights[i][j][k] + (1 - alpha) * other_network->weights[i][j][k];
            }
            network->biases[i][j] = alpha * network->biases[i][j] + (1 - alpha) * other_network->biases[i][j];
        }
    }
    for (int o = 0; o < network->output_size; o++){
        for (int j = 0; j < last_layer_size; j++) {
            network->output_weights[o][j] = alpha * network->output_weights[o][j] + (1 - alpha) * other_network->output_weights[o][j];
        }
    }
}

// pretty print the neural network
void print_nn(nn *network) {
    int last_layer_size = (network->n_hidden_layers > 0)?network->hidden_layer_sizes[network->n_hidden_layers - 1]:network->input_size;
    printf("Neural Network:\n");
    printf("Input size: %d\n", network->input_size);
    printf("Number of hidden layers: %d\n", network->n_hidden_layers);
    for (int i = 0; i < network->n_hidden_layers; i++) {
        printf("Layer %d: size %d\n", i, network->hidden_layer_sizes[i]);
        printf("Weights:\n");
        for (int j = 0; j < network->hidden_layer_sizes[i]; j++) {
            printf("  Neuron %d: ", j);
            for (int k = 0; k < ((i == 0) ? network->input_size : network->hidden_layer_sizes[i - 1]); k++) {
                printf("%f ", network->weights[i][j][k]);
            }
            printf("\n");
            printf("    Bias: %f\n", network->biases[i][j]);
        }
    }
    printf("Output weights:\n");
    for (int o = 0; o < network->output_size; o++){
        printf("  Output %d: ", o);
        for (int j = 0; j < last_layer_size; j++) {
            printf("%f ", network->output_weights[o][j]);
        }
        printf("\n");
    } 
}

#include <stdio.h>
#include <stdlib.h>

void save_nn(const nn* network, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for writing");
        return;
    }

    // Save basic network configuration
    fwrite(&network->input_size, sizeof(int), 1, file);
    fwrite(&network->output_size, sizeof(int), 1, file);
    fwrite(&network->n_hidden_layers, sizeof(int), 1, file);

    // Save hidden layer sizes
    fwrite(network->hidden_layer_sizes, sizeof(int), network->n_hidden_layers, file);

    // Save weights and biases for each hidden layer
    for (int i = 0; i < network->n_hidden_layers; i++) {
        int layer_size = network->hidden_layer_sizes[i];
        for (int j = 0; j < layer_size; j++) {
            int prev_layer_size = (i == 0) ? network->input_size : network->hidden_layer_sizes[i - 1];
            fwrite(network->weights[i][j], sizeof(float), prev_layer_size, file);
        }
        fwrite(network->biases[i], sizeof(float), layer_size, file);
    }

    // Save output weights
    int last_layer_size = (network->n_hidden_layers > 0) ? network->hidden_layer_sizes[network->n_hidden_layers - 1] : network->input_size;
    for (int o = 0; o < network->output_size; o++){
        fwrite(network->output_weights[o], sizeof(float), last_layer_size, file);
    } 

    fclose(file);
}

void load_nn(nn* network, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file for reading");
        return;
    }

    // Load basic network configuration
    fread(&network->input_size, sizeof(int), 1, file);
    fread(&network->output_size, sizeof(int), 1, file);
    fread(&network->n_hidden_layers, sizeof(int), 1, file);

    // Allocate and load hidden layer sizes
    if (network->n_hidden_layers > 0){
        network->hidden_layer_sizes = (int*)malloc(network->n_hidden_layers * sizeof(int));
        fread(network->hidden_layer_sizes, sizeof(int), network->n_hidden_layers, file);
    } else {
        network->hidden_layer_sizes = NULL;
    }

    // Allocate input and output layer
    network->input = (float*)malloc(network->input_size * sizeof(float));
    network->output = (float*)malloc(network->output_size * sizeof(float));
    
    // Allocate and load weights and biases for each hidden layer
    network->weights = (float***)malloc(network->n_hidden_layers * sizeof(float**));
    network->biases = (float**)malloc(network->n_hidden_layers * sizeof(float*));
    for (int i = 0; i < network->n_hidden_layers; i++) {
        int layer_size = network->hidden_layer_sizes[i];
        network->weights[i] = (float**)malloc(layer_size * sizeof(float*));
        int prev_layer_size = (i == 0) ? network->input_size : network->hidden_layer_sizes[i - 1];
        for (int j = 0; j < layer_size; j++) {
            network->weights[i][j] = (float*)malloc(prev_layer_size * sizeof(float));
            fread(network->weights[i][j], sizeof(float), prev_layer_size, file);
        }
        network->biases[i] = (float*)malloc(layer_size * sizeof(float));
        fread(network->biases[i], sizeof(float), layer_size, file);
    }

    // Allocate and load output weights
    int last_layer_size = (network->n_hidden_layers > 0) ? network->hidden_layer_sizes[network->n_hidden_layers - 1] : network->input_size;
    network->output_weights = (float**)malloc(network->output_size * sizeof(float*));
    for (int o = 0; o < network->output_size; o++){
        network->output_weights[o] = malloc(last_layer_size * sizeof(float));
        fread(network->output_weights[o], sizeof(float), last_layer_size, file);
    }

    fclose(file);
}

// float f(float x) {
//     // make the prime counting function using a sieve
//     int y = (int)round(x * 100);
//     int sieve[y + 1];
//     sieve[0] = 0;
//     sieve[1] = 0;
//     sieve[2] = 1;
//     for (int i = 3; i <= y; i++) {
//         sieve[i] = 1; // assume all numbers are prime
//     }
//     for (int i = 2; i * i <= y; i++) {
//         if (sieve[i]) {
//             for (int j = i * i; j <= y; j += i) {
//                 sieve[j] = 0; // mark multiples of i as non-prime
//             }
//         }
//     }
//     int count = 0;
//     for (int i = 2; i <= y; i++) {
//         if (sieve[i]) {
//             count++;
//         }
//     }
//     return (float)count;
// }



// make a quick test of the neural network
#ifdef TEST_NN_BACKPROP
float f(float x){
    return x*x*x+x*x-x+1;
}

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
#endif

#ifdef TEST_NN_BACKPROP_BATCH
void f(float *input, float *output) {
    output[0] = input[0] + input[1];
    output[1] = input[0] * input[1];
}

int main() {
    nn network;
    int hidden_layer_sizes[] = {10, 10}, output_size = 2;
    init_nn(&network, 2, 2, hidden_layer_sizes, output_size, 0.1f, 0);

    int batch_size = 8; // Define the batch size
    float **batch_input = (float **)malloc(batch_size * sizeof(float *));
    float **batch_target = (float **)malloc(batch_size * sizeof(float *));

    for (int i = 0; i < batch_size; i++) {
        batch_input[i] = (float *)malloc(network.input_size * sizeof(float));
        batch_target[i] = (float *)malloc(network.output_size * sizeof(float));
    }

    // Train the network on f
    for (int j = 0; j < 1000; j++) {
        // printf("Epoch %d\n", j);
        float total_mse = 0.0f;

        for (int i = 0; i < 10000; i += batch_size) {
            // Prepare the batch
            for (int b = 0; b < batch_size; b++) {
                for (int k = 0; k < network.input_size; k++){
                    batch_input[b][k] = (float)(rand() % 1000000) / 1000000;
                }
                f(batch_input[b], batch_target[b]);
            }

            batched_backpropagate(&network, batch_input, batch_target, batch_size, ReLU, ReLU_derivative, 0.001f);

            // Calculate MSE for the batch
            float batch_mse = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                for (int k = 0; k < network.input_size; k++){
                    network.input[k] = batch_input[b][k];
                }
                feed_forward(&network, ReLU);
                // printf("Input: %.2f, Output: %.6f, Target: %.6f\n", batch_input[b][0], network.output, batch_target[b]);
                float error = 0;
                for (int k = 0; k < network.output_size; k++){
                    error += (network.output[k] - batch_target[b][k]) * (network.output[k] - batch_target[b][k]);
                }
                batch_mse += error;
            }
            batch_mse /= batch_size;
            total_mse += batch_mse;

            // Log MSE for the batch
            // printf("Batch %d: MSE = %.6f\n", i / batch_size, batch_mse);
        }

        // Log average MSE for the epoch
        printf("Average MSE for epoch %d: %.6f\n", j, total_mse / (10000 / batch_size));
    }

    // Free batch memory
    for (int i = 0; i < batch_size; i++) {
        free(batch_input[i]);
    }
    free(batch_input);
    free(batch_target);

    // Test the network
    // network.input[0] = 0;
    // feed_forward(&network, ReLU);
    // float prev = network.output;
    // for (int i = 1; i < 100; i++) {
    //     network.input[0] = (float)i / 100.0f;
    //     feed_forward(&network, ReLU);
    //     printf("%f\n", network.output - f((float)i / 100.0f));
    //     // prev = network.output;
    // }

    free_nn(&network);
    return 0;
}
#endif

#ifdef TEST_NN_AVG
int main(){
    nn a, b;

    int n = 2;
    init_nn(&a, 1, 1, &n, 2, 1, 1);
    init_nn(&b, 1, 1, &n, 2, 1, 2);

    print_nn(&a);
    print_nn(&b);
    
    weight_avg_nn(&a, &b, 0.5f);
    print_nn(&a);

    free_nn(&a);
    free_nn(&b);
}
#endif

#ifdef TEST_SAVE_LOAD
int main(){
    nn a, b, c, d;

    int n[] = {2, 2};
    init_nn(&a, 2, 2, n, 2, 1, 1);
    print_nn(&a);
    save_nn(&a, "network_test.model");
    load_nn(&b, "network_test.model");
    print_nn(&b);
    
    init_nn(&c, 2, 0, NULL, 2, 1, 1);
    print_nn(&c);
    save_nn(&c, "network_test.model");
    load_nn(&d, "network_test.model");
    print_nn(&d);

    free_nn(&a);
    free_nn(&b);
    free_nn(&c);
    free_nn(&d);
}
#endif