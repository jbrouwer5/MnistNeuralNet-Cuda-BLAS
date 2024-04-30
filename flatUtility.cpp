#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <random>

double** initActivationsMatrixFlat(int nl, int inputSize, int outputSize, int nh, int ni){
    double** activations = (double**)malloc((nl+2) * sizeof(double*));
    for (int i = 0; i < nl+2; i++) {
        int length = (i == 0) ? inputSize : (i == nl+1) ? outputSize : nh; 
        activations[i] = (double*)malloc(ni*length * sizeof(double));
    }
    return activations;
}

double** initWeightsFlat(int nl, int inputSize, int outputSize, int nh){
    double **weights = (double**)malloc((nl + 1) * sizeof(double**)); 

    std::random_device rd; 
    std::mt19937 gen(rd()); 
    
    for (int i = 0; i < nl + 1; i++){
        int rows = (i == 0) ? nh : ((i == nl) ? outputSize : nh);
        int cols = (i == 0) ? inputSize : nh;
        std::normal_distribution<double> d(0.0, sqrt(2.0/(cols))); 
        weights[i] = (double*)malloc(rows * cols * sizeof(double));
        for (int j = 0; j < rows; j++){
            for (int k = 0; k < cols; k++){
                weights[i][j*cols+k] = d(gen); 
            }
        }
    }
    return weights; 
}

double** initDeltasMatrixFlat(int nl, int outputSize, int nh, int ni){
    double **deltas = (double**)malloc((nl+1) * sizeof(double*));
    for (int j = 0; j < nl+1; j++) {
        int length = (j == nl) ? outputSize : nh; 
        deltas[j] = (double*)malloc(ni * length * sizeof(double));
    }
    return deltas;
}

double** initScoresMatrixFlat(int nl, int outputSize, int nh, int ni){
    double **scores = (double**)malloc((nl+2) * sizeof(double**));
    for (int j = 0; j < nl+2; j++) {
        int length = (j == nl+1) ? outputSize : nh; 
        scores[j] = (double*)malloc(ni * length * sizeof(double));
    }
    return scores; 
}

// relu activation
void flatRelu(double* results, double* inputs, int length, int numImages){
    for (int k = 0; k < numImages; k++){
        for (int i = 0; i < length; i++){
            results[k*length+i] = std::max(inputs[k*length+i], 0.0); 
        }
    }
}

void flatSoftmax(double* results, double* inputs, int length, int numImages){
    for (int k = 0; k < numImages; k++){
        double maxInput = inputs[k*length];
        for (int i = 1; i < length; i++){
            if (inputs[k*length+i] > maxInput) {
                maxInput = inputs[k*length+i];
            }
        }

        double sum = 0;
        for (int j = 0; j < length; j++){
            sum += std::exp(inputs[k*length+j] - maxInput); 
        }

        for (int i = 0; i < length; i++){
            results[k*length+i] = std::exp(inputs[k*length+i] - maxInput) / sum; 
        }
    }
}

double** initActivationDeltaSumFlat(int nl, int inputSize, int outputSize, int nh){
    double **sumAD = (double**)malloc((nl + 1) * sizeof(double*)); 
    for (int i = 0; i < nl + 1; i++){
        int rows = (i == 0) ? nh : ((i == nl) ? outputSize : nh);
        int cols = (i == 0) ? inputSize : nh;
        sumAD[i] = (double*)malloc(rows * cols * sizeof(double));
    }
    return sumAD; 
}

// double flatCrossEntropy(const int* a, const double* b, int size, int imageCount){
//     double sum = 0; 
//     for (int k = 0; k < imageCount; k++){
//         for (int c = 0; c < size; c++){
//             sum -= a[k*size+c] * log(b[k*size+c]); 
//         }
//     }
//     return sum; 
// }

// relu derivative mask onto A of B
void flatMatrixMask(double* matrixA, double* matrixB, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrixA[i*cols+j] *= (matrixB[i*cols+j] > 0);
        }
    }
}

// Function to reverse the endianess of a 32-bit unsigned integer.
uint32_t reverse_endianFlat(uint32_t n) {
    return ((n >> 24) & 0xFF) |
           ((n << 8) & 0xFF0000) |
           ((n >> 8) & 0xFF00) |
           ((n << 24) & 0xFF000000);
}

// Read the MNIST label file
uint8_t* readMnistLabelsFlat(const std::string& filename, uint32_t& number_of_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        uint32_t magic_number = 0;

        // Read the magic number
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = reverse_endianFlat(magic_number);

        // Ensure that we are reading the right data
        if (magic_number != 2049) {
            std::cerr << "Invalid MNIST label file!" << std::endl;
            return nullptr;
        }

        // Read the number of labels
        file.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
        number_of_labels = reverse_endianFlat(number_of_labels);

        // Allocate memory for the labels
        uint8_t* labels = new uint8_t[number_of_labels];

        // Read the label data directly into the array
        file.read(reinterpret_cast<char*>(labels), number_of_labels);

        return labels;
    } else {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return nullptr;
    }
}

int* encodeHotLabelsFlat(const std::string& filename, uint32_t imageCount, int outputSize){
    uint8_t* hotLabels = readMnistLabelsFlat(filename, imageCount);
    int* labels = (int*)malloc(imageCount * outputSize * sizeof(int));
    for (int i = 0; i < imageCount; i++){
        for (int j = 0; j < outputSize; j++){
            labels[i*outputSize+j] = 0;
        }
        labels[i*outputSize+(hotLabels[i])] = 1;
    }
    delete[] hotLabels; 
    return labels;
}