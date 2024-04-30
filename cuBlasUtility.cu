#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <random>
#include<cuda.h>

// Function to reverse the endianess of a 32-bit unsigned integer.
uint32_t reverse_endian(uint32_t n) {
    return ((n >> 24) & 0xFF) |
           ((n << 8) & 0xFF0000) |
           ((n >> 8) & 0xFF00) |
           ((n << 24) & 0xFF000000);
}

// Read the dataset and load it into a pointer to pointers format.
float** readMnistImages(const std::string& filename, uint32_t& number_of_images) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        uint32_t magic_number = 0;
        uint32_t number_of_rows = 0, number_of_columns = 0;

        // Read the magic number
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = reverse_endian(magic_number);
        
        // Ensure that we are reading the right data
        if (magic_number != 2051) {
            std::cerr << "Invalid MNIST image file!" << std::endl;
            return nullptr;
        }

        // Read metadata
        file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
        number_of_images = reverse_endian(number_of_images);
        file.read(reinterpret_cast<char*>(&number_of_rows), sizeof(number_of_rows));
        number_of_rows = reverse_endian(number_of_rows);
        file.read(reinterpret_cast<char*>(&number_of_columns), sizeof(number_of_columns));
        number_of_columns = reverse_endian(number_of_columns);

        // Calculate the total image size
        uint32_t image_size = number_of_rows * number_of_columns;

        // Allocate memory for all images
        uint8_t** images = (uint8_t**)malloc(number_of_images * sizeof(uint8_t*));
        float** mnistImages = (float**)malloc(number_of_images * sizeof(float*)); 

        for (int i = 0; i < number_of_images; ++i) {
            // Allocate memory for each image
            images[i] = (uint8_t*)malloc(image_size * sizeof(uint8_t));
            mnistImages[i] = (float*)malloc(image_size * sizeof(float)); 
            // Read the image pixel data directly into the matrix
            file.read(reinterpret_cast<char*>(images[i]), image_size);
            for (int j = 0; j < image_size; j++){
                mnistImages[i][j] = float(images[i][j] / 255.0);
            }
            free(images[i]);
        }
        free(images);
        return mnistImages;

    } else {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return nullptr;
    }
}

// Read the MNIST label file
uint8_t* readMnistLabelsFlat(const std::string& filename, uint32_t& number_of_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        uint32_t magic_number = 0;

        // Read the magic number
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = reverse_endian(magic_number);

        // Ensure that we are reading the right data
        if (magic_number != 2049) {
            std::cerr << "Invalid MNIST label file!" << std::endl;
            return nullptr;
        }

        // Read the number of labels
        file.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
        number_of_labels = reverse_endian(number_of_labels);

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

float** initActivationsMatrixFlat(int nl, int inputSize, int outputSize, int nh, int ni){
    float** activations = (float**)malloc((nl+2) * sizeof(float*));
    for (int i = 0; i < nl+2; i++) {
        int length = (i == 0) ? inputSize : (i == nl+1) ? outputSize : nh; 
        activations[i] = (float*)malloc(ni*length * sizeof(float));
    }
    return activations;
}

float** d_initActivationsMatrix(int nl, int inputSize, int outputSize, int nh, int ni){
    float** activations = (float**)malloc((nl+2) * sizeof(float*));
    for (int i = 0; i < nl+2; i++) {
        int length = (i == 0) ? inputSize : (i == nl+1) ? outputSize : nh; 
        cudaMalloc((void**)&activations[i], ni*length * sizeof(float));
    }
    return activations;
}

float** d_initWeights(int nl, int inputSize, int outputSize, int nh){
    float **weights = (float**)malloc((nl + 1) * sizeof(float**)); 

    std::random_device rd; 
    std::mt19937 gen(rd()); 
    
    for (int i = 0; i < nl + 1; i++){
        int rows = (i == 0) ? nh : ((i == nl) ? outputSize : nh);
        int cols = (i == 0) ? inputSize : nh;
        std::normal_distribution<float> d(0.0, sqrt(2.0/(cols))); 

        cudaMalloc((void**)&weights[i], rows * cols * sizeof(float));
        float* temp_layer = (float*)malloc(rows * cols * sizeof(float));
        for (int j = 0; j < rows; j++){
            for (int k = 0; k < cols; k++){
                temp_layer[j*cols+k] = d(gen); 
            }
        }
        cudaMemcpy(weights[i], temp_layer, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
        free(temp_layer); 
    }
    return weights; 
}

float** d_initDeltasMatrix(int nl, int outputSize, int nh, int ni){
    float **deltas = (float**)malloc((nl+1) * sizeof(float*));
    for (int j = 0; j < nl+1; j++) {
        int length = (j == nl) ? outputSize : nh; 
        cudaMalloc((void**)&deltas[j], ni * length * sizeof(float));
    }
    return deltas;
}

float** d_initScoresMatrix(int nl, int outputSize, int nh, int ni){
    float **scores = (float**)malloc((nl+2) * sizeof(float**));
    for (int j = 0; j < nl+2; j++) {
        int length = (j == nl+1) ? outputSize : nh; 
        cudaMalloc((void**)&scores[j], ni * length * sizeof(float));
    }
    return scores; 
}

float** d_initBiases(int nl, int outputSize, int nh){
    std::random_device rd; 
    std::mt19937 gen(rd()); 

    float **biases = (float**)malloc((nl+1) * sizeof(float*));
    for (int i = 0; i < nl+1; i++) {
        int length = (i == nl) ? outputSize : nh; 
        std::normal_distribution<float> d(0.0, sqrt(2.0/(length))); 
        cudaMalloc((void**)&biases[i], length * sizeof(float));
        float* temp_layer = (float*)malloc(length * sizeof(float));
        for (int j = 0; j < length; j++){
            temp_layer[j] = d(gen); 
        }
        cudaMemcpy( biases[i], temp_layer, length * sizeof(float), cudaMemcpyHostToDevice);
        free(temp_layer); 
    }
    return biases; 
}

float** d_initDeltaSum(int nl, int outputSize, int nh){
    float **sumD = (float**)malloc((nl+1) * sizeof(float*));
    for (int i = 0; i < nl+1; i++) {
        int length = (i == nl) ? outputSize : nh; 
        cudaMalloc((void**)&sumD[i], length * sizeof(float));
    }
    return sumD; 
}

float** d_initActivationDeltaSum(int nl, int inputSize, int outputSize, int nh){
    float **sumAD = (float**)malloc((nl + 1) * sizeof(float*)); 
    for (int i = 0; i < nl + 1; i++){
        int rows = (i == 0) ? nh : ((i == nl) ? outputSize : nh);
        int cols = (i == 0) ? inputSize : nh;
        cudaMalloc((void**)&sumAD[i], rows * cols * sizeof(float));
    }
    return sumAD; 
}

int* initOrder(int imageCount){
    int* order = (int*)malloc(imageCount * sizeof(int));
    for (int i = 0; i < imageCount; i++){
        order[i] = i; 
    }
    return order; 
}

// shuffles an array
void shuffle(int *array, size_t n)
{
    size_t i;
    for (i = 0; i < n - 1; i++) 
    {
        size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}