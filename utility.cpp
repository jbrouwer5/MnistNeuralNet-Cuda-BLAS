#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <random>

// Function to reverse the endianess of a 32-bit unsigned integer.
uint32_t reverse_endian(uint32_t n) {
    return ((n >> 24) & 0xFF) |
           ((n << 8) & 0xFF0000) |
           ((n >> 8) & 0xFF00) |
           ((n << 24) & 0xFF000000);
}

// Read the dataset and load it into a pointer to pointers format.
double** readMnistImages(const std::string& filename, uint32_t& number_of_images) {
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
        double** mnistImages = (double**)malloc(number_of_images * sizeof(double*)); 

        for (int i = 0; i < number_of_images; ++i) {
            // Allocate memory for each image
            images[i] = (uint8_t*)malloc(image_size * sizeof(uint8_t));
            mnistImages[i] = (double*)malloc(image_size * sizeof(double)); 
            // Read the image pixel data directly into the matrix
            file.read(reinterpret_cast<char*>(images[i]), image_size);
            for (int j = 0; j < image_size; j++){
                mnistImages[i][j] = double(images[i][j] / 255.0);
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
uint8_t* readMnistLabels(const std::string& filename, uint32_t& number_of_labels) {
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

int** encodeHotLabels(const std::string& filename, uint32_t imageCount, int outputSize){
    uint8_t* hotLabels = readMnistLabels(filename, imageCount);
    int** labels = (int**)malloc(imageCount * sizeof(int*));
    for (int i = 0; i < imageCount; i++){
        labels[i] = (int*)malloc(outputSize * sizeof(int));
        for (int j = 0; j < outputSize; j++){
            labels[i][j] = 0;
        }
        labels[i][int(hotLabels[i])] = 1;
    }
    delete[] hotLabels; 
    return labels;
}

double*** initActivationsMatrix(int nl, int inputSize, int outputSize, int nh, int ni){
    double*** activations = (double***)malloc((nl+2) * sizeof(double**));
    for (int i = 0; i < nl+2; i++) {
        activations[i] = (double**)malloc(ni * sizeof(double*));
        for (int j = 0; j < ni; j++){
            int length = (i == 0) ? inputSize : (i == nl+1) ? outputSize : nh; 
            activations[i][j] = (double*)malloc(length * sizeof(double));
        }
    }
    return activations;
}

double*** initWeights(int nl, int inputSize, int outputSize, int nh){
    double ***weights = (double***)malloc((nl + 1) * sizeof(double**)); 

    std::random_device rd; 
    std::mt19937 gen(rd()); 
    
    for (int i = 0; i < nl + 1; i++){
        int rows = (i == 0) ? nh : ((i == nl) ? outputSize : nh);
        int cols = (i == 0) ? inputSize : nh;
        std::normal_distribution<double> d(0.0, sqrt(2.0/(cols))); 
        weights[i] = (double**)malloc(rows * sizeof(double*));
        for (int j = 0; j < rows; j++){
            weights[i][j] = (double*)malloc(cols * sizeof(double));
            for (int k = 0; k < cols; k++){
                weights[i][j][k] = d(gen); 
            }
        }
    }
    return weights; 
}

double*** initDeltasMatrix(int nl, int outputSize, int nh, int ni){
    double ***deltas = (double***)malloc((nl+1) * sizeof(double**));
    for (int j = 0; j < nl+1; j++) {
        deltas[j] = (double**)malloc(ni * sizeof(double*));
        for (int i = 0; i < ni; i++){
            int length = (j == nl) ? outputSize : nh; 
            deltas[j][i] = (double*)malloc(length * sizeof(double));
        }
    }
    return deltas;
}

double*** initScoresMatrix(int nl, int outputSize, int nh, int ni){
    double ***scores = (double***)malloc((nl+2) * sizeof(double**));
    for (int j = 0; j < nl+2; j++) {
        scores[j] = (double**)malloc(ni * sizeof(double*));
        for (int i = 0; i < ni; i++){
            int length = (j == nl+1) ? outputSize : nh; 
            scores[j][i] = (double*)malloc(length * sizeof(double));
        }
    }
    return scores; 
}

double** initBiases(int nl, int outputSize, int nh){
    std::random_device rd; 
    std::mt19937 gen(rd()); 

    double **biases = (double**)malloc((nl+1) * sizeof(double*));
    for (int i = 0; i < nl+1; i++) {
        int length = (i == nl) ? outputSize : nh; 
        std::normal_distribution<double> d(0.0, sqrt(2.0/(length))); 
        biases[i] = (double*)malloc(length * sizeof(double));
        for (int j = 0; j < length; j++){
            biases[i][j] = d(gen); 
        }
    }
    return biases; 
}

int* initOrder(int imageCount){
    int* order = (int*)malloc(imageCount * sizeof(int));
    for (int i = 0; i < imageCount; i++){
        order[i] = i; 
    }
    return order; 
}

double** initDeltaSum(int nl, int outputSize, int nh){
    double **sumD = (double**)malloc((nl+1) * sizeof(double*));
    for (int i = 0; i < nl+1; i++) {
        int length = (i == nl) ? outputSize : nh; 
        sumD[i] = (double*)malloc(length * sizeof(double));
    }
    return sumD; 
}

double*** initActivationDeltaSum(int nl, int inputSize, int outputSize, int nh){
    double ***sumAD = (double***)malloc((nl + 1) * sizeof(double**)); 
    for (int i = 0; i < nl + 1; i++){
        int rows = (i == 0) ? nh : ((i == nl) ? outputSize : nh);
        int cols = (i == 0) ? inputSize : nh;
        sumAD[i] = (double**)malloc(rows * sizeof(double*));
        for (int j = 0; j < rows; j++){
            sumAD[i][j] = (double*)malloc(cols * sizeof(double));
            for (int k = 0; k < cols; k++){
                sumAD[i][j][k] = 0; 
            }
        }
    }
    return sumAD; 
}

// Deallocate matrix
void freeMatrix(double **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void freeMatrix(int** matrix, int rows){
    for (int i = 0; i < rows; i++){
        free(matrix[i]); 
    }
    free(matrix); 
}

void free3DMatrix(double ***matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++){
            free(matrix[i][j]); 
        }
        free(matrix[i]);
    }
    free(matrix);
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

// relu activation
void relu(double* results, double* inputs, int length){
    for (int i = 0; i < length; i++){
        results[i] = std::max(inputs[i], 0.0); 
    }
}

void rectifyGradient(double** deltas, double** scores, int cols, int actualChunkSize){
    for (int k = 0; k < actualChunkSize; k++){
        for (int j = 0; j < cols; ++j) {
            deltas[j][k] = (scores[j][k] > 0) ? deltas[j][k] : 0;
        }
    }
}

void softmax(double* results, double* inputs, int length){
    double maxInput = inputs[0];
    for (int i = 1; i < length; i++){
        if (inputs[i] > maxInput) {
            maxInput = inputs[i];
        }
    }

    double sum = 0;
    for (int j = 0; j < length; j++){
        sum += std::exp(inputs[j] - maxInput); 
    }

    for (int i = 0; i < length; i++){
        results[i] = std::exp(inputs[i] - maxInput) / sum; 
    }
}

// elementwise addition between two matrices 
void matrixAdd(double** matrixA, double** matrixB, double** resultMatrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            resultMatrix[i][j] = matrixA[i][j] + matrixB[i][j];
        }
    }
}

// relu derivative mask onto A of B
void matrixMask(double** matrixA, double** matrixB, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrixA[i][j] *= (matrixB[i][j] > 0);
        }
    }
}

// elementwise addition between two matrices 
void matrixVectorAdd(double** matrixA, double* matrixB, double* resultMatrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            resultMatrix[j] += matrixA[i][j];
        }
    }
}

// elementwise subtraction between two matrices 
void matrixSubtract(double** matrixA, int** matrixB, double** resultMatrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            resultMatrix[i][j] = matrixA[i][j] - matrixB[i][j];
        }
    }
}

// element wise addition between two vectors 
void vectorAdd(double* vectorA, double* vectorB, double* resultMatrix, int length) {
    for (int i = 0; i < length; ++i) {
            resultMatrix[i] = vectorA[i] + vectorB[i];
    }
}

// matrix multiplication between a matrix and a vector
void matrixVectorMultiply(double** matrixA, double* matrixB, double* resultMatrix, int rowsA, int colsA) {
    for (int i = 0; i < rowsA; ++i) {
        resultMatrix[i] = 0; // Initialize the current element
        for (int k = 0; k < colsA; ++k) { // Or rowsB
            resultMatrix[i] += matrixA[i][k] * matrixB[k];
        }
    }
}

// b is transposed
void matrixMultiplyTranspose(double** A, double** B, double** C, int A_rows, int A_cols, int B_rows) {
    // Initialize the result matrix C to zero
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_rows; ++j) {
            C[i][j] = 0.0;
        }
    }

    // Perform matrix multiplication with A and the transpose of B
    for (int i = 0; i < A_rows; ++i) { // Iterate over rows of A and C
        for (int j = 0; j < B_rows; ++j) { // Iterate over rows of B (columns of B's transpose) and C
            for (int k = 0; k < A_cols; ++k) { // Iterate over columns of A and rows of B (since B is transposed)
                C[i][j] += A[i][k] * B[j][k]; // Note the use of B[j][k] instead of B[k][j]
            }
        }
    }
}

void matrixMultiplyTransposeA(double** A, double** B, double** C, int A_rows, int A_cols, int B_cols) {
    double** temp = (double**)malloc(A_cols * sizeof(double*));
    for (int i = 0; i < A_cols; i++) {
        temp[i] = (double*)malloc(B_cols * sizeof(double));
        for (int j = 0; j < B_cols; j++){
            temp[i][j] = 0; 
        }
    }
    
    // Perform matrix multiplication with the transpose of A and B
    for (int i = 0; i < A_cols; ++i) { // Iterate over columns of A (rows of A's transpose) and rows of C
        for (int j = 0; j < B_cols; ++j) { // Iterate over columns of B and columns of C
            for (int k = 0; k < A_rows; ++k) { // Iterate over rows of A (columns of A's transpose)
                temp[i][j] += A[k][i] * B[k][j]; // Note the use of A[k][i] instead of A[i][k] since A is transposed
            }
        }
    }

    for (int i = 0; i < A_cols; i++) {
        for (int j = 0; j < B_cols; j++){
            C[i][j] = temp[i][j]; 
        }
        free(temp[i]);
    }
    free(temp); 
}

void matrixMultiplyTransposeAAdd(double** deltasMatrix, double** activationsMatrix, double** sumAD, int actualChunkSize, int cols, int lowCols) {
    // Assuming deltasMatrix is of size [actualChunkSize x cols]
    // Assuming activationsMatrix is of size [actualChunkSize x lowCols]
    // Assuming sumAD is of size [cols x lowCols]

    // Perform matrix multiplication with the first matrix transposed
    for (int i = 0; i < cols; ++i) { // Loop over rows of the transposed deltasMatrix (which are columns of the original)
        for (int j = 0; j < lowCols; ++j) { // Loop over columns of activationsMatrix
            double sum = 0.0; // Temporary variable to store the dot product
            for (int k = 0; k < actualChunkSize; ++k) { // Loop over elements of the dot product
                sum += deltasMatrix[k][i] * activationsMatrix[k][j]; // Note the use of [k][i] since deltasMatrix is transposed
            }
            sumAD[i][j] += sum; // Add the computed value to the corresponding element in sumAD
        }
    }
}

void matrixVectorAdd(double** matrix, double* vector, int rows, int cols) {
    // Iterate over each row of the matrix
    for (int i = 0; i < rows; ++i) {
        // Iterate over each column of the row
        for (int j = 0; j < cols; ++j) {
            // Add the vector element to the corresponding element in the matrix row
            matrix[i][j] += vector[j];
        }
    }
}

// multiplies matrixA by a scalar and subtracts from resultMatrix
void scalarMatrixMultiplySub(double scalar, double** matrixA, double** resultMatrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            resultMatrix[i][j] -= matrixA[i][j] * scalar; 
        }
    }
}

// subtracts the product of the scalar and the given vector from the result vector 
void scalarVectorMultiplySub(double scalar, double* vectorA, double* resultVector, int length){
    for (int i = 0; i < length; ++i) {
        resultVector[i] -= vectorA[i] * scalar; 
    }
}

double crossEntropy(const int* a, const double* b, int size){
    double sum = 0; 
    for (int c = 0; c < size; c++){
        sum -= a[c] * log(b[c]); 
    }
    return sum; 
}
