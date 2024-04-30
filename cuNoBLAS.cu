#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <cstdint>
#include <chrono> 
#include<cuda.h>
#include<curand.h>
#include<curand_kernel.h>
#include "cudaUtility.cu"

__global__ void doBatch(double ***d_activationsMatrix, double ***d_weights,
        double ***d_deltasMatrix, double ***d_scoresMatrix,
        double **d_biases, int* d_order, double **d_sumD,
        double ***d_sumAD, int inputSize, int outputSize, double** d_MnistImages, 
        int** d_labels, int b, int nb, int nl, int nh, double* cost){
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // initialize the inputs with current image 
    for (int c = 0; c < inputSize; c++){
        d_activationsMatrix[0][tid][c] = d_MnistImages[d_order[b*nb+tid]][c];
    }
    

    // forward propogation
    for (int j = 0; j < nl; j++) {
        int rows = (j == nl) ? outputSize : nh;
        int cols = (j == 0) ? inputSize : nh;
        matrixVectorMultiply(d_weights[j], d_activationsMatrix[j][tid], d_scoresMatrix[j+1][tid], rows, cols);
        vectorAdd(d_scoresMatrix[j+1][tid], d_biases[j], d_scoresMatrix[j+1][tid], rows); 
        relu(d_activationsMatrix[j+1][tid], d_scoresMatrix[j+1][tid], rows); 
    }
    
    matrixVectorMultiply(d_weights[nl], d_activationsMatrix[nl][tid], d_scoresMatrix[nl+1][tid], outputSize, nh);
    vectorAdd(d_scoresMatrix[nl+1][tid], d_biases[nl], d_scoresMatrix[nl+1][tid], outputSize); 
    softmax(d_activationsMatrix[nl+1][tid], d_scoresMatrix[nl+1][tid], outputSize); 
   

    // compute output error
    atomicAdd(cost, crossEntropy(d_labels[d_order[b*nb+tid]], d_activationsMatrix[nl+1][tid], outputSize));
    
    // back prop 
    vectorSubtract(d_activationsMatrix[nl+1][tid], d_labels[d_order[b*nb+tid]], d_deltasMatrix[nl][tid], outputSize);

    addVectorMultiplyTransposeAtomic(d_deltasMatrix[nl][tid], d_activationsMatrix[nl][tid], d_sumAD[nl], outputSize, nh); 
    vectorAddAtomic(d_deltasMatrix[nl][tid], d_sumD[nl],outputSize);

    for (int l = nl; l > 0; l--){
        int rows = (l == nl) ? outputSize : nh;
        int cols = nh;
        int lowCols = (l == 1) ? inputSize : nh;
        matrixVectorMultiplyTranspose(d_weights[l], d_deltasMatrix[l][tid], d_deltasMatrix[l-1][tid], rows, cols); 
        for (int s = 0; s < cols; s++){
            d_deltasMatrix[l-1][tid][s] *= (d_scoresMatrix[l][tid][s] > 0);
        }
        addVectorMultiplyTransposeAtomic(d_deltasMatrix[l-1][tid], d_activationsMatrix[l-1][tid], d_sumAD[l-1], cols, lowCols); 
        vectorAddAtomic(d_deltasMatrix[l-1][tid], d_sumD[l-1],cols);
    }
}


__global__ void updateWeights(int nl, int nh, int outputSize, int inputSize, double alpha, int currNb, 
                              double*** d_sumAD, double*** d_weights, double** d_sumD, 
                              double** d_biases){
    // Update Weights Based on Errors ----------------------------------
    for (int l = nl; l >= 0; l--){
        int rows = (l == 0) ? nh : ((l == nl) ? outputSize : nh);
        int cols = (l == 0) ? inputSize : nh;
        scalarMatrixMultiplySub((alpha/currNb),d_sumAD[l],d_weights[l],rows,cols); 
        scalarVectorMultiplySub((alpha/currNb),d_sumD[l],d_biases[l],rows);  
    }

    // Reset weight errors
    for (int i = 0; i < nl + 1; i++){
        int rows = (i == 0) ? nh : ((i == nl) ? outputSize : nh);
        int cols = (i == 0) ? inputSize : nh;
        for (int j = 0; j < rows; j++){
            for (int k = 0; k < cols; k++){
                d_sumAD[i][j][k] = 0; 
            }
        }
    }

    // Reset bias error
    for (int i = 0; i < nl + 1; i++){
        int rows = (i == nl) ? outputSize : nh;
        for (int j = 0; j < rows; j++){
            d_sumD[i][j] = 0; 
        }
    }
}



__global__ void predictTest(int testImageCount, double*** d_activationsMatrix, 
                            double** d_mnistTestImages, int nh, int nl, int inputSize,
                            int outputSize, double*** d_weights, double*** d_scoresMatrix, 
                            double** d_biases, int** d_testLabels, int* d_numCorrect, int iter){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;     

    // turn this into a kernel function ----------------------------------------
    // Predict the labels of test images ----------------------------------- 

    // initialize the inputs with current image 
    for (int c = 0; c < inputSize; c++){
        d_activationsMatrix[0][tid][c] = d_mnistTestImages[tid+iter*100][c];
    }

    // forward propogation
    for (int j = 0; j < nl; j++) {
        int rows = nh;
        int cols = (j == 0) ? inputSize : nh;
        matrixVectorMultiply(d_weights[j], d_activationsMatrix[j][tid], d_scoresMatrix[j+1][tid], rows, cols);
        vectorAdd(d_scoresMatrix[j+1][tid], d_biases[j], d_scoresMatrix[j+1][tid], rows); 
        relu(d_activationsMatrix[j+1][tid], d_scoresMatrix[j+1][tid], rows); 
    }
    // output layer forward propogation
    matrixVectorMultiply(d_weights[nl], d_activationsMatrix[nl][tid], d_scoresMatrix[nl+1][tid], outputSize, nh);
    vectorAdd(d_scoresMatrix[nl+1][tid], d_biases[nl], d_scoresMatrix[nl+1][tid], outputSize); 
    softmax(d_activationsMatrix[nl+1][tid], d_scoresMatrix[nl+1][tid], outputSize);

    double maxVal = 0; 
    int maxIndex = 0; 
    for (int j = 0; j < outputSize; j++){
        if (d_activationsMatrix[nl+1][tid][j] > maxVal) {
            maxVal = d_activationsMatrix[nl+1][tid][j]; 
            maxIndex = j; 
        }
    }
    if (d_testLabels[tid+iter*100][maxIndex]){
        atomicAdd(d_numCorrect, 1); 
    }
}


int main(int argc, char *argv[]) {
    if (argc != 6 && argc != 7) {
        printf("Usage: %s <nl> <nh> <ne> <nb> <alpha> (<ni>)\n", argv[0]);
        return 1;
    }

    // Parse Arguments ---------------------------------------------------------
    int nl = atoi(argv[1]); // Number of hidden layers
    int nh = atoi(argv[2]); // Number of activations in hidden layers
    int ne = atoi(argv[3]); // Number of epochs
    int nb = atoi(argv[4]); // Batch size
    double alpha = atof(argv[5]); // Learning rate
    int ni = nb; 
    if (argc == 7) {
        ni = atoi(argv[6]);
    }

    // Dimensions
    int inputSize = 784; // MNIST images are 28x28
    int outputSize = 10; // 10 classes for MNIST digits
    uint32_t imageCount = 60000, testImageCount = 10000; 


    // Get mnist data ----------------------------------------------------------
    // load mnist train images 
    double** mnistImages = readMnistImages("../train-images-idx3-ubyte", imageCount);
    double** d_MnistImages; 
    cudaMalloc(&d_MnistImages, imageCount * sizeof(double*));
    for (int i = 0; i < imageCount; ++i) {
        double* deviceImage;
        cudaMalloc(&deviceImage, inputSize * sizeof(double)); 
        cudaMemcpy(deviceImage, mnistImages[i], inputSize * sizeof(double), cudaMemcpyHostToDevice); 
        cudaMemcpy(&d_MnistImages[i], &deviceImage, sizeof(double*), cudaMemcpyHostToDevice); 
    }

    // load mnist test images 
    double** mnistTestImages = readMnistImages("../t10k-images-idx3-ubyte", testImageCount);
    double** d_mnistTestImages; 
    cudaMalloc(&d_mnistTestImages, testImageCount * sizeof(double*));
    for (int i = 0; i < testImageCount; ++i) {
        double* deviceImage;
        cudaMalloc(&deviceImage, inputSize * sizeof(double)); 
        cudaMemcpy(deviceImage, mnistTestImages[i], inputSize * sizeof(double), cudaMemcpyHostToDevice); 
        cudaMemcpy(&d_mnistTestImages[i], &deviceImage, sizeof(double*), cudaMemcpyHostToDevice); 
    }

    // load mnist train labels 
    int** labels = encodeHotLabels("../train-labels-idx1-ubyte", imageCount, outputSize);
    int** d_labels; 
    cudaMalloc(&d_labels, imageCount * sizeof(int*));
    for (int i = 0; i < imageCount; ++i) {
        int* deviceImage;
        cudaMalloc(&deviceImage, outputSize * sizeof(int)); 
        cudaMemcpy(deviceImage, labels[i], outputSize * sizeof(int), cudaMemcpyHostToDevice); 
        cudaMemcpy(&d_labels[i], &deviceImage, sizeof(int*), cudaMemcpyHostToDevice); 
    }

    // load mnist test labels 
    int** testLabels = encodeHotLabels("../t10k-labels-idx1-ubyte", testImageCount, outputSize); 
    int** d_testLabels; 
    cudaMalloc(&d_testLabels, testImageCount * sizeof(int*));
    for (int i = 0; i < testImageCount; ++i) {
        int* deviceImage;
        cudaMalloc(&deviceImage, outputSize * sizeof(int)); 
        cudaMemcpy(deviceImage, testLabels[i], outputSize * sizeof(int), cudaMemcpyHostToDevice); 
        cudaMemcpy(&d_testLabels[i], &deviceImage, sizeof(int*), cudaMemcpyHostToDevice); 
    }

    // Initialize training matrices --------------------------------------------
    double ***activationsMatrix = initActivationsMatrix(nl, inputSize, outputSize, nh, ni);
    double ***d_activationsMatrix = d_initActivationsMatrix(nl, inputSize, outputSize, nh, ni);
    double ***weights = initWeights(nl, inputSize, outputSize, nh); 
    double ***d_weights = d_initWeights(nl, inputSize, outputSize, nh); 
    double ***deltasMatrix = initDeltasMatrix(nl, outputSize, nh, ni);
    double ***d_deltasMatrix = d_initDeltasMatrix(nl, outputSize, nh, ni);
    double ***scoresMatrix = initScoresMatrix(nl, outputSize, nh, ni);
    double ***d_scoresMatrix = d_initScoresMatrix(nl, outputSize, nh, ni);
    double **biases = initBiases(nl, outputSize, nh); 
    double **d_biases = d_initBiases(nl, outputSize, nh); 
    int* order = initOrder(imageCount); 
    int* d_order;
    cudaMalloc((void**)&d_order, imageCount * sizeof(int));
    double **sumD = initDeltaSum(nl, outputSize, nh); 
    double **d_sumD = d_initDeltaSum(nl, outputSize, nh); 
    double ***sumAD = initActivationDeltaSum(nl, inputSize, outputSize, nh); 
    double ***d_sumAD = d_initActivationDeltaSum(nl, inputSize, outputSize, nh); 

    // training loop -----------------------------------------------------------
    double *cost = (double*)malloc(sizeof(double)); 
    double *d_cost; 
    cudaMalloc((void**)&d_cost, sizeof(double));
    for (int e = 0; e < ne; e++) {
        auto start = std::chrono::steady_clock::now();
    
        // shuffle the order of images for this epoch
        shuffle(order, imageCount); 
        cudaMemcpy(d_order, order, imageCount * sizeof(int), cudaMemcpyHostToDevice);

        *cost = 0; 
        cudaMemcpy(d_cost, cost, sizeof(double), cudaMemcpyHostToDevice);
        
        // Iterate Batches -----------------------------------------------------
        for (int b = 0; b < std::ceil(imageCount/double(nb)); b++) {
            // Check if batchs don't fit into number of samples
            int currNb = std::min(nb, int(imageCount - b*nb));
            doBatch<<<currNb, 1>>>(d_activationsMatrix, d_weights,
                d_deltasMatrix, d_scoresMatrix,
                d_biases, d_order, d_sumD,
                d_sumAD, inputSize, outputSize, d_MnistImages, 
                d_labels, b, nb, nl, nh, d_cost);
            cudaDeviceSynchronize();
            
            
            updateWeights<<<1,1>>>(nl, nh, outputSize, inputSize, alpha, currNb, 
                d_sumAD, d_weights, d_sumD, d_biases); 
            cudaDeviceSynchronize();
            
        }
        
        cudaMemcpy(cost, d_cost, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "Loss for epoch " << e << " is " << *cost / (60000) << std::endl; 
        std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count() << " milliseconds" << std::endl;

        int *numCorrect = (int*)malloc(sizeof(int)); 
        int* d_numCorrect; 
        cudaMalloc((void**)&d_numCorrect, sizeof(int));

        *numCorrect = 0; 
        cudaMemcpy(d_numCorrect, numCorrect, sizeof(int), cudaMemcpyHostToDevice);
        
        auto start2 = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; i++){
            predictTest<<<10,10>>>(testImageCount, d_activationsMatrix, 
                d_mnistTestImages, nh, nl, inputSize,
                outputSize, d_weights, d_scoresMatrix, 
                d_biases, d_testLabels, d_numCorrect, i);
            cudaDeviceSynchronize();
        }
    
        cudaError_t error3 = cudaGetLastError();
            if(error3 != cudaSuccess) {
                std::cout << "CUDA error3: " << cudaGetErrorString(error3) << std::endl;
                exit(-1);
            }
        cudaMemcpy(numCorrect, d_numCorrect, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Prediction Rate: " << double(*numCorrect) / testImageCount << std::endl;
        std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start2).count() << " milliseconds" << std::endl;
    }


    // Free Matrices -----------------------------------------------------------
    
    
    return 0;
}

