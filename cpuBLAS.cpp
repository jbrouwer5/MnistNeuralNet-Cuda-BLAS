#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <cstdint>
#include <chrono> 
#include "mkl.h"
#include "utility.cpp"
#include "flatUtility.cpp"

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
    // load mnist test images 
    double** mnistTestImages = readMnistImages("../t10k-images-idx3-ubyte", testImageCount);
    // load mnist train labels 
    int* labels = encodeHotLabelsFlat("../train-labels-idx1-ubyte", imageCount, outputSize); 
    double* labelsBatch = (double*)malloc(ni * outputSize * sizeof(double));
    // load mnist test labels 
    int** testLabels = encodeHotLabels("../t10k-labels-idx1-ubyte", testImageCount, outputSize); 


    // Initialize training matrices --------------------------------------------
    double **activationsMatrix = initActivationsMatrixFlat(nl, inputSize, outputSize, nh, ni);
    double **weights = initWeightsFlat(nl, inputSize, outputSize, nh); 
    double **deltasMatrix = initDeltasMatrixFlat(nl, outputSize, nh, ni);
    double **scoresMatrix = initScoresMatrixFlat(nl, outputSize, nh, ni);
    double **biases = initBiases(nl, outputSize, nh); 
    int* order = initOrder(imageCount); 
    double **sumD = initDeltaSum(nl, outputSize, nh); 
    double **sumAD = initActivationDeltaSumFlat(nl, inputSize, outputSize, nh); 

    // training loop -----------------------------------------------------------
    double cost; 
    for (int e = 0; e < ne; e++) {
        auto start = std::chrono::steady_clock::now();
    
        // shuffle the order of images for this epoch
        shuffle(order, imageCount); 
        cost = 0; 
        
        // Iterate Batches -----------------------------------------------------
        for (int b = 0; b < std::ceil(imageCount/double(nb)); b++) {
            
            // Reset weight errors
            for (int i = 0; i < nl + 1; i++){
                int rows = (i == 0) ? nh : ((i == nl) ? outputSize : nh);
                int cols = (i == 0) ? inputSize : nh;
                for (int j = 0; j < rows; j++){
                    for (int k = 0; k < cols; k++){
                        sumAD[i][j*cols+k] = 0; 
                    }
                }
            }

            // Reset bias error
            for (int i = 0; i < nl + 1; i++){
                int rows = (i == nl) ? outputSize : nh;
                for (int j = 0; j < rows; j++){
                    sumD[i][j] = 0; 
                }
            }

            // Check if batchs don't fit into number of samples
            int currNb = nb; 
            if (b == (imageCount / nb) && imageCount % nb != 0){
                currNb = imageCount % nb; 
            }

            // Breaks batch into chunks of concurrent images -------------------
            int chunkSize = std::min(currNb, ni); 
            for (int i = 0; i < currNb; i += chunkSize) {
                int actualChunkSize = std::min(chunkSize, currNb - i);

                // Put inputs into first activation layer
                for (int k = 0; k < actualChunkSize; k++){
                    for (int c = 0; c < inputSize; c++){
                        activationsMatrix[0][k*inputSize+c] = mnistImages[order[b*nb+i*chunkSize+k]][c];
                    }
                }

                // Forward Propogation -----------------------------------------
                for (int j = 0; j < nl; j++) {
                    int rows = nh;
                    int cols = (j == 0) ? inputSize : nh;
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, actualChunkSize,
                                rows, cols, 1.0, activationsMatrix[j], cols, 
                                weights[j], cols, 0.0, scoresMatrix[j+1], rows);
                    for (int s = 0; s < actualChunkSize; s++) { 
                        cblas_daxpy(rows, 1.0, biases[j], 1, scoresMatrix[j+1] 
                                    + s * rows, 1);                 
                    }
                    flatRelu(activationsMatrix[j+1], scoresMatrix[j+1], rows, actualChunkSize);
                }

                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, actualChunkSize, 
                                outputSize, nh, 1.0, activationsMatrix[nl], nh,             
                                weights[nl], nh, 0.0, scoresMatrix[nl+1], outputSize);
                for (int s = 0; s < actualChunkSize; s++) { 
                    cblas_daxpy(outputSize, 1.0, biases[nl], 1, scoresMatrix[nl+1] 
                                + s * outputSize, 1);                 
                }
                flatSoftmax(activationsMatrix[nl+1], scoresMatrix[nl+1], outputSize, actualChunkSize); 
                
                // Compute Loss
                for (int k = 0; k < actualChunkSize; k++){
                    for (int c = 0; c < outputSize; c++){ 
                        cost -= labels[order[b*nb+i*chunkSize+k]*outputSize+c] * log(activationsMatrix[nl+1][k*outputSize+c]); 
                    }
                }
                
                // Backwards Propogation ---------------------------------------
                for (int k = 0; k < actualChunkSize; k++){
                    for (int m = 0; m < outputSize; m++){
                        labelsBatch[k*outputSize+m] = labels[order[b*nb+i*chunkSize+k]*outputSize+m]; 
                    }
                }

                for (int i = 0; i < actualChunkSize; i++) {
                    cblas_dcopy(outputSize, &activationsMatrix[nl+1][i * outputSize], 1, &deltasMatrix[nl][i * outputSize], 1);
                    cblas_daxpy(outputSize, -1.0, &labelsBatch[i * outputSize], 1, &deltasMatrix[nl][i * outputSize], 1);
                }
                // stopped here
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, outputSize, 
                            nh, actualChunkSize, 1.0, deltasMatrix[nl], outputSize,
                            activationsMatrix[nl], nh, 1.0, sumAD[nl], nh);
                
                
                for (int i = 0; i < actualChunkSize; i++) {
                    cblas_daxpy(outputSize, 1.0, deltasMatrix[nl] + i * outputSize, 1, sumD[nl], 1);
                }
                
                for (int l = nl; l > 0; l--){
                    int rows = (l == nl) ? outputSize : nh;
                    int cols = nh;
                    int lowCols = (l == 1) ? inputSize : nh;

                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                                actualChunkSize, cols, rows, 
                                1.0, deltasMatrix[l], rows, 
                                weights[l], rows, 
                                0.0, deltasMatrix[l-1], cols);

                    flatMatrixMask(deltasMatrix[l-1], scoresMatrix[l], actualChunkSize, cols); 
                    for (int i = 0; i < actualChunkSize; i++) {
                        cblas_daxpy(cols, 1.0, deltasMatrix[l-1] + i * cols, 1, sumD[l-1], 1);
                    }
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                                cols, lowCols, actualChunkSize, 
                                1.0, deltasMatrix[l-1], cols, 
                                activationsMatrix[l-1], lowCols, 
                                1.0, sumAD[l-1], lowCols);

                }
            }
            // Update Weights Based on Errors ----------------------------------
            for (int l = nl; l >= 0; l--){
                int rows = (l == 0) ? nh : ((l == nl) ? outputSize : nh);
                int cols = (l == 0) ? inputSize : nh;
                cblas_daxpy(rows*cols, -(alpha / currNb), sumAD[l], 1, weights[l], 1);
                cblas_daxpy(rows, -(alpha / currNb), sumD[l], 1, biases[l], 1);
            }
        }

        std::cout << "Loss for epoch " << e << " is " << cost / (60000) << std::endl; 
        std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count() << " milliseconds" << std::endl;



        // Predict the labels of test images -----------------------------------
        auto start2 = std::chrono::steady_clock::now();
        int numCorrect = 0; 
        for (int i = 0; i < testImageCount; i += ni) {
            int actualChunkSize = std::min(ni, int(testImageCount) - i);
    
            // Put inputs into first activation layer
            for (int k = 0; k < actualChunkSize; k++){
                for (int c = 0; c < inputSize; c++){
                    activationsMatrix[0][k*inputSize+c] = mnistTestImages[i+k][c];
                }
            }

            // Forward Propogation -----------------------------------------
            for (int j = 0; j < nl; j++) {
                int rows = nh;
                int cols = (j == 0) ? inputSize : nh;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, actualChunkSize,
                            rows, cols, 1.0, activationsMatrix[j], cols, 
                            weights[j], cols, 0.0, scoresMatrix[j+1], rows);
                for (int s = 0; s < actualChunkSize; s++) { 
                    cblas_daxpy(rows, 1.0, biases[j], 1, scoresMatrix[j+1] 
                                + s * rows, 1);                 
                }
                flatRelu(activationsMatrix[j+1], scoresMatrix[j+1], rows, actualChunkSize);
            }

            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, actualChunkSize, 
                            outputSize, nh, 1.0, activationsMatrix[nl], nh,             
                            weights[nl], nh, 0.0, scoresMatrix[nl+1], outputSize);
            for (int s = 0; s < actualChunkSize; s++) { 
                cblas_daxpy(outputSize, 1.0, biases[nl], 1, scoresMatrix[nl+1] 
                            + s * outputSize, 1);                 
            }
            flatSoftmax(activationsMatrix[nl+1], scoresMatrix[nl+1], outputSize, actualChunkSize); 
            for (int s = 0; s < actualChunkSize; s++){
                double maxVal = 0; 
                int maxIndex = 0; 
                for (int j = 0; j < outputSize; j++){
                    if (activationsMatrix[nl+1][s*outputSize+j] > maxVal) {
                        maxVal = activationsMatrix[nl+1][s*outputSize+j]; 
                        maxIndex = j; 
                    }
                }
                numCorrect += testLabels[i+s][maxIndex]; 
            }
        }

        std::cout << "Prediction Rate: " << double(numCorrect) / testImageCount << std::endl;
        std::cout << "Prediction took " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start2).count() << " milliseconds" << std::endl;
    }

    // Free Matrices -----------------------------------------------------------
    // freeMatrix(activationsMatrix, nl+2); 
    // freeMatrix(deltasMatrix, nl+1);
    // freeMatrix(scoresMatrix, nl+2); 
    // freeMatrix(weights, nl+1); 
    // free(sumAD); 
    // freeMatrix(mnistImages, imageCount); 
    // freeMatrix(mnistTestImages, testImageCount);
    // freeMatrix(labels, imageCount); 
    // freeMatrix(testLabels, testImageCount); 
    // freeMatrix(biases, nl+1);
    // freeMatrix(sumD, nl+1); 
    // free(order); 
    
    return 0;
}

