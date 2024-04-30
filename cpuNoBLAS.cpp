#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <cstdint>
#include <chrono> 
#include "utility.cpp"

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
    int** labels = encodeHotLabels("../train-labels-idx1-ubyte", imageCount, outputSize); 
    int** labelsBatch = (int**)malloc(ni * sizeof(int*));
    // load mnist test labels 
    int** testLabels = encodeHotLabels("../t10k-labels-idx1-ubyte", testImageCount, outputSize); 


    // Initialize training matrices --------------------------------------------
    double ***activationsMatrix = initActivationsMatrix(nl, inputSize, outputSize, nh, ni);
    double ***weights = initWeights(nl, inputSize, outputSize, nh); 
    double ***deltasMatrix = initDeltasMatrix(nl, outputSize, nh, ni);
    double ***scoresMatrix = initScoresMatrix(nl, outputSize, nh, ni);
    double **biases = initBiases(nl, outputSize, nh); 
    int* order = initOrder(imageCount); 
    double **sumD = initDeltaSum(nl, outputSize, nh); 
    double ***sumAD = initActivationDeltaSum(nl, inputSize, outputSize, nh); 


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
                        sumAD[i][j][k] = 0; 
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
                        activationsMatrix[0][k][c] = mnistImages[order[b*nb+i*chunkSize+k]][c];
                    }
                }

                // Forward Propogation -----------------------------------------
                for (int j = 0; j < nl; j++) {
                    int rows = nh;
                    int cols = (j == 0) ? inputSize : nh;
                    matrixMultiplyTranspose(activationsMatrix[j], weights[j], scoresMatrix[j+1], actualChunkSize, cols, rows);
                    matrixVectorAdd(scoresMatrix[j+1], biases[j], actualChunkSize, rows); 
                    for (int k = 0; k < actualChunkSize; k++){
                        relu(activationsMatrix[j+1][k], scoresMatrix[j+1][k], rows); 
                    }
                }

                matrixMultiplyTranspose(activationsMatrix[nl], weights[nl], scoresMatrix[nl+1], actualChunkSize, nh, outputSize);
                matrixVectorAdd(scoresMatrix[nl+1], biases[nl], actualChunkSize, outputSize); 
                for (int s = 0; s < actualChunkSize; s++){
                    softmax(activationsMatrix[nl+1][s], scoresMatrix[nl+1][s], outputSize); 
                }

                // Compute Loss
                for (int k = 0; k < actualChunkSize; k++){
                    cost += crossEntropy(labels[order[b*nb+i*chunkSize+k]],activationsMatrix[nl+1][k], outputSize);
                }
                
                // Backwards Propogation ---------------------------------------
                for (int k = 0; k < actualChunkSize; k++){
                    labelsBatch[k] = labels[order[b*nb+i*chunkSize+k]]; 
                }

                matrixSubtract(activationsMatrix[nl+1], labelsBatch, deltasMatrix[nl], actualChunkSize, outputSize); 
                matrixMultiplyTransposeA(deltasMatrix[nl], activationsMatrix[nl], activationsMatrix[nl], actualChunkSize, outputSize, nh);
                matrixAdd(activationsMatrix[nl], sumAD[nl], sumAD[nl], outputSize, nh); 
                matrixVectorAdd(deltasMatrix[nl], sumD[nl], sumD[nl], actualChunkSize, outputSize);
                
                for (int l = nl; l > 0; l--){
                    int rows = (l == nl) ? outputSize : nh;
                    int cols = nh;
                    int lowCols = (l == 1) ? inputSize : nh;

                    matrixMultiplyTranspose(deltasMatrix[l], weights[l], deltasMatrix[l-1], actualChunkSize, cols, rows); 

                    matrixMask(deltasMatrix[l-1], scoresMatrix[l], actualChunkSize, cols); 
                    matrixVectorAdd(deltasMatrix[l-1], sumD[l-1], sumD[l-1], actualChunkSize, cols);
                    matrixMultiplyTransposeAAdd(deltasMatrix[l-1],activationsMatrix[l-1], sumAD[l-1], actualChunkSize, cols, lowCols);
                }
            }

            // Update Weights Based on Errors ----------------------------------
            for (int l = nl; l >= 0; l--){
                int rows = (l == 0) ? nh : ((l == nl) ? outputSize : nh);
                int cols = (l == 0) ? inputSize : nh;
                scalarMatrixMultiplySub((alpha/currNb),sumAD[l],weights[l],rows,cols); 
                scalarVectorMultiplySub((alpha/currNb),sumD[l],biases[l],rows);  
            }
        }

        std::cout << "Loss for epoch " << e << " is " << cost / (60000) << std::endl; 
        std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count() << " milliseconds" << std::endl;



        // Predict the labels of test images -----------------------------------
        auto start2 = std::chrono::steady_clock::now();
        int numCorrect = 0; 
        for (int i = 0; i < testImageCount; i++){

            // initialize the inputs with current image 
            for (int c = 0; c < inputSize; c++){
                activationsMatrix[0][0][c] = mnistTestImages[i][c];
            }

            // forward propogation
            for (int j = 0; j < nl; j++) {
                int rows = nh;
                int cols = (j == 0) ? inputSize : nh;
                matrixVectorMultiply(weights[j], activationsMatrix[j][0], scoresMatrix[j+1][0], rows, cols);
                vectorAdd(scoresMatrix[j+1][0], biases[j], scoresMatrix[j+1][0], rows); 
                relu(activationsMatrix[j+1][0], scoresMatrix[j+1][0], rows); 
            }
            // output layer forward propogation
            matrixVectorMultiply(weights[nl], activationsMatrix[nl][0], scoresMatrix[nl+1][0], outputSize, nh);
            vectorAdd(scoresMatrix[nl+1][0], biases[nl], scoresMatrix[nl+1][0], outputSize); 
            softmax(activationsMatrix[nl+1][0], scoresMatrix[nl+1][0], outputSize);

            double maxVal = 0; 
            int maxIndex = 0; 
            for (int j = 0; j < outputSize; j++){
                if (activationsMatrix[nl+1][0][j] > maxVal) {
                    maxVal = activationsMatrix[nl+1][0][j]; 
                    maxIndex = j; 
                }
            }
            numCorrect += testLabels[i][maxIndex]; 
        }

        std::cout << "Prediction Rate: " << double(numCorrect) / testImageCount << std::endl;
        std::cout << "Prediction took " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start2).count() << " milliseconds" << std::endl;
    }


    // Free Matrices -----------------------------------------------------------
    free3DMatrix(activationsMatrix, nl+2, ni); 
    free3DMatrix(deltasMatrix, nl+1, ni); 
    free3DMatrix(scoresMatrix, nl+2, ni); 
    for (int l = 0; l < nl+1; l++){
        int rows = (l == nl) ? outputSize : nh; 
        freeMatrix(weights[l], rows); 
        freeMatrix(sumAD[l], rows); 
    }
    free(weights); 
    free(sumAD); 
    freeMatrix(mnistImages, imageCount); 
    freeMatrix(mnistTestImages, testImageCount);
    freeMatrix(labels, imageCount); 
    freeMatrix(testLabels, testImageCount); 
    freeMatrix(biases, nl+1);
    freeMatrix(sumD, nl+1); 
    free(order); 
    
    return 0;
}

