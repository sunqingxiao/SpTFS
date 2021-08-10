#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// base feature set for 3d-tensor
typedef struct baseFeatureSet {
    int order;
    int* dim;
    int nnz;
    float density;
    float* ave_nnz;
    float* max_nnz;
    float* min_nnz;
    float* dev_nnz;
    float* bounce;
    float mean_neighbor;
} baseFeatureSet;

// csf features set for 3d-tensor
typedef struct csfFeatureSet {
    int order;
    int* dim;
    int nnz;
    float density;
    int numSlices;
    int numFibers;
    float sliceRatio;
    float fiberRatio;
    float maxNnzPerSlice;
    float minNnzPerSlice;
    float aveNnzPerSlice;
    float devNnzPerSlice;
    float adjNnzPerSlice;
    float maxFibersPerSlice;
    float minFibersPerSlice;
    float aveFibersPerSlice;
    float devFibersPerSlice;
    float adjFibersPerSlice;
    float maxNnzPerFiber;
    float minNnzPerFiber;
    float aveNnzPerFiber;
    float devNnzPerFiber;
    float adjNnzPerFiber;
} csfFeatureSet;

// compare two values
int compare(int x, int y) {
    if (x > y)
        return 1;
    else
        return 0;
}

// get max value
float getMax(int* value, int num) {
    int i;
    float maxValue = value[0];
    for (i = 1; i < num; i++)
        if (value[i] > maxValue)
            maxValue = value[i];
    return maxValue;
}

// get min value
float getMin(int* value, int num) {
    int i;
    float minValue = value[0];
    for (i = 1; i < num; i++)
        if (value[i] < minValue)
            minValue = value[i];
    return minValue;
}

// 3d tensor flatten mapping
int* getFlattenInput(char* tns_dir, int output_resolution) {
    int* imgs = (int*)malloc(sizeof(int) * 3 * output_resolution * output_resolution);
    int order, i, j, k;
    FILE* fpt;
    fpt = fopen(tns_dir, "r");
    if (fpt == NULL)
        printf("Error while opening tensor file!\n");
    // get dims
    fscanf(fpt, "%d", &order);
    int* dim = (int*)malloc(sizeof(int) * order);
    fscanf(fpt, "%d %d %d", &dim[0], &dim[1], &dim[2]);
    // get the dims of three modes
    int* dim_mode = (int*)malloc(sizeof(int) * 3 * 2);
    float* scaledim = (float*)malloc(sizeof(float) * 3 * 2);
    int* mode_valdim = (int*)malloc(sizeof(int) * 3 * 2);
    int* indexdim = (int*)malloc(sizeof(int) * 3 * 2);
    int* valdim = (int*)malloc(sizeof(int) * 3);
    float value;
    dim_mode[0] = dim[0]; dim_mode[1] = dim[1] * dim[2];
    dim_mode[2] = dim[1]; dim_mode[3] = dim[0] * dim[2];
    dim_mode[4] = dim[2]; dim_mode[5] = dim[0] * dim[1];
    // get the scaldim
    for (i = 0; i < 3; i++)
        for (j = 0; j < 2; j++)
            scaledim[i*2+j] = dim_mode[i*2+j] / (output_resolution * 1.0);
    // init imgs
    for (i = 0; i < 3 * output_resolution * output_resolution; i++)
        imgs[i] = 0;
    //get the density sampling of three images
    while (fscanf(fpt, "%d %d %d %f\n", &valdim[0], &valdim[1], &valdim[2], &value) != EOF) {
        mode_valdim[0] = valdim[0]-1; mode_valdim[1] = (valdim[2]-1)*dim[1]+valdim[1]-1;
        mode_valdim[2] = valdim[1]-1; mode_valdim[3] = (valdim[2]-1)*dim[0]+valdim[0]-1;
        mode_valdim[4] = valdim[2]-1; mode_valdim[5] = (valdim[1]-1)*dim[0]+valdim[0]-1;
        for (i = 0; i < 3; i++)
            for (j = 0; j < 2; j++)
                indexdim[i*2+j] = (int)(mode_valdim[i*2+j] / scaledim[i*2+j]);
        for (i = 0; i < 3; i++)
            imgs[output_resolution*(i*output_resolution+indexdim[i*2])+indexdim[i*2+1]] += 1;
    }
    fclose(fpt);

    // free arrays
    free(dim);
    free(scaledim);
    free(dim_mode);
    free(mode_valdim);
    free(indexdim);
    free(valdim);

    return imgs;
}

// 3d tensor map sampling
int* getMapInput(char* tns_dir, int output_resolution) {
    //printf("%s\n", tns_dir);
    int* imgs = (int*)malloc(sizeof(int) * 3 * output_resolution * output_resolution);
    int order, i, j, k;
    FILE* fpt;
    fpt = fopen(tns_dir, "r");
    if (fpt == NULL)
        printf("Error while opening tensor file!\n");
    // get dims
    fscanf(fpt, "%d", &order);
    int* dim = (int*)malloc(sizeof(int) * order);
    fscanf(fpt, "%d %d %d", &dim[0], &dim[1], &dim[2]);
    // density sampling of three images
    int* valdim = (int*)malloc(sizeof(int) * order);
    float* scaledim = (float*)malloc(sizeof(float) * order);
    int* indexdim = (int*)malloc(sizeof(int) * order);
    float value;
    for (i = 0; i < 3; i++)
        scaledim[i] = dim[i] / (output_resolution * 1.0);
    // init imgs
    for (i = 0; i < 3 * output_resolution * output_resolution; i++)
        imgs[i] = 0;
    while (fscanf(fpt, "%d %d %d %f\n", &valdim[0], &valdim[1], &valdim[2], &value) != EOF) {
        for (i = 0; i < 3; i++)
            indexdim[i] = (int)((valdim[i]-1) / scaledim[i]);
        imgs[indexdim[1]*output_resolution+indexdim[2]] += 1;
        imgs[output_resolution*(1*output_resolution+indexdim[0])+indexdim[2]] += 1;
        imgs[output_resolution*(2*output_resolution+indexdim[0])+indexdim[1]] += 1;
    }
    fclose(fpt);

    // free arrays
    free(dim);
    free(valdim);
    free(scaledim);
    free(indexdim);

    return imgs;
}

// the function for calculating csf features
float* getCsfFeatures(char* tns_dir) {
    //printf("%s\n", tns_dir);
    csfFeatureSet ftset;
    FILE* fpt;
    int i;
    fpt = fopen(tns_dir, "r");
    if (fpt == NULL)
        printf("Error while opening tensor file!\n");
    fscanf(fpt, "%d", &ftset.order);
    ftset.dim = (int*)malloc(sizeof(int) * ftset.order);
    // get dims
    fscanf(fpt, "%d %d %d", &ftset.dim[0], &ftset.dim[1], &ftset.dim[2]);
    int* valdim = (int*)malloc(sizeof(int) * ftset.order);
    int* lastdim = (int*)malloc(sizeof(int) * ftset.order);
    float value;
    fscanf(fpt, "%d %d %d %f\n", &lastdim[0], &lastdim[1], &lastdim[2], &value);
    int  nnz = 1, numSlices = 1, numFibers = 1;
    while (fscanf(fpt, "%d %d %d %f\n", &valdim[0], &valdim[1], &valdim[2], &value) != EOF) {
        if (valdim[0] != lastdim[0]) {
            numSlices += 1;
            numFibers += 1;
        }
        else {
            if (valdim[1] != lastdim[1])
                numFibers += 1;
        }
        lastdim[0] = valdim[0];
        lastdim[1] = valdim[1];
        lastdim[2] = valdim[2];   
        nnz += 1;
    }
    fclose(fpt);
    ftset.nnz = nnz;
    ftset.density = ftset.nnz / (ftset.dim[0] * ftset.dim[1] * ftset.dim[2] * 1.0);

    int* fibersPerSlice = (int*)malloc(sizeof(int) * numSlices);
    int* nnzPerSlice = (int*)malloc(sizeof(int) * numSlices);
    int* nnzPerFiber = (int*)malloc(sizeof(int) * numFibers);
    int tmpfibersPerSlice = 1, tmpnnzPerSlice = 1, tmpnnzPerFiber = 1;
    int sliceCounter = 0, fiberCounter = 0;
    fpt = fopen(tns_dir, "r");
    // jump to the third line
    for (i = 0; i < 2; i++)  fscanf(fpt,"%*[^\n]%*c");
    fscanf(fpt, "%d %d %d %f\n", &lastdim[0], &lastdim[1], &lastdim[2], &value);
    while (fscanf(fpt, "%d %d %d %f\n", &valdim[0], &valdim[1], &valdim[2], &value) != EOF) {
        if (valdim[0] == lastdim[0]) {
            tmpnnzPerSlice += 1;
            if (valdim[1] == lastdim[1])
                tmpnnzPerFiber += 1;
            else {
                nnzPerFiber[fiberCounter] = tmpnnzPerFiber;
                tmpnnzPerFiber = 1;
                tmpfibersPerSlice += 1;
                fiberCounter += 1;
            }
        }
        else {
            nnzPerFiber[fiberCounter] = tmpnnzPerFiber;
            fibersPerSlice[sliceCounter] = tmpfibersPerSlice;
            nnzPerSlice[sliceCounter] = tmpnnzPerSlice;
            tmpnnzPerFiber = 1;
            tmpnnzPerSlice = 1;
            tmpfibersPerSlice = 1;
            sliceCounter += 1;
            fiberCounter += 1;
        }
        lastdim[0] = valdim[0];
        lastdim[1] = valdim[1];
        lastdim[2] = valdim[2];
    }
    fclose(fpt);
    fibersPerSlice[sliceCounter] = tmpfibersPerSlice;
    nnzPerSlice[sliceCounter] = tmpnnzPerSlice;
    nnzPerFiber[fiberCounter] = tmpnnzPerFiber;

    // get detailed features
    ftset.numSlices = numSlices;
    ftset.numFibers = numFibers;
    ftset.sliceRatio = ftset.numSlices / (ftset.dim[0] * 1.0);
    ftset.fiberRatio = ftset.numFibers / (ftset.dim[0] * ftset.dim[1] * 1.0);
    ftset.maxNnzPerSlice = getMax(nnzPerSlice, ftset.numSlices);
    ftset.minNnzPerSlice = getMin(nnzPerSlice, ftset.numSlices);
    ftset.devNnzPerSlice = ftset.maxNnzPerSlice - ftset.minNnzPerSlice;
    ftset.aveNnzPerSlice = ftset.nnz / (ftset.numSlices * 1.0);
    ftset.maxFibersPerSlice = getMax(fibersPerSlice, ftset.numSlices);
    ftset.minFibersPerSlice = getMin(fibersPerSlice, ftset.numSlices);
    ftset.devFibersPerSlice = ftset.maxFibersPerSlice - ftset.minFibersPerSlice;
    ftset.aveFibersPerSlice = ftset.numFibers / (ftset.numSlices * 1.0);
    ftset.maxNnzPerFiber = getMax(nnzPerFiber, ftset.numFibers);
    ftset.minNnzPerFiber = getMin(nnzPerFiber, ftset.numFibers);
    ftset.devNnzPerFiber = ftset.maxNnzPerFiber - ftset.minNnzPerFiber;
    ftset.aveNnzPerFiber = ftset.nnz / (ftset.numFibers * 1.0);

    // getadjNnz
    float totaladjFiberPerSlice = 0.0, totaladjNnzPerSlice = 0.0, totaladjNnzPerFiber = 0.0;
    for (i = 1; i < ftset.numSlices; i++) {
        totaladjFiberPerSlice += abs(fibersPerSlice[i] - fibersPerSlice[i-1]);
        totaladjNnzPerSlice += abs(nnzPerSlice[i] - nnzPerSlice[i-1]);
    }
    for (i = 1; i < ftset.numFibers; i++)
        totaladjNnzPerFiber += abs(nnzPerFiber[i] - nnzPerFiber[i-1]);

    if (ftset.numSlices == 1) {
        ftset.adjFibersPerSlice = 0.0;
        ftset.adjNnzPerSlice = 0.0;
    }
    else {
            ftset.adjFibersPerSlice = totaladjFiberPerSlice / (ftset.numSlices - 1);
            ftset.adjNnzPerSlice = totaladjNnzPerSlice / (ftset.numSlices - 1);
    }
    if (ftset.numFibers == 1)  ftset.adjNnzPerFiber = 0.0;
    else
        ftset.adjNnzPerFiber = totaladjNnzPerFiber / (ftset.numFibers - 1);

    // get the features
    float* features = (float*)malloc(sizeof(float) * 24);
    for (i = 0; i < 3; i++) {
        features[i] = ftset.dim[i];
    }
    features[3] = ftset.nnz;
    features[4] = ftset.density;
    features[5] = ftset.numSlices;
    features[6] = ftset.numFibers;
    features[7] = ftset.sliceRatio;
    features[8] = ftset.fiberRatio;
    features[9] = ftset.maxFibersPerSlice;
    features[10] = ftset.minFibersPerSlice;
    features[11] = ftset.aveFibersPerSlice;
    features[12] = ftset.devFibersPerSlice;
    features[13] = ftset.adjFibersPerSlice;
    features[14] = ftset.maxNnzPerSlice;
    features[15] = ftset.minNnzPerSlice;
    features[16] = ftset.aveNnzPerSlice;
    features[17] = ftset.devNnzPerSlice;
    features[18] = ftset.adjNnzPerSlice;
    features[19] = ftset.maxNnzPerFiber;
    features[20] = ftset.minNnzPerFiber;
    features[21] = ftset.aveNnzPerFiber;
    features[22] = ftset.devNnzPerFiber;
    features[23] = ftset.adjNnzPerFiber;

    // free arrays
    free(ftset.dim);
    free(valdim);
    free(lastdim);
    free(fibersPerSlice);
    free(nnzPerSlice);
    free(nnzPerFiber);

    return features;
}

// the function for calculating base features
float* getBaseFeatures(char* tns_dir) {
    //printf("%s\n", tns_dir);
    baseFeatureSet ftset;
    FILE* fpt;
    int i;
    fpt = fopen(tns_dir, "r");
    if (fpt == NULL)
        printf("Error while opening tensor file!\n");
    fscanf(fpt, "%d", &ftset.order);
    ftset.dim = (int*)malloc(sizeof(int) * ftset.order);
    ftset.max_nnz = (float*)malloc(sizeof(float) * ftset.order);
    ftset.min_nnz = (float*)malloc(sizeof(float) * ftset.order);
    ftset.ave_nnz = (float*)malloc(sizeof(float) * ftset.order);
    ftset.dev_nnz = (float*)malloc(sizeof(float) * ftset.order);
    ftset.bounce = (float*)malloc(sizeof(float) * ftset.order);
    
    // get dims
    fscanf(fpt, "%d %d %d", &ftset.dim[0], &ftset.dim[1], &ftset.dim[2]);

    int* dim0_nnz = (int*)malloc(sizeof(int) * ftset.dim[0]);
    int* dim1_nnz = (int*)malloc(sizeof(int) * ftset.dim[1]);
    int* dim2_nnz = (int*)malloc(sizeof(int) * ftset.dim[2]);

    int* valdim = (int*)malloc(sizeof(int) * ftset.order);
    float value = 0.0;
    int  nnz = 0;

    // get total nnz, density, nnz of each dim
    for (i = 0; i < ftset.dim[0]; i++)
        dim0_nnz[i] = 0;
    for (i = 0; i < ftset.dim[1]; i++)
        dim1_nnz[i] = 0;
    for (i = 0; i < ftset.dim[2]; i++)
        dim2_nnz[i] = 0;
    while (fscanf(fpt, "%d %d %d %f\n", &valdim[0], &valdim[1], &valdim[2], &value) != EOF) {
        dim0_nnz[valdim[0]-1] += 1;
        dim1_nnz[valdim[1]-1] += 1;
        dim2_nnz[valdim[2]-1] += 1;
        nnz += 1;
    }
    fclose(fpt);

    ftset.nnz = nnz;
    ftset.density = (nnz * 1.0) / (ftset.dim[0] * ftset.dim[1] * ftset.dim[2]);
    // initialize max_nnz, min_nnz, tot_nnz, etc
    ftset.max_nnz[0]=dim0_nnz[0]; ftset.max_nnz[1]=dim1_nnz[0]; ftset.max_nnz[2]=dim2_nnz[0];
    ftset.min_nnz[0]=dim0_nnz[0]; ftset.min_nnz[1]=dim1_nnz[0]; ftset.min_nnz[2]=dim2_nnz[0];
    ftset.ave_nnz[0]=dim0_nnz[0]; ftset.ave_nnz[1]=dim1_nnz[0]; ftset.ave_nnz[2]=dim2_nnz[0];
    ftset.dev_nnz[0]=dim0_nnz[0]; ftset.dev_nnz[1]=dim1_nnz[0]; ftset.dev_nnz[2]=dim2_nnz[0];
    ftset.bounce[0]=dim0_nnz[0]; ftset.bounce[1]=dim1_nnz[0]; ftset.bounce[2]=dim2_nnz[0];
    
    float* tot_nnz = (float*)malloc(sizeof(float) * ftset.order);
    int* dim_counter = (int*)malloc(sizeof(int) * ftset.order);
    float* adj_nnz = (float*)malloc(sizeof(float) * ftset.order);
    
    tot_nnz[0]=dim0_nnz[0]; tot_nnz[1]=dim1_nnz[0]; tot_nnz[2]=dim2_nnz[0];
    dim_counter[0]=compare(dim0_nnz[0],0); dim_counter[1]=compare(dim1_nnz[0],0); dim_counter[2]=compare(dim2_nnz[0],0);
    adj_nnz[0] = 0.0; adj_nnz[1] = 0.0; adj_nnz[2] = 0.0;
    
    // get the values of above variables
    for (i = 1; i < ftset.dim[0]; i++) {
        if (dim0_nnz[i] > ftset.max_nnz[0])
            ftset.max_nnz[0] = dim0_nnz[i];
        if (dim0_nnz[i] < ftset.min_nnz[0])
            ftset.min_nnz[0] = dim0_nnz[i];
        adj_nnz[0] += abs(dim0_nnz[i] - dim0_nnz[i-1]);
        tot_nnz[0] += dim0_nnz[i];
        dim_counter[0] += compare(dim0_nnz[i], 0);
    }

    for (i = 1; i < ftset.dim[1]; i++) {
        if (dim1_nnz[i] > ftset.max_nnz[1])
            ftset.max_nnz[1] = dim1_nnz[i];
        if (dim1_nnz[i] < ftset.min_nnz[1])
            ftset.min_nnz[1] = dim1_nnz[i];
        adj_nnz[1] += abs(dim1_nnz[i] - dim1_nnz[i-1]);
        tot_nnz[1] += dim1_nnz[i];
        dim_counter[1] += compare(dim1_nnz[i], 0);
    }

    for (i = 1; i < ftset.dim[2]; i++) {
        if (dim2_nnz[i] > ftset.max_nnz[2])
            ftset.max_nnz[2] = dim2_nnz[i];
        if (dim2_nnz[i] < ftset.min_nnz[2])
            ftset.min_nnz[2] = dim2_nnz[i];
        adj_nnz[2] += abs(dim2_nnz[i] - dim2_nnz[i-1]);
        tot_nnz[2] += dim2_nnz[i];
        dim_counter[2] += compare(dim2_nnz[i], 0);
    }

    // get the values of ave_nnz, dev_nnz and bounce
    for (i = 0; i < 3; i++) {
        ftset.dev_nnz[i] = ftset.max_nnz[i] - ftset.min_nnz[i];
        ftset.ave_nnz[i] = tot_nnz[i] / dim_counter[i];
        if (ftset.dim[i] < 2)
            ftset.bounce[i] = 0.0;
        else
            ftset.bounce[i] = adj_nnz[i] / (ftset.dim[i] - 1);
    }
   
    // get the features
    float* features = (float*)malloc(sizeof(float) * (ftset.order * 6 + 2));
    for (i = 0; i < 3; i++) {
        features[i] = ftset.dim[i];
    }
    features[3] = ftset.nnz; features[4] = ftset.density;
    for (i = 0; i < 3; i++) {
        features[i + 5] = ftset.max_nnz[i];
    }
    for (i = 0; i < 3; i++) {
        features[i + 8] = ftset.min_nnz[i];
    }
    for (i = 0; i < 3; i++) {
        features[i + 11] = ftset.dev_nnz[i];
    }
    for (i = 0; i < 3; i++) {
        features[i + 14] = ftset.ave_nnz[i];
    }
    for (i = 0; i < 3; i++) {
        features[i + 17] = ftset.bounce[i];
    }

    // free arrays
    free(ftset.dim);
    free(ftset.max_nnz);
    free(ftset.min_nnz);
    free(ftset.ave_nnz);
    free(ftset.dev_nnz);
    free(ftset.bounce);
    free(dim0_nnz);
    free(dim1_nnz);
    free(dim2_nnz);
    free(valdim);
    free(tot_nnz);
    free(dim_counter);
    free(adj_nnz);

    return features;
}
