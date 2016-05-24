#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "cnn.h"

// OpenCL includes
#include <CL/cl.h>
#include "kernel_cl.h"

// Sequential CNN implementation
void CONV(float Cout[NUM][IMROW][IMROW], float Cin[NUM][INIMROW][INIMROW],
          float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
{
	for(int i=0; i<NUM; i++) {
		for(int h=0; h<IMROW; h++) {
			for(int w=0; w<IMROW; w++)
				Cout[i][h][w] = bias[i];
		}
	}
	for(int i=0; i<NUM; i++) {
		for(int j=0; j<NUM; j++) {
			for(int h=0; h<IMROW; h++) {
				for(int w=0; w<IMROW; w++) {
					for(int p=0; p<KERNEL; p++) {
						for(int q=0; q<KERNEL; q++)
							Cout[i][h][w] += weight[i][j][p][q]*Cin[j][1*h+p][1*w+q];
					}
				}
			}
		}
	}
}

void parallel_CONV(float Cout[NUM][IMROW][IMROW], float Cin[NUM][INIMROW][INIMROW],
          float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
{
	size_t size_float = sizeof(float);


    // Use this to check the output of each API call
    cl_int status;  

    // Retrieve the number of platforms
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);

    // Allocate enough space for each platform
    cl_platform_id *platforms = NULL;
    platforms = (cl_platform_id*)malloc(
        numPlatforms*sizeof(cl_platform_id));

    // Fill in the platforms
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

	// Find GPU
	int platform_index = -1;
	char vendor[128];
	for (int i = 0; i < numPlatforms; i++)
	{
		clGetPlatformInfo (platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
		// char vendorF[7];
		// memcpy((void*)vendorF, (void*)vendor, 6);
		// vendorF[6] = '\0';
		// fprintf(stderr, "%s\n", vendorF);
		if (strstr(vendor, "NVIDIA") != NULL)
		{
			platform_index = i;
			break;
		}
	}
	if (platform_index == -1){
		printf("Didn't find GPU platform!\n");
		exit(1);
	}

	printf("Selected platform '%s' . %d\n", vendor, platform_index);

  	// Retrieve the number of devices
    cl_uint numDevices = 0;
    status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, 0, 
        NULL, &numDevices);

	printf("#devices: %d, status %d\n", numDevices, status);
    // Allocate enough space for each device
    cl_device_id *devices;
    devices = (cl_device_id*)malloc(
        numDevices*sizeof(cl_device_id));

    // Fill in the devices 
    status = clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU,        
        numDevices, devices, NULL);

    for(int i=0; i<numDevices; i++){
    	printf("devices: %d\n", devices[i]);
    }

    // Create a context and associate it with the devices
    cl_context context;
    context = clCreateContext(NULL, numDevices, devices, NULL, 
        NULL, &status);

    // Create a command queue and associate it with the device
    // cl_command_queue cmdQueue;
    // cmdQueue = clCreateCommandQueue(context, devices[0], 0, 
    //     &status);    
    cl_command_queue *cmdQueue;
    cmdQueue = (cl_command_queue*)malloc(numDevices*sizof(cl_command_queue));
    for(int i=0; i<numDevices; i++){
        cmdQueue[i] = clCreateCommandQueue(context, devices[i], 0, &status);
    } 

    // Create a buffer object that will contain the data 
    // from the host array Cin
    cl_mem bufCin;
    int Cin_size = NUM * INIMROW * INIMROW * size_float;
    bufCin = clCreateBuffer(context, CL_MEM_READ_ONLY, Cin_size,
    	NULL, &status);

    // Create a buffer object that will contain the data 
    // from the host array weight
    cl_mem bufWeight;
    int Weight_size = NUM * NUM * KERNEL * KERNEL * size_float;
    bufWeight = clCreateBuffer(context, CL_MEM_READ_ONLY, Weight_size,                        
        NULL, &status);

    // Create a buffer object that will contain the data 
    // from the host array bias
    cl_mem bufBias;
    int Bias_size = NUM * size_float;
    bufBias = clCreateBuffer(context, CL_MEM_READ_ONLY, Bias_size,                        
        NULL, &status);

    // Create a buffer object that will hold the output data Cout
    cl_mem bufCout;
    int Cout_size = NUM * IMROW * IMROW * size_float;
    bufCout = clCreateBuffer(context, CL_MEM_WRITE_ONLY, Cout_size,
        NULL, &status);


	// Write input array Cin to the device buffer bufCin
    for(int i=0; i<NUM; i++){
    	for(int j=0; j<INIMROW; j++){
			status = clEnqueueWriteBuffer(cmdQueue, bufCin, CL_FALSE, 
			    (i*INIMROW*INIMROW+j*INIMROW)*size_float, INIMROW*size_float, &(Cin[i][j][0]), 0, NULL, NULL);    			
    	} 
    }

    // Write input array weight to the device buffer bufWeight
    for(int i=0; i<NUM; i++){
    	for(int j=0; j<NUM; j++){
    		for(int k=0; k<KERNEL; k++){
				status = clEnqueueWriteBuffer(cmdQueue, bufWeight, CL_FALSE, 
			    	(i*NUM*KERNEL*KERNEL+j*KERNEL*KERNEL+k*KERNEL)*size_float, KERNEL*size_float, 
			    	&(weight[i][j][k][0]), 0, NULL, NULL);    			
    		}
    	}
    }

    // Write input array bias to the device buffer bufBias
    status = clEnqueueWriteBuffer(cmdQueue, bufBias, CL_FALSE, 
        0, NUM*size_float, bias, 0, NULL, NULL);



    // Create a program with source code
    cl_program program = clCreateProgramWithSource(context, 1, 
        (const char**)&kernel_cl, NULL, &status);    

    // Build (compile) the program for the device
    status = clBuildProgram(program, numDevices, devices, 
        NULL, NULL, NULL);

    // Create the vector addition kernel
    cl_kernel kernel;
    kernel = clCreateKernel(program, "CONV", &status);

    // Associate the input and output buffers with the kernel
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufCout); 
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufCin);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufWeight);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufBias);

   	// Define an index space (global work size) of work 
    // items for execution. A workgroup size (local work size) 
    // is not required, but can be used.
    size_t globalWorkSize[1];
    size_t localWorkSize[1];

    // There are 32 * 512 work-items
    globalWorkSize[0] = 32 * 512;
    // There are work-groups
    localWorkSize[0] = 512;

    // Execute the kernel for execution
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, 
        globalWorkSize, localWorkSize, 0, NULL, NULL);

    // Read the device output buffer to the host output array
    for(int i=0; i<NUM; i++){
    	for(int j=0; j<IMROW; j++){
    		clEnqueueReadBuffer(cmdQueue, bufCout, CL_TRUE, (i*IMROW*IMROW+j*IMROW)*size_float, 
        		IMROW*size_float, &(Cout[i][j][0]), 0, NULL, NULL); 
    	}
    }
    // clEnqueueReadBuffer(cmdQueue, bufCout, CL_TRUE, 0, NUM*IMROW*IMROW, Cout, 0, NULL, NULL);

    // Free OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufCout);
    clReleaseMemObject(bufCin);
    clReleaseMemObject(bufWeight);
    clReleaseMemObject(bufBias);
    clReleaseContext(context);

    // Free host resources
    free(platforms);
    free(devices);
    free(cmdQueue);

    return;
}


int main()
{
	static float Cout[NUM][IMROW][IMROW];
	static float Cin[NUM][INIMROW][INIMROW];
	static float weight[NUM][NUM][KERNEL][KERNEL];
	static float bias[NUM];
	struct timeval begin_time;
	struct timeval end_time;

	LoadData(Cin, weight, bias);

	fprintf(stderr, "Start cnn computation\n");
	// long beginTime = clock();
	gettimeofday(&begin_time, NULL);
	parallel_CONV(Cout, Cin, weight, bias);
	// CONV(Cout, Cin, weight, bias);
	// long endTime=clock();
	gettimeofday(&end_time, NULL);
	// fprintf(stderr, "time: %f\n", (float)(endTime - beginTime) / (float) CLOCKS_PER_SEC);
	fprintf(stderr, "time: %f\n", (float)(end_time.tv_sec - begin_time.tv_sec)+ (end_time.tv_usec - begin_time.tv_usec)/1000000.0);
	int error = Verify(Cout);
	if(error != 0)
		fprintf(stderr, "error ocurrs %d\n", error);
	else
		fprintf(stderr, "all right!\n");

	return 0;
}

