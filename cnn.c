#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
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

// // parallel CNN implementation
// void parallel_CONV(float Cout[NUM][IMROW][IMROW], float Cin[NUM][INIMROW][INIMROW],
//           float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
// {
// 	for(int i=0; i<NUM; i++) {
// 		for(int h=0; h<IMROW; h++) {
// 			for(int w=0; w<IMROW; w++)
// 				Cout[i][h][w] = bias[i];
// 		}
// 	}
// 	for(int i=0; i<NUM; i++) {
// 		for(int j=0; j<NUM; j++) {
// 			for(int h=0; h<IMROW; h++) {
// 				for(int w=0; w<IMROW; w++) {
// 					for(int p=0; p<KERNEL; p++) {
// 						for(int q=0; q<KERNEL; q++)
// 							Cout[i][h][w] += weight[i][j][p][q]*Cin[j][1*h+p][1*w+q];
// 					}
// 				}
// 			}
// 		}
// 	}
// }


int main()
{
	static float Cout[NUM][IMROW][IMROW];
	static float Cin[NUM][INIMROW][INIMROW];
	static float weight[NUM][NUM][KERNEL][KERNEL];
	static float bias[NUM];

	LoadData(Cin, weight, bias);

	fprintf(stderr, "Start cnn computation\n");
	long beginTime = clock();
	CONV(Cout, Cin, weight, bias);
	long endTime=clock();
	fprintf(stderr, "time: %f\n", (float)(endTime - beginTime) / (float) CLOCKS_PER_SEC);

	int error = Verify(Cout);
	if(error != 0)
		fprintf(stderr, "error ocurrs %d\n", error);
	else
		fprintf(stderr, "all right!\n");

	return 0;
}

