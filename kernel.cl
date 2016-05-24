#define NUM 512
#define IMROW 224
#define INIMROW 226
#define KERNEL 3

// // Sequential CNN implementation
// void CONV(float Cout[NUM][IMROW][IMROW], float Cin[NUM][INIMROW][INIMROW],
//           float weight[NUM][NUM][KERNEL][KERNEL], float bias[NUM])
// {
//   for(int i=0; i<NUM; i++) {
//     for(int h=0; h<IMROW; h++) {
//       for(int w=0; w<IMROW; w++)
//         Cout[i][h][w] = bias[i];
//     }
//   }
//   for(int i=0; i<NUM; i++) {
//     for(int j=0; j<NUM; j++) {
//       for(int h=0; h<IMROW; h++) {
//         for(int w=0; w<IMROW; w++) {
//           for(int p=0; p<KERNEL; p++) {
//             for(int q=0; q<KERNEL; q++)
//               Cout[i][h][w] += weight[i][j][p][q]*Cin[j][1*h+p][1*w+q];
//           }
//         }
//       }
//     }
//   }
// }

__kernel 
void CONV(
	__global float * Cout,
	__global float * Cin,
	__global float * weight,
	__global float * bias) {

  int gid = get_global_id(0);
  int lid = get_local_id(0);
  // int tnum = get_global_size(0);

  // int start = gid * (512 / tnum);
  // int end = start + (512 / tnum);

  // for (int i = gid; i < end; i++) {
  //   for (int h = 0; h < 224; h++) {
  //     for (int w = 0; w < 224; w++) {
  //       Cout[i * (224 * 224) + h * 224 + w] = bias[i];
  //     }
  //   }
  // }
  int i = gid/32;
  int h = lid;
  int tmp1 = i * (224 * 224) + h * 224;
  for (int w = 0; w < 224; w++) {
    Cout[tmp1 + w] = bias[i];
  }

  // for (int i = start; i < end; i++) {
  //   for (int j = 0; j < 512; j++) {
  //     for (int h = 0; h < 224; h++) {
  //       for (int w = 0; w < 224; w++) {
  //         for (int p = 0; p < 3; p++) {
  //           for (int q = 0; q < 3; q++) {
  //             Cout[i * (224 * 224) + h * 224 + w] += weight[i * (512 * 
  //               3 * 3) + j * (3 * 3) + p * 3 + q] * Cin[j * (226 * 
  //               226) + (h + p) * 226 + (w + q)];
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  int cout_i = i * (224 * 224);
  int weight_i = i * (512 * 3 * 3);
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (int j = lid; j < 512; j+=16) {
    int cin_j = j * (226 * 226);
    for (int h = 0; h < 224; h++) {
      int cout_h = h * 224;
      for (int w = 0; w < 224; w++) {
        for (int p = 0; p < 3; p++) {
          for (int q = 0; q < 3; q++) {
            Cout[cout_i + cout_h + w] += weight[weight_i + 
              j * (3 * 3) + p * 3 + q] * Cin[cin_j + 
              (h + p) * 226 + (w + q)];
          }
        }
      }
    }
  }
}
