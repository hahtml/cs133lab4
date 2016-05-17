#define NUM 512
#define IMROW 224
#define INIMROW 226
#define KERNEL 3

__kernel 
void CONV(
	__global float * Cout,
	__global float * Cin,
	__global float * weight,
	__global float * bias) {

  int gid = get_global_id(0);
  int tnum = get_global_size(0);
  int start = gid * (512 / tnum);
  int end = start + (512 / tnum);

  for (int i = start; i < end; i++) {
    for (int h = 0; h < 224; h++) {
      for (int w = 0; w < 224; w++) {
        Cout[i * (224 * 224) + h * 224 + w] = bias[i];
      }
    }
  }

  for (int i = start; i < end; i++) {
    for (int j = 0; j < 512; j++) {
      for (int h = 0; h < 224; h++) {
        for (int w = 0; w < 224; w++) {
          for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
              Cout[i * (224 * 224) + h * 224 + w] += weight[i * (512 * 3 * 3) + j * (3 * 3) + p * 3 + q] * Cin[j * (226 * 226) + (h + p) * 226 + (w + q)];
            }
          }
        }
      }
    }
  }
}
