/*
3d elastic wave propagation in isotropic medium
Coalesced array computation
*/

#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <assert.h>
#include "conio.h"
#include <iomanip>

using namespace std;

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void kersigmaxyz(int *ISLBEGIN, int *JSLBEGIN, int *KSLBEGIN, float *cp, float *cs, float *rho, float *DELTAT, int *DDIMX, int *DDIMY, int *DDIMZ, float *memory_dvx_dx, float *memory_dvy_dy, float *memory_dvz_dz, float *a_x_half, float *a_y, float *a_z, float *b_x_half, float *b_y, float *b_z, float *K_x_half, float *K_y, float *K_z, float *sigmaxx, float *sigmayy, float *sigmazz, float *ONE_OVER_DELTAX, float *ONE_OVER_DELTAY, float *ONE_OVER_DELTAZ, float *vx, float *vy, float *vz) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = index_x + index_y*DDIMX[0] + index_z*DDIMX[0] * DDIMZ[0];
	int right = offset + 1;
	int ybottom = offset - DDIMX[0];
	int zbottom = offset - DDIMX[0] * DDIMY[0];
	int iglobal = index_x + ISLBEGIN[0];
	int jglobal = index_y + JSLBEGIN[0];
	int kglobal = index_z + KSLBEGIN[0];


	if ((index_z >= 2) && (index_z <= DDIMZ[0])) {
		if ((index_y >= 2) && (index_y <= DDIMY[0])) {
			if ((index_x >= 1) && (index_z <= DDIMX[0] - 1)) {
				float vp = cp[offset];
				float vs = cs[offset];
				float rhos = rho[offset];

				float lambda = rhos*(vp*vp - 2 * vs*vs);
				float lambdaplus2mu = rhos*vp*vp;

				float DELTAT_lambdaplus2mu = DELTAT[0] * lambdaplus2mu;
				float DELTAT_lambda = DELTAT[0] * lambda;

				float value_dvx_dx = (vx[right] - vx[offset])*ONE_OVER_DELTAX[0];
				float value_dvy_dy = (vy[offset] - vy[ybottom])*ONE_OVER_DELTAY[0];
				float value_dvz_dz = (vz[offset] - vz[zbottom])*ONE_OVER_DELTAZ[0];

				memory_dvx_dx[offset] = b_x_half[iglobal] * memory_dvx_dx[offset] + a_x_half[iglobal] * value_dvx_dx;
				memory_dvy_dy[offset] = b_y[jglobal] * memory_dvy_dy[offset] + a_y[jglobal] * value_dvy_dy;
				memory_dvz_dz[offset] = b_z[kglobal] * memory_dvz_dz[offset] + a_z[kglobal] * value_dvz_dz;

				value_dvx_dx = value_dvx_dx / K_x_half[iglobal] + memory_dvx_dx[offset];
				value_dvy_dy = value_dvy_dy / K_y[jglobal] + memory_dvy_dy[offset];
				value_dvz_dz = value_dvz_dz / K_z[kglobal] + memory_dvz_dz[offset];

				sigmaxx[offset] = DELTAT_lambdaplus2mu * value_dvx_dx + DELTAT_lambda * (value_dvy_dy + value_dvz_dz) + sigmaxx[offset];
				sigmayy[offset] = DELTAT_lambda * (value_dvx_dx + value_dvz_dz) + DELTAT_lambdaplus2mu * value_dvy_dy + sigmayy[offset];
				sigmazz[offset] = DELTAT_lambda * (value_dvx_dx + value_dvy_dy) + DELTAT_lambdaplus2mu * value_dvz_dz + sigmazz[offset];
			}
		}
	}

}

__global__ void kersigmaxy(int *ISLBEGIN, int *JSLBEGIN, int *KSLBEGIN, float *cp, float *cs, float *rho, int *DDIMX, int *DDIMY, int *DDIMZ, float *DELTAT, float *memory_dvy_dx, float *memory_dvx_dy, float *a_x, float *a_y_half, float *b_x, float *b_y_half, float *K_x, float *K_y_half, float *ONE_OVER_DELTAX, float *ONE_OVER_DELTAY, float *vx, float *vy, float *sigmaxy) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = index_x + index_y*DDIMX[0] + index_z*DDIMX[0] * DDIMZ[0];
	int left = offset - 1;
	int ytop = offset + DDIMX[0];
	int iglobal = index_x + ISLBEGIN[0];
	int jglobal = index_y + JSLBEGIN[0];
	int kglobal = index_z + KSLBEGIN[0];


	if ((index_z >= 1) && (index_z <= DDIMZ[0])) {
		if ((index_y >= 1) && (index_y <= DDIMY[0] - 1)) {
			if ((index_x >= 2) && (index_z <= DDIMX[0])) {
				float vs = (cs[left] + cs[ytop]) / 2;
				float rhos = (rho[left] + rho[ytop]) / 2;

				float mu = rhos*vs*vs;

				float DELTAT_mu = DELTAT[0] * mu;

				float value_dvy_dx = (vy[offset] - vy[left])*ONE_OVER_DELTAX[0];
				float value_dvx_dy = (vx[ytop] - vx[offset])*ONE_OVER_DELTAY[0];

				memory_dvy_dx[offset] = b_x[iglobal] * memory_dvy_dx[offset] + a_x[iglobal] * value_dvy_dx;
				memory_dvx_dy[offset] = b_y_half[jglobal] * memory_dvx_dy[offset] + a_y_half[jglobal] * value_dvx_dy;

				value_dvy_dx = value_dvy_dx / K_x[iglobal] + memory_dvy_dx[offset];
				value_dvx_dy = value_dvx_dy / K_y_half[jglobal] + memory_dvx_dy[offset];

				sigmaxy[offset] = DELTAT_mu * (value_dvy_dx + value_dvx_dy) + sigmaxy[offset];
			}
		}
	}

}

__global__ void kersigmaxzyz(int *ISLBEGIN, int *JSLBEGIN, int *KSLBEGIN, float *cp, float *cs, float *rho, int *DDIMX, int *DDIMY, int *DDIMZ, float *DELTAT, float *memory_dvz_dx, float *memory_dvx_dz, float *memory_dvz_dy, float *memory_dvy_dz, float *a_x, float *a_z, float *a_y_half, float *a_z_half, float *b_x, float *b_y_half, float *b_z_half, float *K_x, float *K_y_half, float *K_z_half, float *ONE_OVER_DELTAX, float *ONE_OVER_DELTAY, float *ONE_OVER_DELTAZ, float *vx, float *vy, float *vz, float *sigmaxz, float *sigmayz) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = index_x + index_y*DDIMX[0] + index_z*DDIMX[0] * DDIMZ[0];
	int left = offset - 1;
	int ztop = offset + DDIMX[0] * DDIMY[0];
	int ytop = offset + DDIMX[0];
	int iglobal = index_x + ISLBEGIN[0];
	int jglobal = index_y + JSLBEGIN[0];
	int kglobal = index_z + KSLBEGIN[0];


	if ((index_z >= 1) && (index_z <= DDIMZ[0])) {
		//sigmaxz
		if ((index_y >= 1) && (index_y <= DDIMY[0])) {
			if ((index_x >= 2) && (index_z <= DDIMX[0])) {
				float vs = (cs[left] + cs[ztop]) / 2;
				float rhos = (rho[left] + rho[ztop]) / 2;

				float mu = rhos*vs*vs;

				float DELTAT_mu = DELTAT[0] * mu;

				float value_dvz_dx = (vz[offset] - vz[left]) * ONE_OVER_DELTAX[0];
				float value_dvx_dz = (vx[ztop] - vx[offset]) * ONE_OVER_DELTAZ[0];

				memory_dvz_dx[offset] = b_x[iglobal] * memory_dvz_dx[offset] + a_x[iglobal] * value_dvz_dx;
				memory_dvx_dz[offset] = b_z_half[kglobal] * memory_dvx_dz[offset] + a_z_half[kglobal] * value_dvx_dz;

				value_dvz_dx = value_dvz_dx / K_x[iglobal] + memory_dvz_dx[offset];
				value_dvx_dz = value_dvx_dz / K_z_half[kglobal] + memory_dvx_dz[offset];

				sigmaxz[offset] = DELTAT_mu * (value_dvz_dx + value_dvx_dz) + sigmaxz[offset];
			}
		}

		//sigmayz
		if ((index_y >= 1) && (index_y <= DDIMY[0] - 1)) {
			if ((index_x >= 1) && (index_z <= DDIMX[0])) {
				float vs = (cs[ytop] + cs[ztop]) / 2;
				float rhos = (rho[ytop] + rho[ztop]) / 2;

				float mu = rhos*vs*vs;

				float DELTAT_mu = DELTAT[0] * mu;

				float value_dvz_dy = (vz[ytop] - vz[offset]) * ONE_OVER_DELTAY[0];
				float value_dvy_dz = (vy[ztop] - vy[offset]) * ONE_OVER_DELTAZ[0];

				memory_dvz_dy[offset] = b_y_half[jglobal] * memory_dvz_dy[offset] + a_y_half[jglobal] * value_dvz_dy;
				memory_dvy_dz[offset] = b_z_half[kglobal] * memory_dvy_dz[offset] + a_z_half[kglobal] * value_dvy_dz;

				value_dvz_dy = value_dvz_dy / K_y_half[jglobal] + memory_dvz_dy[offset];
				value_dvy_dz = value_dvy_dz / K_z_half[kglobal] + memory_dvy_dz[offset];

				sigmayz[offset] = DELTAT_mu * (value_dvz_dy + value_dvy_dz) + sigmayz[offset];
			}
		}
	}

}

__global__ void kervxvy(int *ISLBEGIN, int *JSLBEGIN, int *KSLBEGIN, float *rho, int *DDIMX, int *DDIMY, int *DDIMZ, float *DELTAT, float *sigmaxx, float *sigmaxy, float *sigmaxz, float *sigmayy, float *sigmayz, float *memory_dsigmaxx_dx, float *memory_dsigmaxy_dy, float *memory_dsigmaxz_dz, float *memory_dsigmaxy_dx, float *memory_dsigmayy_dy, float *memory_dsigmayz_dz, float *a_x, float *a_y, float *a_z, float *a_x_half, float *a_y_half, float *b_x, float *b_y, float *b_z, float *b_x_half, float *b_y_half, float *K_x, float *K_y, float *K_z, float *K_x_half, float *K_y_half, float *ONE_OVER_DELTAX, float *ONE_OVER_DELTAY, float *ONE_OVER_DELTAZ, float *vx, float *vy) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = index_x + index_y*DDIMX[0] + index_z*DDIMX[0] * DDIMZ[0];
	int left = offset - 1;
	int ybottom = offset - DDIMX[0];
	int zbottom = offset - DDIMX[0] * DDIMY[0];
	int right = offset + 1;
	int ytop = offset + DDIMX[0];
	int iglobal = index_x + ISLBEGIN[0];
	int jglobal = index_y + JSLBEGIN[0];
	int kglobal = index_z + KSLBEGIN[0];


	if ((index_z >= 2) && (index_z <= DDIMZ[0])) {
		//vx
		if ((index_y >= 2) && (index_y <= DDIMY[0])) {
			if ((index_x >= 2) && (index_z <= DDIMX[0])) {
				float rhos = (rho[offset] + rho[left]) / 2;

				float DELTAT_over_rho = DELTAT[0] / rhos;

				float value_dsigmaxx_dx = (sigmaxx[offset] - sigmaxx[left]) * ONE_OVER_DELTAX[0];
				float value_dsigmaxy_dy = (sigmaxy[offset] - sigmaxy[ybottom]) * ONE_OVER_DELTAY[0];
				float value_dsigmaxz_dz = (sigmaxz[offset] - sigmaxz[zbottom]) * ONE_OVER_DELTAZ[0];

				memory_dsigmaxx_dx[offset] = b_x[iglobal] * memory_dsigmaxx_dx[offset] + a_x[iglobal] * value_dsigmaxx_dx;
				memory_dsigmaxy_dy[offset] = b_y[jglobal] * memory_dsigmaxy_dy[offset] + a_y[jglobal] * value_dsigmaxy_dy;
				memory_dsigmaxz_dz[offset] = b_z[kglobal] * memory_dsigmaxz_dz[offset] + a_z[kglobal] * value_dsigmaxz_dz;

				value_dsigmaxx_dx = value_dsigmaxx_dx / K_x[iglobal] + memory_dsigmaxx_dx[offset];
				value_dsigmaxy_dy = value_dsigmaxy_dy / K_y[jglobal] + memory_dsigmaxy_dy[offset];
				value_dsigmaxz_dz = value_dsigmaxz_dz / K_z[kglobal] + memory_dsigmaxz_dz[offset];

				vx[offset] = DELTAT_over_rho * (value_dsigmaxx_dx + value_dsigmaxy_dy + value_dsigmaxz_dz) + vx[offset];
			}
		}

		//vy
		if ((index_y >= 1) && (index_y <= DDIMY[0] - 1)) {
			if ((index_x >= 1) && (index_z <= DDIMX[0] - 1)) {
				float rhos = (rho[offset] + rho[ytop]) / 2;

				float DELTAT_over_rho = DELTAT[0] / rhos;

				float value_dsigmaxy_dx = (sigmaxy[right] - sigmaxy[offset]) * ONE_OVER_DELTAX[0];
				float value_dsigmayy_dy = (sigmayy[ytop] - sigmayy[offset]) * ONE_OVER_DELTAY[0];
				float value_dsigmayz_dz = (sigmayz[offset] - sigmayz[zbottom]) * ONE_OVER_DELTAZ[0];

				memory_dsigmaxy_dx[offset] = b_x_half[iglobal] * memory_dsigmaxy_dx[offset] + a_x_half[iglobal] * value_dsigmaxy_dx;
				memory_dsigmayy_dy[offset] = b_y_half[jglobal] * memory_dsigmayy_dy[offset] + a_y_half[jglobal] * value_dsigmayy_dy;
				memory_dsigmayz_dz[offset] = b_z[kglobal] * memory_dsigmayz_dz[offset] + a_z[kglobal] * value_dsigmayz_dz;

				value_dsigmaxy_dx = value_dsigmaxy_dx / K_x_half[iglobal] + memory_dsigmaxy_dx[offset];
				value_dsigmayy_dy = value_dsigmayy_dy / K_y_half[jglobal] + memory_dsigmayy_dy[offset];
				value_dsigmayz_dz = value_dsigmayz_dz / K_z[kglobal] + memory_dsigmayz_dz[offset];

				vy[offset] = DELTAT_over_rho * (value_dsigmaxy_dx + value_dsigmayy_dy + value_dsigmayz_dz) + vy[offset];
			}
		}
	}

}

__global__ void kervz(int *ISLBEGIN, int *JSLBEGIN, int *KSLBEGIN, float *rho, int *DDIMX, int *DDIMY, int *DDIMZ, float *DELTAT, float *sigmaxz, float *sigmayz, float *sigmazz, float *memory_dsigmaxz_dx, float *memory_dsigmayz_dy, float *memory_dsigmazz_dz, float *b_x_half, float *b_y, float *b_z_half, float *a_x_half, float *a_y, float *a_z_half, float *K_x_half, float *K_y, float *K_z_half, float *ONE_OVER_DELTAX, float *ONE_OVER_DELTAY, float *ONE_OVER_DELTAZ, float *vz) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = index_x + index_y*DDIMX[0] + index_z*DDIMX[0] * DDIMZ[0];
	int right = offset + 1;
	int ybottom = offset - DDIMX[0];
	int ztop = offset + DDIMX[0] * DDIMY[0];
	int iglobal = index_x + ISLBEGIN[0];
	int jglobal = index_y + JSLBEGIN[0];
	int kglobal = index_z + KSLBEGIN[0];


	if ((index_z >= 1) && (index_z <= DDIMZ[0] - 1)) {
		if ((index_y >= 2) && (index_y <= DDIMY[0])) {
			if ((index_x >= 1) && (index_z <= DDIMX[0] - 1)) {
				float rhos = (rho[offset] + rho[ztop]) / 2;

				float DELTAT_over_rho = DELTAT[0] / rhos;

				float value_dsigmaxz_dx = (sigmaxz[right] - sigmaxz[offset]) * ONE_OVER_DELTAX[0];
				float value_dsigmayz_dy = (sigmayz[offset] - sigmayz[ybottom]) * ONE_OVER_DELTAY[0];
				float value_dsigmazz_dz = (sigmazz[ztop] - sigmazz[offset]) * ONE_OVER_DELTAZ[0];

				memory_dsigmaxz_dx[offset] = b_x_half[iglobal] * memory_dsigmaxz_dx[offset] + a_x_half[iglobal] * value_dsigmaxz_dx;
				memory_dsigmayz_dy[offset] = b_y[jglobal] * memory_dsigmayz_dy[offset] + a_y[jglobal] * value_dsigmayz_dy;
				memory_dsigmazz_dz[offset] = b_z_half[kglobal] * memory_dsigmazz_dz[offset] + a_z_half[kglobal] * value_dsigmazz_dz;

				value_dsigmaxz_dx = value_dsigmaxz_dx / K_x_half[iglobal] + memory_dsigmaxz_dx[offset];
				value_dsigmayz_dy = value_dsigmayz_dy / K_y[jglobal] + memory_dsigmayz_dy[offset];
				value_dsigmazz_dz = value_dsigmazz_dz / K_z_half[kglobal] + memory_dsigmazz_dz[offset];

				vz[offset] = DELTAT_over_rho * (value_dsigmaxz_dx + value_dsigmayz_dy + value_dsigmazz_dz) + vz[offset];
			}
		}
	}

}

int main(void) {
	int DIMGLOBX = 1400;
	int DIMGLOBY = 1400;
	int DIMGLOBZ = 200;
	int *DDIMGLOBX, *DDIMGLOBY, *DDIMGLOBZ;
	HANDLE_ERROR(cudaMalloc((void**)&DDIMGLOBX, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&DDIMGLOBY, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&DDIMGLOBZ, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(DDIMGLOBX, &DIMGLOBX, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DDIMGLOBY, &DIMGLOBY, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DDIMGLOBZ, &DIMGLOBZ, sizeof(int), cudaMemcpyHostToDevice));

	int DIMX = 600;
	int DIMY = 600;
	int DIMZ = 100;

	int offsetperslice = 2; //stagered grid 2nd order space = 2;

	int NSTEP = 2;
	float DELTATT = 1e-3;
	int sampgat = 2; //tsamp = sampgat*Deltat
	int IT_OUTPUT = 200;

	int DELTAX, DELTAY, DELTAZ;
	DELTAX = 10; DELTAY = DELTAX; DELTAZ = DELTAX;
	float ONE_OVER_DELTAXX, ONE_OVER_DELTAYY, ONE_OVER_DELTAZZ;
	ONE_OVER_DELTAXX = 1 / float(DELTAX);
	ONE_OVER_DELTAZZ = ONE_OVER_DELTAXX; ONE_OVER_DELTAYY = ONE_OVER_DELTAXX;

	float *ONE_OVER_DELTAX, *ONE_OVER_DELTAY, *ONE_OVER_DELTAZ;
	HANDLE_ERROR(cudaMalloc((void**)&ONE_OVER_DELTAX, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&ONE_OVER_DELTAY, sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&ONE_OVER_DELTAZ, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(ONE_OVER_DELTAX, &ONE_OVER_DELTAXX, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ONE_OVER_DELTAY, &ONE_OVER_DELTAYY, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(ONE_OVER_DELTAZ, &ONE_OVER_DELTAZZ, sizeof(float), cudaMemcpyHostToDevice));

	float *tempcp = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempcs = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *temprho = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	for (int k = 0; k < (DIMGLOBZ + 1); k++) {
		for (int j = 0; j < (DIMGLOBY + 1); j++) {
			for (int i = 0; i < (DIMGLOBX + 1); i++) {
				int ijk = i + j*DIMGLOBX + k*DIMGLOBX*DIMGLOBY;
				tempcp[ijk] = 3300;
				tempcs[ijk] = 3300 / 1.732;
				temprho[ijk] = 3000;
				if (k >= 50) {
					tempcp[ijk] = 2000;
					tempcs[ijk] = 2000 / 1.732;
					temprho[ijk] = 1700;
				}
			}
		}
	}
	float *cp, *cs, *rho;
	HANDLE_ERROR(cudaMalloc((void**)&cp, ((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1))*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&cs, ((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1))*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&rho, ((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1))*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(cp, tempcp, sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(cs, tempcs, sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(rho, temprho, sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)), cudaMemcpyHostToDevice));
	free(tempcp); free(tempcs); free(temprho);
	
	float *DELTAT;
	HANDLE_ERROR(cudaMalloc((void**)&DELTAT, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(DELTAT, &DELTATT, sizeof(float), cudaMemcpyHostToDevice));

	float f0, tt0, factorr;
	f0 = 35;
	tt0 = 1.2 / f0;
	float *ff0, *t0;
	HANDLE_ERROR(cudaMalloc((void**)&t0, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(t0, &tt0, sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&ff0, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(ff0, &f0, sizeof(float), cudaMemcpyHostToDevice));
	factorr = 1e+7;
	float *factor;
	HANDLE_ERROR(cudaMalloc((void**)&factor, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(factor, &factorr, sizeof(float), cudaMemcpyHostToDevice));
	int NPOINTS_PML = 10;
	int *DPML;
	HANDLE_ERROR(cudaMalloc((void**)&DPML, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(DPML, &NPOINTS_PML, sizeof(int), cudaMemcpyHostToDevice));

	int ISOURCEE, KSOURCEE, JSOURCEE;
	ISOURCEE = DIMGLOBX / 2;
	JSOURCEE = DIMGLOBY / 2;
	KSOURCEE = DIMGLOBZ / 2;
	int *ISOURCE, *KSOURCE, *JSOURCE;
	HANDLE_ERROR(cudaMalloc((void**)&ISOURCE, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(ISOURCE, &ISOURCEE, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&JSOURCE, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(JSOURCE, &JSOURCEE, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&KSOURCE, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(KSOURCE, &KSOURCEE, sizeof(int), cudaMemcpyHostToDevice));

	float ANGLE_FORCEE = 0;
	float *ANGLE_FORCE;
	HANDLE_ERROR(cudaMalloc((void**)&ANGLE_FORCE, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(ANGLE_FORCE, &ANGLE_FORCEE, sizeof(float), cudaMemcpyHostToDevice));

	float PI = 3.141592653589793238462643;
	float *DPI;
	HANDLE_ERROR(cudaMalloc((void**)&DPI, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(DPI, &PI, sizeof(float), cudaMemcpyHostToDevice));
	float DEGREES_TO_RADIANSS = PI / 180;
	float *DEGREES_TO_RADIANS;
	HANDLE_ERROR(cudaMalloc((void**)&DEGREES_TO_RADIANS, sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(DEGREES_TO_RADIANS, &DEGREES_TO_RADIANSS, sizeof(float), cudaMemcpyHostToDevice));

	float NPOWER = 2;
	float K_MAX_PML = 1;
	float ALPHA_MAX_PML = 2 * PI*(f0 / 2);

	float *d_x, *K_x, *alpha_x, *a_x, *b_x, *d_x_half, *K_x_half, *alpha_x_half, *a_x_half, *b_x_half;
	HANDLE_ERROR(cudaMalloc((void**)&d_x, (DIMGLOBX + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_x, (DIMGLOBX + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_x, (DIMGLOBX + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_x, (DIMGLOBX + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_x, (DIMGLOBX + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_x_half, (DIMGLOBX + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_x_half, (DIMGLOBX + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_x_half, (DIMGLOBX + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_x_half, (DIMGLOBX + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_x_half, (DIMGLOBX + 1)*sizeof(float)));

	float *d_y, *K_y, *alpha_y, *a_y, *b_y, *d_y_half, *K_y_half, *alpha_y_half, *a_y_half, *b_y_half;
	HANDLE_ERROR(cudaMalloc((void**)&d_y, (DIMGLOBY + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_y, (DIMGLOBY + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_y, (DIMGLOBY + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_y, (DIMGLOBY + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_y, (DIMGLOBY + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_y_half, (DIMGLOBY + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_y_half, (DIMGLOBY + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_y_half, (DIMGLOBY + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_y_half, (DIMGLOBY + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_y_half, (DIMGLOBY + 1)*sizeof(float)));

	float *d_z, *K_z, *alpha_z, *a_z, *b_z, *d_z_half, *K_z_half, *alpha_z_half, *a_z_half, *b_z_half;
	HANDLE_ERROR(cudaMalloc((void**)&d_z, (DIMGLOBZ + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_z, (DIMGLOBZ + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_z, (DIMGLOBZ + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_z, (DIMGLOBZ + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_z, (DIMGLOBZ + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&d_z_half, (DIMGLOBZ + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&K_z_half, (DIMGLOBZ + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&alpha_z_half, (DIMGLOBZ + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&a_z_half, (DIMGLOBZ + 1)*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&b_z_half, (DIMGLOBZ + 1)*sizeof(float)));

	float thickness_PML_x, thickness_PML_y, thickness_PML_z;
	float xoriginleft, xoriginright, yoriginbottom, yorigintop, zoriginbottom, zorigintop;
	float Rcoef, d0_x, d0_y, d0_z, xval, yval, zval, abscissa_in_PML, abscissa_normalized;

	float Courant_number;

	thickness_PML_x = NPOINTS_PML * DELTAX;
	thickness_PML_y = NPOINTS_PML * DELTAY;
	thickness_PML_z = NPOINTS_PML * DELTAZ;
	Rcoef = 0.001;

	float vpml = 3000;
	d0_x = -(NPOWER + 1) * vpml * logf(Rcoef) / (2.0 * thickness_PML_x);
	d0_y = -(NPOWER + 1) * vpml * logf(Rcoef) / (2.0 * thickness_PML_y);
	d0_z = -(NPOWER + 1) * vpml * logf(Rcoef) / (2.0 * thickness_PML_z);

	//------------------PML X
	float *tempd_x = (float*)malloc(sizeof(float)*(DIMGLOBX + 1));
	float *tempd_x_half = (float*)malloc(sizeof(float)*(DIMGLOBX + 1));
	float *tempa_x = (float*)malloc(sizeof(float)*(DIMGLOBX + 1));
	float *tempa_x_half = (float*)malloc(sizeof(float)*(DIMGLOBX + 1));
	float *tempb_x = (float*)malloc(sizeof(float)*(DIMGLOBX + 1));
	float *tempb_x_half = (float*)malloc(sizeof(float)*(DIMGLOBX + 1));
	float *tempK_x = (float*)malloc(sizeof(float)*(DIMGLOBX + 1));
	float *tempK_x_half = (float*)malloc(sizeof(float)*(DIMGLOBX + 1));
	float *tempalpha_x = (float*)malloc(sizeof(float)*(DIMGLOBX + 1));
	float *tempalpha_x_half = (float*)malloc(sizeof(float)*(DIMGLOBX + 1));

	for (int i = 1; i <= DIMGLOBX; i++) {
		tempd_x[i] = 0.0;
		tempd_x_half[i] = 0.0;
		tempK_x[i] = 1.0;
		tempK_x_half[i] = 1.0;
		tempalpha_x[i] = 0.0;
		tempalpha_x_half[i] = 0.0;
		tempa_x[i] = 0.0;
		tempa_x_half[i] = 0.0;
		tempb_x[i] = 0.0;
		tempb_x_half[i] = 0.0;
	}

	xoriginleft = thickness_PML_x;
	xoriginright = (DIMGLOBX - 1)*DELTAX - thickness_PML_x;
	for (int i = 1; i <= DIMGLOBX; i++) {
		xval = DELTAX*float(i - 1);
		abscissa_in_PML = xoriginleft - xval;//PML XMIN
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			tempd_x[i] = d0_x*powf(abscissa_normalized, NPOWER);
			tempK_x[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_x[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = xoriginleft - (xval + DELTAX / 2.0);
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			tempd_x_half[i] = d0_x*powf(abscissa_normalized, NPOWER);
			tempK_x_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_x_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = xval - xoriginright;//PML XMAX
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			tempd_x[i] = d0_x*powf(abscissa_normalized, NPOWER);
			tempK_x[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_x[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = xval + DELTAX / 2.0 - xoriginright;
		if (abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			tempd_x_half[i] = d0_x*powf(abscissa_normalized, NPOWER);
			tempK_x_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_x_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}
		if (tempalpha_x[i] < 0.0) { tempalpha_x[i] = 0.0; }
		if (tempalpha_x_half[i] < 0.0) { tempalpha_x_half[i] = 0.0; }
		tempb_x[i] = expf(-(tempd_x[i] / tempK_x[i] + tempalpha_x[i])*DELTATT);
		tempb_x_half[i] = expf(-(tempd_x_half[i] / tempK_x_half[i] + tempalpha_x_half[i])*DELTATT);

		if (fabs(tempd_x[i]) > 1e-6) { tempa_x[i] = tempd_x[i] * (tempb_x[i] - 1.0) / (tempK_x[i] * (tempd_x[i] + tempK_x[i] * tempalpha_x[i])); }
		if (fabs(tempd_x_half[i]) > 1e-6) { tempa_x_half[i] = tempd_x_half[i] * (tempb_x_half[i] - 1.0) / (tempK_x_half[i] * (tempd_x_half[i] + tempK_x_half[i] * tempalpha_x_half[i])); }
	}

	HANDLE_ERROR(cudaMemcpy(d_x, tempd_x, sizeof(float)*(DIMGLOBX + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_x_half, tempd_x_half, sizeof(float)*(DIMGLOBX + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_x, tempa_x, sizeof(float)*(DIMGLOBX + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_x_half, tempa_x_half, sizeof(float)*(DIMGLOBX + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_x, tempalpha_x, sizeof(float)*(DIMGLOBX + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_x_half, tempalpha_x_half, sizeof(float)*(DIMGLOBX + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_x, tempb_x, sizeof(float)*(DIMGLOBX + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_x_half, tempb_x_half, sizeof(float)*(DIMGLOBX + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_x, tempK_x, sizeof(float)*(DIMGLOBX + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_x_half, tempK_x_half, sizeof(float)*(DIMGLOBX + 1), cudaMemcpyHostToDevice));

	//-----------------PML Y
	float *tempd_y = (float*)malloc(sizeof(float)*(DIMGLOBY + 1));
	float *tempd_y_half = (float*)malloc(sizeof(float)*(DIMGLOBY + 1));
	float *tempa_y = (float*)malloc(sizeof(float)*(DIMGLOBY + 1));
	float *tempa_y_half = (float*)malloc(sizeof(float)*(DIMGLOBY + 1));
	float *tempb_y = (float*)malloc(sizeof(float)*(DIMGLOBY + 1));
	float *tempb_y_half = (float*)malloc(sizeof(float)*(DIMGLOBY + 1));
	float *tempK_y = (float*)malloc(sizeof(float)*(DIMGLOBY + 1));
	float *tempK_y_half = (float*)malloc(sizeof(float)*(DIMGLOBY + 1));
	float *tempalpha_y = (float*)malloc(sizeof(float)*(DIMGLOBY + 1));
	float *tempalpha_y_half = (float*)malloc(sizeof(float)*(DIMGLOBY + 1));

	for (int i = 1; i < (DIMGLOBY + 1); i++) {
		tempd_y[i] = 0.0;
		tempd_y_half[i] = 0.0;
		tempK_y[i] = 1.0;
		tempK_y_half[i] = 1.0;
		tempalpha_y[i] = 0.0;
		tempalpha_y_half[i] = 0.0;
		tempa_y[i] = 0.0;
		tempa_y_half[i] = 0.0;
		tempb_y[i] = 0.0;
		tempb_y_half[i] = 0.0;
	}

	yoriginbottom = thickness_PML_y;
	yorigintop = (DIMGLOBY - 1)*DELTAY - thickness_PML_y;
	for (int i = 1; i <= DIMGLOBY; i++) {
		yval = DELTAY*float(i - 1);
		abscissa_in_PML = yoriginbottom - yval;//PML YMIN
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_y[i] = d0_y*powf(abscissa_normalized, NPOWER);
			tempK_y[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_y[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = yoriginbottom - (yval + DELTAY / 2.0);
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_y_half[i] = d0_y*powf(abscissa_normalized, NPOWER);
			tempK_y_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_y_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = yval - yorigintop;//PML YMAX
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_y[i] = d0_y*powf(abscissa_normalized, NPOWER);
			tempK_y[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_y[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = yval + DELTAY / 2.0 - yorigintop;
		if (abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_y_half[i] = d0_y*powf(abscissa_normalized, NPOWER);
			tempK_y_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_y_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		tempb_y[i] = expf(-(tempd_y[i] / tempK_y[i] + tempalpha_y[i])*DELTATT);
		tempb_y_half[i] = expf(-(tempd_y_half[i] / tempK_y_half[i] + tempalpha_y_half[i])*DELTATT);

		if (fabs(tempd_y[i]) > 1e-6) { tempa_y[i] = tempd_y[i] * (tempb_y[i] - 1.0) / (tempK_y[i] * (tempd_y[i] + tempK_y[i] * tempalpha_y[i])); }
		if (fabs(tempd_y_half[i]) > 1e-6) { tempa_y_half[i] = tempd_y_half[i] * (tempb_y_half[i] - 1.0) / (tempK_y_half[i] * (tempd_y_half[i] + tempK_y_half[i] * tempalpha_y_half[i])); }
	}

	HANDLE_ERROR(cudaMemcpy(d_y, tempd_y, sizeof(float)*(DIMGLOBY + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_y_half, tempd_y_half, sizeof(float)*(DIMGLOBY + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_y, tempa_y, sizeof(float)*(DIMGLOBY + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_y_half, tempa_y_half, sizeof(float)*(DIMGLOBY + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_y, tempalpha_y, sizeof(float)*(DIMGLOBY + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_y_half, tempalpha_y_half, sizeof(float)*(DIMGLOBY + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_y, tempb_y, sizeof(float)*(DIMGLOBY + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_y_half, tempb_y_half, sizeof(float)*(DIMGLOBY + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_y, tempK_y, sizeof(float)*(DIMGLOBY + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_y_half, tempK_y_half, sizeof(float)*(DIMGLOBY + 1), cudaMemcpyHostToDevice));

	//-----------------PML Z
	float *tempd_z = (float*)malloc(sizeof(float)*(DIMGLOBZ + 1));
	float *tempd_z_half = (float*)malloc(sizeof(float)*(DIMGLOBZ + 1));
	float *tempa_z = (float*)malloc(sizeof(float)*(DIMGLOBZ + 1));
	float *tempa_z_half = (float*)malloc(sizeof(float)*(DIMGLOBZ + 1));
	float *tempb_z = (float*)malloc(sizeof(float)*(DIMGLOBZ + 1));
	float *tempb_z_half = (float*)malloc(sizeof(float)*(DIMGLOBZ + 1));
	float *tempK_z = (float*)malloc(sizeof(float)*(DIMGLOBZ + 1));
	float *tempK_z_half = (float*)malloc(sizeof(float)*(DIMGLOBZ + 1));
	float *tempalpha_z = (float*)malloc(sizeof(float)*(DIMGLOBZ + 1));
	float *tempalpha_z_half = (float*)malloc(sizeof(float)*(DIMGLOBZ + 1));

	for (int i = 1; i < (DIMGLOBZ + 1); i++) {
		tempd_z[i] = 0.0;
		tempd_z_half[i] = 0.0;
		tempK_z[i] = 1.0;
		tempK_z_half[i] = 1.0;
		tempalpha_z[i] = 0.0;
		tempalpha_z_half[i] = 0.0;
		tempa_z[i] = 0.0;
		tempa_z_half[i] = 0.0;
		tempb_z[i] = 0.0;
		tempb_z_half[i] = 0.0;
	}

	zoriginbottom = thickness_PML_z;
	zorigintop = (DIMGLOBZ - 1)*DELTAZ - thickness_PML_z;
	for (int i = 1; i <= DIMGLOBZ; i++) {
		zval = DELTAZ*float(i - 1);
		abscissa_in_PML = zoriginbottom - zval;//PML ZMIN
		// disable pml zmin for free surface condition
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_z[i] = d0_z*powf(abscissa_normalized, NPOWER);
			tempK_z[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_y[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}
		abscissa_in_PML = yoriginbottom - (yval + DELTAY / 2.0);
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_z_half[i] = d0_z*powf(abscissa_normalized, NPOWER);
			tempK_z_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_z_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = zval - zorigintop;//PML ZMAX
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_z;
			tempd_z[i] = d0_z*powf(abscissa_normalized, NPOWER);
			tempK_z[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_z[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = zval + DELTAZ / 2.0 - zorigintop;
		if (abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			tempd_z_half[i] = d0_z*powf(abscissa_normalized, NPOWER);
			tempK_z_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			tempalpha_z_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		tempb_z[i] = expf(-(tempd_z[i] / tempK_z[i] + tempalpha_z[i])*DELTATT);
		tempb_z_half[i] = expf(-(tempd_z_half[i] / tempK_z_half[i] + tempalpha_z_half[i])*DELTATT);

		if (fabs(tempd_z[i]) > 1e-6) { tempa_z[i] = tempd_z[i] * (tempb_z[i] - 1.0) / (tempK_z[i] * (tempd_z[i] + tempK_z[i] * tempalpha_z[i])); }
		if (fabs(tempd_z_half[i]) > 1e-6) { tempa_z_half[i] = tempd_z_half[i] * (tempb_z_half[i] - 1.0) / (tempK_z_half[i] * (tempd_z_half[i] + tempK_z_half[i] * tempalpha_z_half[i])); }
	}

	HANDLE_ERROR(cudaMemcpy(d_z, tempd_z, sizeof(float)*(DIMGLOBZ + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_z_half, tempd_z_half, sizeof(float)*(DIMGLOBZ + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_z, tempa_z, sizeof(float)*(DIMGLOBZ + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(a_z_half, tempa_z_half, sizeof(float)*(DIMGLOBZ + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_z, tempalpha_z, sizeof(float)*(DIMGLOBZ + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(alpha_z_half, tempalpha_z_half, sizeof(float)*(DIMGLOBZ + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_z, tempb_y, sizeof(float)*(DIMGLOBZ + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(b_z_half, tempb_z_half, sizeof(float)*(DIMGLOBZ + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_z, tempK_z, sizeof(float)*(DIMGLOBZ + 1), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(K_z_half, tempK_z_half, sizeof(float)*(DIMGLOBZ + 1), cudaMemcpyHostToDevice));

	int *DDIMX, *DDIMY, *DDIMZ;
	HANDLE_ERROR(cudaMalloc((void**)&DDIMX, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&DDIMY, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&DDIMZ, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(DDIMX, &DIMX, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DDIMY, &DIMY, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DDIMZ, &DIMZ, sizeof(int), cudaMemcpyHostToDevice));

	int *DDIMGLOBX, *DDIMGLOBY, *DDIMGLOBZ;
	HANDLE_ERROR(cudaMalloc((void**)&DDIMGLOBX, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&DDIMGLOBY, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&DDIMGLOBZ, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(DDIMGLOBX, &DIMGLOBX, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DDIMGLOBY, &DIMGLOBY, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(DDIMGLOBZ, &DIMGLOBZ, sizeof(int), cudaMemcpyHostToDevice));
	
	float *tempvx = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempvy = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempvz = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempsigmaxx = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempsigmaxy = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempsigmayy = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempsigmazz = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempsigmaxz = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempsigmayz = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dvx_dx = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dvx_dy = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dvx_dz = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dvy_dx = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dvy_dy = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dvy_dz = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dvz_dx = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dvz_dy = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dvz_dz = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dsigmaxx_dx = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dsigmayy_dy = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dsigmazz_dz = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dsigmaxy_dx = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dsigmaxy_dy = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dsigmaxz_dx = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dsigmaxz_dz = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dsigmayz_dy = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	float *tempmemory_dsigmayz_dz = (float*)malloc(sizeof(float)*((DIMGLOBX + 1)*(DIMGLOBY + 1)*(DIMGLOBZ + 1)));
	for (int k = 0; k <= DIMGLOBZ; k++) {
		for (int j = 0; j <= DIMGLOBY; j++) {
			for (int i = 0; i <= DIMGLOBX; i++) {
				int ijk = i + j*DIMGLOBX + k*DIMGLOBX*DIMGLOBY;
				tempvx[ijk] = 0;
				tempvy[ijk] = 0;
				tempvz[ijk] = 0;
				tempsigmaxx[ijk] = 0;
				tempsigmaxy[ijk] = 0;
				tempsigmayy[ijk] = 0;
				tempsigmazz[ijk] = 0;
				tempsigmaxz[ijk] = 0;
				tempsigmayz[ijk] = 0;
				tempmemory_dvx_dx[ijk] = 0;
				tempmemory_dvx_dy[ijk] = 0;
				tempmemory_dvx_dz[ijk] = 0;
				tempmemory_dvy_dx[ijk] = 0;
				tempmemory_dvy_dy[ijk] = 0;
				tempmemory_dvy_dz[ijk] = 0;
				tempmemory_dvz_dx[ijk] = 0;
				tempmemory_dvz_dy[ijk] = 0;
				tempmemory_dvz_dz[ijk] = 0;
				tempmemory_dsigmaxx_dx[ijk] = 0;
				tempmemory_dsigmayy_dy[ijk] = 0;
				tempmemory_dsigmazz_dz[ijk] = 0;
				tempmemory_dsigmaxy_dx[ijk] = 0;
				tempmemory_dsigmaxy_dy[ijk] = 0;
				tempmemory_dsigmaxz_dx[ijk] = 0;
				tempmemory_dsigmaxz_dz[ijk] = 0;
				tempmemory_dsigmayz_dy[ijk] = 0;
				tempmemory_dsigmayz_dz[ijk] = 0;
			}
		}
	}

	for (int it = 1; it <= NSTEP; it++) {
		for (int kk = 2; kk < DIMGLOBZ; kk += DIMZ) {
			for (int jj = 2; jj < DIMGLOBY; jj += DIMY) {
				for (int ii = 2; ii < DIMGLOBX; ii += DIMX) {
					// ukuran per slice ---------------------------------------------------------------------------------------------
					int DLOCALDIMZ = DIMZ + offsetperslice;
					int DLOCALDIMY = DIMY + offsetperslice;
					int DLOCALDIMX = DIMX + offsetperslice;
					if ((kk + DIMZ) > DIMGLOBZ) {
						DLOCALDIMZ = (DIMGLOBZ - kk) + offsetperslice;
					}
					if ((jj + DIMY) > DIMGLOBY) {
						DLOCALDIMY = (DIMGLOBY - jj) + offsetperslice;
					}
					if ((ii + DIMX) > DIMGLOBX) {
						DLOCALDIMX = (DIMGLOBX - ii) + offsetperslice;
					}
					int dd = (DLOCALDIMX + 1)*(DLOCALDIMY + 1)*(DLOCALDIMZ + 1);

					int *DDLOCALDIMX, *DDLOCALDIMY, *DDLOCALDIMZ;
					HANDLE_ERROR(cudaMalloc((void**)&DDLOCALDIMX, sizeof(int)));
					HANDLE_ERROR(cudaMalloc((void**)&DDLOCALDIMY, sizeof(int)));
					HANDLE_ERROR(cudaMalloc((void**)&DDLOCALDIMZ, sizeof(int)));
					HANDLE_ERROR(cudaMemcpy(DDLOCALDIMX, &DLOCALDIMX, sizeof(int), cudaMemcpyHostToDevice));
					HANDLE_ERROR(cudaMemcpy(DDLOCALDIMY, &DLOCALDIMY, sizeof(int), cudaMemcpyHostToDevice));
					HANDLE_ERROR(cudaMemcpy(DDLOCALDIMZ, &DLOCALDIMZ, sizeof(int), cudaMemcpyHostToDevice));

					// slicing -------------------------------------------------------------------------------------------------
					int kslbegin = kk - 1;
					int kslend = kk + DIMZ - 1;
					int jslbegin = jj - 1;
					int jslend = jj + DIMY - 1;
					int islbegin = ii - 1;
					int islend = ii + DIMX - 1;
					if ((kk + DIMZ) > DIMGLOBZ) {
						kslbegin = kk - 1;
						kslend = DIMGLOBZ;
					}
					if ((jj + DIMY) > DIMGLOBY) {
						jslbegin = jj - 1;
						jslend = DIMGLOBY;
					}
					if ((ii + DIMX) > DIMGLOBX) {
						islbegin = ii - 1;
						islend = DIMGLOBX;
					}

					int *ISLBEGIN, *JSLBEGIN, *KSLBEGIN;
					HANDLE_ERROR(cudaMalloc((void**)&ISLBEGIN, sizeof(int)));
					HANDLE_ERROR(cudaMalloc((void**)&JSLBEGIN, sizeof(int)));
					HANDLE_ERROR(cudaMalloc((void**)&KSLBEGIN, sizeof(int)));
					HANDLE_ERROR(cudaMemcpy(ISLBEGIN, &islbegin, sizeof(int), cudaMemcpyHostToDevice));
					HANDLE_ERROR(cudaMemcpy(JSLBEGIN, &jslbegin, sizeof(int), cudaMemcpyHostToDevice));
					HANDLE_ERROR(cudaMemcpy(KSLBEGIN, &kslbegin, sizeof(int), cudaMemcpyHostToDevice));
					int *ISLEND, *JSLEND, *KSLEND;
					HANDLE_ERROR(cudaMalloc((void**)&ISLEND, sizeof(int)));
					HANDLE_ERROR(cudaMalloc((void**)&JSLEND, sizeof(int)));
					HANDLE_ERROR(cudaMalloc((void**)&KSLEND, sizeof(int)));
					HANDLE_ERROR(cudaMemcpy(ISLEND, &islend, sizeof(int), cudaMemcpyHostToDevice));
					HANDLE_ERROR(cudaMemcpy(JSLEND, &jslend, sizeof(int), cudaMemcpyHostToDevice));
					HANDLE_ERROR(cudaMemcpy(KSLEND, &kslend, sizeof(int), cudaMemcpyHostToDevice));

					cout << endl << "kslbegin = " << kslbegin;
					cout << endl << "kslend = " << kslend;
					cout << endl << "DLOCAL DIMZ = " << DLOCALDIMZ;
					cout << endl;
					cout << endl << "jslbegin = " << jslbegin;
					cout << endl << "jslend = " << jslend;
					cout << endl << "DLOCAL DIMY = " << DLOCALDIMY;
					cout << endl;
					cout << endl << "islbegin = " << islbegin;
					cout << endl << "islend = " << islend;
					cout << endl << "DLOCAL DIMX = " << DLOCALDIMX;
					cout << endl;
					cout << "------------------------------------" << endl;
					//getch();

					//alokasi memory ---------------------------------------------------------------------------------------
					float *tempvx1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk1 = i + j*DIMX + k*DIMX*DIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempvx1[ijk1] = tempvx[ijk2];
							}
						}
					}
					float *vx;
					HANDLE_ERROR(cudaMalloc((void**)&vx, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(vx, tempvx1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempvx1);

					float *tempvy1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempvy1[ijk] = tempvy[ijk2];
							}
						}
					}
					float *vy;
					HANDLE_ERROR(cudaMalloc((void**)&vy, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(vy, tempvy1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempvy1);

					float *tempvz1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempvz1[ijk] = tempvz[ijk2];
							}
						}
					}
					float *vz;
					HANDLE_ERROR(cudaMalloc((void**)&vz, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(vz, tempvz1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempvz1);

					float *tempsigmaxx1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmaxx1[ijk] = tempsigmaxx[ijk2];
							}
						}
					}
					float *sigmaxx;
					HANDLE_ERROR(cudaMalloc((void**)&sigmaxx, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(sigmaxx, tempsigmaxx1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempsigmaxx1);

					float *tempsigmaxy1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmaxy1[ijk] = tempsigmaxy[ijk2];
							}
						}
					}
					float *sigmaxy;
					HANDLE_ERROR(cudaMalloc((void**)&sigmaxy, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(sigmaxy, tempsigmaxy1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempsigmaxy1);

					float *tempsigmayy1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmayy1[ijk] = tempsigmayy[ijk2];
							}
						}
					}
					float *sigmayy;
					HANDLE_ERROR(cudaMalloc((void**)&sigmayy, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(sigmayy, tempsigmayy1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempsigmayy1);

					float *tempsigmazz1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmazz1[ijk] = tempsigmazz[ijk2];
							}
						}
					}
					float *sigmazz;
					HANDLE_ERROR(cudaMalloc((void**)&sigmazz, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(sigmazz, tempsigmazz1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempsigmazz1);

					float *tempsigmaxz1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmaxz1[ijk] = tempsigmaxz[ijk2];
							}
						}
					}
					float *sigmaxz;
					HANDLE_ERROR(cudaMalloc((void**)&sigmaxz, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(sigmaxz, tempsigmaxz1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempsigmaxz1);

					float *tempsigmayz1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmayz1[ijk] = tempsigmayz[ijk2];
							}
						}
					}
					float *sigmayz;
					HANDLE_ERROR(cudaMalloc((void**)&sigmayz, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(sigmayz, tempsigmayz1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempsigmayz1);

					float *tempmemory_dvx_dx1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvx_dx1[ijk] = tempmemory_dvx_dx[ijk2];
							}
						}
					}
					float *memory_dvx_dx;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dvx_dx, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dvx_dx, tempmemory_dvx_dx1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dvx_dx1);

					float *tempmemory_dvx_dy1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvx_dy1[ijk] = tempmemory_dvx_dy[ijk2];
							}
						}
					}
					float *memory_dvx_dy;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dvx_dy, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dvx_dy, tempmemory_dvx_dy1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dvx_dy1);

					float *tempmemory_dvx_dz1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvx_dz1[ijk] = tempmemory_dvx_dz[ijk2];
							}
						}
					}
					float *memory_dvx_dz;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dvx_dz, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dvx_dz, tempmemory_dvx_dz1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dvx_dz1);

					float *tempmemory_dvy_dx1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvy_dx1[ijk] = tempmemory_dvy_dx[ijk2];
							}
						}
					}
					float *memory_dvy_dx;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dvy_dx, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dvy_dx, tempmemory_dvy_dx1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dvy_dx1);

					float *tempmemory_dvy_dy1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvy_dy1[ijk] = tempmemory_dvy_dy[ijk2];
							}
						}
					}
					float *memory_dvy_dy;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dvy_dy, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dvy_dy, tempmemory_dvy_dy1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dvy_dy1);

					float *tempmemory_dvy_dz1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvy_dz1[ijk] = tempmemory_dvy_dz[ijk2];
							}
						}
					}
					float *memory_dvy_dz;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dvy_dz, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dvy_dz, tempmemory_dvy_dz1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dvy_dz1);

					float *tempmemory_dvz_dx1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvz_dx1[ijk] = tempmemory_dvz_dx[ijk2];
							}
						}
					}
					float *memory_dvz_dx;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dvz_dx, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dvz_dx, tempmemory_dvz_dx1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dvz_dx1);

					float *tempmemory_dvz_dy1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvz_dy1[ijk] = tempmemory_dvz_dy[ijk2];
							}
						}
					}
					float *memory_dvz_dy;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dvz_dy, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dvz_dy, tempmemory_dvz_dy1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dvz_dy1);

					float *tempmemory_dvz_dz1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvz_dz1[ijk] = tempmemory_dvz_dz[ijk2];
							}
						}
					}
					float *memory_dvz_dz;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dvz_dz, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dvz_dz, tempmemory_dvz_dz1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dvz_dz1);

					float *tempmemory_dsigmaxx_dx1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmaxx_dx1[ijk] = tempmemory_dsigmaxx_dx[ijk2];
							}
						}
					}
					float *memory_dsigmaxx_dx;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmaxx_dx, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dsigmaxx_dx, tempmemory_dsigmaxx_dx1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dsigmaxx_dx1);

					float *tempmemory_dsigmayy_dy1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmayy_dy1[ijk] = tempmemory_dsigmayy_dy[ijk2];
							}
						}
					}
					float *memory_dsigmayy_dy;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmayy_dy, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dsigmayy_dy, tempmemory_dsigmayy_dy1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dsigmayy_dy1);

					float *tempmemory_dsigmazz_dz1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmazz_dz1[ijk] = tempmemory_dsigmazz_dz[ijk2];
							}
						}
					}
					float *memory_dsigmazz_dz;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmazz_dz, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dsigmazz_dz, tempmemory_dsigmazz_dz1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dsigmazz_dz1);

					float *tempmemory_dsigmaxy_dx1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmaxy_dx1[ijk] = tempmemory_dsigmaxy_dx[ijk2];
							}
						}
					}
					float *memory_dsigmaxy_dx;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmaxy_dx, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dsigmaxy_dx, tempmemory_dsigmaxy_dx1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dsigmaxy_dx1);

					float *tempmemory_dsigmaxy_dy1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmaxy_dy1[ijk] = tempmemory_dsigmaxy_dy[ijk2];
							}
						}
					}
					float *memory_dsigmaxy_dy;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmaxy_dy, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dsigmaxy_dy, tempmemory_dsigmaxy_dy1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dsigmaxy_dy1);

					float *tempmemory_dsigmaxz_dx1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmaxz_dx1[ijk] = tempmemory_dsigmaxz_dx[ijk2];
							}
						}
					}
					float *memory_dsigmaxz_dx;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmaxz_dx, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dsigmaxz_dx, tempmemory_dsigmaxz_dx1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dsigmaxz_dx1);

					float *tempmemory_dsigmaxz_dz1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmaxz_dz1[ijk] = tempmemory_dsigmaxz_dz[ijk2];
							}
						}
					}
					float *memory_dsigmaxz_dz;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmaxz_dz, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dsigmaxz_dz, tempmemory_dsigmaxz_dz1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dsigmaxz_dz1);

					float *tempmemory_dsigmayz_dy1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmayz_dy1[ijk] = tempmemory_dsigmayz_dy[ijk2];
							}
						}
					}
					float *memory_dsigmayz_dy;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmayz_dy, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dsigmayz_dy, tempmemory_dsigmayz_dy1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dsigmayz_dy1);

					float *tempmemory_dsigmayz_dz1 = (float*)malloc(sizeof(float)*dd);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmayz_dz1[ijk] = tempmemory_dsigmayz_dz[ijk2];
							}
						}
					}
					float *memory_dsigmayz_dz;
					HANDLE_ERROR(cudaMalloc((void**)&memory_dsigmayz_dz, dd*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(memory_dsigmayz_dz, tempmemory_dsigmayz_dz1, sizeof(float)*dd, cudaMemcpyHostToDevice));
					free(tempmemory_dsigmayz_dz1);


					//run fungsi -----------------------------------------------------------------------------------------------------
					dim3 threads;
					threads.x = 10;
					threads.y = 10;
					threads.z = 10;

					dim3 blocks;
					blocks.x = DLOCALDIMX / threads.x;
					blocks.y = DLOCALDIMY / threads.y;
					blocks.z = DLOCALDIMZ / threads.z;



					//copy perslice -> total -----------------------------------------------------------------------------------------
					float *tempvx1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempvx1, vx, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(vx);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempvx[ijk2] = tempvx1[ijk];
							}
						}
					}
					free(tempvx1);

					float *tempvy1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempvy1, vy, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(vy);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempvy[ijk2] = tempvy1[ijk];
							}
						}
					}
					free(tempvy1);

					float *tempvz1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempvz1, vz, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(vz);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempvz[ijk2] = tempvz1[ijk];
							}
						}
					}
					free(tempvz1);

					float *tempsigmaxx1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempsigmaxx1, sigmaxx, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(sigmaxx);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmaxx[ijk2] = tempsigmaxx1[ijk];
							}
						}
					}
					free(tempsigmaxx1);

					float *tempsigmaxy1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempsigmaxy1, sigmaxy, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(sigmaxy);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmaxy[ijk2] = tempsigmaxy1[ijk];
							}
						}
					}
					free(tempsigmaxy1);

					float *tempsigmayy1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempsigmayy1, sigmayy, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(sigmayy);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmayy[ijk2] = tempsigmayy1[ijk];
							}
						}
					}
					free(tempsigmayy1);

					float *tempsigmazz1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempsigmazz1, sigmazz, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(sigmazz);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmazz[ijk2] = tempsigmazz1[ijk];
							}
						}
					}
					free(tempsigmazz1);

					float *tempsigmaxz1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempsigmaxz1, sigmaxz, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(sigmaxz);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmaxz[ijk2] = tempsigmaxz1[ijk];
							}
						}
					}
					free(tempsigmaxz1);

					float *tempsigmayz1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempsigmayz1, sigmayz, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(sigmayz);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempsigmayz[ijk2] = tempsigmayz1[ijk];
							}
						}
					}
					free(tempsigmayz1);

					float *tempmemory_dvx_dx1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dvx_dx1, memory_dvx_dx, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dvx_dx);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvx_dx[ijk2] = tempmemory_dvx_dx1[ijk];
							}
						}
					}
					free(tempmemory_dvx_dx1);

					float *tempmemory_dvx_dy1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dvx_dy1, memory_dvx_dy, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dvx_dy);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvx_dy[ijk2] = tempmemory_dvx_dy1[ijk];
							}
						}
					}
					free(tempmemory_dvx_dy1);

					float *tempmemory_dvx_dz1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dvx_dz1, memory_dvx_dz, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dvx_dz);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvx_dz[ijk2] = tempmemory_dvx_dz1[ijk];
							}
						}
					}
					free(tempmemory_dvx_dz1);

					float *tempmemory_dvy_dx1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dvy_dx1, memory_dvy_dx, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dvy_dx);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvy_dx[ijk2] = tempmemory_dvy_dx1[ijk];
							}
						}
					}
					free(tempmemory_dvy_dx1);

					float *tempmemory_dvy_dy1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dvy_dy1, memory_dvy_dy, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dvy_dy);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvy_dy[ijk2] = tempmemory_dvy_dy1[ijk];
							}
						}
					}
					free(tempmemory_dvy_dy1);

					float *tempmemory_dvy_dz1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dvy_dz1, memory_dvy_dz, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dvy_dz);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvy_dz[ijk2] = tempmemory_dvy_dz1[ijk];
							}
						}
					}
					free(tempmemory_dvy_dz1);

					float *tempmemory_dvz_dx1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dvz_dx1, memory_dvz_dx, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dvz_dx);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvz_dx[ijk2] = tempmemory_dvz_dx1[ijk];
							}
						}
					}
					free(tempmemory_dvz_dx1);

					float *tempmemory_dvz_dy1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dvz_dy1, memory_dvz_dy, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dvz_dy);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvz_dy[ijk2] = tempmemory_dvz_dy1[ijk];
							}
						}
					}
					free(tempmemory_dvz_dy1);

					float *tempmemory_dvz_dz1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dvz_dz1, memory_dvz_dz, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dvz_dz);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dvz_dz[ijk2] = tempmemory_dvz_dz1[ijk];
							}
						}
					}
					free(tempmemory_dvz_dz1);

					float *tempmemory_dsigmaxx_dx1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dsigmaxx_dx1, memory_dsigmaxx_dx, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dsigmaxx_dx);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmaxx_dx[ijk2] = tempmemory_dsigmaxx_dx1[ijk];
							}
						}
					}
					free(tempmemory_dsigmaxx_dx1);

					float *tempmemory_dsigmayy_dy1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dsigmayy_dy1, memory_dsigmayy_dy, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dsigmayy_dy);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmayy_dy[ijk2] = tempmemory_dsigmayy_dy1[ijk];
							}
						}
					}
					free(tempmemory_dsigmayy_dy1);

					float *tempmemory_dsigmazz_dz1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dsigmazz_dz1, memory_dsigmazz_dz, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dsigmazz_dz);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmazz_dz[ijk2] = tempmemory_dsigmazz_dz1[ijk];
							}
						}
					}
					free(tempmemory_dsigmazz_dz1);

					float *tempmemory_dsigmaxy_dx1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dsigmaxy_dx1, memory_dsigmaxy_dx, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dsigmaxy_dx);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmaxy_dx[ijk2] = tempmemory_dsigmaxy_dx1[ijk];
							}
						}
					}
					free(tempmemory_dsigmaxy_dx1);

					float *tempmemory_dsigmaxy_dy1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dsigmaxy_dy1, memory_dsigmaxy_dy, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dsigmaxy_dy);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmaxy_dy[ijk2] = tempmemory_dsigmaxy_dy1[ijk];
							}
						}
					}
					free(tempmemory_dsigmaxy_dy1);

					float *tempmemory_dsigmaxz_dx1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dsigmaxz_dx1, memory_dsigmaxz_dx, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dsigmaxz_dx);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmaxz_dx[ijk2] = tempmemory_dsigmaxz_dx1[ijk];
							}
						}
					}
					free(tempmemory_dsigmaxz_dx1);

					float *tempmemory_dsigmaxz_dz1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dsigmaxz_dz1, memory_dsigmaxz_dz, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dsigmaxz_dz);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmaxz_dz[ijk2] = tempmemory_dsigmaxz_dz1[ijk];
							}
						}
					}
					free(tempmemory_dsigmaxz_dz1);

					float *tempmemory_dsigmayz_dy1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dsigmayz_dy1, memory_dsigmayz_dy, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dsigmayz_dy);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmayz_dy[ijk2] = tempmemory_dsigmayz_dy1[ijk];
							}
						}
					}
					free(tempmemory_dsigmayz_dy1);

					float *tempmemory_dsigmayz_dz1 = (float*)malloc(sizeof(float)*dd);
					HANDLE_ERROR(cudaMemcpy(tempmemory_dsigmayz_dz1, memory_dsigmayz_dz, sizeof(float)*dd, cudaMemcpyDeviceToHost));
					cudaFree(memory_dsigmayz_dz);
					for (int k = 0; k <= DLOCALDIMZ; k++) {
						for (int j = 0; j <= DLOCALDIMY; j++) {
							for (int i = 0; i <= DLOCALDIMX; i++) {
								int ijk = i + j*DLOCALDIMX + k*DLOCALDIMX*DLOCALDIMY;
								int ijk2 = (i + ii - 1) + (j + jj - 1)*DIMGLOBX + (k + kk - 1)*DIMGLOBX*DIMGLOBY;
								tempmemory_dsigmayz_dz[ijk2] = tempmemory_dsigmayz_dz1[ijk];
							}
						}
					}
					free(tempmemory_dsigmayz_dz1);

					//output file -----------------------------------------------------------------------------------------------------------------------
				}
			}
		}
	}
	return 0;
}
