/*
!
! SEISMIC_CPML Version 1.1.1, November 2009.
!
! Copyright Universite de Pau et des Pays de l'Adour, CNRS and INRIA, France.
! Contributor: Dimitri Komatitsch, komatitsch aT lma DOT cnrs-mrs DOT fr
!
! This software is a computer program whose purpose is to solve
! the three-dimensional isotropic elastic wave equation
! using a finite-difference method with Convolutional Perfectly Matched
! Layer (C-PML) conditions.
!
! This software is a computer program whose purpose is to solve
! the two-dimensional viscoelastic anisotropic or poroelastic wave equation
! using a spectral-element method (SEM).
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 2 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
! The full text of the license is available in file "LICENSE".
program seismic_CPML_3D_iso_MPI_OpenMP
! 3D elastic finite-difference code in velocity and stress formulation
! with Convolutional-PML (C-PML) absorbing conditions.
! Dimitri Komatitsch, University of Pau, France, April 2007.
! The second-order staggered-grid formulation of Madariaga (1976) and Virieux (1986) is used.
! The C-PML implementation is based in part on formulas given in Roden and Gedney (2000).
!
! Parallel implementation based on both MPI and OpenMP.
! Type for instance "setenv OMP_NUM_THREADS 4" before running in OpenMP if you want 4 tasks.
! The C-PML implementation is based in part on formulas given in Roden and Gedney (2000).
! If you use this code for your own research, please cite some (or all) of these
! articles:
!
! @ARTICLE{MaKoEz08,
! author = {Roland Martin and Dimitri Komatitsch and Abdela\^aziz Ezziani},
! title = {An unsplit convolutional perfectly matched layer improved at grazing
! incidence for seismic wave equation in poroelastic media},
! journal = {Geophysics},
! year = {2008},
! volume = {73},
! pages = {T51-T61},
! number = {4},
! doi = {10.1190/1.2939484}}
!
! @ARTICLE{MaKo09,
! author = {Roland Martin and Dimitri Komatitsch},
! title = {An unsplit convolutional perfectly matched layer technique improved
! at grazing incidence for the viscoelastic wave equation},
! journal = {Geophysical Journal International},
! year = {2009},
! volume = {179},
! pages = {333-344},
! number = {1},
! doi = {10.1111/j.1365-246X.2009.04278.x}}
!
! @ARTICLE{MaKoGe08,
! author = {Roland Martin and Dimitri Komatitsch and Stephen D. Gedney},
! title = {A variational formulation of a stabilized unsplit convolutional perfectly
! matched layer for the isotropic or anisotropic seismic wave equation},
! journal = {Computer Modeling in Engineering and Sciences},
! year = {2008},
! volume = {37},
! pages = {274-304},
! number = {3}}
!
! @ARTICLE{KoMa07,
! author = {Dimitri Komatitsch and Roland Martin},
! title = {An unsplit convolutional {P}erfectly {M}atched {L}ayer improved
!          at grazing incidence for the seismic wave equation},
! journal = {Geophysics},
! year = {2007},
! volume = {72},
! number = {5},
! pages = {SM155-SM167},
! doi = {10.1190/1.2757586}}
!
! The original CPML technique for Maxwell's equations is described in:
!
! @ARTICLE{RoGe00,
! author = {J. A. Roden and S. D. Gedney},
! title = {Convolution {PML} ({CPML}): {A}n Efficient {FDTD} Implementation
!          of the {CFS}-{PML} for Arbitrary Media},
! journal = {Microwave and Optical Technology Letters},
! year = {2000},
! volume = {27},
! number = {5},
! pages = {334-339},
! doi = {10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A}}
! To display the results as color images in the selected 2D cut plane, use:
!
!   " display image*.gif " or " gimp image*.gif "
!
! or
!
!   " montage -geometry +0+3 -rotate 90 -tile 1x21 image*Vx*.gif allfiles_Vx.gif "
!   " montage -geometry +0+3 -rotate 90 -tile 1x21 image*Vy*.gif allfiles_Vy.gif "
!   then " display allfiles_Vx.gif " or " gimp allfiles_Vx.gif "
!   then " display allfiles_Vy.gif " or " gimp allfiles_Vy.gif "
!
! IMPORTANT : all our CPML codes work fine in single precision as well (which is significantly faster).
!             If you want you can thus force automatic conversion to single precision at compile time
!             or change all the declarations and constants in the code from double precision to single.
*/

/*
25-Oct-2015
conversion from Fortran90 to C++.
replacing openMP and MPI parallelization into CUDA C++ in a single CPU unit.
generate input from generate_3d_model_anticline.m (https://github.com/thejatmik/ela3d/blob/master/generate_3d_model_anticline.m)
GPU memory management using unified memory (cudaMallocManaged, cudaMemcpyAsync), no need for cudaFree
need improvement in displaying results
jatmikatejasukmana@gmail.com
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

__global__ void kersigmaxyz(double *cp, double *cs, double *rho, double *DELTAT, int *DDIMX, int *DDIMY, int *DDIMZ, double *memory_dvx_dx, double *memory_dvy_dy, double *memory_dvz_dz, double *a_x_half, double *a_y, double *a_z, double *b_x_half, double *b_y, double *b_z, double *K_x_half, double *K_y, double *K_z, double *sigmaxx, double *sigmayy, double *sigmazz, double *ONE_OVER_DELTAX, double *ONE_OVER_DELTAY, double *ONE_OVER_DELTAZ, double *vx, double *vy, double *vz) {
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


	if ((index_z >= 1) && (index_z < DDIMZ[0])) {
		if ((index_y >= 1) && (index_y < DDIMY[0])) {
			if ((index_x >= 0) && (index_x < DDIMX[0] - 1)) {
				double vp = cp[offset];
				double vs = cs[offset];
				double rhos = rho[offset];

				double lambda = rhos*(vp*vp - 2 * vs*vs);
				double lambdaplus2mu = rhos*vp*vp;

				double DELTAT_lambdaplus2mu = DELTAT[0] * lambdaplus2mu;
				double DELTAT_lambda = DELTAT[0] * lambda;

				double value_dvx_dx = (vx[right] - vx[offset])*ONE_OVER_DELTAX[0];
				double value_dvy_dy = (vy[offset] - vy[ybottom])*ONE_OVER_DELTAY[0];
				double value_dvz_dz = (vz[offset] - vz[zbottom])*ONE_OVER_DELTAZ[0];

				memory_dvx_dx[offset] = b_x_half[index_x] * memory_dvx_dx[offset] + a_x_half[index_x] * value_dvx_dx;
				memory_dvy_dy[offset] = b_y[index_y] * memory_dvy_dy[offset] + a_y[index_y] * value_dvy_dy;
				memory_dvz_dz[offset] = b_z[index_z] * memory_dvz_dz[offset] + a_z[index_z] * value_dvz_dz;

				value_dvx_dx = value_dvx_dx / K_x_half[index_x] + memory_dvx_dx[offset];
				value_dvy_dy = value_dvy_dy / K_y[index_y] + memory_dvy_dy[offset];
				value_dvz_dz = value_dvz_dz / K_z[index_z] + memory_dvz_dz[offset];

				sigmaxx[offset] = DELTAT_lambdaplus2mu * value_dvx_dx + DELTAT_lambda * (value_dvy_dy + value_dvz_dz) + sigmaxx[offset];
				sigmayy[offset] = DELTAT_lambda * (value_dvx_dx + value_dvz_dz) + DELTAT_lambdaplus2mu * value_dvy_dy + sigmayy[offset];
				sigmazz[offset] = DELTAT_lambda * (value_dvx_dx + value_dvy_dy) + DELTAT_lambdaplus2mu * value_dvz_dz + sigmazz[offset];
			}
		}
	}

}

__global__ void kersigmaxy(double *cp, double *cs, double *rho, int *DDIMX, int *DDIMY, int *DDIMZ, double *DELTAT, double *memory_dvy_dx, double *memory_dvx_dy, double *a_x, double *a_y_half, double *b_x, double *b_y_half, double *K_x, double *K_y_half, double *ONE_OVER_DELTAX, double *ONE_OVER_DELTAY, double *vx, double *vy, double *sigmaxy) {
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


	if ((index_z >= 0) && (index_z < DDIMZ[0])) {
		if ((index_y >= 0) && (index_y < DDIMY[0] - 1)) {
			if ((index_x >= 1) && (index_x < DDIMX[0])) {
				double vs = (cs[left] + cs[ytop]) / 2;
				double rhos = (rho[left] + rho[ytop]) / 2;

				double mu = rhos*vs*vs;

				double DELTAT_mu = DELTAT[0] * mu;

				double value_dvy_dx = (vy[offset] - vy[left])*ONE_OVER_DELTAX[0];
				double value_dvx_dy = (vx[ytop] - vx[offset])*ONE_OVER_DELTAY[0];

				memory_dvy_dx[offset] = b_x[index_x] * memory_dvy_dx[offset] + a_x[index_x] * value_dvy_dx;
				memory_dvx_dy[offset] = b_y_half[index_y] * memory_dvx_dy[offset] + a_y_half[index_y] * value_dvx_dy;

				value_dvy_dx = value_dvy_dx / K_x[index_x] + memory_dvy_dx[offset];
				value_dvx_dy = value_dvx_dy / K_y_half[index_y] + memory_dvx_dy[offset];

				sigmaxy[offset] = DELTAT_mu * (value_dvy_dx + value_dvx_dy) + sigmaxy[offset];
			}
		}
	}

}

__global__ void kersigmaxzyz(double *cp, double *cs, double *rho, int *DDIMX, int *DDIMY, int *DDIMZ, double *DELTAT, double *memory_dvz_dx, double *memory_dvx_dz, double *memory_dvz_dy, double *memory_dvy_dz, double *a_x, double *a_z, double *a_y_half, double *a_z_half, double *b_x, double *b_y_half, double *b_z_half, double *K_x, double *K_y_half, double *K_z_half, double *ONE_OVER_DELTAX, double *ONE_OVER_DELTAY, double *ONE_OVER_DELTAZ, double *vx, double *vy, double *vz, double *sigmaxz, double *sigmayz) {
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


	if ((index_z >= 0) && (index_z < DDIMZ[0]-1)) {
		//sigmaxz
		if ((index_y >= 0) && (index_y < DDIMY[0])) {
			if ((index_x >= 1) && (index_x < DDIMX[0])) {
				double vs = (cs[left] + cs[ztop]) / 2;
				double rhos = (rho[left] + rho[ztop]) / 2;

				double mu = rhos*vs*vs;

				double DELTAT_mu = DELTAT[0] * mu;

				double value_dvz_dx = (vz[offset] - vz[left]) * ONE_OVER_DELTAX[0];
				double value_dvx_dz = (vx[ztop] - vx[offset]) * ONE_OVER_DELTAZ[0];

				memory_dvz_dx[offset] = b_x[index_x] * memory_dvz_dx[offset] + a_x[index_x] * value_dvz_dx;
				memory_dvx_dz[offset] = b_z_half[index_z] * memory_dvx_dz[offset] + a_z_half[index_z] * value_dvx_dz;

				value_dvz_dx = value_dvz_dx / K_x[index_x] + memory_dvz_dx[offset];
				value_dvx_dz = value_dvx_dz / K_z_half[index_z] + memory_dvx_dz[offset];

				sigmaxz[offset] = DELTAT_mu * (value_dvz_dx + value_dvx_dz) + sigmaxz[offset];
			}
		}

		//sigmayz
		if ((index_y >= 0) && (index_y < DDIMY[0] - 1)) {
			if ((index_x >= 0) && (index_x < DDIMX[0])) {
				double vs = (cs[ytop] + cs[ztop]) / 2;
				double rhos = (rho[ytop] + rho[ztop]) / 2;

				double mu = rhos*vs*vs;

				double DELTAT_mu = DELTAT[0] * mu;

				double value_dvz_dy = (vz[ytop] - vz[offset]) * ONE_OVER_DELTAY[0];
				double value_dvy_dz = (vy[ztop] - vy[offset]) * ONE_OVER_DELTAZ[0];

				memory_dvz_dy[offset] = b_y_half[index_y] * memory_dvz_dy[offset] + a_y_half[index_y] * value_dvz_dy;
				memory_dvy_dz[offset] = b_z_half[index_z] * memory_dvy_dz[offset] + a_z_half[index_z] * value_dvy_dz;

				value_dvz_dy = value_dvz_dy / K_y_half[index_y] + memory_dvz_dy[offset];
				value_dvy_dz = value_dvy_dz / K_z_half[index_z] + memory_dvy_dz[offset];

				sigmayz[offset] = DELTAT_mu * (value_dvz_dy + value_dvy_dz) + sigmayz[offset];
			}
		}
	}

}

__global__ void kervxvy(double *rho, int *DDIMX, int *DDIMY, int *DDIMZ, double *DELTAT, double *sigmaxx, double *sigmaxy, double *sigmaxz, double *sigmayy, double *sigmayz, double *memory_dsigmaxx_dx, double *memory_dsigmaxy_dy, double *memory_dsigmaxz_dz, double *memory_dsigmaxy_dx, double *memory_dsigmayy_dy, double *memory_dsigmayz_dz, double *a_x, double *a_y, double *a_z, double *a_x_half, double *a_y_half, double *b_x, double *b_y, double *b_z, double *b_x_half, double *b_y_half, double *K_x, double *K_y, double *K_z, double *K_x_half, double *K_y_half, double *ONE_OVER_DELTAX, double *ONE_OVER_DELTAY, double *ONE_OVER_DELTAZ, double *vx, double *vy) {
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


	if ((index_z >= 1) && (index_z < DDIMZ[0])) {
		//vx
		if ((index_y >= 1) && (index_y < DDIMY[0])) {
			if ((index_x >= 1) && (index_x < DDIMX[0])) {
				double rhos = (rho[offset] + rho[left]) / 2;

				double DELTAT_over_rho = DELTAT[0] / rhos;

				double value_dsigmaxx_dx = (sigmaxx[offset] - sigmaxx[left]) * ONE_OVER_DELTAX[0];
				double value_dsigmaxy_dy = (sigmaxy[offset] - sigmaxy[ybottom]) * ONE_OVER_DELTAY[0];
				double value_dsigmaxz_dz = (sigmaxz[offset] - sigmaxz[zbottom]) * ONE_OVER_DELTAZ[0];

				memory_dsigmaxx_dx[offset] = b_x[index_x] * memory_dsigmaxx_dx[offset] + a_x[index_x] * value_dsigmaxx_dx;
				memory_dsigmaxy_dy[offset] = b_y[index_y] * memory_dsigmaxy_dy[offset] + a_y[index_y] * value_dsigmaxy_dy;
				memory_dsigmaxz_dz[offset] = b_z[index_z] * memory_dsigmaxz_dz[offset] + a_z[index_z] * value_dsigmaxz_dz;

				value_dsigmaxx_dx = value_dsigmaxx_dx / K_x[index_x] + memory_dsigmaxx_dx[offset];
				value_dsigmaxy_dy = value_dsigmaxy_dy / K_y[index_y] + memory_dsigmaxy_dy[offset];
				value_dsigmaxz_dz = value_dsigmaxz_dz / K_z[index_z] + memory_dsigmaxz_dz[offset];

				vx[offset] = DELTAT_over_rho * (value_dsigmaxx_dx + value_dsigmaxy_dy + value_dsigmaxz_dz) + vx[offset];
			}
		}

		//vy
		if ((index_y >= 0) && (index_y < DDIMY[0] - 1)) {
			if ((index_x >= 0) && (index_x < DDIMX[0] - 1)) {
				double rhos = (rho[offset] + rho[ytop]) / 2;

				double DELTAT_over_rho = DELTAT[0] / rhos;

				double value_dsigmaxy_dx = (sigmaxy[right] - sigmaxy[offset]) * ONE_OVER_DELTAX[0];
				double value_dsigmayy_dy = (sigmayy[ytop] - sigmayy[offset]) * ONE_OVER_DELTAY[0];
				double value_dsigmayz_dz = (sigmayz[offset] - sigmayz[zbottom]) * ONE_OVER_DELTAZ[0];

				memory_dsigmaxy_dx[offset] = b_x_half[index_x] * memory_dsigmaxy_dx[offset] + a_x_half[index_x] * value_dsigmaxy_dx;
				memory_dsigmayy_dy[offset] = b_y_half[index_y] * memory_dsigmayy_dy[offset] + a_y_half[index_y] * value_dsigmayy_dy;
				memory_dsigmayz_dz[offset] = b_z[index_z] * memory_dsigmayz_dz[offset] + a_z[index_z] * value_dsigmayz_dz;

				value_dsigmaxy_dx = value_dsigmaxy_dx / K_x_half[index_x] + memory_dsigmaxy_dx[offset];
				value_dsigmayy_dy = value_dsigmayy_dy / K_y_half[index_y] + memory_dsigmayy_dy[offset];
				value_dsigmayz_dz = value_dsigmayz_dz / K_z[index_z] + memory_dsigmayz_dz[offset];

				vy[offset] = DELTAT_over_rho * (value_dsigmaxy_dx + value_dsigmayy_dy + value_dsigmayz_dz) + vy[offset];
			}
		}
	}

}

__global__ void kervz(double *rho, int *DDIMX, int *DDIMY, int *DDIMZ, double *DELTAT, double *sigmaxz, double *sigmayz, double *sigmazz, double *memory_dsigmaxz_dx, double *memory_dsigmayz_dy, double *memory_dsigmazz_dz, double *b_x_half, double *b_y, double *b_z_half, double *a_x_half, double *a_y, double *a_z_half, double *K_x_half, double *K_y, double *K_z_half, double *ONE_OVER_DELTAX, double *ONE_OVER_DELTAY, double *ONE_OVER_DELTAZ, double *vz) {
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


	if ((index_z >= 0) && (index_z < DDIMZ[0] - 1)) {
		if ((index_y >= 1) && (index_y < DDIMY[0])) {
			if ((index_x >= 0) && (index_x < DDIMX[0] - 1)) {
				double rhos = (rho[offset] + rho[ztop]) / 2;

				double DELTAT_over_rho = DELTAT[0] / rhos;

				double value_dsigmaxz_dx = (sigmaxz[right] - sigmaxz[offset]) * ONE_OVER_DELTAX[0];
				double value_dsigmayz_dy = (sigmayz[offset] - sigmayz[ybottom]) * ONE_OVER_DELTAY[0];
				double value_dsigmazz_dz = (sigmazz[ztop] - sigmazz[offset]) * ONE_OVER_DELTAZ[0];

				memory_dsigmaxz_dx[offset] = b_x_half[index_x] * memory_dsigmaxz_dx[offset] + a_x_half[index_x] * value_dsigmaxz_dx;
				memory_dsigmayz_dy[offset] = b_y[index_y] * memory_dsigmayz_dy[offset] + a_y[index_y] * value_dsigmayz_dy;
				memory_dsigmazz_dz[offset] = b_z_half[index_z] * memory_dsigmazz_dz[offset] + a_z_half[index_z] * value_dsigmazz_dz;

				value_dsigmaxz_dx = value_dsigmaxz_dx / K_x_half[index_x] + memory_dsigmaxz_dx[offset];
				value_dsigmayz_dy = value_dsigmayz_dy / K_y[index_y] + memory_dsigmayz_dy[offset];
				value_dsigmazz_dz = value_dsigmazz_dz / K_z_half[index_z] + memory_dsigmazz_dz[offset];

				vz[offset] = DELTAT_over_rho * (value_dsigmaxz_dx + value_dsigmayz_dy + value_dsigmazz_dz) + vz[offset];
			}
		}
	}

}

__global__ void keraddSource(double *lesrcx, double *lesrcy, double *lesrcz, double *sigmaxx, double *sigmayy, double *sigmazz, double *cp, double *cs, double *rho, int *DDIMX, int *DDIMY, int *DDIMZ, int *iit, int *ISOURCE, int *JSOURCE, int *KSOURCE, double *ANGLE_FORCE, double *DEGREES_TO_RADIANS, double *DELTAT, double *factor, double *t0, double *ff0, double *DPI, double *vx, double *vy, double *vz) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = index_x + index_y*DDIMX[0] + index_z*DDIMX[0] * DDIMY[0];
	int left = offset - 1;
	int ytop = offset + DDIMX[0];
	int ztop = offset + DDIMX[0] * DDIMY[0];

	double lambdaplus2mu = rho[offset] * cp[offset] * cp[offset];

	double a = DPI[0] * DPI[0] * ff0[0] * ff0[0];
	double t = double(iit[0] - 1)*DELTAT[0];
	int iitt = iit[0];
	//Gaussian
	//double source_term = factor * expf(-a*powf((t - t0), 2));

	//first derivative of a Gaussian
	//double source_term = -factor[0] * 2.0*a*(t - t0[0])*expf(-a*powf((t - t0[0]), 2));

	//Ricker source time function(second derivative of a Gaussian)
	double source_term = factor[0]*(1.0 - 2.0*a*powf((t - t0[0]), 2))*expf(-a*powf(t - t0[0], 2));

	double force_x = sinf(ANGLE_FORCE[0] * DEGREES_TO_RADIANS[0])*source_term;
	double force_y = cosf(ANGLE_FORCE[0] * DEGREES_TO_RADIANS[0])*source_term;
	double force_z = cosf(ANGLE_FORCE[0] * DEGREES_TO_RADIANS[0])*source_term;

	if (index_z == KSOURCE[0]) {
		if (index_y == JSOURCE[0]) {
			if (index_x == ISOURCE[0]) {
				//earthquake event source
				/*vx[offset] = vx[offset] + source_term*DELTAT[0] / ((rho[offset] + rho[left]) / 2);
				vy[offset] = vy[offset] + source_term*DELTAT[0] / ((rho[offset] + rho[ytop]) / 2);
				vz[offset] = vz[offset] + source_term*DELTAT[0] / ((rho[offset] + rho[ztop]) / 2);
				*/
				/*explosives source*/
				sigmaxx[offset] = sigmaxx[offset] + force_x*DELTAT[0] * lambdaplus2mu;
				sigmayy[offset] = sigmayy[offset] + force_x*DELTAT[0] * lambdaplus2mu;
				sigmazz[offset] = sigmazz[offset] + force_y*DELTAT[0] * lambdaplus2mu;
				//sigmaxx[offset] = force_x*DELTAT[0] * lambdaplus2mu;
				//sigmayy[offset] = force_x*DELTAT[0] * lambdaplus2mu;
				//sigmazz[offset] = force_y*DELTAT[0] * lambdaplus2mu;

				lesrcx[iitt] = sigmaxx[offset];
				lesrcy[iitt] = sigmayy[offset];
				lesrcz[iitt] = sigmazz[offset];
				
			}
		}
	}
}

__global__ void kerGather(int *ISOURCE, int *JSOURCE, int *KSOURCE, int *DDIMX, int *DDIMY, int *DDIMZ, double *gatvx, double *gatvz, double *gatvy, int *DIMX, int *DIMY, int *DIMZ, int *gatx, int *gaty, int *DPML, double *vx, double *vy, double *vz) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int index_z = blockIdx.z * blockDim.z + threadIdx.z;

	int blkId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int aaaa = blockDim.x * blockDim.y * blockDim.z;
	int bbbb = threadIdx.z * (blockDim.x * blockDim.y);
	int cccc = (threadIdx.y * blockDim.x) + threadIdx.x;
	int offset = index_x + index_y*DDIMX[0] + index_z*DDIMX[0] * DDIMZ[0];

	int xlen = (DDIMX[0] - 2 * DPML[0]) / (gatx[0]);
	int ylen = (DDIMY[0] - 2 * DPML[0]) / (gaty[0]);

	if (index_z == (KSOURCE[0] + 1)) { //gather permukaan
		int gxy = index_x + index_y*DDIMX[0];
		gatvx[gxy] = vx[offset];
		gatvy[gxy] = vy[offset];
		gatvz[gxy] = vz[offset];
	}
}

int main(void) {
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	
	int NIMX, NIMY, NIMZ;
	NIMX = 373;
	NIMY = 373;
	NIMZ = 373;
		
	int *DDIMX, *DDIMY, *DDIMZ;
	HANDLE_ERROR(cudaMallocManaged((void**)&DDIMX, sizeof(int)));
	HANDLE_ERROR(cudaMallocManaged((void**)&DDIMY, sizeof(int)));
	HANDLE_ERROR(cudaMallocManaged((void**)&DDIMZ, sizeof(int)));
	DDIMX[0] = NIMX;
	DDIMY[0] = NIMY;
	DDIMZ[0] = NIMZ;

	int *gatx, *gaty;
	HANDLE_ERROR(cudaMallocManaged((void**)&gatx, sizeof(int)));
	HANDLE_ERROR(cudaMallocManaged((void**)&gaty, sizeof(int)));
	gatx[0] = 100;
	gaty[0] = 6;

	double *gatvx, *gatvy, *gatvz;
	HANDLE_ERROR(cudaMallocManaged((void**)&gatvx, sizeof(double)*(NIMX*NIMY)));
	HANDLE_ERROR(cudaMallocManaged((void**)&gatvy, sizeof(double)*(NIMX*NIMY)));
	HANDLE_ERROR(cudaMallocManaged((void**)&gatvz, sizeof(double)*(NIMX*NIMY)));
	for (int i = 0; i < NIMX; i++){
		for (int j = 0; j < NIMY; j++) {
			int ij = i + j*NIMX;
			gatvx[ij] = 0.0;
			gatvy[ij] = 0.0;
			gatvz[ij] = 0.0;
		}
	}

	int NSTEP = 200;
	double *lesrcx, *lesrcy, *lesrcz;
	HANDLE_ERROR(cudaMallocManaged((void**)&lesrcx, sizeof(double)*(NSTEP + 1)));
	HANDLE_ERROR(cudaMallocManaged((void**)&lesrcy, sizeof(double)*(NSTEP + 1)));
	HANDLE_ERROR(cudaMallocManaged((void**)&lesrcz, sizeof(double)*(NSTEP + 1)));
	for (int i = 0; i < NSTEP + 1; i++) {
		lesrcx[i] = 0;
		lesrcy[i] = 0;
		lesrcz[i] = 0;
	}
	double DELTATT = 1e-3;
	int sampgat = 1; //tsamp = sampgat*Deltat
	int IT_OUTPUT = 200;

	int DELTAX, DELTAY, DELTAZ;
	DELTAX = 10; DELTAY = DELTAX; DELTAZ = DELTAX;
	double ONE_OVER_DELTAXX, ONE_OVER_DELTAYY, ONE_OVER_DELTAZZ;
	ONE_OVER_DELTAXX = 1 / double(DELTAX);
	ONE_OVER_DELTAYY = ONE_OVER_DELTAXX;
	ONE_OVER_DELTAZZ = ONE_OVER_DELTAXX;

	double *ONE_OVER_DELTAX, *ONE_OVER_DELTAY, *ONE_OVER_DELTAZ;
	HANDLE_ERROR(cudaMallocManaged((void**)&ONE_OVER_DELTAX, sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&ONE_OVER_DELTAY, sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&ONE_OVER_DELTAZ, sizeof(double)));
	HANDLE_ERROR(cudaMemcpyAsync(ONE_OVER_DELTAX, &ONE_OVER_DELTAXX, sizeof(double), cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(ONE_OVER_DELTAY, &ONE_OVER_DELTAYY, sizeof(double), cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(ONE_OVER_DELTAZ, &ONE_OVER_DELTAZZ, sizeof(double), cudaMemcpyHostToDevice, 0));

	double *cp;
	HANDLE_ERROR(cudaMallocManaged((void**)&cp, NIMX*NIMY*NIMZ*sizeof(double)));
	double *cs;
	HANDLE_ERROR(cudaMallocManaged((void**)&cs, NIMX*NIMY*NIMZ*sizeof(double)));
	double *rho;
	HANDLE_ERROR(cudaMallocManaged((void**)&rho, NIMX*NIMY*NIMZ*sizeof(double)));
	for (int k = 0; k < (NIMZ); k++) {
		for (int j = 0; j < (NIMY); j++) {
			for (int i = 0; i < (NIMX); i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cp[ijk] = 3300.0;
				cs[ijk] = 2700.0;
				rho[ijk] = 2800.0;
				/*if (k >= 50) {
					tempcp[ijk] = 2000;
					tempcs[ijk] = 2000 / 1.732;
					temprho[ijk] = 1700;
				}*/
			}
		}
	}

	char nmfilet[20];
	sprintf_s(nmfilet, "tesfile.bin");
	std::ofstream foutt(nmfilet, ios::out | ios::binary);
	//sprintf_s(nmfile3, "vx%05i.txt", it);
	//std::ofstream fout3(nmfile3, ios::out);
	for (int kk = 0; kk < NIMZ; kk++) {
		for (int jj = 0; jj < NIMY; jj++) {
			for (int ii = 0; ii < NIMX; ii++) {
				int ijk = ii + jj*NIMX + kk*NIMX*NIMY;
				foutt.write((char *)&rho[ijk], sizeof rho[ijk]);
			}
		}
	}
	std::string line;
	/*
	line = "modelcp.bin";
	char *inname1 = const_cast<char*>(line.c_str());
	FILE* file = fopen(inname1, "rb");
	if (file != NULL){
		cout << endl << "Model Vp";
	}
	for (int k = 0; k < NIMZ; k++) {
		for (int j = 0; j < NIMY; j++) {
			for (int i = 0; i < NIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				double f;
				fread(&f, sizeof(double), 1, file);
				cp[ijk] = f;
			}
		}
	}
	line = "modelcs.bin";
	char *inname2 = const_cast<char*>(line.c_str());
	FILE* file2 = fopen(inname2, "rb");
	if (file2 != NULL){
		cout << endl << "Model Vs";
	}
	for (int k = 0; k < NIMZ; k++) {
		for (int j = 0; j < NIMY; j++) {
			for (int i = 0; i < NIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				double f;
				fread(&f, sizeof(double), 1, file2);
				cs[ijk] = f;
			}
		}
	}
	line = "modelrh.bin";
	char *inname3 = const_cast<char*>(line.c_str());
	FILE* file3 = fopen(inname3, "rb");
	if (file3 != NULL){
		cout << endl << "Model Rho";
	}
	for (int k = 0; k < NIMZ; k++) {
		for (int j = 0; j < NIMY; j++) {
			for (int i = 0; i < NIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				double f;
				fread(&f, sizeof(double), 1, file3);
				rho[ijk] = f;
			}
		}
	}
	*/
	
	double *DELTAT;
	HANDLE_ERROR(cudaMallocManaged((void**)&DELTAT, sizeof(double)));
	HANDLE_ERROR(cudaMemcpyAsync(DELTAT, &DELTATT, sizeof(double), cudaMemcpyHostToDevice, 0));

	double f0, tt0, factorr;
	f0 = 17;
	tt0 = 1.2 / f0;
	double *ff0, *t0;
	HANDLE_ERROR(cudaMallocManaged((void**)&t0, sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&ff0, sizeof(double)));
	HANDLE_ERROR(cudaMemcpyAsync(ff0, &f0, sizeof(double), cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(t0, &tt0, sizeof(double), cudaMemcpyHostToDevice, 0));
	factorr = 1e+10;
	double *factor;
	HANDLE_ERROR(cudaMallocManaged((void**)&factor, sizeof(double)));
	HANDLE_ERROR(cudaMemcpyAsync(factor, &factorr, sizeof(double), cudaMemcpyHostToDevice, 0));
	int NPOINTS_PML = 10;
	int *DPML;
	HANDLE_ERROR(cudaMallocManaged((void**)&DPML, sizeof(int)));
	HANDLE_ERROR(cudaMemcpyAsync(DPML, &NPOINTS_PML, sizeof(int), cudaMemcpyHostToDevice, 0));
	int ISOURCEE, JSOURCEE, KSOURCEE;
	ISOURCEE = NIMX / 2;
	JSOURCEE = NIMY / 2;
	KSOURCEE = NIMZ / 2;

	int *ISOURCE, *KSOURCE, *JSOURCE;
	HANDLE_ERROR(cudaMallocManaged((void**)&ISOURCE, sizeof(int)));
	HANDLE_ERROR(cudaMallocManaged((void**)&JSOURCE, sizeof(int)));
	HANDLE_ERROR(cudaMallocManaged((void**)&KSOURCE, sizeof(int)));
	HANDLE_ERROR(cudaMemcpyAsync(ISOURCE, &ISOURCEE, sizeof(int), cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(JSOURCE, &JSOURCEE, sizeof(int), cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(KSOURCE, &KSOURCEE, sizeof(int), cudaMemcpyHostToDevice, 0));

	double ANGLE_FORCEE = 0;
	double *ANGLE_FORCE;
	HANDLE_ERROR(cudaMallocManaged((void**)&ANGLE_FORCE, sizeof(double)));
	HANDLE_ERROR(cudaMemcpyAsync(ANGLE_FORCE, &ANGLE_FORCEE, sizeof(double), cudaMemcpyHostToDevice, 0));
	
	double PI = 3.141592653589793238462643;
	double *DPI;
	HANDLE_ERROR(cudaMallocManaged((void**)&DPI, sizeof(double)));
	HANDLE_ERROR(cudaMemcpyAsync(DPI, &PI, sizeof(double), cudaMemcpyHostToDevice, 0));
	double DEGREES_TO_RADIANSS = PI / 180;
	double *DEGREES_TO_RADIANS;
	HANDLE_ERROR(cudaMallocManaged((void**)&DEGREES_TO_RADIANS, sizeof(double)));
	HANDLE_ERROR(cudaMemcpyAsync(DEGREES_TO_RADIANS, &DEGREES_TO_RADIANSS, sizeof(double), cudaMemcpyHostToDevice, 0));

	double NPOWER = 2;
	double K_MAX_PML = 1;
	double ALPHA_MAX_PML = 2 * PI*(f0 / 2);

	double *vx;
	double *vy;
	double *vz;
	double *sigmaxx;
	double *sigmaxy;
	double *sigmayy;
	double *sigmazz;
	double *sigmaxz;
	double *sigmayz;
	double *memory_dvx_dx;
	double *memory_dvx_dy;
	double *memory_dvx_dz;
	double *memory_dvy_dx;
	double *memory_dvy_dy;
	double *memory_dvy_dz;
	double *memory_dvz_dx;
	double *memory_dvz_dy;
	double *memory_dvz_dz;
	double *memory_dsigmaxx_dx;
	double *memory_dsigmayy_dy;
	double *memory_dsigmazz_dz;
	double *memory_dsigmaxy_dx;
	double *memory_dsigmaxy_dy;
	double *memory_dsigmaxz_dx;
	double *memory_dsigmaxz_dz;
	double *memory_dsigmayz_dy;
	double *memory_dsigmayz_dz;

	HANDLE_ERROR(cudaMallocManaged((void**)&vx, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&vy, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&vz, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&sigmaxx, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&sigmaxy, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&sigmayy, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&sigmazz, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&sigmaxz, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&sigmayz, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dvx_dx, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dvx_dy, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dvx_dz, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dvy_dx, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dvy_dy, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dvy_dz, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dsigmaxx_dx, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dsigmayy_dy, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dsigmazz_dz, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dsigmaxy_dy, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dsigmaxy_dx, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dsigmaxz_dx, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dsigmaxz_dz, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dsigmayz_dy, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dsigmayz_dz, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dvz_dz, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dvz_dy, NIMX*NIMY*NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&memory_dvz_dx, NIMX*NIMY*NIMZ*sizeof(double)));

	for (int k = 0; k < NIMZ; k++) {
		for (int j = 0; j < NIMY; j++) {
			for (int i = 0; i < NIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				vx[ijk] = 0;
				vy[ijk] = 0;
				vz[ijk] = 0;
				sigmaxx[ijk] = 0;
				sigmaxy[ijk] = 0;
				sigmayy[ijk] = 0;
				sigmazz[ijk] = 0;
				sigmaxz[ijk] = 0;
				sigmayz[ijk] = 0;
				memory_dvx_dx[ijk] = 0;
				memory_dvx_dy[ijk] = 0;
				memory_dvx_dz[ijk] = 0;
				memory_dvy_dx[ijk] = 0;
				memory_dvy_dy[ijk] = 0;
				memory_dvy_dz[ijk] = 0;
				memory_dvz_dx[ijk] = 0;
				memory_dvz_dy[ijk] = 0;
				memory_dvz_dz[ijk] = 0;
				memory_dsigmaxx_dx[ijk] = 0;
				memory_dsigmayy_dy[ijk] = 0;
				memory_dsigmazz_dz[ijk] = 0;
				memory_dsigmaxy_dx[ijk] = 0;
				memory_dsigmaxy_dy[ijk] = 0;
				memory_dsigmaxz_dx[ijk] = 0;
				memory_dsigmaxz_dz[ijk] = 0;
				memory_dsigmayz_dy[ijk] = 0;
				memory_dsigmayz_dz[ijk] = 0;
			}
		}
	}
	
	
	double *d_x, *K_x, *alpha_x, *a_x, *b_x, *d_x_half, *K_x_half, *alpha_x_half, *a_x_half, *b_x_half;
	HANDLE_ERROR(cudaMallocManaged((void**)&d_x, NIMX*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&K_x, NIMX*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&alpha_x, NIMX*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&a_x, NIMX*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&b_x, NIMX*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&d_x_half, NIMX*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&K_x_half, NIMX*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&alpha_x_half, NIMX*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&a_x_half, NIMX*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&b_x_half, NIMX*sizeof(double)));

	double *d_y, *K_y, *alpha_y, *a_y, *b_y, *d_y_half, *K_y_half, *alpha_y_half, *a_y_half, *b_y_half;
	HANDLE_ERROR(cudaMallocManaged((void**)&d_y, NIMY*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&K_y, NIMY*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&alpha_y, NIMY*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&a_y, NIMY*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&b_y, NIMY*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&d_y_half, NIMY*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&K_y_half, NIMY*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&alpha_y_half, NIMY*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&a_y_half, NIMY*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&b_y_half, NIMY*sizeof(double)));

	double *d_z, *K_z, *alpha_z, *a_z, *b_z, *d_z_half, *K_z_half, *alpha_z_half, *a_z_half, *b_z_half;
	HANDLE_ERROR(cudaMallocManaged((void**)&d_z, NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&K_z, NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&alpha_z, NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&a_z, NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&b_z, NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&d_z_half, NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&K_z_half, NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&alpha_z_half, NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&a_z_half, NIMZ*sizeof(double)));
	HANDLE_ERROR(cudaMallocManaged((void**)&b_z_half, NIMZ*sizeof(double)));

	double thickness_PML_x, thickness_PML_y, thickness_PML_z;
	double xoriginleft, xoriginright, yoriginbottom, yorigintop, zoriginbottom, zorigintop;
	double Rcoef, d0_x, d0_y, d0_z, xval, yval, zval, abscissa_in_PML, abscissa_normalized;

	double Courant_number;

	thickness_PML_x = NPOINTS_PML * DELTAX;
	thickness_PML_y = NPOINTS_PML * DELTAY;
	thickness_PML_z = NPOINTS_PML * DELTAZ;
	Rcoef = 0.001;

	double vpml = 3000;
	d0_x = -(NPOWER + 1) * vpml * logf(Rcoef) / (2.0 * thickness_PML_x);
	d0_y = -(NPOWER + 1) * vpml * logf(Rcoef) / (2.0 * thickness_PML_y);
	d0_z = -(NPOWER + 1) * vpml * logf(Rcoef) / (2.0 * thickness_PML_z);

	//------------------PML X
	double *tempd_x = (double*)malloc(sizeof(double)*NIMX);
	double *tempd_x_half = (double*)malloc(sizeof(double)*NIMX);
	double *tempa_x = (double*)malloc(sizeof(double)*NIMX);
	double *tempa_x_half = (double*)malloc(sizeof(double)*NIMX);
	double *tempb_x = (double*)malloc(sizeof(double)*NIMX);
	double *tempb_x_half = (double*)malloc(sizeof(double)*NIMX);
	double *tempK_x = (double*)malloc(sizeof(double)*NIMX);
	double *tempK_x_half = (double*)malloc(sizeof(double)*NIMX);
	double *tempalpha_x = (double*)malloc(sizeof(double)*NIMX);
	double *tempalpha_x_half = (double*)malloc(sizeof(double)*NIMX);

	for (int i = 0; i < NIMX; i++) {
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
	xoriginright = (NIMX - 1)*DELTAX - thickness_PML_x;
	for (int i = 0; i < NIMX; i++) {
		xval = DELTAX*double(i);
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

	HANDLE_ERROR(cudaMemcpyAsync(d_x, tempd_x, sizeof(double)*NIMX, cudaMemcpyHostToDevice,0));
	HANDLE_ERROR(cudaMemcpyAsync(d_x_half, tempd_x_half, sizeof(double)*NIMX, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(a_x, tempa_x, sizeof(double)*NIMX, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(a_x_half, tempa_x_half, sizeof(double)*NIMX, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(alpha_x, tempalpha_x, sizeof(double)*NIMX, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(alpha_x_half, tempalpha_x_half, sizeof(double)*NIMX, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(b_x, tempb_x, sizeof(double)*NIMX, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(b_x_half, tempb_x_half, sizeof(double)*NIMX, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(K_x, tempK_x, sizeof(double)*NIMX, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(K_x_half, tempK_x_half, sizeof(double)*NIMX, cudaMemcpyHostToDevice, 0));

	//-----------------PML Y
	double *tempd_y = (double*)malloc(sizeof(double)*NIMY);
	double *tempd_y_half = (double*)malloc(sizeof(double)*NIMY);
	double *tempa_y = (double*)malloc(sizeof(double)*NIMY);
	double *tempa_y_half = (double*)malloc(sizeof(double)*NIMY);
	double *tempb_y = (double*)malloc(sizeof(double)*NIMY);
	double *tempb_y_half = (double*)malloc(sizeof(double)*NIMY);
	double *tempK_y = (double*)malloc(sizeof(double)*NIMY);
	double *tempK_y_half = (double*)malloc(sizeof(double)*NIMY);
	double *tempalpha_y = (double*)malloc(sizeof(double)*NIMY);
	double *tempalpha_y_half = (double*)malloc(sizeof(double)*NIMY);

	for (int i = 0; i < NIMY; i++) {
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
	yorigintop = (NIMY - 1)*DELTAY - thickness_PML_y;
	for (int i = 0; i < NIMY; i++) {
		yval = DELTAY*double(i);
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

	HANDLE_ERROR(cudaMemcpyAsync(d_y, tempd_y, sizeof(double)*NIMY, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(d_y_half, tempd_y_half, sizeof(double)*NIMY, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(a_y, tempa_y, sizeof(double)*NIMY, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(a_y_half, tempa_y_half, sizeof(double)*NIMY, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(alpha_y, tempalpha_y, sizeof(double)*NIMY, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(alpha_y_half, tempalpha_y_half, sizeof(double)*NIMY, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(b_y, tempb_y, sizeof(double)*NIMY, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(b_y_half, tempb_y_half, sizeof(double)*NIMY, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(K_y, tempK_y, sizeof(double)*NIMY, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(K_y_half, tempK_y_half, sizeof(double)*NIMY, cudaMemcpyHostToDevice, 0));

	//-----------------PML Z
	double *tempd_z = (double*)malloc(sizeof(double)*NIMZ);
	double *tempd_z_half = (double*)malloc(sizeof(double)*NIMZ);
	double *tempa_z = (double*)malloc(sizeof(double)*NIMZ);
	double *tempa_z_half = (double*)malloc(sizeof(double)*NIMZ);
	double *tempb_z = (double*)malloc(sizeof(double)*NIMZ);
	double *tempb_z_half = (double*)malloc(sizeof(double)*NIMZ);
	double *tempK_z = (double*)malloc(sizeof(double)*NIMZ);
	double *tempK_z_half = (double*)malloc(sizeof(double)*NIMZ);
	double *tempalpha_z = (double*)malloc(sizeof(double)*NIMZ);
	double *tempalpha_z_half = (double*)malloc(sizeof(double)*NIMZ);

	for (int i = 0; i < NIMZ; i++) {
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
	zorigintop = (NIMZ - 1)*DELTAZ - thickness_PML_z;
	for (int i = 0; i < NIMZ; i++) {
		zval = DELTAZ*double(i);
		abscissa_in_PML = zoriginbottom - zval;
		//PML ZMIN
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

	HANDLE_ERROR(cudaMemcpyAsync(d_z, tempd_z, sizeof(double)*NIMZ, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(d_z_half, tempd_z_half, sizeof(double)*NIMZ, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(a_z, tempa_z, sizeof(double)*NIMZ, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(a_z_half, tempa_z_half, sizeof(double)*NIMZ, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(alpha_z, tempalpha_z, sizeof(double)*NIMZ, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(alpha_z_half, tempalpha_z_half, sizeof(double)*NIMZ, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(b_z, tempb_y, sizeof(double)*NIMZ, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(b_z_half, tempb_z_half, sizeof(double)*NIMZ, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(K_z, tempK_z, sizeof(double)*NIMZ, cudaMemcpyHostToDevice, 0));
	HANDLE_ERROR(cudaMemcpyAsync(K_z_half, tempK_z_half, sizeof(double)*NIMZ, cudaMemcpyHostToDevice, 0));

	dim3 threads;
	threads.x = 5;
	threads.y = 5;
	threads.z = 5;

	dim3 blocks;
	blocks.x = NIMX / threads.x;
	blocks.y = NIMY / threads.y;
	blocks.z = NIMZ / threads.z;

	int *iit;
	HANDLE_ERROR(cudaMalloc((void**)&iit, sizeof(int)));

	for (int it = 1; it <= NSTEP; it++){
		HANDLE_ERROR(cudaMemcpy(iit, &it, sizeof(int), cudaMemcpyHostToDevice));

		kersigmaxyz << <blocks, threads >> >(cp, cs, rho, DELTAT, DDIMX, DDIMY, DDIMZ, memory_dvx_dx, memory_dvy_dy, memory_dvz_dz, a_x_half, a_y, a_z, b_x_half, b_y, b_z, K_x_half, K_y, K_z, sigmaxx, sigmayy, sigmazz, ONE_OVER_DELTAX, ONE_OVER_DELTAY, ONE_OVER_DELTAZ, vx, vy, vz);

		kersigmaxy << <blocks, threads >> >(cp, cs, rho, DDIMX, DDIMY, DDIMZ, DELTAT, memory_dvy_dx, memory_dvx_dy, a_x, a_y_half, b_x, b_y_half, K_x, K_y_half, ONE_OVER_DELTAX, ONE_OVER_DELTAY, vx, vy, sigmaxy);

		kersigmaxzyz << <blocks, threads >> >(cp, cs, rho, DDIMX, DDIMY, DDIMZ, DELTAT, memory_dvz_dx, memory_dvx_dz, memory_dvz_dy, memory_dvy_dz, a_x, a_z, a_y_half, a_z_half, b_x, b_y_half, b_z_half, K_x, K_y_half, K_z_half, ONE_OVER_DELTAX, ONE_OVER_DELTAY, ONE_OVER_DELTAZ, vx, vy, vz, sigmaxz, sigmayz);

		kervxvy << <blocks, threads >> >(rho, DDIMX, DDIMY, DDIMZ, DELTAT, sigmaxx, sigmaxy, sigmaxz, sigmayy, sigmayz, memory_dsigmaxx_dx, memory_dsigmaxy_dy, memory_dsigmaxz_dz, memory_dsigmaxy_dx, memory_dsigmayy_dy, memory_dsigmayz_dz, a_x, a_y, a_z, a_x_half, a_y_half, b_x, b_y, b_z, b_x_half, b_y_half, K_x, K_y, K_z, K_x_half, K_y_half, ONE_OVER_DELTAX, ONE_OVER_DELTAY, ONE_OVER_DELTAZ, vx, vy);

		kervz << <blocks, threads >> >(rho, DDIMX, DDIMY, DDIMZ, DELTAT, sigmaxz, sigmayz, sigmazz, memory_dsigmaxz_dx, memory_dsigmayz_dy, memory_dsigmazz_dz, b_x_half, b_y, b_z_half, a_x_half, a_y, a_z_half, K_x_half, K_y, K_z_half, ONE_OVER_DELTAX, ONE_OVER_DELTAY, ONE_OVER_DELTAZ, vz);

		kerGather << <blocks, threads >> >(ISOURCE, JSOURCE, KSOURCE, DDIMX, DDIMY, DDIMZ, gatvx, gatvz, gatvy, DDIMX, DDIMY, DDIMZ, gatx, gaty, DPML, vx, vy, vz);

		keraddSource << <blocks, threads >> >(lesrcx, lesrcy, lesrcz, sigmaxx, sigmayy, sigmazz, cp, cs, rho, DDIMX, DDIMY, DDIMZ, iit, ISOURCE, JSOURCE, KSOURCE, ANGLE_FORCE, DEGREES_TO_RADIANS, DELTAT, factor, t0, ff0, DPI, vx, vy, vz);
		
		if (fmod(it, sampgat) == 0) {
			char nmfile4[20], nmfile5[20], nmfile6[20];

			int xlen = NIMX / gatx[0];
			int ylen = NIMY / gaty[0];

			
			//sprintf(nmfile4, "rechorvx.bin");
			//std::ofstream fout4(nmfile4, ios::out | ios::app | ios::binary);
			sprintf(nmfile4, "rechorvx.txt");
			std::ofstream fout4;
			fout4.open(nmfile4, ios::app);
			for (int j = 0; j < NIMY; j += ylen) {
				for (int i = 0; i < NIMX; i += xlen) {
					int kk = i + j*NIMX;
					fout4 << gatvx[kk] << " ";
				}
			}fout4 << endl;

			//sprintf(nmfile5, "rechorvy.bin");
			//std::ofstream fout5(nmfile5, ios::out | ios::app | ios::binary);
			sprintf(nmfile5, "rechorvy.txt");
			std::ofstream fout5;
			fout5.open(nmfile5, ios::app);
			for (int j = 0; j < NIMY; j += ylen) {
				for (int i = 0; i < NIMX; i += xlen) {
					int kk = i + j*NIMX;
					fout5 << gatvy[kk] << " ";
				}
			}fout5 << endl;

			//sprintf(nmfile6, "rechorvz.bin");
			//std::ofstream fout6(nmfile6, ios::out | ios::app | ios::binary);
			sprintf(nmfile6, "rechorvz.txt");
			std::ofstream fout6;
			fout6.open(nmfile6, ios::app);
			for (int j = 0; j < NIMY; j += ylen) {
				for (int i = 0; i < NIMX; i += xlen) {
					int kk = i + j*NIMX;
					fout6 << gatvz[kk] << " ";
				}
			}fout6 << endl;
		}

		if (fmod(it, IT_OUTPUT) == 0){

			//save to file
			char nmfile1[20]; char nmfile2[20]; char nmfile3[20];

			sprintf_s(nmfile1, "vz%05i.bin", it);
			std::ofstream fout1(nmfile1, ios::out | ios::binary);
			//sprintf_s(nmfile1, "vz%05i.txt", it);
			//std::ofstream fout1(nmfile1, ios::out);
			for (int kk = 0; kk < NIMZ; kk++) {
				for (int jj = 0; jj < NIMY; jj++) {
					for (int ii = 0; ii < NIMX; ii++) {
						int ijk = ii + jj*NIMX + kk*NIMX*NIMY;
						fout1.write((char *)&vz[ijk], sizeof vz[ijk]);
					}
				}
			}

			sprintf_s(nmfile2, "vy%05i.bin", it);
			std::ofstream fout2(nmfile2, ios::out | ios::binary);
			//sprintf_s(nmfile2, "vy%05i.txt", it);
			//std::ofstream fout2(nmfile2, ios::out);
			for (int kk = 0; kk < NIMZ; kk++) {
				for (int jj = 0; jj < NIMY; jj++) {
					for (int ii = 0; ii < NIMX; ii++) {
						int ijk = ii + jj*NIMX + kk*NIMX*NIMY;
						fout2.write((char *)&vy[ijk], sizeof vy[ijk]);
					}
				}
			}

			sprintf_s(nmfile3, "vx%05i.bin", it);
			std::ofstream fout3(nmfile3, ios::out | ios::binary);
			//sprintf_s(nmfile3, "vx%05i.txt", it);
			//std::ofstream fout3(nmfile3, ios::out);
			for (int kk = 0; kk < NIMZ; kk++) {
				for (int jj = 0; jj < NIMY; jj++) {
					for (int ii = 0; ii < NIMX; ii++) {
						int ijk = ii + jj*NIMX + kk*NIMX*NIMY;
						fout3.write((char *)&vx[ijk], sizeof vx[ijk]);
					}
				}
			}

			//save to file END
		}
	}

	char nmfilesx[20], nmfilesy[20], nmfilesz[20];

	sprintf_s(nmfilesx, "recsrcx.txt");
	std::ofstream foutsx;
	foutsx.open(nmfilesx, ios::out);
	for (int kk = 0; kk < NSTEP + 1; kk++) {
		foutsx << lesrcx[kk] << " ";
	}

	sprintf_s(nmfilesy, "recsrcy.txt");
	std::ofstream foutsy;
	foutsy.open(nmfilesy, ios::out);
	for (int kk = 0; kk < NSTEP + 1; kk++) {
		foutsy << lesrcy[kk] << " ";
	}

	sprintf_s(nmfilesz, "recsrcz.txt");
	std::ofstream foutsz;
	foutsz.open(nmfilesz, ios::out);
	for (int kk = 0; kk < NSTEP + 1; kk++) {
		foutsz << lesrcz[kk] << " ";
	}
	
	return 0;
}
