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
#define _CRT_SECURE_NO_WARNINGS

using namespace std;

int main(void) {
	int NIMX, NIMY, NIMZ;
	NIMX = 500;
	NIMY = 500;
	NIMZ = 500;

	int DIMX, DIMY, DIMZ;
	DIMX = NIMX + 1; DIMY = NIMY + 1; DIMZ = NIMZ + 1;

	int Ngatx = 100;
	int Ngaty = 5;

	int NSTEP = 2000;
	float DELTATT = 1e-3;
	int sampgat = 2; //tsamp = sampgat*Deltat
	int IT_OUTPUT = 200;

	int DELTAX, DELTAY, DELTAZ;
	DELTAX = 10; DELTAY = DELTAX; DELTAZ = DELTAX;
	float ONE_OVER_DELTAXX, ONE_OVER_DELTAYY, ONE_OVER_DELTAZZ;
	ONE_OVER_DELTAXX = 1 / float(DELTAX);
	ONE_OVER_DELTAZZ = ONE_OVER_DELTAXX; ONE_OVER_DELTAYY = ONE_OVER_DELTAXX;

	float *ccp = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	float *ccs = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	float *crho = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));

	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				ccp[ijk] = 3300;
				ccs[ijk] = 3300 / 1.732;
				crho[ijk] = 3000;
				if (k >= 50) {
					ccp[ijk] = 2000;
					ccs[ijk] = 2000 / 1.732;
					crho[ijk] = 1700;
				}
			}
		}
	}

	float f0, t0, factor;
	f0 = 35;
	t0 = 1.2 / f0;
	factor = 1e+7;

	int NPOINTS_PML = 10;

	int ISOURCEE, KSOURCEE, JSOURCEE;
	ISOURCEE = DIMX / 2;
	JSOURCEE = DIMY / 2;
	KSOURCEE = 5;
	float ANGLE_FORCEE = 0;
	float PI = 3.141592653589793238462643;
	float DEGREES_TO_RADIANSS = PI / 180;
	float NPOWER = 2;
	float K_MAX_PML = 1;
	float ALPHA_MAX_PML = 2 * PI*(f0 / 2);

	float *cvx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cvx[ijk] = 0;
			}
		}
	}

	float *cvy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cvy[ijk] = 0;
			}
		}
	}

	float *cvz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cvz[ijk] = 0;
			}
		}
	}

	float *csigmaxx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				csigmaxx[ijk] = 0;
			}
		}
	}

	float *csigmaxy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				csigmaxy[ijk] = 0;
			}
		}
	}

	float *csigmayy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				csigmayy[ijk] = 0;
			}
		}
	}

	float *csigmazz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				csigmazz[ijk] = 0;
			}
		}
	}

	float *csigmaxz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				csigmaxz[ijk] = 0;
			}
		}
	}

	float *csigmayz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				csigmayz[ijk] = 0;
			}
		}
	}

	float *cmemory_dvx_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dvx_dx[ijk] = 0;
			}
		}
	}

	float *cmemory_dvx_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dvx_dy[ijk] = 0;
			}
		}
	}

	float *cmemory_dvx_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dvx_dz[ijk] = 0;
			}
		}
	}

	float *cmemory_dvy_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dvy_dx[ijk] = 0;
			}
		}
	}

	float *cmemory_dvy_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dvy_dy[ijk] = 0;
			}
		}
	}

	float *cmemory_dvy_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dvy_dz[ijk] = 0;
			}
		}
	}

	float *cmemory_dvz_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dvz_dx[ijk] = 0;
			}
		}
	}

	float *cmemory_dvz_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dvz_dy[ijk] = 0;
			}
		}
	}

	float *cmemory_dvz_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dvz_dz[ijk] = 0;
			}
		}
	}

	float *cmemory_dsigmaxx_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dsigmaxx_dx[ijk] = 0;
			}
		}
	}

	float *cmemory_dsigmayy_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dsigmayy_dy[ijk] = 0;
			}
		}
	}

	float *cmemory_dsigmazz_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dsigmazz_dz[ijk] = 0;
			}
		}
	}

	float *cmemory_dsigmaxy_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dsigmaxy_dx[ijk] = 0;
			}
		}
	}

	float *cmemory_dsigmaxy_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dsigmaxy_dy[ijk] = 0;
			}
		}
	}

	float *cmemory_dsigmaxz_dx = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dsigmaxz_dx[ijk] = 0;
			}
		}
	}

	float *cmemory_dsigmaxz_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dsigmaxz_dz[ijk] = 0;
			}
		}
	}

	float *cmemory_dsigmayz_dy = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dsigmayz_dy[ijk] = 0;
			}
		}
	}

	float *cmemory_dsigmayz_dz = (float*)malloc(sizeof(float)*(DIMX*DIMY*DIMZ));
	for (int k = 0; k < DIMZ; k++) {
		for (int j = 0; j < DIMY; j++) {
			for (int i = 0; i < DIMX; i++) {
				int ijk = i + j*NIMX + k*NIMX*NIMY;
				cmemory_dsigmayz_dz[ijk] = 0;
			}
		}
	}

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

	float *cd_x = (float*)malloc(sizeof(float)*DIMX);
	float *cd_x_half = (float*)malloc(sizeof(float)*DIMX);
	float *ca_x = (float*)malloc(sizeof(float)*DIMX);
	float *ca_x_half = (float*)malloc(sizeof(float)*DIMX);
	float *cb_x = (float*)malloc(sizeof(float)*DIMX);
	float *cb_x_half = (float*)malloc(sizeof(float)*DIMX);
	float *cK_x = (float*)malloc(sizeof(float)*DIMX);
	float *cK_x_half = (float*)malloc(sizeof(float)*DIMX);
	float *calpha_x = (float*)malloc(sizeof(float)*DIMX);
	float *calpha_x_half = (float*)malloc(sizeof(float)*DIMX);

	for (int i = 1; i < DIMX; i++) {
		cd_x[i] = 0.0;
		cd_x_half[i] = 0.0;
		cK_x[i] = 1.0;
		cK_x_half[i] = 1.0;
		calpha_x[i] = 0.0;
		calpha_x_half[i] = 0.0;
		ca_x[i] = 0.0;
		ca_x_half[i] = 0.0;
		cb_x[i] = 0.0;
		cb_x_half[i] = 0.0;
	}

	xoriginleft = thickness_PML_x;
	xoriginright = (NIMX - 1)*DELTAX - thickness_PML_x;
	for (int i = 1; i <= NIMX; i++) {
		xval = DELTAX*float(i - 1);
		abscissa_in_PML = xoriginleft - xval;//PML XMIN
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			cd_x[i] = d0_x*powf(abscissa_normalized, NPOWER);
			cK_x[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_x[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = xoriginleft - (xval + DELTAX / 2.0);
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			cd_x_half[i] = d0_x*powf(abscissa_normalized, NPOWER);
			cK_x_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_x_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = xval - xoriginright;//PML XMAX
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			cd_x[i] = d0_x*powf(abscissa_normalized, NPOWER);
			cK_x[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_x[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = xval + DELTAX / 2.0 - xoriginright;
		if (abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / thickness_PML_x;
			cd_x_half[i] = d0_x*powf(abscissa_normalized, NPOWER);
			cK_x_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_x_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}
		if (calpha_x[i] < 0.0) { calpha_x[i] = 0.0; }
		if (calpha_x_half[i] < 0.0) { calpha_x_half[i] = 0.0; }
		cb_x[i] = expf(-(cd_x[i] / cK_x[i] + calpha_x[i])*DELTATT);
		cb_x_half[i] = expf(-(cd_x_half[i] / cK_x_half[i] + calpha_x_half[i])*DELTATT);

		if (fabs(cd_x[i]) > 1e-6) { ca_x[i] = cd_x[i] * (cb_x[i] - 1.0) / (cK_x[i] * (cd_x[i] + cK_x[i] * calpha_x[i])); }
		if (fabs(cd_x_half[i]) > 1e-6) { calpha_x_half[i] = cd_x_half[i] * (cb_x_half[i] - 1.0) / (cK_x_half[i] * (cd_x_half[i] + cK_x_half[i] * calpha_x_half[i])); }
	}

	float *cd_y = (float*)malloc(sizeof(float)*DIMY);
	float *cd_y_half = (float*)malloc(sizeof(float)*DIMY);
	float *ca_y = (float*)malloc(sizeof(float)*DIMY);
	float *ca_y_half = (float*)malloc(sizeof(float)*DIMY);
	float *cb_y = (float*)malloc(sizeof(float)*DIMY);
	float *cb_y_half = (float*)malloc(sizeof(float)*DIMY);
	float *cK_y = (float*)malloc(sizeof(float)*DIMY);
	float *cK_y_half = (float*)malloc(sizeof(float)*DIMY);
	float *calpha_y = (float*)malloc(sizeof(float)*DIMY);
	float *calpha_y_half = (float*)malloc(sizeof(float)*DIMY);

	for (int i = 1; i < DIMY; i++) {
		cd_y[i] = 0.0;
		cd_y_half[i] = 0.0;
		cK_y[i] = 1.0;
		cK_y_half[i] = 1.0;
		calpha_y[i] = 0.0;
		calpha_y_half[i] = 0.0;
		ca_y[i] = 0.0;
		ca_y_half[i] = 0.0;
		cb_y[i] = 0.0;
		cb_y_half[i] = 0.0;
	}

	yoriginbottom = thickness_PML_y;
	yorigintop = (NIMY - 1)*DELTAY - thickness_PML_y;
	for (int i = 1; i <= NIMY; i++) {
		yval = DELTAY*float(i - 1);
		abscissa_in_PML = yoriginbottom - yval;//PML YMIN
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			cd_y[i] = d0_y*powf(abscissa_normalized, NPOWER);
			cK_y[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_y[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = yoriginbottom - (yval + DELTAY / 2.0);
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			cd_y_half[i] = d0_y*powf(abscissa_normalized, NPOWER);
			cK_y_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_y_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = yval - yorigintop;//PML YMAX
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			cd_y[i] = d0_y*powf(abscissa_normalized, NPOWER);
			cK_y[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_y[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = yval + DELTAY / 2.0 - yorigintop;
		if (abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			cd_y_half[i] = d0_y*powf(abscissa_normalized, NPOWER);
			cK_y_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_y_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		cb_y[i] = expf(-(cd_y[i] / cK_y[i] + calpha_y[i])*DELTATT);
		cb_y_half[i] = expf(-(cd_y_half[i] / cK_y_half[i] + calpha_y_half[i])*DELTATT);

		if (fabs(cd_y[i]) > 1e-6) { ca_y[i] = cd_y[i] * (cb_y[i] - 1.0) / (cK_y[i] * (cd_y[i] + cK_y[i] * calpha_y[i])); }
		if (fabs(cd_y_half[i]) > 1e-6) { ca_y_half[i] = cd_y_half[i] * (cb_y_half[i] - 1.0) / (cK_y_half[i] * (cd_y_half[i] + cK_y_half[i] * calpha_y_half[i])); }
	}

	//-----------------PML Z
	float *cd_z = (float*)malloc(sizeof(float)*DIMZ);
	float *cd_z_half = (float*)malloc(sizeof(float)*DIMZ);
	float *ca_z = (float*)malloc(sizeof(float)*DIMZ);
	float *ca_z_half = (float*)malloc(sizeof(float)*DIMZ);
	float *cb_z = (float*)malloc(sizeof(float)*DIMZ);
	float *cb_z_half = (float*)malloc(sizeof(float)*DIMZ);
	float *cK_z = (float*)malloc(sizeof(float)*DIMZ);
	float *cK_z_half = (float*)malloc(sizeof(float)*DIMZ);
	float *calpha_z = (float*)malloc(sizeof(float)*DIMZ);
	float *calpha_z_half = (float*)malloc(sizeof(float)*DIMZ);

	for (int i = 1; i < DIMZ; i++) {
		cd_z[i] = 0.0;
		cd_z_half[i] = 0.0;
		cK_z[i] = 1.0;
		cK_z_half[i] = 1.0;
		calpha_z[i] = 0.0;
		calpha_z_half[i] = 0.0;
		ca_z[i] = 0.0;
		ca_z_half[i] = 0.0;
		cb_z[i] = 0.0;
		cb_z_half[i] = 0.0;
	}

	zoriginbottom = thickness_PML_z;
	zorigintop = (NIMZ - 1)*DELTAZ - thickness_PML_z;
	for (int i = 1; i <= NIMZ; i++) {
		zval = DELTAZ*float(i - 1);
		abscissa_in_PML = zoriginbottom - zval;
		//PML ZMIN

		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			cd_z[i] = d0_z*powf(abscissa_normalized, NPOWER);
			cK_z[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_z[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}
		abscissa_in_PML = yoriginbottom - (yval + DELTAY / 2.0);
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			cd_z_half[i] = d0_z*powf(abscissa_normalized, NPOWER);
			cK_z_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_z_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}


		abscissa_in_PML = zval - zorigintop;//PML ZMAX
		if (abscissa_in_PML >= 0.0) {
			abscissa_normalized = abscissa_in_PML / thickness_PML_z;
			cd_z[i] = d0_z*powf(abscissa_normalized, NPOWER);
			cK_z[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_z[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		abscissa_in_PML = zval + DELTAZ / 2.0 - zorigintop;
		if (abscissa_in_PML >= 0.0){
			abscissa_normalized = abscissa_in_PML / thickness_PML_y;
			cd_z_half[i] = d0_z*powf(abscissa_normalized, NPOWER);
			cK_z_half[i] = 1.0 + (K_MAX_PML - 1.0)*powf(abscissa_normalized, NPOWER);
			calpha_z_half[i] = ALPHA_MAX_PML*(1.0 - abscissa_normalized) + 0.1*ALPHA_MAX_PML;
		}

		cb_z[i] = expf(-(cd_z[i] / cK_z[i] + calpha_z[i])*DELTATT);
		cb_z_half[i] = expf(-(cd_z_half[i] / cK_z_half[i] + calpha_z_half[i])*DELTATT);

		if (fabs(cd_z[i]) > 1e-6) { ca_z[i] = cd_z[i] * (cb_z[i] - 1.0) / (cK_z[i] * (cd_z[i] + cK_z[i] * calpha_z[i])); }
		if (fabs(cd_z_half[i]) > 1e-6) { ca_z_half[i] = cd_z_half[i] * (cb_z_half[i] - 1.0) / (cK_z_half[i] * (cd_z_half[i] + cK_z_half[i] * calpha_z_half[i])); }
	}

	float *gatvx = (float*)malloc(sizeof(float)*(DIMX*DIMY));
	float *gatvy = (float*)malloc(sizeof(float)*(DIMX*DIMY));
	float *gatvz = (float*)malloc(sizeof(float)*(DIMX*DIMY));
	for (int i = 0; i < DIMX; i++){
		for (int j = 0; j < DIMY; j++) {
			int ij = i + j*DIMX;
			gatvx[ij] = 0.0;
			gatvy[ij] = 0.0;
			gatvz[ij] = 0.0;
		}
	}

	for (int it = 1; it <= NSTEP; it++) {
		//sigmaxyz
		for (int k = 2; k <= NIMZ; k++) {
			for (int j = 2; j <= NIMY; j++) {
				for (int i = 1; i <= NIMX - 1; i++) {
					int offset = i + j*NIMX + k*NIMX * NIMY;
					int right = offset + 1;
					int ybottom = offset - NIMX;
					int zbottom = offset - NIMX * NIMY;

					float vp = ccp[offset];
					float vs = ccs[offset];
					float rhos = crho[offset];

					float lambda = rhos*(vp*vp - 2 * vs*vs);
					float lambdaplus2mu = rhos*vp*vp;

					float DELTAT_lambdaplus2mu = DELTATT * lambdaplus2mu;
					float DELTAT_lambda = DELTATT * lambda;

					float value_dvx_dx = (cvx[right] - cvx[offset])*ONE_OVER_DELTAXX;
					float value_dvy_dy = (cvy[offset] - cvy[ybottom])*ONE_OVER_DELTAYY;
					float value_dvz_dz = (cvz[offset] - cvz[zbottom])*ONE_OVER_DELTAZZ;

					cmemory_dvx_dx[offset] = cb_x_half[i] * cmemory_dvx_dx[offset] + ca_x_half[i] * value_dvx_dx;
					cmemory_dvy_dy[offset] = cb_y[j] * cmemory_dvy_dy[offset] + ca_y[j] * value_dvy_dy;
					cmemory_dvz_dz[offset] = cb_z[k] * cmemory_dvz_dz[offset] + ca_z[k] * value_dvz_dz;

					value_dvx_dx = value_dvx_dx / cK_x_half[i] + cmemory_dvx_dx[offset];
					value_dvy_dy = value_dvy_dy / cK_y[j] + cmemory_dvy_dy[offset];
					value_dvz_dz = value_dvz_dz / cK_z[k] + cmemory_dvz_dz[offset];

					csigmaxx[offset] = DELTAT_lambdaplus2mu * value_dvx_dx + DELTAT_lambda * (value_dvy_dy + value_dvz_dz) + csigmaxx[offset];
					csigmayy[offset] = DELTAT_lambda * (value_dvx_dx + value_dvz_dz) + DELTAT_lambdaplus2mu * value_dvy_dy + csigmayy[offset];
					csigmazz[offset] = DELTAT_lambda * (value_dvx_dx + value_dvy_dy) + DELTAT_lambdaplus2mu * value_dvz_dz + csigmazz[offset];
				}
			}
		}

		//sigmaxy
		for (int k = 1; k <= NIMZ; k++) {
			for (int j = 1; j <= NIMY - 1; j++) {
				for (int i = 2; i <= NIMX; i++) {
					int offset = i + j*NIMX + k*NIMX * NIMY;
					int left = offset - 1;
					int ytop = offset + NIMX;

					float vs = (ccs[left] + ccs[ytop]) / 2;
					float rhos = (crho[left] + crho[ytop]) / 2;

					float mu = rhos*vs*vs;

					float DELTAT_mu = DELTATT * mu;

					float value_dvy_dx = (cvy[offset] - cvy[left])*ONE_OVER_DELTAXX;
					float value_dvx_dy = (cvx[ytop] - cvx[offset])*ONE_OVER_DELTAYY;

					cmemory_dvy_dx[offset] = cb_x[i] * cmemory_dvy_dx[offset] + ca_x[i] * value_dvy_dx;
					cmemory_dvx_dy[offset] = cb_y_half[j] * cmemory_dvx_dy[offset] + ca_y_half[j] * value_dvx_dy;

					value_dvy_dx = value_dvy_dx / cK_x[i] + cmemory_dvy_dx[offset];
					value_dvx_dy = value_dvx_dy / cK_y_half[j] + cmemory_dvx_dy[offset];

					csigmaxy[offset] = DELTAT_mu * (value_dvy_dx + value_dvx_dy) + csigmaxy[offset];
				}
			}
		}

		//sigmaxzyz
		for (int k = 1; k <= NIMZ; k++) {
			for (int j = 1; j <= NIMY; j++) {
				for (int i = 2; i <= NIMX; i++) {
					int offset = i + j*NIMX + k*NIMX * NIMY;
					int left = offset - 1;
					int ztop = offset + NIMX * NIMY;
					int ytop = offset + NIMX;

					float vs = (ccs[left] + ccs[ztop]) / 2;
					float rhos = (crho[left] + crho[ztop]) / 2;

					float mu = rhos*vs*vs;

					float DELTAT_mu = DELTATT * mu;

					float value_dvz_dx = (cvz[offset] - cvz[left]) * ONE_OVER_DELTAXX;
					float value_dvx_dz = (cvx[ztop] - cvx[offset]) * ONE_OVER_DELTAZZ;

					cmemory_dvz_dx[offset] = cb_x[i] * cmemory_dvz_dx[offset] + ca_x[i] * value_dvz_dx;
					cmemory_dvx_dz[offset] = cb_z_half[k] * cmemory_dvx_dz[offset] + ca_z_half[k] * value_dvx_dz;

					value_dvz_dx = value_dvz_dx / cK_x[i] + cmemory_dvz_dx[offset];
					value_dvx_dz = value_dvx_dz / cK_z_half[k] + cmemory_dvx_dz[offset];

					csigmaxz[offset] = DELTAT_mu * (value_dvz_dx + value_dvx_dz) + csigmaxz[offset];
				}
			}

			for (int j = 1; j <= NIMY - 1; j++) {
				for (int i = 1; i <= NIMX; i++) {
					int offset = i + j*NIMX + k*NIMX * NIMY;
					int left = offset - 1;
					int ztop = offset + NIMX * NIMY;
					int ytop = offset + NIMX;

					float vs = (ccs[ytop] + ccs[ztop]) / 2;
					float rhos = (crho[ytop] + crho[ztop]) / 2;

					float mu = rhos*vs*vs;

					float DELTAT_mu = DELTATT * mu;

					float value_dvz_dy = (cvz[ytop] - cvz[offset]) * ONE_OVER_DELTAYY;
					float value_dvy_dz = (cvy[ztop] - cvy[offset]) * ONE_OVER_DELTAZZ;

					cmemory_dvz_dy[offset] = cb_y_half[j] * cmemory_dvz_dy[offset] + ca_y_half[j] * value_dvz_dy;
					cmemory_dvy_dz[offset] = cb_z_half[k] * cmemory_dvy_dz[offset] + ca_z_half[k] * value_dvy_dz;

					value_dvz_dy = value_dvz_dy / cK_y_half[j] + cmemory_dvz_dy[offset];
					value_dvy_dz = value_dvy_dz / cK_z_half[k] + cmemory_dvy_dz[offset];

					csigmayz[offset] = DELTAT_mu * (value_dvz_dy + value_dvy_dz) + csigmayz[offset];
				}
			}
		}

		//vxvy
		for (int k = 2; k <= NIMZ; k++) {
			for (int j = 2; j <= NIMY; j++) {
				for (int i = 2; i <= NIMX; i++) {
					int offset = i + j*NIMX + k*NIMX * NIMY;
					int left = offset - 1;
					int ybottom = offset - NIMX;
					int zbottom = offset - NIMX * NIMY;
					int right = offset + 1;
					int ytop = offset + NIMX;

					float rhos = (crho[offset] + crho[left]) / 2;

					float DELTAT_over_rho = DELTATT / rhos;

					float value_dsigmaxx_dx = (csigmaxx[offset] - csigmaxx[left]) * ONE_OVER_DELTAXX;
					float value_dsigmaxy_dy = (csigmaxy[offset] - csigmaxy[ybottom]) * ONE_OVER_DELTAYY;
					float value_dsigmaxz_dz = (csigmaxz[offset] - csigmaxz[zbottom]) * ONE_OVER_DELTAZZ;

					cmemory_dsigmaxx_dx[offset] = cb_x[i] * cmemory_dsigmaxx_dx[offset] + ca_x[i] * value_dsigmaxx_dx;
					cmemory_dsigmaxy_dy[offset] = cb_y[j] * cmemory_dsigmaxy_dy[offset] + ca_y[j] * value_dsigmaxy_dy;
					cmemory_dsigmaxz_dz[offset] = cb_z[k] * cmemory_dsigmaxz_dz[offset] + ca_z[k] * value_dsigmaxz_dz;

					value_dsigmaxx_dx = value_dsigmaxx_dx / cK_x[i] + cmemory_dsigmaxx_dx[offset];
					value_dsigmaxy_dy = value_dsigmaxy_dy / cK_y[j] + cmemory_dsigmaxy_dy[offset];
					value_dsigmaxz_dz = value_dsigmaxz_dz / cK_z[k] + cmemory_dsigmaxz_dz[offset];

					cvx[offset] = DELTAT_over_rho * (value_dsigmaxx_dx + value_dsigmaxy_dy + value_dsigmaxz_dz) + cvx[offset];
				}
			}

			for (int j = 1; j <= NIMY - 1; j++) {
				for (int i = 1; i <= NIMX - 1; i++) {
					int offset = i + j*NIMX + k*NIMX * NIMY;
					int left = offset - 1;
					int ybottom = offset - NIMX;
					int zbottom = offset - NIMX * NIMY;
					int right = offset + 1;
					int ytop = offset + NIMX;

					float rhos = (crho[offset] + crho[ytop]) / 2;

					float DELTAT_over_rho = DELTATT / rhos;

					float value_dsigmaxy_dx = (csigmaxy[right] - csigmaxy[offset]) * ONE_OVER_DELTAXX;
					float value_dsigmayy_dy = (csigmayy[ytop] - csigmayy[offset]) * ONE_OVER_DELTAYY;
					float value_dsigmayz_dz = (csigmayz[offset] - csigmayz[zbottom]) * ONE_OVER_DELTAZZ;

					cmemory_dsigmaxy_dx[offset] = cb_x_half[i] * cmemory_dsigmaxy_dx[offset] + ca_x_half[i] * value_dsigmaxy_dx;
					cmemory_dsigmayy_dy[offset] = cb_y_half[j] * cmemory_dsigmayy_dy[offset] + ca_y_half[j] * value_dsigmayy_dy;
					cmemory_dsigmayz_dz[offset] = cb_z[k] * cmemory_dsigmayz_dz[offset] + ca_z[k] * value_dsigmayz_dz;

					value_dsigmaxy_dx = value_dsigmaxy_dx / cK_x_half[i] + cmemory_dsigmaxy_dx[offset];
					value_dsigmayy_dy = value_dsigmayy_dy / cK_y_half[j] + cmemory_dsigmayy_dy[offset];
					value_dsigmayz_dz = value_dsigmayz_dz / cK_z[k] + cmemory_dsigmayz_dz[offset];

					cvy[offset] = DELTAT_over_rho * (value_dsigmaxy_dx + value_dsigmayy_dy + value_dsigmayz_dz) + cvy[offset];
				}
			}
		}

		//vz
		for (int k = 1; k <= NIMZ - 1; k++) {
			for (int j = 2; j <= NIMY; j++) {
				for (int i = 1; i <= NIMX - 1; i++) {
					int offset = i + j*NIMX + k*NIMX * NIMY;
					int right = offset + 1;
					int ybottom = offset - NIMX;
					int ztop = offset + NIMX * NIMY;

					float rhos = (crho[offset] + crho[ztop]) / 2;

					float DELTAT_over_rho = DELTATT / rhos;

					float value_dsigmaxz_dx = (csigmaxz[right] - csigmaxz[offset]) * ONE_OVER_DELTAXX;
					float value_dsigmayz_dy = (csigmayz[offset] - csigmayz[ybottom]) * ONE_OVER_DELTAYY;
					float value_dsigmazz_dz = (csigmazz[ztop] - csigmazz[offset]) * ONE_OVER_DELTAZZ;

					cmemory_dsigmaxz_dx[offset] = cb_x_half[i] * cmemory_dsigmaxz_dx[offset] + ca_x_half[i] * value_dsigmaxz_dx;
					cmemory_dsigmayz_dy[offset] = cb_y[j] * cmemory_dsigmayz_dy[offset] + ca_y[j] * value_dsigmayz_dy;
					cmemory_dsigmazz_dz[offset] = cb_z_half[k] * cmemory_dsigmazz_dz[offset] + ca_z_half[k] * value_dsigmazz_dz;

					value_dsigmaxz_dx = value_dsigmaxz_dx / cK_x_half[i] + cmemory_dsigmaxz_dx[offset];
					value_dsigmayz_dy = value_dsigmayz_dy / cK_y[j] + cmemory_dsigmayz_dy[offset];
					value_dsigmazz_dz = value_dsigmazz_dz / cK_z_half[k] + cmemory_dsigmazz_dz[offset];

					cvz[offset] = DELTAT_over_rho * (value_dsigmaxz_dx + value_dsigmayz_dy + value_dsigmazz_dz) + cvz[offset];
				}
			}
		}

		//gathar
		for (int j = 1; j <= NIMY; j++) {
			for (int i = 1; i <= NIMX; i++) {
				int k = KSOURCEE + 1;
				int offset = i + j*NIMX + k*NIMX * NIMY;
				int gxy = i + j*NIMX;
				gatvx[gxy] = cvx[offset];
				gatvy[gxy] = cvy[offset];
				gatvz[gxy] = cvz[offset];
			}
		}

		//addsource
		for (int k = 0; k <= NIMZ; k++) {
			for (int j = 0; j <= NIMY; j++) {
				for (int i = 0; i <= NIMX; i++) {
					if ((i == ISOURCEE) && (j == JSOURCEE) && (k = KSOURCEE)) {
						int offset = i + j*NIMX + k*NIMX * NIMY;
						float lambdaplus2mu = crho[offset] * ccp[offset] * ccp[offset];
						float a = PI * PI * f0 * f0;
						float t = float(it - 1)*DELTATT;
						float source_term = -factor * 2.0*a*(t - t0)*expf(-a*powf((t - t0), 2));
						float force_x = sinf(ANGLE_FORCEE * DEGREES_TO_RADIANSS)*source_term;
						float force_y = cosf(ANGLE_FORCEE * DEGREES_TO_RADIANSS)*source_term;

						/*earthquake event source
						vx[offset] = vx[offset] + force_x*DELTAT[0] / ((rho[offset] + rho[left]) / 2);
						vy[offset] = vy[offset] + force_y*DELTAT[0] / ((rho[offset] + rho[ytop]) / 2);
						*/

						/*explosives source*/
						csigmaxx[offset] = csigmaxx[offset] + force_x*DELTATT * lambdaplus2mu;
						csigmayy[offset] = csigmayy[offset] + force_x*DELTATT * lambdaplus2mu;
						csigmazz[offset] = csigmazz[offset] + force_y*DELTATT * lambdaplus2mu;
					}
				}
			}
		}

		if (fmod(it, sampgat) == 0) {
			char nmfile4[20], nmfile5[20], nmfile6[20];

			int xlen = NIMX / Ngatx;
			int ylen = (NIMY - (2 * NPOINTS_PML)) / Ngaty;
			cout << endl << xlen << " " << ylen;
			//sprintf(nmfile4, "rechorvx.bin");
			//std::ofstream fout4(nmfile4, ios::out | ios::app | ios::binary);
			sprintf_s(nmfile4, "rechorvx.txt");
			std::ofstream fout4;
			fout4.open(nmfile4, ios::app);
			for (int j = 0; j < DIMY; j += ylen) {
				for (int i = 0; i < DIMX; i += xlen) {
					int kk = i + j*NIMX;
					fout4 << gatvx[kk] << " ";
				}
			}fout4 << endl;

			//sprintf(nmfile5, "rechorvy.bin");
			//std::ofstream fout5(nmfile5, ios::out | ios::app | ios::binary);
			sprintf_s(nmfile5, "rechorvy.txt");
			std::ofstream fout5;
			fout5.open(nmfile5, ios::app);
			for (int j = 0; j < DIMY; j += ylen) {
				for (int i = 0; i < DIMX; i += xlen) {
					int kk = i + j*NIMX;
					fout5 << gatvy[kk] << " ";
				}
			}fout5 << endl;

			//sprintf(nmfile6, "rechorvz.bin");
			//std::ofstream fout6(nmfile6, ios::out | ios::app | ios::binary);
			sprintf_s(nmfile6, "rechorvz.txt");
			std::ofstream fout6;
			fout6.open(nmfile6, ios::app);
			for (int j = 0; j < DIMY; j += ylen) {
				for (int i = 0; i < DIMX; i += xlen) {
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
			for (int kk = 0; kk < DIMZ; kk++) {
				for (int jj = 0; jj < DIMY; jj++) {
					for (int ii = 0; ii < DIMX; ii++) {
						int ijk = ii + jj*NIMX + kk*NIMX*NIMY;
						fout1.write((char *)&cvz[ijk], sizeof cvz[ijk]);
					}
				}
			}

			sprintf_s(nmfile2, "vy%05i.bin", it);
			std::ofstream fout2(nmfile2, ios::out | ios::binary);
			//sprintf_s(nmfile2, "vy%05i.txt", it);
			//std::ofstream fout2(nmfile2, ios::out);
			for (int kk = 0; kk < DIMZ; kk++) {
				for (int jj = 0; jj < DIMY; jj++) {
					for (int ii = 0; ii < DIMX; ii++) {
						int ijk = ii + jj*NIMX + kk*NIMX*NIMY;
						fout2.write((char *)&cvy[ijk], sizeof cvy[ijk]);
					}
				}
			}

			sprintf_s(nmfile3, "vx%05i.bin", it);
			std::ofstream fout3(nmfile3, ios::out | ios::binary);
			//sprintf_s(nmfile3, "vx%05i.txt", it);
			//std::ofstream fout3(nmfile3, ios::out);
			for (int kk = 0; kk < DIMZ; kk++) {
				for (int jj = 0; jj < DIMY; jj++) {
					for (int ii = 0; ii < DIMX; ii++) {
						int ijk = ii + jj*NIMX + kk*NIMX*NIMY;
						fout3.write((char *)&cvx[ijk], sizeof cvx[ijk]);
					}
				}
			}

			//save to file END
		}
	}//end it

	free(cvx); free(cvy); free(cvz);
}
