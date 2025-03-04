/** Interpolated densities - Net = sum all of species contributions */

#include <stdio.h>

#include <assert.h>
#include <stdlib.h>

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#include "InterpDensNet.h"
#include <cuda_runtime_api.h>

/** allocated interpolated densities per species */
void interp_dens_net_allocate(struct grid *grd, struct interpDensNet *idn) {

  // charge density defined on nodes and center cell
  idn->rhon = newArr3<FPinterp>(&idn->rhon_flat, grd->nxn, grd->nyn, grd->nzn);
  idn->rhoc = newArr3<FPinterp>(&idn->rhoc_flat, grd->nxc, grd->nyc, grd->nzc);
  // current
  idn->Jx = newArr3<FPinterp>(&idn->Jx_flat, grd->nxn, grd->nyn, grd->nzn);
  idn->Jy = newArr3<FPinterp>(&idn->Jy_flat, grd->nxn, grd->nyn, grd->nzn);
  idn->Jz = newArr3<FPinterp>(&idn->Jz_flat, grd->nxn, grd->nyn, grd->nzn);
  // pressure tensor
  idn->pxx = newArr3<FPinterp>(&idn->pxx_flat, grd->nxn, grd->nyn, grd->nzn);
  idn->pxy = newArr3<FPinterp>(&idn->pxy_flat, grd->nxn, grd->nyn, grd->nzn);
  idn->pxz = newArr3<FPinterp>(&idn->pxz_flat, grd->nxn, grd->nyn, grd->nzn);
  idn->pyy = newArr3<FPinterp>(&idn->pyy_flat, grd->nxn, grd->nyn, grd->nzn);
  idn->pyz = newArr3<FPinterp>(&idn->pyz_flat, grd->nxn, grd->nyn, grd->nzn);
  idn->pzz = newArr3<FPinterp>(&idn->pzz_flat, grd->nxn, grd->nyn, grd->nzn);
}

/** deallocate interpolated densities per species */
void interp_dens_net_deallocate(struct grid *grd, struct interpDensNet *idn) {

  // charge density
  delArr3(idn->rhon, grd->nxn, grd->nyn);
  delArr3(idn->rhoc, grd->nxc, grd->nyc);
  // current
  delArr3(idn->Jx, grd->nxn, grd->nyn);
  delArr3(idn->Jy, grd->nxn, grd->nyn);
  delArr3(idn->Jz, grd->nxn, grd->nyn);
  // pressure
  delArr3(idn->pxx, grd->nxn, grd->nyn);
  delArr3(idn->pxy, grd->nxn, grd->nyn);
  delArr3(idn->pxz, grd->nxn, grd->nyn);
  delArr3(idn->pyy, grd->nxn, grd->nyn);
  delArr3(idn->pyz, grd->nxn, grd->nyn);
  delArr3(idn->pzz, grd->nxn, grd->nyn);
}

/** set all the densities to zero */
void setZeroDensities(struct interpDensNet *idn, struct interpDensSpecies *ids,
                      struct grid *grd, int ns) {

  //////////////////////////////////////
  // Net densities
  // calculate the coordinates - Nodes
  for (int i = 0; i < grd->nxn; i++)
    for (int j = 0; j < grd->nyn; j++)
      for (int k = 0; k < grd->nzn; k++) {

        // charge density
        idn->rhon[i][j][k] = 0.0; // quantities defined on node
        // current
        idn->Jx[i][j][k] = 0.0; // quantities defined on node
        idn->Jy[i][j][k] = 0.0; // quantities defined on node
        idn->Jz[i][j][k] = 0.0; // quantities defined on node
        // pressure
        idn->pxx[i][j][k] = 0.0; // quantities defined on node
        idn->pxy[i][j][k] = 0.0; // quantities defined on node
        idn->pxz[i][j][k] = 0.0; // quantities defined on node
        idn->pyy[i][j][k] = 0.0; // quantities defined on node
        idn->pyz[i][j][k] = 0.0; // quantities defined on node
        idn->pzz[i][j][k] = 0.0; // quantities defined on node
      }

  // center cell rhoc
  for (int i = 0; i < grd->nxc; i++)
    for (int j = 0; j < grd->nyc; j++)
      for (int k = 0; k < grd->nzc; k++) {

        idn->rhoc[i][j][k] = 0.0; // quantities defined on center cells
      }

  //////////////////////////////////
  // Densities per species
  for (int is = 0; is < ns; is++)
    for (int i = 0; i < grd->nxn; i++)
      for (int j = 0; j < grd->nyn; j++)
        for (int k = 0; k < grd->nzn; k++) {

          // charge density
          ids[is].rhon[i][j][k] = 0.0; // quantities defined on node
          // current
          ids[is].Jx[i][j][k] = 0.0; // quantities defined on node
          ids[is].Jy[i][j][k] = 0.0; // quantities defined on node
          ids[is].Jz[i][j][k] = 0.0; // quantities defined on node
          // pressure
          ids[is].pxx[i][j][k] = 0.0; // quantities defined on node
          ids[is].pxy[i][j][k] = 0.0; // quantities defined on node
          ids[is].pxz[i][j][k] = 0.0; // quantities defined on node
          ids[is].pyy[i][j][k] = 0.0; // quantities defined on node
          ids[is].pyz[i][j][k] = 0.0; // quantities defined on node
          ids[is].pzz[i][j][k] = 0.0; // quantities defined on node
        }

  //////////////////////////////////
  //  rhoc  - center cell
  for (int is = 0; is < ns; is++)
    for (int i = 0; i < grd->nxc; i++)
      for (int j = 0; j < grd->nyc; j++)
        for (int k = 0; k < grd->nzc; k++) {
          ids[is].rhoc[i][j][k] = 0.0;
        }
}

__global__ void sum_kernel(struct interpDensNet *idn,
                           struct interpDensSpecies *ids, int nxn, int nyn,
                           int nzn) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int totalNodes = nxn * nyn * nzn;

  // Set values for quantities defined on nodes
  if (index < totalNodes) {
    atomicAdd(&idn->rhon_flat[index], ids->rhon_flat[index]);
  }
}

void sumOverSpeciesGPU(struct interpDensNet *idn, struct interpDensSpecies *ids,
                       struct grid *grd, int ns) {
  int threadsPerBlock = 256;
  int blocksPerGrid =
      (grd->nxn * grd->nyn * grd->nzn + threadsPerBlock - 1) / threadsPerBlock;

  for (int is = 0; is < ns; is++) {
    sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(idn, &ids[is], grd->nxn,
                                                   grd->nyn, grd->nzn);
  }
  cudaDeviceSynchronize();
}

/** set all the densities to zero */
void sumOverSpecies(struct interpDensNet *idn, struct interpDensSpecies *ids,
                    struct grid *grd, int ns) {
  for (int is = 0; is < ns; is++)
    for (register int i = 0; i < grd->nxn; i++)
      for (register int j = 0; j < grd->nyn; j++)
        for (register int k = 0; k < grd->nzn; k++) {

          // density
          idn->rhon[i][j][k] += ids[is].rhon[i][j][k];

          // These are not really needed for the algoritm
          // They might needed for the algorithm
          // J
          idn->Jx[i][j][k] += ids[is].Jx[i][j][k];
          idn->Jy[i][j][k] += ids[is].Jy[i][j][k];
          idn->Jz[i][j][k] += ids[is].Jz[i][j][k];
          // pressure
          idn->pxx[i][j][k] += ids[is].pxx[i][j][k];
          idn->pxy[i][j][k] += ids[is].pxy[i][j][k];
          idn->pxz[i][j][k] += ids[is].pxz[i][j][k];
          idn->pyy[i][j][k] += ids[is].pyy[i][j][k];
          idn->pyz[i][j][k] += ids[is].pyz[i][j][k];
          idn->pzz[i][j][k] += ids[is].pzz[i][j][k];
        }
}
