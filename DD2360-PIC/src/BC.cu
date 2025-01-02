#include "Alloc.h"
#include "BC.h"
#include "InterpDensNet.h"

//////////
// POPULATE GHOST CELL ON NODES
//////////

__global__ void face_X(interpDensNet *idn, int nxn, int nyn, int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (j < nyn - 1 && k < nzn - 1) {
    idn->rhon_flat[get_idx(0, j, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(nxn - 3, j, k, nyn, nzn)];
    idn->rhon_flat[get_idx(nxn - 1, j, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(2, j, k, nyn, nzn)];
  }
}

__global__ void face_Y(interpDensNet *idn, int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i < nxn - 1 && k < nzn - 1) {
    idn->rhon_flat[get_idx(i, 0, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, nyn - 3, k, nyn, nzn)];
    idn->rhon_flat[get_idx(i, nyn - 1, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, 2, k, nyn, nzn)];
  }
}

__global__ void face_Z(interpDensNet *idn, int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i < nxn - 1 && j < nyn - 1) {
    idn->rhon_flat[get_idx(i, j, 0, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, j, nzn - 3, nyn, nzn)];
    idn->rhon_flat[get_idx(i, j, nzn - 1, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, j, 2, nyn, nzn)];
  }
}

__global__ void edge_X(interpDensNet *idn, int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i < nxn - 1) {
    idn->rhon_flat[get_idx(i, nyn - 1, nzn - 1, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, 2, 2, nyn, nzn)];
    idn->rhon_flat[get_idx(i, 0, 0, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, nyn - 3, nzn - 3, nyn, nzn)];
    idn->rhon_flat[get_idx(i, 0, nzn - 1, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, nyn - 3, 2, nyn, nzn)];
    idn->rhon_flat[get_idx(i, nyn - 1, 0, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, 2, nzn - 3, nyn, nzn)];
  }
}

__global__ void edge_Y(interpDensNet *idn, int nxn, int nyn, int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (j < nyn - 1) {
    idn->rhon_flat[get_idx(nxn - 1, j, nzn - 1, nyn, nzn)] =
        idn->rhon_flat[get_idx(2, j, 2, nyn, nzn)];
    idn->rhon_flat[get_idx(0, j, 0, nyn, nzn)] =
        idn->rhon_flat[get_idx(nxn - 3, j, nzn - 3, nyn, nzn)];
    idn->rhon_flat[get_idx(0, j, nzn - 1, nyn, nzn)] =
        idn->rhon_flat[get_idx(nxn - 3, j, 2, nyn, nzn)];
    idn->rhon_flat[get_idx(nxn - 1, j, 0, nyn, nzn)] =
        idn->rhon_flat[get_idx(2, j, nzn - 3, nyn, nzn)];
  }
}

__global__ void edge_Z(interpDensNet *idn, int nxn, int nyn, int nzn) {
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (k < nzn - 1) {
    idn->rhon_flat[get_idx(nxn - 1, nyn - 1, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(2, 2, k, nyn, nzn)];
    idn->rhon_flat[get_idx(0, 0, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(nxn - 3, nyn - 3, k, nyn, nzn)];
    idn->rhon_flat[get_idx(nxn - 1, 0, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(2, nyn - 3, k, nyn, nzn)];
    idn->rhon_flat[get_idx(0, nyn - 1, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(nxn - 3, 2, k, nyn, nzn)];
  }
}

__global__ void corners(interpDensNet *idn, int nxn, int nyn, int nzn) {
  idn->rhon_flat[get_idx(nxn - 1, nyn - 1, nzn - 1, nyn, nzn)] =
      idn->rhon_flat[get_idx(2, 2, 2, nyn, nzn)];
  idn->rhon_flat[get_idx(0, nyn - 1, nzn - 1, nyn, nzn)] =
      idn->rhon_flat[get_idx(nxn - 3, 2, 2, nyn, nzn)];
  idn->rhon_flat[get_idx(nxn - 1, 0, nzn - 1, nyn, nzn)] =
      idn->rhon_flat[get_idx(2, nyn - 3, 2, nyn, nzn)];
  idn->rhon_flat[get_idx(0, 0, nzn - 1, nyn, nzn)] =
      idn->rhon_flat[get_idx(nxn - 3, nyn - 3, 2, nyn, nzn)];
  idn->rhon_flat[get_idx(nxn - 1, nyn - 1, 0, nyn, nzn)] =
      idn->rhon_flat[get_idx(2, 2, nzn - 3, nyn, nzn)];
  idn->rhon_flat[get_idx(0, nyn - 1, 0, nyn, nzn)] =
      idn->rhon_flat[get_idx(nxn - 3, 2, nzn - 3, nyn, nzn)];
  idn->rhon_flat[get_idx(nxn - 1, 0, 0, nyn, nzn)] =
      idn->rhon_flat[get_idx(2, nyn - 3, nzn - 3, nyn, nzn)];
  idn->rhon_flat[get_idx(0, 0, 0, nyn, nzn)] =
      idn->rhon_flat[get_idx(nxn - 3, nyn - 3, nzn - 3, nyn, nzn)];
}

__global__ void face_XF(interpDensNet *idn, int nxn, int nyn, int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j < nyn - 1 && k < nzn - 1) {
    idn->rhon_flat[get_idx(0, j, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(nxn - 3, j, k, nyn, nzn)];
    idn->rhon_flat[get_idx(nxn - 1, j, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(2, j, k, nyn, nzn)];
  }
}

__global__ void face_YF(interpDensNet *idn, int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i < nxn - 1 && k < nzn - 1) {
    idn->rhon_flat[get_idx(i, 0, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, nyn - 3, k, nyn, nzn)];
    idn->rhon_flat[get_idx(i, nyn - 1, k, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, 2, k, nyn, nzn)];
  }
}

__global__ void face_ZF(interpDensNet *idn, int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i < nxn - 1 && j < nyn - 1) {
    idn->rhon_flat[get_idx(i, j, 0, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, j, nzn - 3, nyn, nzn)];
    idn->rhon_flat[get_idx(i, j, nzn - 1, nyn, nzn)] =
        idn->rhon_flat[get_idx(i, j, 2, nyn, nzn)];
  }
}

/** Apply BC to scalar interp quantity defined on nodes - Interpolation quantity
 */
void applyBCscalarDensNGPU(struct interpDensNet *d_idn, grid *grd,
                           parameters *param) {

  ///////////////////////
  ///
  ///    FACE
  ///

  int threads_per_block = 256;
  int x_blocks = (grd->nyn - 2 + threads_per_block - 1) / threads_per_block;
  int y_blocks = (grd->nxn - 2 + threads_per_block - 1) / threads_per_block;
  int z_blocks = (grd->nxn - 2 + threads_per_block - 1) / threads_per_block;

  dim3 gridX(y_blocks, z_blocks);
  dim3 gridY(x_blocks, z_blocks);
  dim3 gridZ(x_blocks, y_blocks);

  // X direction
  if (param->PERIODICX == true) {
    face_X<<<gridX, threads_per_block>>>(d_idn, grd->nxn, grd->nyn, grd->nzn);
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == true) {
    face_Y<<<gridY, threads_per_block>>>(d_idn, grd->nxn, grd->nyn, grd->nzn);
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == true) {
    face_Z<<<gridZ, threads_per_block>>>(d_idn, grd->nxn, grd->nyn, grd->nzn);
  }

  ///////////////////////
  ///
  ///    EDGES
  ///

  // X-EDGE
  if (param->PERIODICY == true || param->PERIODICZ == true) {
    edge_X<<<x_blocks, threads_per_block>>>(d_idn, grd->nxn, grd->nyn,
                                            grd->nzn);
  }

  // Y-EDGE
  if (param->PERIODICX == true || param->PERIODICZ == true) {
    edge_Y<<<y_blocks, threads_per_block>>>(d_idn, grd->nxn, grd->nyn,
                                            grd->nzn);
  }
  // Z-EDGE
  if (param->PERIODICX == true || param->PERIODICY == true) {
    edge_Z<<<z_blocks, threads_per_block>>>(d_idn, grd->nxn, grd->nyn,
                                            grd->nzn);
  }
  // Corners
  if (param->PERIODICX == true || param->PERIODICY == true ||
      param->PERIODICZ == true) {
    corners<<<1, 1>>>(d_idn, grd->nxn, grd->nyn, grd->nzn);
  }
  // FACE NON PERIODIC
  // X direction
  if (param->PERIODICX == false) {
    face_XF<<<gridX, threads_per_block>>>(d_idn, grd->nxn, grd->nyn, grd->nzn);
  }
  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == false) {
    face_YF<<<gridY, threads_per_block>>>(d_idn, grd->nxn, grd->nyn, grd->nzn);
  }
  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == false) {
    face_ZF<<<gridZ, threads_per_block>>>(d_idn, grd->nxn, grd->nyn, grd->nzn);
  }
  cudaDeviceSynchronize();
}

/** Apply BC to scalar interp quantity defined on nodes - Interpolation quantity
 */
void applyBCscalarDensN(FPinterp ***scalarN, grid *grd, parameters *param) {

  ///////////////////////
  ///
  ///    FACE
  ///

  // X direction
  if (param->PERIODICX == true) {
    for (register int j = 1; j < grd->nyn - 1; j++)
      for (register int k = 1; k < grd->nzn - 1; k++) {
        scalarN[0][j][k] = scalarN[grd->nxn - 3][j][k];
        scalarN[grd->nxn - 1][j][k] = scalarN[2][j][k];
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == true) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int k = 1; k < grd->nzn - 1; k++) {

        // rhon
        scalarN[i][0][k] = scalarN[i][grd->nyn - 3][k];
        scalarN[i][grd->nyn - 1][k] = scalarN[i][2][k];
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == true) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int j = 1; j < grd->nyn - 1; j++) {

        // rhon
        scalarN[i][j][0] = scalarN[i][j][grd->nzn - 3];
        scalarN[i][j][grd->nzn - 1] = scalarN[i][j][2];
      }
  }

  ///////////////////////
  ///
  ///    EDGES
  ///

  // X-EDGE
  if (param->PERIODICY == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nxn - 1); i++) {
      scalarN[i][grd->nyn - 1][grd->nzn - 1] = scalarN[i][2][2];
      scalarN[i][0][0] = scalarN[i][grd->nyn - 3][grd->nzn - 3];
      scalarN[i][0][grd->nzn - 1] = scalarN[i][grd->nyn - 3][2];
      scalarN[i][grd->nyn - 1][0] = scalarN[i][2][grd->nzn - 3];
    }
  }

  // Y-EDGE
  if (param->PERIODICX == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nyn - 1); i++) {
      scalarN[grd->nxn - 1][i][grd->nzn - 1] = scalarN[2][i][2];
      scalarN[0][i][0] = scalarN[grd->nxn - 3][i][grd->nzn - 3];
      scalarN[0][i][grd->nzn - 1] = scalarN[grd->nxn - 3][i][2];
      scalarN[grd->nxn - 1][i][0] = scalarN[2][i][grd->nzn - 3];
    }
  }

  // Z-EDGE
  if (param->PERIODICX == true || param->PERIODICY == true) {
    for (register int i = 1; i < (grd->nzn - 1); i++) {
      scalarN[grd->nxn - 1][grd->nyn - 1][i] = scalarN[2][2][i];
      scalarN[0][0][i] = scalarN[grd->nxn - 3][grd->nyn - 3][i];
      scalarN[grd->nxn - 1][0][i] = scalarN[2][grd->nyn - 3][i];
      scalarN[0][grd->nyn - 1][i] = scalarN[grd->nxn - 3][2][i];
    }
  }

  // Corners
  if (param->PERIODICX == true || param->PERIODICY == true ||
      param->PERIODICZ == true) {

    scalarN[grd->nxn - 1][grd->nyn - 1][grd->nzn - 1] = scalarN[2][2][2];
    scalarN[0][grd->nyn - 1][grd->nzn - 1] = scalarN[grd->nxn - 3][2][2];
    scalarN[grd->nxn - 1][0][grd->nzn - 1] = scalarN[2][grd->nyn - 3][2];
    scalarN[0][0][grd->nzn - 1] = scalarN[grd->nxn - 3][grd->nyn - 3][2];
    scalarN[grd->nxn - 1][grd->nyn - 1][0] = scalarN[2][2][grd->nzn - 3];
    scalarN[0][grd->nyn - 1][0] = scalarN[grd->nxn - 3][2][grd->nzn - 3];
    scalarN[grd->nxn - 1][0][0] = scalarN[2][grd->nyn - 3][grd->nzn - 3];
    scalarN[0][0][0] = scalarN[grd->nxn - 3][grd->nyn - 3][grd->nzn - 3];
  }

  // FACE NON PERIODIC
  // X direction
  if (param->PERIODICX == false) {
    for (register int j = 1; j < grd->nyn - 1; j++)
      for (register int k = 1; k < grd->nzn - 1; k++) {
        scalarN[0][j][k] = scalarN[1][j][k];
        scalarN[grd->nxn - 1][j][k] = scalarN[grd->nxn - 2][j][k];
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == false) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int k = 1; k < grd->nzn - 1; k++) {
        scalarN[i][0][k] = scalarN[i][1][k];
        scalarN[i][grd->nyn - 1][k] = scalarN[i][grd->nyn - 2][k];
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == false) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int j = 1; j < grd->nyn - 1; j++) {
        scalarN[i][j][0] = scalarN[i][j][1];
        scalarN[i][j][grd->nzn - 1] = scalarN[i][j][grd->nzn - 2];
      }
  }
}

/** Apply BC to scalar interp quantity defined on nodes - Interpolation quantity
 */
void applyBCscalarFieldN(FPfield ***scalarN, grid *grd, parameters *param) {

  ///////////////////////
  ///
  ///    FACE
  ///

  // X direction
  if (param->PERIODICX == true) {
    for (register int j = 1; j < grd->nyn - 1; j++)
      for (register int k = 1; k < grd->nzn - 1; k++) {
        scalarN[0][j][k] = scalarN[grd->nxn - 3][j][k];
        scalarN[grd->nxn - 1][j][k] = scalarN[2][j][k];
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == true) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int k = 1; k < grd->nzn - 1; k++) {

        // rhon
        scalarN[i][0][k] = scalarN[i][grd->nyn - 3][k];
        scalarN[i][grd->nyn - 1][k] = scalarN[i][2][k];
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == true) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int j = 1; j < grd->nyn - 1; j++) {

        // rhon
        scalarN[i][j][0] = scalarN[i][j][grd->nzn - 3];
        scalarN[i][j][grd->nzn - 1] = scalarN[i][j][2];
      }
  }

  ///////////////////////
  ///
  ///    EDGES
  ///

  // X-EDGE
  if (param->PERIODICY == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nxn - 1); i++) {
      scalarN[i][grd->nyn - 1][grd->nzn - 1] = scalarN[i][2][2];
      scalarN[i][0][0] = scalarN[i][grd->nyn - 3][grd->nzn - 3];
      scalarN[i][0][grd->nzn - 1] = scalarN[i][grd->nyn - 3][2];
      scalarN[i][grd->nyn - 1][0] = scalarN[i][2][grd->nzn - 3];
    }
  }

  // Y-EDGE
  if (param->PERIODICX == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nyn - 1); i++) {
      scalarN[grd->nxn - 1][i][grd->nzn - 1] = scalarN[2][i][2];
      scalarN[0][i][0] = scalarN[grd->nxn - 3][i][grd->nzn - 3];
      scalarN[0][i][grd->nzn - 1] = scalarN[grd->nxn - 3][i][2];
      scalarN[grd->nxn - 1][i][0] = scalarN[2][i][grd->nzn - 3];
    }
  }

  // Z-EDGE
  if (param->PERIODICX == true || param->PERIODICY == true) {
    for (register int i = 1; i < (grd->nzn - 1); i++) {
      scalarN[grd->nxn - 1][grd->nyn - 1][i] = scalarN[2][2][i];
      scalarN[0][0][i] = scalarN[grd->nxn - 3][grd->nyn - 3][i];
      scalarN[grd->nxn - 1][0][i] = scalarN[2][grd->nyn - 3][i];
      scalarN[0][grd->nyn - 1][i] = scalarN[grd->nxn - 3][2][i];
    }
  }

  // Corners
  if (param->PERIODICX == true || param->PERIODICY == true ||
      param->PERIODICZ == true) {

    scalarN[grd->nxn - 1][grd->nyn - 1][grd->nzn - 1] = scalarN[2][2][2];
    scalarN[0][grd->nyn - 1][grd->nzn - 1] = scalarN[grd->nxn - 3][2][2];
    scalarN[grd->nxn - 1][0][grd->nzn - 1] = scalarN[2][grd->nyn - 3][2];
    scalarN[0][0][grd->nzn - 1] = scalarN[grd->nxn - 3][grd->nyn - 3][2];
    scalarN[grd->nxn - 1][grd->nyn - 1][0] = scalarN[2][2][grd->nzn - 3];
    scalarN[0][grd->nyn - 1][0] = scalarN[grd->nxn - 3][2][grd->nzn - 3];
    scalarN[grd->nxn - 1][0][0] = scalarN[2][grd->nyn - 3][grd->nzn - 3];
    scalarN[0][0][0] = scalarN[grd->nxn - 3][grd->nyn - 3][grd->nzn - 3];
  }

  // FACE NON PERIODIC
  // X direction
  if (param->PERIODICX == false) {
    for (register int j = 1; j < grd->nyn - 1; j++)
      for (register int k = 1; k < grd->nzn - 1; k++) {
        scalarN[0][j][k] = scalarN[1][j][k];
        scalarN[grd->nxn - 1][j][k] = scalarN[grd->nxn - 2][j][k];
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == false) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int k = 1; k < grd->nzn - 1; k++) {
        scalarN[i][0][k] = scalarN[i][1][k];
        scalarN[i][grd->nyn - 1][k] = scalarN[i][grd->nyn - 2][k];
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == false) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int j = 1; j < grd->nyn - 1; j++) {
        scalarN[i][j][0] = scalarN[i][j][1];
        scalarN[i][j][grd->nzn - 1] = scalarN[i][j][grd->nzn - 2];
      }
  }
}

///////// USE THIS TO IMPOSE BC TO ELECTRIC FIELD
///////// NOW THIS IS FIXED TO ZERO

/** Apply BC to scalar interp quantity defined on nodes - Interpolation quantity
 */
void applyBCscalarFieldNzero(FPfield ***scalarN, grid *grd, parameters *param) {

  ///////////////////////
  ///
  ///    FACE
  ///

  // X direction
  if (param->PERIODICX == true) {
    for (register int j = 1; j < grd->nyn - 1; j++)
      for (register int k = 1; k < grd->nzn - 1; k++) {
        scalarN[0][j][k] = scalarN[grd->nxn - 3][j][k];
        scalarN[grd->nxn - 1][j][k] = scalarN[2][j][k];
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == true) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int k = 1; k < grd->nzn - 1; k++) {

        // rhon
        scalarN[i][0][k] = scalarN[i][grd->nyn - 3][k];
        scalarN[i][grd->nyn - 1][k] = scalarN[i][2][k];
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == true) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int j = 1; j < grd->nyn - 1; j++) {

        // rhon
        scalarN[i][j][0] = scalarN[i][j][grd->nzn - 3];
        scalarN[i][j][grd->nzn - 1] = scalarN[i][j][2];
      }
  }

  ///////////////////////
  ///
  ///    EDGES
  ///

  // X-EDGE
  if (param->PERIODICY == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nxn - 1); i++) {
      scalarN[i][grd->nyn - 1][grd->nzn - 1] = scalarN[i][2][2];
      scalarN[i][0][0] = scalarN[i][grd->nyn - 3][grd->nzn - 3];
      scalarN[i][0][grd->nzn - 1] = scalarN[i][grd->nyn - 3][2];
      scalarN[i][grd->nyn - 1][0] = scalarN[i][2][grd->nzn - 3];
    }
  }

  // Y-EDGE
  if (param->PERIODICX == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nyn - 1); i++) {
      scalarN[grd->nxn - 1][i][grd->nzn - 1] = scalarN[2][i][2];
      scalarN[0][i][0] = scalarN[grd->nxn - 3][i][grd->nzn - 3];
      scalarN[0][i][grd->nzn - 1] = scalarN[grd->nxn - 3][i][2];
      scalarN[grd->nxn - 1][i][0] = scalarN[2][i][grd->nzn - 3];
    }
  }

  // Z-EDGE
  if (param->PERIODICX == true || param->PERIODICY == true) {
    for (register int i = 1; i < (grd->nzn - 1); i++) {
      scalarN[grd->nxn - 1][grd->nyn - 1][i] = scalarN[2][2][i];
      scalarN[0][0][i] = scalarN[grd->nxn - 3][grd->nyn - 3][i];
      scalarN[grd->nxn - 1][0][i] = scalarN[2][grd->nyn - 3][i];
      scalarN[0][grd->nyn - 1][i] = scalarN[grd->nxn - 3][2][i];
    }
  }

  // Corners
  if (param->PERIODICX == true || param->PERIODICY == true ||
      param->PERIODICZ == true) {

    scalarN[grd->nxn - 1][grd->nyn - 1][grd->nzn - 1] = scalarN[2][2][2];
    scalarN[0][grd->nyn - 1][grd->nzn - 1] = scalarN[grd->nxn - 3][2][2];
    scalarN[grd->nxn - 1][0][grd->nzn - 1] = scalarN[2][grd->nyn - 3][2];
    scalarN[0][0][grd->nzn - 1] = scalarN[grd->nxn - 3][grd->nyn - 3][2];
    scalarN[grd->nxn - 1][grd->nyn - 1][0] = scalarN[2][2][grd->nzn - 3];
    scalarN[0][grd->nyn - 1][0] = scalarN[grd->nxn - 3][2][grd->nzn - 3];
    scalarN[grd->nxn - 1][0][0] = scalarN[2][grd->nyn - 3][grd->nzn - 3];
    scalarN[0][0][0] = scalarN[grd->nxn - 3][grd->nyn - 3][grd->nzn - 3];
  }

  // FACE NON PERIODIC
  // X direction
  if (param->PERIODICX == false) {
    for (register int j = 1; j < grd->nyn - 1; j++)
      for (register int k = 1; k < grd->nzn - 1; k++) {
        scalarN[0][j][k] = 0.0;
        scalarN[grd->nxn - 1][j][k] = 0.0;
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == false) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int k = 1; k < grd->nzn - 1; k++) {
        scalarN[i][0][k] = 0.0;
        scalarN[i][grd->nyn - 1][k] = 0.0;
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == false) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int j = 1; j < grd->nyn - 1; j++) {
        scalarN[i][j][0] = 0.0;
        scalarN[i][j][grd->nzn - 1] = 0.0;
      }
  }
}

///////////////
////
////    add Densities
////
////
///////////////

__global__ void applyBC_X_kernel(struct interpDensSpecies *ids, int nxn,
                                 int nyn, int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (j < nyn - 1 && k < nzn - 1) {
    // rhon
    ids->rhon_flat[get_idx(1, j, k, nyn, nzn)] +=
        ids->rhon_flat[get_idx(nxn - 2, j, k, nyn, nzn)];
    ids->rhon_flat[get_idx(nxn - 2, j, k, nyn, nzn)] =
        ids->rhon_flat[get_idx(1, j, k, nyn, nzn)];
  }
}

__global__ void applyBC_Y_kernel(struct interpDensSpecies *ids, int nxn,
                                 int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // Index in the X direction
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1; // Index in the Z direction

  if (i < nxn - 1 && k < nzn - 1) { // Ensuring within bounds

    // rhon
    ids->rhon_flat[get_idx(i, 1, k, nyn, nzn)] +=
        ids->rhon_flat[get_idx(i, nyn - 2, k, nyn, nzn)];
    ids->rhon_flat[get_idx(i, nyn - 2, k, nyn, nzn)] =
        ids->rhon_flat[get_idx(i, 1, k, nyn, nzn)];
  }
}

__global__ void applyBC_Z_kernel(struct interpDensSpecies *ids, int nxn,
                                 int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // Index in the X direction
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // Index in the Y direction

  if (i < nxn - 1 && j < nyn - 1) { // Ensuring within bounds

    // rhon
    ids->rhon_flat[get_idx(i, j, 1, nyn, nzn)] +=
        ids->rhon_flat[get_idx(i, j, nzn - 2, nyn, nzn)];
    ids->rhon_flat[get_idx(i, j, nzn - 2, nyn, nzn)] =
        ids->rhon_flat[get_idx(i, j, 1, nyn, nzn)];
  }
}

__global__ void applyBC_XF_kernel(struct interpDensSpecies *ids, int nxn,
                                  int nyn, int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // Index in the Y direction
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1; // Index in the Z direction

  if (j < nyn - 1 && k < nzn - 1) { // Ensure within bounds

    // Multiply by 2 at the X boundaries (1 and nxn-2)
    // rhon
    ids->rhon_flat[get_idx(1, j, k, nyn, nzn)] *= 2;
    ids->rhon_flat[get_idx(nxn - 2, j, k, nyn, nzn)] *= 2;
  }
}

__global__ void applyBC_YF_kernel(struct interpDensSpecies *ids, int nxn,
                                  int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // Index in the X direction
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1; // Index in the Z direction

  if (i < nxn - 1 && k < nzn - 1) { // Ensure within bounds

    // Multiply by 2 at the Y boundaries (1 and nyn-2)
    // rhon
    ids->rhon_flat[get_idx(i, 1, k, nyn, nzn)] *= 2;
    ids->rhon_flat[get_idx(i, nyn - 2, k, nyn, nzn)] *= 2;
  }
}

__global__ void applyBC_ZF_kernel(struct interpDensSpecies *ids, int nxn,
                                  int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // Index in the X direction
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // Index in the Y direction

  if (i < nxn - 1 && j < nyn - 1) { // Ensure within bounds

    // Multiply by 2 at the Z boundaries (1 and nzn-2)
    // rhon
    ids->rhon_flat[get_idx(i, j, 1, nyn, nzn)] *= 2;
    ids->rhon_flat[get_idx(i, j, nzn - 2, nyn, nzn)] *= 2;
  }
}

// apply boundary conditions to species interpolated densities GPU
void applyBCidsGPU(struct interpDensSpecies *ids, struct grid *grd,
                   struct parameters *param) {

  /////////////////(///
  // apply BC on X
  /////
  dim3 nthreads(16, 16);
  dim3 nblocksXY[param->ns];
  dim3 nblocksYZ[param->ns];
  dim3 nblocksXZ[param->ns];

  for (int is = 0; is < param->ns; is++) {
    nblocksXY[is] = dim3((grd->nyn + nthreads.x - 1) / nthreads.x,
                         (grd->nzn + nthreads.y - 1) / nthreads.y);
    nblocksYZ[is] = dim3((grd->nxn + nthreads.x - 1) / nthreads.x,
                         (grd->nzn + nthreads.y - 1) / nthreads.y);
    nblocksXZ[is] = dim3((grd->nxn + nthreads.x - 1) / nthreads.x,
                         (grd->nyn + nthreads.y - 1) / nthreads.y);
  }

  for (int is = 0; is < param->ns; is++) {
    // X direction
    if (param->PERIODICX == true)
      applyBC_X_kernel<<<nblocksXY[is], nthreads>>>(&ids[is], grd->nxn,
                                                    grd->nyn, grd->nzn);
    // end of periodic in X direction
    else
      applyBC_XF_kernel<<<nblocksXY[is], nthreads>>>(&ids[is], grd->nxn,
                                                     grd->nyn, grd->nzn);
    // Periodic boundary conditions in Y direction
    if (param->PERIODICY == true)
      applyBC_Y_kernel<<<nblocksYZ[is], nthreads>>>(&ids[is], grd->nxn,
                                                    grd->nyn, grd->nzn);
    // end of PERIODICY
    else
      applyBC_YF_kernel<<<nblocksXZ[is], nthreads>>>(&ids[is], grd->nxn,
                                                     grd->nyn, grd->nzn);

    // Periodic boundary conditions in Z direction
    if (param->PERIODICZ == true)
      applyBC_Z_kernel<<<nblocksXZ[is], nthreads>>>(&ids[is], grd->nxn,
                                                    grd->nyn, grd->nzn);
    else
      applyBC_ZF_kernel<<<nblocksYZ[is], nthreads>>>(&ids[is], grd->nxn,
                                                     grd->nyn, grd->nzn);
  }

  cudaDeviceSynchronize();
}

// apply boundary conditions to species interpolated densities
void applyBCids(struct interpDensSpecies *ids, struct grid *grd,
                struct parameters *param) {

  /////////////////(///
  // apply BC on X
  /////

  // X direction
  if (param->PERIODICX == true) {
    for (register int j = 1; j < grd->nyn - 1; j++)
      for (register int k = 1; k < grd->nzn - 1; k++) {
        // rhon
        ids->rhon[1][j][k] += ids->rhon[grd->nxn - 2][j][k];
        ids->rhon[grd->nxn - 2][j][k] =
            ids->rhon[1][j][k]; // second is = not +=

        // Jx
        ids->Jx[1][j][k] += ids->Jx[grd->nxn - 2][j][k];
        ids->Jx[grd->nxn - 2][j][k] = ids->Jx[1][j][k]; // second is = not +=
        // Jy
        ids->Jy[1][j][k] += ids->Jy[grd->nxn - 2][j][k];
        ids->Jy[grd->nxn - 2][j][k] = ids->Jy[1][j][k]; // second is = not +=
        // Jz
        ids->Jz[1][j][k] += ids->Jz[grd->nxn - 2][j][k];
        ids->Jz[grd->nxn - 2][j][k] = ids->Jz[1][j][k]; // second is = not +=

        // pxx
        ids->pxx[1][j][k] += ids->pxx[grd->nxn - 2][j][k];
        ids->pxx[grd->nxn - 2][j][k] = ids->pxx[1][j][k]; // second is = not +=
        // pxy
        ids->pxy[1][j][k] += ids->pxy[grd->nxn - 2][j][k];
        ids->pxy[grd->nxn - 2][j][k] = ids->pxy[1][j][k]; // second is = not +=
        // pxz
        ids->pxz[1][j][k] += ids->pxz[grd->nxn - 2][j][k];
        ids->pxz[grd->nxn - 2][j][k] = ids->pxz[1][j][k]; // second is = not +=

        // pyy
        ids->pyy[1][j][k] += ids->pyy[grd->nxn - 2][j][k];
        ids->pyy[grd->nxn - 2][j][k] = ids->pyy[1][j][k]; // second is = not +=
        // pyz
        ids->pyz[1][j][k] += ids->pyz[grd->nxn - 2][j][k];
        ids->pyz[grd->nxn - 2][j][k] = ids->pyz[1][j][k]; // second is = not +=
        // pzz
        ids->pzz[1][j][k] += ids->pzz[grd->nxn - 2][j][k];
        ids->pzz[grd->nxn - 2][j][k] = ids->pzz[1][j][k]; // second is = not +=

      } // end of loop over the grid
  } // end of periodic in X direction

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == true) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int k = 1; k < grd->nzn - 1; k++) {

        // rhon
        ids->rhon[i][1][k] += ids->rhon[i][grd->nyn - 2][k];
        ids->rhon[i][grd->nyn - 2][k] =
            ids->rhon[i][1][k]; // second is = not +=

        // Jx
        ids->Jx[i][1][k] += ids->Jx[i][grd->nyn - 2][k];
        ids->Jx[i][grd->nyn - 2][k] = ids->Jx[i][1][k]; // second is = not +=
        // Jy
        ids->Jy[i][1][k] += ids->Jy[i][grd->nyn - 2][k];
        ids->Jy[i][grd->nyn - 2][k] = ids->Jy[i][1][k]; // second is = not +=
        // Jz
        ids->Jz[i][1][k] += ids->Jz[i][grd->nyn - 2][k];
        ids->Jz[i][grd->nyn - 2][k] = ids->Jz[i][1][k]; // second is = not +=

        // pxx
        ids->pxx[i][1][k] += ids->pxx[i][grd->nyn - 2][k];
        ids->pxx[i][grd->nyn - 2][k] = ids->pxx[i][1][k]; // second is = not +=
        // pxy
        ids->pxy[i][1][k] += ids->pxy[i][grd->nyn - 2][k];
        ids->pxy[i][grd->nyn - 2][k] = ids->pxy[i][1][k]; // second is = not +=
        // pxz
        ids->pxz[i][1][k] += ids->pxz[i][grd->nyn - 2][k];
        ids->pxz[i][grd->nyn - 2][k] = ids->pxz[i][1][k]; // second is = not +=

        // pyy
        ids->pyy[i][1][k] += ids->pyy[i][grd->nyn - 2][k];
        ids->pyy[i][grd->nyn - 2][k] = ids->pyy[i][1][k]; // second is = not +=
        // pyz
        ids->pyz[i][1][k] += ids->pyz[i][grd->nyn - 2][k];
        ids->pyz[i][grd->nyn - 2][k] = ids->pyz[i][1][k]; // second is = not +=
        // pzz
        ids->pzz[i][1][k] += ids->pzz[i][grd->nyn - 2][k];
        ids->pzz[i][grd->nyn - 2][k] = ids->pzz[i][1][k]; // second is = not +=

      } // end of loop

  } // end of PERIODICY

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == true) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int j = 1; j < grd->nyn - 1; j++) {

        // rhon
        ids->rhon[i][j][1] += ids->rhon[i][j][grd->nzn - 2];
        ids->rhon[i][j][grd->nzn - 2] =
            ids->rhon[i][j][1]; // second is = not +=

        // Jx
        ids->Jx[i][j][1] += ids->Jx[i][j][grd->nzn - 2];
        ids->Jx[i][j][grd->nzn - 2] = ids->Jx[i][j][1]; // second is = not +=
        // Jy
        ids->Jy[i][j][1] += ids->Jy[i][j][grd->nzn - 2];
        ids->Jy[i][j][grd->nzn - 2] = ids->Jy[i][j][1]; // second is = not +=
        // Jz
        ids->Jz[i][j][1] += ids->Jz[i][j][grd->nzn - 2];
        ids->Jz[i][j][grd->nzn - 2] = ids->Jz[i][j][1]; // second is = not +=

        // pxx
        ids->pxx[i][j][1] += ids->pxx[i][j][grd->nzn - 2];
        ids->pxx[i][j][grd->nzn - 2] = ids->pxx[i][j][1]; // second is = not +=
        // pxy
        ids->pxy[i][j][1] += ids->pxy[i][j][grd->nzn - 2];
        ids->pxy[i][j][grd->nzn - 2] = ids->pxy[i][j][1]; // second is = not +=
        // pxz
        ids->pxz[i][j][1] += ids->pxz[i][j][grd->nzn - 2];
        ids->pxz[i][j][grd->nzn - 2] = ids->pxz[i][j][1]; // second is = not +=
        // pyy
        ids->pyy[i][j][1] += ids->pyy[i][j][grd->nzn - 2];
        ids->pyy[i][j][grd->nzn - 2] = ids->pyy[i][j][1]; // second is = not +=
        // pyz
        ids->pyz[i][j][1] += ids->pyz[i][j][grd->nzn - 2];
        ids->pyz[i][j][grd->nzn - 2] = ids->pyz[i][j][1]; // second is = not +=
        // pzz
        ids->pzz[i][j][1] += ids->pzz[i][j][grd->nzn - 2];
        ids->pzz[i][j][grd->nzn - 2] = ids->pzz[i][j][1]; // second is = not +=
      }
  }

  // apply BC if BC are not periodic
  // X direction
  if (param->PERIODICX == false) {
    for (register int j = 1; j < grd->nyn - 1; j++)
      for (register int k = 1; k < grd->nzn - 1; k++) {
        // rhon
        ids->rhon[1][j][k] *= 2;
        ids->rhon[grd->nxn - 2][j][k] *= 2; // second is = not +=

        // Jx
        ids->Jx[1][j][k] *= 2;
        ids->Jx[grd->nxn - 2][j][k] *= 2;
        // Jy
        ids->Jy[1][j][k] *= 2;
        ids->Jy[grd->nxn - 2][j][k] *= 2;
        // Jz
        ids->Jz[1][j][k] *= 2;
        ids->Jz[grd->nxn - 2][j][k] *= 2;

        // pxx
        ids->pxx[1][j][k] *= 2;
        ids->pxx[grd->nxn - 2][j][k] *= 2;
        // pxy
        ids->pxy[1][j][k] *= 2;
        ids->pxy[grd->nxn - 2][j][k] *= 2;
        // pxz
        ids->pxz[1][j][k] *= 2;
        ids->pxz[grd->nxn - 2][j][k] *= 2;

        // pyy
        ids->pyy[1][j][k] *= 2;
        ids->pyy[grd->nxn - 2][j][k] *= 2;
        // pyz
        ids->pyz[1][j][k] *= 2;
        ids->pyz[grd->nxn - 2][j][k] *= 2;
        // pzz
        ids->pzz[1][j][k] *= 2;
        ids->pzz[grd->nxn - 2][j][k] *= 2;

      } // end of loop over the grid
  } // end of not periodic in X direction

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == false) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int k = 1; k < grd->nzn - 1; k++) {

        // rhon
        ids->rhon[i][1][k] *= 2;
        ids->rhon[i][grd->nyn - 2][k] *= 2;

        // Jx
        ids->Jx[i][1][k] *= 2;
        ids->Jx[i][grd->nyn - 2][k] *= 2;
        // Jy
        ids->Jy[i][1][k] *= 2;
        ids->Jy[i][grd->nyn - 2][k] *= 2;
        // Jz
        ids->Jz[i][1][k] *= 2;
        ids->Jz[i][grd->nyn - 2][k] *= 2;

        // pxx
        ids->pxx[i][1][k] *= 2;
        ids->pxx[i][grd->nyn - 2][k] *= 2;
        // pxy
        ids->pxy[i][1][k] *= 2;
        ids->pxy[i][grd->nyn - 2][k] *= 2;
        // pxz
        ids->pxz[i][1][k] *= 2;
        ids->pxz[i][grd->nyn - 2][k] *= 2;

        // pyy
        ids->pyy[i][1][k] *= 2;
        ids->pyy[i][grd->nyn - 2][k] *= 2;
        // pyz
        ids->pyz[i][1][k] *= 2;
        ids->pyz[i][grd->nyn - 2][k] *= 2;
        // pzz
        ids->pzz[i][1][k] *= 2;
        ids->pzz[i][grd->nyn - 2][k] *= 2;

      } // end of loop

  } // end of non PERIODICY

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == false) {
    for (register int i = 1; i < grd->nxn - 1; i++)
      for (register int j = 1; j < grd->nyn - 1; j++) {

        // rhon
        ids->rhon[i][j][1] *= 2;
        ids->rhon[i][j][grd->nzn - 2] *= 2;

        // Jx
        ids->Jx[i][j][1] *= 2;
        ids->Jx[i][j][grd->nzn - 2] *= 2;
        // Jy
        ids->Jy[i][j][1] *= 2;
        ids->Jy[i][j][grd->nzn - 2] *= 2;
        // Jz
        ids->Jz[i][j][1] *= 2;
        ids->Jz[i][j][grd->nzn - 2] *= 2;

        // pxx
        ids->pxx[i][j][1] *= 2;
        ids->pxx[i][j][grd->nzn - 2] *= 2;
        // pxy
        ids->pxy[i][j][1] *= 2;
        ids->pxy[i][j][grd->nzn - 2] *= 2;
        // pxz
        ids->pxz[i][j][1] *= 2;
        ids->pxz[i][j][grd->nzn - 2] *= 2;
        // pyy
        ids->pyy[i][j][1] *= 2;
        ids->pyy[i][j][grd->nzn - 2] *= 2;
        // pyz
        ids->pyz[i][j][1] *= 2;
        ids->pyz[i][j][grd->nzn - 2] *= 2;
        // pzz
        ids->pzz[i][j][1] *= 2;
        ids->pzz[i][j][grd->nzn - 2] *= 2;
      }
  } // end of non X periodic
}

//////////
// POPULATE GHOST CELL ON CELL CENTERS
//////////

/** Apply BC to scalar interp quantity defined on nodes - Interpolation quantity
 */
void applyBCscalarDensC(FPinterp ***scalarC, grid *grd, parameters *param) {

  ///////////////////////
  ///
  ///    FACE
  ///

  // X direction
  if (param->PERIODICX == true) {
    for (register int j = 1; j < grd->nyc - 1; j++)
      for (register int k = 1; k < grd->nzc - 1; k++) {
        scalarC[0][j][k] = scalarC[grd->nxc - 2][j][k];
        scalarC[grd->nxc - 1][j][k] = scalarC[1][j][k];
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == true) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int k = 1; k < grd->nzc - 1; k++) {

        // rhon
        scalarC[i][0][k] = scalarC[i][grd->nyc - 2][k];
        scalarC[i][grd->nyc - 1][k] = scalarC[i][1][k];
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == true) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int j = 1; j < grd->nyc - 1; j++) {

        // rhon
        scalarC[i][j][0] = scalarC[i][j][grd->nzc - 2];
        scalarC[i][j][grd->nzc - 1] = scalarC[i][j][1];
      }
  }

  ///////////////////////
  ///
  ///    EDGES
  ///

  // X-EDGE
  if (param->PERIODICY == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nxc - 1); i++) {
      scalarC[i][grd->nyc - 1][grd->nzc - 1] = scalarC[i][1][1];
      scalarC[i][0][0] = scalarC[i][grd->nyc - 2][grd->nzc - 2];
      scalarC[i][0][grd->nzc - 1] = scalarC[i][grd->nyc - 2][1];
      scalarC[i][grd->nyc - 1][0] = scalarC[i][1][grd->nzc - 2];
    }
  }

  // Y-EDGE
  if (param->PERIODICX == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nyn - 1); i++) {
      scalarC[grd->nxc - 1][i][grd->nzc - 1] = scalarC[1][i][1];
      scalarC[0][i][0] = scalarC[grd->nxc - 2][i][grd->nzc - 2];
      scalarC[0][i][grd->nzc - 1] = scalarC[grd->nxc - 2][i][1];
      scalarC[grd->nxc - 1][i][0] = scalarC[1][i][grd->nzc - 2];
    }
  }

  // Z-EDGE
  if (param->PERIODICX == true || param->PERIODICY == true) {
    for (register int i = 1; i < (grd->nzc - 1); i++) {
      scalarC[grd->nxc - 1][grd->nyc - 1][i] = scalarC[1][1][i];
      scalarC[0][0][i] = scalarC[grd->nxc - 2][grd->nyc - 2][i];
      scalarC[grd->nxc - 1][0][i] = scalarC[1][grd->nyc - 2][i];
      scalarC[0][grd->nyc - 1][i] = scalarC[grd->nxc - 2][1][i];
    }
  }

  // Corners
  if (param->PERIODICX == true || param->PERIODICY == true ||
      param->PERIODICZ == true) {

    scalarC[grd->nxc - 1][grd->nyc - 1][grd->nzc - 1] = scalarC[1][1][1];
    scalarC[0][grd->nyc - 1][grd->nzc - 1] = scalarC[grd->nxc - 2][1][1];
    scalarC[grd->nxc - 1][0][grd->nzc - 1] = scalarC[1][grd->nyc - 2][1];
    scalarC[0][0][grd->nzc - 1] = scalarC[grd->nxc - 2][grd->nyc - 2][1];
    scalarC[grd->nxc - 1][grd->nyc - 1][0] = scalarC[1][1][grd->nzc - 2];
    scalarC[0][grd->nyc - 1][0] = scalarC[grd->nxc - 2][1][grd->nzc - 2];
    scalarC[grd->nxc - 1][0][0] = scalarC[1][grd->nyc - 2][grd->nzc - 2];
    scalarC[0][0][0] = scalarC[grd->nxc - 2][grd->nyc - 2][grd->nzc - 2];
  }

  // FACE NON PERIODIC: PUT Neuman condition absence of something else
  // X direction
  if (param->PERIODICX == false) {
    for (register int j = 1; j < grd->nyc - 1; j++)
      for (register int k = 1; k < grd->nzc - 1; k++) {
        scalarC[0][j][k] = scalarC[1][j][k];
        scalarC[grd->nxc - 1][j][k] = scalarC[grd->nxc - 2][j][k];
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == false) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int k = 1; k < grd->nzc - 1; k++) {
        scalarC[i][0][k] = scalarC[i][1][k];
        scalarC[i][grd->nyc - 1][k] = scalarC[i][grd->nyc - 2][k];
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == false) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int j = 1; j < grd->nyc - 1; j++) {
        scalarC[i][j][0] = scalarC[i][j][1];
        scalarC[i][j][grd->nzc - 1] = scalarC[i][j][grd->nzc - 2];
      }
  }
}

/** Apply BC to scalar field quantity defined on center - Interpolation quantity
 */
void applyBCscalarFieldC(FPfield ***scalarC, grid *grd, parameters *param) {
  ///////////////////////
  ///
  ///    FACE
  ///

  // X direction
  if (param->PERIODICX == true) {
    for (register int j = 1; j < grd->nyc - 1; j++)
      for (register int k = 1; k < grd->nzc - 1; k++) {
        scalarC[0][j][k] = scalarC[grd->nxc - 2][j][k];
        scalarC[grd->nxc - 1][j][k] = scalarC[1][j][k];
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == true) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int k = 1; k < grd->nzc - 1; k++) {

        // rhon
        scalarC[i][0][k] = scalarC[i][grd->nyc - 2][k];
        scalarC[i][grd->nyc - 1][k] = scalarC[i][1][k];
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == true) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int j = 1; j < grd->nyc - 1; j++) {

        // rhon
        scalarC[i][j][0] = scalarC[i][j][grd->nzc - 2];
        scalarC[i][j][grd->nzc - 1] = scalarC[i][j][1];
      }
  }

  ///////////////////////
  ///
  ///    EDGES
  ///

  // X-EDGE
  if (param->PERIODICY == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nxc - 1); i++) {
      scalarC[i][grd->nyc - 1][grd->nzc - 1] = scalarC[i][1][1];
      scalarC[i][0][0] = scalarC[i][grd->nyc - 2][grd->nzc - 2];
      scalarC[i][0][grd->nzc - 1] = scalarC[i][grd->nyc - 2][1];
      scalarC[i][grd->nyc - 1][0] = scalarC[i][1][grd->nzc - 2];
    }
  }

  // Y-EDGE
  if (param->PERIODICX == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nyn - 1); i++) {
      scalarC[grd->nxc - 1][i][grd->nzc - 1] = scalarC[1][i][1];
      scalarC[0][i][0] = scalarC[grd->nxc - 2][i][grd->nzc - 2];
      scalarC[0][i][grd->nzc - 1] = scalarC[grd->nxc - 2][i][1];
      scalarC[grd->nxc - 1][i][0] = scalarC[1][i][grd->nzc - 2];
    }
  }

  // Z-EDGE
  if (param->PERIODICX == true || param->PERIODICY == true) {
    for (register int i = 1; i < (grd->nzc - 1); i++) {
      scalarC[grd->nxc - 1][grd->nyc - 1][i] = scalarC[1][1][i];
      scalarC[0][0][i] = scalarC[grd->nxc - 2][grd->nyc - 2][i];
      scalarC[grd->nxc - 1][0][i] = scalarC[1][grd->nyc - 2][i];
      scalarC[0][grd->nyc - 1][i] = scalarC[grd->nxc - 2][1][i];
    }
  }

  // Corners
  if (param->PERIODICX == true || param->PERIODICY == true ||
      param->PERIODICZ == true) {

    scalarC[grd->nxc - 1][grd->nyc - 1][grd->nzc - 1] = scalarC[1][1][1];
    scalarC[0][grd->nyc - 1][grd->nzc - 1] = scalarC[grd->nxc - 2][1][1];
    scalarC[grd->nxc - 1][0][grd->nzc - 1] = scalarC[1][grd->nyc - 2][1];
    scalarC[0][0][grd->nzc - 1] = scalarC[grd->nxc - 2][grd->nyc - 2][1];
    scalarC[grd->nxc - 1][grd->nyc - 1][0] = scalarC[1][1][grd->nzc - 2];
    scalarC[0][grd->nyc - 1][0] = scalarC[grd->nxc - 2][1][grd->nzc - 2];
    scalarC[grd->nxc - 1][0][0] = scalarC[1][grd->nyc - 2][grd->nzc - 2];
    scalarC[0][0][0] = scalarC[grd->nxc - 2][grd->nyc - 2][grd->nzc - 2];
  }

  // FACE NON PERIODIC: PUT Neuman condition absence of something else
  // X direction
  if (param->PERIODICX == false) {
    for (register int j = 1; j < grd->nyc - 1; j++)
      for (register int k = 1; k < grd->nzc - 1; k++) {
        scalarC[0][j][k] = scalarC[1][j][k];
        scalarC[grd->nxc - 1][j][k] = scalarC[grd->nxc - 2][j][k];
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == false) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int k = 1; k < grd->nzc - 1; k++) {
        scalarC[i][0][k] = scalarC[i][1][k];
        scalarC[i][grd->nyc - 1][k] = scalarC[i][grd->nyc - 2][k];
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == false) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int j = 1; j < grd->nyc - 1; j++) {
        scalarC[i][j][0] = scalarC[i][j][1];
        scalarC[i][j][grd->nzc - 1] = scalarC[i][j][grd->nzc - 2];
      }
  }
}

/** Apply BC to scalar field quantity defined on nodes - Interpolation quantity
 */
// set to zero ghost cell
void applyBCscalarFieldCzero(FPfield ***scalarC, grid *grd, parameters *param) {

  ///////////////////////
  ///
  ///    FACE
  ///

  // X direction
  if (param->PERIODICX == true) {
    for (register int j = 1; j < grd->nyc - 1; j++)
      for (register int k = 1; k < grd->nzc - 1; k++) {
        scalarC[0][j][k] = scalarC[grd->nxc - 2][j][k];
        scalarC[grd->nxc - 1][j][k] = scalarC[1][j][k];
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == true) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int k = 1; k < grd->nzc - 1; k++) {

        // rhon
        scalarC[i][0][k] = scalarC[i][grd->nyc - 2][k];
        scalarC[i][grd->nyc - 1][k] = scalarC[i][1][k];
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == true) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int j = 1; j < grd->nyc - 1; j++) {

        // rhon
        scalarC[i][j][0] = scalarC[i][j][grd->nzc - 2];
        scalarC[i][j][grd->nzc - 1] = scalarC[i][j][1];
      }
  }

  ///////////////////////
  ///
  ///    EDGES
  ///

  // X-EDGE
  if (param->PERIODICY == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nxc - 1); i++) {
      scalarC[i][grd->nyc - 1][grd->nzc - 1] = scalarC[i][1][1];
      scalarC[i][0][0] = scalarC[i][grd->nyc - 2][grd->nzc - 2];
      scalarC[i][0][grd->nzc - 1] = scalarC[i][grd->nyc - 2][1];
      scalarC[i][grd->nyc - 1][0] = scalarC[i][1][grd->nzc - 2];
    }
  }

  // Y-EDGE
  if (param->PERIODICX == true || param->PERIODICZ == true) {
    for (register int i = 1; i < (grd->nyn - 1); i++) {
      scalarC[grd->nxc - 1][i][grd->nzc - 1] = scalarC[1][i][1];
      scalarC[0][i][0] = scalarC[grd->nxc - 2][i][grd->nzc - 2];
      scalarC[0][i][grd->nzc - 1] = scalarC[grd->nxc - 2][i][1];
      scalarC[grd->nxc - 1][i][0] = scalarC[1][i][grd->nzc - 2];
    }
  }

  // Z-EDGE
  if (param->PERIODICX == true || param->PERIODICY == true) {
    for (register int i = 1; i < (grd->nzc - 1); i++) {
      scalarC[grd->nxc - 1][grd->nyc - 1][i] = scalarC[1][1][i];
      scalarC[0][0][i] = scalarC[grd->nxc - 2][grd->nyc - 2][i];
      scalarC[grd->nxc - 1][0][i] = scalarC[1][grd->nyc - 2][i];
      scalarC[0][grd->nyc - 1][i] = scalarC[grd->nxc - 2][1][i];
    }
  }

  // Corners
  if (param->PERIODICX == true || param->PERIODICY == true ||
      param->PERIODICZ == true) {

    scalarC[grd->nxc - 1][grd->nyc - 1][grd->nzc - 1] = scalarC[1][1][1];
    scalarC[0][grd->nyc - 1][grd->nzc - 1] = scalarC[grd->nxc - 2][1][1];
    scalarC[grd->nxc - 1][0][grd->nzc - 1] = scalarC[1][grd->nyc - 2][1];
    scalarC[0][0][grd->nzc - 1] = scalarC[grd->nxc - 2][grd->nyc - 2][1];
    scalarC[grd->nxc - 1][grd->nyc - 1][0] = scalarC[1][1][grd->nzc - 2];
    scalarC[0][grd->nyc - 1][0] = scalarC[grd->nxc - 2][1][grd->nzc - 2];
    scalarC[grd->nxc - 1][0][0] = scalarC[1][grd->nyc - 2][grd->nzc - 2];
    scalarC[0][0][0] = scalarC[grd->nxc - 2][grd->nyc - 2][grd->nzc - 2];
  }

  // FACE NON PERIODIC: PUT Neuman condition absence of something else
  // X direction
  if (param->PERIODICX == false) {
    for (register int j = 1; j < grd->nyc - 1; j++)
      for (register int k = 1; k < grd->nzc - 1; k++) {
        scalarC[0][j][k] = 0.0;
        scalarC[grd->nxc - 1][j][k] = 0.0;
      }
  }

  // Periodic boundary conditions in Y direction
  if (param->PERIODICY == false) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int k = 1; k < grd->nzc - 1; k++) {
        scalarC[i][0][k] = 0.0;
        scalarC[i][grd->nyc - 1][k] = 0.0;
      }
  }

  // Periodic boundary conditions in Z direction
  if (param->PERIODICZ == false) {
    for (register int i = 1; i < grd->nxc - 1; i++)
      for (register int j = 1; j < grd->nyc - 1; j++) {
        scalarC[i][j][0] = 0.0;
        scalarC[i][j][grd->nzc - 1] = 0.0;
      }
  }
}
