/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous
 * systems **/

#include <stdio.h>
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

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"
#include <assert.h>

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensNet.h"
#include "InterpDensSpecies.h"

// Field structure
#include "EMfield.h"     // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

int main(int argc, char **argv) {

  // Read the inputfile and fill the param structure
  parameters param;
  // Read the input file name from command line
  readInputFile(&param, argc, argv);
  printParameters(&param);
  saveParameters(&param);

  // Timing variables
  double iStart = cpuSecond();
  double iMover, iInterp, eMover = 0.0, eInterp = 0.0;
  double iBC, eBC = 0.0;
  double iSUM, eSUM = 0.0;
  double iDens, eDens = 0.0;
  double iZero, eZero = 0.0;
  double iWrite, eWrite1 = 0.0;

  // Set-up the grid information
  grid grd;
  setGrid(&param, &grd);

  // Allocate Fields
  EMfield field;
  field_allocate(&grd, &field);
  EMfield_aux field_aux;
  field_aux_allocate(&grd, &field_aux);

  // Allocate Interpolated Quantities
  // per species
  interpDensSpecies *ids = new interpDensSpecies[param.ns];
  for (int is = 0; is < param.ns; is++)
    interp_dens_species_allocate(&grd, &ids[is], is);
  // Net densities
  interpDensNet idn;
  interp_dens_net_allocate(&grd, &idn);

  // Allocate Particles
  particles *part = new particles[param.ns];
  // allocation
  for (int is = 0; is < param.ns; is++) {
    particle_allocate(&param, &part[is], is);
  }

  // Initialization
  initGEM(&param, &grd, &field, &field_aux, part, ids);

  // Copy parameters to device
  parameters *d_param;
  gpuErrchk(cudaMalloc(&d_param, sizeof(parameters)));
  gpuErrchk(
      cudaMemcpy(d_param, &param, sizeof(parameters), cudaMemcpyHostToDevice));

  // Copy grid to device
  grid *d_grid;
  gpuErrchk(cudaMalloc(&d_grid, sizeof(grid)));
  gpuErrchk(cudaMemcpy(d_grid, &grd, sizeof(grid), cudaMemcpyHostToDevice));

  FPfield *d_XN, *d_YN, *d_ZN;
  int grid_size = grd.nxn * grd.nyn * grd.nzn * sizeof(FPfield);

  // Allocate 1D arrays
  gpuErrchk(cudaMalloc(&d_XN, grid_size));
  gpuErrchk(cudaMalloc(&d_YN, grid_size));
  gpuErrchk(cudaMalloc(&d_ZN, grid_size));

  // Set pointers in device
  gpuErrchk(cudaMemcpy(&(d_grid->XN_flat), &d_XN, sizeof(FPfield *),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(&(d_grid->YN_flat), &d_YN, sizeof(FPfield *),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(&(d_grid->ZN_flat), &d_ZN, sizeof(FPfield *),
                       cudaMemcpyHostToDevice));
  // Copy 1D arrays
  gpuErrchk(cudaMemcpy(d_XN, grd.XN_flat, grid_size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_YN, grd.YN_flat, grid_size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_ZN, grd.ZN_flat, grid_size, cudaMemcpyHostToDevice));

  // Copy EMfield to device
  EMfield *d_field;
  gpuErrchk(cudaMalloc(&d_field, sizeof(EMfield)));
  gpuErrchk(
      cudaMemcpy(d_field, &field, sizeof(EMfield), cudaMemcpyHostToDevice));

  FPfield *d_field_Ex, *d_field_Ey, *d_field_Ez, *d_field_Bxn, *d_field_Byn,
      *d_field_Bzn;
  int field_size = grd.nxn * grd.nyn * grd.nzn * sizeof(FPfield);

  // Allocate 1D arrays
  gpuErrchk(cudaMalloc(&d_field_Ex, field_size));
  gpuErrchk(cudaMalloc(&d_field_Ey, field_size));
  gpuErrchk(cudaMalloc(&d_field_Ez, field_size));
  gpuErrchk(cudaMalloc(&d_field_Bxn, field_size));
  gpuErrchk(cudaMalloc(&d_field_Byn, field_size));
  gpuErrchk(cudaMalloc(&d_field_Bzn, field_size));

  // Set pointers in device
  gpuErrchk(cudaMemcpy(&(d_field->Ex_flat), &d_field_Ex, sizeof(FPfield *),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(&(d_field->Ey_flat), &d_field_Ey, sizeof(FPfield *),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(&(d_field->Ez_flat), &d_field_Ez, sizeof(FPfield *),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(&(d_field->Bxn_flat), &d_field_Bxn, sizeof(FPfield *),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(&(d_field->Byn_flat), &d_field_Byn, sizeof(FPfield *),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(&(d_field->Bzn_flat), &d_field_Bzn, sizeof(FPfield *),
                       cudaMemcpyHostToDevice));

  // Copy 1D arrays
  gpuErrchk(cudaMemcpy(d_field_Ex, field.Ex_flat, field_size,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_field_Ey, field.Ey_flat, field_size,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_field_Ez, field.Ez_flat, field_size,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_field_Bxn, field.Bxn_flat, field_size,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_field_Byn, field.Byn_flat, field_size,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_field_Bzn, field.Bzn_flat, field_size,
                       cudaMemcpyHostToDevice));

  interpDensNet *d_idn;
  gpuErrchk(cudaMalloc(&d_idn, sizeof(interpDensNet)));

  FPinterp *d_rhon_flat_n;

  int interp_size = grd.nxn * grd.nyn * grd.nzn * sizeof(FPinterp);

  // Allocate 1D arrays for d_idn
  gpuErrchk(cudaMalloc(&d_rhon_flat_n, interp_size));

  // Set pointers in device for d_idn
  gpuErrchk(cudaMemcpy(&(d_idn->rhon_flat), &d_rhon_flat_n, sizeof(FPinterp *),
                       cudaMemcpyHostToDevice));
  // Copy 1D arrays to device
  gpuErrchk(cudaMemcpy(d_rhon_flat_n, idn.rhon_flat, interp_size,
                       cudaMemcpyHostToDevice));

  // Allocate interpDensSpecies array on the GPU
  interpDensSpecies *d_ids;
  gpuErrchk(cudaMalloc(&d_ids, param.ns * sizeof(interpDensSpecies)));

  FPinterp *d_rhon_flat[param.ns];

  // Allocate and copy fields for each species
  for (int is = 0; is < param.ns; is++) {
    gpuErrchk(cudaMemcpy(&d_ids[is], &ids[is], sizeof(interpDensSpecies),
                         cudaMemcpyHostToDevice));

    int interp_size = grd.nxn * grd.nyn * grd.nzn * sizeof(FPinterp);

    gpuErrchk(cudaMalloc(&d_rhon_flat[is], interp_size));

    gpuErrchk(cudaMemcpy(&(d_ids[is].rhon_flat), &d_rhon_flat[is],
                         sizeof(FPinterp *), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_rhon_flat[is], ids[is].rhon_flat, interp_size,
                         cudaMemcpyHostToDevice));
  }

  // Allocate particles array on the GPU
  particles *d_part;
  gpuErrchk(cudaMalloc(&d_part, param.ns * sizeof(particles)));

  // Loop over each species and allocate/copy fields to GPU
  for (int is = 0; is < param.ns; is++) {
    gpuErrchk(cudaMemcpy(&d_part[is], &part[is], sizeof(particles),
                         cudaMemcpyHostToDevice));

    FPpart *d_x, *d_y, *d_z, *d_u, *d_v, *d_w;
    FPinterp *d_q;

    int part_size = part[is].npmax * sizeof(FPpart);
    int q_size = part[is].npmax * sizeof(FPinterp);

    // Allocate device memory for fields
    gpuErrchk(cudaMalloc(&d_x, part_size));
    gpuErrchk(cudaMalloc(&d_y, part_size));
    gpuErrchk(cudaMalloc(&d_z, part_size));
    gpuErrchk(cudaMalloc(&d_u, part_size));
    gpuErrchk(cudaMalloc(&d_v, part_size));
    gpuErrchk(cudaMalloc(&d_w, part_size));
    gpuErrchk(cudaMalloc(&d_q, q_size));

    // Update device structure with pointers to device fields
    gpuErrchk(cudaMemcpy(&(d_part[is].x), &d_x, sizeof(FPpart *),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_part[is].y), &d_y, sizeof(FPpart *),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_part[is].z), &d_z, sizeof(FPpart *),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_part[is].u), &d_u, sizeof(FPpart *),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_part[is].v), &d_v, sizeof(FPpart *),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_part[is].w), &d_w, sizeof(FPpart *),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_part[is].q), &d_q, sizeof(FPinterp *),
                         cudaMemcpyHostToDevice));

    // Copy data from host to device
    gpuErrchk(cudaMemcpy(d_x, part[is].x, part_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, part[is].y, part_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_z, part[is].z, part_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, part[is].u, part_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, part[is].v, part_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_w, part[is].w, part_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_q, part[is].q, q_size, cudaMemcpyHostToDevice));
  }

  // **********************************************************//
  // **** Start the Simulation!  Cycle index start from 1  *** //
  // **********************************************************//
  for (int cycle = param.first_cycle_n;
       cycle < (param.first_cycle_n + param.ncycles); cycle++) {

    iZero = cpuSecond(); // start timer for zeroing the densities

    std::cout << std::endl;
    std::cout << "***********************" << std::endl;
    std::cout << "   cycle = " << cycle << std::endl;
    std::cout << "***********************" << std::endl;

    cudaMemset(d_rhon_flat_n, 0, interp_size);

    for (int is = 0; is < param.ns; is++) {
      gpuErrchk(cudaMemset(d_rhon_flat[is], 0, interp_size));
    }

    eZero += (cpuSecond() - iZero); // stop timer for zeroing the densities

    // implicit mover
    iMover = cpuSecond(); // start timer for mover
    mover_PC_gpu(d_part, d_field, d_grid, d_param, param.n_sub_cycles, part,
                 param.ns, &param);
    eMover += (cpuSecond() - iMover); // stop timer for mover

    // interpolation particle to grid
    iInterp = cpuSecond(); // start timer for the interpolation step
                           // interpolate species
    interpP2G_gpu(d_part, d_ids, d_grid, part, param.ns);

    eInterp += (cpuSecond() - iInterp); // stop timer for interpolation

    iBC = cpuSecond(); // start timer for boundary conditions

    // apply BC to interpolated densities
    applyBCidsGPU(d_ids, &grd, &param);

    eBC += (cpuSecond() - iBC); // stop timer for boundary conditions

    iSUM = cpuSecond(); // start timer for summing densities

    // sum over species
    sumOverSpeciesGPU(d_idn, d_ids, &grd, param.ns);

    eSUM += (cpuSecond() - iSUM); // stop timer for summing densities

    iDens = cpuSecond(); // start timer for density calculation

    // interpolate charge density from center to node
    applyBCscalarDensNGPU(d_idn, &grd, &param);

    eDens += (cpuSecond() - iDens); // stop timer for density calculation

    // write E, B, rho to disk
    if (cycle % param.FieldOutputCycle == 0) {

      iWrite = cpuSecond();

      gpuErrchk(cudaMemcpy(field.Ex_flat, d_field_Ex, field_size,
                           cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(field.Ey_flat, d_field_Ey, field_size,
                           cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(field.Ez_flat, d_field_Ez, field_size,
                           cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(field.Bxn_flat, d_field_Bxn, field_size,
                           cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(field.Byn_flat, d_field_Byn, field_size,
                           cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(field.Bzn_flat, d_field_Bzn, field_size,
                           cudaMemcpyDeviceToHost));

      // Copy interpDensSpecies to host
      for (int is = 0; is < 2; is++) {
        gpuErrchk(cudaMemcpy(ids[is].rhon_flat, d_rhon_flat[is], interp_size,
                             cudaMemcpyDeviceToHost));
      }

      // Copy 1D arrays to host
      gpuErrchk(cudaMemcpy(idn.rhon_flat, d_rhon_flat_n, interp_size,
                           cudaMemcpyDeviceToHost));

      VTK_Write_Vectors(cycle, &grd, &field);
      VTK_Write_Scalars(cycle, &grd, ids, &idn);

      eWrite1 = cpuSecond() - iWrite;
    }

  } // end of one PIC cycle

  /// Release the resources
  // deallocate field
  grid_deallocate(&grd);
  field_deallocate(&grd, &field);
  // interp
  interp_dens_net_deallocate(&grd, &idn);

  // Deallocate interpolated densities and particles
  for (int is = 0; is < param.ns; is++) {
    interp_dens_species_deallocate(&grd, &ids[is]);
    particle_deallocate(&part[is]);
  }

  // Free parameters on device
  gpuErrchk(cudaFree(d_param));

  // Free grid on device
  gpuErrchk(cudaFree(d_grid));

  // Free 1D arrays
  gpuErrchk(cudaFree(d_XN));
  gpuErrchk(cudaFree(d_YN));
  gpuErrchk(cudaFree(d_ZN));

  // Free EMfield on device
  gpuErrchk(cudaFree(d_field));

  // Free 1D arrays
  gpuErrchk(cudaFree(d_field_Ex));
  gpuErrchk(cudaFree(d_field_Ey));
  gpuErrchk(cudaFree(d_field_Ez));
  gpuErrchk(cudaFree(d_field_Bxn));
  gpuErrchk(cudaFree(d_field_Byn));
  gpuErrchk(cudaFree(d_field_Bzn));

  // Free interpDensNet on device
  gpuErrchk(cudaFree(d_idn));

  // Free interpDensSpecies on device
  gpuErrchk(cudaFree(d_ids));

  // Free particles on device
  gpuErrchk(cudaFree(d_part));

  // stop timer
  double iElaps = cpuSecond() - iStart;

  // Print timing of simulation
  std::cout << std::endl;
  std::cout << "**************************************" << std::endl;
  std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
  std::cout << "   Zero Time / Cycle    (s) = " << eZero / param.ncycles
            << std::endl;
  std::cout << "   Mover Time / Cycle   (s) = " << eMover / param.ncycles
            << std::endl;
  std::cout << "   Interp. Time / Cycle (s) = " << eInterp / param.ncycles
            << std::endl;
  std::cout << "   BC Time / Cycle      (s) = " << eBC / param.ncycles
            << std::endl;
  std::cout << "   SUM Time / Cycle     (s) = " << eSUM / param.ncycles
            << std::endl;
  std::cout << "   Dens Time / Cycle    (s) = " << eDens / param.ncycles
            << std::endl;
  std::cout << "   IO Time / Cycle    (s) = " << eWrite1 / param.ncycles
            << std::endl;
  std::cout << "**************************************" << std::endl;

  // exit
  return 0;
}
