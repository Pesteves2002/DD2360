#ifndef INTERPDENSNET_H
#define INTERPDENSNET_H

/** Interpolated densities - Net = sum all of species contributions */

#include "Alloc.h"
#include "Grid.h"
#include "InterpDensSpecies.h"
#include "PrecisionTypes.h"

struct __align__(16) interpDensNet {

  /** charged densities */
  FPinterp ***rhon;
  FPinterp *rhon_flat; // rho defined on nodes
  FPinterp ***rhoc;
  FPinterp *rhoc_flat; // rho defined at center cell
  /** J current densities */
  FPinterp ***Jx;
  FPinterp *Jx_flat;
  FPinterp ***Jy;
  FPinterp *Jy_flat;
  FPinterp ***Jz;
  FPinterp *Jz_flat; // on nodes
  /** p = pressure tensor*/
  FPinterp ***pxx;
  FPinterp *pxx_flat;
  FPinterp ***pxy;
  FPinterp *pxy_flat;
  FPinterp ***pxz;
  FPinterp *pxz_flat; // on nodes
  FPinterp ***pyy;
  FPinterp *pyy_flat;
  FPinterp ***pyz;
  FPinterp *pyz_flat;
  FPinterp ***pzz;
  FPinterp *pzz_flat; // on nodes
};

/** allocated interpolated densities per species */
void interp_dens_net_allocate(struct grid *, struct interpDensNet *);

/** deallocate interpolated densities per species */
void interp_dens_net_deallocate(struct grid *, struct interpDensNet *);

/** set all the densities to zero */
void setZeroDensities(struct interpDensNet *, struct interpDensSpecies *,
                      struct grid *, int);

/** sum over species */
void sumOverSpecies(struct interpDensNet *, struct interpDensSpecies *,
                    struct grid *, int);

/** sum over species GPU */
void sumOverSpeciesGPU(struct interpDensNet *idn, struct interpDensSpecies *ids,
                       struct grid *grd, int ns);

#endif
