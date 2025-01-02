#ifndef SMOOTHING_H
#define SMOOTHING_H

#include "Grid.h"
#include "Parameters.h"
#include "PrecisionTypes.h"

/** Smmoth Interpolation Quantity defined on Center */
void smoothInterpScalarC(FPinterp ***, grid *, parameters *);

/** Smmoth Interpolation Quantity defined on Nodes */
void smoothInterpScalarN(FPinterp ***, grid *, parameters *);

#endif
