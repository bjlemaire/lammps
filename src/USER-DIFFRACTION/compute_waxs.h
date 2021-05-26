/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(waxs,ComputeWAXS)

#else

#ifndef LMP_COMPUTE_WAXS_H
#define LMP_COMPUTE_WAXS_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeWAXS : public Compute {
 public:
  ComputeWAXS(class LAMMPS *, int, char **);
  ~ComputeWAXS();
  void init();
  void compute_array();
  double memory_usage();

 private:
  std::vector<double> polar_to_cart(double theta, double phi);
  double compute_intensity(std::vector<std::vector<double>>& data, double qx, double qy, double qz, double th);

  int     me;
  int     *ztype;            // Atomic number of the different atom types
  double  Min2Theta;         // Minimum 2theta value (input in 2theta rad)
  double  Max2Theta;         // Maximum 2theta value (input in 2theta rad)
  double  ThetaStepSize;     // Step size for the theta angle.
  double  PhiStepSize;       // Step size for the phi angle.
  int     nTheta;            // Number of theta values evaluated.
  int     nPhi;              // Number of phi values evaluated.
  double  beam[3];           // Direction of the incident beam
  double  Kmax;              // Maximum reciprocal distance to explore
  double  c[3];              // Resolution parameters for reciprocal space explored
  int     Knmax[3];          // maximum integer value for K points in each dimension
  double  dK[3];             // Parameters controlling resolution of reciprocal space explored
  double  prd_inv[3];        // Inverse spacing of unit cell
  int     LP;                // Switch to turn on Lorentz-Polarization factor 1=on
  bool    echo;              // echo compute_array progress
  bool    manual;            // Turn on manual recpiprocal map
  bool    radflag;           // Values of angles in radian or degree.

  int    ntypes;
  int    nlocalgroup;
  double lambda;             // Radiation wavelenght (distance units)
  int    *store_tmp;         // Temporary array that stores qx, qy, qz for each entry of the output array

};

}

#endif
#endif