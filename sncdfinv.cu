/*

Copyright 2016 Thomas Luu

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

/*

File: sncdfinv.cu

Computation of the skew-normal quantile function.

Based on:

Luu, T; (2016) Fast and accurate parallel computation of quantile functions for 
random number generation. Doctoral thesis, UCL (University College London).

http://discovery.ucl.ac.uk/1482128/

*/

#ifndef SNCDFINV
#define SNCDFINV

#include "plog.cu"

#include <math_constants.h>

__host__ __device__ double sncdfinv(double u, double a)
{
  double tol = 0.01;

  if (u == 0.0) {
#ifdef __CUDA_ARCH__
    return -CUDART_INF;
#else
    return -INFINITY;
#endif
  }

  if (u == 1.0) {
#ifdef __CUDA_ARCH__
    return CUDART_INF;
#else
    return INFINITY;
#endif
  }

  /*
   * Change of variable + special cases
   */
  if (a == 1) u = sqrt(u);
  if (a == -1) u = sqrt(1-u);
  double z = normcdfinv(u);
  if (a == 0) return z;
  else if (a == 1) return z;
  else if (a == -1) return -z;
  if (a < 0) z = -z;

  double A = fabs(a);

  double right_limit = erf(erfcinv(2 * tol) / A);

  /*
   * Tails
   */
  if (a > 0 && u > right_limit) {
    return 1.4142135623730950488 * erfinv(u);
  } else if (a < 0 && (1-u) > right_limit) {
    return -1.4142135623730950488 * erfcinv(u);
  }

  double x = normcdfinv(0.5 - 0.31830988618379067154 * atan(A));

  double expon = exp(0.5 * ( - x*x));
  double errfn = 1.0;
  double efder = expon * 0.79788456080286535588 * A / errfn;

  double c0 = 0;
  double c1 = expon / errfn;
  double c2 = - expon*(efder + errfn*x) / (2*errfn*errfn);
  double c3 = 0.16666666666666666667 * expon*(3*efder*efder + errfn*errfn*(-1 + x*x) + expon*expon + efder*(3*errfn*x)) / (errfn*errfn*errfn);
  double c4 = - 0.041666666666666666667 * expon*(15*efder*efder*efder + errfn*errfn*errfn*x*(-3 + x*x) + 6*errfn*expon*expon*x + efder*efder*(18*errfn*x) + efder*(errfn*errfn*(-4 + 7*x*x) + expon*expon*(7 - A*A))) / (errfn*errfn*errfn*errfn);
  double c5 = 0.0083333333333333333333 * expon*(105*efder*efder*efder*efder + errfn*errfn*errfn*errfn*(3 - 6*x*x + x*x*x*x) + 5*errfn*errfn*expon*expon*(-2 + 5*x*x) + expon*expon*expon*expon*(7) + 15*efder*efder*efder*(10*errfn*x) + efder*(5*errfn*errfn*errfn*x*(-5 + 3*x*x) + 10*errfn*expon*expon*x*(7 - A*A)) + 5*efder*efder*(3*errfn*errfn*(-2 + 5*x*x) + expon*expon*(-3*(-4 + A*A)))) / (errfn*errfn*errfn*errfn*errfn);

  //double h = 0.5 * pow(fabs(tol / c5), 0.2);
  double h = 0.75 * pow(fabs(tol / c5), 0.2);
  //double h = 0.9 * pow(fabs(tol / c5), 0.2);

  double left_limit = x - h;
  if (z < left_limit) {
    if (a > 0) {
      return -sqrt(2 * plog(1 / (6.2831853071795864769 * u * a)) / (1 + a*a));
    } else {
      return sqrt(2 * plog(1 / (6.2831853071795864769 * (1-u) * fabs(a))) / (1 + a*a));
    }
  }

  // otherwise eval central series 
  h = z - x;
  double res = c0 + h*(c1 + h*(c2 + h*(c3 + h*(c4 + h*c5))));

  return a < 0 ? -res : res;
}

#endif
