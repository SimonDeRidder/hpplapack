#ifndef LAPACK_HEADER
#define LAPACK_HEADER

#include <limits>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

#include "Blas.h"


template<class real>
class Lapack
{
public:
    // constants

    static constexpr real NEGONE= real(-1.0);
    static constexpr real ZERO  = real(0.0);
    static constexpr real QURTR = real(0.25);
    static constexpr real HALF  = real(0.5);
    static constexpr real ONE   = real(1.0);
    static constexpr real TWO   = real(2.0);
    static constexpr real THREE = real(3.0);
    static constexpr real FOUR  = real(4.0);
    static constexpr real EIGHT = real(8.0);
    static constexpr real TEN   = real(10.0);
    static constexpr real HNDRD = real(100.0);

    // LAPACK INSTALL (alphabetically)

    /* dlamch determines double precision machine parameters.
     * Parameters: cmach: cmach[0] Specifies the value to be returned by DLAMCH:
     *                    'E' or 'e', DLAMCH := eps: relative machine precision
     *                    'S' or 's , DLAMCH := sfmin: safe minimum, such that 1 / sfmin does not
     *                                                 overflow
     *                    'B' or 'b', DLAMCH := base: base of the machine (radix)
     *                    'P' or 'p', DLAMCH := eps*base
     *                    'N' or 'n', DLAMCH := t: number of(base) digits in the mantissa
     *                    'R' or 'r', DLAMCH := rnd: 1.0 when rounding occurs in addition, 0.0
     *                                               otherwise
     *                    'M' or 'm', DLAMCH := emin: minimum exponent before(gradual) underflow
     *                    'U' or 'u', DLAMCH := rmin: underflow threshold - base^(emin - 1)
     *                    'L' or 'l', DLAMCH := emax: largest exponent before overflow
     *                    'O' or 'o', DLAMCH := rmax: overflow threshold - (base^emax)*(1 - eps)
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.																*/
    static real dlamch(char const* cmach)
    {
        real eps;
        // Assume rounding, not chopping.Always.
        real rnd = ONE;
        if (ONE==rnd)
        {
            eps = std::numeric_limits<real>::epsilon() * HALF;
        }
        else
        {
            eps = std::numeric_limits<real>::epsilon();
        }
        real sfmin, small;
        switch (toupper(cmach[0]))
        {
            case 'E':
                return eps;
            case 'S':
                sfmin = std::numeric_limits<real>::min();
                small = ONE / std::numeric_limits<real>::max();
                if (small>sfmin)
                {
                    //Use SMALL plus a bit, to avoid the possibility of rounding
                    // causing overflow when computing  1 / sfmin.
                    sfmin = small * (ONE+eps);
                }
                return sfmin;
            case 'B':
                return std::numeric_limits<real>::radix;
            case 'P':
                return eps * std::numeric_limits<real>::radix;
            case 'N':
                return std::numeric_limits<real>::digits;
            case 'R':
                return rnd;
            case 'M':
                return std::numeric_limits<real>::min_exponent;
            case 'U':
                return std::numeric_limits<real>::min();
            case 'L':
                return std::numeric_limits<real>::max_exponent;
            case 'O':
                return std::numeric_limits<real>::max();
            default:
                return ZERO;
        }
    }

    /* dlamc3 is intended to force A and B to be stored prior to doing the addition of A and B,
     * for use in situations where optimizers might hold one of these in a register.
     * Author:
     *     LAPACK is a software package provided by Univ. of Tennessee,
     *     Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.
     * Date: December 2016
     * Paramters: A,
     *            B: The values A and B.                                                         */
    real dlamc3(real A, real B)
    {
        return A + B;
    }

    // LAPACK SRC (alphabetically)

    /* dbdsqr computes the singular values and, optionally, the right and/or left singular vectors
     * from the singular value decomposition (SVD) of a real n-by-n (upper or lower) bidiagonal
     * matrix B using the implicit zero-shift QR algorithm. The SVD of B has the form
     *     B = Q * S * P^T
     * where S is the diagonal matrix of singular values, Q is an orthogonal matrix of left
     * singular vectors, and P is an orthogonal matrix of right singular vectors. If left singular
     * vectors are requested, this subroutine actually returns U*Q instead of Q, and, if right
     * singular vectors are requested, this subroutine returns P^T*Vt instead of P^T, for given
     * real input matrices U and Vt. When U and Vt are the orthogonal matrices that reduce a
     * general matrix A to bidiagonal form: A = U*B*Vt, as computed by DGEBRD, then
     *     A = (U*Q) * S * (P^T*Vt)
     * is the SVD of A. Optionally, the subroutine may also compute Q^T*C for a given real input
     * matrix C.
     * See "Computing Small Singular Values of Bidiagonal Matrices With Guaranteed High Relative
     *     Accuracy," by J. Demmel and W. Kahan, LAPACK Working Note #3 (or SIAM J. Sci. Statist.
     *     Comput. vol. 11, no. 5, pp. 873-912, Sept 1990)
     * and "Accurate singular values and differential qd algorithms," by B. Parlett and
     *     V. Fernando, Technical Report CPAM-554, Mathematics Department, University of California
     *     at Berkeley, July 1992
     * for a detailed description of the algorithm.
     * Parameters: uplo: ='U': B is upper bidiagonal.
     *                   ='L': B is lower bidiagonal.
     *             n: The order of the matrix B. n >= 0.
     *             ncvt: The number of columns of the matrix Vt. ncvt >= 0.
     *             nru: The number of rows of the matrix U. nru >= 0.
     *             ncc: The number of columns of the matrix C. ncc >= 0.
     *             d: an array, dimension (n)
     *                On entry, the n diagonal elements of the bidiagonal matrix B.
     *                On exit, if info=0, the singular values of B in decreasing order.
     *             e: an array, dimension (n-1)
     *                On entry, the n-1 offdiagonal elements of the bidiagonal matrix B.
     *                On exit, if info==0, e is destroyed; if info>0, d and e will contain the
     *                         diagonal and superdiagonal elements of a bidiagonal matrix
     *                         orthogonally equivalent to the one given as input.
     *             Vt: an array, dimension (ldvt, ncvt)
     *                 On entry, an n-by-ncvt matrix Vt.
     *                 On exit, Vt is overwritten by P^T * Vt.
     *                 Not referenced if ncvt==0.
     *             ldvt: The leading dimension of the array Vt.
     *                   ldvt>=max(1,n) if ncvt>0; ldvt>=1 if ncvt==0.
     *             U: an array, dimension (ldu, n)
     *                On entry, an nru-by-n matrix U.
     *                On exit, U is overwritten by U * Q.
     *                Not referenced if nru==0.
     *             ldu: The leading dimension of the array U. ldu >= max(1,nru).
     *             C: an array, dimension (ldc, ncc)
     *                On entry, an n-by-ncc matrix C.
     *                On exit, C is overwritten by Q^T * C.
     *                Not referenced if ncc = 0.
     *             ldc: The leading dimension of the array C.
     *                  ldc>=max(1,n) if ncc>0; ldc>=1 if ncc==0.
     *             work: an array, dimension (4*n)
     *             info: =0: successful exit
     *                   <0: If info = -i, the i-th argument had an illegal value
     *                   >0: if ncvt = nru = ncc = 0,
     *                       = 1, a split was marked by a positive value in e
     *                       = 2, current block of Z not diagonalized after 30*n iterations
     *                            (in inner while loop)
     *                       = 3, termination criterion of outer while loop not met
     *                            (program created more than n unreduced blocks)
     *                       else
     *                            the algorithm did not converge; d and e contain the elements of a
     *                            bidiagonal matrix which is orthogonally similar to the input
     *                            matrix B; if info==i, i elements of e have not converged to zero.
     * Internal Parameters: TOLMUL: default = max(10,min(100,EPS^(-1/8)))
     *                              TOLMUL controls the convergence criterion of the QR loop.
     *                              If it is positive, TOLMUL*EPS is the desired relative precision
     *                                  in the computed singular values.
     *                              If it is negative, abs(TOLMUL*EPS*sigma_max) is the desired
     *                                  absolute accuracy in the computed singular values
     *                                  (corresponds to relative accuracy abs(TOLMUL*EPS) in the
     *                                  largest singular value.)
     *                              abs(TOLMUL) should be between 1 and 1/EPS, and preferably
     *                                  between 10 (for fast convergence) and 0.1/EPS
     *                                  (for there to be some accuracy in the results).
     *                              Default is to lose at either one eighth or 2 of the available
     *                                  decimal digits in each computed singular value
     *                                  (whichever is smaller).
     *                      MAXITR: default = 6
     *                              MAXITR controls the maximum number of passes of the algorithm
     *                              through its inner loop. The algorithms stops (and so fails to
     *                              converge) if the number of passes through the inner loop
     *                              exceeds MAXITR*n^2.
     * Note:
     *     Bug report from Cezary Dendek.
     *         On March 23rd 2017, the integer variable MAXIT = MAXITR*n^2 is removed since it can
     *         overflow pretty easily (for n larger or equal than 18,919). We instead use
     *         maxitdivn = MAXITR*n.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2017                                                                            */
    static void dbdsqr(char const* uplo, int n, int ncvt, int nru, int ncc, real* d, real* e,
                       real* Vt, int ldvt, real* U, int ldu, real* C, int ldc, real* work,
                       int& info)
    {
        const real MEIGTH = real(-0.125);
        const real HNDRTH = real(0.01);
        const int MAXITR = 6;
        // Test the input parameters.
        info = 0;
        bool lower = (std::toupper(uplo[0])=='L');
        if (!(std::toupper(uplo[0])=='U') && !lower)
        {
            info = -1;
        }
        else if (n<0)
        {
            info = -2;
        }
        else if (ncvt<0)
        {
            info = -3;
        }
        else if (nru<0)
        {
            info = -4;
        }
        else if (ncc<0)
        {
            info = -5;
        }
        else if ((ncvt==0 && ldvt<1) || (ncvt>0 && (ldvt<1 || ldvt<n)))
        {
            info = -9;
        }
        else if (ldu<1 || ldu<nru)
        {
            info = -11;
        }
        else if ((ncc==0 && ldc<1) || (ncc>0 && (ldc<1 || ldc<n)))
        {
            info = -13;
        }
        if (info!=0)
        {
            xerbla("DBDSQR", -info);
            return;
        }
        if (n==0)
        {
            return;
        }
        int i;
        real smin;
        if (n!=1)
        {
            // rotate is true if any singular vectors desired, false otherwise
            bool rotate = ((ncvt>0) || (nru>0) || (ncc>0));
            // If no singular vectors desired, use qd algorithm
            if (!rotate)
            {
                dlasq1(n, d, e, work, info);
                // If info equals 2, dqds didn't finish, try to finish
                if (info!=2)
                {
                    return;
                }
                info = 0;
            }
            int nm1 = n - 1;
            int nm12 = nm1 + nm1;
            int nm13 = nm12 + nm1;
            int IDIR = 0;
            // Get machine constants
            real eps = dlamch("Epsilon");
            real unfl = dlamch("Safe minimum");
            // If matrix lower bidiagonal, rotate to be upper bidiagonal by applying
            // Givens rotations on the left
            real cs, sn, r;
            if (lower)
            {
                for (i=0; i<n-1; i++)
                {
                    dlartg(d[i], e[i], cs, sn, r);
                    d[i] = r;
                    e[i] = sn*d[i+1];
                    d[i+1] = cs*d[i+1];
                    work[i] = cs;
                    work[nm1+i] = sn;
                }
                // Update singular vectors if desired
                if (nru>0)
                {
                    dlasr("R", "V", "F", nru, n, &work[0], &work[n-1], U, ldu);
                }
                if (ncc>0)
                {
                    dlasr("L", "V", "F", n, ncc, &work[0], &work[n-1], C, ldc);
                }
            }
            // Compute singular values to relative accuracy tol (By setting tol to be negative,
            // algorithm will compute singular values to absolute accuracy
            // abs(tol)*norm(input matrix))
            real tolmul = std::pow(eps, MEIGTH);
            if (HNDRD<tolmul)
            {
                tolmul = HNDRD;
            }
            if (TEN>tolmul)
            {
                tolmul = TEN;
            }
            real tol = tolmul * eps;
            // Compute approximate maximum, minimum singular values
            real smax = ZERO, temp;
            for (i=0; i<n; i++)
            {
                temp = std::fabs(d[i]);
                if (temp>smax)
                {
                    smax = temp;
                }
            }
            for (i=0; i<n-1; i++)
            {
                temp = std::fabs(e[i]);
                if (temp>smax)
                {
                    smax = temp;
                }
            }
            real sminl = ZERO, sminoa, mu, thresh;
            if (tol>=ZERO)
            {
                // Relative accuracy desired
                sminoa = std::fabs(d[0]);
                if (sminoa!=ZERO)
                {
                    mu = sminoa;
                    for (i=1; i<n; i++)
                    {
                        mu = std::fabs(d[i]) * (mu/(mu+std::fabs(e[i-1])));
                        if (mu<sminoa)
                        {
                            sminoa = mu;
                        }
                        if (sminoa==ZERO)
                        {
                            break;
                        }
                    }
                }
                sminoa = sminoa / std::sqrt(real(n));
                temp = MAXITR * (n*(n*unfl));
                thresh = tol * sminoa;
                if (temp>thresh)
                {
                    thresh = temp;
                }
            }
            else
            {
                // Absolute accuracy desired
                temp = MAXITR * (n*(n*unfl));
                thresh = std::fabs(tol) * smax;
                if (temp>thresh)
                {
                    thresh = temp;
                }
            }
            // Prepare for main iteration loop for the singular values (MAXIT is the maximum number
            // of passes through the inner loop permitted before nonconvergence signalled.)
            int maxitdivn = MAXITR*n;
            int iterdivn = 0;
            int iter = -1;
            int oldll = -1;
            int oldm = -1;
            // m points to last element of unconverged part of matrix
            int m = n-1;
            // Begin main iteration loop
            int ll, lll;
            real abse, abss, cosl, cosr, f, g, h, oldcs, oldsn, shift, sigmn, sigmx, sinl, sinr,
                 sll;
            bool breakloop1 = false, breakloop2;
            while (true)
            {
                // Check for convergence or exceeding iteration count
                if (m<=0)
                {
                    break;
                }
                if (iter>=n)
                {
                    iter -= n;
                    iterdivn++;
                    if (iterdivn>=maxitdivn)
                    {
                        breakloop1 = true;
                        break;
                    }
                }
                // Find diagonal block of matrix to work on
                if (tol<ZERO && std::fabs(d[m])<=thresh)
                {
                    d[m] = ZERO;
                }
                smax = std::fabs(d[m]);
                smin = smax;
                breakloop2 = false;
                for (lll=0; lll<m; lll++)
                {
                    ll = m - lll + 1;
                    abss = std::fabs(d[ll]);
                    abse = std::fabs(e[ll]);
                    if (tol<ZERO && abss<=thresh)
                    {
                        d[ll] = ZERO;
                    }
                    if (abse<=thresh)
                    {
                        breakloop2 = true;
                        break;
                    }
                    if (abss<smin)
                    {
                        smin = abss;
                    }
                    temp = std::max(abss, abse);
                    if (temp>smax)
                    {
                        smax = temp;
                    }
                }
                if (breakloop2)
                {
                    e[ll] = ZERO;
                    // Matrix splits since E[ll] = 0
                    if (ll==m-1)
                    {
                        // Convergence of bottom singular value, return to top of loop
                        m--;
                        continue;
                    }
                }
                else
                {
                    ll = -1;
                }
                ll++;
                // E[ll] through E[m-1] are nonzero, E[ll-1] is zero
                if (ll==m-1)
                {
                    // 2 by 2 block, handle separately
                    dlasv2(d[m-1], e[m-1], d[m], sigmn, sigmx, sinr, cosr, sinl, cosl);
                    d[m-1] = sigmx;
                    e[m-1] = ZERO;
                    d[m] = sigmn;
                    // Compute singular vectors, if desired
                    if (ncvt>0)
                    {
                        Blas<real>::drot(ncvt, &Vt[m-1/*ldvt*0*/], ldvt, &Vt[m/*+ldvt*0*/], ldvt,
                                      cosr, sinr);
                    }
                    if (nru>0)
                    {
                        Blas<real>::drot(nru, &U[/*0+*/ldu*(m-1)], 1, &U[/*0+*/ldu*m], 1, cosl,
                                         sinl);
                    }
                    if (ncc>0)
                    {
                        Blas<real>::drot(ncc, &C[m-1/*+ldc*0*/], ldc, &C[m/*+ldc*0*/], ldc, cosl,
                                      sinl);
                    }
                    m -= 2;
                    continue;
                }
                // If working on new submatrix, choose shift direction
                // (from larger end diagonal element towards smaller)
                if (ll>oldm || m<oldll)
                {
                    if (std::fabs(d[ll]) >= std::fabs(d[m]))
                    {
                        // Chase bulge from top (big end) to bottom (small end)
                        IDIR = 1;
                    }
                    else
                    {
                        // Chase bulge from bottom (big end) to top (small end)
                        IDIR = 2;
                    }
                }
                // Apply convergence tests
                if (IDIR==1)
                {
                    // Run convergence test in forward direction
                    // First apply standard test to bottom of matrix
                    if (std::fabs(e[m-1])<=std::fabs(tol)*std::fabs(d[m])
                        || (tol<ZERO && std::fabs(e[m-1])<=thresh))
                    {
                        e[m-1] = ZERO;
                        continue;
                    }
                    if (tol>=ZERO)
                    {
                        // If relative accuracy desired, apply convergence criterion forward
                        mu = std::fabs(d[ll]);
                        sminl = mu;
                        breakloop2 = false;
                        for (lll=ll; lll<m; lll++)
                        {
                            if (std::fabs(e[lll]<=tol*mu))
                            {
                                e[lll] = ZERO;
                                breakloop2 = true;
                                break;
                            }
                            mu = std::fabs(d[lll+1]) * (mu/(mu+std::fabs(e[lll])));
                            if (mu<sminl)
                            {
                                sminl = mu;
                            }
                        }
                        if (breakloop2)
                        {
                            continue;
                        }
                    }
                }
                else
                {
                    // Run convergence test in backward direction
                    // First apply standard test to top of matrix
                    if (std::fabs(e[ll])<=std::fabs(tol)*std::fabs(d[ll])
                        || (tol<ZERO && std::fabs(e[ll])<=thresh))
                    {
                        e[ll] = ZERO;
                        continue;
                    }
                    if (tol>=ZERO)
                    {
                        // If relative accuracy desired, apply convergence criterion backward
                        mu = std::fabs(d[m]);
                        sminl = mu;
                        breakloop2 = false;
                        for (lll=m-1; lll>=ll; lll--)
                        {
                            if (std::fabs(e[lll])<=tol*mu)
                            {
                                e[lll] = ZERO;
                                breakloop2 = true;
                                break;
                            }
                            mu = std::fabs(d[lll]) * (mu / (mu+std::fabs(e[lll])));
                            if (mu<sminl)
                            {
                                sminl = mu;
                            }
                        }
                        if (breakloop2)
                        {
                            continue;
                        }
                    }
                }
                oldll = ll;
                oldm = m;
                // Compute shift. First, test if shifting would ruin relative accuracy,
                // and if so set the shift to zero.
                temp = HNDRTH*tol;
                if (tol>=ZERO && n*tol*(sminl/smax)<=((eps>temp)?eps:temp))
                {
                    // Use a zero shift to avoid loss of relative accuracy
                    shift = ZERO;
                }
                else
                {
                    // Compute the shift from 2-by-2 block at end of matrix
                    if (IDIR==1)
                    {
                        sll = std::fabs(d[ll]);
                        dlas2(d[m-1], e[m-1], d[m], shift, r);
                    }
                    else
                    {
                        sll = std::fabs(d[m]);
                        dlas2(d[ll], e[ll], d[ll+1], shift, r);
                    }
                    // Test if shift negligible, and if so set to zero
                    if (sll>ZERO)
                    {
                        temp = shift / sll;
                        if (temp*temp<eps)
                        {
                            shift = ZERO;
                        }
                    }
                }
                // Increment iteration count
                iter += m - ll;
                // If SHIFT = 0, do simplified QR iteration
                if (shift==ZERO)
                {
                    if (IDIR==1)
                    {
                        // Chase bulge from top to bottom
                        // Save cosines and sines for later singular vector updates
                        cs = ONE;
                        oldcs = ONE;
                        for (i=ll; i<m; i++)
                        {
                            dlartg(d[i]*cs, e[i], cs, sn, r);
                            if (i>ll)
                            {
                                e[i-1] = oldsn*r;
                            }
                            dlartg(oldcs*r, d[i+1]*sn, oldcs, oldsn, d[i]);
                            work[i-ll] = cs;
                            work[i-ll+nm1] = sn;
                            work[i-ll+nm12] = oldcs;
                            work[i-ll+nm13] = oldsn;
                        }
                        h = d[m]*cs;
                        d[m] = h*oldcs;
                        e[m-1] = h*oldsn;
                        // Update singular vectors
                        if (ncvt>0)
                        {
                            dlasr("L", "V", "F", m-ll+1, ncvt, &work[0], &work[n-1],
                                  &Vt[ll/*+ldvt*0*/], ldvt);
                        }
                        if (nru>0)
                        {
                            dlasr("R", "V", "F", nru, m-ll+1, &work[nm12], &work[nm13],
                                  &U[/*0+*/ldu*ll], ldu);
                        }
                        if (ncc>0)
                        {
                            dlasr("L", "V", "F", m-ll+1, ncc, &work[nm12], &work[nm13+1],
                                  &C[ll/*+ldc*0*/], ldc);
                        }
                        // Test convergence
                        if (std::fabs(e[m-1])<=thresh)
                        {
                            e[m-1] = ZERO;
                        }
                    }
                    else
                    {
                        // Chase bulge from bottom to top
                        // Save cosines and sines for later singular vector updates
                        cs = ONE;
                        oldcs = ONE;
                        for (i=m; i>ll; i--)
                        {
                            dlartg(d[i]*cs, e[i-1], cs, sn, r);
                            if (i<m)
                            {
                                e[i] = oldsn*r;
                            }
                            dlartg(oldcs*r, d[i-1]*sn, oldcs, oldsn, d[i]);
                            work[i-ll-1] = cs;
                            work[i-ll-1+nm1] = -sn;
                            work[i-ll-1+nm12] = oldcs;
                            work[i-ll-1+nm13] = -oldsn;
                        }
                        h = d[ll]*cs;
                        d[ll] = h*oldcs;
                        e[ll] = h*oldsn;
                        // Update singular vectors
                        if (ncvt>0)
                        {
                            dlasr("L", "V", "B", m-ll+1, ncvt, &work[nm12], &work[nm13],
                                  &Vt[ll/*+ldvt*0*/], ldvt);
                        }
                        if (nru>0)
                        {
                            dlasr("R", "V", "B", nru, m-ll+1, &work[0], &work[n-1],
                                  &U[/*0+*/ldu*ll], ldu);
                        }
                        if (ncc>0)
                        {
                            dlasr("L", "V", "B", m-ll+1, ncc, &work[0], &work[n-1],
                                  &C[ll/*+ldc*0*/], ldc);
                        }
                        // Test convergence
                        if (std::fabs(e[ll])<=thresh)
                        {
                            e[ll] = ZERO;
                        }
                    }
                }
                else
                {
                    // Use nonzero shift
                    if (IDIR==1)
                    {
                        // Chase bulge from top to bottom
                        // Save cosines and sines for later singular vector updates
                        f = (std::fabs(d[ll]) - shift)
                            * ((real(ZERO<=d[ll])-real(ZERO>d[ll])) + shift/d[ll]);
                        g = e[ll];
                        for (i=ll; i<m; i++)
                        {
                            dlartg(f, g, cosr, sinr, r);
                            if (i>ll)
                            {
                                e[i-1] = r;
                            }
                            f    = cosr*d[i] + sinr*e[i];
                            e[i] = cosr*e[i] - sinr*d[i];
                            g      = sinr*d[i+1];
                            d[i+1] = cosr*d[i+1];
                            dlartg(f, g, cosl, sinl, r);
                            d[i] = r;
                            f      = cosl*e[i]   + sinl*d[i+1];
                            d[i+1] = cosl*d[i+1] - sinl*e[i];
                            if (i+1<m)
                            {
                                g = sinl*e[i+1];
                                e[i+1] = cosl*e[i+1];
                            }
                            work[i-ll] = cosr;
                            work[i-ll+nm1] = sinr;
                            work[i-ll+nm12] = cosl;
                            work[i-ll+nm13] = sinl;
                        }
                        e[m-1] = f;
                        // Update singular vectors
                        if (ncvt>0)
                        {
                            dlasr("L", "V", "F", m-ll+1, ncvt, &work[0], &work[n-1],
                                  &Vt[ll/*+ldvt*0*/], ldvt);
                        }
                        if (nru>0)
                        {
                            dlasr("R", "V", "F", nru, m-ll+1, &work[nm12], &work[nm13],
                                  &U[/*0+*/ldu*ll], ldu);
                        }
                        if (ncc>0)
                        {
                            dlasr("L", "V", "F", m-ll+1, ncc, &work[nm12], &work[nm13],
                                  &C[ll/*+ldc*0*/], ldc);
                        }
                        // Test convergence
                        if (std::fabs(e[m-1])<=thresh)
                        {
                            e[m-2] = ZERO;
                        }
                    }
                    else
                    {
                        // Chase bulge from bottom to top
                        // Save cosines and sines for later singular vector updates
                        f = (std::fabs(d[m])-shift) * ((real(ZERO<=d[m])-real(ZERO>d[m]))+shift/d[m]);
                        g = e[m-1];
                        for (i=m; i>ll; i--)
                        {
                            dlartg(f, g, cosr, sinr, r);
                            if (i<m)
                            {
                                e[i] = r;
                            }
                            f      = cosr*d[i] + sinr*e[i-1];
                            e[i-1] = cosr*e[i-1] - sinr*d[i];
                            g      = sinr*d[i-1];
                            d[i-1] = cosr*d[i-1];
                            dlartg(f, g, cosl, sinl, r);
                            d[i] = r;
                            f      = cosl*e[i-1] + sinl*d[i-1];
                            d[i-1] = cosl*d[i-1] - sinl*e[i-1];
                            if (i>ll+1)
                            {
                                g = sinl*e[i-2];
                                e[i-2] = cosl*e[i-2];
                            }
                            work[i-ll-1] = cosr;
                            work[i-ll-1+nm1] = -sinr;
                            work[i-ll-1+nm12] = cosl;
                            work[i-ll-1+nm13] = -sinl;
                        }
                        e[ll] = f;
                        // Test convergence
                        if (std::fabs(e[ll])<=thresh)
                        {
                            e[ll] = ZERO;
                        }
                        // Update singular vectors if desired
                        if (ncvt>0)
                        {
                            dlasr("L", "V", "B", m-ll+1, ncvt, &work[nm12], &work[nm13],
                                  &Vt[ll/*+ldvt*0*/], ldvt);
                        }
                        if (nru>0)
                        {
                            dlasr("R", "V", "B", nru, m-ll+1, &work[0], &work[n-1],
                                  &U[/*0+*/ldu*ll], ldu);
                        }
                        if (ncc>0)
                        {
                            dlasr("L", "V", "B", m-ll+1, ncc, &work[0], &work[n-1],
                                  &C[ll/*+ldc*0*/], ldc);
                        }
                    }
                }
                // QR iteration finished, go back and check convergence
            }
            if (breakloop1)
            {
                // Maximum number of iterations exceeded, failure to converge
                info = 0;
                for (i=0; i<n-1; i++)
                {
                    if (e[i]!=ZERO)
                    {
                        info++;
                    }
                }
                return;
            }
        }
        // All singular values converged, so make them positive
        for (i=0; i<n; i++)
        {
            if (d[i]<ZERO)
            {
                d[i] = -d[i];
                // Change sign of singular vectors, if desired
                if (ncvt>0)
                {
                    Blas<real>::dscal(ncvt, NEGONE, &Vt[i/*+lda*0*/], ldvt);
                }
            }
        }
        // Sort the singular values into decreasing order (insertion sort on singular values,
        // but only one transposition per singular vector)
        int isub, j;
        for (i=0; i<n-1; i++)
        {
            // Scan for smallest d[i]
            isub = 0;
            smin = d[0];
            for (j=1; j<n-i; j++)
            {
                if (d[j]<=smin)
                {
                    isub = j;
                    smin = d[j];
                }
            }
            if (isub!=n-i-1)
            {
                // Swap singular values and vectors
                d[isub] = d[n-i-1];
                d[n-i-1] = smin;
                if (ncvt>0)
                {
                    Blas<real>::dswap(ncvt, &Vt[isub], ldvt, &Vt[n-i-1], ldvt);
                }
                if (nru>0)
                {
                    Blas<real>::dswap(nru, &U[ldu*isub], 1, &U[ldu*(n-i-1)], 1);
                }
                if (ncc>0)
                {
                    Blas<real>::dswap(ncc, &C[isub], ldc, &C[n-i-1], ldc);
                }
            }
        }
    }

    /* dgebal balances a general real matrix A. This involves, first, permuting A by a similarity
     * transformation to isolate eigenvalues in the first 0 to ilo-1 and last ihi+1 to N elements
     * on the diagonal; and second, applying a diagonal similarity transformation to rows and
     * columns ilo to ihi to make the rows and columns as close in norm as possible. Both steps are
     * optional. Balancing may reduce the 1-norm of the matrix, and improve the accuracy of the
     * computed eigenvalues and/or eigenvectors.
     * Parameters: job: Specifies the operations to be performed on A:
     *                  'N': none: simply set ilo = 0, ihi = n-1, scale[i] = 1.0 for i = 0,...,n-1;
     *                  'P':  permute only;
     *                  'S':  scale only;
     *                  'B':  both permute and scale.
     *             n: The order of the matrix A. n >= 0.
     *             A: an array of dimension (lda,n)
     *                On entry, the input matrix A.
     *                On exit, A is overwritten by the balanced matrix.
     *                If job = 'N', A is not referenced. See Further Details.
     *             lda: The leading dimension of the array A. lda >= max(1,n).
     *             ilo,
     *             ihi: ilo and ihi are set to integers such that on exit A[i,j] = 0 if i > j and
     *                  j = 0,...,ilo-1 or i = ihi+1,...,n-1. If job=='N' or 'S', ilo==0 and
     *                  ihi==n-1.
     *                  NOTE: zero-based indices!
     *             scale: an array of dimension (n)
     *                    Details of the permutations and scaling factors applied to A. If P[j] is
     *                    the index of the row and column interchanged with row and column j and
     *                    D[j] is the scaling factor applied to row and column j, then
     *                    scale[j] = P[j]    for j = 0,...,ilo-1
     *                             = D[j]    for j = ilo,...,ihi
     *                             = P[j]    for j = ihi+1,...,n-1.
     *                    The order in which the interchanges are made is n-1 to ihi+1, then 0 to
     *                    ilo-1.
     *             info: = 0:  successful exit.
     *                   < 0:  if info = -i, the i-th argument had an illegal value.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: June 2017
     * Further Details:
     *     The permutations consist of row and column interchanges which put the matrix in the form
     *             ( T1   X   Y  )
     *     P A P = (  0   B   Z  )
     *             (  0   0   T2 )
     *     where T1 and T2 are upper triangular matrices whose eigenvalues lie along the diagonal.
     *     The column indices ilo and ihi mark the starting and ending columns of the submatrix B.
     *     Balancing consists of applying a diagonal similarity transformation inv(D) * B * D to
     *     make the 1-norms of each row of B and its corresponding column nearly equal.
     *     The output matrix is
     *     ( T1     X*D          Y    )
     *     (  0  inv(D)*B*D  inv(D)*Z ).
     *     (  0      0           T2   )
     *     Information about the permutations P and the diagonal matrix D is returned in the
     *     vector scale.
     *     This subroutine is based on the EISPACK routine BALANC.
     *     Modified by Tzu-Yi Chen, Computer Science Division, University of California at
     *     Berkeley, USA                                                                         */
    static void dgebal(char const* job, int n, real* A, int lda, int& ilo, int& ihi, real* scale,
                       int& info)
    {
        const real sclfac = TWO;
        const real factor = real(0.95);
        // Test the input parameters
        info = 0;
        char upjob = toupper(job[0]);
        if (upjob!='N' && upjob!='P' && upjob!='S' && upjob!='B')
        {
            info = -1;
        }
        else if (n<0)
        {
            info = -2;
        }
        else if (lda<1 || lda<n)
        {
            info = -4;
        }
        if (info!=0)
        {
            xerbla("dgebal", -info);
            return;
        }
        int k = 0;
        int l = n-1;
        if (n==0)
        {
            ilo = k;
            ihi = l;
            return;
        }
        int i;
        if (upjob=='N')
        {
            for (i=0; i<n; i++)
            {
                scale[i] = ONE;
            }
            ilo = k;
            ihi = l;
            return;
        }
        if (upjob!='S')
        {
            // Permutation to isolate eigenvalues if possible
            bool backToStart, continueLoop, doRowSearch;
            int iexc = -1;
            int j, m;
            bool notFirstLoop = false;
            while (true)
            {
                doRowSearch = true;
                if (notFirstLoop)
                {
                    // Row and column exchange.
                    scale[m] = j+1;
                    if (j!=m)
                    {
                        Blas<real>::dswap(l+1, &A[/*0+*/lda*j], 1, &A[/*0+*/lda*m], 1);
                        Blas<real>::dswap(n-k, &A[j+lda*k], lda, &A[m+lda*k], lda);
                    }
                    if (iexc==1)
                    {
                        k++;
                        doRowSearch = false;
                    }
                    else
                    {
                        if (l==0)
                        {
                            ilo = k;
                            ihi = l;
                            return;
                        }
                        l--;
                    }
                }
                notFirstLoop = true;
                if (doRowSearch)
                {
                    // Search for rows isolating an eigenvalue and push them down.
                    backToStart = false;
                    for (j=l; j>=0; j--)
                    {
                        continueLoop = false;
                        for (i=0; i<=l; i++)
                        {
                            if (i==j)
                            {
                                continue;
                            }
                            if (A[j+lda*i]!=ZERO)
                            {
                                continueLoop = true;
                                break;
                            }
                        }
                        if (!continueLoop)
                        {
                            m = l;
                            iexc = 0;
                            backToStart = true;
                            break;
                        }
                    }
                    if (backToStart)
                    {
                        continue;
                    }
                }
                // Search for columns isolating an eigenvalue and push them left.
                backToStart = false;
                for (j=k; j<=l; j++)
                {
                    continueLoop = false;
                    for (i=k; i<=l; i++)
                    {
                        if (i==j)
                        {
                            continue;
                        }
                        if (A[i+lda*j]!=ZERO)
                        {
                            continueLoop = true;
                            break;
                        }
                    }
                    if (!continueLoop)
                    {
                        m = k;
                        iexc = 1;
                        backToStart = true;
                        break;
                    }
                }
                if (!backToStart)
                {
                    break;
                }
            }
        }
        for (i=k; i<=l; i++)
        {
            scale[i] = ONE;
        }
        if (upjob!='P')
        {
            // Balance the submatrix in rows k to l.
            real c, ca, f, g, r, ra, s, sfmax1, sfmax2, sfmin1, sfmin2;
            int ica, ira;
            // Iterative loop for norm reduction
            sfmin1 = dlamch("Safemin") / dlamch("Precision");
            sfmax1 = ONE / sfmin1;
            sfmin2 = sfmin1 * sclfac;
            sfmax2 = ONE / sfmin2;
            bool noconv = true;
            while (noconv)
            {
                noconv = false;
                for (i=k; i<=l; i++)
                {
                    c = Blas<real>::dnrm2(l-k+1, &A[k+lda*i], 1);
                    r = Blas<real>::dnrm2(l-k+1, &A[i+lda*k], lda);
                    ica = Blas<real>::idamax(l+1, &A[/*0+*/lda*i], 1);
                    ca = fabs(A[ica+lda*i]);
                    ira = Blas<real>::idamax(n-k, &A[i+lda*k], lda);
                    ra = fabs(A[i+lda*(ira+k)]);
                    // Guard against zero c or R due to underflow.
                    if (c==0.0 || r==0.0)
                    {
                        continue;
                    }
                    g = r / sclfac;
                    f = ONE;
                    s = c + r;
                    while (c<g && f<sfmax2 && c<sfmax2 && ca<sfmax2 && r>sfmin2 && g>sfmin2
                           && ra>sfmin2)
                    {
                        if (std::isnan(c+f+ca+r+g+ra))
                        {
                            //Exit if NaN to avoid infinite loop
                            info = -3;
                            xerbla("DGEBAL", -info);
                            return;
                        }
                        f *= sclfac;
                        c *= sclfac;
                        ca *= sclfac;
                        r /= sclfac;
                        g /= sclfac;
                        ra /= sclfac;
                    }
                    g = c / sclfac;
                    while (g>=r && r<sfmax2 && ra<sfmax2 && f>sfmin2 && c>sfmin2 && g>sfmin2
                           && ca>sfmin2)
                    {
                        f /= sclfac;
                        c /= sclfac;
                        g /= sclfac;
                        ca /= sclfac;
                        r *= sclfac;
                        ra *= sclfac;
                    }
                    //Now balance.
                    if ((c+r)>=factor*s)
                    {
                        continue;
                    }
                    if (f<1.0 && scale[i]<ONE)
                    {
                        if (f*scale[i]<=sfmin1)
                        {
                            continue;
                        }
                    }
                    if (f>1.0 && scale[i]>ONE)
                    {
                        if (scale[i]>=sfmax1/f)
                        {
                            continue;
                        }
                    }
                    g = ONE / f;
                    scale[i] *= f;
                    noconv = true;
                    Blas<real>::dscal(n-k, g, &A[i+lda*k], lda);
                    Blas<real>::dscal(l+1, f, &A[/*0+*/lda*i], 1);
                }
            }
        }
        ilo = k;
        ihi = l;
    }

    /* dgebd2 reduces a real general m by n matrix A to upper or lower bidiagonal form B by an
     * orthogonal transformation: Q^T*A*P = B.
     * If m>=n, B is upper bidiagonal; if m<n, B is lower bidiagonal.
     * Parameters: m: The number of rows in the matrix A. m>=0.
     *             n: The number of columns in the matrix A. n>=0.
     *             A: an array, dimension (lda,n)
     *                On entry, the m by n general matrix to be reduced.
     *                On exit,
     *                  if m>=n, the diagonal and the first superdiagonal are overwritten with the
     *                    upper bidiagonal matrix B; the elements below the diagonal, with the
     *                    array tauq, represent the orthogonal matrix Q as a product of elementary
     *                    reflectors, and the elements above the first superdiagonal, with the
     *                    array taup, represent the orthogonal matrix P as a product of elementary
     *                    reflectors;
     *                  if m<n, the diagonal and the first subdiagonal are overwritten with the
     *                    lower bidiagonal matrix B; the elements below the first subdiagonal, with
     *                    the array tauq, represent the orthogonal matrix Q as a product of
     *                    elementary reflectors, and the elements above the diagonal, with the
     *                    array taup, represent the orthogonal matrix P as a product of elementar
     *                    reflectors.
     *                See Further Details.
     *             lda: The leading dimension of the array A. lda >= max(1,m).
     *             d: an array, dimension (min(m,n))
     *                The diagonal elements of the bidiagonal matrix B: d[i] = A[i,i].
     *             e: an array, dimension (min(m,n)-1)
     *                The off-diagonal elements of the bidiagonal matrix B:
     *                  if m >= n, e[i] = A[i,i+1] for i = 0,1,...,n-2;
     *                  if m < n, e[i] = A[i+1,i] for i = 0,1,...,m-2.
     *             tauq: an array, dimension (min(m,n))
     *                   The scalar factors of the elementary reflectors which represent the
     *                   orthogonal matrix Q. See Further Details.
     *             taup: an array, dimension (min(m,n))
     *                   The scalar factors of the elementary reflectors which represent the
     *                   orthogonal matrix P. See Further Details.
     *             work: an array, dimension (max(m,n))
     *             info: ==0: successful exit.
     *                    <0: if info==-i, the i-th argument had an illegal value.
     * Authors: Univ. of Tennessee
     *          Univ. of California Berkeley
     *          Univ. of Colorado Denver
     *          NAG Ltd.
     * Date: June 2017
     * Further Details:
     *     The matrices Q and P are represented as products of elementary reflectors:
     *     If m >= n,
     *       Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)
     *     Each H(i) and G(i) has the form:
     *       H(i) = I - tauq * v * v^T  and G(i) = I - taup * u * u^T
     *     where tauq and taup are real scalars, and v and u are real vectors;
     *       v[0:i-1] = 0, v[i] = 1, and v[i+1:m-1] is stored on exit in A[i+1:m-1,i];
     *       u[0:i] = 0, u[i+1] = 1, and u[i+2:n-1] is stored on exit in A[i,i+2:n-1];
     *     tauq is stored in tauq[i] and taup in taup[i].
     *     If m < n,
     *       Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)
     *     Each H(i) and G(i) has the form:
     *       H(i) = I - tauq * v * v^T  and G(i) = I - taup * u * u^T
     *     where tauq and taup are real scalars, and v and u are real vectors;
     *       v[0:i] = 0, v[i+1] = 1, and v[i+2:m-1] is stored on exit in A[i+2:m-1,i];
     *       u[0:i-1] = 0, u[i] = 1, and u[i+1:n-1] is stored on exit in A[i,i+1:n-1];
     *     tauq is stored in tauq[i] and taup in taup[i].
     *     The contents of A on exit are illustrated by the following examples:
     *       m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):
     *       (  d   e   u1  u1  u1 )           (  d   u1  u1  u1  u1  u1 )
     *       (  v1  d   e   u2  u2 )           (  e   d   u2  u2  u2  u2 )
     *       (  v1  v2  d   e   u3 )           (  v1  e   d   u3  u3  u3 )
     *       (  v1  v2  v3  d   e  )           (  v1  v2  e   d   u4  u4 )
     *       (  v1  v2  v3  v4  d  )           (  v1  v2  v3  e   d   u5 )
     *       (  v1  v2  v3  v4  v5 )
     *     where d and e denote diagonal and off-diagonal elements of B, vi denotes an element of
     *     the vector defining H(i), and ui an element of the vector defining G(i).              */
    static void dgebd2(int m, int n, real* A, int lda, real* d, real* e, real* tauq, real* taup,
                       real* work, int& info)
    {
        // Test the input parameters
        info = 0;
        if (m<0)
        {
            info = -1;
        }
        else if (n<0)
        {
            info = -2;
        }
        else if (lda<1 || lda<m)
        {
            info = -4;
        }
        if (info<0)
        {
            xerbla("DGEBD2", -info);
            return;
        }
        int i, itemp, ldai, ildai;
        if (m>=n)
        {
            // Reduce to upper bidiagonal form
            for (i=0; i<n; i++)
            {
                // Generate elementary reflector H[i] to annihilate A[i+1:m-1,i]
                itemp = std::min(i+1, m-1);
                ldai = lda*i;
                ildai = i+ldai;
                dlarfg(m-i, A[ildai], &A[itemp+ldai], 1, tauq[i]);
                d[i] = A[ildai];
                A[ildai] = ONE;
                // Apply H(i) to A[i:m-1,i+1:n-1] from the left
                if (i<n-1)
                {
                    dlarf("Left", m-i, n-1-i, &A[ildai], 1, tauq[i], &A[ildai+lda], lda, work);
                }
                A[ildai] = d[i];
                if (i<n-1)
                {
                    // Generate elementary reflector G(i) to annihilate A[i,i+2:n-1]
                    itemp = std::min(i+2, n-1);
                    dlarfg(n-1-i, A[ildai+lda], &A[i+lda*itemp], lda, taup[i]);
                    e[i] = A[ildai+lda];
                    A[ildai+lda] = ONE;
                    // Apply G(i) to A[i+1:m-1,i+1:n-1] from the right
                    dlarf("Right", m-1-i, n-1-i, &A[ildai+lda], lda, taup[i], &A[1+ildai+lda], lda,
                          work);
                    A[ildai+lda] = e[i];
                }
                else
                {
                    taup[i] = ZERO;
                }
            }
        }
        else
        {
            // Reduce to lower bidiagonal form
            for (i=0; i<m; i++)
            {
                // Generate elementary reflector G(i) to annihilate A[i,i+1:n-1]
                itemp = std::min(i+1, n-1);
                ldai = lda*i;
                ildai = i+ldai;
                dlarfg(n-i, A[ildai], &A[i+lda*itemp], lda, taup[i]);
                d[i] = A[ildai];
                A[ildai] = ONE;
                // Apply G(i) to A[i+1:m-1,i:n-1] from the right
                if (i<m-1)
                {
                    dlarf("Right", m-1-i, n-i, &A[ildai], lda, taup[i], &A[1+ildai], lda, work);
                }
                A[ildai] = d[i];
                if (i<m-1)
                {
                    // Generate elementary reflector H(i) to annihilate A[i+2:m-1,i]
                    itemp = std::min(i+2, m-1);
                    dlarfg(m-1-i, A[1+ildai], &A[itemp+ldai], 1, tauq[i]);
                    e[i] = A[1+ildai];
                    A[1+ildai] = ONE;
                    // Apply H(i) to A[i+1:m-1,i+1:n-1] from the left
                    dlarf("Left", m-1-i, n-1-i, &A[1+ildai], 1, tauq[i], &A[1+ildai+lda], lda,
                          work);
                    A[1+ildai] = e[i];
                }
                else
                {
                    tauq[i] = ZERO;
                }
            }
        }
    }

    /* dgeqp3 computes a QR factorization with column pivoting of a matrix A: A*P = Q*R using
     * Level 3 BLAS.
     * Parameters: m: The number of rows of the matrix A. m >= 0.
     *             n: The number of columns of the matrix A. n >= 0.
     *             A: an array, dimension(lda, n)
     *                On entry, the m-by-n matrix A.
     *                On exit, the upper triangle of the array contains the min(m, n)-by-n upper
     *                         trapezoidal matrix R;
     *                the elements below the diagonal, together with the array tau, represent the
     *                orthogonal matrix Q as a product of min(m, n) elementary reflectors.
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     *             jpvt: an integer array, dimension(n)
     *                   On entry, if jpvt[j]!= -1, the j-th column of A is permuted to the front
     *                                              of A*P(a leading column);
     *                             if jpvt[j] = -1, the j-th column of A is a free column.
     *                   On exit,  if jpvt[j] = k,  then the j-th column of A*P was the the k-th
     *                                              column of A.
     *                   Note: this array contains zero-based indices!
     *             tau: an array, dimension(min(m, n))
     *                  The scalar factors of the elementary reflectors.
     *             work: an array, dimension(MAX(1, lwork))
     *                   On exit, if info = 0, work[0] returns the optimal lwork.
     *             lwork: The dimension of the array work. lwork >= 3 * n + 1.
     *                    For optimal performance lwork >= 2*n+(n+1)*nb, where nb is the optimal
     *                    blocksize. If lwork==-1, then a workspace query is assumed; the routine
     *                    only calculates the optimal size of the work array, returns this value as
     *                    the first entry of the work array, and no error message related to lwork
     *                    is issued by xerbla.
     *             info: =0: successful exit.
     *                   <0: if info = -i, the i-th argument had an illegal value.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016
     * Further Details:
     *     The matrix Q is represented as a product of elementary reflectors
     *         Q = H(1) H(2) . ..H(k), where k = min(m, n).
     *     Each H(i) has the form
     *         H(i) = I - tau * v * v^T
     *     where tau is a real scalar, and v is a real/complex vector with v[0:i-1]==0 and v[i]==1;
     *     v[i+1:m-1] is stored on exit in A[i+1:m-1, i], and tau in tau[i].                     */
    static void dgeqp3(int m, int n, real* A, int lda, int* jpvt, real* tau, real* work, int lwork,
                       int& info)
    {
        const int INB = 1, INBMIN = 2, IXOVER = 3;
        // Test input arguments
        info = 0;
        bool lquery = (lwork == -1);
        if (m<0)
        {
            info = -1;
        }
        else if (n<0)
        {
            info = -2;
        }
        else if (lda<1 || lda<m)
        {
            info = -4;
        }
        int iws = 0, minmn = 0, nb;
        if (info==0)
        {
            minmn = std::min(m, n);
            if (minmn == 0)
            {
                iws = 1;
                work[0] = 1;
            }
            else
            {
                iws = 3*n + 1;
                nb = ilaenv(INB, "DGEQRF", " ", m, n, -1, -1);
                work[0] = 2*n + (n+1)*nb;
            }
            if ((lwork<iws) && !lquery)
            {
                info = -8;
            }
        }
        if (info!=0)
        {
            xerbla("DGEQP3", -info);
            return;
        }
        else if (lquery)
        {
            return;
        }
        // Move initial columns up front.
        int j;
        int nfxd = 1;
        for (j=0; j<n; j++)
        {
            if (jpvt[j] != -1)
            {
                if (j != nfxd-1)
                {
                    Blas<real>::dswap(m, &A[/*0+*/lda*j], 1, &A[/*0+*/lda*(nfxd-1)], 1);
                    jpvt[j] = jpvt[nfxd-1];
                    jpvt[nfxd-1] = j;
                }
                else
                {
                    jpvt[j] = j;
                }
                nfxd++;
            }
            else
            {
                jpvt[j] = j;
            }
        }
        nfxd--;
        // Factorize fixed columns
        // Compute the QR factorization of fixed columns and update remaining columns.
        if (nfxd>0)
        {
            int na = std::min(m, nfxd);
            // dgeqr2(m, na, A, lda, tau, work, info);
            dgeqrf(m, na, A, lda, tau, work, lwork, info);
            iws = std::max(iws, int(work[0]));
            if (na<n)
            {
                dormqr("Left", "Transpose", m, n-na, na, A, lda, tau, &A[/*0+*/lda*na], lda, work,
                       lwork, info);
                iws = std::max(iws, int(work[0]));
            }
        }
        // Factorize free columns
        if (nfxd<minmn)
        {
            int sm = m - nfxd;
            int sn = n - nfxd;
            int sminmn = minmn - nfxd;
            // Determine the block size.
            nb = ilaenv(INB, "DGEQRF", " ", sm, sn, -1, -1);
            int nbmin = 2;
            int nx = 0;
            if ((nb>1) && (nb<sminmn))
            {
                // Determine when to cross over from blocked to unblocked code.
                nx = ilaenv(IXOVER, "DGEQRF", " ", sm, sn, -1, -1);
                if (0>nx)
                {
                    nx = 0;
                }
                if (nx<sminmn)
                {
                    // Determine if workspace is large enough for blocked code.
                    int minws = 2*sn + (sn+1)*nb;
                    iws = std::max(iws, minws);
                    if (lwork<minws)
                    {
                        // Not enough workspace to use optimal nb: Reduce nb and determine the
                        // minimum value of nb.
                        nb = (lwork-2*sn) / (sn+1);
                        nbmin = ilaenv(INBMIN, "DGEQRF", " ", sm, sn, -1, -1);
                        if (nbmin<2)
                        {
                            nbmin = 2;
                        }
                    }
                }
            }
            // Initialize partial column norms. The first n elements of work store the exact column
            // norms.
            for (j=nfxd; j<n; j++)
            {
                work[j] = Blas<real>::dnrm2(sm, &A[nfxd+lda*j], 1);
                work[n+j] = work[j];
            }
            if ((nb>=nbmin) && (nb<sminmn) && (nx<sminmn))
            {
                // Use blocked code initially.
                j = nfxd;
                // Compute factorization: while loop.
                int topbmn = minmn - nx;
                int fjb, jb;
                while (j<topbmn)
                {
                    jb = topbmn - j;
                    if (nb<jb)
                    {
                        jb = nb;
                    }
                    // Factorize jb columns among columns j:n-1.
                    dlaqps(m, n-j, j, jb, fjb, &A[lda*j], lda, &jpvt[j], &tau[j], &work[j],
                           &work[n+j], &work[2*n], &work[2*n+jb], n-j);
                    j += fjb;
                }
            }
            else
            {
                j = nfxd;
            }
            // Use unblocked code to factor the last or only block.
            if (j<minmn)
            {

                dlaqp2(m, n-j, j, &A[lda*j], lda, &jpvt[j], &tau[j], &work[j], &work[n+j],
                       &work[2*n]);
            }
        }
        work[0] = iws;
    }

    /* dgeqr2 computes a QR factorization of a real m by n matrix A : A = Q * R.
     * Parameters: m: The number of rows of the matrix A. m >= 0.
     *             n: The number of columns of the matrix A. n >= 0.
     *             A: an array, dimension(lda, n)
     *                On entry, the m by n matrix A.
     *                On exit, the elements on and above the diagonal of the array contain the
     *                         min(m, n) by n upper trapezoidal matrix R (R is upper triangular
     *                         if m>=n);
     *                the elements below the diagonal, with the array tau, represent the orthogonal
     *                matrix Q as a product of elementary reflectors(see Further Details).
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     *             tau: an array, dimension(min(m, n))
     *                  The scalar factors of the elementary reflectors(see Further Details).
     *             work: an array, dimension(n)
     *             info: 0:  successful exit
     *                  <0: if info = -i, the i - th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Further Details:
     *     The matrix Q is represented as a product of elementary reflectors
     *         Q = H(1) H(2) . ..H(k), where k = min(m, n).
     *     Each H(i) has the form
     *         H(i) = I - tau * v * v^T
     *     where tau is a real scalar, and v is a real vector with v[0:i-1] = 0 and v[i] = 1;
     *     v[i+1:m-1] is stored on exit in A[i+1:m-1,i], and tau in tau[i].			 */
    static void dgeqr2(int m, int n, real* A, int lda, real* tau, real* work, int& info)
    {
        int i, k, coli;
        real AII;
        // Test the input arguments
        info = 0;
        if (m<0)
        {
            info = -1;
        }
        else if (n<0)
        {
            info = -2;
        }
        else if (lda<m || lda<1)
        {
            info = -4;
        }
        if (info!=0)
        {
            xerbla("DGEQR2", -info);
            return;
        }
        k = std::min(m, n);
        for (i=0; i<k; i++)
        {
            coli = lda*i;
            // Generate elementary reflector H[i] to annihilate A[i+1:m-1, i]
            dlarfg(m-i, A[i+coli], &A[((i+1<m-1)?i+1:m-1)+coli], 1, tau[i]);
            if (i<(n-1))
            {
                // Apply H[i] to A[i:m-1, i+1:n-1] from the left
                AII = A[i+coli];
                A[i+coli] = ONE;
                dlarf("Left", m-i, n-i-1, &A[i+coli], 1, tau[i], &A[i+coli+lda], lda, work);
                A[i+coli] = AII;
            }
        }
    }

    /* dgeqrf computes a QR factorization of a real m-by-n matrix A:
     * A = Q * R.
     * Parameters: m: The number of rows of the matrix A. m >= 0.
     *             n: The number of columns of the matrix A. n >= 0.
     *             A: an array, dimension(lda, n)
     *                On entry, the m-by-n matrix A.
     *                On exit, the elements on and above the diagonal of the array
     *                contain the min(m, n)-by-n upper trapezoidal matrix R
     *                (R is upper triangular if m >= n); the elements below the diagonal,
     *                with the array tau, represent the orthogonal matrix Q as a
     *                product of min(m, n) elementary reflectors(see Further Details).
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     *             tau: an array, dimension(min(m, n))
     *                  The scalar factors of the elementary reflectors(see Further Details).
     *             work: an array, dimension(max(1, lwork))
     *                   On exit, if info = 0, work[0] returns the optimal lwork.
     *             lwork: The dimension of the array work. lwork >= max(1, n).
     *                    For optimum performance lwork >= n*nb, where nb is
     *                    the optimal blocksize.
     *                    If lwork = -1, then a workspace query is assumed; the routine
     *                    only calculates the optimal size of the work array, returns
     *                    this value as the first entry of the work array, and no error
     *                    message related to lwork is issued by xerbla.
     *             info: =0:  successful exit
     *                   <0:  if info = -i, the i-th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Further Details:
     *     The matrix Q is represented as a product of elementary reflectors
     *         Q = H(1) H(2) . ..H(k), where k = min(m, n).
     *     Each H(i) has the form
     *         H(i) = I - tau * v * v^T
     *     where tau is a real scalar, and v is a real vector with v[0:i-1] = 0 and
     *     v[i] = 1; v[i+1:m-1] is stored on exit in A[i+1:m-1, i], and tau in tau[i].        */
    static void dgeqrf(int m, int n, real* A, int lda, real* tau, real* work, int lwork, int& info)
    {
        // Test the input arguments
        info = 0;
        int nb = ilaenv(1, "DGEQRF", " ", m, n, -1, -1);
        work[0] = n*nb;
        bool lquery = (lwork==-1);
        if (m<0)
        {
            info = -1;
        }
        else if (n<0)
        {
            info = -2;
        }
        else if (lda<1 || lda<m)
        {
            info = -4;
        }
        else if ((lwork<1 || lwork<n) && !lquery)
        {
            info = -7;
        }
        if (info!=0)
        {
            xerbla("DGEQRF", -info);
            return;
        }
        else if (lquery)
        {
            return;
        }
        // Quick return if possible
        int k = std::min(m, n);
        if (k==0)
        {
            work[0] = 1;
            return;
        }
        int nbmin = 2;
        int nx = 0;
        int iws = n;
        int ldwork = 0;
        if (nb>1 && nb<k)
        {
            // Determine when to cross over from blocked to unblocked code.
            nx = ilaenv(3, "DGEQRF", " ", m, n, -1, -1);
            if (nx<0)
            {
                nx = 0;
            }
            if (nx<k)
            {
                // Determine if workspace is large enough for blocked code.
                ldwork = n;
                iws = ldwork*nb;
                if (lwork<iws)
                {
                    //Not enough workspace to use optimal nb: reduce nb and determine the minimum value of nb.
                    nb = lwork / ldwork;
                    nbmin = ilaenv(2, "DGEQRF", " ", m, n, -1, -1);
                    if (nbmin<2)
                    {
                        nbmin = 2;
                    }
                }
            }
        }
        int i, ib, iinfo, aind;
        if (nb>=nbmin && nb<k && nx<k)
        {
            // Use blocked code initially
            for (i=0; i<(k-nx); i+=nb)
            {
                ib = k - i;
                if (ib>nb)
                {
                    ib = nb;
                }
                aind = i + lda*i;
                // Compute the QR factorization of the current block
                //     A[i:m-1, i:i+ib-1]
                dgeqr2(m-i, ib, &A[aind], lda, &tau[i], work, iinfo);
                if ((i+ib)<n)
                {
                    // Form the triangular factor of the block reflector
                    //     H = H(i) H(i + 1) . ..H(i + ib - 1)
                    dlarft("Forward", "Columnwise", m-i, ib, &A[aind], lda, &tau[i], work, ldwork);
                    // Apply H^T to A[i:m-1, i+ib:n-1] from the left
                    dlarfb("Left", "Transpose", "Forward", "Columnwise", m-i, n-i-ib, ib, &A[aind],
                           lda, work, ldwork, &A[aind+lda*ib], lda, &work[ib], ldwork);
                }
            }
        }
        else
        {
            i = 0;
        }
        // Use unblocked code to factor the last or only block.
        if (i<k)
        {

            dgeqr2(m-i, n-i, &A[i+lda*i], lda, &tau[i], work, iinfo);
        }
        work[0] = iws;
    }

    /* disnan replaced by std::isnan */

    /* dlabad takes as input the values computed by dlamch for underflow and overflow, and returns
     * the square root of each of these values if the log of large is sufficiently large. This
     * subroutine is intended to identify machines with a large exponent range, such as the Crays,
     * and redefine the underflow and overflow limits to be the square roots of the values computed
     * by dlamch. This subroutine is needed because dlamch does not compensate for poor arithmetic
     * in the upper half of the exponent range, as is found on a Cray.
     * Parameters: small: On entry, the underflow threshold as computed by dlamch.
     *                    On exit, if log10(large) is sufficiently large, the square root of small,
     *                             otherwise unchanged.
     *             large: On entry, the overflow threshold as computed by dlamch.
     *                    On exit, if log10(large) is sufficiently large, the square root of large,
     *                             otherwise unchanged.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlabad(real& small, real& large)
    {
        // If it looks like we're on a Cray, take the square root of small and large to avoid
        // overflow and underflow problems.
        if (std::log10(large)>real(2000.0))
        {
            small = std::sqrt(small);
            large = std::sqrt(large);
        }
    }

    /* dlacpy copies all or part of a two-dimensional matrix A to another matrix B.
     * Parameters: uplo: Specifies the part of the matrix A to be copied to B.
     *                   'U': Upper triangular part
     *                   'L': Lower triangular part
     *                   Otherwise: All of the matrix A
     *             m: The number of rows of the matrix A. m>=0.
     *             n: The number of columns of the matrix A. n>=0.
     *             A: an array, dimension (lda,n)
     *                The m by n matrix A.
     *                If uplo=='U', only the upper triangle or trapezoid is accessed;
     *                if uplo=='L', only the lower triangle or trapezoid is accessed.
     *             lda: The leading dimension of the array A. lda>=max(1,m).
     *             B: an array, dimension (ldb,n)
     *                On exit, B = A in the locations specified by uplo.
     *             ldb: The leading dimension of the array B. ldb>=max(1,m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016                                                                       */
    static void dlacpy(char const* uplo, int m, int n, real const* A, int lda, real* B, int ldb)
    {
        int i, j, ldaj;
        if (toupper(uplo[0])=='U')
        {
            for (j=0; j<n; j++)
            {
                ldaj = lda*j;
                for (i=0; i<=j && i<m; i++)
                {
                    B[i+ldaj] = A[i+ldaj];
                }
            }
        }
        else if (toupper(uplo[0])=='L')
        {
            for (j=0; j<n; j++)
            {
                ldaj = lda*j;
                for (i=j; i<m; i++)
                {
                    B[i+ldaj] = A[i+ldaj];
                }
            }
        }
        else
        {
            for (j=0; j<n; j++)
            {
                ldaj = lda*j;
                for (i=0; i<m; i++)
                {
                    B[i+ldaj] = A[i+ldaj];
                }
            }
        }
    }

    /* dlaed6 computes the positive or negative root (closest to the origin) of
     *                   z(1)        z(2)         z(3)
     *  f(x) =   rho + --------- + ---------- + ---------
     *                  d(1)-x      d(2)-x       d(3)-x
     * It is assumed that
     *     if orgati==true the root is between d(2) and d(3);
     *     otherwise it is between d(1) and d(2)
     * This routine will be called by dlaed4 when necessary. In most cases, the root sought is the
     * smallest in magnitude, though it might not be in some extremely rare situations.
     * Parameters: kniter: Refer to dlaed4 for its significance.
     *                     NOTE: zero-based
     *             orgati: If orgati is true, the needed root is between d(2) and d(3); otherwise
     *                     it is between d(1) and d(2). See dlaed4 for further details.
     *             rho: Refer to the equation f(x) above.
     *             d: an array, dimension (3)
     *                d satisfies d[0] < d[1] < d[2].
     *             z: an array, dimension (3)
     *                Each of the elements in z must be positive.
     *             finit: The value of f at 0. It is more accurate than the one evaluated inside
     *                    this routine (if someone wants to do so).
     *             tau: The root of the equation f(x).
     *             info: ==0: successful exit
     *                   > 0: if info = 1, failure to converge
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016
     * Further Details:
     *     10/02/03: This version has a few statements commented out for thread safety
     *               (machine parameters are computed on each entry). SJH.
     *     05/10/06: Modified from a new version of Ren-Cang Li, use Gragg-Thornton-Warner cubic
     *               convergent scheme for better stability.
     * Contributors:
     *     Ren-Cang Li, Computer Science Division, University of California at Berkeley, USA     */
    static void dlaed6(int kniter, bool orgati, real rho, real const* d, real const* z, real finit,
                       real& tau, int& info)
    {
        const int MAXIT = 40;
        info = 0;
        real lbd, ubd;
        if (orgati)
        {
            lbd = d[1];
            ubd = d[2];
        }
        else
        {
            lbd = d[0];
            ubd = d[1];
        }
        if (finit < ZERO)
        {
            lbd = ZERO;
        }
        else
        {
            ubd = ZERO;
        }
        int niter = 0;
        tau = ZERO;
        real a, b, c, temp;
        if (kniter==1)
        {
            if (orgati)
            {
                temp = (d[2]-d[1]) / TWO;
                c = rho + z[0] / ((d[0]-d[1])-temp);
                a = c*(d[1]+d[2]) + z[1] + z[2];
                b = c*d[1]*d[2] + z[1]*d[2] + z[2]*d[1];
            }
            else
            {
                temp = (d[0]-d[1]) / TWO;
                c = rho + z[2] / ((d[2]-d[1])-temp);
                a = c*(d[0]+d[1]) + z[0] + z[1];
                b = c*d[0]*d[1] + z[0]*d[1] + z[1]*d[0];
            }
            temp = std::max(std::max(std::fabs(a), std::fabs(b)), std::fabs(c));
            a /= temp;
            b /= temp;
            c /= temp;
            if (c==ZERO)
            {
                tau = b / a;
            }
            else if (a<=ZERO)
            {
                tau = (a-std::sqrt(std::fabs(a*a-FOUR*b*c))) / (TWO*c);
            }
            else
            {
                tau = TWO*b / (a+std::sqrt(std::fabs(a*a-FOUR*b*c)));
            }
            if (tau<lbd || tau>ubd)
            {
                tau = (lbd+ubd) / TWO;
            }
            if (d[0]==tau || d[1]==tau || d[2]==tau)
            {
                tau = ZERO;
            }
            else
            {
                temp = finit + tau*z[0]/(d[0]*(d[0]-tau))
                             + tau*z[1]/(d[1]*(d[1]-tau))
                             + tau*z[2]/(d[2]*(d[2]-tau));
                if (temp<=ZERO)
                {
                   lbd = tau;
                }
                else
                {
                   ubd = tau;
                }
                if (std::fabs(finit)<=std::fabs(temp))
                {
                    tau = ZERO;
                }
            }
        }
        // get machine parameters for possible scaling to avoid overflow
        // modified by Sven: parameters small1, sminv1, small2, sminv2, eps are not saved anymore
        //                   between one call to the others but recomputed at each call
        real eps = dlamch("Epsilon");
        real base = dlamch("Base");
        real small1 = std::pow(base, int(std::log(dlamch("SafMin")) / std::log(base) / THREE));
        real sminv1 = ONE / small1;
        real small2 = small1*small1;
        real sminv2 = sminv1*sminv1;
        // Determine if scaling of inputs necessary to avoid overflow when computing 1/temp^3
        if (orgati)
        {
            temp = std::min(std::fabs(d[1]-tau), std::fabs(d[2]-tau));
        }
        else
        {
            temp = std::min(std::fabs(d[0]-tau), std::fabs(d[1]-tau));
        }
        bool scale = false;
        int i;
        real sclfac, sclinv;
        real dscale[3], zscale[3];
        if (temp<=small1)
        {
            scale = true;
            if (temp<=small2)
            {
                // Scale up by power of radix nearest 1/SAFMIN^(2/3)
                sclfac = sminv2;
                sclinv = small2;
            }
            else
            {
                // Scale up by power of radix nearest 1/SAFMIN^(1/3)
                sclfac = sminv1;
                sclinv = small1;
            }
            // Scaling up safe because d, z, tau scaled elsewhere to be O(1)
            for (i=0; i<3; i++)
            {
                dscale[i] = d[i]*sclfac;
                zscale[i] = z[i]*sclfac;
            }
            tau *= sclfac;
            lbd *= sclfac;
            ubd *= sclfac;
        }
        else
        {
            // Copy d and z to dscale and zscale
            for (i=0; i<3; i++)
            {
                dscale[i] = d[i];
                zscale[i] = z[i];
            }
        }
        real fc = ZERO;
        real df = ZERO;
        real ddf = ZERO;
        real temp1, temp2, temp3;
        for (i=0; i<3; i++)
        {
            temp = ONE / (dscale[i]-tau);
            temp1 = zscale[i]*temp;
            temp2 = temp1*temp;
            temp3 = temp2*temp;
            fc += temp1 / dscale[i];
            df += temp2;
            ddf += temp3;
        }
        real f = finit + tau*fc;
        if (std::fabs(f)>ZERO)
        {
            if (f<=ZERO)
            {
                lbd = tau;
            }
            else
            {
                ubd = tau;
            }
            // Iteration begins -- Use Gragg-Thornton-Warner cubic convergent scheme
            // It is not hard to see that
            //     1) Iterations will go up monotonically if finit < 0;
            //     2) Iterations will go down monotonically if finit > 0.
            int iter = niter + 1;
            real erretm, eta, temp4;
            for (niter=iter; niter<MAXIT; niter++)
            {
                if (orgati)
                {
                    temp1 = dscale[1] - tau;
                    temp2 = dscale[2] - tau;
                }
                else
                {
                    temp1 = dscale[0] - tau;
                    temp2 = dscale[1] - tau;
                }
                a = (temp1+temp2)*f - temp1*temp2*df;
                b = temp1*temp2*f;
                c = f - (temp1+temp2)*df + temp1*temp2*ddf;
                temp = std::max(std::max(std::fabs(a), std::fabs(b)), std::fabs(c));
                a /= temp;
                b /= temp;
                c /= temp;
                if (c==ZERO)
                {
                    eta = b / a;
                }
                else if (a<=ZERO)
                {
                    eta = (a-std::sqrt(std::fabs(a*a-FOUR*b*c))) / (TWO*c);
                }
                else
                {
                    eta = TWO*b / (a+std::sqrt(std::fabs(a*a-FOUR*b*c)));
                }
                if (f*eta>=ZERO)
                {
                    eta = -f / df;
                }
                tau += eta;
                if (tau<lbd || tau>ubd)
                {
                    tau = (lbd + ubd)/TWO;
                }
                fc = ZERO;
                erretm = ZERO;
                df = ZERO;
                ddf = ZERO;
                for (i=0; i<3; i++)
                {
                    if ((dscale[i]-tau)!=ZERO)
                    {
                        temp = ONE / (dscale[i]-tau);
                        temp1 = zscale[i]*temp;
                        temp2 = temp1*temp;
                        temp3 = temp2*temp;
                        temp4 = temp1 / dscale[i];
                        fc += temp4;
                        erretm += std::fabs(temp4);
                        df += temp2;
                        ddf += temp3;
                    }
                    else
                    {
                        // Undo scaling
                        if (scale)
                        {
                            tau *= sclinv;
                        }
                        return;
                    }
                }
                f = finit + tau*fc;
                erretm = EIGHT*(std::fabs(finit)+std::fabs(tau)*erretm) + std::fabs(tau)*df;
                if (std::fabs(f)<=FOUR*eps*erretm || (ubd-lbd)<=FOUR*eps*std::fabs(tau))
                {
                    // Undo scaling
                    if (scale)
                    {
                        tau *= sclinv;
                    }
                    return;
                }
                if (f<=ZERO)
                {
                   lbd = tau;
                }
                else
                {
                   ubd = tau;
                }
            }
            info = 1;
        }
        // Undo scaling
        if (scale)
        {
            tau *= sclinv;
        }
    }

    /* dlamrg will create a permutation list which will merge the elements of a (which is composed
     * of two independently sorted sets) into a single set which is sorted in ascending order.
     * Parameters: n1,
     *             n2: These arguments contain the respective lengths of the two sorted lists to be
     *                 merged.
     *             a: an array, dimension (n1+n2)
     *                The first n1 elements of a contain a list of numbers which are sorted in
     *                either ascending or descending order. Likewise for the final n2 elements.
     *             dtrd1,
     *             dtrd2: These are the strides to be taken through the array a. Allowable strides
     *                    are 1 and -1. They indicate whether a subset of a is sorted in ascending
     *                    (DTRDx==1) or descending (DTRDx==-1) order.
     *             index: an integer array, dimension (n1+n2)
     *                    On exit this array will contain a permutation such that if
     *                    b[i]==a[index[i]] for i=0,n1+n2-1,
     *                    then b will be sorted in ascending order.
     *                    NOTE: Zero-based indices!
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: June 2016                                                                           */
    static void dlamrg(int n1, int n2, real const* a, int dtrd1, int dtrd2, int* index)
    {
        int ind1, ind2;
        if (dtrd1>0)
        {
            ind1 = 0;
        }
        else
        {
            ind1 = n1-1;
        }
        if (dtrd2>0)
        {
            ind2 = n1;
        }
        else
        {
            ind2 = n1 + n2 - 1;
        }
        int i = 0;
        while (n1>0 && n2>0)
        {
            if (a[ind1]<=a[ind2])
            {
                index[i] = ind1;
                i++;
                ind1 += dtrd1;
                n1--;
            }
            else
            {
                index[i] = ind2;
                i++;
                ind2 += dtrd2;
                n2--;
            }
        }
        if (n1==0)
        {
            for (n1=1; n1<=n2; n1++)
            {
                index[i] = ind2;
                i++;
                ind2 += dtrd2;
            }
        }
        else
        {
            // n2==0
            for (n2=1; n2<=n1; n2++)
            {
                index[i] = ind1;
                i++;
                ind1 += dtrd1;
            }
        }
    }

    /* dlange returns the value of the 1-norm, Frobenius norm, infinity-norm, or the largest
     * absolute value of any element of a general real rectangular matrix A.
     * Parameters: norm: Specifies the value to be returned in dlange as described above.
     *             m: The number of rows of the matrix A. m>=0. When m==0, dlange is set to zero.
     *             n: The number of columns of the matrix A. n>=0.
     *                When n==0, dlange is set to zero.
     *             A: an array, dimension (lda,n)
     *                The m by n matrix A.
     *             lda: The leading dimension of the array A. lda>=max(m,1).
     *             work: an array, dimension (MAX(1,lwork)), where lwork>=m when norm = 'I';
     *                   otherwise, work is not referenced.
     * return: dlange = max(abs(A(i,j))), norm = 'M' or 'm'
     *                  norm1(A),         norm = '1', 'O' or 'o'
     *                  normI(A),         norm = 'I' or 'i'
     *                  normF(A),         norm = 'F', 'f', 'E' or 'e'
     *         where norm1 denotes the  one norm of a matrix (maximum column sum), normI denotes
     *         the infinity norm of a matrix (maximum row sum) and normF denotes the Frobenius norm
     *         of a matrix (square root of sum of squares). Note that max(abs(A(i,j))) is not a
     *         consistent matrix norm.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016                                                                       */
    static real dlange(char const* norm, int m, int n, real const* A, int lda, real* work)
    {
        int i, j, ldacol;
        real scale, sum, dlange=ZERO, temp;
        char upNorm = toupper(norm[0]);
        if (m==0 || n==0)
        {
            dlange = ZERO;
        }
        else if (upNorm=='M')
        {
            //Find max(abs(A(i,j))).
            dlange = ZERO;
            for (j=0; j<n; j++)
            {
                ldacol = lda*j;
                for (i=0; i<m; i++)
                {
                    temp = fabs(A[i+ldacol]);
                    if (dlange<temp || std::isnan(temp))
                    {
                        dlange = temp;
                    }
                }
            }
        }
        else if ((upNorm=='O') || (upNorm=='1'))
        {
            //Find norm1(A).
            dlange = ZERO;
            for (j=0; j<n; j++)
            {
                sum = ZERO;
                ldacol = lda*j;
                for (i=0; i<m; i++)
                {
                    sum += fabs(A[i+ldacol]);
                }
                if (dlange<sum || std::isnan(sum))
                {
                    dlange = sum;
                }
            }
        }
        else if (upNorm=='I')
        {
            //Find normI(A).
            for (i=0; i<m; i++)
            {
                work[i] = ZERO;
            }
            for (j=0; j<n; j++)
            {
                ldacol = lda*j;
                for (i=0; i<m; i++)
                {
                    work[i] += fabs(A[i+ldacol]);
                }
            }
            dlange = ZERO;
            for (i=0; i<m; m++)
            {
                temp = work[i];
                if (dlange<temp || std::isnan(temp))
                {
                    dlange = temp;
                }
            }
        }
        else if ((upNorm=='F') || (upNorm=='E'))
        {
            // Find normF(A).
            scale = ZERO;
            sum = ONE;
            for (j=0; j<n; j++)
            {
                dlassq(m, &A[/*0+*/lda*j], 1, scale, sum);
            }
            dlange = scale*std::sqrt(sum);
        }
        return dlange;
    }

    /* dlapy2 returns sqrt(x^2 + y^2), taking care not to cause unnecessary overflow.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.																*/
    static real dlapy2(real x, real Y)
    {
        real w, xabs, yabs, z;
        xabs = fabs(x);
        yabs = fabs(Y);
        if (xabs>yabs)
        {
            w = xabs;
            z = yabs;
        }
        else
        {
            w = yabs;
            z = xabs;
        }
        if (z==ZERO)
        {
            return w;
        }
        else
        {
            return w * sqrt(ONE + (z/w)*(z/w));
        }
    }

    /* dlaqp2 computes a QR factorization with column pivoting of the block A[offset:m-1, 0:n-1].
     * The block A[0:offset-1, 0:n-1] is accordingly pivoted, but not factorized.
     * Parameters: m: The number of rows of the matrix A. m>=0.
     *             n: The number of columns of the matrix A. n>=0.
     *             offset: The number of rows of the matrix A that must be pivoted but not
     *                     factorized. offset>=0.
     *             A: an array, dimension(lda, n)
     *                On entry, the m-by-n matrix A.
     *                On exit, the upper triangle of block A[offset:m-1, 0:n-1] is the triangular
     *                         factor obtained; the elements in block A[offset:m-1, 0:n-1] below
     *                         the diagonal, together with the array tau, represent the orthogonal
     *                         matrix Q as a product of elementary reflectors. Block
     *                         A[0:offset-1,0:n-1] has been accordingly pivoted,but not factorized.
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     *             jpvt: an integer array, dimension(n)
     *                   On entry, if jpvt[i] != -1, the i-th column of A is permuted to the front
     *                                               of A*P (a leading column);
     *                             if jpvt[i] == -1, the i-th column of A is a free column.
     *                   On exit,  if jpvt[i] == k, then the i-th column of A*P was the k-th column
     *                                              of A.
     *                   Note: this array contains zero-based indices
     *             tau: an array, dimension(min(m, n))
     *                  The scalar factors of the elementary reflectors.
     *             vn1: an array, dimension(n)
     *                  The vector with the partial column norms.
     *             vn2: an array, dimension(n)
     *                  The vector with the exact column norms.
     *             work: an array, dimension(n)
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                         */
    static void dlaqp2(int m, int n, int offset, real* A, int lda, int* jpvt, real* tau, real* vn1,
                       real* vn2, real* work)
    {
        int mn = std::min(m-offset, n);
        real tol3z = sqrt(dlamch("Epsilon"));
        // Compute factorization.
        int i, itemp, j, offpi, pvt, acoli;
        real aii, temp, temp2;
        for (i=0; i<mn; i++)
        {
            offpi = offset + i;
            // Determine ith pivot column and swap if necessary.
            pvt = i + Blas<real>::idamax(n-i, &vn1[i], 1);
            acoli = lda*i;
            if (pvt!=i)
            {
                Blas<real>::dswap(m, &A[lda*pvt], 1, &A[acoli], 1);
                itemp = jpvt[pvt];
                jpvt[pvt] = jpvt[i];
                jpvt[i] = itemp;
                vn1[pvt] = vn1[i];
                vn2[pvt] = vn2[i];
            }
            // Generate elementary reflector H(i).
            if (offpi<m-1)
            {
                dlarfg(m-offpi, A[offpi+acoli], &A[offpi+1+acoli], 1, tau[i]);
            }
            else
            {
                dlarfg(1, A[m-1+acoli], &A[m-1+acoli], 1, tau[i]);
            }
            if (i+1<n)
            {
                // Apply H(i)^T to A[offset+i:m-1, i+1:n-1] from the left.
                aii = A[offpi+acoli];
                A[offpi+acoli] = ONE;
                dlarf("Left", m-offpi, n-i-1, &A[offpi+acoli], 1, tau[i], &A[offpi+acoli+lda], lda,
                      work);
                A[offpi+acoli] = aii;
            }
            // Update partial column norms.
            for (j=i+1; j<n; j++)
            {
                if (vn1[j]!=ZERO)
                {
                    // NOTE: The following 6 lines follow from the analysis in Lapack Working Note 176.
                    temp = fabs(A[offpi+lda*j]) / vn1[j];
                    temp = ONE - temp*temp;
                    temp = std::max(temp, ZERO);
                    temp2 = vn1[j] / vn2[j];
                    temp2 = temp * temp2 * temp2;
                    if (temp2<=tol3z)
                    {
                        if (offpi<m-1)
                        {
                            vn1[j] = Blas<real>::dnrm2(m-offpi-1, &A[offpi+1+lda*j], 1);
                            vn2[j] = vn1[j];
                        }
                        else
                        {
                            vn1[j] = ZERO;
                            vn2[j] = ZERO;
                        }
                    }
                    else
                    {
                        vn1[j] *= sqrt(temp);
                    }
                }
            }
        }
    }

    /* dlaqps computes a step of QR factorization with column pivotingof a real m-by-n matrix A by
     * using Blas-3. It tries to factorize nb columns from A starting from the row offset + 1, and
     * updates all of the matrix with Blas-3 xgemm. In some cases, due to catastrophic
     * cancellations, it cannot factorize nb columns. Hence, the actual number of factorized
     * columns is returned in kb. Block A[0:offset-1, 0:n-1] is accordingly pivoted, but not
     * factorized.
     * Parameters: m: The number of rows of the matrix A. m >= 0.
     *             n: The number of columns of the matrix A. n >= 0
     *             offset: The number of rows of A that have been factorized in previous steps.
     *             nb: The number of columns to factorize.
     *             kb: The number of columns actually factorized.
     *             A: an array, dimension(lda, n)
     *                On entry, the m-by-n matrix A.
     *                On exit, block A[offset:m-1, 0:kb-1] is the triangular factor obtained and
     *                block A[0:offset-1, 0:n-1] has been accordingly pivoted, but no factorized.
     *                The rest of the matrix, block A[offset:m-1, kb:n-1] has been updated.
     *             lda: The leading dimension of the array A.lda >= max(1, m).
     *             jpvt: an integer array, dimension(n)
     *                   jpvt[i] = k <==> Column k of the full matrix A has been permuted into
     *                                    position i in AP.
     *                   Note: this array contains zero-based indices
     *             tau: an array, dimension(kb)
     *                  The scalar factors of the elementary reflectors.
     *             vn1: an array, dimension(n)
     *                  The vector with the partial column norms.
     *             vn2: an array, dimension(n)
     *                  The vector with the exact column norms.
     *             auxv: an array, dimension(nb)
     *                   Auxiliar vector.
     *             F: an array, dimension(ldf, nb)
     *                Matrix F^T = L*Y^T*A.
     *             ldf: The leading dimension of the array F. ldf >= max(1, n).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                         */
    static void dlaqps(int m, int n, int offset, int nb, int& kb, real* A, int lda, int* jpvt,
                       real* tau, real* vn1, real* vn2, real* auxv, real* F, int ldf)
    {
        int lastrk = std::min(m, n+offset);
        int lsticc = -1;
        int k = -1;
        real tol3z = sqrt(dlamch("Epsilon"));
        // Beginning of while loop.
        int itemp, j, pvt, rk, acolk;
        real akk, temp, temp2;
        while ((k+1<nb) && (lsticc==-1))
        {
            k++;
            rk = offset + k;
            acolk = lda*k;
            // Determine i-th pivot column and swap if necessary
            pvt = k + Blas<real>::idamax(n-k, &vn1[k], 1);
            if (pvt!=k)
            {
                Blas<real>::dswap(m, &A[lda*pvt], 1, &A[acolk], 1);
                Blas<real>::dswap(k, &F[pvt], ldf, &F[k], ldf);
                itemp = jpvt[pvt];
                jpvt[pvt] = jpvt[k];
                jpvt[k] = itemp;
                vn1[pvt] = vn1[k];
                vn2[pvt] = vn2[k];
            }
            // Apply previous Householder reflectors to column k:
            //     A[rk:m-1, k] -= A[rk:m-1, 0:k-1] * F[k, 0:k-1]^T.
            if (k>0)
            {
                Blas<real>::dgemv("No transpose", m-rk, k, -ONE, &A[rk], lda, &F[k], ldf, ONE,
                               &A[rk+acolk], 1);
            }
            // Generate elementary reflector H(k).
            if (rk<m-1)
            {
                dlarfg(m-rk, A[rk+acolk], &A[rk+1+acolk], 1, tau[k]);
            }
            else
            {
                dlarfg(1, A[rk+acolk], &A[rk+acolk], 1, tau[k]);
            }
            akk = A[rk+acolk];
            A[rk+acolk] = ONE;
            // Compute k-th column of F:
            // Compute  F[k+1:n-1, k] = tau[k] * A[rk:m-1, k+1:n-1]^T * A[rk:m-1, k].
            if (k<n-1)
            {
                Blas<real>::dgemv("Transpose", m-rk, n-k-1, tau[k], &A[rk+acolk+lda], lda,
                               &A[rk+acolk], 1, ZERO, &F[k+1+ldf*k], 1);
            }
            // Padding F[0:k, k] with zeros.
            for (j=0; j<=k; j++)
            {
                F[j+ldf*k] = ZERO;
            }
            // Incremental updating of F:
            // F[0:n-1, k] -= tau[k] * F[0:n-1, 0:k-1] * A[rk:m-1, 0:k-1]^T * A[rk:m-1, k].
            if (k>0)
            {
                Blas<real>::dgemv("Transpose", m-rk, k, -tau[k], &A[rk], lda, &A[rk+acolk], 1,
                                  ZERO, auxv, 1);
                Blas<real>::dgemv("No transpose", n, k, ONE, F, ldf, auxv, 1, ONE, &F[ldf*k], 1);
            }
            // Update the current row of A:
            // A[rk, k+1:n-1] -= A[rk, 0:k] * F[k+1:n-1, 0:k]^T.
            if (k<n-1)
            {
                Blas<real>::dgemv("No transpose", n-k-1, k+1, -ONE, &F[k+1], ldf, &A[rk], lda,
                               ONE, &A[rk+acolk+lda], lda);
            }
            // Update partial column norms.
            if (rk < lastrk-1)
            {
                for (j=k+1; j<n; j++)
                {
                    if (vn1[j]!=ZERO)
                    {
                        // NOTE: The following 6 lines follow from the analysis in Lapack Working
                        //       Note 176.
                        temp = fabs(A[rk+lda*j]) / vn1[j];
                        temp = (ONE+temp) * (ONE-temp);
                        temp = std::max(ZERO, temp);
                        temp2 = vn1[j] / vn2[j];
                        temp2 = temp * temp2 * temp2;
                        if (temp2<=tol3z)
                        {
                            vn2[j] = real(lsticc+1);
                            lsticc = j;
                        }
                        else
                        {
                            vn1[j] *= sqrt(temp);
                        }
                    }
                }
            }
            A[rk+acolk] = akk;
        }
        kb = k + 1;
        rk = offset + kb - 1;
        // Apply the block reflector to the rest of the matrix:
        // A[offset+kb:m-1, kb:n-1] -= A[offset+kb:m-1, 0:kb-1] * F[kb:n-1, 0:kb-1]^T.
        if (kb < ((n<=m-offset)?n:m-offset))
        {
            Blas<real>::dgemm("No transpose", "Transpose", m-rk-1, n-kb, kb, -ONE, &A[rk+1], lda,
                           &F[kb], ldf, ONE, &A[rk+1+lda*kb], lda);
        }
        // Recomputation of difficult columns.
        while (lsticc>=0)
        {
            itemp = std::round(vn2[lsticc]) - 1;
            vn1[lsticc] = Blas<real>::dnrm2(m-rk-1, &A[rk+1+lda*lsticc], 1);
            // NOTE: The computation of vn1[lsticc] relies on the fact that dnrm2 does not fail on
            // vectors with norm below the value of sqrt(dlamch("S"))
            vn2[lsticc] = vn1[lsticc];
            lsticc = itemp;
        }
    }

    /* dlarf applies a real elementary reflector H to a real m by n matrix
     * C, from either the left or the right.H is represented in the form
     *     H = I - tau * v * v^T
     * where tau is a real scalar and v is a real vector.
     * If tau = 0, then H is taken to be the unit matrix.
     * Parameters: side: 'L': form  H * C
     *                   'R': form  C * H
     *             m: The number of rows of the matrix C.
     *             n: The number of columns of the matrix C.
     *             v: an array, dimension (1 + (m - 1)*abs(incv)) if side = 'L'
     *                                 or (1 + (n - 1)*abs(incv)) if side = 'R'
     *                The vector v in the representation of H. v is not used if tau = 0.
     *             incv: The increment between elements of v. incv <> 0.
     *             tau: The value tau in the representation of H.
     *             C: an array, dimension(ldc, n)
     *                          On entry, the m by n matrix C.
     *                          On exit, C is overwritten by the matrix H * C if side = 'L',
     *                                                               or C * H if side = 'R'.
     *             ldc: The leading dimension of the array C. ldc >= max(1, m).
     *             work: an array, dimension (n) if side = 'L'
     *                                    or (m) if side = 'R'
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                         */
    static void dlarf(char const* side, int m, int n, real const* v, int incv, real tau, real* C,
                      int ldc, real* work)
    {
        bool applyleft;
        int i, lastv=0, lastc=0;
        applyleft = (toupper(side[0])=='L');
        if (tau!=ZERO)
        {
            //Set up variables for scanning v. LASTV begins pointing to the end of v.
            if (applyleft)
            {
                lastv = m;
            }
            else
            {
                lastv = n;
            }
            if (incv>0)
            {
                i = (lastv-1) * incv;
            }
            else
            {
                i = 0;
            }
            // Look for the last non - zero row in v.
            while (lastv>0 && v[i]==ZERO)
            {
                lastv--;
                i -= incv;
            }
            if (applyleft)
            {
                // Scan for the last non - zero column in C[0:lastv-1,:].
                lastc = iladlc(lastv, n, C, ldc) + 1;
            }
            else
            {
                // Scan for the last non - zero row in C[:,0:lastv-1].
                lastc = iladlr(m, lastv, C, ldc) + 1;
            }
        }
        // Note that lastc==0 renders the BLAS operations null;
        // no special case is needed at this level.
        if (applyleft)
        {
            //Form  H * C
            if (lastv>0)
            {
                // work[0:lastc-1] = C[0:lastv-1,0:last-1]^T * v[0:lastv-1]
                Blas<real>::dgemv("Transpose", lastv, lastc, ONE, C, ldc, v, incv, ZERO, work, 1);
                // C[0:lastv-1,0:lastc-1] -= v[0:lastv-1] * work[0:lastc-1]^T
                Blas<real>::dger(lastv, lastc, -tau, v, incv, work, 1, C, ldc);
            }
        }
        else
        {
            // Form  C * H
            if (lastv>0)
            {
                // work[0:lastc-1] = C[0:lastc-1,0:lastv-1] * v[0:lastv-1]
                Blas<real>::dgemv("No transpose", lastc, lastv, ONE, C, ldc, v, incv, ZERO, work,
                                  1);
                // C[0:lastc-1,0:lastv-1] -= work[0:lastc-1] * v[0:lastv-1]^T
                Blas<real>::dger(lastc, lastv, -tau, work, 1, v, incv, C, ldc);
            }
        }
    }

    /* dlarfb applies a real block reflector H or its transpose H^T to a real m by n matrix C,
     * from either the left or the right.
     * Parameters: side: 'L': apply H or H^T from the Left
     *                   'R': apply H or H**T from the Right
     *             trans: 'N': apply H (No transpose)
     *                    'T': apply H^T (Transpose)
     *             direct: Indicates how H is formed from a product of elementary reflectors
     *                     'F': H = H(1) H(2) . ..H(k) (Forward)
     *                     'B': H = H(k) . ..H(2) H(1) (Backward)
     *             storev: Indicates how the vectors which define the elementary reflectors
     *                     are stored:
     *                     'C': Columnwise
     *                     'R': Rowwise
     *             m: The number of rows of the matrix C.
     *             n: The number of columns of the matrix C.
     *             k: The order of the matrix T(= the number of elementary reflectors
     *                whose product defines the block reflector).
     *             V: an array, dimension (ldv, k) if storev = 'C'
     *                                    (ldv, m) if storev = 'R' and side = 'L'
     *                                    (ldv, n) if storev = 'R' and side = 'R'
     *             ldv: he leading dimension of the array V.
     *                  If storev = 'C' and side = 'L', ldv >= max(1, m);
     *                  if storev = 'C' and side = 'R', ldv >= max(1, n);
     *                  if storev = 'R', ldv >= k.
     *             T: an array, dimension(ldt, k)
     *                The triangular k by k matrix T in the representation of the block reflector.
     *             ldt: The leading dimension of the array T. ldt >= k.
     *             C: an array, dimension(ldc, n)
     *                On entry, the m by n matrix C.
     *                On exit, C is overwritten by H * C or H^T * C or C * H or C * H^T.
     *             ldc: The leading dimension of the array C. ldc >= max(1, m).
     *             Work: an array, dimension(ldwork, k)
     *             ldwork: The leading dimension of the array Work.
     *                     If side = 'L', ldwork >= max(1, n);
     *                     if side = 'R', ldwork >= max(1, m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Further Details:
     *     The shape of the matrix V and the storage of the vectors which define the H(i) is best
     *     illustrated by the following example with n==5 and k==3. The elements equal to 1 are not
     *     stored; the corresponding array elements are modified but restored on exit. The rest of
     *     the array is not used.
     *     direct = 'F' and storev = 'C':        direct = 'F' and storev = 'R':
     *                  V = (1)                  V = (1 v1 v1 v1 v1)
     *                      (v1  1)                     (1 v2 v2 v2)
     *                      (v1 v2  1)                     (1 v3 v3)
     *                      (v1 v2 v3)
     *                      (v1 v2 v3)
     *     direct = 'B' and storev = 'C':         direct = 'B' and storev = 'R' :
     *                  V = (v1 v2 v3)                 V = (v1 v1  1)
     *                      (v1 v2 v3)                     (v2 v2 v2  1)
     *                      (1 v2 v3)                      (v3 v3 v3 v3  1)
     *                      (1 v3)
     *                      (1)                                                                  */
    static void dlarfb(char const* side, char const* trans, char const* direct, char const* storev,
                       int m, int n, int k, real* V, int ldv, real const* T, int ldt, real* C,
                       int ldc, real* Work, int ldwork)
    {
        // Quick return if possible
        if (m<0 || n<0)
        {
            return;
        }
        char const* transt;
        if (toupper(trans[0])=='N')
        {
            transt = "Transpose";
        }
        else
        {
            transt = "No transpose";
        }
        int i, j, ccol, workcol;
        char upstorev = toupper(storev[0]);
        char updirect = toupper(direct[0]);
        char upside = toupper(side[0]);
        if (upstorev=='C')
        {
            if (updirect=='F')
            {
                // Let V = (V1) (first k rows)
                //         (V2)
                // where V1 is unit lower triangular.
                if (upside=='L')
                {
                    // Form  H * C or H^T * C  where  C = (C1)
                    //                                    (C2)
                    // W = C^T * V = (C1^T * V1 + C2^T * V2)  (stored in Work)
                    // W = C1^T
                    for (j=0; j<k; j++)
                    {
                        Blas<real>::dcopy(n, &C[j], ldc, &Work[ldwork*j], 1);
                    }
                    // W : = W * V1
                    Blas<real>::dtrmm("Right", "Lower", "No transpose", "Unit", n, k, ONE, V, ldv,
                                      Work, ldwork);
                    if (m>k)
                    {
                        // W = W + C2^T * V2
                        Blas<real>::dgemm("Transpose", "No transpose", n, k, m-k, ONE, &C[k], ldc,
                                          &V[k], ldv, ONE, Work, ldwork);
                    }
                    // W = W * T^T or W * T
                    Blas<real>::dtrmm("Right", "Upper", transt, "Non-unit", n, k, ONE, T, ldt,
                                      Work, ldwork);
                    // C = C - V * W^T
                    if (m>k)
                    {
                        // C2 = C2 - V2 * W^T
                        Blas<real>::dgemm("No transpose", "Transpose", m-k, n, k, -ONE, &V[k], ldv,
                                          Work, ldwork, ONE, &C[k], ldc);
                    }
                    // W = W * V1^T
                    Blas<real>::dtrmm("Right", "Lower", "Transpose", "Unit", n, k, ONE, V, ldv,
                                      Work, ldwork);
                    // C1 = C1 - W^T
                    for (j=0; j<k; j++)
                    {
                        workcol = ldwork * j;
                        for (i=0; i<n; i++)
                        {
                            C[j+ldc*i] -= Work[i+workcol];
                        }
                    }
                }
                else if (upside=='R')
                {
                    // Form  C * H or C * H^T  where  C = (C1  C2)
                    // W = C * V = (C1*V1 + C2*V2)  (stored in Work)
                    // W = C1
                    for (j=0; j<k; j++)
                    {
                        Blas<real>::dcopy(m, &C[ldc*j], 1, &Work[ldwork*j], 1);
                    }
                    // W = W * V1
                    Blas<real>::dtrmm("Right", "Lower", "No transpose", "Unit", m, k, ONE, V, ldv,
                                      Work, ldwork);
                    if (n>k)
                    {
                        // W = W + C2 * V2
                        Blas<real>::dgemm("No transpose", "No transpose", m, k, n-k, ONE,
                                          &C[ldc*k], ldc, &V[k], ldv, ONE, Work, ldwork);
                    }
                    // W = W * T or W * T^T
                    Blas<real>::dtrmm("Right", "Upper", trans, "Non-unit", m, k, ONE, T, ldt,
                                      Work, ldwork);
                    // C = C - W * V^T
                    if (n>k)
                    {
                        // C2 = C2 - W * V2^T
                        Blas<real>::dgemm("No transpose", "Transpose", m, n-k, k, -ONE, Work,
                                          ldwork, &V[k], ldv, ONE, &C[ldc*k], ldc);
                    }
                    // W = W * V1^T
                    Blas<real>::dtrmm("Right", "Lower", "Transpose", "Unit", m, k, ONE, V, ldv,
                                      Work, ldwork);
                    // C1 = C1 - W
                    for (j=0; j<k; j++)
                    {
                        ccol = ldc * j;
                        workcol = ldwork * j;
                        for (i=0; i<m; i++)
                        {
                            C[i+ccol] -= Work[i+workcol];
                        }
                    }
                }
            }
            else
            {
                // Let V = (V1)
                //         (V2) (last k rows)
                // where V2 is unit upper triangular.
                if (upside=='L')
                {
                    // Form  H * C or H^T * C  where  C = (C1)
                    //                                    (C2)
                    // W = C^T * V = (C1^T * V1 + C2^T * V2)  (stored in Work)
                    // W = C2^T
                    for (j=0; j<k; j++)
                    {
                        Blas<real>::dcopy(n, &C[m-k+j], ldc, &Work[ldwork*j], 1);
                    }
                    // W = W * V2
                    Blas<real>::dtrmm("Right", "Upper", "No transpose", "Unit", n, k, ONE, &V[m-k],
                                      ldv, Work, ldwork);
                    if (m>k)
                    {
                        // W = W + C1^T * V1
                        Blas<real>::dgemm("Transpose", "No transpose", n, k, m-k, ONE, C, ldc, V,
                                          ldv, ONE, Work, ldwork);
                    }
                    // W = W * T^T or W * T
                    Blas<real>::dtrmm("Right", "Lower", transt, "Non-unit", n, k, ONE, T, ldt,
                                      Work, ldwork);
                    // C = C - V * W^T
                    if (m>k)
                    {
                        // C1 = C1 - V1 * W^T
                        Blas<real>::dgemm("No transpose", "Transpose", m-k, n, k, -ONE, V, ldv,
                                          Work, ldwork, ONE, C, ldc);
                    }
                    // W = W * V2^T
                    Blas<real>::dtrmm("Right", "Upper", "Transpose", "Unit", n, k, ONE, &V[m-k],
                                      ldv, Work, ldwork);
                    // C2 = C2 - W^T
                    for (j=0; j<k; j++)
                    {
                        ccol = m - k + j;
                        workcol = ldwork * j;
                        for (i = 0; i<n; i++)
                        {
                            C[ccol+ldc*i] -= Work[i+workcol];
                        }
                    }
                }
                else if (upside=='R')
                {
                    // Form  C * H or C * H^T  where  C = (C1  C2)
                    // W = C * V = (C1*V1 + C2*V2)  (stored in Work)
                    // W = C2
                    ccol = ldc * (n-k);
                    for (j=0; j<k; j++)
                    {
                        Blas<real>::dcopy(m, &C[ccol+ldc*j], 1, &Work[ldwork*j], 1);
                    }
                    // W = W * V2
                    Blas<real>::dtrmm("Right", "Upper", "No transpose", "Unit", m, k, ONE, &V[n-k],
                                      ldv, Work, ldwork);
                    if (n>k)
                    {
                        // W = W + C1 * V1
                        Blas<real>::dgemm("No transpose", "No transpose", m, k, n-k, ONE, C, ldc,
                                          V, ldv, ONE, Work, ldwork);
                    }
                    // W : = W * T or W * T^T
                    Blas<real>::dtrmm("Right", "Lower", trans, "Non-unit", m, k, ONE, T, ldt,
                                      Work, ldwork);
                    // C = C - W * V^T
                    if (n>k)
                    {
                        // C1 = C1 - W * V1^T
                        Blas<real>::dgemm("No transpose", "Transpose", m, n-k, k, -ONE, Work,
                                          ldwork, V, ldv, ONE, C, ldc);
                    }
                    // W = W * V2^T
                    Blas<real>::dtrmm("Right", "Upper", "Transpose", "Unit", m, k, ONE, &V[n-k],
                                      ldv, Work, ldwork);
                    // C2 = C2 - W
                    for (j=0; j<k; j++)
                    {
                        ccol = ldc * (n-k+j);
                        workcol = ldwork * j;
                        for (i=0; i<m; i++)
                        {
                            C[i+ccol] -= Work[i+workcol];
                        }
                    }
                }
            }
        }
        else if (upstorev=='R')
        {
            if (updirect=='F')
            {
                // Let V = (V1  V2)   (V1: first k columns)
                // where V1 is unit upper triangular.
                if (upside=='L')
                {
                    // Form H * C or H^T * C where C = (C1)
                    //                                 (C2)
                    // W = C^T * V^T = (C1^T * V1^T + C2^T * V2^T) (stored in Work)
                    // W = C1^T
                    for (j=0; j<k; j++)
                    {
                        Blas<real>::dcopy(n, &C[j], ldc, &Work[ldwork*j], 1);
                    }
                    // W = W * V1^T
                    Blas<real>::dtrmm("Right", "Upper", "Transpose", "Unit", n, k, ONE, V, ldv,
                                      Work, ldwork);
                    if (m>k)
                    {
                        // W = W + C2^T * V2^T
                        Blas<real>::dgemm("Transpose", "Transpose", n, k, m-k, ONE, &C[k], ldc,
                                          &V[ldv*k], ldv, ONE, Work, ldwork);
                    }
                    // W = W * T^T or W * T
                    Blas<real>::dtrmm("Right", "Upper", transt, "Non-unit", n, k, ONE, T, ldt,
                                      Work, ldwork);
                    // C = C - V^T * W^T
                    if (m>k)
                    {
                        // C2 = C2 - V2^T * W^T
                        Blas<real>::dgemm("Transpose", "Transpose", m-k, n, k, -ONE, &V[ldv*k],
                                          ldv, Work, ldwork, ONE, &C[k], ldc);
                    }
                    // W = W * V1
                    Blas<real>::dtrmm("Right", "Upper", "No transpose", "Unit", n, k, ONE, V, ldv,
                                      Work, ldwork);
                    // C1 = C1 - W^T
                    for (j=0; j<k; j++)
                    {
                        workcol = ldwork*j;
                        for (i=0; i<n; i++)
                        {
                            C[j+ldc*i] -= Work[i+workcol];
                        }
                    }
                }
                else if (upside=='R')
                {
                    // Form C * H or C * H^T where  C = (C1  C2)
                    // W = C * V^T = (C1*V1^T + C2*V2^T)  (stored in Work)
                    // W = C1
                    for (j=0; j<k; j++)
                    {
                        Blas<real>::dcopy(m, &C[ldc*j], 1, &Work[ldwork*j], 1);
                    }
                    // W = W * V1^T
                    Blas<real>::dtrmm("Right", "Upper", "Transpose", "Unit", m, k, ONE, V, ldv,
                                      Work, ldwork);
                    if (n>k)
                    {
                        // W = W + C2 * V2^T
                        Blas<real>::dgemm("No transpose", "Transpose", m, k, n-k, ONE, &C[ldc*k],
                                          ldc, &V[ldv*k], ldv, ONE, Work, ldwork);
                    }
                    // W = W * T or W * T^T
                    Blas<real>::dtrmm("Right", "Upper", trans, "Non-unit", m, k, ONE, T, ldt,
                                      Work, ldwork);
                    // C = C - W * V
                    if (n>k)
                    {
                        // C2 = C2 - W * V2
                        Blas<real>::dgemm("No transpose", "No transpose", m, n-k, k, -ONE, Work,
                                          ldwork, &V[ldv*k], ldv, ONE, &C[ldc*k], ldc);
                    }
                    // W = W * V1
                    Blas<real>::dtrmm("Right", "Upper", "No transpose", "Unit", m, k, ONE, V, ldv,
                                      Work, ldwork);
                    // C1 = C1 - W
                    for (j=0; j<k; j++)
                    {
                        ccol = ldc * j;
                        workcol = ldwork * j;
                        for (i=0; i<m; i++)
                        {
                            C[i+ccol] -= Work[i+workcol];
                        }
                    }
                }
            }
            else
            {
                // Let V = (V1  V2)   (V2: last k columns)
                // where V2 is unit lower triangular.
                if (upside=='L')
                {
                    // Form H * C or H^T * C where C = (C1)
                    //                                 (C2)
                    // W = C^T * V^T = (C1^T * V1^T + C2^T * V2^T) (stored in Work)
                    // W = C2^T
                    for (j=0; j<k; j++)
                    {
                        Blas<real>::dcopy(n, &C[m-k+j], ldc, &Work[ldwork*j], 1);
                    }
                    // W = W * V2^T
                    Blas<real>::dtrmm("Right", "Lower", "Transpose", "Unit", n, k, ONE,
                                      &V[ldv*(m-k)], ldv, Work, ldwork);
                    if (m>k)
                    {
                        // W = W + C1^T * V1^T
                        Blas<real>::dgemm("Transpose", "Transpose", n, k, m-k, ONE, C, ldc, V, ldv,
                                          ONE, Work, ldwork);
                    }
                    // W = W * T^T or W * T
                    Blas<real>::dtrmm("Right", "Lower", transt, "Non-unit", n, k, ONE, T, ldt,
                                      Work, ldwork);
                    // C = C - V^T * W^T
                    if (m>k)
                    {
                        // C1 = C1 - V1^T * W^T
                        Blas<real>::dgemm("Transpose", "Transpose", m-k, n, k, -ONE, V, ldv, Work,
                                          ldwork, ONE, C, ldc);
                    }
                    // W = W * V2
                    Blas<real>::dtrmm("Right", "Lower", "No transpose", "Unit", n, k, ONE,
                                      &V[ldv*(m-k)], ldv, Work, ldwork);
                    // C2 = C2 - W^T
                    for (j=0; j<k; j++)
                    {
                        ccol = m - k + j;
                        workcol = ldwork * j;
                        for (i=0; i<n; i++)
                        {
                            C[ccol+ldc*i] -= Work[i+workcol];
                        }
                    }
                }
                else if (upside=='R')
                {
                    // Form  C * H or C * H^T where C = (C1  C2)
                    // W = C * V^T = (C1*V1^T + C2*V2^T) (stored in Work)
                    // W = C2
                    ccol = ldc * (n-k);
                    for (j=0; j<k; j++)
                    {
                        Blas<real>::dcopy(m, &C[ccol+ldc*j], 1, &Work[ldwork*j], 1);
                    }
                    // W = W * V2^T
                    Blas<real>::dtrmm("Right", "Lower", "Transpose", "Unit", m, k, ONE,
                                      &V[ldv*(n-k)], ldv, Work, ldwork);
                    if (n>k)
                    {
                        // W = W + C1 * V1^T
                        Blas<real>::dgemm("No transpose", "Transpose", m, k, n-k, ONE, C, ldc, V,
                                          ldv, ONE, Work, ldwork);
                    }
                    // W = W * T or W * T^T
                    Blas<real>::dtrmm("Right", "Lower", trans, "Non-unit", m, k, ONE, T, ldt,
                                      Work, ldwork);
                    // C = C - W * V
                    if (n>k)
                    {
                        // C1 = C1 - W * V1
                        Blas<real>::dgemm("No transpose", "No transpose", m, n-k, k, -ONE, Work,
                                          ldwork, V, ldv, ONE, C, ldc);
                    }
                    // W = W * V2
                    Blas<real>::dtrmm("Right", "Lower", "No transpose", "Unit", m, k, ONE,
                                      &V[ldv*(n-k)], ldv, Work, ldwork);
                    // C1 = C1 - W
                    for (j=0; j<k; j++)
                    {
                        ccol = ldc * (n-k+j);
                        workcol = ldwork * j;
                        for (i=0; i<m; i++)
                        {
                            C[i+ccol] -= Work[i+workcol];
                        }
                    }
                }
            }
        }
    }

    /* dlarfg generates a real elementary reflector H of order n, such that
     *           H * (alpha) = (beta), H^T * H = I.
     *               (x    )   (0   )
     * where alpha and beta are scalars, and x is an (n - 1)-element real vector.
     * H is represented in the form
     *           H = I - tau * (1) * (1 v^T),
     *                         (v)
     * where tau is a real scalar and v is a real (n - 1)-element vector.
     * If the elements of x are all zero, then tau = 0 and H is taken to be the unit matrix.
     * Otherwise  1 <= tau <= 2.
     * Parameters: n: The order of the elementary reflector.
     *             alpha: On entry, the value alpha.
     *                    On exit, it is overwritten with the value beta.
     *             x: array, dimension (1 + (n - 2)*abs(incx))
     *                On entry, the vector x.
     *                On exit, it is overwritten with the vector v.
     *             incx: The increment between elements of x. incx > 0.
     *             tau:  The value tau.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                         */
    static void dlarfg(int n, real& alpha, real* x, int incx, real& tau)
    {
        int j, knt;
        real beta, rsafmin, safmin, xnorm;
        if (n<=1)
        {
            tau = ZERO;
            return;
        }
        xnorm = Blas<real>::dnrm2(n-1, x, incx);
        if (xnorm==ZERO)
        {
            // H = I
            tau = ZERO;
        }
        else
        {
            // general case
            beta = -std::copysign(dlapy2(alpha, xnorm), alpha);
            safmin = dlamch("SafeMin") / dlamch("Epsilon");
            knt = 0;
            if (fabs(beta)<safmin)
            {
                // xnorm, beta may be inaccurate; scale x and recompute them
                rsafmin = ONE / safmin;
                do
                {
                    knt++;
                    Blas<real>::dscal(n-1, rsafmin, x, incx);
                    beta *= rsafmin;
                    alpha *= rsafmin;
                } while (fabs(beta)<safmin);
                // New beta is at most 1, at least SAFMIN
                xnorm = Blas<real>::dnrm2(n-1, x, incx);
                beta = -std::copysign(dlapy2(alpha, xnorm), alpha);
            }
            tau = (beta-alpha) / beta;
            Blas<real>::dscal(n-1, ONE/(alpha-beta), x, incx);
            // If alpha is subnormal, it may lose relative accuracy
            for (j=0; j<knt; j++)
            {
                beta *= safmin;
            }
            alpha = beta;
        }
    }

    /* dlarft forms the triangular factor T of a real block reflector H
     * of order n, which is defined as a product of k elementary reflectors.
     *     If direct = 'F', H = H(1) H(2) . ..H(k) and T is upper triangular;
     *     If direct = 'B', H = H(k) . ..H(2) H(1) and T is lower triangular.
     *     If storev = 'C', the vector which defines the elementary reflector
     *                      H(i) is stored in the i - th column of the array V, and
     *                      H = I - V * T * V^T
     *     If storev = 'R', the vector which defines the elementary reflector
     *                      H(i) is stored in the i - th row of the array V, and
     *                      H = I - V^T * T * V
     * Parameters: direct:  Specifies the order in which the elementary reflectors are
     *                      multiplied to form the block reflector:
     *                      'F': H = H(1) H(2) . ..H(k) (Forward)
     *                      'B': H = H(k) . ..H(2) H(1) (Backward)
     *             storev: Specifies how the vectors which define the elementary
     *                     reflectors are stored(see also Further Details) :
     *                     'C': columnwise
     *                     'R': rowwise
     *             n: The order of the block reflector H. n >= 0.
     *             k: The order of the triangular factor T(= the number of elementary reflectors).
     *                k >= 1.
     *             V: an array, dimension (ldv, k) if storev = 'C'
     *                                    (ldv, n) if storev = 'R'
     *             ldv: The leading dimension of the array V.
     *                  If storev = 'C', ldv >= max(1, n); if storev = 'R', ldv >= k.
     *             tau: an array, dimension(k)
     *                  tau(i) must contain the scalar factor of the elementary reflector H(i).
     *             T: an array, dimension(lda, k)
     *                The k by k triangular factor T of the block reflector.
     *                If direct=='F', T is upper triangular;
     *                if direct=='B', T is lower triangular. The rest of the array is not used.
     *             lda: The leading dimension of the array T. lda >= k.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Further Details:
     *     The shape of the matrix V and the storage of the vectors which define the H(i) is best
     *     illustrated by the following example with n==5 and k==3. The elements equal to 1 are not
     *     stored.
     * direct = 'F' and storev = 'C':         direct = 'F' and storev = 'R':
     *                V = (1)                 V = (1 v1 v1 v1 v1)
     *                    (v1  1)                    (1 v2 v2 v2)
     *                    (v1 v2  1)                    (1 v3 v3)
     *                    (v1 v2 v3)
     *                    (v1 v2 v3)
     * direct = 'B' and storev = 'C':         direct = 'B' and storev = 'R' :
     *                V = (v1 v2 v3)                 V = (v1 v1  1)
     *                    (v1 v2 v3)                     (v2 v2 v2  1)
     *                    (1 v2 v3)                      (v3 v3 v3 v3  1)
     *                    (1 v3)
     *                    (1)                                                                    */
    static void dlarft(char const* direct, char const* storev, int n, int k, real const* V, int ldv,
                       real const* tau, real* T, int lda)
    {
        // Quick return if possible
        if (n==0)
        {
            return;
        }
        char updirect = toupper(direct[0]);
        char upstorev = toupper(storev[0]);
        int i, j, prevlastv, lastv, tcoli, vcol;
        if (updirect=='F')
        {
            prevlastv = n-1;
            for (i=0; i<k; i++)
            {
                tcoli = lda * i;
                if (i>prevlastv)
                {
                    prevlastv = i;
                }
                if (tau[i]==ZERO)
                {
                    // H(i) = I
                    for (j=0; j<=i; j++)
                    {
                        T[j+tcoli] = ZERO;
                    }
                }
                else
                {
                    // general case
                    if (upstorev=='C')
                    {
                        vcol = ldv * i;
                        // Skip any trailing zeros.
                        for (lastv=n-1; lastv>i; lastv--)
                        {
                            if (V[lastv+vcol]!=ZERO)
                            {
                                break;
                            }
                        }
                        for (j = 0; j<i; j++)
                        {
                            T[j+tcoli] = -tau[i]*V[i+ldv*j];
                        }
                        j = std::min(lastv, prevlastv);
                        // T[0:i-1, i] = -tau[i] * V[i:j, 0:i-1]^T * V[i:j, i]
                        Blas<real>::dgemv("Transpose", j-i, i, -tau[i], &V[i+1], ldv, &V[i+1+vcol],
                                          1, ONE, &T[tcoli], 1);
                    }
                    else
                    {
                        // Skip any trailing zeros.
                        for (lastv=n-1; lastv>i; lastv--)
                        {
                            if (V[i+ldv*lastv]!=ZERO)
                            {
                                break;
                            }
                        }
                        vcol = ldv*i;
                        for (j=0; j<i; j++)
                        {
                            T[j+tcoli] = -tau[i]*V[j+vcol];
                        }
                        j = std::min(lastv, prevlastv);
                        // T[0:i-1, i] = -tau[i] * V[0:i-1, i:j] * V[i, i:j]^T
                        Blas<real>::dgemv("No transpose", i, j-i, -tau[i], &V[vcol+ldv], ldv,
                                          &V[i+vcol+ldv], ldv, ONE, &T[tcoli], 1);
                    }
                    // T[0:i-1, i] = T[0:i-1, 0:i-1] * T[0:i-1, i]
                    Blas<real>::dtrmv("Upper", "No transpose", "Non-unit", i, T, lda, &T[tcoli],
                                      1);
                    T[i+tcoli] = tau[i];
                    if (i>0)
                    {
                        if (lastv>prevlastv)
                        {
                            prevlastv = lastv;
                        }
                    }
                    else
                    {
                        prevlastv = lastv;
                    }
                }
            }
        }
        else
        {
            prevlastv = 0;
            for (i=k-1; i>=0; i--)
            {
                tcoli = lda * i;
                if (tau[i]==ZERO)
                {
                    // H(i) = I
                    for (j=i; j<k; j++)
                    {
                        T[j+tcoli] = ZERO;
                    }
                }
                else
                {
                    // general case
                    if (i<k-1)
                    {
                        if (upstorev=='C')
                        {
                            vcol = ldv * i;
                            // Skip any leading zeros.
                            for (lastv=0; lastv<i; lastv++)
                            {
                                if (V[lastv+vcol]!=ZERO)
                                {
                                    break;
                                }
                            }
                            vcol = n - k - i; // misuse: not a col, but a row
                            for (j=(i+1); j<k; j++)
                            {
                                T[j+tcoli] = -tau[i] * V[vcol+ldv*j];
                            }
                            j = std::max(lastv, prevlastv);
                            // T[i+1:k-1, i] = -tau[i] * V[j:n-k+i, i+1:k-1]^T * V[j:n-k+i, i]
                            Blas<real>::dgemv("Transpose", vcol-j, k-1-i, -tau[i], &V[j+ldv*(i+1)],
                                              ldv, &V[j+ldv*i], 1, ONE, &T[i+1+tcoli], 1);
                        }
                        else
                        {
                            // Skip any leading zeros.
                            for (lastv=0; lastv<i; lastv++)
                            {
                                if (V[i+ldv*lastv]!=ZERO)
                                {
                                    break;
                                }
                            }
                            vcol = ldv * (n-k+i);
                            for (j=(i+1); j<k; j++)
                            {
                                T[j+tcoli] = -tau[i]*V[j+vcol];
                            }
                            j = std::max(lastv, prevlastv);
                            // T[i+1:k-1, i] = -tau[i] * V[i+1:k-1, j:n-k+i] * V[i, j:n-k+i]^T
                            vcol = ldv*j;
                            Blas<real>::dgemv("No transpose", k-1-i, n-k+i-j, -tau[i],
                                              &V[i+1+vcol], ldv, &V[i+vcol], ldv, ONE,
                                              &T[i+1+tcoli], 1);
                        }
                        // T[i+1:k-1, i] = T[i+1:k-1, i+1:k-1] * T[i+1:k-1, i]
                        Blas<real>::dtrmv("Lower", "No transpose", "Non-unit", k-i-1,
                                          &T[i+1+lda*(i+1)], lda, &T[i+1+tcoli], 1);
                        if (i>0)
                        {
                            if (lastv<prevlastv)
                            {
                                prevlastv = lastv;
                            }
                        }
                        else
                        {
                            prevlastv = lastv;
                        }
                    }
                    T[i+tcoli] = tau[i];
                }
            }
        }
    }

    /* dlarnv returns a vector of n random real numbers from a uniform or normal distribution.
     * Parameters: idist: Specifies the distribution of the random numbers:
     *                    ==1: uniform (0,1)
     *                    ==2: uniform (-1,1)
     *                    ==3: normal (0,1)
     *             iseed: an integer array, dimension (4)
     *                    On entry, the seed of the random number generator; the array elements
     *                              must be between 0 and 4095, and iseed[3] must be odd.
     *                    On exit, the seed is updated.
     *             n: The number of random numbers to be generated.
     *             x: an array, dimension (n)
     *                The generated random numbers.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Further Details:
     *     This routine calls the auxiliary routine dlaruv to generate random real numbers from a
     *     uniform (0,1) distribution, in batches of up to 128 using vectorisable code. The Box-
     *     Muller method is used to transform numbers from a uniform to a normal distribution.   */
    static void dlarnv(int idist, int* iseed, int n, real* x)
    {
        const real TWOPI = real(6.2831853071795864769252867663);
        const int LV = 128;
        int i, il, il2, iv;
        real u[LV];
        for (iv=0; iv<n; iv+=LV/2)
        {
            il = LV/2;
            if (n-iv<il)
            {
                il = n-iv;
            }
            if (idist==3)
            {
                il2 = 2*il;
            }
            else
            {
                il2 = il;
            }
            // Call dlaruv to generate il2 numbers from a uniform (0,1) distribution (il2 <= LV)
            dlaruv(iseed, il2, u);
            if (idist==1)
            {
                // Copy generated numbers
                for (i=0; i<il; i++)
                {
                    x[iv+i] = u[i];
                }
            }
            else if (idist==2)
            {
                // Convert generated numbers to uniform (-1,1) distribution
                for (i=0; i<il; i++)
                {
                    x[iv+i] = TWO*u[i] - ONE;
                }
            }
            else if (idist==3)
            {
                // Convert generated numbers to normal (0,1) distribution
                for (i=0; i<il; i++)
                {
                    x[iv+i] = std::sqrt(-TWO*std::log(u[2*i])) * std::cos(TWOPI*u[2*i+1]);
                }
            }
        }
    }

    /* dlartg generates a plane rotation so that
     *     [  cs  sn  ]  .  [ f ]  =  [ r ]   where cs^2 + sn^2 = 1.
     *     [ -sn  cs  ]     [ g ]     [ 0 ]
     * This is a slower, more accurate version of the BLAS1 routine DROTG,
     * with the following other differences:
     *     f and g are unchanged on return.
     *     If g==0, then cs==1 and sn==0.
     *     If f==0 and g!=0, then cs==0 and sn==1 without doing any floating point operations
     *         (saves work in DBDSQR when there are zeros on the diagonal).
     *     If f exceeds g in magnitude, cs will be positive.
     * Parameters: f: The first component of vector to be rotated.
     *             g: The second component of vector to be rotated.
     *             cs: The cosine of the rotation.
     *             sn: The sine of the rotation.
     *             r: The nonzero component of the rotated vector.
     * This version has a few statements commented out for thread safety (machine parameters are
     *   computed on each entry). 10 feb 03, SJH.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlartg(real f, real g, real& cs, real& sn, real& r)
    {
        //static bool first = true;
        /*static*/ real safmn2, safmx2;
        //if (first)
        //{
        real safmin = dlamch("S");
        real eps = dlamch("E");
        real base = dlamch("B");
        safmn2 = std::pow(base, std::log(safmin/eps)/std::log(base)/TWO);
        safmx2 = ONE/safmn2;
        //first = false;
        //}
        if (g==ZERO)
        {
            cs = ONE;
            sn = ZERO;
            r = f;
        }
        else if (f==ZERO)
        {
            cs = ZERO;
            sn = ONE;
            r = g;
        }
        else
        {
            real f1 = f;
            real g1 = g;
            real scale = fabs(f1);
            scale = std::max(scale, fabs(g1));
            int i, count = 0;
            if (scale>=safmx2)
            {
                do
                {
                    count++;
                    f1 *= safmn2;
                    g1 *= safmn2;
                    scale = fabs(f1);
                    scale = std::max(scale, fabs(g1));
                } while (scale>=safmx2);
                r = std::sqrt(f1*f1+g1*g1);
                cs = f1 / r;
                sn = g1 / r;
                for (i=0; i<count; i++)
                {
                    r *= safmx2;
                }
            }
            else if (scale<=safmn2)
            {
                do
                {
                    count++;
                    f1 *= safmx2;
                    g1 *= safmx2;
                    scale = fabs(f1);
                    scale = std::max(scale, fabs(g1));
                } while (scale<=safmn2);
                r = std::sqrt(f1*f1+g1*g1);
                cs = f1 / r;
                sn = g1 / r;
                for (i=0; i<count; i++)
                {
                    r *= safmn2;
                }
            }
            else
            {
                r = std::sqrt(f1*f1+g1*g1);
                cs = f1 / r;
                sn = g1 / r;
            }
            if (fabs(f)>fabs(g) && cs<ZERO)
            {
                cs = -cs;
                sn = -sn;
                r = -r;
            }
        }
    }

    /* dlaruv returns a vector of n random real numbers from a uniform (0,1) distribution (n<=128).
     * This is an auxiliary routine called by dlarnv and zlarnv.
     * Parameters: iseed: an integer array, dimension (4)
     *                    On entry, the seed of the random number generator; the array elements
     *                              must be between 0 and 4095, and iseed[3] must be odd.
     *                    On exit, the seed is updated.
     *             n: The number of random numbers to be generated. n<=128.
     *             x: an array, dimension (n)
     *                The generated random numbers.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Further Details:
     *    This routine uses a multiplicative congruential method with modulus 2^48 and multiplier
     *    33952834046453 (see G.S.Fishman, 'Multiplicative congruential random number generators
     *    with modulus 2^b: an exhaustive analysis for b = 32 and a partial analysis for b = 48',
     *    Math. Comp. 189, pp 331-344, 1990).
     *    48-bit integers are stored in 4 integer array elements with 12 bits per element. Hence
     *    the routine is portable across machines with integers of 32 bits or more.              */
    static void dlaruv(int* iseed, int n, real* x)
    {
        const int LV = 128;
        const int IPW2 = 4096;
        const real R = ONE / IPW2;
        const int MM[4*LV] = {
            /*  0*/  494,  322, 2508, 2549,
            /*  1*/ 2637,  789, 3754, 1145,
            /*  2*/  255, 1440, 1766, 2253,
            /*  3*/ 2008,  752, 3572,  305,
            /*  4*/ 1253, 2859, 2893, 3301,
            /*  5*/ 3344,  123,  307, 1065,
            /*  6*/ 4084, 1848, 1297, 3133,
            /*  7*/ 1739,  643, 3966, 2913,
            /*  8*/ 3143, 2405,  758, 3285,
            /*  9*/ 3468, 2638, 2598, 1241,
            /* 10*/  688, 2344, 3406, 1197,
            /* 11*/ 1657,   46, 2922, 3729,
            /* 12*/ 1238, 3814, 1038, 2501,
            /* 13*/ 3166,  913, 2934, 1673,
            /* 14*/ 1292, 3649, 2091,  541,
            /* 15*/ 3422,  339, 2451, 2753,
            /* 16*/ 1270, 3808, 1580,  949,
            /* 17*/ 2016,  822, 1958, 2361,
            /* 18*/  154, 2832, 2055, 1165,
            /* 19*/ 2862, 3078, 1507, 4081,
            /* 20*/  697, 3633, 1078, 2725,
            /* 21*/ 1706, 2970, 3273, 3305,
            /* 22*/  491,  637,   17, 3069,
            /* 23*/  931, 2249,  854, 3617,
            /* 24*/ 1444, 2081, 2916, 3733,
            /* 25*/  444, 4019, 3971,  409,
            /* 26*/ 3577, 1478, 2889, 2157,
            /* 27*/ 3944,  242, 3831, 1361,
            /* 28*/ 2184,  481, 2621, 3973,
            /* 29*/ 1661, 2075, 1541, 1865,
            /* 30*/ 3482, 4058,  893, 2525,
            /* 31*/  657,  622,  736, 1409,
            /* 32*/ 3023, 3376, 3992, 3445,
            /* 33*/ 3618,  812,  787, 3577,
            /* 34*/ 1267,  234, 2125,   77,
            /* 35*/ 1828,  641, 2364, 3761,
            /* 36*/  164, 4005, 2460, 2149,
            /* 37*/ 3798, 1122,  257, 1449,
            /* 38*/ 3087, 3135, 1574, 3005,
            /* 39*/ 2400, 2640, 3912,  225,
            /* 40*/ 2870, 2302, 1216,   85,
            /* 41*/ 3876,   40, 3248, 3673,
            /* 42*/ 1905, 1832, 3401, 3117,
            /* 43*/ 1593, 2247, 2124, 3089,
            /* 44*/ 1797, 2034, 2762, 1349,
            /* 45*/ 1234, 2637,  149, 2057,
            /* 46*/ 3460, 1287, 2245,  413,
            /* 47*/  328, 1691,  166,   65,
            /* 48*/ 2861,  496,  466, 1845,
            /* 49*/ 1950, 1597, 4018,  697,
            /* 50*/  617, 2394, 1399, 3085,
            /* 51*/ 2070, 2584,  190, 3441,
            /* 52*/ 3331, 1843, 2879, 1573,
            /* 53*/  769,  336,  153, 3689,
            /* 54*/ 1558, 1472, 2320, 2941,
            /* 55*/ 2412, 2407,   18,  929,
            /* 56*/ 2800,  433,  712,  533,
            /* 57*/  189, 2096, 2159, 2841,
            /* 58*/  287, 1761, 2318, 4077,
            /* 59*/ 2045, 2810, 2091,  721,
            /* 60*/ 1227,  566, 3443, 2821,
            /* 61*/ 2838,  442, 1510, 2249,
            /* 62*/  209,   41,  449, 2397,
            /* 63*/ 2770, 1238, 1956, 2817,
            /* 64*/ 3654, 1086, 2201,  245,
            /* 65*/ 3993,  603, 3137, 1913,
            /* 66*/  192,  840, 3399, 1997,
            /* 67*/ 2253, 3168, 1321, 3121,
            /* 68*/ 3491, 1499, 2271,  997,
            /* 69*/ 2889, 1084, 3667, 1833,
            /* 70*/ 2857, 3438, 2703, 2877,
            /* 71*/ 2094, 2408,  629, 1633,
            /* 72*/ 1818, 1589, 2365,  981,
            /* 73*/  688, 2391, 2431, 2009,
            /* 74*/ 1407,  288, 1113,  941,
            /* 75*/  634,   26, 3922, 2449,
            /* 76*/ 3231,  512, 2554,  197,
            /* 77*/  815, 1456,  184, 2441,
            /* 78*/ 3524,  171, 2099,  285,
            /* 79*/ 1914, 1677, 3228, 1473,
            /* 80*/  516, 2657, 4012, 2741,
            /* 81*/  164, 2270, 1921, 3129,
            /* 82*/  303, 2587, 3452,  909,
            /* 83*/ 2144, 2961, 3901, 2801,
            /* 84*/ 3480, 1970,  572,  421,
            /* 85*/  119, 1817, 3309, 4073,
            /* 86*/ 3357,  676, 3171, 2813,
            /* 87*/  837, 1410,  817, 2337,
            /* 88*/ 2826, 3723, 3039, 1429,
            /* 89*/ 2332, 2803, 1696, 1177,
            /* 90*/ 2089, 3185, 1256, 1901,
            /* 91*/ 3780,  184, 3715,   81,
            /* 92*/ 1700,  663, 2077, 1669,
            /* 93*/ 3712,  499, 3019, 2633,
            /* 94*/  150, 3784, 1497, 2269,
            /* 95*/ 2000, 1631, 1101,  129,
            /* 96*/ 3375, 1925,  717, 1141,
            /* 97*/ 1621, 3912,   51,  249,
            /* 98*/ 3090, 1398,  981, 3917,
            /* 99*/ 3765, 1349, 1978, 2481,
            /*100*/ 1149, 1441, 1813, 3941,
            /*101*/ 3146, 2224, 3881, 2217,
            /*102*/   33, 2411,   76, 2749,
            /*103*/ 3082, 1907, 3846, 3041,
            /*104*/ 2741, 3192, 3694, 1877,
            /*105*/  359, 2786, 1682,  345,
            /*106*/ 3316,  382,  124, 2861,
            /*107*/ 1749,   37, 1660, 1809,
            /*108*/  185,  759, 3997, 3141,
            /*109*/ 2784, 2948,  479, 2825,
            /*110*/ 2202, 1862, 1141,  157,
            /*111*/ 2199, 3802,  886, 2881,
            /*112*/ 1364, 2423, 3514, 3637,
            /*113*/ 1244, 2051, 1301, 1465,
            /*114*/ 2020, 2295, 3604, 2829,
            /*115*/ 3160, 1332, 1888, 2161,
            /*116*/ 2785, 1832, 1836, 3365,
            /*117*/ 2772, 2405, 1990,  361,
            /*118*/ 1217, 3638, 2058, 2685,
            /*119*/ 1822, 3661,  692, 3745,
            /*120*/ 1245,  327, 1194, 2325,
            /*121*/ 2252, 3660,   20, 3609,
            /*122*/ 3904,  716, 3285, 3821,
            /*123*/ 2774, 1842, 2046, 3537,
            /*124*/  997, 3987, 2107,  517,
            /*125*/ 2573, 1368, 3508, 3017,
            /*126*/ 1148, 1848, 3525, 2141,
            /*127*/  545, 2366, 3801, 1537};
        int i1 = iseed[0];
        int i2 = iseed[1];
        int i3 = iseed[2];
        int i4 = iseed[3];
        int i, im4, it1=0, it2=0, it3=0, it4=0;
        for (i=0; i<n && i<LV; i++)
        {
            im4 = 4*i;
            do
            {
                // Multiply the seed by i-th power of the multiplier modulo 2^48
                it4 = i4  * MM[3+im4];
                it3 = it4 / IPW2;
                it4 = it4 - IPW2*it3;
                it3 = it3 + i3*MM[3+im4] + i4*MM[2+im4];
                it2 = it3 / IPW2;
                it3 = it3 - IPW2*it2;
                it2 = it2 + i2*MM[3+im4] + i3*MM[2+im4] + i4*MM[1+im4];
                it1 = it2 / IPW2;
                it2 = it2 - IPW2*it1;
                it1 = it1 + i1*MM[3+im4] + i2*MM[2+im4] + i3*MM[1+im4] + i4*MM[im4];
                it1 = it1 % IPW2;
                // Convert 48-bit integer to a real number in the interval (0,1)
                x[i] = R*(real(it1)+R*(real(it2)+R*(real(it3)+R*real(it4))));
                if (x[i]==ONE)
                {
                    // If a real number has n bits of precision, and the first n bits of the 48-bit
                    // integer above happen to be all 1 (which will occur about once every 2^n calls),
                    // then X[i] will be rounded to exactly 1.0.
                    // Since X[i] is not supposed to return exactly 0.0 or 1.0, the statistically
                    // correct thing to do in this situation is simply to iterate again.
                    // N.B. the case X[i] = 0.0 should not be possible.
                    i1 += 2;
                    i2 += 2;
                    i3 += 2;
                    i4 += 2;
                }
            } while (x[i]==ONE);
        }
        // Return final value of seed
        iseed[0] = it1;
        iseed[1] = it2;
        iseed[2] = it3;
        iseed[3] = it4;
    }

    /* dlas2 computes the singular values of the 2-by-2 matrix
     *     [ f  g ]
     *     [ 0  h ].
     * On return, ssmin is the smaller singular value and ssmax is the larger singular value.
     * Parameters: f: The [0,0] element of the 2-by-2 matrix.
     *             g: The [0,1] element of the 2-by-2 matrix.
     *             h: The [1,1] element of the 2-by-2 matrix.
     *             ssmin: The smaller singular value.
     *             ssmax: The larger singular value.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Further Details:
     *     Barring over/underflow, all output quantities are correct to within a few units in the
     *     last place (ulps), even in the absence of a guard digit in addition/subtraction.
     *     In IEEE arithmetic, the code works correctly if one matrix element is infinite.
     *     Overflow will not occur unless the largest singular value itself overflows, or is within
     *     a few ulps of overflow. (On machines with partial overflow, like the Cray, overflow may
     *     occur if the largest singular value is within a factor of 2 of overflow.)
     *     Underflow is harmless if underflow is gradual. Otherwise, results may correspond to a
     *     matrix modified by perturbations of size near the underflow threshold.                */
    static void dlas2(real f, real g, real h, real& ssmin, real& ssmax)
    {
        real fa = fabs(f);
        real ga = fabs(g);
        real ha = fabs(h);
        real fhmn, fhmx;
        if (fa>ha)
        {
            fhmn = ha;
            fhmx = fa;
        }
        else
        {
            fhmn = fa;
            fhmx = ha;
        }
        if (fhmn==ZERO)
        {
            ssmin = ZERO;
            if (fhmx==ZERO)
            {
                ssmax = ga;
            }
            else
            {
                if (fhmx>ga)
                {
                    ssmax = ga/fhmx;
                    ssmax = fhmx*std::sqrt(ONE+ssmax*ssmax);
                }
                else
                {
                    ssmax = fhmx/ga;
                    ssmax = ga*std::sqrt(ONE+ssmax*ssmax);
                }
            }
        }
        else
        {
            real as, at, au, c;
            if (ga<fhmx)
            {
                as = ONE + fhmn/fhmx;
                at = (fhmx-fhmn) / fhmx;
                au = ga / fhmx;
                au = au*au;
                c = TWO / (std::sqrt(as*as+au)+std::sqrt(at*at+au));
                ssmin = fhmn * c;
                ssmax = fhmx / c;
            }
            else
            {
                au = fhmx / ga;
                if (au==ZERO)
                {
                    // Avoid possible harmful underflow if exponent range asymmetric
                    // (true SSMIN may not underflow even if AU underflows)
                    ssmin = (fhmn*fhmx) / ga;
                    ssmax = ga;
                }
                else
                {
                    as = ONE + fhmn/fhmx;
                    at = (fhmx-fhmn) / fhmx;
                    c = ONE / (std::sqrt(ONE+(as*au)*(as*au))+std::sqrt(ONE+(at*au)*(at*au)));
                    ssmin = (fhmn*c) * au;
                    ssmin = ssmin + ssmin;
                    ssmax = ga / (c+c);
                }
            }
        }
    }

    /* dlascl multiplies the m by n real matrix A by the real scalar cto/cfrom. This is done
     * without over/underflow as long as the final result cto*A[i,j]/cfrom does not over/underflow.
     * type specifies that A may be full, upper triangular, lower triangular, upper Hessenberg, or
     * banded.
     * Parameters: type: indices the storage type of the input matrix.
     *                   ='G': A is a full matrix.
     *                   ='L': A is a lower triangular matrix.
     *                   ='U': A is an upper triangular matrix.
     *                   ='H': A is an upper Hessenberg matrix.
     *                   ='B': A is a symmetric band matrix with lower bandwidth kl and upper
     *                         bandwidth ku and with the only the lower half stored.
     *                   ='Q': A is a symmetric band matrix with lower bandwidth kl and upper
     *                         bandwidth ku and with the only the upper half stored.
     *                   ='Z': A is a band matrix with lower bandwidth kl and upper bandwidth ku.
     *                         See DGBTRF for storage details.
     *             kl: The lower bandwidth of A. Referenced only if type=='B', 'Q' or 'Z'.
     *             ku: The upper bandwidth of A. Referenced only if type=='B', 'Q' or 'Z'.
     *             cfrom,
     *             cto: The matrix A is multiplied by cto/cfrom. A[i,j] is computed without
     *                  over/underflow if the final result cto*A[i,j]/cfrom can be represented
     *                  without over/underflow. cfrom must be nonzero.
     *             m: The number of rows of the matrix A. m>=0.
     *             n: The number of columns of the matrix A. n>=0.
     *             A: an array, dimension (lda,n)
     *                The matrix to be multiplied by cto/cfrom. See type for the storage type.
     *             lda: The leading dimension of the array A.
     *                  If type=='G', 'L', 'U', 'H': lda >= max(1,m);
     *                     type=='B'               : lda >= kl+1;
     *                     type=='Q'               : lda >= ku+1;
     *                     type=='Z'               : lda >= 2*kl+ku+1.
     *             info: ==0: successful exit
     *                    <0: if info = -i, the i-th argument had an illegal value.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: June 2016                                                                           */
    static void dlascl(char const* type, int kl, int ku, real cfrom, real cto, int m, int n,
                       real* A, int lda, int& info)
    {
        // Test the input arguments
        info = 0;
        int itype;
        char uptype = std::toupper(type[0]);
        if (uptype=='G')
        {
            itype = 0;
        }
        else if (uptype=='L')
        {
            itype = 1;
        }
        else if (uptype=='U')
        {
            itype = 2;
        }
        else if (uptype=='H')
        {
            itype = 3;
        }
        else if (uptype=='B')
        {
            itype = 4;
        }
        else if (uptype=='Q')
        {
            itype = 5;
        }
        else if (uptype=='Z')
        {
            itype = 6;
        }
        else
        {
            itype = -1;
        }
        if (itype==-1)
        {
            info = -1;
        }
        else if (cfrom==ZERO || std::isnan(cfrom))
        {
            info = -4;
        }
        else if (std::isnan(cto))
        {
            info = -5;
        }
        else if (m<0)
        {
            info = -6;
        }
        else if (n<0 || (itype==4 && n!=m) || (itype==5 && n!=m))
        {
            info = -7;
        }
        else if (itype<=3 && (lda<1 || lda<m))
        {
            info = -9;
        }
        else if (itype>=4)
        {
            if (kl<0 || (kl>(m-1) && kl>0))
            {
                info = -2;
            }
            else if (ku<0 || (ku>(n-1) && ku>0) || ((itype==4 || itype==5) && kl!=ku))
            {
                info = -3;
            }
            else if ((itype==4 && lda<(kl+1)) || (itype==5 && lda<(ku+1))
                     || (itype==6 && lda<(2*kl+ku+1)))
            {
                info = -9;
            }
        }
        if (info!=0)
        {
            xerbla("DLASCL", -info);
            return;
        }
        // Quick return if possible
        if (n==0 || m==0)
        {
            return;
        }
        // Get machine parameters
        real smlnum = dlamch("S");
        real bignum = ONE / smlnum;
        real cfromc = cfrom;
        real ctoc = cto;
        int i, j, ldaj, k1, k2, k3, k4;
        real cfrom1, cto1=0, mul;
        bool done;
        do
        {
            cfrom1 = cfromc*smlnum;
            if (cfrom1==cfromc)
            {
                // CFROMC is an inf. Multiply by a correctly signed zero for finite CTOC,
                // or a NaN if CTOC is infinite.
                mul = ctoc / cfromc;
                done = true;
                cto1 = ctoc;
            }
            else
            {
                cto1 /= bignum;
                if (cto1==ctoc)
                {
                    // CTOC is either 0 or an inf.
                    // In both cases, CTOC itself serves as the correct multiplication factor.
                    mul = ctoc;
                    done = true;
                    cfromc = ONE;
                }
                else if (fabs(cfrom1)>fabs(ctoc) && ctoc!=ZERO)
                {
                    mul = smlnum;
                    done = false;
                    cfromc = cfrom1;
                }
                else if (fabs(cto1)>fabs(cfromc))
                {
                    mul = bignum;
                    done = false;
                    ctoc = cto1;
                }
                else
                {
                    mul = ctoc / cfromc;
                    done = true;
                }
            }
            if (itype==0)
            {
                // Full matrix
                for (j=0; j<n; j++)
                {
                    ldaj = lda*j;
                    for (i=0; i<m; i++)
                    {
                        A[i+ldaj] *= mul;
                    }
                }
            }
            else if (itype==1)
            {
                // Lower triangular matrix
                for (j=0; j<n; j++)
                {
                    ldaj = lda*j;
                    for (i=j; i<m; i++)
                    {
                        A[i+ldaj] *= mul;
                    }
                }
            }
            else if (itype==2)
            {
                // Upper triangular matrix
                for (j=0; j<n; j++)
                {
                    ldaj = lda*j;
                    for (i=0; i<=j && i<m; i++)
                    {
                        A[i+ldaj] *= mul;
                    }
                }
            }
            else if (itype==3)
            {
                // Upper Hessenberg matrix
                for (j=0; j<n; j++)
                {
                    ldaj = lda*j;
                    for (i=0; i<=(j+1) && i<m; i++)
                    {
                        A[i+ldaj] *= mul;
                    }
                }
            }
            else if (itype==4)
            {
                // Lower half of a symmetric band matrix
                for (j=0; j<n; j++)
                {
                    ldaj = lda*j;
                    for (i=0; i<=kl && i<(n-j); i++)
                    {
                        A[i+ldaj] *= mul;
                    }
                }
            }
            else if (itype==5)
            {
                // Upper half of a symmetric band matrix
                for (j=0; j<n; j++)
                {
                    ldaj = lda*j;
                    k1 = ku-j;
                    for (i=((k1>0)?k1:0); i<=ku; i++)
                    {
                        A[i+ldaj] *= mul;
                    }
                }
            }
            else if (itype==6)
            {
                // Band matrix
                k1 = kl + ku;
                k3 = 2*kl + ku + 1;
                k4 = k1 + m;
                for (j=0; j<n; j++)
                {
                    ldaj = lda*j;
                    k2 = k1-j;
                    for (i=((k2>kl)?k2:kl); i<k3 && i<(k4-j); i++)
                    {
                        A[i+ldaj] *= mul;
                    }
                }
            }
        } while (!done);
    }

    /* Using a divide and conquer approach, dlasd0 computes the singular value decomposition (SVD)
     * of a real upper bidiagonal n-by-m matrix B with diagonal d and offdiagonal e, where
     * m = n+sqre. The algorithm computes orthogonal matrices U and Vt such that B = U * S * Vt.
     * The singular values S are overwritten on d.
     * A related subroutine, dlasda, computes only the singular values, and optionally, the
     * singular vectors in compact form.
     * Parameters: n: On entry, the row dimension of the upper bidiagonal matrix.
     *                This is also the dimension of the main diagonal array d.
     *             sqre: Specifies the column dimension of the bidiagonal matrix.
     *                   ==0: The bidiagonal matrix has column dimension m = n;
     *                   ==1: The bidiagonal matrix has column dimension m = n+1;
     *             d: an array, dimension (n)
     *                On entry d contains the main diagonal of the bidiagonal matrix.
     *                On exit d, if info==0, contains its singular values.
     *             e: an array, dimension (m-1)
     *                Contains the subdiagonal entries of the bidiagonal matrix.
     *                On exit, e has been destroyed.
     *             U: an array, dimension (ldu, n)
     *                On exit, U contains the left singular vectors.
     *             ldu: On entry, leading dimension of U.
     *             Vt: an array, dimension (ldvt, m)
     *                 On exit, Vt^T contains the right singular vectors.
     *             ldvt: On entry, leading dimension of Vt.
     *             smlsiz: On entry, maximum size of the subproblems at the bottom of the
     *                     computation tree.
     *             iwork: an integer array, dimension (8*n)
     *             work: an array, dimension (3*m**2+2*m)
     *             info: ==0: successful exit.
     *                   < 0: if info==-i, the i-th argument had an illegal value.
     *                   > 0: if info==1, a singular value did not converge
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: June 2017
     * Contributors:
     *     Ming Gu and Huan Ren, Computer Science Division,
     *     University of California at Berkeley, USA                                             */
    static void dlasd0(int n, int sqre, real* d, real* e, real* U, int ldu, real* Vt, int ldvt,
                       int smlsiz, int* iwork, real* work, int& info)
    {
        // Test the input parameters.
        info = 0;
        if (n<0)
        {
            info = -1;
        }
        else if (sqre<0 || sqre>1)
        {
            info = -2;
        }
        int m = n + sqre;
        if (ldu<n)
        {
            info = -6;
        }
        else if (ldvt<m)
        {
            info = -8;
        }
        else if (smlsiz<3)
        {
            info = -9;
        }
        if (info!=0)
        {
            xerbla("DLASD0", -info);
            return;
        }
        // If the input matrix is too small, call dlasdq to find the SVD.
        if (n<=smlsiz)
        {
            dlasdq("U", sqre, n, m, n, 0, d, e, Vt, ldvt, U, ldu, U, ldu, work, info);
            return;
        }
        // Set up the computation tree.
        int inode = 0;
        int ndiml = inode + n;
        int ndimr = ndiml + n;
        int idxq  = ndimr + n;
        int iwk   = idxq  + n;
        int nlvl;
        int nd;
        dlasdt(n, nlvl, nd, &iwork[inode], &iwork[ndiml], &iwork[ndimr], smlsiz);
        // For the nodes on bottom level of the tree, solve their subproblems by dlasdq.
        int ndb1 = (nd+1) / 2;
        int ncc = 0;
        int i, ic, itemp, j, nl, nlf, nr, nrf, nrp1, sqrei;
        for (i=ndb1-1; i<nd; i++)
        {
            // ic : center row of each node
            // nl : number of rows of left  subproblem
            // nr : number of rows of right subproblem
            // nlf: starting row of the left  subproblem
            // nrf: starting row of the right subproblem
            ic = iwork[inode+i];
            nl = iwork[ndiml+i];
            nr = iwork[ndimr+i];
            nlf = ic - nl;
            nrf = ic + 1;
            sqrei = 1;
            dlasdq("U", sqrei, nl, nl+1, nl, ncc, &d[nlf], &e[nlf], &Vt[nlf+ldvt*nlf], ldvt,
                   &U[nlf+ldu*nlf], ldu, &U[nlf+ldu*nlf], ldu, work, info);
            if (info!=0)
            {
                return;
            }
            itemp = idxq + nlf;
            for (j=0; j<nl; j++)
            {
                iwork[itemp+j] = j;
            }
            if (i==nd-1)
            {
                sqrei = sqre;
            }
            else
            {
                sqrei = 1;
            }
            nrp1 = nr + sqrei;
            dlasdq("U", sqrei, nr, nrp1, nr, ncc, &d[nrf], &e[nrf], &Vt[nrf+ldvt*nrf], ldvt,
                   &U[nrf+ldu*nrf], ldu, &U[nrf+ldu*nrf], ldu, work, info);
            if (info!=0)
            {
                return;
            }
            itemp = idxq + ic + 1;
            for (j=0; j<nr; j++)
            {
                iwork[itemp+j] = j;
            }
        }
        // Now conquer each subproblem bottom-up.
        int idxqc, lf, ll, lvl;
        real alpha, beta;
        for (lvl=nlvl-1; lvl>=0; lvl--)
        {
            // Find the first node lf and last node ll on the current level lvl.
            if (lvl==0)
            {
                lf = 0;
                ll = 0;
            }
            else
            {
                lf = (1 << lvl) - 1;
                ll = 2*lf;
            }
            for (i=lf; i<=ll; i++)
            {
                ic = iwork[inode+i];
                nl = iwork[ndiml+i];
                nr = iwork[ndimr+i];
                nlf = ic - nl;
                if (sqre==0 && i==ll)
                {
                    sqrei = sqre;
                }
                else
                {
                    sqrei = 1;
                }
                idxqc = idxq + nlf;
                alpha = d[ic];
                beta = e[ic];
                dlasd1(nl, nr, sqrei, &d[nlf], alpha, beta, &U[nlf+ldu*nlf], ldu,
                       &Vt[nlf+ldvt*nlf], ldvt, &iwork[idxqc], &iwork[iwk], work, info);
                // Report the possible convergence failure.
                if (info!=0)
                {
                    return;
                }
            }
        }
    }

    /* dlasd1 computes the SVD of an upper bidiagonal n-by-m matrix B, where n = nl+nr+1 and
     * m = n+sqre. dlasd1 is called from dlasd0.
     * A related subroutine dlasd7 handles the case in which the singular values (and the singular
     * vectors in factored form) are desired.
     * dlasd1 computes the SVD as follows:
     *                 ( d1(in)   0    0      0 )
     *     B = U(in) * (   Z1^T   a   Z2^T    b ) * Vt(in)
     *                 (   0      0   D2(in)  0 )
     *       = U(out) * ( d(out) 0) * Vt(out)
     * where Z^T = (Z1^T a Z2^T b) = u^T Vt^T, and u is a vector of dimension m with alpha and beta
     * in the nl+1 and nl+2 th entries and zeros elsewhere; and the entry b is empty if sqre==0.
     * The left singular vectors of the original matrix are stored in U, and the transpose of the
     * right singular vectors are stored in Vt, and the singular values are in d. The algorithm
     * consists of three stages:
     *   The first stage consists of deflating the size of the problem when there are multiple
     *     singular values or when there are zeros in the Z vector. For each such occurrence the
     *     dimension of the secular equation problem is reduced by one. This stage is performed by
     *     the routine dlasd2.
     *   The second stage consists of calculating the updated singular values. This is done by
     *     finding the square roots of the roots of the secular equation via the routine dlasd4
     *     (as called by DLASD3). This routine also calculates the singular vectors of the current
     *     problem.
     *   The final stage consists of computing the updated singular vectors directly using the
     *     updated singular values. The singular vectors for the current problem are multiplied
     *     with the singular vectors from the overall problem.
     * Parameters: nl: The row dimension of the upper block. nl>=1.
     *             nr: The row dimension of the lower block. nr>=1.
     *             sqre: ==0: the lower block is an nr-by-nr square matrix.
     *                   ==1: the lower block is an nr-by-(nr+1) rectangular matrix.
     *                   The bidiagonal matrix has row dimension n = nl+nr+1, and column dimension
     *                   m = n+sqre.
     *             d: an array, dimension (n = nl+nr+1).
     *                On entry d[0:nl-1,0:nl-1] contains the singular values of the upper block;
     *                and d[nl+1:n-1] contains the singular values of the lower block.
     *                On exit d[0:n-1] contains the singular values of the modified matrix.
     *             alpha: Contains the diagonal element associated with the added row.
     *             beta: Contains the off-diagonal element associated with the added row.
     *             U: an array, dimension(ldu,n)
     *                On entry U[0:nl-1,0:nl-1] contains the left singular vectors of the upper
     *                block; U[nl+1:n-1,nl+1:n-1] contains the left singular vectors of the lower
     *                block.
     *                On exit U contains the left singular vectors of the bidiagonal matrix.
     *             ldu: The leading dimension of the array U. ldu>=max(1, n).
     *             Vt: an array, dimension(ldvt,m) where m = n+sqre.
     *                 On entry Vt[0:nl, 0:nl]^T contains the right singular vectors of the upper
     *                 block; Vt[nl+1:m-1,nl+1:m-1]^T contains the right singular vectors of the
     *                 lower block.
     *                 On exit Vt^T contains the right singular vectors of the bidiagonal matrix.
     *             ldvt: The leading dimension of the array Vt. ldvt>=max(1, m).
     *             idxq: an integer array, dimension(n)
     *                   This contains the permutation which will reintegrate the subproblem just
     *                   solved back into sorted order, i.e. d[idxq[0:n-1]] will be in ascending
     *                   order.
     *                   NOTE: zero-based indexing!
     *             iwork: an integer array, dimension(4*n)
     *             work: an array, dimension(3*m^2 + 2*m)
     *             info: ==0: successful exit.
     *                   < 0: if info==-i, the i-th argument had an illegal value.
     *                   > 0: if info==1, a singular value did not converge
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: June 2016
     * Contributors:
     *     Ming Gu and Huan Ren, Computer Science Division,
     *     University of California at Berkeley, USA                                             */
    static void dlasd1(int nl, int nr, int sqre, real* d, real& alpha, real& beta, real* U,
                       int ldu, real* Vt, int ldvt, int* idxq, int* iwork, real* work, int& info)
    {
        // Test the input parameters.
        info = 0;
        if (nl<1)
        {
            info = -1;
        }
        else if (nr<1)
        {
            info = -2;
        }
        else if (sqre<0 || sqre>1)
        {
            info = -3;
        }
        if (info!=0)
        {
            xerbla("DLASD1", -info);
            return;
        }
        int n = nl + nr + 1;
        int m = n + sqre;
        // The following values are for bookkeeping purposes only. They are integer pointers which
        // indicate the portion of the workspace used by a particular array in dlasd2 and dlasd3.
        int ldu2 = n;
        int ldvt2 = m;
        int iz = 0;
        int isigma = iz + m;
        int iu2 = isigma + n;
        int ivt2 = iu2 + ldu2*n;
        int iq = ivt2 + ldvt2*m;
        int idx = 0;
        int idxc = idx + n;
        int coltyp = idxc + n;
        int idxp = coltyp + n;
        // Scale.
        real orgnrm = std::max(std::fabs(alpha), std::fabs(beta));
        d[nl] = ZERO;
        for (int i=0; i<n; i++)
        {
            if (std::fabs(d[i])>orgnrm)
            {
                orgnrm = std::fabs(d[i]);
            }
        }
        dlascl("G", 0, 0, orgnrm, ONE, n, 1, d, n, info);
        alpha /= orgnrm;
        beta /= orgnrm;
        // Deflate singular values.
        int k;
        dlasd2(nl, nr, sqre, k, d, &work[iz], alpha, beta, U, ldu, Vt, ldvt, &work[isigma],
               &work[iu2], ldu2, &work[ivt2], ldvt2, &iwork[idxp], &iwork[idx], &iwork[idxc], idxq,
               &iwork[coltyp], info);
        // Solve Secular Equation and update singular vectors.
        int ldq = k;
        dlasd3(nl, nr, sqre, k, d, &work[iq], ldq, &work[isigma], U, ldu, &work[iu2], ldu2, Vt,
               ldvt, &work[ivt2], ldvt2, &iwork[idxc], &iwork[coltyp], &work[iz], info);
        // Report the convergence failure.
        if (info!=0)
        {
            return;
        }
        // Unscale.
        dlascl("G", 0, 0, ONE, orgnrm, n, 1, d, n, info);
        // Prepare the idxq sorting permutation.
        dlamrg(k, n-k, d, 1, -1, idxq);
    }

    /* dlasd2 merges the two sets of singular values together into a single sorted set. Then it
     * tries to deflate the size of the problem. There are two ways in which deflation can occur:
     * when two or more singular values are close together or if there is a tiny entry in the z
     * vector. For each such occurrence the order of the related secular equation problem is
     * reduced by one.
     * dlasd2 is called from dlasd1.
     * Parameters: nl: The row dimension of the upper block. nl>=1.
     *             nr: The row dimension of the lower block. nr>=1.
     *             sqre: ==0: the lower block is an nr-by-nr square matrix.
     *                   ==1: the lower block is an nr-by-(nr+1) rectangular matrix.
     *                   The bidiagonal matrix has n = nl+nr+1 rows and m = n+sqre >= n columns.
     *             k: Contains the dimension of the non-deflated matrix, This is the order of the
     *                related secular equation. 1<=k<=n.
     *             d: an array, dimension(n)
     *                On entry d contains the singular values of the two submatrices to be
     *                combined.
     *                On exit d contains the trailing (n-k) updated singular values (those which
     *                were deflated) sorted into increasing order.
     *             z: an array, dimension(n)
     *                On exit z contains the updating row vector in the secular equation.
     *             alpha: Contains the diagonal element associated with the added row.
     *             beta: Contains the off-diagonal element associated with the added row.
     *             U: an array, dimension(ldu,n)
     *                On entry U contains the left singular vectors of two submatrices in the two
     *                square blocks with corners at (1,1), (nl, nl), and (nl+2, nl+2), (n,n).
     *                On exit U contains the trailing (n-k) updated left singular vectors (those
     *                which were deflated) in its last n-k columns.
     *             ldu: The leading dimension of the array U. ldu>=n.
     *             Vt: an array, dimension(ldvt,m)
     *                 On entry Vt^T contains the right singular vectors of two submatrices in the
     *                 two square blocks with corners at (1,1), (nl+1, nl+1), and (nl+2, nl+2),
     *                 (m,m).
     *                 On exit Vt^T contains the trailing (n-k) updated right singular vectors
     *                 (those which were deflated) in its last n-k columns.
     *                 In case sqre==1, the last row of Vt spans the right null space.
     *             ldvt: The leading dimension of the array Vt. ldvt>=m.
     *             dsigma: an array, dimension (n)
     *                     Contains a copy of the diagonal elements (k-1 singular values and one
     *                     zero) in the secular equation.
     *             U2: an array, dimension(ldu2,n)
     *                 Contains a copy of the first k-1 left singular vectors which will be used by
     *                 dlasd3 in a matrix multiply (dgemm) to solve for the new left singular
     *                 vectors. U2 is arranged into four blocks. The first block contains a column
     *                 with 1 at nl+1 and zero everywhere else; the second block contains non-zero
     *                 entries only at and above nl; the third contains non-zero entries only below
     *                 nl+1; and the fourth is dense.
     *             ldu2: The leading dimension of the array U2. ldu2>=n.
     *             Vt2: an array, dimension(ldvt2,n)
     *                  Vt2^T contains a copy of the first k right singular vectors which will be
     *                  used by dlasd3 in a matrix multiply (dgemm) to solve for the new right
     *                  singular vectors. Vt2 is arranged into three blocks. The first block
     *                  contains a row that corresponds to the special 0 diagonal element in SIGMA;
     *                  the second block contains non-zeros only at and before nl +1; the third
     *                  block contains non-zeros only at and after nl +2.
     *             ldvt2: The leading dimension of the array Vt2. ldvt2>=m.
     *             idxp: an integer array, dimension(n)
     *                   This will contain the permutation used to place deflated values of d at
     *                   the end of the array. On output idxp[1:k-1] points to the nondeflated
     *                   d-values and idxp[k:n-1] points to the deflated singular values.
     *                   NOTE: zero-based indices!
     *             idx: an integer array, dimension(n)
     *                  This will contain the permutation used to sort the contents of d into
     *                  ascending order.
     *                  NOTE: zero-based indices!
     *             idxc: an integer array, dimension(n)
     *                   This will contain the permutation used to arrange the columns of the
     *                   deflated U matrix into three groups: the first group contains non-zero
     *                   entries only at and above nl, the second contains non-zero entries only
     *                   below nl+2, and the third is dense.
     *                   NOTE: zero-based indices!
     *             idxq: an integer array, dimension(n)
     *                   This contains the permutation which separately sorts the two sub-problems
     *                   in d into ascending order. Note that entries in the first half of this
     *                   permutation must first be moved one position backward; and entries in the
     *                   second half must first have nl+1 added to their values.
     *                   NOTE: zero-based indices!
     *             coltyp: an integer array, dimension(n)
     *                     As workspace, this will contain a label which will indicate which of the
     *                     following types a column in the U2 matrix or a row in the Vt2 matrix is:
     *                         0: non-zero in the upper half only
     *                         1: non-zero in the lower half only
     *                         2: dense
     *                         3: deflated
     *                     On exit, it is an array of dimension 4, with coltyp[I] being the
     *                     dimension of the I-th type columns.
     *                     NOTE: zero-based indices!
     *             info: ==0: successful exit.
     *                   < 0: if info==-i, the i-th argument had an illegal value.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Contributors:
     *     Ming Gu and Huan Ren, Computer Science Division,
     *     University of California at Berkeley, USA                                             */
    static void dlasd2(int nl, int nr, int sqre, int& k, real* d, real* z, real alpha, real beta,
                       real* U, int ldu, real* Vt, int ldvt, real* dsigma, real* U2, int ldu2,
                       real* Vt2, int ldvt2, int* idxp, int* idx, int* idxc, int* idxq,
                       int* coltyp, int& info)
    {
        // Test the input parameters.
        info = 0;
        if (nl<1)
        {
            info = -1;
        }
        else if (nr<1)
        {
            info = -2;
        }
        else if (sqre!=1 && sqre!=0)
        {
            info = -3;
        }
        int n = nl + nr + 1;
        int m = n + sqre;
        if (ldu<n)
        {
            info = -10;
        }
        else if (ldvt<m)
        {
            info = -12;
        }
        else if (ldu2<n)
        {
            info = -15;
        }
        else if (ldvt2<m)
        {
            info = -17;
        }
        if (info!=0)
        {
            xerbla("DLASD2", -info);
            return;
        }
        // Generate the first part of the vector z; and move the singular values in the first part
        // of d one position backward.
        real z1 = alpha*Vt[nl+ldvt*nl];
        z[0] = z1;
        int i, tind=ldvt*nl;
        int nlp1 = nl + 1;
        for (i=nl-1; i>=0; i--)
        {
            z[i+1] = alpha*Vt[i+tind];
            d[i+1] = d[i];
            idxq[i+1] = idxq[i] + 1;
        }
        // Generate the second part of the vector z.
        tind = ldvt*nlp1;
        for (i=nlp1; i<m; i++)
        {
            z[i] = beta*Vt[i+tind];
        }
        // Initialize some reference arrays.
        for (i=1; i<nlp1; i++)
        {
            coltyp[i] = 0;
        }
        for (i=nlp1; i<n; i++)
        {
            coltyp[i] = 1;
        }
        // Sort the singular values into increasing order
        for (i=nlp1; i<n; i++)
        {
            idxq[i] += nlp1;
        }
        // dsigma, idxc, idxc, and the first column of U2 are used as storage space.
        for (i=1; i<n; i++)
        {
            dsigma[i] = d[idxq[i]];
            U2[i] = z[idxq[i]];
            idxc[i] = coltyp[idxq[i]];
        }
        dlamrg(nl, nr, &dsigma[1], 1, 1, &idx[1]);
        int idxi;
        for (i=1; i<n; i++)
        {
            idxi = 1 + idx[i];
            d[i] = dsigma[idxi];
            z[i] = U2[idxi];
            coltyp[i] = idxc[idxi];
        }
        // Calculate the allowable deflation tolerance
        real eps = dlamch("Epsilon");
        real tol = std::max(std::fabs(alpha), std::fabs(beta));
        tol = EIGHT*eps*std::max(std::fabs(d[n-1]), tol);
        /* There are 2 kinds of deflation -- first a value in the z-vector is small, second two
         * (or more) singular values are very close together (their difference is small).
         * If the value in the z-vector is small, we simply permute the array so that the
         * corresponding singular value is moved to the end.
         * If two values in the d-vector are close, we perform a two-sided rotation designed to
         * make one of the corresponding z-vector entries zero, and then permute the array so that
         * the deflated singular value is moved to the end.
         * If there are multiple singular values then the problem deflates. Here the number of
         * equal singular values are found. As each equal singular value is found, an elementary
         * reflector is computed to rotate the corresponding singular subspace so that the
         * corresponding components of z are zero in this new basis.                             */
        k = 1;
        int k2 = n;
        bool skip_deflation = false;
        int j, jprev;
        for (j=1; j<n; j++)
        {
            if (std::fabs(z[j])<=tol)
            {
                // Deflate due to small z component.
                k2--;
                idxp[k2] = j;
                coltyp[j] = 3;
                if (j==n-1)
                {
                    skip_deflation = true;
                    break;
                }
            }
            else
            {
                jprev = j;
                break;
            }
        }
        int idxj;
        real c, s;
        if (!skip_deflation)
        {
            int idxjp;
            real tau;
            j = jprev;
            while (true)
            {
                j++;
                if (j>=n)
                {
                    break;
                }
                if (std::fabs(z[j])<=tol)
                {
                    // Deflate due to small z component.
                    k2--;
                    idxp[k2] = j;
                    coltyp[j] = 3;
                }
                else
                {
                    // Check if singular values are close enough to allow deflation.
                    if (std::fabs(d[j]-d[jprev])<=tol)
                    {
                        // Deflation is possible.
                        s = z[jprev];
                        c = z[j];
                        // Find sqrt(a^2+b^2) without overflow or destructive underflow.
                        tau = dlapy2(c, s);
                        c /= tau;
                        s = -s / tau;
                        z[j] = tau;
                        z[jprev] = ZERO;
                        // Apply back the Givens rotation to the left and right singular vector
                        // matrices.
                        idxjp = idxq[idx[jprev]+1];
                        idxj = idxq[idx[j]+1];
                        if (idxjp<nlp1)
                        {
                            idxjp--;
                        }
                        if (idxj<nlp1)
                        {
                            idxj--;
                        }
                        Blas<real>::drot(n, &U[ldu*idxjp], 1, &U[ldu*idxj], 1, c, s);
                        Blas<real>::drot(m, &Vt[idxjp], ldvt, &Vt[idxj], ldvt, c, s);
                        if (coltyp[j]!=coltyp[jprev])
                        {
                            coltyp[j] = 2;
                        }
                        coltyp[jprev] = 3;
                        k2--;
                        idxp[k2] = jprev;
                        jprev = j;
                    }
                    else
                    {
                        k++;
                        U2[k-1] = z[jprev];
                        dsigma[k-1] = d[jprev];
                        idxp[k-1] = jprev;
                        jprev = j;
                    }
                }
            }
            // Record the last singular value.
            k++;
            U2[k-1] = z[jprev];
            dsigma[k-1] = d[jprev];
            idxp[k-1] = jprev;
        }
        // Count up the total number of the various types of columns, then form a permutation which
        // positions the four column types into four groups of uniform structure (although one or
        // more of these groups may be empty).
        int ctot[4], psm[4];
        int ct;
        for (j=0; j<4; j++)
        {
            ctot[j] = 0;
        }
        for (j=1; j<n; j++)
        {
            ct = coltyp[j];
            ctot[ct]++;
        }
        // psm[*] = Position in SubMatrix (of types 0 through 3) (zero-based!)
        psm[0] = 1;
        psm[1] = 1 + ctot[0];
        psm[2] = psm[1] + ctot[1];
        psm[3] = psm[2] + ctot[2];
        // Fill out the idxc array so that the permutation which it induces will place all type-1
        // columns first, all type-2 columns next, then all type-3's, and finally all type-4's,
        // starting from the second column. This applies similarly to the rows of Vt.
        int jp;
        for (j=1; j<n; j++)
        {
            jp = idxp[j];
            ct = coltyp[jp];
            idxc[psm[ct]] = j;
            psm[ct]++;
        }
        // Sort the singular values and corresponding singular vectors into dsigma, U2, and Vt2
        // respectively. The singular values/vectors which were not deflated go into the first k
        // slots of dsigma, U2, and Vt2 respectively, while those which were deflated go into the
        // last n-k slots, except that the first column/row will be treated separately.
        for (j=1; j<n; j++)
        {
            jp = idxp[j];
            dsigma[j] = d[jp];
            idxj = idxq[idx[idxp[idxc[j]]]+1];
            if (idxj<nlp1)
            {
               idxj--;
            }
            Blas<real>::dcopy(n, &U[ldu*idxj], 1, &U2[ldu2*j], 1);
            Blas<real>::dcopy(m, &Vt[idxj], ldvt, &Vt2[j], ldvt2);
        }
        // Determine dsigma[0], dsigma[1] and z[0]
        dsigma[0] = ZERO;
        real hlftol = tol / TWO;
        if (std::fabs(dsigma[1])<=hlftol)
        {
            dsigma[1] = hlftol;
        }
        if (m>n)
        {
            z[0] = dlapy2(z1, z[m-1]);
            if (z[0]<=tol)
            {
                c = ONE;
                s = ZERO;
                z[0] = tol;
            }
            else
            {
                c = z1 / z[0];
                s = z[m-1] / z[0];
            }
        }
        else
        {
            if (std::fabs(z1)<=tol)
            {
                z[0] = tol;
            }
            else
            {
                z[0] = z1;
            }
        }
        // Move the rest of the updating row to z.
        Blas<real>::dcopy(k-1, &U2[1], 1, &z[1], 1);
        // Determine the first column of U2, the first row of Vt2 and the last row of Vt.
        dlaset("A", n, 1, ZERO, ZERO, U2, ldu2);
        U2[nl] = ONE;
        if (m>n)
        {
            for (i=0; i<nlp1; i++)
            {
                tind = ldvt*i;
                Vt[m-1+tind] = -s*Vt[nl+tind];
                Vt2[ldvt2*i] = c*Vt[nl+tind];
            }
            for (i=nlp1; i<m; i++)
            {
                tind = m-1+ldvt*i;
                Vt2[ldvt2*i] = s*Vt[tind];
                Vt[tind] *= c;
            }
        }
        else
        {
            Blas<real>::dcopy(m, &Vt[nl], ldvt, Vt2, ldvt2);
        }
        if (m>n)
        {
            Blas<real>::dcopy(m, &Vt[m-1], ldvt, &Vt2[m-1], ldvt2);
        }
        // The deflated singular values and their corresponding vectors go into the back of d, U,
        // and V respectively.
        if (n>k)
        {
            Blas<real>::dcopy(n-k, dsigma[k], 1, d[k], 1);
            dlacpy("A", n, n-k, &U2[ldu2*k], ldu2, &U[ldu*k], ldu);
            dlacpy("A", n-k, m, &Vt2[k], ldvt2, &Vt[k], ldvt);
        }
        // Copy ctot into coltyp for referencing in dlasd3.
        for (j=0; j<4; j++)
        {
            coltyp[j] = ctot[j];
        }
    }

    /* dlasd3 finds all the square roots of the roots of the secular equation, as defined by the
     * values in d and z. It makes the appropriate calls to dlasd4 and then updates the singular
     * vectors by matrix multiplication.
     * This code makes very mild assumptions about floating point arithmetic. It will work on
     * machines with a guard digit in add/subtract, or on those binary machines without guard
     * digits which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2. It could
     * conceivably fail on hexadecimal or decimal machines without guard digits, but we know of
     * none.
     * dlasd3 is called from dlasd1.
     * Parameters: nl: The row dimension of the upper block. nl>=1.
     *             nr: The row dimension of the lower block. nr>=1.
     *             sqre: ==0: the lower block is an nr-by-nr square matrix.
     *                   ==1: the lower block is an nr-by-(nr+1) rectangular matrix.
     *                   The bidiagonal matrix has n = nl+nr+1 rows and m = n+sqre >= n columns.
     *             k: The size of the secular equation, 1 =< k = < n.
     *             d: an array, dimension(k)
     *                On exit the square roots of the roots of the secular equation, in ascending
     *                order.
     *             Q: an array, dimension (ldq,k)
     *             ldq: The leading dimension of the array Q.  ldq >= k.
     *             dsigma: an array, dimension(k)
     *                     The first k elements of this array contain the old roots of the deflated
     *                     updating problem. These are the poles of the secular equation.
     *             U: an array, dimension (ldu, n)
     *                The last n - k columns of this matrix contain the deflated left singular
     *                vectors.
     *             ldu: The leading dimension of the array U. ldu >= n.
     *             U2: an array, dimension (ldu2, n)
     *                 The first k columns of this matrix contain the non-deflated left singular
     *                 vectors for the split problem.
     *             ldu2: The leading dimension of the array U2. ldu2 >= n.
     *             Vt: an array, dimension (ldvt, m)
     *                 The last m - k columns of Vt^T contain the deflated right singular vectors.
     *             ldvt: The leading dimension of the array Vt. ldvt >= n.
     *             Vt2: an array, dimension (ldvt2, n)
     *                  The first k columns of Vt2^T contain the non-deflated right singular
     *                  vectors for the split problem.
     *             ldvt2: The leading dimension of the array Vt2. ldvt2 >= n.
     *             idxc: an integer array, dimension (n)
     *                   The permutation used to arrange the columns of U (and rows of Vt) into
     *                   three groups: the first group contains non-negative entries only at and
     *                   above (or before) nl; the second contains non-negative entries only at and
     *                   below (or after) nl+1; and the third is dense. The first column of U and
     *                   the row of Vt are treated separately, however.
     *                   The rows of the singular vectors found by dlasd4 must be likewise permuted
     *                   before the matrix multiplies can take place.
     *                   NOTE: zero-based indices!
     *             ctot: an integer array, dimension (4)
     *                   A count of the total number of the various types of columns in U (or rows
     *                   in Vt), as described in idxc. The fourth column type is any column which
     *                   has been deflated.
     *             z: an array, dimension (k)
     *                The first k elements of this array contain the components of the
     *                deflation-adjusted updating row vector.
     *             info: ==0:  successful exit.
     *                   < 0:  if info = -i, the i-th argument had an illegal value.
     *                   > 0:  if info = 1, a singular value did not converge
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: June 2017
     * Contributors:
     *     Ming Gu and Huan Ren, Computer Science Division,
     *     University of California at Berkeley, USA                                             */
    static void dlasd3(int nl, int nr, int sqre, int k, real* d, real* Q, int ldq, real* dsigma,
                       real* U, int ldu, real const* U2, int ldu2, real* Vt, int ldvt, real* Vt2,
                       int ldvt2, int const* idxc, int const* ctot, real* z, int& info)
    {
        // Test the input parameters.
        info = 0;
        if (nl<1)
        {
            info = -1;
        }
        else if (nr<1)
        {
            info = -2;
        }
        else if (sqre!=1 && sqre!=0)
        {
            info = -3;
        }
        int n = nl + nr + 1;
        int m = n + sqre;
        int nlp1 = nl + 1;
        if (k<1 || k>n)
        {
            info = -4;
        }
        else if (ldq<k)
        {
            info = -7;
        }
        else if (ldu<n)
        {
            info = -10;
        }
        else if (ldu2<n)
        {
            info = -12;
        }
        else if (ldvt<m)
        {
            info = -14;
        }
        else if (ldvt2<m)
        {
            info = -16;
        }
        if (info!=0)
        {
            xerbla("DLASD3", -info);
            return;
        }
        // Quick return if possible
        int i;
        if (k==1)
        {
            d[0] = std::fabs(z[0]);
            Blas<real>::dcopy(m, Vt2, ldvt2, Vt, ldvt);
            if (z[0]>ZERO)
            {
                Blas<real>::dcopy(n, U2, 1, U, 1);
            }
            else
            {
                for (i=0; i<n; i++)
                {
                    &U[i] = -U2[i];
                }
            }
            return;
        }
        /* Modify values dsigma[i] to make sure all dsigma[i]-dsigma[j] can be computed with high
         * relative accuracy (barring over/underflow). This is a problem on machines without a
         * guard digit in add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2). The following
         * code replaces dsigma[i] by 2*dsigma[i]-dsigma[i], which on any of these machines zeros
         * out the bottommost bit of dsigma[i] if it is 1; this makes the subsequent subtractions
         * dsigma[i]-dsigma[j] unproblematic when cancellation occurs. On binary machines with a
         * guard digit (almost all machines) it does not change dsigma[i] at all. On hexadecimal
         * and decimal machines with a guard digit, it slightly changes the bottommost bits of
         * dsigma[i]. It does not account for hexadecimal or decimal machines without guard digits
         * (we know of none). We use a subroutine call to compute 2*dsigma[i] to prevent optimizing
         * compilers from eliminating this code.                                                 */
        for (i=0; i<k; i++)
        {
            dsigma[i] = dlamc3(dsigma[i], dsigma[i]) - dsigma[i];
        }
        // Keep a copy of z.
        Blas<real>::dcopy(k, z, 1, Q, 1);
        // Normalize z.
        real rho = Blas<real>::dnrm2(k, z, 1);
        dlascl("G", 0, 0, rho, ONE, k, 1, z, k, info);
        rho *= rho;
        // Find the new singular values.
        int j;
        for (j=0; j<k; j++)
        {
            dlasd4(k, j, dsigma, z, &U[ldu*j], rho, &d[j], &Vt[ldvt*j], info);
            // If the zero finder fails, report the convergence failure.
            if (info!=0)
            {
                return;
            }
        }
        // Compute updated z.
        for (i=0; i<k; i++)
        {
            z[i] = U[i+ldu*(k-1)]*Vt[i+ldvt*(k-1)];
            for (j=0; j<i; j++)
            {
                z[i] *= (U[i+ldu*j]*Vt[i+ldvt*j] / (dsigma[i]-dsigma[j]) / (dsigma[i]+dsigma[j]));
            }
            for (j=i; j<k-1; j++)
            {
                z[i] *= (U[i+ldu*j]*Vt[i+ldvt*j] / (dsigma[i]-dsigma[j+1]) / (dsigma[i]+dsigma[j+1]));
            }
            z[i] = std::sqrt(std::fabs(z[i])) * ((ZERO<Q[i])-(Q[i]<ZERO));
        }
        // Compute left singular vectors of the modified diagonal matrix, and store related
        // information for the right singular vectors.
        int jc, ldui, ldvti;
        real temp;
        for (i=0; i<k; i++)
        {
            ldui = ldu*i;
            ldvti = ldvt*i;
            Vt[ldvti] = z[0] / U[ldui] / Vt[ldvti];
            U[ldui] = NEGONE;
            for (j=1; j<k; j++)
            {
                Vt[j+ldvti] = z[j] / U[j+ldui] / Vt[j+ldvti];
                U[j+ldui] = dsigma[j]*Vt[j+ldvti];
            }
            temp = Blas<real>::dnrm2(k, &U[ldui], 1);
            Q[ldq*i] = U[ldui] / temp;
            for (j=1; j<k; j++)
            {
                jc = idxc[j];
                Q[j+ldq*i] = U[jc+ldui] / temp;
            }
        }
        // Update the left singular vector matrix.
        int ctemp, ktempm1;
        if (k==2)
        {
            Blas<real>::dgemm("N", "N", n, k, k, ONE, U2, ldu2, Q, ldq, ZERO, U, ldu);
        }
        else
        {
            if (ctot[0]>0)
            {
                Blas<real>::dgemm("N", "N", nl, k, ctot[0], ONE, &U2[ldu2], ldu2, &Q[1], ldq, ZERO,
                                  U, ldu);
                if (ctot[2]>0)
                {
                    ktempm1 = 1 + ctot[0] + ctot[1];
                    Blas<real>::dgemm("N", "N", nl, k, ctot[2], ONE, &U2[ldu2*ktempm1], ldu2,
                                      &Q[ktempm1], ldq, ONE, U, ldu);
                }
            }
            else if (ctot[2]>0)
            {
                ktempm1 = 1 + ctot[0] + ctot[1];
                Blas<real>::dgemm("N", "N", nl, k, ctot[2], ONE, &U2[ldu2*ktempm1], ldu2, &Q[ktempm1],
                                  ldq, ZERO, U, ldu);
            }
            else
            {
                dlacpy("F", nl, k, U2, ldu2, U, ldu);
            }
            Blas<real>::dcopy(k, Q, ldq, &U[nl], ldu);
            ktempm1 = 1 + ctot[0];
            ctemp = ctot[1] + ctot[2];
            Blas<real>::dgemm("N", "N", nr, k, ctemp, ONE, &U2[nlp1+ldu2*ktempm1], ldu2, &Q[ktempm1], ldq,
                              ZERO, &U[nlp1], ldu);
        }
        // Generate the right singular vectors.
        for (i=0; i<k; i++)
        {
            ldvti = ldvt*i;
            temp = dnrm2(k, &Vt[ldvti], 1);
            Q[i] = Vt[ldvti] / temp;
            for (j=1; j<k; j++)
            {
                jc = idxc[j];
                Q[i+ldq*j] = Vt[jc+ldvti] / temp;
            }
        }
        // Update the right singular vector matrix.
        if (k==2)
        {
            Blas<real>::dgemm("N", "N", k, m, k, ONE, Q, ldq, Vt2, ldvt2, ZERO, Vt, ldvt);
            return;
        }
        ktempm1 = ctot[0];
        Blas<real>::dgemm("N", "N", k, nlp1, ktempm1+1, ONE, Q, ldq, Vt2, ldvt2, ZERO, Vt, ldvt);
        ktempm1 = 1 + ctot[0] + ctot[1];
        if (ktempm1<ldvt2)
        {
            Blas<real>::dgem("N", "N", k, nlp1, ctot[2], ONE, &Q[ldq*ktempm1], ldq, &Vt2[ktempm1],
                             ldvt2, ONE, Vt, ldvt);
        }
        ktempm1 = ctot[0];
        int nrp1 = nr + sqre;
        int qind = ldq*ktempm1;
        if (ktempm1>0)
        {
            for (i=0; i<k; i++)
            {
                Q[i+qind] = Q[i];
            }
            for (i=nlp1; i<m; i++)
            {
                Vt2[ktempm1+ldvt2*i] = Vt2[ldvt2*i];
            }
        }
        ctemp = 1 + ctot[1] + ctot[2];
        Blas<real>::dgem("N", "N", k, nrp1, ctemp, ONE, &Q[qind], ldq, &Vt2[ktempm1+ldvt2*nlp1],
                         ldvt2, ZERO, &Vt[ldvt*nlp1], ldvt);
    }

    /* dlasd4: This subroutine computes the square root of the i-th updated eigenvalue of a
     * positive symmetric rank-one modification to a positive diagonal matrix whose entries are
     * given as the squares of the corresponding entries in the array d, and that
     *     0 <= d[i] < d[j]  for  i < j
     * and that rho > 0. This is arranged by the calling routine, and is no loss in generality.
     * The rank-one modified system is thus
     *     diag( d ) * diag( d ) +  rho * z * z_transpose.
     * where we assume the Euclidean norm of z is 1.
     * The method consists of approximating the rational functions in the secular equation by
     * simpler interpolating rational functions.
     * Parameters: n: The length of all arrays.
     *             i: The index of the eigenvalue to be computed. 0<=i<n.
     *                NOTE: zero-based index!
     *             d: an array, dimension (n)
     *                The original eigenvalues. It is assumed that they are in order,
     *                0 <= d[i] < d[j] for i < j.
     *             z: an array, dimension (n)
     *                The components of the updating vector.
     *             delta: an array, dimension (n)
     *                    If n!=1, delta contains (d[j] - sigma_I) in its j-th component.
     *                    If n==1, then delta[0] = 1. The vector delta contains the information
     *                             necessary to construct the (singular) eigenvectors.
     *             rho: The scalar in the symmetric updating formula.
     *             sigma: The computed sigma_i, the i-th updated eigenvalue.
     *             work: an array, dimension (n)
     *                   If n!=1, work contains (d[j] + sigma_I) in its j-th component.
     *                   If n==1, then work[0] = 1.
     *             info: ==0: successful exit
     *                   > 0: if info==1, the updating process failed.
     * Internal Parameters:
     *     Logical variable orgati (origin-at-i?) is used for distinguishing whether d[i] or d[i+1]
     *       is treated as the origin.
     *         orgati==true    origin at i
     *         orgati==false   origin at i+1
     *     Logical variable swtch3 (switch-for-3-poles?) is for noting if we are working with THREE
     *       poles!
     *     MAXIT is the maximum number of iterations allowed for each eigenvalue.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016
     * Contributors:
     *     Ren-Cang Li, Computer Science Division, University of California at Berkeley, USA     */
    static void dlasd4(int n, int i, real const* d, real const* z, real* delta, real rho,
                       real& sigma, real* work, int& info)
    {
        const int MAXIT = 400;
        // Since this routine is called in an inner loop, we do no argument checking.
        // Quick return for n==1 and 2.
        info = 0;
        if (n==1)
        {
            // Presumably, i==0 upon entry
            sigma = std::sqrt(d[0]*d[0]+rho*z[0]*z[0]);
            delta[0] = ONE;
            work[0] = ONE;
            return;
        }
        if (n==2)
        {
            dlasd5(i, d, z, delta, rho, sigma, work);
            return;
        }
        // Compute machine epsilon
        real eps = dlamch("Epsilon");
        real rhoinv = ONE / rho;
        real tau2 = ZERO;
        int ii, iter, j, niter;
        int nm1 = n-1;
        int nm2 = n-2;
        real a, b, c, delsq, dphi, dpsi, erretm, eta, phi, psi, tau, temp, temp1, w;
        // The case i==n-1
        if (i==nm1)
        {
            // Initialize some basic variables
            ii = nm2;
            niter = 0;
            // Calculate initial guess
            temp = rho / TWO;
            // If ||z||_2 is not one, then temp should be set to rho * ||z||_2^2 / TWO
            temp1 = temp / (d[nm1]+std::sqrt(d[nm1]*d[nm1]+temp));
            for (j=0; j<n; j++)
            {
                work[j] = d[j] + d[nm1] + temp1;
                delta[j] = (d[j]-d[nm1]) - temp1;
            }
            psi = ZERO;
            for (j=0; j<nm2; j++)
            {
                psi += z[j]*z[j] / (delta[j]*work[j]);
            }
            c = rhoinv + psi;
            w = c + z[ii]*z[ii]/(delta[ii]*work[ii]) + z[nm1]*z[nm1]/(delta[nm1]*work[nm1]);
            if (w<=ZERO)
            {
                temp1 = std::sqrt(d[nm1]*d[nm1]+rho);
                temp = z[nm2]*z[nm2] / ((d[nm2]+temp1)*(d[nm1]-d[nm2]+rho/(d[nm1]+temp1)))
                     + z[nm1]*z[nm1] / rho;
                // The following tau2 is to approximate SIGMA_n^2 - d[nm1]*d[nm1]
                if (c<=temp)
                {
                    tau = rho;
                }
                else
                {
                    delsq = (d[nm1]-d[nm2])*(d[nm1]+d[nm2]);
                    a = -c*delsq + z[nm2]*z[nm2] + z[nm1]*z[nm1];
                    b = z[nm1]*z[nm1]*delsq;
                    if (a<ZERO)
                    {
                       tau2 = TWO*b / (std::sqrt(a*a+FOUR*b*c)-a);
                    }
                    else
                    {
                       tau2 = (a+std::sqrt(a*a+FOUR*b*c)) / (TWO*c);
                    }
                    tau = tau2 / (d[nm1]+std::sqrt(d[nm1]*d[nm1]+tau2));
                }
                // It can be proven that
                // d[nm1]^2+rho/2 <= sigma_n^2 < d[nm1]^2+tau2 <= d[nm1]^2+rho
            }
            else
            {
                delsq = (d[nm1]-d[nm2])*(d[nm1]+d[nm2]);
                a = -c*delsq + z[nm2]*z[nm2] + z[nm1]*z[nm1];
                b = z[nm1]*z[nm1]*delsq;
                // The following tau2 is to approximate sigma_n^2 - d[nm1]*d[nm1]
                if (a<ZERO)
                {
                   tau2 = TWO*b / (std::sqrt(a*a+FOUR*b*c)-a);
                }
                else
                {
                   tau2 = (a+std::sqrt(a*a+FOUR*b*c)) / (TWO*c);
                }
                tau = tau2 / (d[nm1]+std::sqrt(d[nm1]*d[nm1]+tau2));
                // It can be proven that d[nm1]^2 < d[nm1]^2+tau2 < sigma[nm1]^2 < d[nm1]^2+rho/2
            }
            // The following tau is to approximate sigma_n - d[nm1]
            //tau = tau2 / (d[nm1]+std::sqrt(d[nm1]*d[nm1]+tau2));
            sigma = d[nm1] + tau;
            for (j=0; j<n; j++)
            {
                delta[j] = (d[j]-d[nm1]) - tau;
                work[j] = d[j] + d[nm1] + tau;
            }
            // Evaluate psi and the derivative dpsi
            dpsi = ZERO;
            psi = ZERO;
            erretm = ZERO;
            for (j=0; j<=ii; j++)
            {
                temp = z[j] / (delta[j]*work[j]);
                psi += z[j]*temp;
                dpsi += temp*temp;
                erretm += psi;
            }
            erretm = std::fabs(erretm);
            // Evaluate phi and the derivative dphi
            temp = z[nm1] / (delta[nm1]*work[nm1]);
            phi = z[nm1]*temp;
            dphi = temp*temp;
            erretm = EIGHT*(-phi-psi) + erretm - phi + rhoinv;
                     //+ std::fabs(tau2)*(dpsi+dphi);
            w = rhoinv + phi + psi;
            // Test for convergence
            if (std::fabs(w)<=eps*erretm)
            {
                return;
            }
            // Calculate the new step
            niter++;
            real dtnsq1 = work[nm2]*delta[nm2];
            real dtnsq = work[nm1]*delta[nm1];
            c = w - dtnsq1*dpsi - dtnsq*dphi;
            a = (dtnsq+dtnsq1)*w - dtnsq*dtnsq1*(dpsi+dphi);
            b = dtnsq*dtnsq1*w;
            if (c<ZERO)
            {
                c = std::fabs(c);
            }
            if (c==ZERO)
            {
                eta = rho - sigma*sigma;
            }
            else if (a>=ZERO)
            {
                eta = (a+std::sqrt(std::fabs(a*a-FOUR*b*c))) / (TWO*c);
            }
            else
            {
                eta = TWO*b / (a-std::sqrt(std::fabs(a*a-FOUR*b*c)));
            }
            // Note, eta should be positive if w is negative, and eta should be negative otherwise.
            // However, if for some reason caused by roundoff, eta*w > 0, we simply use one Newton
            // step instead. This way will guarantee eta*w < 0.
            if (w*eta>ZERO)
            {
                eta = -w / (dpsi+dphi);
            }
            temp = eta - dtnsq;
            if (temp>rho)
            {
                eta = rho + dtnsq;
            }
            eta /= (sigma+std::sqrt(eta+sigma*sigma));
            tau += eta;
            sigma += eta;
            for (j=0; j<n; j++)
            {
                delta[j] -= eta;
                work[j] += eta;
            }
            // Evaluate psi and the derivative dpsi
            dpsi = ZERO;
            psi = ZERO;
            erretm = ZERO;
            for (j=0; j<=ii; j++)
            {
                temp = z[j] / (work[j]*delta[j]);
                psi += z[j]*temp;
                dpsi += temp*temp;
                erretm += psi;
            }
            erretm = std::fabs(erretm);
            // Evaluate phi and the derivative dphi
            tau2 = work[nm1]*delta[nm1];
            temp = z[nm1] / tau2;
            phi = z[nm1]*temp;
            dphi = temp*temp;
            erretm = EIGHT*(-phi-psi) + erretm - phi + rhoinv;
                     //+ std::fabs(tau2)*(dpsi+dphi);
            w = rhoinv + phi + psi;
            // Main loop to update the values of the array delta
            iter = niter + 1;
            for (niter=iter; niter<MAXIT; niter++)
            {
                // Test for convergence
                if (std::fabs(w)<=eps*erretm)
                {
                    return;
                }
                // Calculate the new step
                dtnsq1 = work[nm2]*delta[nm2];
                dtnsq = work[nm1]*delta[nm1];
                c = w - dtnsq1*dpsi - dtnsq*dphi;
                a = (dtnsq+dtnsq1)*w - dtnsq1*dtnsq*(dpsi+dphi);
                b = dtnsq1*dtnsq*w;
                if (a>=ZERO)
                {
                   eta = (a+std::sqrt(std::fabs(a*a-FOUR*b*c))) / (TWO*c);
                }
                else
                {
                   eta = TWO*b / (a-std::sqrt(std::fabs(a*a-FOUR*b*c)));
                }
                // Note, eta should be positive if w is negative, and eta should be negative
                // otherwise. However, if for some reason caused by roundoff, eta*w > 0, we simply
                // use one Newton step instead. This way will guarantee eta*w < 0.
                if (w*eta>ZERO)
                {
                    eta = -w / (dpsi+dphi);
                }
                temp = eta - dtnsq;
                if (temp<=ZERO)
                {
                    eta /= TWO;
                }
                eta /= (sigma+std::sqrt(eta+sigma*sigma));
                tau += eta;
                for (j=0; j<n; j++)
                {
                    delta[j] -= eta;
                    work[j] += eta;
                }
                // Evaluate psi and the derivative dpsi
                dpsi = ZERO;
                psi = ZERO;
                erretm = ZERO;
                for (j=0; j<=ii; j++)
                {
                    temp = z[j] / (work[j]*delta[j]);
                    psi += z[j]*temp;
                    dpsi += temp*temp;
                    erretm += psi;
                }
                erretm = std::fabs(erretm);
                // Evaluate phi and the derivative dphi
                tau2 = work[nm1]*delta[nm1];
                temp = z[nm1] / tau2;
                phi = z[nm1]*temp;
                dphi = temp*temp;
                erretm = EIGHT*(-phi-psi) + erretm - phi + rhoinv;
                         //+ std::fabs(tau2)*(dpsi+dphi);
                w = rhoinv + phi + psi;
            }
            // Return with info = 1, niter = MAXIT-1 and not converged
            info = 1;
            return;
            // End for the case i==n-1
        }
        else
        {
            // The case for i < n-1
            niter = 0;
            int ip1 = i + 1;
            // Calculate initial guess
            delsq = (d[ip1]-d[i])*(d[ip1]+d[i]);
            real delsq2 = delsq / TWO;
            real sq2 = std::sqrt((d[i]*d[i]+d[ip1]*d[ip1]) / TWO);
            temp = delsq2 / (d[i]+sq2);
            for (j=0; j<n; j++)
            {
                work[j] = d[j] + d[i] + temp;
                delta[j] = (d[j]-d[i]) - temp;
            }
            psi = ZERO;
            for (j=0; j<i; j++)
            {
                psi += z[j]*z[j] / (work[j]*delta[j]);
            }
            phi = ZERO;
            for (j=nm1; j>i+1; j--)
            {
               phi += z[j]*z[j] / (work[j]*delta[j]);
            }
            c = rhoinv + psi + phi;
            w = c + z[i]*z[i]/(work[i]*delta[i]) + z[ip1]*z[ip1]/(work[ip1]*delta[ip1]);
            bool orgati;
            real sglb, sgub;
            bool geomavg = false;
            if (w>ZERO)
            {
                // d(i)^2 < the ith sigma^2 < (d(i)^2+d(i+1)^2)/2
                // We choose d(i) as origin.
                orgati = true;
                ii = i;
                sglb = ZERO;
                sgub = delsq2  / (d[i]+sq2);
                a = c*delsq + z[i]*z[i] + z[ip1]*z[ip1];
                b = z[i]*z[i]*delsq;
                if (a>ZERO)
                {
                   tau2 = TWO*b / (a+std::sqrt(std::fabs(a*a-FOUR*b*c)));
                }
                else
                {
                   tau2 = (a-std::sqrt(std::fabs(a*a-FOUR*b*c))) / (TWO*c);
                }
                // tau2 now is an estimation of sigma^2 - d[i]^2. The following, however,
                // is the corresponding estimation of sigma - d[i].
                tau = tau2 / (d[i]+std::sqrt(d[i]*d[i]+tau2));
                temp = std::sqrt(eps);
                if (d[i]<=temp*d[ip1] && std::fabs(z[i])<=temp && d[i]>ZERO)
                {
                    tau = std::min(TEN*d[i], sgub);
                    geomavg = true;
                }
            }
            else
            {
                // (d(i)^2+d(i+1)^2)/2 <= the ith sigma^2 < d(i+1)^2/2
                // We choose d(i+1) as origin.
                orgati = false;
                ii = ip1;
                sglb = -delsq2  / (d[ii]+sq2);
                sgub = ZERO;
                a = c*delsq - z[i]*z[i] - z[ip1]*z[ip1];
                b = z[ip1]*z[ip1]*delsq;
                if (a<ZERO)
                {
                   tau2 = TWO*b / (a-std::sqrt(std::fabs(a*a+FOUR*b*c)));
                }
                else
                {
                   tau2 = -(a+std::sqrt(std::fabs(a*a+FOUR*b*c))) / (TWO*c);
                }
                // tau2 now is an estimation of sigma^2 - d[ip1]^2. The following, however,
                // is the corresponding estimation of sigma - d[ip1].
                tau = tau2 / (d[ip1]+std::sqrt(std::fabs(d[ip1]*d[ip1]+tau2)));
            }
            sigma = d[ii] + tau;
            for (j=0; j<n; j++)
            {
                work[j] = d[j] + d[ii] + tau;
                delta[j] = (d[j]-d[ii]) - tau;
            }
            int iim1 = ii - 1;
            int iip1 = ii + 1;
            // Evaluate psi and the derivative dpsi
            dpsi = ZERO;
            psi = ZERO;
            erretm = ZERO;
            for (j=0; j<=iim1; j++)
            {
                temp = z[j] / (work[j]*delta[j]);
                psi += z[j]*temp;
                dpsi += temp*temp;
                erretm += psi;
            }
            erretm = std::fabs(erretm);
            // Evaluate phi and the derivative dphi
            dphi = ZERO;
            phi = ZERO;
            for (j=nm1; j>=iip1; j--)
            {
                temp = z[j] / (work[j]*delta[j]);
                phi += z[j]*temp;
                dphi += temp*temp;
                erretm += phi;
            }
            w = rhoinv + phi + psi;
            // w is the value of the secular function with its ii-th element removed.
            bool swtch3 = false;
            if (orgati)
            {
                if (w<ZERO)
                {
                    swtch3 = true;
                }
            }
            else
            {
                if (w>ZERO)
                {
                    swtch3 = true;
                }
            }
            if (ii==0 || ii==nm1)
            {
                swtch3 = false;
            }
            temp = z[ii] / (work[ii]*delta[ii]);
            real dw = dpsi + dphi + temp*temp;
            temp *= z[ii];
            w += temp;
            erretm = EIGHT*(phi-psi) + erretm + TWO*rhoinv + THREE*std::fabs(temp);
                     //+ std::fabs(tau2)*dw;
            // Test for convergence
            if (std::fabs(w)<=eps*erretm)
            {
                return;
            }
            if (w<=ZERO)
            {
               sglb = std::max(sglb, tau);
            }
            else
            {
               sgub = std::min(sgub, tau);
            }
            // Calculate the new step
            real dtiim, dtiip, dtipsq, dtisq;
            real dd[3], zz[3];
            niter++;
            if (!swtch3)
            {
                dtipsq = work[ip1]*delta[ip1];
                dtisq = work[i]*delta[i];
                if (orgati)
                {
                   c = w - dtipsq*dw + delsq*(z[i]/dtisq)*(z[i]/dtisq);
                }else{
                   c = w - dtisq*dw - delsq*(z[ip1]/dtipsq)*(z[ip1]/dtipsq);
                }
                a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw;
                b = dtipsq*dtisq*w;
                if (c==ZERO)
                {
                    if (a==ZERO)
                    {
                        if (orgati)
                        {
                            a = z[i]*z[i] + dtipsq*dtipsq*(dpsi+dphi);
                        }
                        else
                        {
                            a = z[ip1]*z[ip1] + dtisq*dtisq*(dpsi+dphi);
                        }
                    }
                    eta = b / a;
                }
                else if (a<=ZERO)
                {
                    eta = (a-std::sqrt(std::fabs(a*a-FOUR*b*c))) / (TWO*c);
                }
                else
                {
                    eta = TWO*b / (a+std::sqrt(std::fabs(a*a-FOUR*b*c)));
                }
            }
            else
            {
                // Interpolation using THREE most relevant poles
                dtiim = work[iim1]*delta[iim1];
                dtiip = work[iip1]*delta[iip1];
                temp = rhoinv + psi + phi;
                if (orgati)
                {
                    temp1 = z[iim1] / dtiim;
                    temp1 *= temp1;
                    c = (temp-dtiip*(dpsi+dphi)) - (d[iim1]-d[iip1])*(d[iim1]+d[iip1])*temp1;
                    zz[0] = z[iim1]*z[iim1];
                    if (dpsi<temp1)
                    {
                       zz[2] = dtiip*dtiip*dphi;
                    }
                    else
                    {
                       zz[2] = dtiip*dtiip*((dpsi-temp1)+dphi);
                    }
                }
                else
                {
                    temp1 = z[iip1] / dtiip;
                    temp1 *= temp1;
                    c = (temp - dtiim*(dpsi+dphi)) - (d[iip1]-d[iim1])*(d[iim1]+d[iip1])*temp1;
                    if (dphi<temp1)
                    {
                        zz[0] = dtiim*dtiim*dpsi;
                    }
                    else
                    {
                        zz[0] = dtiim*dtiim*(dpsi+(dphi-temp1));
                    }
                    zz[2] = z[iip1]*z[iip1];
                }
                zz[1] = z[ii]*z[ii];
                dd[0] = dtiim;
                dd[1] = delta[ii]*work[ii];
                dd[2] = dtiip;
                dlaed6(niter, orgati, c, dd, zz, w, eta, info);
                if (info!=0)
                {
                    // If info is not 0, i.e., dlaed6 failed, switch back to 2 pole interpolation.
                    swtch3 = false;
                    info = 0;
                    dtipsq = work[ip1]*delta[ip1];
                    dtisq = work[i]*delta[i];
                    if (orgati)
                    {
                        c = w - dtipsq*dw + delsq*(z[i]/dtisq)*(z[i]/dtisq);
                    }
                    else
                    {
                        c = w - dtisq*dw - delsq*(z[ip1]/dtipsq)*(z[ip1]/dtipsq);
                    }
                    a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw;
                    b = dtipsq*dtisq*w;
                    if (c==ZERO)
                    {
                        if (a==ZERO)
                        {
                            if (orgati)
                            {
                                a = z[i]*z[i] + dtipsq*dtipsq*(dpsi+dphi);
                            }
                            else
                            {
                               a = z[ip1]*z[ip1] + dtisq*dtisq*(dpsi+dphi);
                            }
                        }
                        eta = b / a;
                    }
                    else if (a<=ZERO)
                    {
                        eta = (a-std::sqrt(std::fabs(a*a-FOUR*b*c))) / (TWO*c);
                    }
                    else
                    {
                        eta = TWO*b / (a+std::sqrt(std::fabs(a*a-FOUR*b*c)));
                    }
                }
            }
            // Note, eta should be positive if w is negative, and eta should be negative otherwise.
            // However, if for some reason caused by roundoff, eta*w > 0, we simply use one Newton
            // step instead. This way will guarantee eta*w < 0.
            if (w*eta>=ZERO)
            {
                eta = -w / dw;
            }
            eta /= (sigma+std::sqrt(sigma*sigma+eta));
            temp = tau + eta;
            if (temp>sgub || temp<sglb)
            {
                if (w<ZERO)
                {
                    eta = (sgub-tau) / TWO;
                }
                else
                {
                    eta = (sglb-tau) / TWO;
                }
                if (geomavg)
                {
                    if (w < ZERO)
                    {
                        if (tau > ZERO)
                        {
                            eta = std::sqrt(sgub*tau)-tau;
                        }
                    }
                    else
                    {
                        if (sglb > ZERO)
                        {
                            eta = std::sqrt(sglb*tau)-tau;
                        }
                    }
                }
            }
            real prew = w;
            tau += eta;
            sigma += eta;
            for (j=0; j<n; j++)
            {
                work[j] += eta;
                delta[j] -= eta;
            }
            // Evaluate psi and the derivative dpsi
            dpsi = ZERO;
            psi = ZERO;
            erretm = ZERO;
            for (j=0; j<=iim1; j++)
            {
                temp = z[j] / (work[j]*delta[j]);
                psi += z[j]*temp;
                dpsi += temp*temp;
                erretm += psi;
            }
            erretm = std::fabs(erretm);
            // Evaluate phi and the derivative dphi
            dphi = ZERO;
            phi = ZERO;
            for (j=nm1; j>=iip1; j--)
            {
                temp = z[j] / (work[j]*delta[j]);
                phi += z[j]*temp;
                dphi += temp*temp;
                erretm += phi;
            }
            tau2 = work[ii]*delta[ii];
            temp = z[ii] / tau2;
            dw = dpsi + dphi + temp*temp;
            temp *= z[ii];
            w = rhoinv + phi + psi + temp;
            erretm = EIGHT*(phi-psi) + erretm + TWO*rhoinv + THREE*std::fabs(temp);
                     //+ std::fabs(tau2)*dw
            bool swtch = false;
            if (orgati)
            {
                if (-w>std::fabs(prew) / TEN)
                {
                    swtch = true;
                }
            }
            else
            {
                if (w>std::fabs(prew) / TEN)
                {
                    swtch = true;
                }
            }
            // Main loop to update the values of the array delta and work
            real temp2;
            iter = niter + 1;
            for (niter=iter; niter<MAXIT; niter++)
            {
                // Test for convergence
                if (std::fabs(w)<=eps*erretm)
                    //|| (sgub-sglb)<=EIGHT*std::fabs(sgub+sglb)){
                {
                    return;
                }
                if (w<=ZERO)
                {
                   sglb = std::max(sglb, tau);
                }
                else
                {
                   sgub = std::min(sgub, tau);
                }
                // Calculate the new step
                if (!swtch3)
                {
                    dtipsq = work[ip1]*delta[ip1];
                    dtisq = work[i]*delta[i];
                    if (!swtch)
                    {
                        if (orgati)
                        {
                            c = w - dtipsq*dw + delsq*(z[i]/dtisq)*(z[i]/dtisq);
                        }
                        else
                        {
                            c = w - dtisq*dw - delsq*(z[ip1]/dtipsq)*(z[ip1]/dtipsq);
                        }
                    }
                    else
                    {
                        temp = z[ii] / (work[ii]*delta[ii]);
                        if (orgati)
                        {
                            dpsi += temp*temp;
                        }
                        else
                        {
                            dphi += temp*temp;
                        }
                        c = w - dtisq*dpsi - dtipsq*dphi;
                    }
                    a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw;
                    b = dtipsq*dtisq*w;
                    if (c==ZERO)
                    {
                        if (a==ZERO)
                        {
                            if (!swtch)
                            {
                                if (orgati)
                                {
                                    a = z[i]*z[i] + dtipsq*dtipsq*(dpsi+dphi);
                                }
                                else
                                {
                                    a = z[ip1]*z[ip1] + dtisq*dtisq*(dpsi+dphi);
                                }
                            }
                            else
                            {
                               a = dtisq*dtisq*dpsi + dtipsq*dtipsq*dphi;
                            }
                        }
                        eta = b / a;
                    }
                    else if (a<=ZERO)
                    {
                        eta = (a-std::sqrt(std::fabs(a*a-FOUR*b*c))) / (TWO*c);
                    }
                    else
                    {
                        eta = TWO*b / (a+std::sqrt(std::fabs(a*a-FOUR*b*c)));
                    }
                }
                else
                {
                    // Interpolation using THREE most relevant poles
                    dtiim = work[iim1]*delta[iim1];
                    dtiip = work[iip1]*delta[iip1];
                    temp = rhoinv + psi + phi;
                    if (swtch)
                    {
                        c = temp - dtiim*dpsi - dtiip*dphi;
                        zz[0] = dtiim*dtiim*dpsi;
                        zz[2] = dtiip*dtiip*dphi;
                    }
                    else
                    {
                        if (orgati)
                        {
                            temp1 = z[iim1] / dtiim;
                            temp1 *= temp1;
                            temp2 = (d[iim1]-d[iip1]) * (d[iim1]+d[iip1]) * temp1;
                            c = temp - dtiip*(dpsi+dphi) - temp2;
                            zz[0] = z[iim1]*z[iim1];
                            if (dpsi<temp1)
                            {
                                zz[2] = dtiip*dtiip*dphi;
                            }
                            else
                            {
                                zz[2] = dtiip*dtiip*((dpsi-temp1)+dphi);
                            }
                        }
                        else
                        {
                            temp1 = z[iip1] / dtiip;
                            temp1 *= temp1;
                            temp2 = (d[iip1]-d[iim1]) * (d[iim1]+d[iip1]) * temp1;
                            c = temp - dtiim*(dpsi+dphi) - temp2;
                            if (dphi<temp1)
                            {
                                zz[0] = dtiim*dtiim*dpsi;
                            }
                            else
                            {
                                zz[0] = dtiim*dtiim*(dpsi+(dphi-temp1));
                            }
                            zz[2] = z[iip1]*z[iip1];
                        }
                    }
                    dd[0] = dtiim;
                    dd[1] = delta[ii]*work[ii];
                    dd[2] = dtiip;
                    dlaed6(niter, orgati, c, dd, zz, w, eta, info);
                    if (info!=0)
                    {
                        // If info is not 0, i.e., dlaed6 failed, switch back to two pole
                        // interpolation
                        swtch3 = false;
                        info = 0;
                        dtipsq = work[ip1]*delta[ip1];
                        dtisq = work[i]*delta[i];
                        if (!swtch)
                        {
                            if (orgati)
                            {
                                c = w - dtipsq*dw + delsq*(z[i]/dtisq)*(z[i]/dtisq);
                            }
                            else
                            {
                                c = w - dtisq*dw - delsq*(z[ip1]/dtipsq)*(z[ip1]/dtipsq);
                            }
                        }
                        else
                        {
                            temp = z[ii] / (work[ii]*delta[ii]);
                            if (orgati)
                            {
                               dpsi += temp*temp;
                            }
                            else
                            {
                               dphi += temp*temp;
                            }
                            c = w - dtisq*dpsi - dtipsq*dphi;
                        }
                        a = (dtipsq+dtisq)*w - dtipsq*dtisq*dw;
                        b = dtipsq*dtisq*w;
                        if (c==ZERO)
                        {
                            if (a==ZERO)
                            {
                                if (!swtch)
                                {
                                    if (orgati)
                                    {
                                        a = z[i]*z[i] + dtipsq*dtipsq*(dpsi+dphi);
                                    }
                                    else
                                    {
                                        a = z[ip1]*z[ip1] + dtisq*dtisq*(dpsi+dphi);
                                    }
                                }
                                else
                                {
                                    a = dtisq*dtisq*dpsi + dtipsq*dtipsq*dphi;
                                }
                            }
                            eta = b / a;
                        }
                        else if (a<=ZERO)
                        {
                            eta = (a-std::sqrt(std::fabs(a*a-FOUR*b*c))) / (TWO*c);
                        }
                        else
                        {
                            eta = TWO*b / (a+std::sqrt(std::fabs(a*a-FOUR*b*c)));
                        }
                    }
                }
                // Note, eta should be positive if w is negative, and eta should be negative
                // otherwise. However, if for some reason caused by roundoff, eta*w > 0, we simply
                // use one Newton step instead. This way will guarantee eta*w < 0.
                if (w*eta>=ZERO)
                {
                    eta = -w / dw;
                }
                eta /= (sigma+std::sqrt(sigma*sigma+eta));
                temp = tau+eta;
                if (temp>sgub || temp<sglb)
                {
                    if (w<ZERO)
                    {
                        eta = (sgub-tau) / TWO;
                    }
                    else
                    {
                        eta = (sglb-tau) / TWO;
                    }
                    if (geomavg)
                    {
                        if (w < ZERO)
                        {
                            if (tau > ZERO)
                            {
                                eta = std::sqrt(sgub*tau)-tau;
                            }
                        }
                        else
                        {
                            if (sglb > ZERO)
                            {
                                eta = std::sqrt(sglb*tau)-tau;
                            }
                        }
                    }
                }
                prew = w;
                tau += eta;
                sigma += eta;
                for (j=0; j<n; j++)
                {
                    work[j] += eta;
                    delta[j] -= eta;
                }
                // Evaluate psi and the derivative dpsi
                dpsi = ZERO;
                psi = ZERO;
                erretm = ZERO;
                for (j=0; j<=iim1; j++)
                {
                    temp = z[j] / (work[j]*delta[j]);
                    psi += z[j]*temp;
                    dpsi += temp*temp;
                    erretm += psi;
                }
                erretm = std::fabs(erretm);
                // Evaluate phi and the derivative dphi
                dphi = ZERO;
                phi = ZERO;
                for (j=nm1; j>=iip1; j--)
                {
                    temp = z[j] / (work[j]*delta[j]);
                    phi += z[j]*temp;
                    dphi += temp*temp;
                    erretm += phi;
                }
                tau2 = work[ii]*delta[ii];
                temp = z[ii] / tau2;
                dw = dpsi + dphi + temp*temp;
                temp *= z[ii];
                w = rhoinv + phi + psi + temp;
                erretm = EIGHT*(phi-psi) + erretm + TWO*rhoinv + THREE*std::fabs(temp);
                         //+ std::fabs(tau2)*dw;
                if (w*prew>ZERO && std::fabs(w)>std::fabs(prew) / TEN)
                {
                    swtch = !swtch;
                }
            }
            // Return with info = 1, niter = MAXIT-1 and not converged
            info = 1;
        }
    }

    /* dlasd5: This subroutine computes the square root of the i-th eigenvalue of a positive
     * symmetric rank-one modification of a 2-by-2 diagonal matrix
     *     diag( D ) * diag( D ) +  rho * z * transpose(z).
     * The diagonal entries in the array d are assumed to satisfy
     *     0 <= d[i] < d[j]  for  i < j .
     * We also assume rho > 0 and that the Euclidean norm of the vector z is one.
     * Parameters: i: The index of the eigenvalue to be computed. i==0 or i==1.
     *                NOTE: zero-based index!
     *             d: an array, dimension (2)
     *                The original eigenvalues. We assume 0 <= d[0] < d[1].
     *             z: an array, dimension (2)
     *                The components of the updating vector.
     *             delta: an array, dimension (2)
     *                    Contains (d[j] - sigma_I) in its  j-th component. The vector delta
     *                    contains the information necessary to construct the eigenvectors.
     *             rho: The scalar in the symmetric updating formula.
     *             dsigma: The computed sigma_I, the i-th updated eigenvalue.
     *             work: an array, dimension (2)
     *                   work contains (d[j] + sigma_I) in its  j-th component.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Contributors:
     *     Ren-Cang Li, Computer Science Division, University of California at Berkeley, USA     */
    static void dlasd5(int i, real const* d, real const* z, real* delta, real rho, real& dsigma,
                       real* work)
    {
        real b, c, tau;
        real del = d[1] - d[0];
        real delsq = del*(d[1]+d[0]);
        if (i==0)
        {
            real w = ONE
                    + FOUR * rho * (z[1]*z[1]/(d[0]+THREE*d[1])-z[0]*z[0]/(THREE*d[0]+d[1])) / del;
            if (w>ZERO)
            {
                b = delsq + rho*(z[0]*z[0]+z[1]*z[1]);
                c = rho*z[0]*z[0]*delsq;
                // b > ZERO, always
                // The following tau is dsigma * dsigma - d[0] * d[0]
                tau = TWO*c / (b+std::sqrt(std::fabs(b*b-FOUR*c)));
                // The following tau is dsigma - d[0]
                tau /= (d[0]+std::sqrt(d[0]*d[0]+tau));
                dsigma = d[0] + tau;
                delta[0] = -tau;
                delta[1] = del - tau;
                work[0] = TWO*d[0] + tau;
                work[1] = (d[0]+tau) + d[1];
                //delta[0] = -z[0] / tau;
                //delta[1] = z[1] / (del-tau);
            }
            else
            {
                b = -delsq + rho*(z[0]*z[0]+z[1]*z[1]);
                c = rho*z[1]*z[1]*delsq;
                // The following tau is dsigma * dsigma - d[1] * d[1]
                if (b>ZERO)
                {
                   tau = -TWO*c / (b+std::sqrt(b*b+FOUR*c));
                }
                else
                {
                   tau = (b-std::sqrt(b*b+FOUR*c)) / TWO;
                }
                // The following tau is dsigma - d[1]
                tau /= (d[1]+std::sqrt(std::fabs(d[1]*d[1]+tau)));
                dsigma = d[1] + tau;
                delta[0] = -(del+tau);
                delta[1] = -tau;
                work[0] = d[0] + tau + d[1];
                work[1] = TWO*d[1] + tau;
                //delta[0] = -z[0] / (del+tau);
                //delta[1] = -z[1] / tau;
            }
            //temp = std::sqrt(delta[0]*delta[0]+delta[1]*delta[1]);
            //delta[0] /= temp;
            //delta[1] /= temp;
        }
        else
        {
            // Now i==1
            b = -delsq + rho*(z[0]*z[0]+z[1]*z[1]);
            c = rho*z[1]*z[1]*delsq;
            // The following tau is dsigma * dsigma - d[1] * d[1]
            if (b>ZERO)
            {
               tau = (b+std::sqrt(b*b+FOUR*c)) / TWO;
            }
            else
            {
               tau = TWO*c / (-b+std::sqrt(b*b+FOUR*c));
            }
            // The following tau is dsigma - d[1]
            tau /= (d[1]+std::sqrt(d[1]*d[1]+tau));
            dsigma = d[1] + tau;
            delta[0] = -(del+tau);
            delta[1] = -tau;
            work[0] = d[0] + tau + d[1];
            work[1] = TWO*d[1] + tau;
            //delta[0] = -z[0] / (del+tau);
            //delta[1] = -z[1] / tau;
            //temp = std::sqrt(delta[0]*delta[0]+delta[1]*delta[1]);
            //delta[0] /= temp;
            //delta[1] /= temp;
        }
    }

    /* dlasd6 computes the SVD of an updated upper bidiagonal matrix B obtained by merging two
     * smaller ones by appending a row. This routine is used only for the problem which requires
     * all singular values and optionally singular vector matrices in factored form. B is an n-by-m
     * matrix with n = nl+nr+1 and m = n+sqre. A related subroutine, dlasd1, handles the case in
     * which all singular values and singular vectors of the bidiagonal matrix are desired.
     * dlasd6 computes the SVD as follows:
     *                 ( D1(in)   0    0      0 )
     *     B = U(in) * (   z1^T   a   z2^T    b ) * VT(in)
     *                 (   0      0   D2(in)  0 )
     *       = U(out) * ( d(out) 0) * VT(out)
     * where z^T = (z1^T a z2^T b) = u^T VT^T, and u is a vector of dimension m with alpha and beta
     * in the nl+1 and nl+2 th entries and zeros elsewhere; and the entry b is empty if sqre==0.
     * The singular values of B can be computed using D1, D2, the first components of all the right
     * singular vectors of the lower block, andthe last components of all the right singular
     * vectors of the upper block. These components are stored and updated in vf and vl,
     * respectively, in dlasd6. Hence U and VT are not explicitly referenced.
     * The singular values are stored in d. The algorithm consists of two stages:
     *   The first stage consists of deflating the size of the problem when there are multiple
     *     singular values or if there is a zero in the z vector. For each such occurrence the
     *     dimension of the secular equation problem is reduced by one. This stage is performed by
     *     the routine dlasd7.
     *   The second stage consists of calculating the updated
     *     singular values. This is done by finding the roots of the secular equation via the
     *     routine dlasd4 (as called by dlasd8). This routine also updates vf and vl and computes
     *     the distances between the updated singular values and the old singular values.
     * dlasd6 is called from dlasda.
     * Parameters: icompq: Specifies whether singular vectors are to be computed in factored form:
     *                     ==0: Compute singular values only.
     *                     ==1: Compute singular vectors in factored form as well.
     *             nl: The row dimension of the upper block. nl>=1.
     *             nr: The row dimension of the lower block. nr>=1.
     *             sqre: ==0: the lower block is an nr-by-nr square matrix.
     *                   ==1: the lower block is an nr-by-(nr+1) rectangular matrix.
     *                   The bidiagonal matrix has row dimension n = nl+nr+1, and column dimension
     *                   m = n+sqre.
     *             d: an array, dimension (nl+nr+1).
     *                On entry d[0:nl-1,0:nl-1] contains the singular values of the upper block,
     *                and d[nl+1:n-1] contains the singular values of the lower block.
     *                On exit d[0:n-1] contains the singular values of the modified matrix.
     *             vf: an array, dimension (m)
     *                 On entry, vf[0:nl] contains the first components of all right singular
     *                 vectors of the upper block; and vf[nl+1:m-1] contains the first components of
     *                 all right singular vectors of the lower block.
     *                 On exit, vf contains the first components of all right singular vectors of
     *                 the bidiagonal matrix.
     *             vl: an array, dimension (m)
     *                 On entry, vl[0:nl] contains the  last components of all right singular
     *                 vectors of the upper block; and vl[nl+1:m-1] contains the last components of
     *                 all right singular vectors of the lower block.
     *                 On exit, vl contains the last components of all right singular vectors of
     *                 the bidiagonal matrix.
     *             alpha: Contains the diagonal element associated with the added row.
     *             beta: Contains the off-diagonal element associated with the added row.
     *             idxq: an integer array, dimension (n)
     *                   This contains the permutation which will reintegrate the subproblem just
     *                   solved back into sorted order, i.e. d[idxq[0:n-1]] will be in
     *                   ascending order.
     *                   NOTE: zero-based indices!
     *             perm: an integer array, dimension (n)
     *                   The permutations (from deflation and sorting) to be applied to each block.
     *                   Not referenced if icompq==0.
     *                   NOTE: zero-based indices!
     *             givptr: The number of Givens rotations which took place in this subproblem.
     *                     Not referenced if icompq==0.
     *             Givcol: an integer array, dimension (ldgcol, 2)
     *                     Each pair of numbers indicates a pair of columns to take place in a
     *                     Givens rotation. Not referenced if icompq==0.
     *                     NOTE: zero-based indices!
     *             ldgcol: leading dimension of Givcol, must be at least n.
     *             Givnum: an array, dimension (ldgnum, 2)
     *                     Each number indicates the c or s value to be used in the corresponding
     *                     Givens rotation. Not referenced if icompq==0.
     *             ldgnum: The leading dimension of Givnum and Poles, must be at least n.
     *             Poles: an array, dimension (ldgnum, 2)
     *                    On exit, Poles[:,0] is an array containing the new singular values
     *                    obtained from solving the secular equation, and Poles[:,1] is an array
     *                    containing the poles in the secular equation.
     *                    Not referenced if icompq==0.
     *             difl: an array, dimension (n)
     *                   On exit, difl[i] is the distance between i-th updated (undeflated)
     *                   singular value and the i-th (undeflated) old singular value.
     *             Difr: an array, dimension (LDDIFR, 2) if icompq==1 and
     *                             dimension (k)         if icompq==0.
     *                   On exit, Difr[i,0] = d[i] - DSIGMA(I+1), Difr[k-1,0] is not defined and
     *                   will not be referenced.
     *                   If icompq==1, Difr[0:k-1,1] is an array containing the normalizing factors
     *                   for the right singular vector matrix.
     *                   See dlasd8 for details on difl and Difr.
     *             z: an array, dimension (m)
     *                The first elements of this array contain the components of the
     *                deflation-adjusted updating row vector.
     *             k: Contains the dimension of the non-deflated matrix,
     *                This is the order of the related secular equation. 1 <= k <= n.
     *             c: c contains garbage if sqre==0 and the c-value of a Givens rotation related to
     *                the right null space if sqre==1.
     *             s: s contains garbage if sqre==0 and the s-value of a Givens rotation related to
     *                the right null space if sqre==1.
     *             work: an array, dimension (4*m)
     *             iwork: an integer array, dimension (3*n)
     *             info: ==0: Successful exit.
     *                   < 0: if info==-i, the i-th argument had an illegal value.
     *                   > 0: if info==1, a singular value did not converge
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: June 2016
     * Contributors:
     *     Ming Gu and Huan Ren, Computer Science Division,
     *     University of California at Berkeley, USA                                             */
    static void dlasd6(int icompq, int nl, int nr, int sqre, real* d, real* vf, real* vl,
                       real& alpha, real& beta, int* idxq, int* perm, int& givptr, int* Givcol,
                       int ldgcol, real* Givnum, int ldgnum, real* Poles, real* difl, real* Difr,
                       real* z, int& k, real& c, real& s, real* work, int* iwork, int& info)
    {
        // Test the input parameters.
        info = 0;
        int n = nl + nr + 1;
        int m = n + sqre;
        if (icompq<0 || icompq>1)
        {
            info = -1;
        }
        else if (nl<1)
        {
            info = -2;
        }
        else if (nr<1)
        {
            info = -3;
        }
        else if (sqre<0 || sqre>1)
        {
            info = -4;
        }
        else if (ldgcol<n)
        {
            info = -14;
        }
        else if (ldgnum<n)
        {
            info = -16;
        }
        if (info!=0)
        {
            xerbla("DLASD6", -info);
            return;
        }
        // The following values are for bookkeeping purposes only. They are integer pointers which
        // indicate the portion of the workspace used by a particular array in dlasd7 and dlasd8.
        int isigma = 0;
        int iw     = isigma + n;
        int ivfw   = iw + m;
        int ivlw   = ivfw + m;
        int idx  = 0;
        int idxc = idx + n;
        int idxp = idxc + n;
        // Scale.
        real orgnrm = std::max(std::fabs(alpha), std::fabs(beta));
        d[nl] = ZERO;
        for (int I=1; I<=n; I++)
        {
            if (std::fabs(d[I-1])>orgnrm)
            {
                orgnrm = std::fabs(d[I-1]);
            }
        }
        dlascl("G", 0, 0, orgnrm, ONE, n, 1, d, n, info);
        alpha /= orgnrm;
        beta  /= orgnrm;
        // Sort and Deflate singular values.
        dlasd7(icompq, nl, nr, sqre, k, d, z, &work[iw], vf, &work[ivfw], vl, &work[ivlw], alpha,
               beta, &work[isigma], &iwork[idx], &iwork[idxp], idxq, perm, givptr, Givcol, ldgcol,
               Givnum, ldgnum, c, s, info);
        // Solve Secular Equation, compute difl, Difr, and update vf, vl.
        dlasd8(icompq, k, d, z, vf, vl, difl, Difr, ldgnum, &work[isigma], &work[iw], info);
        // Report the possible convergence failure.
        if (info!=0)
        {
            return;
        }
        // Save the poles if icompq==1.
        if (icompq==1)
        {
            Blas<real>::dcopy(k, d, 1, Poles, 1);
            Blas<real>::dcopy(k, &work[isigma], 1, &Poles[ldgnum], 1);
        }
        // Unscale.
        dlascl("G", 0, 0, ONE, orgnrm, n, 1, d, n, info);
        // Prepare the idxq sorting permutation.
        dlamrg(k, n-k, d, 1, -1, idxq);
    }

    /* dlasd7 merges the two sets of singular values together into a single sorted set. Then it
     * tries to deflate the size of the problem. There are two ways in which deflation can occur:
     * when two or more singular values are close together or if there is a tiny entry in the z
     * vector. For each such occurrence the order of the related secular equation problem is
     * reduced by one.
     * dlasd7 is called from dlasd6.
     * Parameters: icompq: Specifies whether singular vectors are to be computed in compact form,
     *                     as follows:
     *                     0: Compute singular values only.
     *                     1: Compute singular vectors of upper bidiagonal matrix in compact form.
     *             nl: The row dimension of the upper block. nl>=1.
     *             nr: The row dimension of the lower block. nr>=1.
     *             sqre: ==0: the lower block is an nr-by-nr square matrix.
     *                   ==1: the lower block is an nr-by-(nr+1) rectangular matrix.
     *                   The bidiagonal matrix has n = nl+nr+1 rows and m = n+sqre >= n columns.
     *             k: Contains the dimension of the non-deflated matrix, this is the order of the
     *                related secular equation. 1 <= k <=n.
     *             d: an array, dimension (n)
     *                On entry d contains the singular values of the two submatrices to be
     *                combined. On exit d contains the trailing (n-k) updated singular values
     *                (those which were deflated) sorted into increasing order.
     *             z: an array, dimension (m)
     *                On exit z contains the updating row vector in the secular equation.
     *             zw: an array, dimension (m)
     *                 Workspace for z.
     *             vf: an array, dimension (m)
     *                 On entry, vf[0:nl] contains the first components of all right singular
     *                 vectors of the upper block; and vf[nl+1:m-1] contains the first components
     *                 of all right singular vectors of the lower block.
     *                 On exit, vf contains the first components of all right singular vectors of
     *                 the bidiagonal matrix.
     *             vfw: an array, dimension (m)
     *                  Workspace for vf.
     *             vl: an array, dimension (m)
     *                 On entry, vl[0:nl] contains the last components of all right singular
     *                 vectors of the upper block; and vl[nl+1:m-1] contains the last components of
     *                 all right singular vectors of the lower block.
     *                 On exit, vl contains the last components of all right singular vectors of
     *                 the bidiagonal matrix.
     *             vlw: an array, dimension (m)
     *                  Workspace for vl.
     *             alpha: Contains the diagonal element associated with the added row.
     *             beta: Contains the off-diagonal element associated with the added row.
     *             dsigma: an array, dimension (n)
     *                     Contains a copy of the diagonal elements (k-1 singular values and one
     *                     zero) in the secular equation.
     *             idx: an integer array, dimension (n)
     *                  This will contain the permutation used to sort the contents of d into
     *                  ascending order.
     *                  NOTE: zero-based indices!
     *             idxp: an integer array, dimension (n)
     *                   This will contain the permutation used to place deflated values of d at
     *                   the end of the array. On output idxp[1:k-1] points to the nondeflated
     *                   d-values and idxp[k:n-1] points to the deflated singular values.
     *                   NOTE: zero-based indices!
     *             idxq: an array, dimension (n)
     *                   This contains the permutation which separately sorts the two sub-problems
     *                   in d into ascending order. Note that entries in the first half of this
     *                   permutation must first be moved one position backward; and entries in the
     *                   second half must first have nl+1 added to their values.
     *                   NOTE: zero-based indices!
     *             perm: an integer array, dimension (n)
     *                   The permutations (from deflation and sorting) to be applied to each
     *                   singular block. Not referenced if icompq==0.
     *                   NOTE: zero-based indices!
     *             givptr: The number of Givens rotations which took place in this subproblem.
     *                     Not referenced if icompq==0.
     *             Givcol: an integer array, dimension (ldgcol, 2)
     *                     Each pair of numbers indicates a pair of columns to take place in a
     *                     Givens rotation. Not referenced if icompq==0.
     *                     NOTE: zero-based indices!
     *             ldgcol: The leading dimension of Givcol, must be at least n.
     *             Givnum: an array, dimension (ldgnum, 2)
     *                     Each number indicates the c or s value to be used in the corresponding
     *                     Givens rotation. Not referenced if icompq==0.
     *             ldgnum: The leading dimension of Givnum, must be at least n.
     *             c: contains garbage if sqre==0 and the c-value of a Givens rotation related to
     *                the right null space if sqre==1.
     *             s: contains garbage if sqre==0 and the s-value of a Givens rotation related to
     *                the right null space if sqre==1.
     *             info: ==0: successful exit.
     *                   < 0: if info==-i, the i-th argument had an illegal value.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016
     * Contributors:
     *     Ming Gu and Huan Ren, Computer Science Division,
     *     University of California at Berkeley, USA                                             */
    static void dlasd7(int icompq, int nl, int nr, int sqre, int& k, real* d, real* z, real* zw,
                       real* vf, real* vfw, real* vl, real* vlw, real alpha, real beta,
                       real* dsigma, int* idx, int* idxp, int* idxq, int* perm, int& givptr,
                       int* Givcol, int ldgcol, real* Givnum, int ldgnum, int& c, int& s,
                       int& info)
    {
        // Test the input parameters.
        info = 0;
        int n = nl + nr + 1;
        int m = n + sqre;
        if (icompq<0 || icompq>1)
        {
            info = -1;
        }
        else if (nl<1)
        {
            info = -2;
        }
        else if (nr<1)
        {
            info = -3;
        }
        else if (sqre<0 || sqre>1)
        {
            info = -4;
        }
        else if (ldgcol<n)
        {
            info = -22;
        }
        else if (ldgnum<n)
        {
            info = -24;
        }
        if (info!=0)
        {
            xerbla("DLASD7", -info);
            return;
        }
        int nlp1 = nl + 1;
        if (icompq==1)
        {
            givptr = 0;
        }
        // Generate the first part of the vector z and move the singular values in the first part
        // of d one position backward.
        real z1 = alpha*vl[nl];
        vl[nl] = ZERO;
        real tau = vf[nl];
        int i;
        for (i=nl-1; i>=0; i--)
        {
            z[i+1]    = alpha*vl[i];
            vl[i]     = ZERO;
            vf[i+1]   = vf[i];
            d[i+1]    = d[i];
            idxq[i+1] = idxq[i] + 1;
        }
        vf[0] = tau;
        // Generate the second part of the vector z.
        for (i=nlp1; i<m; i++)
        {
            z[i] = beta*vf[i];
            vf[i] = ZERO;
        }
        // Sort the singular values into increasing order
        for (i=nlp1; i<n; i++)
        {
            idxq[i] += nlp1;
        }
        // dsigma, IDXC, IDXC, and zw are used as storage space.
        for (i=1; i<n; i++)
        {
            dsigma[i] =  d[idxq[i]];
            zw[i]     =  z[idxq[i]];
            vfw[i]    = vf[idxq[i]];
            vlw[i]    = vl[idxq[i]];
        }
        dlamrg(nl, nr, &dsigma[1], 1, 1, &idx[1]);
        int idxi;
        for (i=1; i<n; i++)
        {
            idxi  = 1 + idx[i];
            d[i]  = dsigma[idxi];
            z[i]  = zw[idxi];
            vf[i] = vfw[idxi];
            vl[i] = vlw[idxi];
        }
        // Calculate the allowable deflation tolerence
        real eps = dlamch("Epsilon");
        real tol = std::max(std::fabs(alpha), std::fabs(beta));
        tol = EIGHT * EIGHT * eps * std::max(std::fabs(d[n-1]), tol);
        // There are 2 kinds of deflation -- first a value in the z-vector is small, second two
        // (or more) singular values are very close together (their difference is small).
        // If the value in the z-vector is small, we simply permute the array so that the
        // corresponding singular value is moved to the end.
        // If two values in the d-vector are close, we perform a two-sided rotation designed to
        // make one of the corresponding z-vector entries zero, and then permute the array so that
        // the deflated singular value is moved to the end.
        // If there are multiple singular values then the problem deflates. Here the number of
        // equal singular values are found. As each equal singular value is found, an elementary
        // reflector is computed to rotate the corresponding singular subspace so that the
        // corresponding components of z are zero in this new basis.
        k = 1;
        int k2 = n;
        bool deflate = true;
        int j, jprev, km1;
        for (j=1; j<n; j++)
        {
            if (std::fabs(z[j])<=tol)
            {
                // Deflate due to small z component.
                k2--;
                idxp[k2] = j;
                if (j==n-1)
                {
                    deflate = false;
                    break;
                }
            }
            else
            {
                jprev = j;
                break;
            }
        }
        if (deflate)
        {
            int idxj, idxjp, gptrm1;
            j = jprev;
            while (true)
            {
                j++;
                if (j>=n)
                {
                    break;
                }
                if (std::fabs(z[j])<=tol)
                {
                    // Deflate due to small z component.
                    k2--;
                    idxp[k2] = j;
                }
                else
                {
                    // Check if singular values are close enough to allow deflation.
                    if (std::fabs(d[j]-d[jprev])<=tol)
                    {
                        // Deflation is possible.
                        s = z[jprev];
                        c = z[j];
                        // Find sqrt(a^2+b^2) without overflow or destructive underflow.
                        tau = dlapy2(c, s);
                        z[j] = tau;
                        z[jprev] = ZERO;
                        c /= tau;
                        s = -s / tau;
                        // Record the appropriate Givens rotation
                        if (icompq==1)
                        {
                            givptr++;
                            idxjp = idxq[idx[jprev]+1];
                            idxj  = idxq[idx[j]+1];
                            if (idxjp<=nl)
                            {
                                idxjp--;
                            }
                            if (idxj<=nl)
                            {
                                idxj--;
                            }
                            gptrm1 = givptr-1;
                            Givcol[gptrm1+ldgcol] = idxjp;
                            Givcol[gptrm1]        = idxj;
                            Givnum[gptrm1+ldgnum] = c;
                            Givnum[gptrm1]        = s;
                        }
                        Blas<real>::drot(1, &vf[jprev], 1, &vf[j], 1, c, s);
                        Blas<real>::drot(1, &vl[jprev], 1, &vl[j], 1, c, s);
                        k2--;
                        idxp[k2] = jprev;
                        jprev = j;
                    }
                    else
                    {
                       k++;
                       km1 = k - 1;
                       zw[km1]     = z[jprev];
                       dsigma[km1] = d[jprev];
                       idxp[km1]   = jprev;
                       jprev = j;
                    }
                }
            }
            // Record the last singular value.
            k++;
            km1 = k - 1;
            zw[km1]     = z[jprev];
            dsigma[km1] = d[jprev];
            idxp[km1]   = jprev;
        }
        // Sort the singular values into dsigma. The singular values which were not deflated go
        // into the first k slots of dsigma, except that dsigma[0] is treated separately.
        int jp;
        for (j=1; j<n; j++)
        {
            jp = idxp[j];
            dsigma[j] = d[jp];
            vfw[j]    = vf[jp];
            vlw[j]    = vl[jp];
        }
        if (icompq==1)
        {
            for (j=1; j<n; j++)
            {
                jp = idxp[j];
                perm[j] = idxq[idx[jp]+1];
                if (perm[j]<=nl)
                {
                    perm[j]--;
                }
            }
        }
        // The deflated singular values go back into the last n - k slots of d.
        Blas<real>::dcopy(n-k, &dsigma[k], 1, &d[k], 1);
        // Determine dsigma[0], dsigma[1], z[0], vf[0], vl[0], vf[m-1], and vl[m-1].
        dsigma[0] = ZERO;
        real hlftol = tol / TWO;
        if (std::fabs(dsigma[1])<=hlftol)
        {
            dsigma[1] = hlftol;
        }
        if (m>n)
        {
            z[0] = dlapy2(z1, z[m-1]);
            if (z[0]<=tol)
            {
                c = ONE;
                s = ZERO;
                z[0] = tol;
            }
            else
            {
                c = z1 / z[0];
                s = -z[m-1] / z[0];
            }
            Blas<real>::drot(1, &vf[m-1], 1, &vf[0], 1, c, s);
            Blas<real>::drot(1, &vl[m-1], 1, &vl[0], 1, c, s);
        }
        else
        {
            if (std::fabs(z1)<=tol)
            {
                z[0] = tol;
            }
            else
            {
                z[0] = z1;
            }
        }
        // Restore z, vf, and vl.
        Blas<real>::dcopy(k-1,  &zw[1], 1,  &z[1], 1);
        Blas<real>::dcopy(n-1, &vfw[1], 1, &vf[1], 1);
        Blas<real>::dcopy(n-1, &vlw[1], 1, &vl[1], 1);
    }

    /* dlasd8 finds the square roots of the roots of the secular equation, as defined by the values
     * in dsigma and z. It makes the appropriate calls to dlasd4, and stores, for each  element in
     * d, the distance to its two nearest poles (elements in dsigma). It also updates the arrays vf
     * and vl, the first and last components of all the right singular vectors of the original
     * bidiagonal matrix.
     * dlasd8 is called from dlasd6.
     * Parameters: icompq: Specifies whether singular vectors are to be computed in factored form
     *                     in the calling routine:
     *                     ==0: Compute singular values only.
     *                     ==1: Compute singular vectors in factored form as well.
     *             k: The number of terms in the rational function to be solved by dlasd4. k>=1.
     *             d: an array, dimension (k)
     *                On output, d contains the updated singular values.
     *             z: an array, dimension (k)
     *                On entry, the first k elements of this array contain the components of the
     *                deflation-adjusted updating row vector.
     *                On exit, z is updated.
     *             vf: an array, dimension (k)
     *                 On entry, vf contains information passed through dbede8.
     *                 On exit, vf contains the first k components of the first components of all
     *                 right singular vectors of the bidiagonal matrix.
     *             vl: an array, dimension (k)
     *                 On entry, vl contains information passed through dbede8.
     *                 On exit, vl contains the first k components of the last components of all
     *                 right singular vectors of the bidiagonal matrix.
     *             difl: an array, dimension (k)
     *                   On exit, difl[i] = d[i] - dsigma[i].
     *             Difr: an array, dimension (lddifr, 2) if icompq==1 and
     *                             dimension (k) if icompq==0.
     *                   On exit, Difr[i,0] = d[i] - dsigma[i+1], Difr[k-1,0] is not defined and will
     *                   not be referenced.
     *                   If icompq==1, Difr[0:k-1,1] is an array containing the normalizing factors
     *                   for the right singular vector matrix.
     *             lddifr: The leading dimension of Difr, must be at least k.
     *             dsigma: an array, dimension (k)
     *                     On entry, the first k elements of this array contain the old roots of
     *                     the deflated updating problem. These are the poles of the secular
     *                     equation.
     *                     On exit, the elements of dsigma may be very slightly altered in value.
     *             work: an array, dimension (3*k)
     *             info: ==0: successful exit.
     *                   < 0: if info==-i, the i-th argument had an illegal value.
     *                   > 0: if info==1, a singular value did not converge
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: June 2017
     * Contributors:
     *     Ming Gu and Huan Ren, Computer Science Division,
     *     University of California at Berkeley, USA                                             */
    static void dlasd8(int icompq, int k, real* d, real* z, real* vf, real* vl, real* difl,
                       real* Difr, int lddifr, real* dsigma, real* work, int& info)
    {
        // Test the input parameters.
        info = 0;
        if (icompq<0 || icompq>1)
        {
            info = -1;
        }
        else if (k<1)
        {
            info = -2;
        }
        else if (lddifr<k)
        {
            info = -9;
        }
        if (info!=0)
        {
            xerbla("DLASD8", -info);
            return;
        }
        // Quick return if possible
        if (k==1)
        {
            d[0] = std::fabs(z[0]);
            difl[0] = d[0];
            if (icompq==1)
            {
                difl[1] = ONE;
                Difr[lddifr] = ONE;
            }
            return;
        }
        /* Modify values dsigma[i] to make sure all dsigma[i]-dsigma[j] can be computed with high
         * relative accuracy (barring over/underflow). This is a problem on machines without a
         * guard digit in add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2). The following
         * code replaces dsigma[i] by 2*dsigma[i]-dsigma[i], which on any of these machines zeros
         * out the bottommost bit of dsigma[i] if it is 1; this makes the subsequent subtractions
         * dsigma[i]-dsigma[j] unproblematic when cancellation occurs. On binary machines with a
         * guard digit (almost all machines) it does not change dsigma[i] at all. On hexadecimal
         * and decimal machines with a guard digit, it slightly changes the bottommost bits of
         * dsigma[i]. It does not account for hexadecimal or decimal machines without guard digits
         * (we know of none). We use a subroutine call to compute 2*dlambda[i] to prevent
         * optimizing compilers from eliminating this code.                                      */
        int i;
        for (i=0; i<k; i++)
        {
            dsigma[i] = dlamc3(dsigma[i], dsigma[i]) - dsigma[i];
        }
        // Book keeping.
        int iwk1 = 0;
        int iwk2 = iwk1 + k;
        int iwk3 = iwk2 + k;
        // Normalize z.
        real rho = Blas<real>::dnrm2(k, z, 1);
        dlascl("G", 0, 0, rho, ONE, k, 1, z, k, info);
        rho *= rho;
        // Initialize work[iwk3].
        dlaset("A", k, 1, ONE, ONE, &work[iwk3], k);
        // Compute the updated singular values, the arrays difl, Difr, and the updated z.
        int j;
        for (j=0; j<k; j++)
        {
            dlasd4(k, j, dsigma, z, &work[iwk1], rho, d[j], &work[iwk2], info);
            // If the root finder fails, report the convergence failure.
            if (info!=0)
            {
                return;
            }
            work[iwk3+j] *= work[j]*work[iwk2+j];
            difl[j] = -work[j];
            Difr[j] = -work[j+1];
            for (i=0; i<j; i++)
            {
                work[iwk3+i] *= work[i] * work[iwk2+i] / (dsigma[i]-dsigma[j])
                                 / (dsigma[i]+dsigma[j]);
            }
            for (i=j+1; i<k; i++)
            {
                work[iwk3+i] *= work[i] * work[iwk2+i] / (dsigma[i]-dsigma[j])
                                 / (dsigma[i]+dsigma[j]);
            }
        }
        // Compute updated z.
        for (i=0; i<k; i++)
        {
            z[i] = std::copysign(std::sqrt(std::fabs(work[iwk3+i])), z[i]);
        }
        // Update vf and vl.
        real diflj, difrj, dj, dsigj, dsigjp, temp;
        for (j=0; j<k; j++)
        {
            diflj = difl[j];
            dj = d[j];
            dsigj = -dsigma[j];
            if (j<k-1)
            {
                difrj = -Difr[j];
                dsigjp = -dsigma[j+1];
            }
            work[j] = -z[j] / diflj / (dsigma[j]+dj);
            for (i=0; i<j; i++)
            {
                work[i] = z[i] / (dlamc3(dsigma[i], dsigj)-diflj) / (dsigma[i]+dj);
            }
            for (i=j+1; i<k; i++)
            {
                work[i] = z[i] / (dlamc3(dsigma[i], dsigjp)+difrj) / (dsigma[i]+dj);
            }
            temp = Blas<real>::dnrm2(k, work, 1);
            work[iwk2+j] = Blas<real>::ddot(k, work, 1, vf, 1) / temp;
            work[iwk3+j] = Blas<real>::ddot(k, work, 1, vl, 1) / temp;
            if (icompq==1)
            {
                Difr[j+lddifr] = temp;
            }
        }
        Blas<real>::dcopy(k, &work[iwk2], 1, vf, 1);
        Blas<real>::dcopy(k, &work[iwk3], 1, vl, 1);
    }

    /* dlasdq computes the singular value decomposition (SVD) of a real (upper or lower) bidiagonal
     * matrix with diagonal d and offdiagonal e, accumulating the transformations if desired.
     * Letting B denote the input bidiagonal matrix, the algorithm computes orthogonal matrices Q
     * and P such that B = Q * S * P^T (P^T denotes the transpose of P). The singular values S are
     * overwritten on d.
     * The input matrix U  is changed to U  * Q  if desired.
     * The input matrix Vt is changed to P^T * Vt if desired.
     * The input matrix C  is changed to Q^T * C  if desired.
     * See "Computing  Small Singular Values of Bidiagonal Matrices With Guaranteed High Relative
     * Accuracy," by J. Demmel and W. Kahan, LAPACK Working Note #3, for a detailed description of
     * the algorithm.
     * Parameters: uplo: On entry, uplo specifies whether the input bidiagonal matrix is upper or
     *                             lower bidiagonal, and whether it is square or not.
     *                   uplo=='U' or 'u': B is upper bidiagonal.
     *                   uplo=='L' or 'l': B is lower bidiagonal.
     *             sqre: ==0: then the input matrix is n-by-n.
     *                   ==1: then the input matrix is n-by-(n+1) if UPLU=='U' and (n+1)-by-n if
     *                        UPLU=='L'.
     *                   The bidiagonal matrix has n = nl + nr + 1 rows and
     *                   m = n + sqre >= n columns.
     *             n: On entry, n specifies the number of rows and columns in the matrix.
     *                          n must be at least 0.
     *             ncvt: On entry, ncvt specifies the number of columns of the matrix Vt.
     *                             ncvt must be at least 0.
     *             nru: On entry, nru specifies the number of rows of the matrix U.
     *                            nru must be at least 0.
     *             ncc: On entry, ncc specifies the number of columns of the matrix C.
     *                            ncc must be at least 0.
     *             d: an array, dimension (n)
     *                On entry, d contains the diagonal entries of the bidiagonal matrix whose SVD
     *                          is desired.
     *                On normal exit, d contains the singular values in ascending order.
     *             e: an array. dimension is (n-1) if sqre==0 and n if sqre==1.
     *                On entry, the entries of e contain the offdiagonal entries of the bidiagonal
     *                          matrix whose SVD is desired.
     *                On normal exit, e will contain 0. If the algorithm does not converge, d and e
     *                          will contain the diagonal and superdiagonal entries of a bidiagonal
     *                          matrix orthogonally equivalent to the one given as input.
     *             Vt: an array, dimension (ldvt, ncvt)
     *                 On entry, contains a matrix which on exit has been premultiplied by P^T,
     *                           dimension n-by-ncvt if sqre==0 and (n+1)-by-ncvt if sqre==1
     *                           (not referenced if ncvt=0).
     *             ldvt: On entry, ldvt specifies the leading dimension of Vt as declared in the
     *                             calling (sub) program. ldvt must be at least 1.
     *                             If ncvt is nonzero ldvt must also be at least n.
     *             U: an array, dimension (ldu, n)
     *                On entry, contains a  matrix which on exit has been postmultiplied by Q,
     *                          dimension nru-by-n if sqre==0 and nru-by-(n+1) if sqre==1
     *                          (not referenced if nru=0).
     *             ldu: On entry, ldu specifies the leading dimension of U as declared in the
     *                            calling (sub) program. ldu must be at least max(1, nru).
     *             C: an array, dimension (ldc, ncc)
     *                On entry, contains an n-by-ncc matrix which on exit has been premultiplied by
     *                          Q^T dimension n-by-ncc if sqre==0 and (n+1)-by-ncc if sqre==1
     *                          (not referenced if ncc=0).
     *             ldc: On entry, ldc  specifies the leading dimension of C as declared in the
     *                            calling (sub) program. ldc must be at least 1.
     *                            If ncc is nonzero, ldc must also be at least n.
     *             work: an array, dimension (4*n)
     *                   Workspace. Only referenced if one of ncvt, nru, or ncc is nonzero,
     *                   and if n is at least 2.
     *             info: On exit, a value of 0 indicates a successful exit.
     *                   If info < 0, argument number -info is illegal.
     *                   If info > 0, the algorithm did not converge, and info specifies how many
     *                                superdiagonals did not converge.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2016
     * Contributors:
     * Ming Gu and Huan Ren, Computer Science Division, University of California at Berkeley, USA*/
    static void dlasdq(char const* uplo, int sqre, int n, int ncvt, int nru, int ncc, real* d,
                       real* e, real* Vt, int ldvt, real* U, int ldu, real* C, int ldc, real* work,
                       int& info)
    {
        // Test the input parameters.
        info = 0;
        int iuplo = 0;
        if (std::toupper(uplo[0])=='U')
        {
            iuplo = 1;
        }
        if (std::toupper(uplo[0])=='L')
        {
            iuplo = 2;
        }
        if (iuplo==0)
        {
            info = -1;
        }
        else if (sqre<0 || sqre>1)
        {
            info = -2;
        }
        else if (n<0)
        {
            info = -3;
        }
        else if (ncvt<0)
        {
            info = -4;
        }
        else if (nru<0)
        {
            info = -5;
        }
        else if (ncc<0)
        {
            info = -6;
        }
        else if ((ncvt==0 && ldvt<1) || (ncvt>0 && (ldvt<1 || ldvt<n)))
        {
            info = -10;
        }
        else if (ldu<1 || ldu<nru)
        {
            info = -12;
        }
        else if ((ncc==0 && ldc<1) || (ncc>0 && (ldc<1 || ldc<n)))
        {
           info = -14;
        }
        if (info!=0)
        {
            xerbla("DLASDQ", -info);
            return;
        }
        if (n==0)
        {
            return;
        }
        // rotate is true if any singular vectors desired, false otherwise
        bool rotate = (ncvt>0 || nru>0 || ncc>0);
        int np1 = n + 1;
        // If matrix non-square upper bidiagonal, rotate to be lower bidiagonal.
        // The rotations are on the right.
        int i;
        real cs, r, sn;
        if (iuplo==1 && sqre==1)
        {
            for (i=0; i<n-1; i++)
            {
                dlartg(d[i], e[i], cs, sn, r);
                d[i] = r;
                e[i] = sn*d[i+1];
                d[i+1] = cs*d[i+1];
                if (rotate)
                {
                    work[i] = cs;
                    work[n+i] = sn;
                }
            }
            dlartg(d[n-1], e[n-1], cs, sn, r);
            d[n-1] = r;
            e[n-1] = ZERO;
            if (rotate)
            {
                work[n-1] = cs;
                work[n+n-1] = sn;
            }
            iuplo = 2;
            sqre = 0;
            // Update singular vectors if desired.
            if (ncvt>0)
            {
                dlasr("L", "V", "F", np1, ncvt, work, &work[n], Vt, ldvt);
            }
        }
        // If matrix lower bidiagonal, rotate to be upper bidiagonal by applying Givens rotations
        // on the left.
        if (iuplo==2)
        {
            for (i=0; i<n-1; i++)
            {
                dlartg(d[i], e[i], cs, sn, r);
                d[i] = r;
                e[i] = sn*d[i+1];
                d[i+1] = cs*d[i+1];
                if (rotate)
                {
                    work[i] = cs;
                    work[n+i] = sn;
                }
            }
            // If matrix (n+1)-by-n lower bidiagonal, one additional rotation is needed.
            if (sqre==1)
            {
                dlartg(d[n-1], e[n-1], cs, sn, r);
                d[n-1] = r;
                if (rotate)
                {
                    work[n-1] = cs;
                    work[n+n-1] = sn;
                }
            }
            // Update singular vectors if desired.
            if (nru>0)
            {
                if (sqre==0)
                {
                    dlasr("R", "V", "F", nru, n, work, &work[n], U, ldu);
                }
                else
                {
                    dlasr("R", "V", "F", nru, np1, work, &work[n], U, ldu);
                }
            }
            if (ncc>0)
            {
                if (sqre==0)
                {
                   dlasr("L", "V", "F", n, ncc, work, &work[n], C, ldc);
                }
                else
                {
                   dlasr("L", "V", "F", np1, ncc, work, &work[n], C, ldc);
                }
            }
        }
        // Call dbdsqr to compute the SVD of the reduced real n-by-n upper bidiagonal matrix.
        dbdsqr("U", n, ncvt, nru, ncc, d, e, Vt, ldvt, U, ldu, C, ldc, work, info);
        // Sort the singular values into ascending order (insertion sort on singular values, but
        // only one transposition per singular vector)
        int isub, j;
        real smin;
        for (i=0; i<n; i++)
        {
            // Scan for smallest d[i].
            isub = i;
            smin = d[i];
            for (j=i+1; j<n; j++)
            {
                if (d[j]<smin)
                {
                   isub = j;
                   smin = d[j];
                }
            }
            if (isub!=i)
            {
                // Swap singular values and vectors.
                d[isub] = d[i];
                d[i] = smin;
                if (ncvt>0)
                {
                    Blas<real>::dswap(ncvt, &Vt[isub], ldvt, &Vt[i], ldvt);
                }
                if (nru>0)
                {
                    Blas<real>::dswap(nru, &U[ldu*isub], 1, &U[ldu*i], 1);
                }
                if (ncc>0)
                {
                    Blas<real>::dswap(ncc, &C[isub], ldc, &C[i], ldc);
                }
            }
        }
    }

    /* dlasdt creates a tree of subproblems for bidiagonal divide and sconquer.
     * Parameters: n: On entry, the number of diagonal elements of the bidiagonal matrix.
     *             lvl: On exit, the number of levels on the computation tree.
     *             nd: On exit, the number of nodes on the tree.
     *             inode: an integer array, dimension (n)
     *                    On exit, centers of subproblems.
     *                    NOTE: zero-based indices!
     *             ndiml: an integer array, dimension (n)
     *                    On exit, row dimensions of left children.
     *             ndimr: an integer array, dimension (n)
     *                    On exit, row dimensions of right children.
     *             msub: On entry, the maximum row dimension each subproblem at the bottom of the
     *                   tree can be of.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016
     * Contributors:
     *     Ming Gu and Huan Ren, Computer Science Division,
     *     University of California at Berkeley, USA                                             */
    static void dlasdt(int n, int& lvl, int& nd, int* inode, int* ndiml, int* ndimr, int msub)
    {
        // Find the number of levels on the tree.
        lvl = int(std::log(real(std::max(1, n))/real(msub+1)) / std::log(TWO)) + 1;
        int i = n / 2;
        inode[0] = i;
        ndiml[0] = i;
        ndimr[0] = n - i - 1;
        int il = -1;
        int ir = 0;
        int llst = 1;
        int ncrnt, nlvl;
        for (nlvl=1; nlvl<=lvl-1; nlvl++)
        {
            // Constructing the tree at (nlvl+1)-st level.
            // The number of nodes created on this level is llst*2.
            for (i=0; i<llst; i++)
            {
                il += 2;
                ir += 2;
                ncrnt = llst + i - 1;
                ndiml[il] = ndiml[ncrnt] / 2;
                ndimr[il] = ndiml[ncrnt] - ndiml[il] - 1;
                inode[il] = inode[ncrnt] - ndimr[il] - 1;
                ndiml[ir] = ndimr[ncrnt] / 2;
                ndimr[ir] = ndimr[ncrnt] - ndiml[ir] - 1;
                inode[ir] = inode[ncrnt] + ndiml[ir] + 1;
            }
            llst *= 2;
        }
        nd = llst*2 - 1;
    }

    /* dlaset initializes an m-by-n matrix A to beta on the diagonal and alpha on the offdiagonals.
     * Parameters: uplo: Specifies the part of the matrix A to be set.
     *                   ='U': Upper triangular part is set; the strictly lower triangular part of
     *                         A is not changed.
     *                   ='L': Lower triangular part is set; the strictly upper triangular part of
     *                         A is not changed. Otherwise:  All of the matrix A is set.
     *             m: The number of rows of the matrix A. m>=0.
     *             n: The number of columns of the matrix A. n>=0.
     *             alpha: The constant to which the offdiagonal elements are to be set.
     *             beta: The constant to which the diagonal elements are to be set.
     *             A: an array, dimension (lda,n)
     *                On exit, the leading m-by-n submatrix of A is set as follows:
     *                if uplo = 'U', A(i,j) = alpha, 1<=i<=j-1, 1<=j<=n,
     *                if uplo = 'L', A(i,j) = alpha, j+1<=i<=m, 1<=j<=n,
     *                otherwise,     A(i,j) = alpha, 1<=i<=m, 1<=j<=n, i.ne.j,
     *                and, for all uplo, A(i,i) = beta, 1<=i<=min(m,n).
     *             lda: The leading dimension of the array A. lda>=max(1,m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlaset(char const* uplo, int m, int n, real alpha, real beta, real* A, int lda)
    {
        int i, j, ldaj;
        if (toupper(uplo[0])=='U')
        {
            // Set the strictly upper triangular or trapezoidal part of the array to alpha.
            for (j=1; j<n; j++)
            {
                ldaj = lda*j;
                for (i=0; i<j && i<m; i++)
                {
                    A[i+ldaj] = alpha;
                }
            }
        }
        else if (toupper(uplo[0])=='L')
        {
            // Set the strictly lower triangular or trapezoidal part of the array to alpha.
            for (j=0; j<m && j<n; j++)
            {
                ldaj = lda*j;
                for (i=j+1; i<m; i++)
                {
                    A[i+ldaj] = alpha;
                }
            }
        }
        else
        {
            // Set the leading m-by-n submatrix to alpha.
            for (j=0; j<n; j++)
            {
                ldaj = lda*j;
                for (i=0; i<m; i++)
                {
                    A[i+ldaj] = alpha;
                }
            }
        }
        // Set the first min(m,n) diagonal elements to beta.
        for (i=0; i<m && i<n; i++)
        {
            A[i+lda*i] = beta;
        }
    }

    /* dlasq1 computes the singular values of a real n-by-n bidiagonal matrix with diagonal d and
     * off-diagonal e. The singular values are computed to high relative accuracy, in the absence
     * of denormalization, underflow and overflow. The algorithm was first presented in
     *     "Accurate singular values and differential qd algorithms" by K. V. Fernando and
     *     B. N. Parlett, Numer. Math., Vol-67, No. 2, pp. 191-230, 1994,
     * and the present implementation is described in
     *     "An implementation of the dqds Algorithm (Positive Case)", LAPACK Working Note.
     * Parameters: n: The number of rows and columns in the matrix. n>=0.
     *             d: an array, dimension (n)
     *                On entry, d contains the diagonal elements of the bidiagonal matrix whose SVD
     *                is desired.
     *                On normal exit, d contains the singular values in decreasing order.
     *             e: an array, dimension (n)
     *                On entry, elements e[0:n-2] contain the off-diagonal elements of the
     *                bidiagonal matrix whose SVD is desired.
     *                On exit, e is overwritten.
     *             work: an array, dimension (4*n)
     *             info: =0: successful exit
     *                   <0: if info==-i, the i-th argument had an illegal value
     *                  > 0: the algorithm failed
     *                       =1, a split was marked by a positive value in e
     *                       =2, current block of Z not diagonalized after 100*n iterations
     *                           (in inner while loop). On exit d and e represent a matrix with the
     *                           same singular values which the calling subroutine could use to
     *                           finish the computation, or even feed back into dlasq1
     *                       =3, termination criterion of outer while loop not met
     *                           (program created more than n unreduced blocks)
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlasq1(int n, real* d, real* e, real* work, int& info)
    {
        int i, iinfo;
        real eps, scale, safmin, sigmn, sigmx, temp;
        info = 0;
        if (n<0)
        {
            info = -1;
            xerbla("DLASQ1", -info);
            return;
        }
        else if (n==0)
        {
            return;
        }
        else if (n==1)
        {
            d[0] = fabs(d[0]);
            return;
        }
        else if (n==2)
        {
            dlas2(d[0], e[0], d[1], sigmn, sigmx);
            d[0] = sigmx;
            d[1] = sigmn;
            return;
        }
        //Estimate the largest singular value.
        sigmx = ZERO;
        for (i=0; i<n-1; i++)
        {
            d[i] = fabs(d[i]);
            temp = fabs(e[i]);
            if (temp>sigmx)
            {
                sigmx = temp;
            }
        }
        d[n-1] = fabs(d[n-1]);
        // Early return if SIGMX is zero (matrix is already diagonal).
        if (sigmx==ZERO)
        {
            dlasrt("D", n, d, iinfo);
            return;
        }
        for (i=0; i<n; i++)
        {
            if (d[i]>sigmx)
            {
                sigmx = d[i];
            }
        }
        // Copy d and e into work (in the Z format) and scale (squaring the input data makes
        // scaling by a power of the radix pointless).
        eps = dlamch("Precision");
        safmin = dlamch("Safe minimum");
        scale = std::sqrt(eps/safmin);
        Blas<real>::dcopy(n, d, 1, &work[0], 2);
        Blas<real>::dcopy(n-1, e, 1, &work[1], 2);
        dlascl("G", 0, 0, sigmx, scale, 2*n-1, 1, work, 2*n-1, iinfo);
        // Compute the q's and e's.
        for (i=0; i<2*n-1; i++)
        {
            work[i] *= work[i];
        }
        work[2*n-1] = ZERO;
        dlasq2(n, work, info);
        if (info==0)
        {
            for (i=0; i<n; i++)
            {
                d[i] = std::sqrt(work[i]);
            }
            dlascl("G", 0, 0, scale, sigmx, n, 1, d, n, iinfo);
        }
        else if (info==2)
        {
            // Maximum number of iterations exceeded. Move data from work into d and e so the
            // calling subroutine can try to finish
            for (i=0; i<n; i++)
            {
                d[i] = std::sqrt(work[2*i]);
                e[i] = std::sqrt(work[2*i+1]);
            }
            dlascl("G", 0, 0, scale, sigmx, n, 1, d, n, iinfo);
            dlascl("G", 0, 0, scale, sigmx, n, 1, e, n, iinfo);
        }
    }

    /* dlasq2 computes all the eigenvalues of the symmetric positive definite tridiagonal matrix
     * associated with the qd array Z to high relative accuracy are computed to high relative
     * accuracy, in the absence of denormalization, underflow and overflow.
     * To see the relation of Z to the tridiagonal matrix, let L be a unit lower bidiagonal matrix
     * with subdiagonals Z[1,3,5,,..] and let U be an upper bidiagonal matrix with 1's above and
     * diagonal Z[0,2,4,,..]. The tridiagonal is L*U or, if you prefer, the symmetric tridiagonal
     * to which it is similar.
     * Note: dlasq2 defines a logical variable, IEEE, which is true on machines which follow
     * ieee-754 floating-point standard in their handling of infinities and NaNs, and false
     * otherwise. This variable is passed to dlasq3.
     * Parameters: n: The number of rows and columns in the matrix. n>=0.
     *             Z: an array, dimension (4*n)
     *                On entry Z holds the qd array. On exit, entries 1 to n hold the eigenvalues
     *                in decreasing order, Z[2*n] holds the trace, and Z[2*n+1] holds the sum of
     *                the eigenvalues. If n>2, then Z[2*n+2] holds the iteration count, Z[2*n+3]
     *                holds NDIVS/NIN^2, and Z[2*n+4] holds the percentage of shifts that failed.
     *             info: =0: successful exit
     *                   <0: if the i-th argument is a scalar and had an illegal value, then
     *                       info==-i, if the i-th argument is an array and the j-entry had an
     *                       illegal value, then info = -(i*100+j)
     *                   >0: the algorithm failed
     *                       =1, a split was marked by a positive value in Z
     *                       =2, current block of Z not diagonalized after 100*n iterations
     *                           (in inner while loop). On exit Z holds a qd array with the same
     *                           eigenvalues as the given Z.
     *                       =3, termination criterion of outer while loop not met
     *                           (program created more than n unreduced blocks)
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Further Details:
     *     Local Variables: i0:n0 defines a current unreduced segment of Z. The shifts are
     *     accumulated in SIGMA. Iteration count is in iter. Ping-pong is controlled by PP
     *     (alternates between 0 and 1).                                                         */
    static void dlasq2(int n, real* Z, int& info)
    {
        const real CBIAS = real(1.50);
        bool ieee, loopbreak;
        int i0, i1, i4, iinfo, ipn4, iter, iwhila, iwhilb, k, kmin, n0, n1, nbig, ndiv, nfail, pp,
            splt, ttype;
        real d, dee, deemin, desig, dmin, dmin1, dmin2, dn, dn1, dn2, e, emax, emin, eps, g,
             oldemn, qmax, qmin, s, safmin, sigma, tt, tau, temp, tol, tol2, trace, zmax, tempe,
             tempq;
        // Test the input arguments. (in case dlasq2 is not called by dlasq1)
        info = 0;
        eps = dlamch("Precision");
        safmin = dlamch("Safe minimum");
        tol = eps * HNDRD;
        tol2 = tol * tol;
        if (n<0)
        {
            info = -1;
            xerbla("DLASQ2", 1);
            return;
        }
        else if (n==0)
        {
            return;
        }
        else if (n==1)
        {
            // 1-by-1 case.
            if (Z[0]<ZERO)
            {
                info = -201;
                xerbla("DLASQ2", 2);
            }
            return;
        }
        else if (n==2)
        {
            // 2-by-2 case.
            if (Z[1]<ZERO || Z[2]<ZERO)
            {
                info = -2;
                xerbla("DLASQ2", 2);
                return;
            }
            else if (Z[2]>Z[0])
            {
                d = Z[2];
                Z[2] = Z[0];
                Z[0] = d;
            }
            Z[4] = Z[0] + Z[1] + Z[2];
            if (Z[1]>Z[2]*tol2)
            {
                tt = HALF * ((Z[0]-Z[2])+Z[1]);
                s = Z[2] * (Z[1]/tt);
                if (s<=tt)
                {
                    s = Z[2] * (Z[1]/(tt*(ONE+std::sqrt(ONE+s/tt))));
                }
                else
                {
                    s = Z[2] * (Z[1]/(tt+std::sqrt(tt)*std::sqrt(tt+s)));
                }
                tt = Z[0] + (s+Z[1]);
                Z[2] = Z[2] * (Z[0]/tt);
                Z[0] = tt;
            }
            Z[1] = Z[2];
            Z[5] = Z[1] + Z[0];
            return;
        }
        // Check for negative data and compute sums of q's and e's.
        Z[2*n-1] = ZERO;
        emin = Z[1];
        qmax = ZERO;
        zmax = ZERO;
        d = ZERO;
        e = ZERO;
        for (k=0; k<2*(n-1); k+=2)
        {
            if (Z[k]<ZERO)
            {
                info = -(201+k);
                xerbla("DLASQ2", 2);
                return;
            }
            else if (Z[k+1]<ZERO)
            {
                info = -(201+k+1);
                xerbla("DLASQ2", 2);
                return;
            }
            d += Z[k];
            e += Z[k+1];
            qmax = std::max(qmax, Z[k]);
            emin = std::min(emin, Z[k+1]);
            zmax = std::max(std::max(qmax, zmax), Z[k+1]);
        }
        if (Z[2*n-2]<ZERO)
        {
            info = -(200+2*n-1);
            xerbla("DLASQ2", 2);
            return;
        }
        d += Z[2*n-2];
        qmax = std::max(qmax, Z[2*n-2]);
        zmax = std::max(qmax, zmax);
        // Check for diagonality.
        if (e==ZERO)
        {
            for (k=1; k<n; k++)
            {
                Z[k] = Z[2*n-2];
            }
            dlasrt("D", n, Z, iinfo);
            Z[2*n-2] = d;
            return;
        }
        trace = d + e;
        // Check for zero data.
        if (trace==ZERO)
        {
            Z[2*n-2] = ZERO;
            return;
        }
        // Check whether the machine is IEEE conformable.
        ieee = ilaenv(10, "DLASQ2", "N", 1, 2, 3, 4)==1
            && ilaenv(11, "DLASQ2", "N", 1, 2, 3, 4)==1;
        // Rearrange data for locality: Z=(q1,qq1,e1,ee1,q2,qq2,e2,ee2,...).
        for (k=2*n-1; k>=1; k-=2)
        {
            Z[2*k+1] = ZERO;
            Z[2*k]   = Z[k];
            Z[2*k-1] = ZERO;
            Z[2*k-2] = Z[k-1];
        }
        i0 = 0;
        n0 = n-1;
        // Reverse the qd-array, if warranted.
        if (CBIAS*Z[4*i0]<Z[4*n0])
        {
            ipn4 = 4*(i0+n0+1);
            for (i4=4*i0; i4<=2*(i0+n0-1); i4+=4)
            {
                temp = Z[i4];
                Z[i4] = Z[ipn4-i4-4];
                Z[ipn4-i4-4] = temp;
                temp = Z[i4+2];
                Z[i4+2] = Z[ipn4-i4-6];
                Z[ipn4-i4-6] = temp;
            }
        }
        // Initial split checking via dqd and Li's test.
        pp = 0;
        for (k=0; k<2; k++)
        {
            d = Z[4*n0+pp];
            for (i4=4*(n0-1)+pp; i4>=4*i0+pp; i4-=4)
            {
                if (Z[i4+2]<=tol2*d)
                {
                    Z[i4+2] = -ZERO;
                    d = Z[i4];
                }
                else
                {
                    d = Z[i4] * (d/(d+Z[i4+2]));
                }
            }
            // dqd maps Z to ZZ plus Li's test.
            emin = Z[4*i0+4+pp];
            d = Z[4*i0+pp];
            for (i4=4*i0+pp; i4<=4*(n0-1)+pp; i4+=4)
            {
                Z[i4-2*pp+1] = d + Z[i4+2];
                if (Z[i4+2]<=tol2*d)
                {
                    Z[i4+2] = -ZERO;
                    Z[i4-2*pp+1] = d;
                    Z[i4-2*pp+3] = ZERO;
                    d = Z[i4+4];
                }
                else if (safmin*Z[i4+4]<Z[i4-2*pp+1] && safmin*Z[i4-2*pp+1]<Z[i4+4])
                {
                    temp = Z[i4+4] / Z[i4-2*pp+1];
                    Z[i4-2*pp+3] = Z[i4+2] * temp;
                    d *= temp;
                }
                else
                {
                    Z[i4-2*pp+3] = Z[i4+4] * (Z[i4+2]/Z[i4-2*pp+1]);
                    d = Z[i4+4] * (d/Z[i4-2*pp+1]);
                }
                if (emin>Z[i4-2*pp+3])
                {
                    emin = Z[i4-2*pp+3];
                }
            }
            Z[4*n0-pp+1] = d;
            // Now find qmax.
            qmax = Z[4*i0-pp+1];
            for (i4=4*i0-pp+2; i4<=4*n0-pp-2; i4+=4)
            {
                if (qmax<Z[i4+3])
                {
                    qmax = Z[i4+3];
                }
            }
            // Prepare for the next iteration on k.
            pp = 1 - pp;
        }
        // Initialise variables to pass to DLASQ3.
        ttype = 0;
        dmin1 = ZERO;
        dmin2 = ZERO;
        dn    = ZERO;
        dn1   = ZERO;
        dn2   = ZERO;
        g     = ZERO;
        tau   = ZERO;
        iter = 2;
        nfail = 0;
        ndiv = 2*(n0-i0);
        for (iwhila=0; iwhila<=n; iwhila++)
        {
            if (n0<0)
            {
                break;
            }
            // While array unfinished do
            // e[n0] holds the value of SIGMA when submatrix in i0:n0 splits from the rest of the
            // array, but is negated.
            desig = ZERO;
            if (n0==n-1)
            {
                sigma = ZERO;
            }
            else
            {
                sigma = -Z[4*n0+2];
            }
            if (sigma<ZERO)
            {
                info = 1;
                return;
            }
            // Find last unreduced submatrix's top index i0, find QMAX and EMIN.
            // Find Gershgorin-type bound if Q's much greater than e's.
            emax = ZERO;
            if (n0>i0)
            {
                emin = std::fabs(Z[4*n0-2]);
            }
            else
            {
                emin = ZERO;
            }
            qmin = Z[4*n0];
            qmax = qmin;
            loopbreak = false;
            for (i4=4*n0; i4>=4; i4-=4)
            {
                if (Z[i4-2]<=ZERO)
                {
                    loopbreak = true;
                    break;
                }
                if (qmin>=FOUR*emax)
                {
                    qmin = std::min(qmin, Z[i4]);
                    emax = std::max(emax, Z[i4-2]);
                }
                temp = Z[i4-4] + Z[i4-2];
                qmax = std::max(qmax, temp);
                emin = std::min(emin, Z[i4-2]);
            }
            if (!loopbreak)
            {
                i4 = 0;
            }
            i0 = i4 / 4;
            pp = 0;
            if (n0-i0>1)
            {
                dee = Z[4*i0];
                deemin = dee;
                kmin = i0;
                for (i4=4*i0+1; i4<=4*n0-3; i4+=4)
                {
                    dee = Z[i4+3] * (dee/(dee+Z[i4+1]));
                    if (dee<=deemin)
                    {
                        deemin = dee;
                        kmin = (i4+3)/4;
                    }
                }
                if ((kmin-i0)*2<n0-kmin && deemin<=HALF*Z[4*n0])
                {
                    ipn4 = 4*(i0+n0+1);
                    pp = 2;
                    for (i4=4*i0; i4<=2*(i0+n0-1); i4+=4)
                    {
                        temp         = Z[i4];
                        Z[i4]        = Z[ipn4-i4-4];
                        Z[ipn4-i4-4] = temp;
                        temp         = Z[i4+1];
                        Z[i4+1]      = Z[ipn4-i4-3];
                        Z[ipn4-i4-3] = temp;
                        temp         = Z[i4+2];
                        Z[i4+2]      = Z[ipn4-i4-6];
                        Z[ipn4-i4-6] = temp;
                        temp         = Z[i4+3];
                        Z[i4+3]      = Z[ipn4-i4-5];
                        Z[ipn4-i4-5] = temp;
                    }
                }
            }
            // Put -(initial shift) into DMIN.
            dmin = TWO*std::sqrt(qmin)*std::sqrt(emax) - qmin;
            if (dmin>ZERO)
            {
                dmin = -ZERO;
            }
            // Now i0:n0 is unreduced.
            // PP = 0 for ping, PP = 1 for pong.
            // PP = 2 indicates that flipping was applied to the Z array and and that the tests for
            // deflation upon entry in dlasq3 should not be performed.
            nbig = 100*(n0-i0+1);
            loopbreak = false;
            for (iwhilb=0; iwhilb<nbig; iwhilb++)
            {
                if (i0>n0)
                {
                    loopbreak = true;
                    break;
                }
                // While submatrix unfinished take a good dqds step.
                dlasq3(i0, n0, Z, pp, dmin, sigma, desig, qmax, nfail, iter, ndiv, ieee, ttype,
                       dmin1, dmin2, dn, dn1, dn2, g, tau);
                pp = 1 - pp;
                // When EMIN is very small check for splits.
                if (pp==0 && n0-i0>=3)
                {
                    if (Z[4*n0+3]<=tol2*qmax || Z[4*n0+2]<=tol2*sigma)
                    {
                        splt = i0 - 1;
                        qmax = Z[4*i0];
                        emin = Z[4*i0+2];
                        oldemn = Z[4*i0+3];
                        for (i4=4*i0; i4<=4*(n0-3); i4+=4)
                        {
                            if (Z[i4+3]<=tol2*Z[i4] || Z[i4+2]<=tol2*sigma)
                            {
                                Z[i4+2] = -sigma;
                                splt = i4 / 4;
                                qmax = ZERO;
                                emin = Z[i4+6];
                                oldemn = Z[i4+7];
                            }
                            else
                            {
                                qmax   = std::max(qmax,   Z[i4+4]);
                                emin   = std::min(emin,   Z[i4+2]);
                                oldemn = std::min(oldemn, Z[i4+3]);
                            }
                        }
                        Z[4*n0+2] = emin;
                        Z[4*n0+3] = oldemn;
                        i0 = splt + 1;
                    }
                }
            }
            if (loopbreak)
            {
                continue;
            }
            info = 2;
            // Maximum number of iterations exceeded, restore the shift SIGMA and place the new
            // d's and e's in a qd array. This might need to be done for several blocks.
            i1 = i0;
            n1 = n0;
            while (true)
            {
                tempq = Z[4*i0];
                Z[4*i0] += sigma;
                for (k=i0+1; k<=n0; k++)
                {
                    tempe = Z[4*k-2];
                    Z[4*k-2] *= tempq / Z[4*k-4];
                    tempq = Z[4*k];
                    Z[4*k] += sigma + tempe - Z[4*k-2];
                }
                // Prepare to do this on the previous block if there is one
                if (i1<=0)
                {
                    break;
                }
                n1 = i1 - 1;
                while ((i1>=1) && (Z[4*i1-2]>=ZERO))
                {
                    i1--;
                }
                sigma = -Z[4*n1+2];
            }
            for (k=0; k<n; k++)
            {
                Z[2*k] = Z[4*k];
                // Only the block 0..n0-1 is unfinished. The rest of the e's must be essentially
                // zero, although sometimes other data has been stored in them.
                if (k<n0)
                {
                    Z[2*k+1] = Z[4*k+2];
                }
                else
                {
                    Z[2*k+1] = 0;
                }
            }
            return;
        }
        if (n0>=0)
        {
            info = 3;
            return;
        }
        // Move q's to the front.
        for (k=1; k<n; k++)
        {
            Z[k] = Z[4*k];
        }
        // Sort and compute sum of eigenvalues.
        dlasrt("D", n, Z, iinfo);
        e = ZERO;
        for (k=n-1; k>=0; k--)
        {
            e += Z[k];
        }
        // Store trace, sum(eigenvalues) and information on performance.
        Z[2*n] = trace;
        Z[2*n+1] = e;
        Z[2*n+2] = real(iter);
        Z[2*n+3] = real(ndiv) / real(n*n);
        Z[2*n+4] = HNDRD * nfail / real(iter);
        return;
    }

    /* dlasq3 checks for deflation, computes a shift (tau) and calls dqds. In case of failure it
     * changes shifts, and tries again until output is positive.
     * Parameters: i0: First index. (note: zero-based index!)
     *             n0: Last index. (note: zero-based index!)
     *             Z: an array, dimension (4*n0+4)
     *                Z holds the qd array.
     *             pp: pp=0 for ping, pp=1 for pong.
     *                 pp=2 indicates that flipping was applied to the Z array and that the initial
     *                 tests for deflation should not be performed.
     *             dmin: Minimum value of d.
     *             sigma: Sum of shifts used in current segment.
     *             desig: Lower order part of sigma
     *             qmax: Maximum value of q.
     *             nfail: Increment nfail by 1 each time the shift was too big.
     *             iter: Increment iter by 1 for each iteration.
     *             ndiv: Increment ndiv by 1 for each division.
     *             ieee: Flag for IEEE or non IEEE arithmetic (passed to dlasq5).
     *             ttype: Shift type.
     *             dmin1,
     *             dmin2,
     *             dn,
     *             dn1,
     *             dn2,
     *             g,
     *             tau: These are passed as arguments in order to save their values between calls
     *                  to dlasq3.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2016                                                                            */
    static void dlasq3(int i0, int& n0, real* Z, int& pp, real& dmin, real& sigma, real& desig,
                       real qmax, int& nfail, int& iter, int& ndiv, bool ieee, int& ttype,
                       real& dmin1, real& dmin2, real& dn, real& dn1, real& dn2, real& g,
                       real& tau)
    {
        const real CBIAS = real(1.50);
        int n0in = n0;
        real eps = dlamch("Precision");
        real tol = eps*HNDRD;
        real tol2 = tol*tol;
        int ipn4, j4, nn;
        real s, tt, temp;
        // Check for deflation.
        while (true)
        {
            if (n0<i0)
            {
                return;
            }
            if (n0==i0)
            {
                Z[4*n0] = Z[4*n0+pp] + sigma;
                n0--;
                continue;
            }
            nn = 4*n0 + pp + 4;
            if (n0!=(i0+1))
            {
                // Check whether e[n0-1] is negligible, 1 eigenvalue.
                if (Z[nn-6]<=tol2*(sigma+Z[nn-4]) || Z[nn-2*pp-5]<=tol2*Z[nn-8])
                {
                    Z[4*n0] = Z[4*n0+pp] + sigma;
                    n0--;
                    continue;
                }
                // Check  whether e[n0-2] is negligible, 2 eigenvalues.
                if (Z[nn-10]>tol2*sigma && Z[nn-2*pp-9]>tol2*Z[nn-12])
                {
                    break;
                }
            }
            if (Z[nn-4]>Z[nn-8])
            {
                s = Z[nn-4];
                Z[nn-4] = Z[nn-8];
                Z[nn-8] = s;
            }
            tt = HALF * ((Z[nn-8]-Z[nn-4])+Z[nn-6]);
            if (Z[nn-6]>Z[nn-4]*tol2 && tt!=ZERO)
            {
                s = Z[nn-4] * (Z[nn-6]/tt);
                if (s<=tt)
                {
                    s = Z[nn-4] * (Z[nn-6] / (tt*(ONE+std::sqrt(ONE+s/tt))));
                }
                else
                {
                    s = Z[nn-4] * (Z[nn-6] / (tt+std::sqrt(tt)*std::sqrt(tt+s)));
                }
                tt = Z[nn-8] + (s+Z[nn-6]);
                Z[nn-4] = Z[nn-4] * (Z[nn-8]/tt);
                Z[nn-8] = tt;
            }
            Z[4*n0-4] = Z[nn-8] + sigma;
            Z[4*n0]   = Z[nn-4] + sigma;
            n0 -= 2;
        }
        if (pp==2)
        {
            pp = 0;
        }
        // Reverse the qd-array, if warranted.
        if (dmin<=ZERO || n0<n0in)
        {
            if (CBIAS*Z[4*i0+pp]<Z[4*n0+pp])
            {
                ipn4 = 4*(i0+n0)+7;
                for (j4=4*i0+3; j4<2*(i0+n0+1); j4+=4)
                {
                    temp = Z[j4-3];
                    Z[j4-3] = Z[ipn4-j4-4];
                    Z[ipn4-j4-4] = temp;
                    temp = Z[j4-2];
                    Z[j4-2] = Z[ipn4-j4-3];
                    Z[ipn4-j4-3] = temp;
                    temp = Z[j4-1];
                    Z[j4-1] = Z[ipn4-j4-6];
                    Z[ipn4-j4-6] = temp;
                    temp = Z[j4];
                    Z[j4] = Z[ipn4-j4-5];
                    Z[ipn4-j4-5] = temp;
                }
                if (n0-i0<=4)
                {
                    Z[4*n0+pp+2] = Z[4*i0+pp+2];
                    Z[4*n0-pp+3] = Z[4*i0-pp+3];
                }
                dmin2 = std::min(dmin2, Z[4*n0+pp+2]);
                Z[4*n0+pp+2] = std::min(std::min(Z[4*n0+pp+2], Z[4*i0+pp+2]), Z[4*i0+pp+6]);
                Z[4*n0-pp+3] = std::min(std::min(Z[4*n0-pp+3], Z[4*i0-pp+3]), Z[4*i0-pp+7]);
                qmax = std::max(std::max(qmax, Z[4*i0+pp]), Z[4*i0+pp+4]);
                dmin = -ZERO;
            }
        }
        // Choose a shift.
        dlasq4(i0, n0, Z, pp, n0in, dmin, dmin1, dmin2, dn, dn1, dn2, tau, ttype, g);
        // Call dqds until DMIN>0.
        while (true)// 70 CONTINUE
        {
            dlasq5(i0, n0, Z, pp, tau, sigma, dmin, dmin1, dmin2, dn, dn1, dn2, ieee, eps);
            ndiv += n0 - i0 + 2;
            iter++;
            // Check status.
            if (dmin>=ZERO && dmin1>=ZERO)
            {
                // Success.
                break;
            }
            else if (dmin<ZERO && dmin1>ZERO && Z[4*n0-pp-1]<tol*(sigma+dn1)
                     && fabs(dn)<tol*sigma)
            {
                // Convergence hidden by negative DN.
                Z[4*n0-pp+1] = ZERO;
                dmin = ZERO;
                break;
            }
            else if (dmin<ZERO)
            {
                // tau too big. Select new tau and try again.
                nfail++;
                if (ttype<-22)
                {
                    //Failed twice. Play it safe.
                    tau = ZERO;
                }
                else if (dmin1>ZERO)
                {
                    // Late failure. Gives excellent shift.
                    tau = (tau+dmin)*(ONE-TWO*eps);
                    ttype -= 11;
                }
                else
                {
                    // Early failure. Divide by 4.
                    tau *= QURTR;
                    ttype -= 12;
                }
                continue;
            }
            else if (std::isnan(dmin))
            {
                // NaN.
                if (tau!=ZERO)
                {
                    tau = ZERO;
                    continue;
                }
            }//else {// Possible underflow. Play it safe.}
            // Risk of underflow.
            dlasq6(i0, n0, Z, pp, dmin, dmin1, dmin2, dn, dn1, dn2);
            ndiv += n0 - i0 + 2;
            iter++;
            tau = ZERO;
        }
        if (tau<sigma)
        {
            desig += tau;
            tt = sigma + desig;
            desig -= tt - sigma;
        }
        else
        {
            tt = sigma + tau;
            desig += sigma - (tt-tau);
        }
        sigma = tt;
    }

    /* dlasq4 computes an approximation dn2 to the smallest eigenvalue using values of d from the
     * previous transform.
     * Parameters: i0: First index. (note: zero-based index!)
     *             n0: Last index. (note: zero-based index!)
     *             Z: an array, dimension (4*(n0+1))
     *                Z holds the qd array.
     *             pp: pp=0 for ping, pp=1 for pong.
     *             n0in: The value of n0 at start of EIGTEST. (note: zero-based index!)
     *             dmin: Minimum value of d.
     *             dmin1: Minimum value of d, excluding D[n0].
     *             dmin2: Minimum value of d, excluding D[n0] and D[n0-1].
     *             dn: d(N)
     *             dn1: d(N-1)
     *             dn2: d(N-2)
     *             dn2: This is the shift.
     *             dn2: Shift type.
     *             g: g is passed as an argument in order to save its value between calls to
     *                dlasq4.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2016
     * Further Details:
     *     CNST1 = 9/16                                                                          */
    static void dlasq4(int i0, int n0, real const* Z, int pp, int n0in, real dmin, real dmin1,
                       real dmin2, real dn, real dn1, real dn2, real& tau, int& ttype, real& g)
    {
        const real CNST1 = real(0.563);
        const real CNST2 = real(1.01);
        const real CNST3 = real(1.05);
        const real THIRD = real(0.333);
        // A negative dmin forces the shift to take that absolute value
        // dn2 records the type of shift.
        if (dmin<=ZERO)
        {
            tau = -dmin;
            ttype = -1;
            return;
        }
        int nn = 4*(n0+1) + pp;
        int i4, np;
        real a2, b1, b2, gam, gap1, gap2, s=ZERO, temp;
        if (n0in==n0)
        {
            // No eigenvalues deflated.
            if (dmin==dn || dmin==dn1)
            {
                b1 = std::sqrt(Z[nn-4])*std::sqrt(Z[nn-6]);
                b2 = std::sqrt(Z[nn-8])*std::sqrt(Z[nn-10]);
                a2 = Z[nn-8] + Z[nn-6];
                // Cases 2 and 3.
                if (dmin==dn && dmin1==dn1)
                {
                    gap2 = dmin2 - a2 - dmin2*QURTR;
                    if (gap2>ZERO && gap2>b2)
                    {
                        gap1 = a2 - dn - (b2/gap2)*b2;
                    }
                    else
                    {
                        gap1 = a2 - dn - (b1+b2);
                    }
                    if (gap1>ZERO && gap1>b1)
                    {
                        s = dn-(b1/gap1)*b1;
                        temp = HALF*dmin;
                        if (temp>s)
                        {
                            s = temp;
                        }
                        ttype = -2;
                    }
                    else
                    {
                        s = ZERO;
                        if (dn>b1)
                        {
                            s = dn - b1;
                        }
                        if (a2>(b1+b2))
                        {
                            temp = a2 - (b1+b2);
                            if (temp<s)
                            {
                                s = temp;
                            }
                        }
                        temp = THIRD * dmin;
                        if (s<temp)
                        {
                            s = temp;
                        }
                        ttype = -3;
                    }
                }
                else
                {
                    // Case 4.
                    ttype = -4;
                    s = QURTR*dmin;
                    if (dmin==dn)
                    {
                        gam = dn;
                        a2 = ZERO;
                        if (Z[nn-6]>Z[nn-8])
                        {
                            return;
                        }
                        b2 = Z[nn-6] / Z[nn-8];
                        np = nn - 10;
                    }
                    else
                    {
                        np = nn - 2*pp - 1;
                        gam = dn1;
                        if (Z[np-4]>Z[np-2])
                        {
                            return;
                        }
                        a2 = Z[np-4] / Z[np-2];
                        if (Z[nn-10]>Z[nn-12])
                        {
                            return;
                        }
                        b2 = Z[nn-10] / Z[nn-12];
                        np = nn - 14;
                    }
                    // Approximate contribution to norm squared from i<nn-2.
                    a2 += b2;
                    for (i4=np; i4>=(4*i0+2+pp); i4-=4)
                    {
                        if (b2==ZERO)
                        {
                            break;
                        }
                        b1 = b2;
                        if (Z[i4]>Z[i4-2])
                        {
                            return;
                        }
                        b2 *= Z[i4] / Z[i4-2];
                        a2 += b2;
                        if (HNDRD*(b2>b1?b2:b1)<a2 || CNST1<a2)
                        {
                            break;
                        }
                    }
                    a2 *= CNST3;
                    // Rayleigh quotient residual bound.
                    if (a2<CNST1)
                    {
                        s = gam * (ONE-std::sqrt(a2)) / (ONE+a2);
                    }
                }
            }
            else if (dmin==dn2)
            {
                // Case 5.
                ttype = -5;
                s = QURTR*dmin;
                // Compute contribution to norm squared from i>=nn-2.
                np = nn - 2*pp - 1;
                b1 = Z[np-2];
                b2 = Z[np-6];
                gam = dn2;
                if (Z[np-8]>b2 || Z[np-4]>b1)
                {
                    return;
                }
                a2 = (Z[np-8]/b2) * (ONE+Z[np-4]/b1);
                // Approximate contribution to norm squared from i<nn-3.
                if (n0-i0>2)
                {
                    b2 = Z[nn-14] / Z[nn-16];
                    a2 += b2;
                    for (i4=nn-18; i4>=(4*i0+2+pp); i4-=4)
                    {
                        if (b2==ZERO)
                        {
                            break;
                        }
                        b1 = b2;
                        if (Z[i4]>Z[i4-2])
                        {
                            return;
                        }
                        b2 *= Z[i4] / Z[i4-2];
                        a2 += b2;
                        if (HNDRD*(b2>b1?b2:b1)<a2 || CNST1<a2)
                        {
                            break;
                        }
                    }
                    a2 *= CNST3;
                }
                if (a2<CNST1)
                {
                    s = gam * (ONE-std::sqrt(a2)) / (ONE+a2);
                }
            }
            else
            {
                // Case 6, no information to guide us.
                if (ttype==-6)
                {
                    g += THIRD*(ONE-g);
                }
                else if (ttype==-18)
                {
                    g = QURTR*THIRD;
                }
                else
                {
                    g = QURTR;
                }
                s = g*dmin;
                ttype = -6;
            }
        }
        else if (n0in==(n0+1))
        {
            // One eigenvalue just deflated. Use dmin1, dn1 for dmin and dn.
            if (dmin1==dn1 && dmin2==dn2)
            {
                // Cases 7 and 8.
                ttype = -7;
                s = THIRD*dmin1;
                if (Z[nn-6]>Z[nn-8])
                {
                    return;
                }
                b1 = Z[nn-6] / Z[nn-8];
                b2 = b1;
                if (b2!=ZERO)
                {
                    for (i4=(4*n0-6+pp); i4>=(4*i0+2+pp); i4-=4)
                    {
                        a2 = b1;
                        if (Z[i4]>Z[i4-2])
                        {
                            return;
                        }
                        b1 *= Z[i4] / Z[i4-2];
                        b2 += b1;
                        if (HNDRD*(b1>a2?b1:a2)<b2)
                        {
                            break;
                        }
                    }
                }
                b2 = std::sqrt(CNST3*b2);
                a2 = dmin1 / (ONE+b2*b2);
                gap2 = HALF*dmin2 - a2;
                if (gap2>ZERO && gap2>b2*a2)
                {
                    temp = a2 * (ONE-CNST2*a2*(b2/gap2)*b2);
                    if (s<temp)
                    {
                        s = temp;
                    }
                }
                else
                {
                    temp = a2 * (ONE-CNST2*b2);
                    if (s<temp)
                    {
                        s = temp;
                    }
                    ttype = -8;
                }
            }
            else
            {
                // Case 9.
                s = QURTR * dmin1;
                if (dmin1==dn1)
                {
                    s = HALF*dmin1;
                }
                ttype = -9;
            }
        }
        else if (n0in==(n0+2))
        {
            // Two eigenvalues deflated. Use dmin2, dn2 for dmin and dn.
            // Cases 10 and 11.
            if (dmin2==dn2 && TWO*Z[nn-6]<Z[nn-8])
            {
                ttype = -10;
                s = THIRD*dmin2;
                if (Z[nn-6]>Z[nn-8])
                {
                    return;
                }
                b1 = Z[nn-6] / Z[nn-8];
                b2 = b1;
                if (b2!=ZERO)
                {
                    for (i4=(4*n0-6+pp); i4>=(4*i0+2+pp); i4-=4)
                    {
                        if (Z[i4]>Z[i4-2])
                        {
                            return;
                        }
                        b1 *= Z[i4] / Z[i4-2];
                        b2 += b1;
                        if (HNDRD*b1<b2)
                        {
                            break;
                        }
                    }
                }
                b2 = std::sqrt(CNST3*b2);
                a2 = dmin2 / (ONE+b2*b2);
                gap2 = Z[nn-8] + Z[nn-10] - std::sqrt(Z[nn-12])*std::sqrt(Z[nn-10]) - a2;
                if (gap2>ZERO && gap2>b2*a2)
                {
                    temp = a2 * (ONE-CNST2*a2*(b2/gap2)*b2);
                    if (s<temp)
                    {
                        s = temp;
                    }
                }
                else
                {
                    temp = a2 * (ONE-CNST2*b2);
                    if (s<temp)
                    {
                        s = temp;
                    }
                }
            }
            else
            {
                s = QURTR * dmin2;
                ttype = -11;
            }
        }
        else if (n0in>(n0+2))
        {
            // Case 12, more than two eigenvalues deflated. No information.
            s = ZERO;
            ttype = -12;
        }
        tau = s;
    }

    /* dlasq5 computes one dqds transform in ping-pong form, one version for ieee machines another
     * for non ieee machines.
     * Parameters: i0: First index. (note: zero-based index!)
     *             n0: Last index. (note: zero-based index!)
     *             Z: an array, dimension (4*N)
     *                holds the qd array. EMIN is stored in Z(4*(n0+1)) to avoid an extra argument.
     *             pp: pp=0 for ping, pp=1 for pong.
     *             pp: This is the shift.
     *             sigma: This is the accumulated shift up to this step.
     *             dmin: Minimum value of d.
     *             dmin1: Minimum value of d, excluding d[n0].
     *             dmin2: Minimum value of d, excluding d[n0] and d[n0-1].
     *             dn: d[n0], the last value of d.
     *             dnm1: d[n0-1].
     *             dnm2: d[n0-2].
     *             ieee: Flag for ieee or non ieee arithmetic.
     *             eps: This is the value of epsilon used.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2017                                                                            */
    static void dlasq5(int i0, int n0, real* Z, int pp, real tau, real sigma, real& dmin,
                       real& dmin1, real& dmin2, real& dn, real& dnm1, real& dnm2, bool ieee,
                       real eps)
    {
        if ((n0-i0-1)<=0)
        {
            return;
        }
        int j4, j4p2;
        real d, emin, temp, dthresh;
        dthresh = eps*(sigma+tau);
        if (tau<dthresh*HALF)
        {
            tau = ZERO;
        }
        if (tau!=ZERO)
        {
            j4 = 4*i0 + pp;
            emin = Z[j4+4];
            d = Z[j4] - tau;
            dmin = d;
            dmin1 = -Z[j4];
            if (ieee)
            {
                // Code for IEEE arithmetic.
                if (pp==0)
                {
                    for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
                    {
                        Z[j4-2] = d + Z[j4-1];
                        temp = Z[j4+1] / Z[j4-2];
                        d = d*temp - tau;
                        dmin = std::min(dmin, d);
                        Z[j4] = Z[j4-1]*temp;
                        emin = std::min(Z[j4], emin);
                    }
                }
                else
                {
                    for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
                    {
                        Z[j4-3] = d + Z[j4];
                        temp = Z[j4+2] / Z[j4-3];
                        d = d*temp - tau;
                        dmin = std::min(dmin, d);
                        Z[j4-1] = Z[j4]*temp;
                        emin = std::min(Z[j4-1], emin);
                    }
                }
                // Unroll last two steps.
                dnm2 = d;
                dmin2 = dmin;
                j4 = 4*n0 - 5 - pp;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm2 + Z[j4p2];
                Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                dnm1 = Z[j4p2+2]*(dnm2/Z[j4-2]) - tau;
                dmin = std::min(dmin, dnm1);
                dmin1 = dmin;
                j4 += 4;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm1 + Z[j4p2];
                Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                dn = Z[j4p2+2]*(dnm1/Z[j4-2]) - tau;
                dmin = std::min(dmin, dn);
            }
            else
            {
                // Code for non IEEE arithmetic.
                if (pp==0)
                {
                    for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
                    {
                        Z[j4-2] = d + Z[j4-1];
                        if (d<ZERO)
                        {
                            return;
                        }
                        else
                        {
                            Z[j4] = Z[j4+1] * (Z[j4-1]/Z[j4-2]);
                            d = Z[j4+1]*(d/Z[j4-2]) - tau;
                        }
                        dmin = std::min(dmin, d);
                        emin = std::min(emin, Z[j4]);
                    }
                }
                else
                {
                    for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
                    {
                        Z[j4-3] = d + Z[j4];
                        if (d<ZERO)
                        {
                            return;
                        }
                        else
                        {
                            Z[j4-1] = Z[j4+2] * (Z[j4]/Z[j4-3]);
                            d = Z[j4+2]*(d/Z[j4-3]) - tau;
                        }
                        dmin = std::min(dmin, d);
                        emin = std::min(emin, Z[j4-1]);
                    }
                }
                // Unroll last two steps.
                dnm2 = d;
                dmin2 = dmin;
                j4 = 4*n0 - 5 - pp;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm2 + Z[j4p2];
                if (dnm2<ZERO)
                {
                    return;
                }
                else
                {
                    Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                    dnm1 = Z[j4p2+2]*(dnm2/Z[j4-2]) - tau;
                }
                dmin = std::min(dmin, dnm1);
                dmin1 = dmin;
                j4 += 4;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm1 + Z[j4p2];
                if (dnm1<ZERO)
                {
                    return;
                }
                else
                {
                    Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                    dn = Z[j4p2+2]*(dnm1/Z[j4-2]) - tau;
                }
                dmin = std::min(dmin, dn);
            }
        }
        else
        {
            // This is the version that sets d's to zero if they are small enough
            j4 = 4*i0 + pp;
            emin = Z[j4+4];
            d = Z[j4] - tau;
            dmin = d;
            dmin1 = -Z[j4];
            if (ieee)
            {
                // Code for IEEE arithmetic.
                if (pp==0)
                {
                    for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
                    {
                        Z[j4-2] = d + Z[j4-1];
                        temp = Z[j4+1] / Z[j4-2];
                        d = d*temp - tau;
                        if (d<dthresh)
                        {
                            d = ZERO;
                        }
                        dmin = std::min(dmin, d);
                        Z[j4] = Z[j4-1]*temp;
                        emin = std::min(Z[j4], emin);
                    }
                }
                else
                {
                    for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
                    {
                        Z[j4-3] = d + Z[j4];
                        temp = Z[j4+2] / Z[j4-3];
                        d = d*temp - tau;
                        if (d<dthresh)
                        {
                            d = ZERO;
                        }
                        dmin = std::min(dmin, d);
                        Z[j4-1] = Z[j4]*temp;
                        emin = std::min(Z[j4-1], emin);
                    }
                }
                // Unroll last two steps.
                dnm2 = d;
                dmin2 = dmin;
                j4 = 4*n0 - 5 - pp;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm2 + Z[j4p2];
                Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                dnm1 = Z[j4p2+2]*(dnm2/Z[j4-2]) - tau;
                dmin = std::min(dmin, dnm1);
                dmin1 = dmin;
                j4 += 4;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm1 + Z[j4p2];
                Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                dn = Z[j4p2+2]*(dnm1/Z[j4-2]) - tau;
                dmin = std::min(dmin, dn);
            }
            else
            {
                // Code for non IEEE arithmetic.
                if (pp==0)
                {
                    for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
                    {
                        Z[j4-2] = d + Z[j4-1];
                        if (d<ZERO)
                        {
                            return;
                        }
                        else
                        {
                            Z[j4] = Z[j4+1] * (Z[j4-1]/Z[j4-2]);
                            d = Z[j4+1]*(d/Z[j4-2]) - tau;
                        }
                        if (d<dthresh)
                        {
                            d = ZERO;
                        }
                        dmin = std::min(dmin, d);
                        emin = std::min(Z[j4], emin);
                    }
                }
                else
                {
                    for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
                    {
                        Z[j4-3] = d + Z[j4];
                        if (d<ZERO)
                        {
                            return;
                        }
                        else
                        {
                            Z[j4-1] = Z[j4+2] * (Z[j4]/Z[j4-3]);
                            d = Z[j4+2]*(d/Z[j4-3]) - tau;
                        }
                        if (d<dthresh)
                        {
                            d = ZERO;
                        }
                        dmin = std::min(dmin, d);
                        emin = std::min(Z[j4-1], emin);
                    }
                }
                // Unroll last two steps.
                dnm2 = d;
                dmin2 = dmin;
                j4 = 4*n0 - 5 - pp;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm2 + Z[j4p2];
                if (dnm2<ZERO)
                {
                    return;
                }
                else
                {
                    Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                    dnm1 = Z[j4p2+2]*(dnm2/Z[j4-2]) - tau;
                }
                dmin = std::min(dmin, dnm1);
                dmin1 = dmin;
                j4 += 4;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm1 + Z[j4p2];
                if (dnm1<ZERO)
                {
                    return;
                }
                else
                {
                    Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                    dn = Z[j4p2+2]*(dnm1/Z[j4-2]) - tau;
                }
                dmin = std::min(dmin, dn);
            }
        }
        Z[j4+2] = dn;
        Z[4*n0-pp+3] = emin;
    }

    /* dlasq6 computes one dqd (shift equal to zero) transform in ping-pong form, with protection
     * against underflow and overflow.
     * Parameters: i0: First index. (note: zero-based index!)
     *             n0: Last index. (note: zero-based index!)
     *             Z: an array, dimension (4*N)
     *                Z holds the qd array. EMIN is stored in Z[4*n0+3] to avoid an extra argument.
     *             pp: pp=0 for ping, pp=1 for pong.
     *             dmin: Minimum value of d.
     *             dmin1: Minimum value of d, excluding D[n0].
     *             dmin2: Minimum value of d, excluding D[n0] and D[n0-1].
     *             dn: d[n0], the last value of d.
     *             dnm1: d[n0-1].
     *             dnm2: d[n0-2].
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlasq6(int i0, int n0, real* Z, int pp, real& dmin, real& dmin1, real& dmin2,
                       real& dn, real& dnm1, real& dnm2)
    {
        if ((n0-i0-1)<=0)
        {
            return;
        }
        real safmin = dlamch("Safe minimum");
        int j4 = 4*i0 + pp + 1;
        real emin = Z[j4+4];
        real d = Z[j4];
        dmin = d;
        int j4p2;
        real temp;
        if (pp==0)
        {
            for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
            {
                Z[j4-2] = d + Z[j4-1];
                if (Z[j4-2]==ZERO)
                {
                    Z[j4] = ZERO;
                    d = Z[j4+1];
                    dmin = d;
                    emin = ZERO;
                }
                else if (safmin*Z[j4+1]<Z[j4-2] && safmin*Z[j4-2]<Z[j4+1])
                {
                    temp = Z[j4+1] / Z[j4-2];
                    Z[j4] = Z[j4-1] * temp;
                    d *= temp;
                }
                else
                {
                    Z[j4] = Z[j4+1] * (Z[j4-1]/Z[j4-2]);
                    d = Z[j4+1] * (d/Z[j4-2]);
                }
                dmin = std::min(dmin, d);
                emin = std::min(emin, Z[j4]);
            }
        }
        else
        {
            for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
            {
                Z[j4-3] = d+Z[j4];
                if (Z[j4-3]==ZERO)
                {
                    Z[j4-1] = ZERO;
                    d = Z[j4+2];
                    dmin = d;
                    emin = ZERO;
                }
                else if (safmin*Z[j4+2]<Z[j4-3] && safmin*Z[j4-3]<Z[j4+2])
                {
                    temp = Z[j4+2] / Z[j4-3];
                    Z[j4-1] = Z[j4] * temp;
                    d *= temp;
                }
                else
                {
                    Z[j4-1] = Z[j4+2] * (Z[j4]/Z[j4-3]);
                    d = Z[j4+2] * (d/Z[j4-3]);
                }
                dmin = std::min(dmin, d);
                emin = std::min(emin, Z[j4-1]);
            }
        }
        // Unroll last two steps.
        dnm2 = d;
        dmin2 = dmin;
        j4 = 4*n0 - 5 - pp;
        j4p2 = j4 + 2*pp - 1;
        Z[j4-2] = dnm2 + Z[j4p2];
        if (Z[j4-2]==ZERO)
        {
            Z[j4] = ZERO;
            dnm1 = Z[j4p2+2];
            dmin = dnm1;
            emin = ZERO;
        }
        else if (safmin*Z[j4p2+2]<Z[j4-2] && safmin*Z[j4-2]<Z[j4p2+2])
        {
            temp = Z[j4p2+2] / Z[j4-2];
            Z[j4] = Z[j4p2] * temp;
            dnm1 = dnm2 * temp;
        }
        else
        {
            Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
            dnm1 = Z[j4p2+2] * (dnm2/Z[j4-2]);
        }
        dmin = std::min(dmin, dnm1);
        dmin1 = dmin;
        j4 += 4;
        j4p2 = j4 + 2*pp - 1;
        Z[j4-2] = dnm1 + Z[j4p2];
        if (Z[j4-2]==ZERO)
        {
            Z[j4] = ZERO;
            dn = Z[j4p2+2];
            dmin = dn;
            emin = ZERO;
        }
        else if (safmin*Z[j4p2+2]<Z[j4-2] && safmin*Z[j4-2]<Z[j4p2+2])
        {
            temp = Z[j4p2+2] / Z[j4-2];
            Z[j4] = Z[j4p2] * temp;
            dn = dnm1 * temp;
        }
        else
        {
            Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
            dn = Z[j4p2+2] * (dnm1/Z[j4-2]);
        }
        dmin = std::min(dmin, dn);
        Z[j4+2] = dn;
        Z[4*n0-pp+3] = emin;
    }

    /* dlasr applies a sequence of plane rotations to a real matrix A, from either the left or the
     * right.
     * When side=='L', the transformation takes the form
     *     A := P*A
     * and when side=='R', the transformation takes the form
     *     A := A*P^T
     * where P is an orthogonal matrix consisting of a sequence of z plane rotations, with z = m
     * when side=='L' and z = n when side=='R', and P^T is the transpose of P.
     * When direct=='F' (Forward sequence), then
     *     P = P(z-1) * ... * P(2) * P(1)
     * and when direct=='B' (Backward sequence), then
     *     P = P(1) * P(2) * ... * P(z-1)
     * where P(k) is a plane rotation matrix defined by the 2-by-2 rotation
     *     R(k) = (  c(k)  s(k) )
     *          = ( -s(k)  c(k) ).
     * When pivot=='V' (Variable pivot), the rotation is performed for the plane (k,k+1),
     * i.e., P(k) has the form
     *     P(k) = (  1                                            )
     *            (       ...                                     )
     *            (              1                                )
     *            (                   c(k)  s(k)                  )
     *            (                  -s(k)  c(k)                  )
     *            (                                1              )
     *            (                                     ...       )
     *            (                                            1  )
     * where R(k) appears as a rank-2 modification to the identity matrix in rows and columns
     * k and k+1.
     * When pivot=='T' (Top pivot), the rotation is performed for the plane (1,k+1),
     * so P(k) has the form
     *     P(k) = (  c(k)                    s(k)                 )
     *            (         1                                     )
     *            (              ...                              )
     *            (                     1                         )
     *            ( -s(k)                    c(k)                 )
     *            (                                 1             )
     *            (                                      ...      )
     *            (                                             1 )
     * where R(k) appears in rows and columns 1 and k+1.
     * Similarly, when pivot=='B' (Bottom pivot), the rotation is performed for the plane (k,z),
     * giving P(k) the form
     *     P(k) = ( 1                                             )
     *            (      ...                                      )
     *            (             1                                 )
     *            (                  c(k)                    s(k) )
     *            (                         1                     )
     *            (                              ...              )
     *            (                                     1         )
     *            (                 -s(k)                    c(k) )
     * where R(k) appears in rows and columns k and z. The rotations are performed without ever
     * forming P(k) explicitly.
     * Parameters: side: Specifies whether the plane rotation matrix P is applied to A on the left
     *                   or the right.
     *                   =='L': Left, compute A := P*A
     *                   =='R': Right, compute A:= A*P^T
     *             pivot: Specifies the plane for which P(k) is a plane rotation matrix.
     *                    =='V': Variable pivot, the plane (k,k+1)
     *                    =='T': Top pivot, the plane (1,k+1)
     *                    =='B': Bottom pivot, the plane (k,z)
     *             direct: Specifies whether P is a forward or backward sequence of plane
     *                     rotations.
     *                     =='F': Forward, P = P(z-1)*...*P(2)*P(1)
     *                     =='B': Backward, P = P(1)*P(2)*...*P(z-1)
     *             m: The number of rows of the matrix A. If m<=1, an immediate return is effected.
     *             n: The number of columns of the matrix A. If n<=1, an immediate return is
     *                effected.
     *             c: an array, dimension (m-1) if side=='L'
     *                                    (n-1) if side=='R'
     *                The cosines c(k) of the plane rotations.
     *             s: an array, dimension (m-1) if side=='L'
     *                                    (n-1) if side=='R'
     *                The sines s(k) of the plane rotations. The 2-by-2 plane rotation part of the
     *                matrix P(k), R(k), has the form
     *                    R(k) = (  c(k)  s(k) )
     *                           ( -s(k)  c(k) ).
     *             A: an array, dimension (lda,n)
     *                The m-by-n matrix A. On exit, A is overwritten by P*A if side=='R'
     *                                                            or by A*P^T if side=='L'.
     *             lda: The leading dimension of the array A. lda >= max(1,m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlasr(char const* side, char const* pivot, char const* direct, int m, int n,
                      real const* c, real const* s, real* A, int lda)
    {
        int i, info, j, aind1, aind2, aind3, aind4;
        real ctemp, stemp, temp;
        char upside = std::toupper(side[0]);
        char uppivot = std::toupper(pivot[0]);
        char updirect = std::toupper(direct[0]);
        // Test the input parameters
        info = 0;
        if (!(upside=='L' || upside=='R'))
        {
            info = 1;
        }
        else if (!(uppivot=='V' || uppivot=='T' || uppivot=='B'))
        {
            info = 2;
        }
        else if (!(updirect=='F' || updirect=='B'))
        {
            info = 3;
        }
        else if (m<0)
        {
            info = 4;
        }
        else if (n<0)
        {
            info = 5;
        }
        else if (lda<1 || lda<m)
        {
            info = 9;
        }
        if (info!=0)
        {
            xerbla("DLASR ", info);
            return;
        }
        // Quick return if possible
        if (m==0 || n==0)
        {
            return;
        }
        if (upside=='L')
        {
            // Form P*A
            if (uppivot=='V')
            {
                if (updirect=='F')
                {
                    for (j=0; j<m-1; j++)
                    {
                        ctemp = c[j];
                        stemp = s[j];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            for (i=0; i<n; i++)
                            {
                                aind1 = j+lda*i;
                                aind2 = aind1+1;
                                // A[j+1,i] and A[j,i]
                                temp     = A[aind2];
                                A[aind2] = ctemp*temp - stemp*A[aind1];
                                A[aind1] = stemp*temp + ctemp*A[aind1];
                            }
                        }
                    }
                }
                else if (updirect=='B')
                {
                    for (j=m-2; j>=0; j--)
                    {
                        ctemp = c[j];
                        stemp = s[j];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            for (i=0; i<n; i++)
                            {
                                aind1 = j+lda*i;
                                aind2 = aind1+1;
                                // A[j+1,i] and A[j,i]
                                temp     = A[aind2];
                                A[aind2] = ctemp*temp - stemp*A[aind1];
                                A[aind1] = stemp*temp + ctemp*A[aind1];
                            }
                        }
                    }
                }
            }
            else if (uppivot=='T')
            {
                if (updirect=='F')
                {
                    for (j=1; j<m; j++)
                    {
                        ctemp = c[j-1];
                        stemp = s[j-1];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            for (i=0; i<n; i++)
                            {
                                aind1 = lda*i;
                                aind2 = j+aind1;
                                // A[j,i] and A[0,i]
                                temp     = A[aind2];
                                A[aind2] = ctemp*temp - stemp*A[aind1];
                                A[aind1] = stemp*temp + ctemp*A[aind1];
                            }
                        }
                    }
                }
                else if (updirect=='B')
                {
                    for (j=m-1; j>=1; j--)
                    {
                        ctemp = c[j-1];
                        stemp = s[j-1];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            for (i=0; i<n; i++)
                            {
                                aind1 = lda*i;
                                aind2 = j+aind1;
                                // A[j,i] and A[0,i]
                                temp     = A[aind2];
                                A[aind2] = ctemp*temp - stemp*A[aind1];
                                A[aind1] = stemp*temp + ctemp*A[aind1];
                            }
                        }
                    }
                }
            }
            else if (uppivot=='B')
            {
                if (updirect=='F')
                {
                    for (j=0; j<m-1; j++)
                    {
                        ctemp = c[j];
                        stemp = s[j];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            for (i=0; i<n; i++)
                            {
                                aind1 = lda*i;
                                aind2 = m-1+aind1;
                                aind3 = j+aind1;
                                // A[j,i] and A[m-1,i]
                                temp     = A[aind3];
                                A[aind3] = stemp*A[aind2] + ctemp*temp;
                                A[aind2] = ctemp*A[aind2] - stemp*temp;
                            }
                        }
                    }
                }
                else if (updirect=='B')
                {
                    for (j=m-2; j>=0; j--)
                    {
                        ctemp = c[j];
                        stemp = s[j];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            for (i=0; i<n; i++)
                            {
                                aind1 = lda*i;
                                aind2 = m-1+aind1;
                                aind3 = j+aind1;
                                // A[j,i] and A[m-1,i]
                                temp     = A[aind3];
                                A[aind3] = stemp*A[aind2] + ctemp*temp;
                                A[aind2] = ctemp*A[aind2] - stemp*temp;
                            }
                        }
                    }
                }
            }
        }
        else if (upside=='R')
        {
            // Form A*P^T
            if (uppivot=='V')
            {
                if (updirect=='F')
                {
                    for (j=0; j<n-1; j++)
                    {
                        ctemp = c[j];
                        stemp = s[j];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            aind1 = lda*j;
                            for (i=0; i<m; i++)
                            {
                                aind2 = i+aind1;
                                aind3 = aind2+lda;
                                // A[i,j+1] and A[i,j]
                                temp     = A[aind3];
                                A[aind3] = ctemp*temp - stemp*A[aind2];
                                A[aind2] = stemp*temp + ctemp*A[aind2];
                            }
                        }
                    }
                }
                else if (updirect=='B')
                {
                    for (j=n-2; j>=0; j--)
                    {
                        ctemp = c[j];
                        stemp = s[j];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            aind1 = lda*j;
                            for (i=0; i<m; i++)
                            {
                                aind2 = i+aind1;
                                aind3 = aind2+lda;
                                // A[i,j+1] and A[i,j]
                                temp     = A[aind3];
                                A[aind3] = ctemp*temp - stemp*A[aind2];
                                A[aind2] = stemp*temp + ctemp*A[aind2];
                            }
                        }
                    }
                }
            }
            else if (uppivot=='T')
            {
                if (updirect=='F')
                {
                    for (j=1; j<n; j++)
                    {
                        ctemp = c[j-1];
                        stemp = s[j-1];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            aind1 = lda*j;
                            for (i=0; i<m; i++)
                            {
                                aind2 = i+aind1;
                                // A[i,j] and A[i,0]
                                temp     = A[aind2];
                                A[aind2] = ctemp*temp - stemp*A[i];
                                A[i]     = stemp*temp + ctemp*A[i];
                            }
                        }
                    }
                }
                else if (updirect=='B')
                {
                    for (j=n-1; j>=1; j--)
                    {
                        ctemp = c[j-1];
                        stemp = s[j-1];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            aind1 = lda*j;
                            for (i=0; i<m; i++)
                            {
                                aind2 = i+aind1;
                                // A[i,j] and A[i,0]
                                temp     = A[aind2];
                                A[aind2] = ctemp*temp - stemp*A[i];
                                A[i]     = stemp*temp + ctemp*A[i];
                            }
                        }
                    }
                }
            }
            else if (uppivot=='B')
            {
                aind1 = lda*(n-1);
                if (updirect=='F')
                {
                    for (j=0; j<n-1; j++)
                    {
                        ctemp = c[j];
                        stemp = s[j];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            aind2 = lda*j;
                            for (i=0; i<m; i++)
                            {
                                aind3 = i+aind1;
                                aind4 = i+aind2;
                                // A[i,j] and A[i,n-1]
                                temp     = A[aind4];
                                A[aind4] = stemp*A[aind3] + ctemp*temp;
                                A[aind3] = ctemp*A[aind3] - stemp*temp;
                            }
                        }
                    }
                }
                else if (updirect=='B')
                {
                    for (j=n-2; j>=0; j--)
                    {
                        ctemp = c[j];
                        stemp = s[j];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            aind2 = lda*j;
                            for (i=0; i<m; i++)
                            {
                                aind3 = i+aind1;
                                aind4 = i+aind2;
                                // A[i,j] and A[i,n-1]
                                temp     = A[aind4];
                                A[aind4] = stemp*A[aind3] + ctemp*temp;
                                A[aind3] = ctemp*A[aind3] - stemp*temp;
                            }
                        }
                    }
                }
            }
        }
    }

    /* dlasrt sorts the numbers in d in increasing order (if id = 'I') or in decreasing order
     * (if id = 'D').
     * Use Quick Sort, reverting to Insertion sort on arrays of size<=20. Dimension of 'stack'
     * limits n to about 2^32.
     * Parameters: id: =='I': sort d in increasing order;
     *                 =='D': sort d in decreasing order.
     *             n: The length of the array d.
     *             d: an array, dimension (n)
     *                On entry, the array to be sorted.
     *                On exit, d has been sorted into increasing order (d[0] <= ... <= d[n-1]) or
     *                         into decreasing order (d[0] >= ... >= d[n-1]), depending on id.
     *             info: ==0: successful exit
     *                    <0: if info = -i, the i-th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2016                                                                            */
    static void dlasrt(char const* id, int n, real* d, int& info)
    {
        const int SELECT = 20;
        // Test the input parameters.
        info = 0;
        int dir = -1;
        if (toupper(id[0])=='D')
        {
            dir = 0;
        }
        else if (toupper(id[0])=='I')
        {
            dir = 1;
        }
        if (dir==-1)
        {
            info = -1;
        }
        else if (n<0)
        {
            info = -2;
        }
        if (info!=0)
        {
            xerbla("DLASRT", -info);
            return;
        }
        // Quick return if possible
        if (n<=1)
        {
            return;
        }
        int stack[2*32]; // stack[2, 32]
        int endd, i, j, start, stkpnt;
        real d1, d2, d3, dmnmx, tmp;
        stkpnt = 0;
        stack[0] = 0;     //stack[0,0]
        stack[1] = n - 1; //stack[1,0]
        do
        {
            start = stack[2*stkpnt];   //stack[0,stkpnt]
            endd  = stack[1+2*stkpnt]; //stack[1,stkpnt]
            stkpnt--;
            if (endd-start<=SELECT && endd-start>0)
            {
                // Do Insertion sort on D[start:endd]
                if (dir==0)
                {
                    // Sort into decreasing order
                    for (i=start+1; i<=endd; i++)
                    {
                        for (j=i; j>start; j--)
                        {
                            if (d[j]>d[j-1])
                            {
                                dmnmx = d[j];
                                d[j] = d[j-1];
                                d[j-1] = dmnmx;
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
                }
                else
                {
                    // Sort into increasing order
                    for (i=start+1; i<=endd; i++)
                    {
                        for (j=i; j>start; j--)
                        {
                            if (d[j]<d[j-1])
                            {
                                dmnmx = d[j];
                                d[j] = d[j-1];
                                d[j-1] = dmnmx;
                            }
                            else
                            {
                                break;
                            }
                        }
                    }
                }
            }
            else if (endd-start>SELECT)
            {
                // Partition D[start:endd] and stack parts, largest one first
                // Choose partition entry as median of 3
                d1 = d[start];
                d2 = d[endd];
                i = (start+endd)/2;
                d3 = d[i];
                if (d1<d2)
                {
                    if (d3<d1)
                    {
                        dmnmx = d1;
                    }
                    else if (d3<d2)
                    {
                        dmnmx = d3;
                    }
                    else
                    {
                        dmnmx = d2;
                    }
                }
                else
                {
                    if (d3<d2)
                    {
                        dmnmx = d2;
                    }
                    else if (d3<d1)
                    {
                        dmnmx = d3;
                    }
                    else
                    {
                        dmnmx = d1;
                    }
                }
                if (dir==0)
                {
                    // Sort into decreasing order
                    i = start - 1;
                    j = endd + 1;
                    while (true)
                    {
                        do
                        {
                            j--;
                        } while (d[j]<dmnmx);
                        do
                        {
                            i++;
                        } while (d[i]>dmnmx);
                        if (i>=j)
                        {
                            break;
                        }
                        tmp = d[i];
                        d[i] = d[j];
                        d[j] = tmp;
                    }
                    if (j-start>endd-j-1)
                    {
                        stkpnt++;
                        stack[2*stkpnt] = start; //stack[0,stkpnt]
                        stack[1+2*stkpnt] = j;   //stack[1,stkpnt]
                        stkpnt++;
                        stack[2*stkpnt] = j + 1;  //stack[0,stkpnt]
                        stack[1+2*stkpnt] = endd; //stack[1,stkpnt]
                    }
                    else
                    {
                        stkpnt++;
                        stack[2*stkpnt] = j + 1;  //stack[0,stkpnt]
                        stack[1+2*stkpnt] = endd; //stack[1,stkpnt]
                        stkpnt++;
                        stack[2*stkpnt] = start; //stack[0,stkpnt]
                        stack[1+2*stkpnt] = j;   //stack[1,stkpnt]
                    }
                }
                else
                {
                    // Sort into increasing order
                    i = start - 1;
                    j = endd + 1;
                    while (true)
                    {
                        do
                        {
                            j--;
                        } while (d[j]>dmnmx);
                        do
                        {
                            i++;
                        } while (d[i]<dmnmx);
                        if (i<j)
                        {
                            break;
                        }
                        tmp = d[i];
                        d[i] = d[j];
                        d[j] = tmp;
                    }
                    if (j-start>endd-j-1)
                    {
                        stkpnt++;
                        stack[2*stkpnt] = start; //stack[0,stkpnt]
                        stack[1+2*stkpnt] = j;   //stack[1,stkpnt]
                        stkpnt++;
                        stack[2*stkpnt] = j + 1;  //stack[0,stkpnt]
                        stack[1+2*stkpnt] = endd; //stack[1,stkpnt]
                    }
                    else
                    {
                        stkpnt++;
                        stack[2*stkpnt] = j + 1;  //stack[0,stkpnt]
                        stack[1+2*stkpnt] = endd; //stack[1,stkpnt]
                        stkpnt++;
                        stack[2*stkpnt] = start; //stack[0,stkpnt]
                        stack[1+2*stkpnt] = j;   //stack[1,stkpnt]
                    }
                }
            }
        } while (stkpnt>=0);
    }

    /* dlassq updates a sum of squares represented in scaled form.
     * returns the values scl and smsq such that
     * (scl^2)*smsq = x(0)^2 + ... + x(n-1)^2 + (scale^2)*sumsq,
     * where x(i) = x[i*incx]. The value of sumsq is assumed to be non-negative and scl
     * returns the value
     * scl = max(scale, abs(x(i))).
     * scl and smsq are overwritten on scale and sumsq respectively.
     * The routine makes only one pass through the vector x.
     * Parameters: n: The number of elements to be used from the vector x.
     *             x: an array, dimension (n)
     *                The vector for which a scaled sum of squares is computed.
     *                x(i) = x[i*incx], 0 <= i < n.
     *             incx: The increment between successive values of the vector x.
     *                   incx > 0.
     *             scale: On entry, the value scale in the equation above.
     *                    On exit, scale is overwritten with scl, the scaling factor for
     *                    the sum of squares.
     *             sumsq: On entry, the value sumsq in the equation above.
     *                    On exit, sumsq is overwritten with smsq, the basic sum of squares from
     *                    which scl has been factored out.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016                                                                       */
    static void dlassq(int n, real const* x, int incx, real& scale, real& sumsq)
    {
        if (n>0)
        {
            real absxi, temp;
            for (int ix=0; ix<=(n-1)*incx; ix++)
            {
                absxi = fabs(x[ix]);
                if (absxi>ZERO || std::isnan(absxi))
                {
                    temp = scale/absxi;
                    if (scale<absxi)
                    {
                        sumsq = 1 + sumsq*temp*temp;
                        scale = absxi;
                    }
                    else
                    {
                        sumsq += temp*temp;
                    }
                }
            }
        }
    }

    /* dlasv2 computes the singular value decomposition of a 2-by-2 triangular matrix
     *     [ f  g ]
     *     [ 0  h ].
     * On return, abs(ssmax) is the larger singular value, abs(ssmin) is the smaller singular
     * value, and (csl,snl) and (csr,snr) are the left and right singular vectors for abs(ssmax),
     * giving the decomposition
     *     [ csl snl] [ f  g ] [csr -snr] = [ssmax  0  ]
     *     [-snl csl] [ 0  h ] [snr  csr]   [ 0   ssmin].
     * Parameters: f: The [0,0] element of the 2-by-2 matrix.
     *             g: The [0,1] element of the 2-by-2 matrix.
     *             h: The [1,1] element of the 2-by-2 matrix.
     *             ssmin: abs(ssmin) is the smaller singular value.
     *             ssmax: abs(ssmax) is the larger singular value.
     *             snl,
     *             csl: The vector (csl, snl) is a unit left singular vector for the singular value
     *                  abs(ssmax).
     *             snr,
     *             csr: The vector (csr, snr) is a unit right singular vector for the singular
     *                  value abs(ssmax).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016
     * Further Details:
     *   Any input parameter may be aliased with any output parameter.
     *   Barring over/underflow and assuming a guard digit in subtraction, all output quantities
     *     are correct to within a few units in the last place (ulps).
     *   In IEEE arithmetic, the code works correctly if one matrix element is infinite.
     *   Overflow will not occur unless the largest singular value itself overflows or is within
     *     a few ulps of overflow. (On machines with partial overflow, like the Cray, overflow may
     *     occur if the largest singular value is within a factor of 2 of overflow.)
     *   Underflow is harmless if underflow is gradual. Otherwise, results may correspond to a
     *     matrix modified by perturbations of size near the underflow threshold.                */
    static void dlasv2(real f, real g, real h, real& ssmin, real& ssmax, real& snr, real& csr,
                       real& snl, real& csl)
    {
        bool gasmal, swap;
        int pmax;
        real a, clt, crt, d, fa, ft, ga, gt, ha, ht, l, m, mm, r, s, slt, srt, t, temp, tsign, tt;
        ft = f;
        fa = std::fabs(ft);
        ht = h;
        ha = std::fabs(h);
        // pmax points to the maximum absolute element of matrix
        //  pmax = 0 if f largest in absolute values
        //  pmax = 1 if G largest in absolute values
        //  pmax = 2 if H largest in absolute values
        pmax = 0;
        swap = (ha>fa);
        if (swap)
        {
            pmax = 2;
            temp = ft;
            ft = ht;
            ht = temp;
            temp = fa;
            fa = ha;
            ha = temp;
            // Now fa >= ha
        }
        gt = g;
        ga = std::fabs(gt);
        if (ga==ZERO)
        {
            // Diagonal matrix
            ssmin = ha;
            ssmax = fa;
            clt = ONE;
            crt = ONE;
            slt = ZERO;
            srt = ZERO;
        }
        else
        {
            gasmal = true;
            if (ga>fa)
            {
                pmax = 1;
                if ((fa/ga) < dlamch("EPS"))
                {
                    // Case of very large ga
                    gasmal = false;
                    ssmax = ga;
                    if (ha>ONE)
                    {
                        ssmin = fa / (ga/ha);
                    }
                    else
                    {
                        ssmin = (fa/ga) * ha;
                    }
                    clt = ONE;
                    slt = ht / gt;
                    srt = ONE;
                    crt = ft / gt;
                }
            }
            if (gasmal)
            {
                // Normal case
                d = fa - ha;
                if (d==fa)
                {
                    //Copes with infinite f or h
                    l = ONE;
                }
                else
                {
                    l = d / fa;
                }
                // Note that 0 <= l <= 1
                m = gt / ft;
                // Note that abs(m) <= 1/macheps
                t = TWO - l;
                // Note that tt >= 1
                mm = m*m;
                tt = t*t;
                s = std::sqrt(tt+mm);
                // Note that 1 <= s <= 1 + 1/macheps
                if (l==ZERO)
                {
                    r = std::fabs(m);
                }
                else
                {
                    r = std::sqrt(l*l+mm);
                }
                // Note that 0 <= r <= 1 + 1/macheps
                a = HALF * (s+r);
                // Note that 1 <= a <= 1 + abs(m)
                ssmin = ha / a;
                ssmax = fa * a;
                if (mm==ZERO)
                {
                    // Note that m is very tiny
                    if (l==ZERO)
                    {
                        t = std::copysign(TWO, ft) * real((ZERO<=gt)-(gt<ZERO));
                    }
                    else
                    {
                        t = gt/std::copysign(d, ft) + m/t;
                    }
                }
                else
                {
                    t = (m/(s+t) + m/(r+l)) * (ONE + a);
                }
                l = std::sqrt(t*t+real(4.0));
                crt = TWO / l;
                srt = t / l;
                clt = (crt+srt*m) / a;
                slt = (ht/ft) * srt / a;
            }
        }
        if (swap)
        {
            csl = srt;
            snl = crt;
            csr = slt;
            snr = clt;
        }
        else
        {
            csl = clt;
            snl = slt;
            csr = crt;
            snr = srt;
        }
        // Correct signs of SSMAX and SSMIN
        if (pmax==0)
        {
            tsign = real((ZERO<=csr)-(csr<ZERO)) * real((ZERO<=csl)-(csl<ZERO))
                  * real((ZERO<=f)-(f<ZERO));
        }
        if (pmax==1)
        {
            tsign = real((ZERO<=snr)-(snr<ZERO)) * real((ZERO<=csl)-(csl<ZERO))
                  * real((ZERO<=g)-(g<ZERO));
        }
        if (pmax==2)
        {
            tsign = real((ZERO<=snr)-(snr<ZERO)) * real((ZERO<=snl)-(snl<ZERO))
                  * real((ZERO<=h)-(h<ZERO));
        }
        ssmax = std::copysign(ssmax, tsign);
        ssmin = std::copysign(ssmin, tsign*real((ZERO<=f)-(f<ZERO))*real((ZERO<=h)-(h<ZERO)));
    }

    /* dorg2r generates an m by n real matrix Q with orthonormal columns,
     * which is defined as the first n columns of a product of k elementary reflectors of order m
     *     Q = H(1) H(2) . ..H(k)
     * as returned by dgeqrf.
     * Parameters: m: The number of rows of the matrix Q. m >= 0.
     *             n: The number of columns of the matrix Q. m >= n >= 0.
     *             k: The number of elementary reflectors whose product defines the matrix Q.
     *                n >= k >= 0.
     *             A: an array, dimension(lda, n)
     *                On entry, the i-th column must contain the vector which defines the
     *                elementary reflector H(i), for i = 1, 2, ..., k, as returned by dgeqrf
     *                in the first k columns of its array argument A.
     *                On exit, the m-by-n matrix Q.
     *             lda: The first dimension of the array A. lda >= max(1, m).
     *             tau: an array, dimension(k)
     *                  tau(i) must contain the scalar factor of the elementary reflector H(i),
     *                  as returned by dgeqrf.
     *             work: an array, dimension(n)
     *             info: ==0: successful exit
     *                   < 0: if info = -i, the i-th argument has an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                         */
    static void dorg2r(int m, int n, int k, real* A, int lda, real const* tau, real* work,
                       int& info)
    {
        // Test the input arguments
        info = 0;
        if (m<0)
        {
            info = -1;
        }
        else if (n<0 || n>m)
        {
            info = -2;
        }
        else if (k<0 || k>n)
        {
            info = -3;
        }
        else if (lda<1 || lda<m)
        {
            info = -5;
        }
        if (info!=0)
        {
            xerbla("DORG2R", -info);
            return;
        }
        // Quick return if possible
        if (n<0)
        {
            return;
        }
        int i, j, acoli;
        // Initialise columns k:n-1 to columns of the unit matrix
        for (i=k; i<n; i++)
        {
            acoli = lda*i;
            for (j=0; j<m; j++)
            {
                A[j+acoli] = ZERO;
            }
            A[i+acoli] = ONE;
        }
        for (i=k-1; i>=0; i--)
        {
            acoli = lda*i;
            // Apply H(i) to A[i:m-1, i:n-1] from the left
            if (i<n-1)
            {
                A[i+acoli] = ONE;
                dlarf("Left", m-i, n-i-1, &A[i+acoli], 1, tau[i], &A[i+acoli+lda], lda, work);
            }
            if (i<m-1)
            {
                Blas<real>::dscal(m-i-1, -tau[i], &A[i+1+acoli], 1);
            }
            A[i+acoli] = ONE - tau[i];
            // Set A(1:i - 1, i) to zero
            for (j=0; j<i; j++)
            {
                A[j+acoli] = ZERO;
            }
        }
    }

    /* dorgqr generates an m-by-n real matrix Q with orthonormal columns, which is defined as the
     * first n columns of a product of k elementary reflectors of order m
     *     Q = H(1) H(2) . ..H(k)
     * as returned by dgeqrf.
     * Parameters: m: The number of rows of the matrix Q. m >= 0.
     *             n: The number of columns of the matrix Q. m >= n >= 0.
     *             k: The number of elementary reflectors whose product defines the matrix Q.
     *                n>=k>=0.
     *             A: an array, dimension(lda, n)
     *                On entry, the i-th column must contain the vector which defines the
     *                elementary reflector H(i), for i = 1, 2, ..., k, as returned by dgeqrf in the
     *                first k columns of its array argument A.
     *                On exit, the m-by-n matrix Q.
     *             lda: The first dimension of the array A. lda >= max(1, m).
     *             tau:  array, dimension(k). tau[i] must contain the scalar factor of the
     *                   elementary reflector H(i), as returned by dgeqrf.
     *             work: an array, dimension(max(1, lwork))
     *                   On exit, if info = 0, work[0] returns the optimal lwork.
     *             lwork: The dimension of the array work. lwork >= max(1, n).
     *                    For optimum performance lwork >= n*nb, where nb is the optimal blocksize.
     *                    If lwork = -1, then a workspace query is assumed; the routine only
     *                    calculates the optimal size of the work array, returns this value as the
     *                    first entry of the work array, and no error message related to lwork is
     *                    issued by xerbla.
     *             info: =0:  successful exit
     *                   <0:  if info = -i, the i-th argument has an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                         */
    static void dorgqr(int m, int n, int k, real* A, int lda, real const* tau, real* work,
                       int lwork, int& info)
    {
        // Test the input arguments
        info = 0;
        int nb = ilaenv(1, "DORGQR", " ", m, n, k, -1);
        int lwkopt = ((1>n)?1:n) * nb;
        work[0] = lwkopt;
        bool lquery = (lwork==-1);
        if (m<0)
        {
            info = -1;
        }
        else if (n<0 || n>m)
        {
            info = -2;
        }
        else if (k<0 || k>n)
        {
            info = -3;
        }
        else if (lda<1 || lda<m)
        {
            info = -5;
        }
        else if ((lwork<1 || lwork<n) && !lquery)
        {
            info = -8;
        }
        if (info!=0)
        {
            xerbla("DORGQR", -info);
            return;
        }
        else if (lquery)
        {
            return;
        }
        // Quick return if possible
        if (n<=0)
        {
            work[0] = 1;
            return;
        }
        int nbmin = 2;
        int nx = 0;
        int iws = n;
        int ldwork = 0;
        if (nb>1 && nb<k)
        {
            // Determine when to cross over from blocked to unblocked code.
            nx = ilaenv(3, "DORGQR", " ", m, n, k, -1);
            if (nx<0)
            {
                nx = 0;
            }
            if (nx<k)
            {
                // Determine if workspace is large enough for blocked code.
                ldwork = n;
                iws = ldwork*nb;
                if (lwork<iws)
                {
                    // Not enough workspace to use optimal nb:
                    // reduce nb and determine the minimum value of nb.
                    nb = lwork / ldwork;
                    nbmin = ilaenv(2, "DORGQR", " ", m, n, k, -1);
                    if (nbmin<2)
                    {
                        nbmin = 2;
                    }
                }
            }
        }
        int i, j, kk, ki = 0, acol;
        if (nb>=nbmin && nb<k && nx<k)
        {
            // Use blocked code after the last block. The first kk columns are handled by the block method.
            ki = ((k-nx-1)/nb) * nb;
            kk = ki + nb;
            if (kk>k)
            {
                kk = k;
            }
            // Set A[0:kk-1, kk:n-1] to zero.
            for (j=kk; j<n; j++)
            {
                acol = lda * j;
                for (i=0; i<kk; i++)
                {
                    A[i+acol] = ZERO;
                }
            }
        }
        else
        {
            kk = 0;
        }
        int iinfo;
        // Use unblocked code for the last or only block.
        if (kk<n)
        {
            dorg2r(m-kk, n-kk, k-kk, &A[kk+lda*kk], lda, &tau[kk], work, iinfo);
        }
        if (kk>0)
        {
            int ib, l;
            // Use blocked code
            for (i=ki; i>=0; i-=nb)
            {
                ib = k - i;
                if (ib>nb)
                {
                    ib = nb;
                }
                acol = i + lda*i;
                if (i+ib < n)
                {
                    // Form the triangular factor of the block reflector
                    // H = H(i) H(i + 1) ... H(i + ib - 1)
                    dlarft("Forward", "Columnwise", m-i, ib, &A[acol], lda, &tau[i], work, ldwork);
                    // Apply H to A[i:m-1, i+ib:n-1] from the left
                    dlarfb("Left", "No transpose", "Forward", "Columnwise", m-i, n-i-ib, ib,
                           &A[acol], lda, work, ldwork, &A[acol+lda*ib], lda, &work[ib], ldwork);
                }
                // Apply H to rows i:m-1 of current block
                dorg2r(m-i, ib, ib, &A[acol], lda, &tau[i], work, iinfo);
                // Set rows 0:i-1 of current block to zero
                for (j=i; j<(i+ib); j++)
                {
                    acol = lda * j;
                    for (l=0; l<i; l++)
                    {
                        A[l+acol] = ZERO;
                    }
                }
            }
        }
        work[0] = iws;
    }

    /* dorm2r overwrites the general real m-by-n matrix C with
     *     Q * C   if side = 'L' and trans = 'N', or
     *     Q^T* C  if side = 'L' and trans = 'T', or
     *     C * Q   if side = 'R' and trans = 'N', or
     *     C * Q^T if side = 'R' and trans = 'T',
     * where Q is a real orthogonal matrix defined as the product of k elementary reflectors
     *     Q = H(1) H(2) . ..H(k)
     * as returned by dgeqrf. Q is of order m if side = 'L' and of order n if side = 'R'.
     * Parameters: side: 'L': apply Q or Q^T from the Left
     *                   'R': apply Q or Q^T from the Right
     *             trans: 'N': apply Q (No transpose)
     *                    'T': apply Q^T (Transpose)
     *             m: The number of rows of the matrix C. m >= 0.
     *             n: The number of columns of the matrix C. n >= 0.
     *             k: The number of elementary reflectors whose product defines the matrix Q.
     *                If side = 'L', m >= k >= 0;
     *                if side = 'R', n >= k >= 0.
     *             A: an array, dimension(lda, k)
     *                The i-th column must contain the vector which defines the elementary
     *                reflector H(i), for i = 1, 2, ..., k, as returned by dgeqrf in the first k
     *                columns of its array argument A. A is modified by the routine but restored
     *                on exit.
     *             lda: The leading dimension of the array A.
     *                  If side = 'L', lda >= max(1, m);
     *                  if side = 'R', lda >= max(1, n).
     *             tau: an array, dimension(k)
     *                  tau[i] must contain the scalar factor of the elementary reflector H(i),
     *                  as returned by dgeqrf.
     *             C: an array, dimension(ldc, n)
     *                On entry, the m-by-n matrix C.
     *                On exit, C is overwritten by Q*C or Q^T*C or C*Q^T or C*Q.
     *             ldc: The leading dimension of the array C. ldc >= max(1, m).
     *             work: an array, dimension (n) if side = 'L',
     *                                       (m) if side = 'R'
     *             info: =0: successful exit
     *                   <0: if info = -i, the i-th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void dorm2r(char const* side, char const* trans, int m, int n, int k, real* A, int lda,
                       real const* tau, real* C, int ldc, real* work, int& info)
    {
        // Test the input arguments
        bool left = (toupper(side[0])=='L');
        bool notran = (toupper(trans[0])=='N');
        // nq is the order of Q
        int nq;
        if (left)
        {
            nq = m;
        }
        else
        {
            nq = n;
        }
        info = 0;
        if (!left && (toupper(side[0])!='R'))
        {
            info = -1;
        }
        else if (!notran && (toupper(trans[0])!='T'))
        {
            info = -2;
        }
        else if (m<0)
        {
            info = -3;
        }
        else if (n<0)
        {
            info = -4;
        }
        else if (k<0 || k>nq)
        {
            info = -5;
        }
        else if (lda<1 || lda<nq)
        {
            info = -7;
        }
        else if (ldc<1 || ldc<m)
        {
            info = -10;
        }
        if (info!=0)
        {
            xerbla("DORM2R", -info);
            return;
        }
        // Quick return if possible
        if (m==0 || n==0 || k==0)
        {
            return;
        }
        int i1, i2, i3;
        if ((left && !notran) || (!left && notran))
        {
            i1 = 0;
            i2 = k;
            i3 = 1;
        }
        else
        {
            i1 = k - 1;
            i2 = -1;
            i3 = -1;
        }
        int ic, jc, mi=0, ni=0;
        if (left)
        {
            ni = n;
            jc = 0;
        }
        else
        {
            mi = m;
            ic = 0;
        }
        int i, aind;
        real aii;
        for (i=i1; i!=i2; i+=i3)
        {
            if (left)
            {
                // H(i) is applied to C[i:m-1, 0:n-1]
                mi = m - i;
                ic = i;
            }
            else
            {
                // H(i) is applied to C[0:m-1, i:n-1]
                ni = n - i;
                jc = i;
            }
            // Apply H(i)
            aind = i + lda*i;
            aii = A[aind];
            A[aind] = ONE;
            dlarf(side, mi, ni, &A[aind], 1, tau[i], &C[ic+ldc*jc], ldc, work);
            A[aind] = aii;
        }
    }

    /* dormqr overwrites the general real m-by-n matrix C with
     *                     side = 'L'   side = 'R'
     *     trans = 'N':    Q * C        C * Q
     *     trans = 'T':    Q^T * C      C * Q^T
     * where Q is a real orthogonal matrix defined as the product of k elementary reflectors
     *     Q = H(1) H(2) . ..H(k)
     * as returned by dgeqrf. Q is of order m if side = 'L' and of order n if side = 'R'.
     * Parameters: side: 'L': apply Q or Q^T from the Left;
     *                   'R': apply Q or Q**T from the Right.
     *             trans: 'N':  No transpose, apply Q;
     *                    'T':  Transpose, apply Q^T.
     *             m: The number of rows of the matrix C. m >= 0.
     *             n: The number of columns of the matrix C. n >= 0.
     *             k: The number of elementary reflectors whose product defines the matrix Q.
     *                If side = 'L', m >= k >= 0;
     *                if side = 'R', n >= k >= 0.
     *             A: an array, dimension(lda, k)
     *               The i-th column must contain the vector which defines the elementary reflector
     *               H(i), for i=1, 2, ..., k, as returned by dgeqrf in the first k columns of its
     *               array argument A. A may be modified by the routine but is restored on exit.
     *             lda: The leading dimension of the array A.
     *                  If side = 'L', lda >= max(1, m);
     *                  if side = 'R', lda >= max(1, n).
     *             tau:  array, dimension(k)
     *                   tau(i) must contain the scalar factor of the elementary reflector H(i),
     *                   as returned by dgeqrf.
     *             C: an array, dimension(ldc, n)
     *                On entry, the m-by-n matrix C.
     *                On exit, C is overwritten by Q*C or Q^T*C or C*Q^T or C*Q.
     *             ldc: The leading dimension of the array C. ldc >= max(1, m).
     *             work: an array, dimension(MAX(1, lwork))
     *                   On exit, if info = 0, work[0] returns the optimal lwork.
     *             lwork: The dimension of the array work.
     *                    If side = 'L', lwork >= max(1, n);
     *                    if side = 'R', lwork >= max(1, m).
     *                    For optimum performance lwork >= n*nb if side = 'L',
     *                    and lwork >= m*nb if side = 'R', where nb is the optimal blocksize.
     *                    If lwork = -1, then a workspace query is assumed;
     *                    the routine only calculates the optimal size of the work array,
     *                    returns this value as the first entry of the work array,
     *                    and no error message related to lwork is issued by xerbla.
     *             info: =0:  successful exit
     *                   <0:  if info = -i, the i-th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                         */
    static void dormqr(char const* side, char const* trans, int m, int n, int k, real* A, int lda,
                       real const* tau, real* C, int ldc, real* work, int lwork, int& info)
    {
        const int NBMAX = 64;
        const int LDT = NBMAX + 1;
        real* T = new real[LDT*NBMAX];
        // Test the input arguments
        info = 0;
        char upside = toupper(side[0]);
        char uptrans = toupper(trans[0]);
        bool left = (upside=='L');
        bool notran = (uptrans=='N');
        bool lquery = (lwork==-1);
        // nq is the order of Q and nw is the minimum dimension of work
        int nq, nw;
        if (left)
        {
            nq = m;
            nw = n;
        }
        else
        {
            nq = n;
            nw = m;
        }
        if (!left && (upside!='R'))
        {
            info = -1;
        }
        else if (!notran && (uptrans!='T'))
        {
            info = -2;
        }
        else if (m<0)
        {
            info = -3;
        }
        else if (n<0)
        {
            info = -4;
        }
        else if (k<0 || k>nq)
        {
            info = -5;
        }
        else if (lda<1 || lda<nq)
        {
            info = -7;
        }
        else if (ldc<1 || ldc<m)
        {
            info = -10;
        }
        else if ((lwork<1 || lwork<nw) && !lquery)
        {
            info = -12;
        }
        char opts[2];
        opts[0] = upside;
        opts[1] = uptrans;
        int lwkopt, nb;
        if (info==0)
        {
            // Determine the block size. nb may be at most NBMAX, where NBMAX is used to define the local array T.
            nb = ilaenv(1, "DORMQR", opts, m, n, k, -1);
            if (nb>NBMAX)
            {
                nb = NBMAX;
            }
            lwkopt = ((1>nw)?1:nw) * nb;
            work[0] = lwkopt;
        }
        if (info!=0)
        {
            xerbla("DORMQR", -info);
            return;
        }
        else if (lquery)
        {
            return;
        }
        // Quick return if possible
        if (m==0 || n==0 || k==0)
        {
            work[0] = 1;
            return;
        }
        int nbmin = 2;
        int ldwork = nw;
        int iws;
        if (nb>1 && nb<k)
        {
            iws = nw*nb;
            if (lwork<iws)
            {
                nb = lwork / ldwork;
                nbmin = ilaenv(2, "DORMQR", opts, m, n, k, -1);
                if (nbmin<2)
                {
                    nbmin = 2;
                }
            }
        }
        else
        {
            iws = nw;
        }
        if (nb<nbmin || nb>=k)
        {
            // Use unblocked code
            int iinfo;
            dorm2r(side, trans, m, n, k, A, lda, tau, C, ldc, work, iinfo);
        }
        else
        {
            // Use blocked code
            int i1, i2, i3;
            if ((left && !notran) || (!left && notran))
            {
                i1 = 0;
                i2 = k - 1;
                i3 = nb;
            }
            else
            {
                i1 = ((k-1)/nb) * nb;
                i2 = 0;
                i3 = -nb;
            }
            int ic, jc, mi = 0, ni = 0;
            if (left)
            {
                ni = n;
                jc = 0;
            }
            else
            {
                mi = m;
                ic = 0;
            }
            int i, ib, aind;
            for (i=i1; (i3<0) ? (i>=i2) : (i<=i2); i+=i3)
            {
                ib = k - i;
                if (ib>nb)
                {
                    ib = nb;
                }
                // Form the triangular factor of the block reflector
                aind = i + lda*i;
                //     H = H(i) H(i + 1) . ..H(i + ib - 1)
                dlarft("Forward", "Columnwise", nq-i, ib, &A[aind], lda, &tau[i], T, LDT);
                if (left)
                {
                    // H or H^T is applied to C[i:m-1, 0:n-1]
                    mi = m - i;
                    ic = i;
                }
                else
                {
                    // H or H^T is applied to C[0:m-1, i:n-1]
                    ni = n - i;
                    jc = i;
                }
                // Apply H or H^T
                dlarfb(side, trans, "Forward", "Columnwise", mi, ni, ib, &A[aind], lda, T, LDT,
                       &C[ic+ldc*jc], ldc, work, ldwork);
            }
        }
        work[0] = lwkopt;
        delete[] T;
    }

    /* ieeeck is called from the ilaenv to verify that Infinity and possibly NaN arithmetic is safe
     * (i.e.will not trap).
     * Parameters: ispec: Specifies whether to test just for inifinity arithmetic or whether to
     *                    test for infinity and NaN arithmetic.
     *                    0: Verify infinity arithmetic only.
     *                    1: Verify infinity and NaN arithmetic.
     *             Zero: Must contain the value 0.0
     *                   This is passed to prevent the compiler from optimizing away some code.
     *             One: Must contain the value 1.0
     *                  This is passed to prevent the compiler from optimizing away some code.
     * Returns: 0: Arithmetic failed to produce the correct answers
     *          1: Arithmetic produced the correct answers
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * TODO: Check whether these checks are still valid in c++                                   */
    static int ieeeck(int ispec, real Zero, real One)
    {
        real posinf = One / Zero;
        if (posinf<One)
        {
            return 0;
        }
        real neginf = -One / Zero;
        if (neginf>=Zero)
        {
            return 0;
        }
        real negzro = One / (neginf+One);
        if (negzro!=Zero)
        {
            return 0;
        }
        neginf = One / negzro;
        if (neginf>=Zero)
        {
            return 0;
        }
        real newzro = negzro + Zero;
        if (newzro!=Zero)
        {
            return 0;
        }
        posinf = One / newzro;
        if (posinf<=One)
        {
            return 0;
        }
        neginf = neginf * posinf;
        if (neginf>=Zero)
        {
            return 0;
        }
        posinf = posinf * posinf;
        if (posinf<=One)
        {
            return 0;
        }
        // Return if we were only asked to check infinity arithmetic
        if (ispec==0)
        {
            return 1;
        }
        real NaN1 = posinf + neginf;
        real NaN2 = posinf / neginf;
        real NaN3 = posinf / posinf;
        real NaN4 = posinf * Zero;
        real NaN5 = neginf * negzro;
        real NaN6 = NaN5   * Zero;
        if (NaN1==NaN1)
        {
            return 0;
        }
        if (NaN2==NaN2)
        {
            return 0;
        }
        if (NaN3==NaN3)
        {
            return 0;
        }
        if (NaN4==NaN4)
        {
            return 0;
        }
        if (NaN5==NaN5)
        {
            return 0;
        }
        if (NaN6==NaN6)
        {
            return 0;
        }
        return 1;
    }

    /* iladlc scans A for its last non - zero column.
     * Parameters: m: The number of rows of the matrix A.
     *             n: The number of columns of the matrix A.
     *             A: an array, dimension(lda, n)
     *                The m by n matrix A.
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                           */
    static int iladlc(int m, int n, real const* A, int lda)
    {
        int ila, lastcol=lda*(n-1);
        // Quick test for the common case where one corner is non - zero.
        if (n==0)
        {
            return 0;
        }
        else if (A[lastcol]!=ZERO || A[m-1+lastcol]!=ZERO)
        {
            return n - 1;
        }
        else
        {
            int i;
            // Now scan each column from the end, returning with the first non-zero.
            for (ila=lastcol; ila>=0; ila-=lda)
            {
                for (i=0; i<m; i++)
                {
                    if (A[i+ila] != ZERO)
                    {

                        return ila / lda;
                    }
                }
            }
            return -1;
        }
    }

    /* iladlr scans A for its last non - zero row.
     * Parameters: m: The number of rows of the matrix A.
     *             n: The number of columns of the matrix A.
     *             A: an array, dimension(lda, n)
     *                The m by n matrix A.
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                           */
    static int iladlr(int m, int n, real const* A, int lda)
    {
        int lastrow = m - 1;
        // Quick test for the common case where one corner is non - zero.
        if (m==0)
        {
            return 0;
        }
        else if (A[lastrow]!=ZERO || A[lastrow+lda*(n-1)]!=ZERO)
        {
            return lastrow;
        }
        else
        {
            int i, j;
            // Scan up each column tracking the last zero row seen.
            int colj, ila=0;
            for (j=0; j<n; j++)
            {
                colj = lda * j;
                i = lastrow;
                while ((i>0) && (A[i+colj]==ZERO))
                {
                    i--;
                }
                if (ila<i)
                {
                    ila = i;
                }
            }
            return ila;
        }
    }

    /* ilaenv is called from the LAPACK routines to choose problem-dependent parameters for the
     * local environment.
     * See ispec for a description of the parameters.
     * ilaenv returns an integer: if >=0: ilaenv returns the value of the parameter specified by
     *                                     ispec
     *                            if  <0: if -k, the k-th argument had an illegal value.
     * This version provides a set of parameters which should give good, but not optimal,
     *     performance on many of the currently available computers.Users are encouraged to modify
     *     this subroutine to set the tuning parameters for their particular machine using the
     *     option and problem size information in the arguments.
     * This routine will not function correctly if it is converted to all lower case.
     *     Converting it to all upper case is allowed.
     * Parameters: ispec: Specifies the parameter to be returned.
     *                    1: the optimal blocksize; if this value is 1, an unblocked algorithm will
     *                       give the best performance.
     *                    2: the minimum block size for which the block routine should be used;
     *                       if the usable block size is less than this value, an unblocked routine
     *                       should be used.
     *                    3: the crossover point(in a block routine, for N less than this value, an
     *                       unblocked routine should be used)
     *                    4: the number of shifts, used in the nonsymmetric eigenvalue routines
     *                       (DEPRECATED)
     *                    5: the minimum column dimension for blocking to be used; rectangular
     *                       blocks must have dimension at least k by m, where k is given by
     *                       ILAENV(2, ...) and m by ILAENV(5, ...)
     *                    6: the crossover point for the SVD (when reducing an m by n matrix to
     *                       bidiagonal form, if max(m, n) / min(m, n) exceeds this value, a QR
     *                       factorization is used first to reduce the matrix to triangular form.)
     *                    7: the number of processors
     *                    8: the crossover point for the multishift QR method for nonsymmetric
     *                       eigenvalue problems(DEPRECATED)
     *                    9: maximum size of the subproblems at the bottom of the computation tree
     *                       in the divide-and-conquer algorithm (used by xGELSD and xGESDD)
     *                    10: ieee NaN arithmetic can be trusted not to trap
     *                    11: infinity arithmetic can be trusted not to trap
     *                    12<=ispec<=16: xHSEQR or one of its subroutines, see IPARMQ for
     *                                   detailed explanation
     *             name: The name of the calling subroutine, in either upper case or lower case.
     *             opts: The character options to the subroutine name, concatenated into a single
     *                   character string. For example, UPLO=='U', trans=='T', and DIAG=='N' for a
     *                   triangular routine would be specified as opts=='UTN'.
     *             n1: integer
     *             n2: integer
     *             n3: integer
     *             n4: integer; Problem dimensions for the subroutine name;
     *                          these may not all be required.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Further Details:
     *     The following conventions have been used when calling ilaenv from the LAPACK routines:
     *      1)  opts is a concatenation of all of the character options to subroutine name, in the
     *          same order that they appear in the argument list for name, even if they are not
     *          used in determining the value of the parameter specified by ispec.
     *      2)  The problem dimensions n1, n2, n3, n4 are specified in the order that they appear
     *          in the argument list for name. n1 is used first, n2 second, and so on, and unused
     *          problem dimensions are passed a value of -1.
     *      3)  The parameter value returned by ILAENV is checked for validity in the calling
     *          subroutine. For example, ilaenv is used to retrieve the optimal blocksize for
     *          STRTRI as follows:
     *          nb = ilaenv(1, 'STRTRI', UPLO // DIAG, N, -1, -1, -1)
     *          if(nb<=1) nb = MAX(1, N);
     * TODO: optimize                                                                            */
    static int ilaenv(int ispec, char const* name, char const* opts, int n1, int n2, int n3,
                      int n4)
    {
        bool sname, cname;
        char c1;
        switch (ispec)
        {
            case 1:
            case 2:
            case 3:
                // Convert name to upper case.
                char subnam[6];
                std::strncpy(subnam, name, 6);
                int nb, nbmin;
                for (nb=0; nb<6; nb++)
                {
                    subnam[nb] = toupper(subnam[nb]);
                }
                c1 = subnam[0];
                sname = (c1=='S' || c1=='D');
                cname = (c1=='C' || c1=='Z');
                if (!(cname || sname))
                {
                    return 1;
                }
                char c2[2], c3[3], c4[2];
                std::strncpy(c2, subnam+1, 2);
                std::strncpy(c3, subnam+3, 3);
                std::strncpy(c4, c3+1, 2);
                switch (ispec)
                {
                    case 1:
                        // ispec = 1: block size
                        // In these examples, separate code is provided for setting nb for real and
                        // complex. We assume that nb will take the same value in single or double
                        // precision.
                        nb = 1;
                        if (std::strncmp(c2, "GE", 2)==0)
                        {
                            if (std::strncmp(c3, "TRF", 3)==0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                }
                                else
                                {
                                    nb = 64;
                                }
                            }
                            else if (std::strncmp(c3, "QRF", 3)==0 || std::strncmp(c3, "RQF", 3)==0
                                 || std::strncmp(c3, "LQF", 3)==0 || std::strncmp(c3, "QLF", 3)==0)
                            {
                                if (sname)
                                {
                                    nb = 32;
                                }
                                else
                                {
                                    nb = 32;
                                }
                            }
                            else if (std::strncmp(c3, "HRD", 3)==0)
                            {
                                if (sname)
                                {
                                    nb = 32;
                                }
                                else
                                {
                                    nb = 32;
                                }
                            }
                            else if (std::strncmp(c3, "BRD", 3)==0)
                            {
                                if (sname)
                                {
                                    nb = 32;
                                }
                                else
                                {
                                    nb = 32;
                                }
                            }
                            else if (std::strncmp(c3, "TRI", 3)==0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                }
                                else
                                {
                                    nb = 64;
                                }
                            }
                        }
                        else if (std::strncmp(c2, "PO", 2)==0)
                        {
                            if (std::strncmp(c3, "TRF", 3)==0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                }
                                else
                                {
                                    nb = 64;
                                }
                            }
                        }
                        else if (std::strncmp(c2, "SY", 2)==0)
                        {
                            if (std::strncmp(c3, "TRF", 3)==0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                }
                                else
                                {
                                    nb = 64;
                                }
                            }
                            else if (sname && std::strncmp(c3, "TRD", 3)==0)
                            {
                                nb = 32;
                            }
                            else if (sname && std::strncmp(c3, "GST", 3)==0)
                            {
                                nb = 64;
                            }
                        }
                        else if (cname && std::strncmp(c2, "HE", 2)==0)
                        {
                            if (std::strncmp(c3, "TRF", 3)==0)
                            {
                                nb = 64;
                            }
                            else if (std::strncmp(c3, "TRD", 3)==0)
                            {
                                nb = 32;
                            }
                            else if (std::strncmp(c3, "GST", 3)==0)
                            {
                                nb = 64;
                            }
                        }
                        else if (sname && std::strncmp(c2, "OR", 2)==0)
                        {
                            if (c3[0]=='G')
                            {
                                if (std::strncmp(c4, "QR", 2)==0 || std::strncmp(c4, "RQ", 2)==0
                                    || std::strncmp(c4, "LQ", 2)==0 || std::strncmp(c4, "QL", 2)==0
                                    || std::strncmp(c4, "HR", 2)==0 || std::strncmp(c4, "TR", 2)==0
                                    || std::strncmp(c4, "BR", 2)==0)
                                {
                                    nb = 32;
                                }
                            }
                            else if (c3[0]=='M')
                            {
                                if (std::strncmp(c4, "QR", 2)==0 || std::strncmp(c4, "RQ", 2)==0
                                    || std::strncmp(c4, "LQ", 2)==0 || std::strncmp(c4, "QL", 2)==0
                                    || std::strncmp(c4, "HR", 2)==0 || std::strncmp(c4, "TR", 2)==0
                                    || std::strncmp(c4, "BR", 2)==0)
                                {
                                    nb = 32;
                                }
                            }
                        }
                        else if (cname && std::strncmp(c2, "UN", 2)==0)
                        {
                            if (c3[0]=='G')
                            {
                                if (std::strncmp(c4, "QR", 2)==0 || std::strncmp(c4, "RQ", 2)==0
                                    || std::strncmp(c4, "LQ", 2)==0 || std::strncmp(c4, "QL", 2)==0
                                    || std::strncmp(c4, "HR", 2)==0 || std::strncmp(c4, "TR", 2)==0
                                    || std::strncmp(c4, "BR", 2)==0)
                                {
                                    nb = 32;
                                }
                            }
                            else if (c3[0]=='M')
                            {
                                if (std::strncmp(c4, "QR", 2)==0 || std::strncmp(c4, "RQ", 2)==0
                                    || std::strncmp(c4, "LQ", 2)==0 || std::strncmp(c4, "QL", 2)==0
                                    || std::strncmp(c4, "HR", 2)==0 || std::strncmp(c4, "TR", 2)==0
                                    || std::strncmp(c4, "BR", 2)==0)
                                {
                                    nb = 32;
                                }
                            }
                        }
                        else if (std::strncmp(c2, "GB", 2)==0)
                        {
                            if (std::strncmp(c3, "TRF", 3)==0)
                            {
                                if (sname)
                                {
                                    if (n4<=64)
                                    {
                                        nb = 1;
                                    }
                                    else
                                    {
                                        nb = 32;
                                    }
                                }
                                else
                                {
                                    if (n4<=64)
                                    {
                                        nb = 1;
                                    }
                                    else
                                    {
                                        nb = 32;
                                    }
                                }
                            }
                        }
                        else if (std::strncmp(c2, "PB", 2)==0)
                        {
                            if (std::strncmp(c3, "TRF", 3)==0)
                            {
                                if (sname)
                                {
                                    if (n2<=64)
                                    {
                                        nb = 1;
                                    }
                                    else
                                    {
                                        nb = 32;
                                    }
                                }
                                else
                                {
                                    if (n2<=64)
                                    {
                                        nb = 1;
                                    }
                                    else
                                    {
                                        nb = 32;
                                    }
                                }
                            }
                        }
                        else if (std::strncmp(c2, "TR", 2)==0)
                        {
                            if (std::strncmp(c3, "TRI", 3)==0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                }
                                else
                                {
                                    nb = 64;
                                }
                            }
                        }
                        else if (std::strncmp(c2, "LA", 2)==0)
                        {
                            if (std::strncmp(c3, "UUM", 3)==0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                }
                                else
                                {
                                    nb = 64;
                                }
                            }
                        }
                        else if (sname && std::strncmp(c2, "ST", 2)==0)
                        {
                            if (std::strncmp(c3, "EBZ", 3)==0)
                            {
                                nb = 1;
                            }
                        }
                        return nb;
                        break;
                    case 2:
                        // ispec = 2: minimum block size
                        nbmin = 2;
                        if (std::strncmp(c2, "GE", 2)==0)
                        {
                            if (std::strncmp(c3, "QRF", 3)==0 || std::strncmp(c3, "RQF", 3)==0
                                || std::strncmp(c3, "LQF", 3)==0 || std::strncmp(c3, "QLF", 3)==0)
                            {
                                if (sname)
                                {
                                    nbmin = 2;
                                }
                                else
                                {
                                    nbmin = 2;
                                }
                            }
                            else if (std::strncmp(c3, "HRD", 3)==0)
                            {
                                if (sname)
                                {
                                    nbmin = 2;
                                }
                                else
                                {
                                    nbmin = 2;
                                }
                            }
                            else if (std::strncmp(c3, "BRD", 3)==0)
                            {
                                if (sname)
                                {
                                    nbmin = 2;
                                }
                                else
                                {
                                    nbmin = 2;
                                }
                            }
                            else if (std::strncmp(c3, "TRI", 3)==0)
                            {
                                if (sname)
                                {
                                    nbmin = 2;
                                }
                                else
                                {
                                    nbmin = 2;
                                }
                            }
                        }
                        else if (std::strncmp(c2, "SY", 2)==0)
                        {
                            if (std::strncmp(c3, "TRF", 3)==0)
                            {
                                if (sname)
                                {
                                    nbmin = 8;
                                }
                                else
                                {
                                    nbmin = 8;
                                }
                            }
                            else if (sname && std::strncmp(c3, "TRD", 3)==0)
                            {
                                nbmin = 2;
                            }
                        }
                        else if (cname && std::strncmp(c2, "HE", 2)==0)
                        {
                            if (std::strncmp(c3, "TRD", 3)==0)
                            {
                                nbmin = 2;
                            }
                        }
                        else if (sname && std::strncmp(c2, "OR", 2)==0)
                        {
                            if (c3[0]=='G')
                            {
                                if (std::strncmp(c4, "QR", 2)==0 || std::strncmp(c4, "RQ", 2)==0
                                    || std::strncmp(c4, "LQ", 2)==0 || std::strncmp(c4, "QL", 2)==0
                                    || std::strncmp(c4, "HR", 2)==0 || std::strncmp(c4, "TR", 2)==0
                                    || std::strncmp(c4, "BR", 2)==0)
                                {
                                    nbmin = 2;
                                }
                            }
                            else if (c3[0]=='M')
                            {
                                if (std::strncmp(c4, "QR", 2)==0 || std::strncmp(c4, "RQ", 2)==0
                                    || std::strncmp(c4, "LQ", 2)==0 || std::strncmp(c4, "QL", 2)==0
                                    || std::strncmp(c4, "HR", 2)==0 || std::strncmp(c4, "TR", 2)==0
                                    || std::strncmp(c4, "BR", 2)==0)
                                {
                                    nbmin = 2;
                                }
                            }
                        }
                        else if (cname && std::strncmp(c2, "UN", 2)==0)
                        {
                            if (c3[0]=='G')
                            {
                                if (std::strncmp(c4, "QR", 2)==0 || std::strncmp(c4, "RQ", 2)==0
                                    || std::strncmp(c4, "LQ", 2)==0 || std::strncmp(c4, "QL", 2)==0
                                    || std::strncmp(c4, "HR", 2)==0 || std::strncmp(c4, "TR", 2)==0
                                    || std::strncmp(c4, "BR", 2)==0)
                                {
                                    nbmin = 2;
                                }
                            }
                            else if (c3[0]=='M')
                            {
                                if (std::strncmp(c4, "QR", 2)==0 || std::strncmp(c4, "RQ", 2)==0
                                    || std::strncmp(c4, "LQ", 2)==0 || std::strncmp(c4, "QL", 2)==0
                                    || std::strncmp(c4, "HR", 2)==0 || std::strncmp(c4, "TR", 2)==0
                                    || std::strncmp(c4, "BR", 2)==0)
                                {
                                    nbmin = 2;
                                }
                            }
                        }
                        return nbmin;
                        break;
                    case 3:
                        // ispec = 3: crossover point
                        int nx = 0;
                        if (std::strncmp(c2, "GE", 2)==0)
                        {
                            if (std::strncmp(c3, "QRF", 3)==0 || std::strncmp(c3, "RQF", 3)==0
                                || std::strncmp(c3, "LQF", 3)==0 || std::strncmp(c3, "QLF", 3)==0)
                            {
                                if (sname)
                                {
                                    nx = 128;
                                }
                                else
                                {
                                    nx = 128;
                                }
                            }
                            else if (std::strncmp(c3, "HRD", 3)==0)
                            {
                                if (sname)
                                {
                                    nx = 128;
                                }
                                else
                                {
                                    nx = 128;
                                }
                            }
                            else if (std::strncmp(c3, "BRD", 3)==0)
                            {
                                if (sname)
                                {
                                    nx = 128;
                                }
                                else
                                {
                                    nx = 128;
                                }
                            }
                        }
                        else if (std::strncmp(c2, "SY", 2)==0)
                        {
                            if (sname && std::strncmp(c3, "TRD", 3)==0)
                            {
                                nx = 32;
                            }
                        }
                        else if (cname && std::strncmp(c2, "HE", 2)==0)
                        {
                            if (std::strncmp(c3, "TRD", 3)==0)
                            {
                                nx = 32;
                            }
                        }
                        else if (sname && std::strncmp(c2, "OR", 2)==0)
                        {
                            if (c3[0]=='G')
                            {
                                if (std::strncmp(c4, "QR", 2)==0 || std::strncmp(c4, "RQ", 2)==0
                                    || std::strncmp(c4, "LQ", 2)==0 || std::strncmp(c4, "QL", 2)==0
                                    || std::strncmp(c4, "HR", 2)==0 || std::strncmp(c4, "TR", 2)==0
                                    || std::strncmp(c4, "BR", 2)==0)
                                {
                                    nx = 128;
                                }
                            }
                        }
                        else if (cname && std::strncmp(c2, "UN", 2)==0)
                        {
                            if (c3[0]=='G')
                            {
                                if (std::strncmp(c4, "QR", 2)==0 || std::strncmp(c4, "RQ", 2)==0
                                    || std::strncmp(c4, "LQ", 2)==0 || std::strncmp(c4, "QL", 2)==0
                                    || std::strncmp(c4, "HR", 2)==0 || std::strncmp(c4, "TR", 2)==0
                                    || std::strncmp(c4, "BR", 2)==0)
                                {
                                    nx = 128;
                                }
                            }
                        }
                        return nx;
                        break;
                }
                break;
            case 4:
                // ispec = 4: number of shifts(used by xhseqr)
                return 6;
                break;
            case 5:
                // ispec = 5: minimum column dimension(not used)
                return 2;
                break;
            case 6:
                // ispec = 6: crossover point for SVD(used by xgelss and xgesvd)
                return int(real((n1<n2)?n1:n2) * real(1.6));
                break;
            case 7:
                // ispec = 7: number of processors (not used)
                return 1;
                break;
            case 8:
                // ispec = 8: crossover point for multishift (used by xhseqr)
                return 50;
                break;
            case 9:
                // ispec = 9: maximum size of the subproblems at the bottom of the computation
                // tree in the divide-and-conquer algorithm (used by xgelsd and xgesdd)
                return 25;
                break;
            case 10:
                // ispec = 10: ieee NaN arithmetic can be trusted not to trap
                // return 0;
                return ieeeck(1, ZERO, ONE);
                break;
            case 11:
                // ispec = 11: infinity arithmetic can be trusted not to trap
                // return 0;
                return ieeeck(0, ZERO, ONE);
                break;
            case 12:
            case 13:
            case 14:
            case 15:
            case 16:
                // 12 <= ispec <= 16: xhseqr or one of its subroutines.
                return iparmq(ispec, name, opts, n1, n2, n3, n4);
                break;
            default:
                // Invalid value for ispec
                return -1;
        }
        return -1;
    }

    /* This subroutine returns the LAPACK version.
     * Parameter: vers_major: return the lapack major version
     *            vers_minor: return the lapack minor version from the major version.
     *            vers_patch: return the lapack patch version from the minor version
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: June 2017                                                                           */
    static void ilaver(int& vers_major, int& vers_minor, int& vers_patch)
    {
        vers_major = 3;
        vers_minor = 8;
        vers_patch = 0;
    }

    /* This program sets problem and machine dependent parameters useful for xHSEQR and its
     * subroutines.
     * It is called whenever ilaenv is called with 12 <= ispec <= 16
     * Parameters: ispec: specifies which tunable parameter IPARMQ should return.
     *                    12: (inmin) Matrices of order nmin or less are sent directly to xLAHQR,
     *                        the implicit double shift QR algorithm. NMIN must be at least 11.
     *                    13: (inwin) Size of the deflation window. This is best set greater than
     *                        or equal to the number of simultaneous shifts ns. Larger matrices
     *                        benefit from larger deflation windows.
     *                    14: (inibl) Determines when to stop nibbling and invest in an(expensive)
     *                        multi-shift QR sweep. If the aggressive early deflation subroutine
     *                        finds LD converged eigenvalues from an order nw deflation window and
     *                        LD>(nw*NIBBLE) / 100, then the next QR sweep is skipped and early
     *                        deflation is applied immediately to the remaining active diagonal
     *                        block. Setting iparmq(ispec=14)=0 causes ttqre to skip a multi-shift
     *                        QR sweep whenever early deflation finds a converged eigenvalue.
     *                        Setting iparmq(ispec=14) greater than or equal to 100 prevents ttqre
     *                        from skipping a multi-shift QR sweep.
     *                    15: (nshfts) The number of simultaneous shifts in a multi-shift QR
     *                        iteration.
     *                    16: (iacc22) iparmq is set to 0, 1 or 2 with the following meanings.
     *                        0: During the multi-shift QR sweep, xlaqr5 does not accumulate
     *                           reflections and does not use matrix-matrix multiply to update
     *                           the far-from-diagonal matrix entries.
     *                        1: During the multi-shift QR sweep, xlaqr5 and/or xlaqr accumulates
     *                           reflections and uses matrix-matrix multiply to update the
     *                           far-from-diagonal matrix entries.
     *                        2: During the multi-shift QR sweep. xlaqr5 accumulates reflections
     *                           and takes advantage of 2-by-2 block structure during matrix-matrix
     *                           multiplies.
     *                        (If xTRMM is slower than xgemm, then iparmq(ispec=16)=1 may be more
     *                         efficient than iparmq(ispec=16)=2 despite the greater level of
     *                         arithmetic work implied by the latter choice.)
     *             name: Name of the calling subroutine
     *             opts: This is a concatenation of the string arguments to ttqre.
     *             n: n is the order of the Hessenberg matrix H.
     *             ilo: integer
     *             ihi: integer
     *                  It is assumed that H is already upper triangular in rows and columns
     *                  1:ilo-1 and ihi+1:n.
     *             lwork: The amount of workspace available.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: June 2017
     * Further Details:
     *     Little is known about how best to choose these parameters. It is possible to use
     *         different values of the parameters for each of chseqr, dhseqr, shseqr and zhseqr.
     *     It is probably best to choose different parameters for different matrices and different
     *         parameters at different times during the iteration, but this has not been
     *         implemented yet.
     *     The best choices of most of the parameters depend in an ill-understood way on the
     *         relative execution rate of xlaqr3 and xlaqr5 and on the nature of each particular
     *         eigenvalue problem. Experiment may be the only practical way to determine which
     *         choices are most effective.
     *     Following is a list of default values supplied by iparmq. These defaults may be adjusted
     *         in order to attain better performance in any particular computational environment.
     *     iparmq(ispec = 12) The xlahqr vs xlaqr0 crossover point.
     *         Default: 75. (Must be at least 11.)
     *     iparmq(ispec = 13) Recommended deflation window size.
     *         This depends on ilo, ihi and ns, the number of simultaneous shifts returned by
     *         iparmq(ispec = 15). The default for (ihi-ilo+1)<=500 is ns. The default for
     *         (ihi-ilo+1)>500 is 3*ns/2.
     *     iparmq(ispec = 14) Nibble crossover point. Default: 14.
     *     iparmq(ispec = 15) Number of simultaneous shifts, ns.
     *         a multi-shift QR iteration. If ihi-ilo+1 is ...
     *         greater than or equal to...      but less than...      the default is
     *                                   0                    30                     ns = 2+
     *                                  30                    60                     ns = 4+
     *                                  60                   150                     ns = 10
     *                                 150                   590                     ns = **
     *                                 590                  3000                     ns = 64
     *                                3000                  6000                     ns = 128
     *                                6000                  infinity                 ns = 256
     *         (+) By default matrices of this order are passed to the implicit double shift
     *             routine xlahqr. See iparmq(ispec=12) above. These values of ns are used only in
     *             case of a rare xlahqr failure.
     *         (**) an ad-hoc function increasing from 10 to 64.
     *     iparmq(ispec = 16) Select structured matrix multiply. (See ispec=16 above for details.)
     *         Default: 3.                                                                       */
    static int iparmq(int ispec, char const* name, char const* opts, int n, int ilo, int ihi,
                      int lwork)
    {
        const int INMIN=12, INWIN=13, INIBL=14, ISHFTS=15, IACC22=16, NMIN=75, K22MIN=14,
                  KACMIN=14, NIBBLE=14, KNWSWP=500;
        int nh, ns = 0;
        if ((ispec==ISHFTS) || (ispec==INWIN) || (ispec==IACC22))
        {
            // Set the number simultaneous shifts
            nh = ihi - ilo + 1;
            ns = 2;
            if (nh>=30)
            {
                ns = 4;
            }
            if (nh>=60)
            {
                ns = 10;
            }
            if (nh>=150)
            {
                ns = std::max(10, nh / int(std::log(real(nh))/std::log(TWO)));
            }
            if (nh>=590)
            {
                ns = 64;
            }
            if (nh>=3000)
            {
                ns = 128;
            }
            if (nh>=6000)
            {
                ns = 256;
            }
            ns = std::max(2, ns-(ns%2));
        }
        if (ispec==INMIN)
        {
            // Matrices of order smaller than NMIN get sent to xLAHQR, the classic double shift algorithm. This must be at least 11.
            return NMIN;
        }
        else if (ispec==INIBL)
        {
            // INIBL: skip a multi-shift qr iteration and whenever aggressive early
            // deflation finds at least(NIBBLE*(window size) / 100) deflations.
            return NIBBLE;
        }
        else if (ispec==ISHFTS)
        {
            // NSHFTS: The number of simultaneous shifts
            return ns;
        }
        else if (ispec==INWIN)
        {
            // nw: deflation window size.
            if (nh<=KNWSWP)
            {
                return ns;
            }
            else
            {
                return 3 * ns / 2;
            }
        }
        else if (ispec==IACC22)
        {
            /* IACC22: Whether to accumulate reflections before updating the far-from-diagonal
             * elements and whether to use 2-by-2 block structure while doing it. A small amount of
             * work could be saved by making this choice dependent also upon the nh = ihi-ilo+1. */
            // Convert NAME to upper case if the first character is lower case.
            char subnam[6];
            for (int i=0; i<6; i++)
            {
                subnam[i] = std::toupper(name[i]);
            }
            if (std::strncmp(&subnam[1], "GGHRD", 5)==0 || std::strncmp(&subnam[1], "GGHD3", 5)==0)
            {
                if (nh>=K22MIN)
                {
                    return 2;
                }
                return 1;
            }
            else if (strncmp(&subnam[3], "EXC", 3)==0)
            {
                if (nh>=K22MIN)
                {
                    return 2;
                }
                if (nh>=KACMIN)
                {
                    return 1;
                }
            }
            else if (strncmp(&subnam[1], "HSEQR", 5)==0 || strncmp(&subnam[1], "LAQR", 4)==0)
            {
                if(ns>=K22MIN)
                {
                    return 2;
                }
                if (ns>=KACMIN)
                {
                    return 1;
                }
            }
            return 0;
        }
        else
        {
            // invalid value of ispec
            return -1;
        }
    }

    /* xerbla is an error handler for the LAPACK routines.
     * It is called by an LAPACK routine if an input parameter has an invalid value.
     * A message is printed and execution stops.
     * Installers may consider modifying the STOP statement in order to call
     * system-specific exception-handling facilities.
     * Parameters: srname: The name of the routine which called xerbla.
     *             info: The position of the invalid parameter in the parameter list of the calling
     *                   routine.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                         */
    static void xerbla(char const* srname, int info)
    {
        std::cerr << "On entry to " << srname << " parameter number " << info
                  << " had an illegal value.";
        throw info;
    }
};
#endif