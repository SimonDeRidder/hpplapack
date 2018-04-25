#include <limits>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cmath>

#include "Blas.h"

#ifndef LAPACK_HEADER
#define LAPACK_HEADER

template<class T>
class Lapack
{
public:
    // constants

    static constexpr T ZERO  = T(0.0);
    static constexpr T QURTR = T(0.25);
    static constexpr T HALF  = T(0.5);
    static constexpr T ONE   = T(1.0);
    static constexpr T TWO   = T(2.0);
    static constexpr T HNDRD = T(100.0);

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
    static T dlamch(char const* cmach)
    {
        T eps;
        // Assume rounding, not chopping.Always.
        T rnd = ONE;
        if (ONE==rnd)
        {
            eps = std::numeric_limits<T>::epsilon() * HALF;
        }
        else
        {
            eps = std::numeric_limits<T>::epsilon();
        }
        T sfmin, small;
        switch (toupper(cmach[0]))
        {
            case 'E':
                return eps;
            case 'S':
                sfmin = std::numeric_limits<T>::min();
                small = ONE / std::numeric_limits<T>::max();
                if (small>sfmin)
                {
                    //Use SMALL plus a bit, to avoid the possibility of rounding
                    // causing overflow when computing  1 / sfmin.
                    sfmin = small * (ONE+eps);
                }
                return sfmin;
            case 'B':
                return std::numeric_limits<T>::radix;
            case 'P':
                return eps * std::numeric_limits<T>::radix;
            case 'N':
                return std::numeric_limits<T>::digits;
            case 'R':
                return rnd;
            case 'M':
                return std::numeric_limits<T>::min_exponent;
            case 'U':
                return std::numeric_limits<T>::min();
            case 'L':
                return std::numeric_limits<T>::max_exponent;
            case 'O':
                return std::numeric_limits<T>::max();
            default:
                return ZERO;
        }
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
    static void dbdsqr(char const* uplo, int n, int ncvt, int nru, int ncc, T* d, T* e, T* Vt,
                       int ldvt, T* U, int ldu, T* C, int ldc, T* work, int& info)
    {
        const T NEGONE = T(-1.0);
        const T MEIGTH = T(-0.125);
        const T HNDRTH = T(0.01);
        const T TEN    = T(10.0);
        const int MAXITR = 6;
        // Test the input parameters.
        info = 0;
        bool lower = (std::toupper(UPLO[0])=='L');
        if (!(std::toupper(UPLO[0])=='U') && !lower)
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
        T smin;
        if (n!=1)
        {
            // rotate is true if any singular vectors desired, false otherwise
            bool rotate = ((ncvt>0) || (nru>0) || (ncc>0));
            // If no singular vectors desired, use qd algorithm
            if (!rotate)
            {
                dlasq1(n, d, e, work, info);
                // If INFO equals 2, dqds didn't finish, try to finish
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
            T eps = dlamch("Epsilon");
            T unfl = dlamch("Safe minimum");
            // If matrix lower bidiagonal, rotate to be upper bidiagonal by applying
            // Givens rotations on the left
            T cs, sn, r;
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
            // Compute singular values to relative accuracy TOL (By setting TOL to be negative,
            // algorithm will compute singular values to absolute accuracy
            // abs(tol)*norm(input matrix))
            T tolmul = std::pow(eps, MEIGTH);
            if (HNDRD<tolmul)
            {
                tolmul = HNDRD;
            }
            if (TEN>tolmul)
            {
                tolmul = TEN;
            }
            T tol = tolmul * eps;
            // Compute approximate maximum, minimum singular values
            T smax = ZERO, temp;
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
            T sminl = ZERO, sminoa, mu, thresh;
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
                sminoa = sminoa / std::sqrt(T(n));
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
            T abse, abss, cosl, cosr, f, g, h, oldcs, oldsn, shift, sigmn, sigmx, sinl, sinr, sll;
            bool breakloop1 = false, breakloop2;
            while (true)
            {
                // Check for convergence or exceeding iteration count
                if (m<=0)
                {
                    break
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
                loopbreak2 = false;
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
                        loopbreak2 = true;
                        break;
                    }
                    if (abss<smin)
                    {
                        smin = abss;
                    }
                    temp = ((abss>abse) ? abss : abse);
                    if (temp>smax)
                    {
                        smax = temp;
                    }
                }
                if (loopbreak2)
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
                        Blas<T>::drot(ncvt, &Vt[m-1/*ldvt*0*/], ldvt, &Vt[m/*+ldvt*0*/], ldvt,
                                      cosr, sinr);
                    }
                    if (nru>0)
                    {
                        Blas<T>::drot(nru, &U[/*0+*/ldu*(m-1)], 1, &U[/*0+*/ldu*m], 1, cosl, sinl);
                    }
                    if (ncc>0)
                    {
                        Blas<T>::drot(ncc, &C[m-1/*+ldc*0*/], ldc, &C[m/*+ldc*0*/], ldc, cosl,
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
                        loopbreak2 = false;
                        for (lll=ll; lll<m; lll++)
                        {
                            if (std::fabs(e[lll]<=tol*mu))
                            {
                                e[lll] = ZERO;
                                loopbreak2 = true;
                                break;
                            }
                            mu = std::fabs(d[lll+1]) * (mu/(mu+std::fabs(e[lll])));
                            if (mu<sminl)
                            {
                                sminl = mu;
                            }
                        }
                        if (loopbreak2)
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
                        loopbreak2 = false;
                        for (lll=m-1; lll>=ll; lll--)
                        {
                            if (std::fabs(e[lll])<=tol*mu)
                            {
                                e[lll] = ZERO;
                                loopbreak2 = true;
                                break
                            }
                            mu = std::fabs(d[lll]) * (mu / (mu+std::fabs(e[lll])));
                            if (mu<sminl)
                            {
                                sminl = mu;
                            }
                        }
                        if (loopbreak2)
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
                                  &Vt[ll/*+ldvt*0*/], ldvt)
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
                            * ((T(ZERO<=d[ll])-T(ZERO>d[ll])) + shift/d[ll]);
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
                        f = (std::fabs(d[m])-shift) * ((T(ZERO<=d[m])-T(ZERO>d[m]))+shift/d[m]);
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
            if (loopbreak1)
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
                    Blas<T>::dscal(ncvt, NEGONE, &Vt[i/*+lda*0*/], ldvt);
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
                    Blas<T>::dswap(ncvt, &Vt[isub/*+ldvt*0*/], ldvt, &Vt[n-i-1/*+ldvt*0*/], ldvt);
                }
                if (nru>0)
                {
                    Blas<T>::dswap(nru, &U[/*0+*/ldu*isub], 1, &U[/*0+*/ldu*(n-i-1)], 1);
                }
                if (ncc>0)
                {
                    Blas<T>::dswap(ncc, &C[isub/*+ldc*0*/], ldc, &C[n-i-1/*+ldc*0*/], ldc);
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
    static void dgebal(char const* job, int n, T* A, int lda, int& ilo, int& ihi, T* scale,
                       int& info)
    {
        const T sclfac = TWO;
        const T factor = T(0.95);
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
                        Blas<T>::dswap(l+1, &A[/*0+*/lda*j], 1, &A[/*0+*/lda*m], 1);
                        Blas<T>::dswap(n-k, &A[j+lda*k], lda, &A[m+lda*k], lda);
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
            T c, ca, f, g, r, ra, s, sfmax1, sfmax2, sfmin1, sfmin2;
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
                    c = Blas<T>::dnrm2(l-k+1, &A[k+lda*i], 1);
                    r = Blas<T>::dnrm2(l-k+1, &A[i+lda*k], lda);
                    ica = Blas<T>::idamax(l+1, &A[/*0+*/lda*i], 1);
                    ca = fabs(A[ica+lda*i]);
                    ira = Blas<T>::idamax(n-k, &A[i+lda*k], lda);
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
                    Blas<T>::dscal(n-k, g, &A[i+lda*k], lda);
                    Blas<T>::dscal(l+1, f, &A[/*0+*/lda*i], 1);
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
    static void dgebd2(int m, int n, T* A, int lda, T* d, T* e, T* tauq, T* taup, T* work,
                       int& info)
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
                itemp = ((i+2<m) ? i+1 : m-1);
                ldai = lda*i;
                ildai = i+ldai;
                dlarfg(m-i, A[ildai], &A[itemp+ldai], 1, tauq[i]);
                d[i] = A[ildai];
                A[ildai] = ONE;
                // Apply H(i) to A[i:m-1,i+1:n-1] from the left
                if (i<n-1)
                {
                    dlarf("Left", m-i, n-1-i, &A[ildai], 1, tauq[i], &A[ildai+lda], lda, &work);
                }
                A[ildai] = d[i];
                if (i<n-1)
                {
                    // Generate elementary reflector G(i) to annihilate A[i,i+2:n-1]
                    itemp = ((i+3<n) ? i+2 : n-1);
                    dlarfg(n-1-i, A[ildai+lda], &A[i+lda*itemp], lda, taup[i]);
                    e[i] = A[ildai+lda];
                    A[ildai+lda] = ONE;
                    // Apply G(i) to A[i+1:m-1,i+1:n-1] from the right
                    dlarf("Right", m-1-i, n-1-i, &A[ildai+lda], lda, taup[i], &A[1+ildai+lda], lda,
                          &work);
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
                itemp = ((i+2<n) ? i+1 : n-1);
                ldai = lda*i;
                ildai = i+ldai;
                dlarfg(n-i, A[ildai], &A[i+lda*itemp], lda, taup[i]);
                d[i] = A[ildai];
                A[ildai] = ONE;
                // Apply G(i) to A[i+1:m-1,i:n-1] from the right
                if (i<m-1)
                {
                    DLARF("Right", m-1-i, n-i, &A[ildai], lda, taup[i], &A[1+ildai], lda, &work);
                }
                A[ildai] = d[i];
                if (i<m-1)
                {
                    // Generate elementary reflector H(i) to annihilate A[i+2:m-1,i]
                    itemp = ((i+3<m) ? i+2 : m-1);
                    dlarfg(m-1-i, A[1+ildai], &A[itemp+ldai], 1, tauq[i]);
                    e[i] = A[1+ildai];
                    A[1+ildai] = ONE;
                    // Apply H(i) to A[i+1:m-1,i+1:n-1] from the left
                    dlarf("Left", m-1-i, n-1-i, &A[1+ildai], 1, tauq[i], &A[1+ildai+lda], lda,
                            &work);
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
    static void dgeqp3(int m, int n, T* A, int lda, int* jpvt, T* tau, T* work, int lwork,
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
            minmn = ((m<n) ? m : n);
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
                    Blas<T>::dswap(m, &A[/*0+*/lda*j], 1, &A[/*0+*/lda*(nfxd-1)], 1);
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
            int na = ((m<nfxd) ? m : nfxd);
            // dgeqr2(m, na, A, lda, tau, work, info);
            dgeqrf(m, na, A, lda, tau, work, lwork, info);
            iws = iws > int(work[0]) ? iws : int(work[0]);
            if (na<n)
            {
                dormqr("Left", "Transpose", m, n-na, na, A, lda, tau, &A[/*0+*/lda*na], lda, work,
                       lwork, info);
                iws = iws > int(work[0]) ? iws : int(work[0]);
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
                    iws = ((iws>minws) ? iws : minws);
                    if (lwork<minws)
                    {
                        // Not enough workspace to use optimal nb: Reduce nb and determine the minimum value of nb.
                        nb = (lwork-2*sn) / (sn+1);
                        nbmin = ilaenv(INBMIN, "DGEQRF", " ", sm, sn, -1, -1);
                        if (nbmin<2)
                        {
                            nbmin = 2;
                        }
                    }
                }
            }
            // Initialize partial column norms. The first n elements of work store the exact column norms.
            for (j=nfxd; j<n; j++)
            {
                work[j] = Blas<T>::dnrm2(sm, &A[nfxd+lda*j], 1);
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
    static void dgeqr2(int m, int n, T* A, int lda, T* tau, T* work, int& info)
    {
        int i, k, coli;
        T AII;
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
        k = ((m<n) ? m : n);
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
    static void dgeqrf(int m, int n, T* A, int lda, T* tau, T* work, int lwork, int& info)
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
        int k = ((m<n) ? m : n);
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
    static void dlabad(T& small, T& large)
    {
        // If it looks like we're on a Cray, take the square root of small and large to avoid
        // overflow and underflow problems.
        if (std::log10(large)>T(2000.0))
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
    static void dlacpy(char const* uplo, int m, int n, T const* A, int lda, T* B, int ldb)
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
    static T dlange(char const* norm, int m, int n, T const* A, int lda, T* work)
    {
        int i, j, ldacol;
        T scale, sum, dlange=ZERO, temp;
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
    static T dlapy2(T x, T Y)
    {
        T w, xabs, yabs, z;
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
    static void dlaqp2(int m, int n, int offset, T* A, int lda, int* jpvt, T* tau, T* vn1, T* vn2,
            T* work)
    {
        int mn = ((m-offset<n) ? m-offset : n);
        T tol3z = sqrt(dlamch("Epsilon"));
        // Compute factorization.
        int i, itemp, j, offpi, pvt, acoli;
        T aii, temp, temp2;
        for (i=0; i<mn; i++)
        {
            offpi = offset + i;
            // Determine ith pivot column and swap if necessary.
            pvt = i + Blas<T>::idamax(n-i, &vn1[i], 1);
            acoli = lda*i;
            if (pvt!=i)
            {
                Blas<T>::dswap(m, &A[lda*pvt], 1, &A[acoli], 1);
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
                    temp = ((temp>ZERO) ? temp : ZERO);
                    temp2 = vn1[j] / vn2[j];
                    temp2 = temp * temp2 * temp2;
                    if (temp2<=tol3z)
                    {
                        if (offpi<m-1)
                        {
                            vn1[j] = Blas<T>::dnrm2(m-offpi-1, &A[offpi+1+lda*j], 1);
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
    static void dlaqps(int m, int n, int offset, int nb, int& kb, T* A, int lda, int* jpvt, T* tau,
                       T* vn1, T* vn2, T* auxv, T* F, int ldf)
    {
        int lastrk = ((m<n+offset) ? m : n+offset);
        int lsticc = -1;
        int k = -1;
        T tol3z = sqrt(dlamch("Epsilon"));
        // Beginning of while loop.
        int itemp, j, pvt, rk, acolk;
        T akk, temp, temp2;
        while ((k+1<nb) && (lsticc==-1))
        {
            k++;
            rk = offset + k;
            acolk = lda*k;
            // Determine i-th pivot column and swap if necessary
            pvt = k + Blas<T>::idamax(n-k, &vn1[k], 1);
            if (pvt!=k)
            {
                Blas<T>::dswap(m, &A[lda*pvt], 1, &A[acolk], 1);
                Blas<T>::dswap(k, &F[pvt], ldf, &F[k], ldf);
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
                Blas<T>::dgemv("No transpose", m-rk, k, -ONE, &A[rk], lda, &F[k], ldf, ONE,
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
                Blas<T>::dgemv("Transpose", m-rk, n-k-1, tau[k], &A[rk+acolk+lda], lda,
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
                Blas<T>::dgemv("Transpose", m-rk, k, -tau[k], &A[rk], lda, &A[rk+acolk], 1, ZERO,
                               auxv, 1);
                Blas<T>::dgemv("No transpose", n, k, ONE, F, ldf, auxv, 1, ONE, &F[ldf*k], 1);
            }
            // Update the current row of A:
            // A[rk, k+1:n-1] -= A[rk, 0:k] * F[k+1:n-1, 0:k]^T.
            if (k<n-1)
            {
                Blas<T>::dgemv("No transpose", n-k-1, k+1, -ONE, &F[k+1], ldf, &A[rk], lda,
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
                        temp = ((ZERO>temp) ? ZERO : temp);
                        temp2 = vn1[j] / vn2[j];
                        temp2 = temp * temp2 * temp2;
                        if (temp2<=tol3z)
                        {
                            vn2[j] = T(lsticc+1);
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
            Blas<T>::dgemm("No transpose", "Transpose", m-rk-1, n-kb, kb, -ONE, &A[rk+1], lda,
                           &F[kb], ldf, ONE, &A[rk+1+lda*kb], lda);
        }
        // Recomputation of difficult columns.
        while (lsticc>=0)
        {

            itemp = int(vn2[lsticc]-HALF); // round vn2[lsticc]-1
            vn1[lsticc] = Blas<T>::dnrm2(m-rk-1, &A[rk+1+lda*lsticc], 1);
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
     *          NAG Ltd.                                                                      */
    static void dlarf(char const* side, int m, int n, T const* v, int incv, T tau, T* C, int ldc,
                      T* work)
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
                Blas<T>::dgemv("Transpose", lastv, lastc, ONE, C, ldc, v, incv, ZERO, work, 1);
                // C[0:lastv-1,0:lastc-1] -= v[0:lastv-1] * work[0:lastc-1]^T
                Blas<T>::dger(lastv, lastc, -tau, v, incv, work, 1, C, ldc);
            }
        }
        else
        {
            // Form  C * H
            if (lastv>0)
            {
                // work[0:lastc-1] = C[0:lastc-1,0:lastv-1] * v[0:lastv-1]
                Blas<T>::dgemv("No transpose", lastc, lastv, ONE, C, ldc, v, incv, ZERO, work, 1);
                // C[0:lastc-1,0:lastv-1] -= work[0:lastc-1] * v[0:lastv-1]^T
                Blas<T>::dger(lastc, lastv, -tau, work, 1, v, incv, C, ldc);
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
     *             k: The order of the matrix Tm(= the number of elementary reflectors
     *                whose product defines the block reflector).
     *             V: an array, dimension (ldv, k) if storev = 'C'
     *                                    (ldv, m) if storev = 'R' and side = 'L'
     *                                    (ldv, n) if storev = 'R' and side = 'R'
     *             ldv: he leading dimension of the array V.
     *                  If storev = 'C' and side = 'L', ldv >= max(1, m);
     *                  if storev = 'C' and side = 'R', ldv >= max(1, n);
     *                  if storev = 'R', ldv >= k.
     *             Tm: an array, dimension(ldt, k)
     *                The triangular k by k matrix Tm in the representation of the block reflector.
     *             ldt: The leading dimension of the array Tm. ldt >= k.
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
                       int m, int n, int k, T* V, int ldv, T const* Tm, int ldt, T* C, int ldc,
                       T* Work, int ldwork)
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
                        Blas<T>::dcopy(n, &C[j], ldc, &Work[ldwork*j], 1);
                    }
                    // W : = W * V1
                    Blas<T>::dtrmm("Right", "Lower", "No transpose", "Unit", n, k, ONE, V, ldv,
                                   Work, ldwork);
                    if (m>k)
                    {
                        // W = W + C2^T * V2
                        Blas<T>::dgemm("Transpose", "No transpose", n, k, m-k, ONE, &C[k], ldc,
                                       &V[k], ldv, ONE, Work, ldwork);
                    }
                    // W = W * Tm^T or W * Tm
                    Blas<T>::dtrmm("Right", "Upper", transt, "Non-unit", n, k, ONE, Tm, ldt, Work,
                                   ldwork);
                    // C = C - V * W^T
                    if (m>k)
                    {
                        // C2 = C2 - V2 * W^T
                        Blas<T>::dgemm("No transpose", "Transpose", m-k, n, k, -ONE, &V[k], ldv,
                                       Work, ldwork, ONE, &C[k], ldc);
                    }
                    // W = W * V1^T
                    Blas<T>::dtrmm("Right", "Lower", "Transpose", "Unit", n, k, ONE, V, ldv, Work,
                                   ldwork);
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
                        Blas<T>::dcopy(m, &C[ldc*j], 1, &Work[ldwork*j], 1);
                    }
                    // W = W * V1
                    Blas<T>::dtrmm("Right", "Lower", "No transpose", "Unit", m, k, ONE, V, ldv,
                                   Work, ldwork);
                    if (n>k)
                    {
                        // W = W + C2 * V2
                        Blas<T>::dgemm("No transpose", "No transpose", m, k, n-k, ONE, &C[ldc*k],
                                       ldc, &V[k], ldv, ONE, Work, ldwork);
                    }
                    // W = W * Tm or W * Tm^T
                    Blas<T>::dtrmm("Right", "Upper", trans, "Non-unit", m, k, ONE, Tm, ldt, Work,
                                   ldwork);
                    // C = C - W * V^T
                    if (n>k)
                    {
                        // C2 = C2 - W * V2^T
                        Blas<T>::dgemm("No transpose", "Transpose", m, n-k, k, -ONE, Work, ldwork,
                                       &V[k], ldv, ONE, &C[ldc*k], ldc);
                    }
                    // W = W * V1^T
                    Blas<T>::dtrmm("Right", "Lower", "Transpose", "Unit", m, k, ONE, V, ldv, Work,
                                   ldwork);
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
                        Blas<T>::dcopy(n, &C[m-k+j], ldc, &Work[ldwork*j], 1);
                    }
                    // W = W * V2
                    Blas<T>::dtrmm("Right", "Upper", "No transpose", "Unit", n, k, ONE, &V[m-k],
                                   ldv, Work, ldwork);
                    if (m>k)
                    {
                        // W = W + C1^T * V1
                        Blas<T>::dgemm("Transpose", "No transpose", n, k, m-k, ONE, C, ldc, V, ldv,
                                ONE, Work, ldwork);
                    }
                    // W = W * Tm^T or W * Tm
                    Blas<T>::dtrmm("Right", "Lower", transt, "Non-unit", n, k, ONE, Tm, ldt, Work,
                                   ldwork);
                    // C = C - V * W^T
                    if (m>k)
                    {
                        // C1 = C1 - V1 * W^T
                        Blas<T>::dgemm("No transpose", "Transpose", m-k, n, k, -ONE, V, ldv, Work,
                                ldwork, ONE, C, ldc);
                    }
                    // W = W * V2^T
                    Blas<T>::dtrmm("Right", "Upper", "Transpose", "Unit", n, k, ONE, &V[m-k], ldv,
                                   Work, ldwork);
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
                        Blas<T>::dcopy(m, &C[ccol+ldc*j], 1, &Work[ldwork*j], 1);
                    }
                    // W = W * V2
                    Blas<T>::dtrmm("Right", "Upper", "No transpose", "Unit", m, k, ONE, &V[n-k],
                                   ldv, Work, ldwork);
                    if (n>k)
                    {
                        // W = W + C1 * V1
                        Blas<T>::dgemm("No transpose", "No transpose", m, k, n-k, ONE, C, ldc, V,
                                       ldv, ONE, Work, ldwork);
                    }
                    // W : = W * Tm or W * Tm^T
                    Blas<T>::dtrmm("Right", "Lower", trans, "Non-unit", m, k, ONE, Tm, ldt, Work,
                                   ldwork);
                    // C = C - W * V^T
                    if (n>k)
                    {
                        // C1 = C1 - W * V1^T
                        Blas<T>::dgemm("No transpose", "Transpose", m, n-k, k, -ONE, Work, ldwork,
                                       V, ldv, ONE, C, ldc);
                    }
                    // W = W * V2^T
                    Blas<T>::dtrmm("Right", "Upper", "Transpose", "Unit", m, k, ONE, &V[n-k], ldv,
                                   Work, ldwork);
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
                        Blas<T>::dcopy(n, &C[j], ldc, &Work[ldwork*j], 1);
                    }
                    // W = W * V1^T
                    Blas<T>::dtrmm("Right", "Upper", "Transpose", "Unit", n, k, ONE, V, ldv, Work,
                                   ldwork);
                    if (m>k)
                    {
                        // W = W + C2^T * V2^T
                        Blas<T>::dgemm("Transpose", "Transpose", n, k, m-k, ONE, &C[k], ldc,
                                       &V[ldv*k], ldv, ONE, Work, ldwork);
                    }
                    // W = W * Tm^T or W * Tm
                    Blas<T>::dtrmm("Right", "Upper", transt, "Non-unit", n, k, ONE, Tm, ldt, Work,
                                   ldwork);
                    // C = C - V^T * W^T
                    if (m>k)
                    {
                        // C2 = C2 - V2^T * W^T
                        Blas<T>::dgemm("Transpose", "Transpose", m-k, n, k, -ONE, &V[ldv*k], ldv,
                                       Work, ldwork, ONE, &C[k], ldc);
                    }
                    // W = W * V1
                    Blas<T>::dtrmm("Right", "Upper", "No transpose", "Unit", n, k, ONE, V, ldv,
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
                        Blas<T>::dcopy(m, &C[ldc*j], 1, &Work[ldwork*j], 1);
                    }
                    // W = W * V1^T
                    Blas<T>::dtrmm("Right", "Upper", "Transpose", "Unit", m, k, ONE, V, ldv, Work,
                                   ldwork);
                    if (n>k)
                    {
                        // W = W + C2 * V2^T
                        Blas<T>::dgemm("No transpose", "Transpose", m, k, n-k, ONE, &C[ldc*k], ldc,
                                        &V[ldv*k], ldv, ONE, Work, ldwork);
                    }
                    // W = W * Tm or W * Tm^T
                    Blas<T>::dtrmm("Right", "Upper", trans, "Non-unit", m, k, ONE, Tm, ldt, Work,
                                   ldwork);
                    // C = C - W * V
                    if (n>k)
                    {
                        // C2 = C2 - W * V2
                        Blas<T>::dgemm("No transpose", "No transpose", m, n-k, k, -ONE, Work,
                                       ldwork, &V[ldv*k], ldv, ONE, &C[ldc*k], ldc);
                    }
                    // W = W * V1
                    Blas<T>::dtrmm("Right", "Upper", "No transpose", "Unit", m, k, ONE, V, ldv,
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
                        Blas<T>::dcopy(n, &C[m-k+j], ldc, &Work[ldwork*j], 1);
                    }
                    // W = W * V2^T
                    Blas<T>::dtrmm("Right", "Lower", "Transpose", "Unit", n, k, ONE, &V[ldv*(m-k)],
                                   ldv, Work, ldwork);
                    if (m>k)
                    {
                        // W = W + C1^T * V1^T
                        Blas<T>::dgemm("Transpose", "Transpose", n, k, m-k, ONE, C, ldc, V, ldv,
                                       ONE, Work, ldwork);
                    }
                    // W = W * Tm^T or W * Tm
                    Blas<T>::dtrmm("Right", "Lower", transt, "Non-unit", n, k, ONE, Tm, ldt, Work,
                                   ldwork);
                    // C = C - V^T * W^T
                    if (m>k)
                    {
                        // C1 = C1 - V1^T * W^T
                        Blas<T>::dgemm("Transpose", "Transpose", m-k, n, k, -ONE, V, ldv, Work,
                                       ldwork, ONE, C, ldc);
                    }
                    // W = W * V2
                    Blas<T>::dtrmm("Right", "Lower", "No transpose", "Unit", n, k, ONE,
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
                        Blas<T>::dcopy(m, &C[ccol+ldc*j], 1, &Work[ldwork*j], 1);
                    }
                    // W = W * V2^T
                    Blas<T>::dtrmm("Right", "Lower", "Transpose", "Unit", m, k, ONE, &V[ldv*(n-k)],
                                   ldv, Work, ldwork);
                    if (n>k)
                    {
                        // W = W + C1 * V1^T
                        Blas<T>::dgemm("No transpose", "Transpose", m, k, n-k, ONE, C, ldc, V, ldv,
                                       ONE, Work, ldwork);
                    }
                    // W = W * Tm or W * Tm^T
                    Blas<T>::dtrmm("Right", "Lower", trans, "Non-unit", m, k, ONE, Tm, ldt, Work,
                                   ldwork);
                    // C = C - W * V
                    if (n>k)
                    {
                        // C1 = C1 - W * V1
                        Blas<T>::dgemm("No transpose", "No transpose", m, n-k, k, -ONE, Work,
                                       ldwork, V, ldv, ONE, C, ldc);
                    }
                    // W = W * V2
                    Blas<T>::dtrmm("Right", "Lower", "No transpose", "Unit", m, k, ONE,
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
    static void dlarfg(int n, T& alpha, T* x, int incx, T& tau)
    {
        int j, knt;
        T beta, rsafmin, safmin, xnorm;
        if (n<=1)
        {
            tau = ZERO;
            return;
        }
        xnorm = Blas<T>::dnrm2(n-1, x, incx);
        if (xnorm==ZERO)
        {
            // H = I
            tau = ZERO;
        }
        else
        {
            // general case
            beta = -dlapy2(alpha, xnorm) * T((ZERO<=alpha) - (alpha<ZERO));
            safmin = dlamch("SafeMin") / dlamch("Epsilon");
            knt = 0;
            if (fabs(beta)<safmin)
            {
                // xnorm, beta may be inaccurate; scale x and recompute them
                rsafmin = ONE / safmin;
                do
                {
                    knt++;
                    Blas<T>::dscal(n-1, rsafmin, x, incx);
                    beta *= rsafmin;
                    alpha *= rsafmin;
                } while (fabs(beta)<safmin);
                // New beta is at most 1, at least SAFMIN
                xnorm = Blas<T>::dnrm2(n-1, x, incx);
                beta = -dlapy2(alpha, xnorm) * T((ZERO<=alpha) - (alpha<ZERO));
            }
            tau = (beta-alpha) / beta;
            Blas<T>::dscal(n-1, ONE/(alpha-beta), x, incx);
            // If alpha is subnormal, it may lose relative accuracy
            for (j=0; j<knt; j++)
            {
                beta *= safmin;
            }
            alpha = beta;
        }
    }

    /* dlarft forms the triangular factor A of a real block reflector H
     * of order n, which is defined as a product of k elementary reflectors.
     *     If direct = 'F', H = H(1) H(2) . ..H(k) and A is upper triangular;
     *     If direct = 'B', H = H(k) . ..H(2) H(1) and A is lower triangular.
     *     If storev = 'C', the vector which defines the elementary reflector
     *                      H(i) is stored in the i - th column of the array V, and
     *                      H = I - V * A * V^T
     *     If storev = 'R', the vector which defines the elementary reflector
     *                      H(i) is stored in the i - th row of the array V, and
     *                      H = I - V^T * A * V
     * Parameters: direct:  Specifies the order in which the elementary reflectors are
     *                      multiplied to form the block reflector:
     *                      'F': H = H(1) H(2) . ..H(k) (Forward)
     *                      'B': H = H(k) . ..H(2) H(1) (Backward)
     *             storev: Specifies how the vectors which define the elementary
     *                     reflectors are stored(see also Further Details) :
     *                     'C': columnwise
     *                     'R': rowwise
     *             n: The order of the block reflector H. n >= 0.
     *             k: The order of the triangular factor A(= the number of elementary reflectors).
     *                k >= 1.
     *             V: an array, dimension (ldv, k) if storev = 'C'
     *                                    (ldv, n) if storev = 'R'
     *             ldv: The leading dimension of the array V.
     *                  If storev = 'C', ldv >= max(1, n); if storev = 'R', ldv >= k.
     *             tau: an array, dimension(k)
     *                  tau(i) must contain the scalar factor of the elementary reflector H(i).
     *             A: an array, dimension(lda, k)
     *                The k by k triangular factor A of the block reflector.
     *                If direct=='F', A is upper triangular;
     *                if direct=='B', A is lower triangular. The rest of the array is not used.
     *             lda: The leading dimension of the array A. lda >= k.
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
    static void dlarft(char const* direct, char const* storev, int n, int k, T const* V, int ldv,
                       T const* tau, T* A, int lda)
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
                        A[j+tcoli] = ZERO;
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
                            A[j+tcoli] = -tau[i]*V[i+ldv*j];
                        }
                        j = ((lastv<prevlastv) ? lastv : prevlastv);
                        // T[0:i-1, i] = -tau[i] * V[i:j, 0:i-1]^T * V[i:j, i]
                        Blas<T>::dgemv("Transpose", j-i, i, -tau[i], &V[i+1], ldv, &V[i+1+vcol], 1,
                                       ONE, &A[tcoli], 1);
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
                            A[j+tcoli] = -tau[i]*V[j+vcol];
                        }
                        j = ((lastv<prevlastv) ? lastv : prevlastv);
                        // T[0:i-1, i] = -tau[i] * V[0:i-1, i:j] * V[i, i:j]^T
                        Blas<T>::dgemv("No transpose", i, j-i, -tau[i], &V[vcol+ldv], ldv,
                                       &V[i+vcol+ldv], ldv, ONE, &A[tcoli], 1);
                    }
                    // T[0:i-1, i] = T[0:i-1, 0:i-1] * T[0:i-1, i]
                    Blas<T>::dtrmv("Upper", "No transpose", "Non-unit", i, A, lda, &A[tcoli], 1);
                    A[i+tcoli] = tau[i];
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
                        A[j+tcoli] = ZERO;
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
                                A[j+tcoli] = -tau[i] * V[vcol+ldv*j];
                            }
                            j = ((lastv>prevlastv) ? lastv : prevlastv);
                            // T[i+1:k-1, i] = -tau[i] * V[j:n-k+i, i+1:k-1]^T * V[j:n-k+i, i]
                            dgemv("Transpose", vcol-j, k-1-i, -tau[i], &V[j+ldv*(i+1)], ldv,
                                  &V[j+ldv*i], 1, ONE, &A[i+1+tcoli], 1);
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
                                A[j+tcoli] = -tau[i]*V[j+vcol];
                            }
                            j = ((lastv>prevlastv) ? lastv : prevlastv);
                            // T[i+1:k-1, i] = -tau[i] * V[i+1:k-1, j:n-k+i] * V[i, j:n-k+i]^T
                            vcol = ldv*j;
                            dgemv("No transpose", k-1-i, n-k+i-j, -tau[i], &V[i+1+vcol], ldv,
                                  &V[i+vcol], ldv, ONE, &A[i+1+tcoli], 1);
                        }
                        // T[i+1:k-1, i] = T[i+1:k-1, i+1:k-1] * T[i+1:k-1, i]
                        Blas<T>::dtrmv("Lower", "No transpose", "Non-unit", k-i-1,
                                       &A[i+1+lda*(i+1)], lda, &A[i+1+tcoli], 1);
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
                    A[i+tcoli] = tau[i];
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
    static void dlarnv(int idist, int* iseed, int n, T* x)
    {
        const T TWOPI = T(6.2831853071795864769252867663);
        const int LV = 128;
        int i, il, il2, iv;
        T* u[LV];
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
    static void dlartg(T f, T g, T& cs, T& sn, T& r)
    {
        //static bool first = true;
        /*static*/ T safmn2, safmx2;
        //if (first)
        //{
        T safmin = dlamch("S");
        T eps = dlamch("E");
        T base = dlamch("B");
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
            T f1 = f;
            T g1 = g;
            T scale = fabs(f1);
            scale = ((scale>fabs(g1)) ? scale : fabs(g1));
            int i, count = 0;
            if (scale>=safmx2)
            {
                do
                {
                    count++;
                    f1 *= safmn2;
                    g1 *= safmn2;
                    scale = fabs(f1);
                    scale = ((scale>fabs(g1)) ? scale : fabs(g1));
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
                    f1 *= safmx2
                    g1 *= safmx2
                    scale = fabs(f1);
                    scale = ((scale>fabs(g1)) ? scale : fabs(g1));
                } while (scale<=safmn2)
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
    static void dlaruv(int* iseed, int n, T* x)
    {
        const int LV = 128;
        const int IPW2 = 4096;
        const T R = ONE / IPW2;
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
        int i, imul4, it1, it2, it3, it4;
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
                x[i] = R*(T(it1)+R*(T(it2)+R*(T(it3)+R*T(it4))));
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
            } while (x[i]==ONE)
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
    static void dlas2(T f, T g, T h, T& ssmin, T& ssmax)
    {
        T fa = fabs(f);
        T ga = fabs(g);
        T ha = fabs(h);
        T fhmn, fhmx;
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
            T as, at, au, c;
            if (ga<fhmx)
            {
                as = ONE + fhmn/fhmx;
                at = (fhmx-fhmn) / fhmx;
                au = ga / fhmx;
                au = au*au;
                c = TWO / (srd::sqrt(as*as+au)+std::sqrt(at*at+au));
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
     * without over/underflow as long as the final result cto*A(I,J)/cfrom does not over/underflow.
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
     * Date June 2016                                                                            */
    static void dlascl(char const* type, int kl, int ku, T cfrom, T cto, int m, int n, T* A,
                       int lda, int& info)
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
        T smlnum = dlamch("S");
        T bignum = ONE / smlnum;
        T cfromc = cfrom;
        T ctoc = cto;
        int i, j, ldaj, k1, k2, k3, k4;
        T cfrom1, cto1, mul;
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
        } while (!done)
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
    static void dlaset(char const* uplo, int m, int n, T alpha, T beta, T* A, int lda)
    {
        int i, j, ldaj;
        if (toupper(uplo[0])=='U')
        {
            // Set the strictly upper triangular or trapezoidal part of the array to ALPHA.
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
            // Set the strictly lower triangular or trapezoidal part of the array to ALPHA.
            for (j=0, j<m && j<n; j++)
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
            // Set the leading m-by-n submatrix to ALPHA.
            for (j=0; j<n; j++)
            {
                ldaj = lda*j;
                for (i=0; i<m; i++)
                {
                    A[i+ldaj] = alpha;
                }
            }
        }
        // Set the first min(M,N) diagonal elements to BETA.
        for (i=0, i<m && i<n; i++)
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
    static void dlasq1(int n, T* d, T* e, T* work, int& info)
    {
        int i, iinfo;
        T eps, scale, safmin, sigmn, sigmx, temp;
        info = 0
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
            temp = fabs(e[i]));
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
        Blas<T>::dcopy(n, d, 1, work[0], 2);
        Blas<T>::dcopy(n-1, e, 1, work[1], 2);
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
    static void dlasq2(int n, int Z, int& info)
    {
        const T CBIAS = T(1.50);
        const T FOUR  = T(4.0);
        bool ieee, loopbreak;
        int i0, i1, i4, iinfo, ipn4, iter, iwhila, iwhilb, k, kmin, n0, n1, nbig, ndiv, nfail, pp,
            splt, ttype;
        T d, dee, deemin, desig, dmin, dmin1, dmin2, dn, dn1, dn2, e, emax, emin, eps, g, oldemn,
          qmax, qmin, s, safmin, sigma, tt, tau, temp, tol, tol2, trace, zmax, tempe, tempq;
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
            qmax = ((qmax>Z[k])   ? qmax : Z[k]);
            emin = ((emin<Z[k+1]) ? emin : Z[k+1]);
            zmax = ((qmax>zmax)   ? qmax : zmax);
            zmax = ((zmax>Z[k+1]) ? zmax : Z[k+1]);
        }
        if (Z[2*n-2]<ZERO)
        {
            info = -(200+2*n-1);
            xerbla("DLASQ2", 2);
            return;
        }
        d += Z[2*n-2];
        qmax = ((qmax>Z[2*n-2]) ? qmax : Z[2*n-2]);
        zmax = ((qmax>zmax)     ? qmax : zmax);
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
        for (k=2*n-1; k>=1, k-=2)
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
                    Z[i4-2*pp+3] = Z[i4+4] * (Z[i4+2]/Z(i4-2*pp+2));
                    d = Z[i4+4] * (d/Z[i4-2*pp+1]);
                }
                if (emin>Z[i4-2*pp+3])
                {
                    emin = Z[i4-2*pp+3];
                }
            }
            Z[4*n0-pp+1] = d;
            // Now find qmax.
            qmax = Z[4*n0-pp+1];
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
        NDIV = 2*(n0-i0);
        for (iwhila=0; iwhila<=n; iwhila++)
        {
            if (n0<0)
            {
                break;
            }
            // While array unfinished do
            // E[n0] holds the value of SIGMA when submatrix in i0:n0 splits from the rest of the
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
            // Find Gershgorin-type bound if Q's much greater than E's.
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
                    qmin = ((qmin<Z[i4])   ? qmin : Z[i4]);
                    emax = ((emax>Z[i4-2]) ? emax : Z[i4-2]);
                }
                temp = Z[i4-4] + Z[i4-2];
                qmax = ((qmax>temp)    ? qmax : temp);
                emin = ((emin<Z[i4-2]) ? emin : Z[i4-2]);
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
                dmin = ZERO;
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
                dlasq3(i0, n0, Z, pp, dmin, sigma, desig, qmax, nfail, iter, NDIV, ieee, ttype,
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
                                qmax   = ((qmax>Z[i4+4])   ? qmax   : Z[i4+4]);
                                emin   = ((emin<Z[i4+2])   ? emin   : Z[i4+2]);
                                oldemn = ((oldemn<Z[i4+3]) ? oldemn : Z[i4+3]);
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
        Z[2*n+2] = T(iter);
        Z[2*n+3] = T(NDIV) / T(n*n);
        Z[2*n+4] = HNDRD * nfail / T(iter);
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
    static void dlasq3(int i0, int& n0, T* Z, int& pp, T& dmin, T& sigma, T& desig, T qmax,
                       int& nfail, int& iter, int& ndiv, bool ieee, int& ttype, T& dmin1, T& dmin2,
                       T& dn, T& dn1, T& dn2, T& g, T& tau)
    {
        const T CBIAS = T(1.50);
        int n0in = n0;
        T eps = dlamch("Precision");
        T tol = eps*HNDRD;
        T tol2 = tol*tol;
        int ipn4, j4, nn;
        T s, tt, temp;
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
                // Check whether E[n0-1] is negligible, 1 eigenvalue.
                if (Z[nn-6]<=tol2*(sigma+Z[nn-4]) || Z[nn-2*pp-5]<=tol2*Z[nn-8])
                {
                    Z[4*n0] = Z[4*n0+pp] + sigma;
                    n0--;
                    continue;
                }
                // Check  whether E[n0-2] is negligible, 2 eigenvalues.
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
                temp = Z[4*n0+pp+2];
                dmin2 = ((dmin2<temp) ? dmin2 : temp);
                temp = Z[4*n0+pp+2];
                temp = ((temp<Z[4*i0+pp+2]) ? temp : Z[4*i0+pp+2]);
                Z[4*n0+pp+2] = ((temp<Z[4*i0+pp+6]) ? temp : Z[4*i0+pp+6]);
                temp = Z[4*n0-pp+3];
                temp = ((temp<Z[4*i0-pp+3]) ? temp : Z[4*i0-pp+3]);
                Z[4*n0-pp+3] = ((temp<Z[4*i0-pp+7]) ? temp : Z[4*i0-pp+7]);
                temp = Z[4*i0+pp];
                temp = ((qmax>temp) ? qmax : temp);
                qmax = ((temp>Z[4*i0+pp+4]) ? temp : Z[4*i0+pp+4]);
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
                // TAU too big. Select new TAU and try again.
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
    static void dlasq4(int i0, int n0, T const* Z, int pp, int n0in, T dmin, T dmin1, T dmin2,
                       T dn, T dn1, T dn2, T& tau, int& ttype, T& g)
    {
        const T CNST1 = T(0.563);
        const T CNST2 = T(1.01);
        const T CNST3 = T(1.05);
        const T THIRD = T(0.333);
        // A negative dmin forces the shift to take that absolute value
        // dn2 records the type of shift.
        if (dmin<=ZERO)
        {
            ttype = -1;
            return;
        }
        int nn = 4*(n0+1) + pp;
        int i4, np;
        T a2, b1, b2, gam, gap1, gap2, s, temp;
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
                    // Approximate contribution to norm squared from I<nn-1.
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
                // Compute contribution to norm squared from I>nn-2.
                np = nn - 2*pp - 1;
                b1 = Z[np-2];
                b2 = Z[np-6];
                gam = dn2;
                if (Z[np-8]>b2 || Z[np-4]>b1)
                {
                    return;
                }
                a2 = (Z[np-8]/b2) * (ONE+Z[np-4]/b1);
                // Approximate contribution to norm squared from I<nn-2.
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
    static void dlasq5(int i0, int n0, T* Z, int pp, T tau, T sigma, T& dmin, T& dmin1, T& dmin2,
                       T& dn, T& dnm1, T& dnm2, bool ieee, T eps)
    {
        if ((n0-i0-1)<=0)
        {
            return;
        }
        int j4, j4p2;
        T d, emin, temp, dthresh;
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
                        dmin = ((dmin<d) ? dmin : d);
                        Z[j4] = Z[j4-1]*temp;
                        emin = ((Z[j4]<emin) ? Z[j4] : emin);
                    }
                }
                else
                {
                    for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
                    {
                        Z[j4-3] = d + Z[j4];
                        temp = Z[j4+2] / Z[j4-3];
                        d = d*temp - tau;
                        dmin = ((dmin<d) ? dmin : d);
                        Z[j4-1] = Z[j4]*temp;
                        emin = ((Z[j4-1]<emin) ? Z[j4-1] : emin);
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
                dmin = ((dmin<dnm1) ? dmin : dnm1);
                dmin1 = dmin;
                j4 += 4;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm1 + Z[j4p2];
                Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                dn = Z[j4p2+2]*(dnm1/Z[j4-2]) - tau;
                dmin = ((dmin<dn) ? dmin : dn);
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
                        dmin = ((dmin<d) ? dmin : d);
                        emin = ((emin<Z[j4]) ? emin : Z[j4]);
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
                        dmin = ((dmin<d) ? dmin : d);
                        emin = ((emin<Z[j4-1]) ? emin : Z[j4-1]);
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
                dmin = ((dmin<dnm1) ? dmin : dnm1);
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
                dmin = ((dmin<dn) ? dmin : dn);
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
                    for (j4=4*i0+3; j4<4*(n0-2), j4+=4)
                    {
                        Z[j4-2] = d + Z[j4-1];
                        temp = Z[j4+1] / Z[j4-2];
                        d = d*temp - tau;
                        if (d<dthresh)
                        {
                            d = ZERO;
                        }
                        dmin = ((dmin<d) ? dmin : d);
                        Z[j4] = Z[j4-1]*temp;
                        emin = ((Z[j4]<emin) ? Z[j4] : emin);
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
                        dmin = ((dmin<d) ? dmin : d);
                        Z[j4-1] = Z[j4]*temp;
                        emin = ((Z[j4-1]<emin) ? Z[j4-1] : emin);
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
                dmin = ((dmin<dnm1) ? dmin : dnm1);
                dmin1 = dmin;
                j4 += 4;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm1 + Z[j4p2];
                Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                dn = Z[j4p2+2]*(dnm1/Z[j4-2]) - tau;
                dmin = ((dmin<dn) ? dmin : dn);
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
                        dmin = ((dmin<d) ? dmin : d);
                        emin = ((Z[j4]<emin) ? Z[j4] : emin);
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
                        dmin = ((dmin<d) ? dmin : d);
                        emin = ((Z[j4-1]<emin) ? Z[j4-1] : emin);
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
                dmin = ((dmin<dnm1) ? dmin : dnm1);
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
                dmin = ((dmin<dn) ? dmin : dn);
            }
        }
        Z[j4+2] = dn;
        Z(4*(n0+1)-pp) = emin;
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
    static void dlasq6(int i0, int n0, T* Z, int pp, T& dmin, T& dmin1, T& dmin2, T& dn, T& dnm1,
                       T& dnm2)
    {
        if ((n0-i0-1)<=0)
        {
            return;
        }
        T safmin = dlamch("Safe minimum");
        int j4 = 4*i0 + pp + 1;
        T emin = Z[j4+4];
        T d = Z[j4];
        dmin = d;
        int j4p2;
        T temp;
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
                dmin = ((dmin<d) ? dmin : d);
                emin = ((emin<Z[j4]) ? emin : Z[j4]);
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
                dmin = ((dmin<d) ? dmin : d);
                emin = ((emin<Z[j4-1]) ? emin : Z[j4-1]);
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
        dmin = ((dmin<dnm1) ? dmin : dnm1);
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
        dmin = ((dmin<dn) ? dmin : dn);
        Z[j4+2] = dn;
        Z(4*n0+4-pp) = emin;
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
                      T const* c, T const* s, T* A, int lda)
    {
        int i, info, j, aind1, aind2, aind3, aind4;
        T ctemp, stemp, temp;
        char upside = std::toupper(side[0]);
        char uppivot = std::toupper(pivot[0]);
        char updirect = std::toupper(direct[0]);
        // Test the input parameters
        info = 0
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
        else if (lda<MAX(1, m))
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
    static void dlasrt(char const* id, int n, T* d, int& info)
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
        T d1, d2, d3, dmnmx, tmp;
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
    static void dlassq(int n, T const* x, int incx, int& scale, int& sumsq)
    {
        if (n>0)
        {
            T absxi, temp;
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
    static void dlasv2(T f, T g, T h, T& ssmin, T& ssmax, T& snr, T& csr, T& snl, T& csl)
    {
        bool gasmal, swap;
        int pmax;
        T a, clt, crt, d, fa, ft, ga, gt, ha, ht, l, m, mm, r, s, slt, srt, t, temp, tsign, tt;
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
                a = T(0.5) * (s+r);
                // Note that 1 <= a <= 1 + abs(M)
                ssmin = ha / a;
                ssmax = fa * a;
                if (mm==ZERO)
                {
                    // Note that m is very tiny
                    if (l==ZERO)
                    {
                        t = T((ZERO<=ft)-(ft<ZERO)) * TWO * T((ZERO<=gt)-(gt<ZERO));
                    }
                    else
                    {
                        t = gt / (T((ZERO<=ft)-(ft<ZERO))*std::fabs(d)) + m / t;
                    }
                }
                else
                {
                    t = (m/(s+t) + m/(r+l)) * (ONE + a);
                }
                l = std::sqrt(t*t+T(4.0));
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
            tsign = T((ZERO<=csr)-(csr<ZERO))*T((ZERO<=csl)-(csl<ZERO))*T((ZERO<=f)-(f<ZERO));
        }
        if (pmax==1)
        {
            tsign = T((ZERO<=snr)-(snr<ZERO))*T((ZERO<=csl)-(csl<ZERO))*T((ZERO<=g)-(g<ZERO));
        }
        if (pmax==2)
        {
            tsign = T((ZERO<=snr)-(snr<ZERO))*T((ZERO<=snl)-(snl<ZERO))*T((ZERO<=h)-(h<ZERO));
        }
        ssmax = tsign*std::fabs(ssmax);
        ssmin = tsign*T((ZERO<=f)-(f<ZERO))*T((ZERO<=h)-(h<ZERO))*std::fabs(ssmin);
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
    static void dorg2r(int m, int n, int k, T* A, int lda, T const* tau, T* work, int& info)
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
                Blas<T>::dscal(m-i-1, -tau[i], &A[i+1+acoli], 1);
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
    static void dorgqr(int m, int n, int k, T* A, int lda, T const* tau, T* work, int lwork,
                       int& info)
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
    static void dorm2r(char const* side, char const* trans, int m, int n, int k, T* A, int lda,
                       T const* tau, T* C, int ldc, T* work, int& info)
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
        T aii;
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
    static void dormqr(char const* side, char const* trans, int m, int n, int k, T* A, int lda,
                       T const* tau, T* C, int ldc, T* work, int lwork, int& info)
    {
        const int NBMAX = 64;
        const int LDT = NBMAX + 1;
        T* Tm = new T[LDT*NBMAX];
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
            {
                nb = ilaenv(1, "DORMQR", opts, m, n, k, -1);
            }
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
            for (i=i1; i!=i2+i3; i+=i3)
            {
                ib = k - i;
                if (ib>nb)
                {
                    ib = nb;
                }
                // Form the triangular factor of the block reflector
                aind = i + lda*i;
                //     H = H(i) H(i + 1) . ..H(i + ib - 1)
                dlarft("Forward", "Columnwise", nq-i, ib, &A[aind], lda, &tau[i], Tm, LDT);
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
                dlarfb(side, trans, "Forward", "Columnwise", mi, ni, ib, &A[aind], lda, Tm, LDT,
                       &C[ic+ldc*jc], ldc, work, ldwork);
            }
        }
        work[0] = lwkopt;
        delete[] Tm;
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
    static int ieeeck(int ispec, T Zero, T One)
    {
        T posinf = One / Zero;
        if (posinf<One)
        {
            return 0;
        }
        T neginf = -One / Zero;
        if (neginf>=Zero)
        {
            return 0;
        }
        T negzro = One / (neginf+One);
        if (negzro!=Zero)
        {
            return 0;
        }
        neginf = One / negzro;
        if (neginf>=Zero)
        {
            return 0;
        }
        T newzro = negzro + Zero;
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
        T NaN1 = posinf + neginf;
        T NaN2 = posinf / neginf;
        T NaN3 = posinf / posinf;
        T NaN4 = posinf * Zero;
        T NaN5 = neginf * negzro;
        T NaN6 = NaN5   * Zero;
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
    static int iladlc(int m, int n, T const* A, int lda)
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
    static int iladlr(int m, int n, T const* A, int lda)
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
            int i, j, ipos;
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
                return int(T((n1<n2)?n1:n2) * 1.6);
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
        int nh, ns = 0, nstemp;
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
                nstemp = nh / int(std::log(T(nh))/std::log(TWO));
                ns = ((10>nstemp) ? 10 : nstemp);
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
            nstemp = ns - (ns%2);
            ns = ((2>nstemp) ? 2 : nstemp);
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
            // IACC22: Whether to accumulate reflections before updating the far-from-diagonal elements
            // and whether to use 2-by-2 block structure while doing it. A small amount of work could be
            // saved by making this choice dependent also upon the nh = ihi-ilo+1.
            if (ns>=KACMIN)
            {
                return 1;
            }
            if (ns>=K22MIN)
            {
                return 2;
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
}
#endif