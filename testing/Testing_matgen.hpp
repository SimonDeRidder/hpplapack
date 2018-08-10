#ifndef TESTING_MATGEN_HEADER
#define TESTING_MATGEN_HEADER

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

#include "Blas.hpp"
#include "Lapack_dyn.hpp"

template<class real>
class Testing_matgen : public Lapack_dyn<real>
{
public:
    // constants

    const real ZERO  = real(0.0);
    const real HALF  = real(0.5);
    const real ONE   = real(1.0);
    const real TWO   = real(2.0);
    const real TWOPI = real(6.2831853071795864769252867663);
    // LAPACK TESTING MATGEN

    /* dlagge generates a real general m by n matrix A, by pre- and post- multiplying a real
     * diagonal matrix D with random orthogonal matrices: A = U*D*V. The lower and upper bandwidths
     * may then be reduced to kl and ku by additional orthogonal transformations.
     * Parameters: m: The number of rows of the matrix A. m>=0.
     *             n: The number of columns of the matrix A. n>=0.
     *             kl: The number of nonzero subdiagonals within the band of A. 0<=kl<=m-1.
     *             ku: The number of nonzero superdiagonals within the band of A. 0<=ku<=n-1.
     *             d: an array, dimension (min(m,n))
     *                The diagonal elements of the diagonal matrix D.
     *             A: an array, dimension (lda,n)
     *                The generated m by n matrix A.
     *             lda: The leading dimension of the array A. lda>=m.
     *             iseed: an integer array, dimension (4)
     *                    On entry, the seed of the random number generator; the array elements
     *                              must be between 0 and 4095, and iseed[3] must be odd.
     *                    On exit, the seed is updated.
     *             work: an array, dimension (m+n)
     *             info: ==0: successful exit
     *                    <0: if info==-i, the i-th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void dlagge(int m, int n, int kl, int ku, real const* d, real* A, int lda, int* iseed,
                       real* work, int& info)
    {
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
        else if (kl<0 || kl>m-1)
        {
            info = -3;
        }
        else if (ku<0 || ku>n-1)
        {
            info = -4;
        }
        else if (lda<1 || lda<m)
        {
            info = -7;
        }
        if (info<0)
        {
            this->xerbla("DLAGGE", -info);
            return;
        }
        int i, j, aind;
        // initialize A to diagonal matrix
        for (j=0; j<n; j++)
        {
            aind = lda*j;
            for (i=0; i<m; i++)
            {
                A[i+aind] = ZERO;
            }
        }
        for (i=0; i<m && i<n; i++)
        {
           A[i+lda*i] = d[i];
        }
        // Quick exit if the user wants a diagonal matrix
        if ((kl==0) && (ku==0))
        {
            return;
        }
        real tau, wa, wb, wn;
        // pre- and post-multiply A by random orthogonal matrices
        for (i=std::min(m, n)-1; i>=0; i--)
        {
            aind = i+lda*i;
            if (i<m-1)
            {
                // generate random reflection
                this->dlarnv(3, iseed, m-i, work);
                wn = Blas<real>::dnrm2(m-i, work, 1);
                wa = std::copysign(wn, work[0]);
                if (wn==ZERO)
                {
                    tau = ZERO;
                }
                else
                {
                    wb = work[0] + wa;
                    Blas<real>::dscal(m-i-1, ONE/wb, &work[1], 1);
                    work[0] = ONE;
                    tau = wb / wa;
                }
                // multiply A[i:m-1,i:n-1] by random reflection from the left
                Blas<real>::dgemv("Transpose", m-i, n-i, ONE, &A[aind], lda, work, 1, ZERO,
                                  &work[m], 1);
                Blas<real>::dger(m-i, n-i, -tau, work, 1, &work[m], 1, &A[aind], lda);
            }
            if (i<n-1)
            {
                // generate random reflection
                this->dlarnv(3, iseed, n-i, work);
                wn = Blas<real>::dnrm2(n-i, work, 1);
                wa = std::copysign(wn, work[0]);
                if (wn==ZERO)
                {
                    tau = ZERO;
                }
                else
                {
                    wb = work[0] + wa;
                    Blas<real>::dscal(n-i-1, ONE / wb, &work[1], 1);
                    work[0] = ONE;
                    tau = wb / wa;
                }
                // multiply A[i:m-1,i:n-1] by random reflection from the right
                Blas<real>::dgemv("No transpose", m-i, n-i, ONE, &A[aind], lda, work, 1, ZERO,
                                  &work[n], 1);
                Blas<real>::dger(m-i, n-i, -tau, &work[n], 1, work, 1, &A[aind], lda);
            }
        }
        // Reduce number of subdiagonals to KL and number of superdiagonals to KU
        for (i=0; i<std::max(m-1-kl, n-1-ku); i++)
        {
            if (kl<=ku)
            {
                // annihilate subdiagonal elements first (necessary if KL==0)
                if (i<m-1-kl && i<n)
                {
                    aind = kl+i+lda*i;
                    // generate reflection to annihilate A[kl+i+1:m-1,i]
                    wn = Blas<real>::dnrm2(m-kl-i, &A[aind], 1);
                    wa = std::copysign(wn, A[aind]);
                    if (wn==ZERO)
                    {
                        tau = ZERO;
                    }
                    else
                    {
                        wb = A[aind] + wa;
                        Blas<real>::dscal(m-1-kl-i, ONE/wb, &A[1+aind], 1);
                        A[aind] = ONE;
                        tau = wb / wa;
                    }
                    // apply reflection to A[kl+i:m-1,i+1:n-1] from the left
                    Blas<real>::dgemv("Transpose", m-kl-i, n-i-1, ONE, &A[aind+lda], lda, &A[aind],
                                      1, ZERO, work, 1);
                    Blas<real>::dger(m-kl-i, n-i-1, -tau, &A[aind], 1, work, 1, &A[aind+lda], lda);
                    A[kl+i+lda*i] = -wa;
                }
                if (i<n-1-ku && i<m)
                {
                    aind = i+lda*(ku+i);
                    // generate reflection to annihilate A[i,ku+i+1:n-1]
                    wn = Blas<real>::dnrm2(n-ku-i, &A[aind], lda);
                    wa = std::copysign(wn, A[aind]);
                    if (wn==ZERO)
                    {
                        tau = ZERO;
                    }
                    else
                    {
                        wb = A[aind] + wa;
                        Blas<real>::dscal(n-ku-i-1, ONE/wb, &A[aind+lda], lda);
                        A[aind] = ONE;
                        tau = wb / wa;
                    }
                    // apply reflection to A[i+1:m-1,ku+i:n-1] from the right
                    Blas<real>::dgemv("No transpose", m-i-1, n-ku-i, ONE, &A[1+aind], lda,
                                      &A[aind], lda, ZERO, work, 1);
                    Blas<real>::dger(m-i-1, n-ku-i, -tau, work, 1, &A[aind], lda, &A[1+aind], lda);
                    A[aind] = -wa;
                }
            }
            else
            {
                // annihilate superdiagonal elements first (necessary if ku = 0)
                if (i<n-1-ku && i<m)
                {
                    aind = i+lda*(ku+i);
                    // generate reflection to annihilate A[i,ku+i+1:n-1]
                    wn = Blas<real>::dnrm2(n-ku-i, &A[aind], lda);
                    wa = std::copysign(wn, A[aind]);
                    if (wn==ZERO)
                    {
                        tau = ZERO;
                    }
                    else
                    {
                        wb = A[aind] + wa;
                        Blas<real>::dscal(n-ku-i-1, ONE/wb, &A[aind+lda], lda);
                        A[aind] = ONE;
                        tau = wb / wa;
                    }
                    // apply reflection to A[i+1:m-1,ku+i:n-1] from the right
                    Blas<real>::dgemv("No transpose", m-i-1, n-ku-i, ONE, &A[1+aind], lda,
                                      &A[aind], lda, ZERO, work, 1);
                    Blas<real>::dger(m-i-1, n-ku-i, -tau, work, 1, &A[aind], lda, &A[1+aind], lda);
                    A[aind] = -wa;
                }
                if (i<m-1-kl && i<n)
                {
                    aind = kl+i+lda*i;
                    // generate reflection to annihilate A[kl+i+1:m-1,i]
                    wn = Blas<real>::dnrm2(m-kl-i, &A[aind], 1);
                    wa = std::copysign(wn, A[aind]);
                    if (wn==ZERO)
                    {
                        tau = ZERO;
                    }
                    else
                    {
                        wb = A[aind] + wa;
                        Blas<real>::dscal(m-kl-i-1, ONE/wb, &A[1+aind], 1);
                        A[aind] = ONE;
                        tau = wb / wa;
                    }
                    // apply reflection to A[kl+i:m-1,i+1:n-1] from the left
                    Blas<real>::dgemv("Transpose", m-kl-i, n-i-1, ONE, &A[aind+lda], lda, &A[aind],
                                      1, ZERO, work, 1);
                    Blas<real>::dger(m-kl-i, n-i-1, -tau, &A[aind], 1, work, 1, &A[aind+lda], lda);
                    A[aind] = -wa;
                }
            }
            if (i<n)
            {
                aind = lda*i;
                for (j=kl+i+1; j<m; j++)
                {
                    A[j+aind] = ZERO;
                }
            }
            if (i<m)
            {
                for (j=ku+i+1; j<n; j++)
                {
                    A[i+lda*j] = ZERO;
                }
            }
        }
    }

    /* dlagsy generates a real symmetric matrix A, by pre- and post- multiplying a real diagonal
     * matrix D with a random orthogonal matrix: A = U*D*U'. The semi-bandwidth may then be reduced
     * to k by additional orthogonal transformations.
     * Parameters: n: The order of the matrix A. n>=0.
     *             k: The number of nonzero subdiagonals within the band of A. 0<=k<=n-1.
     *             d: an array, dimension (n)
     *                The diagonal elements of the diagonal matrix D.
     *             A: an array, dimension (lda,n)
     *                The generated n by n symmetric matrix A (the full matrix is stored).
     *             lda: The leading dimension of the array A. lda>=n.
     *             iseed: an integer array, dimension (4)
     *                    On entry, the seed of the random number generator; the array elements
     *                              must be between 0 and 4095, and iseed[3] must be odd.
     *                    On exit, the seed is updated.
     *             work: an array, dimension (2*n)
     *             info: ==0: successful exit
     *                    <0: if info==-i, the i-th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void dlagsy(int n, int k, real const* d, real* A, int lda, int* iseed, real* work,
                       int& info)
    {
        // Test the input arguments
        info = 0;
        if (n<0)
        {
            info = -1;
        }
        else if (k<0||k>n-1)
        {
            info = -2;
        }
        else if (lda<1||lda<n)
        {
            info = -5;
        }
        if (info<0)
        {
            this->xerbla("DLAGSY", -info);
            return;
        }
        int i, j, aind;
        // initialize lower triangle of A to diagonal matrix
        for (j=0; j<n; j++)
        {
            aind = lda*j;
            for (i=j+1; i<n; i++)
            {
                A[i+aind] = ZERO;
            }
        }
        for (i=0; i<n; i++)
        {
            A[i+lda*i] = d[i];
        }
        // Generate lower triangle of symmetric matrix
        real alpha, tau, wa, wb, wn;
        for (i=n-2; i>=0; i--)
        {
            // generate random reflection
            this->dlarnv(3, iseed, n-i, work);
            wn = Blas<real>::dnrm2(n-i, work, 1);
            wa = std::copysign(wn, work[0]);
            if (wn==ZERO)
            {
                tau = ZERO;
            }
            else
            {
                wb = work[0]+wa;
                Blas<real>::dscal(n-i-1, ONE/wb, &work[1], 1);
                work[0] = ONE;
                tau = wb/wa;
            }
            // apply random reflection to A[i:n-1,i:n-1] from the left and the right
            aind = i+lda*i;
            // compute  y := tau * A * u
            Blas<real>::dsymv("Lower", n-i, tau, &A[aind], lda, work, 1, ZERO, &work[n], 1);
            // compute  v := y - 1/2 * tau * (y, u) * u
            alpha = -HALF * tau * Blas<real>::ddot(n-i, &work[n], 1, work, 1);
            Blas<real>::daxpy(n-i, alpha, work, 1, &work[n], 1);
            // apply the transformation as a rank-2 update to A[i:n-1,i:n-1]
            Blas<real>::dsyr2("Lower", n-i, -ONE, work, 1, &work[n], 1, &A[aind], lda);
        }
        // Reduce number of subdiagonals to K
        for (i=0; i<n-1-k; i++)
        {
            aind = k+i+lda*i;
            // generate reflection to annihilate A[k+i+1:n-1,i]
            wn = Blas<real>::dnrm2(n-k-i, &A[aind], 1);
            wa = std::copysign(wn, A[aind]);
            if (wn==ZERO)
            {
                tau = ZERO;
            }
            else
            {
                wb = A[aind]+wa;
                Blas<real>::dscal(n-k-i-1, ONE/wb, &A[1+aind], 1);
                A[aind] = ONE;
                tau = wb/wa;
            }
            // apply reflection to A[k+i:n-1,i+1:k+i-1] from the left
            Blas<real>::dgemv("Transpose", n-k-i, k-1, ONE, &A[aind+lda], lda, &A[aind], 1, ZERO,
                              work, 1);
            Blas<real>::dger(n-k-i, k-1, -tau, &A[aind], 1, work, 1, &A[aind+lda], lda);
            // apply reflection to A[k+i:n-1,k+i:n-1] from the left and the right
            // compute  y := tau * A * u
            Blas<real>::dsymv("Lower", n-k-i, tau, &A[aind+lda*k], lda, &A[aind], 1, ZERO, work,
                              1);
            // compute  v := y - 1/2 * tau * (y, u) * u
            alpha = -HALF * tau * Blas<real>::ddot(n-k-i, work, 1, &A[aind], 1);
            Blas<real>::daxpy(n-k-i, alpha, &A[aind], 1, work, 1);
            // apply symmetric rank-2 update to A[k+i:n-1,k+i:n-1]
            Blas<real>::dsyr2("Lower", n-k-i, -ONE, &A[aind], 1, work, 1, &A[aind+lda*k], lda);
            A[aind] = -wa;
            aind = lda*i;
            for (j=k+i+1; j<n; j++)
            {
                A[j+aind] = ZERO;
            }
        }
        // Store full symmetric matrix
        for (j=0; j<n; j++)
        {
            aind = lda*j;
            for (i=j+1; i<n; i++)
            {
                A[j+lda*i] = A[i+aind];
            }
        }
    }

    /* dlaran returns a random real number from a uniform (0,1) distribution.
     * Parameters: iseed: an integer array, dimension (4)
     *                    On entry, the seed of the random number generator; the array elements
     *                              must be between 0 and 4095, and iseed[3] must be odd.
     *                    On exit, the seed is updated.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Further Details:
     *     This routine uses a multiplicative congruential method with modulus 2^48 and multiplier
     *     33952834046453 (see G.S.Fishman, 'Multiplicative congruential random number generators
     *     with modulus 2**b: an exhaustive analysis for b = 32 and a partial analysis for b = 48',
     *     Math. Comp. 189, pp 331-344, 1990).
     *     48-bit integers are stored in 4 integer array elements with 12 bits per element. Hence
     *     the routine is portable across machines with integers of 32 bits or more.             */
    real dlaran(int* iseed)
    {
        const int M1 = 494;
        const int M2 = 322;
        const int M3 = 2508;
        const int M4 = 2549;
        const int IPW2 = 4096;
        const real R = ONE / IPW2;
        int it1, it2, it3, it4;
        real rndout;
        do
        {
            // multiply the seed by the multiplier modulo 2^48
            it4 = iseed[3]*M4;
            it3 = it4 / IPW2;
            it4 = it4 - IPW2*it3;
            it3 = it3 + iseed[2]*M4 + iseed[3]*M3;
            it2 = it3 / IPW2;
            it3 = it3 - IPW2*it2;
            it2 = it2 + iseed[1]*M4 + iseed[2]*M3 + iseed[3]*M2;
            it1 = it2 / IPW2;
            it2 = it2 - IPW2*it1;
            it1 = it1 + iseed[0]*M4 + iseed[1]*M3 + iseed[2]*M2 + iseed[3]*M1;
            it1 = it1 % IPW2;
            // return updated seed
            iseed[0] = it1;
            iseed[1] = it2;
            iseed[2] = it3;
            iseed[3] = it4;
            // convert 48-bit integer to a real number in the interval (0,1)
            rndout = R*(real(it1)+R*(real(it2)+R*(real(it3)+R*(real(it4)))));
            // If a real number has n bits of precision, and the first n bits of the 48-bit integer
            // above happen to be all 1 (which will occur about once every 2^n calls), then dlaran
            // will be rounded to exactly 1.0. Since DLARAN is not supposed to return exactly 0.0
            // or 1.0 (and some callers of dlaran, such as clarnd, depend on that), the
            // statistically correct thing to do in this situation is simply to iterate again.
            // N.B. the case dlaran = 0.0 should not be possible.
        } while (rndout==ONE);
        return rndout;
    }

    /* dlarnd returns a random real number from a uniform or normal distribution.
     * Paramters: idist: Specifies the distribution of the random numbers:
     *                   ==1: uniform (0,1)
     *                   ==2: uniform (-1,1)
     *                   ==3: normal (0,1)
     *            iseed: an integer array, dimension (4)
     *                   On entry, the seed of the random number generator; the array elements must
     *                             be between 0 and 4095, and iseed[3] must be odd.
     *                   On exit, the seed is updated.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Further Details:
     *     This routine calls the auxiliary routine dlaran to generate a random real number from a
     *     uniform (0,1) distribution. The Box-Muller method is used to transform numbers from a
     *     uniform to a normal distribution.                                                     */
    real dlarnd(int idist, int* iseed)
    {
        // Generate a real random number from a uniform (0,1) distribution
        real t1 = dlaran(iseed);
        if (idist==1)
        {
            // uniform (0,1)
            return t1;
        }
        else if (idist==2)
        {
            // uniform (-1,1)
            return TWO*t1 - ONE;
        }
        else if (idist==3)
        {
            // normal (0,1)
            real t2 = dlaran(iseed);
            return std::sqrt(-TWO*std::log(t1))*std::cos(TWOPI*t2);
        }
        return t1;// to satisfy compiler
    }

    /* dlarot applies a (Givens) rotation to two adjacent rows or columns, where one element of the
     * first and/or last column/row for use on matrices stored in some format other than GE, so
     * that elements of the matrix may be used or modified for which no array element is provided.
     * One example is a symmetric matrix in SB format (bandwidth=4), for which UPLO='L': Two
     * adjacent rows will have the format:
     *     row j:     C> C> C> C> C> .  .  .  .
     *     row j+1:      C> C> C> C> C> .  .  .  .
     * '*' indicates elements for which storage is provided,
     * '.' indicates elements for which no storage is provided, but are not necessarily zero; their
     *     values are determined by symmetry.
     * ' ' indicates elements which are necessarily zero, and have no storage provided.
     * Those columns which have two '*'s can be handled by drot.
     * Those columns which have no '*'s can be ignored, since as long as the Givens rotations are
     *     carefully applied to preserve symmetry, their values are determined.
     * Those columns which have one '*' have to be handled separately, by using separate variables
     *     "p" and "q":
     *     row j:     C> C> C> C> C> p  .  .  .
     *     row j+1:   q  C> C> C> C> C> .  .  .  .
     * The element p would have to be set correctly, then that column is rotated, setting p to its
     * new value. The next call to dlarot would rotate columns j and j+1, using p, and restore
     * symmetry. The element q would start out being zero, and be made non-zero by the rotation.
     * Later, rotations would presumably be chosen to zero q out.
     * Typical Calling Sequences: rotating the i-th and (i+1)-st rows.
     * ------- ------- ---------
     * General dense matrix:
     *     dlarot(true, false, false, N, c, s, &A[i,0], lda, DUMMY, DUMMY);
     * General banded matrix in GB format:
     *     j = MAX(0, i-KL);
     *     nl = MIN(N, i+KU+2) - j;
     *     dlarot(true, i-KL>=0, i+KU<N-1, nl, c, s, &A[KU+i-j,j], lda-1, xleft, xright);
     *         [ note that i-j is just MIN(i,KL) ]
     * Symmetric banded matrix in SY format, bandwidth K, lower triangle only:
     *     j = MAX(0, i-K);
     *     nl = MIN(K+1, i+1) + 1;
     *     dlarot(true, i-K>=0, true, nl, c, s, &A[i,j], lda, xleft, xright);
     * Same, but upper triangle only:
     *     nl = MIN(K+2, N-i);
     *     dlarot(true, true, i+K<N-1, nl, c, s, &A[i,i], lda, xleft, xright);
     * Symmetric banded matrix in SB format, bandwidth K, lower triangle only:
     *         [ same as for SY, except:]
     *     ..., &A[i-j,j], lda-1, xleft, xright);
     *         [ note that i-j is just MIN(i,K) ]
     * Same, but upper triangle only:
     *     ..., &A[K,i], lda-1, xleft, xright);
     * Rotating columns is just the transpose of rotating rows, except for GB and SB:
     * (rotating columns i and i+1)
     * GB:
     *     j = MAX(0, i-KU);
     *     nl = MIN(N, i+KL+2) - j;
     *     dlarot(true, i-KU>=0, i+KL<N-1, nl, c, s, &A[KU+j-i,i], lda-1, XTOP, XBOTTM);
     *         [note that KU+j-i is just MAX(0,KU-i)]
     * SB: (upper triangle)
     *     ..., &A[K+j-i,i], lda-1, XTOP, XBOTTM);
     * SB: (lower triangle)
     *     ..., &A[0,i], lda-1, XTOP, XBOTTM);
     * Parameters: lrows: If true, then dlarot will rotate two rows.
     *                    If false, then it will rotate two columns.
     *             lleft: If true, then xleft will be used instead of the corresponding element of
     *                             A for the first element in the second row (if lrows==false) or
     *                             column (if lrows==true)
     *                    If false, then the corresponding element of A will be used.
     *             lright: If true, then xright will be used instead of the corresponding element
     *                              of A for the last element in the first row (if lrows==false)
     *                              or column (if lrows==true)
     *                     If false, then the corresponding element of A will be used.
     *             nl: The length of the rows (if lrows==true) or columns (if lrows==false) to be
     *                 rotated. If xleft and/or xright are used, the columns/rows they are in
     *                 should be included in nl, e.g., if lleft==lright==true, then nl must be at
     *                 least 2. The number of rows/columns to be rotated exclusive of those
     *                 involving xleft and/or xright may not be negative, i.e., nl minus how many
     *                 of lleft and lright are true must be at least zero; if not, XERBLA will be
     *                 called.
     *             c,
     *             s: Specify the Givens rotation to be applied.
     *                If lrows is true, then the matrix ( c  s )
     *                                                  (-s  c ) is applied from the left;
     *                if false, then the transpose thereof is applied from the right. For a Givens
     *                          rotation, c^2 + s^2 should be 1, but this is not checked.
     *             A: The array containing the rows/columns to be rotated. The first element of A
     *                should be the upper left element to be rotated.
     *             lda: The "effective" leading dimension of A. If A contains a matrix stored in GE
     *                  or SY format, then this is just the leading dimension of A as dimensioned
     *                  in the calling routine. If A contains a matrix stored in band (GB or SB)
     *                  format, then this should be *one less* than the leading dimension used in
     *                  the calling routine. Thus, if A were dimensioned A(lda,*) in dlarot, then
     *                  A(1,j) would be the j-th element in the first of the two rows to be
     *                  rotated, and A(2,j) would be the j-th in the second, regardless of how the
     *                  array may be stored in the calling routine. [A cannot, however, actually be
     *                  dimensioned thus, since for band format, the row number may exceed lda
     *                  which is not legal.]
     *                  If lrows==true, then lda must be at least 1, otherwise it must be at least
     *                  nl minus the number of true values in xleft and xright.
     *             xleft: If lleft is true, then xleft will be used and modified instead of A(2,1)
     *                    (if lrows==true) or A(1,2) (if lrows==false).
     *             xright: If lright is true, then xright will be used and modified instead of
     *                     A(1,nl) (if lrows==true) or A(nl,1) (if lrows==false).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void dlarot(bool lrows, bool lleft, bool lright, int nl, real c, real s, real* A,
                       int lda, real& xleft, real& xright)
    {
        // Set up indices, arrays for ends
        int iinc, inext;
        if (lrows)
        {
            iinc = lda;
            inext = 1;
        }
        else
        {
            iinc = 1;
            inext = lda;
        }
        int ix, iy, nt;
        real xt[2];
        real yt[2];
        if (lleft)
        {
            nt = 1;
            ix = iinc;
            iy = 1 + lda;
            xt[0] = A[0];
            yt[0] = xleft;
        }
        else
        {
            nt = 0;
            ix = 0;
            iy = inext;
        }
        int iyt;
        if (lright)
        {
            iyt = inext + (nl-1)*iinc;
            xt[nt] = xright;
            yt[nt] = A[iyt];
            nt++;
        }
        // Check for errors
        if (nl<nt)
        {
            this->xerbla("DLAROT", 4);
            return;
        }
        if (lda<=0 || (!lrows && lda<nl-nt))
        {
            this->xerbla("DLAROT", 8);
            return;
        }
        // Rotate
        Blas<real>::drot(nl-nt, &A[ix], iinc, &A[iy], iinc, c, s);
        Blas<real>::drot(nt, xt, 1, yt, 1, c, s);
        // Stuff values back into xleft, xright, etc.
        if (lleft)
        {
            A[0]  = xt[0];
            xleft = yt[0];
        }
        if (lright)
        {
            xright = xt[nt-1];
            A[iyt] = yt[nt-1];
        }
    }

    /* dlatm1 computes the entries of d[0:n-1] as specified by mode, cond and irsign. idist and
     * iseed determine the generation of random numbers. dlatm1 is called by dlatmr to generate
     * random test matrices for LAPACK programs.
     * Parameters: mode: On entry describes how d is to be computed:
     *                   mode==0 means do not change d.
     *                   mode==1 sets d[0]=1 and d[1:n-1]=1.0/cond
     *                   mode==2 sets d[0:n-2]=1 and d[n-1]=1.0/cond
     *                   mode==3 sets d[i]=cond^(-(I-1)/(n-1))
     *                   mode==4 sets d[i]=1 - (i-1)/(n-1)*(1-1/cond)
     *                   mode==5 sets d to random numbers in the range (1/cond, 1) such that their
     *                           logarithms are uniformly distributed.
     *                   mode==6 set d to random numbers from same distribution as the rest of the
     *                           matrix.
     *                   mode <0 has the same meaning as abs(mode), except that the order of the
     *                           elements of d is reversed.
     *                           Thus if mode is positive, d has entries ranging from 1 to 1/cond,
     *                           if negative, from 1/cond to 1.
     *             cond: On entry, used as described under mode above.
     *                   If used, it must be >=1.
     *             irsign: On entry, if mode neither -6, 0 nor 6, determines sign of entries of d
     *                     0 => leave entries of d unchanged
     *                     1 => multiply each entry of d by 1 or -1 with probability 0.5
     *             idist: On entry, idist specifies the type of distribution to be used to generate
     *                    a random matrix.
     *                    1 => UNIFORM(0, 1)
     *                    2 => UNIFORM(-1, 1)
     *                    3 => NORMAL(0, 1)
     *             iseed: an INTEGER array, dimension (4)
     *                    On entry iseed specifies the seed of the random number generator. The
     *                    random number generator uses a linear congruential sequence limited to
     *                    small integers, and so should produce machine independent random numbers.
     *                    The values of iseed are changed on exit, and can be used in the next call
     *                    to dlatm1 to continue the same random number sequence.
     *             d: an array, dimension (n)
     *                Array to be computed according to mode, cond and irsign.
     *                May be changed on exit if mode is nonzero.
     *             n: Number of entries of d.
     *             info: 0 => normal termination
     *                  -1 => if mode not in range -6 to 6
     *                  -2 => if mode neither -6, 0 nor 6, and irsign neither 0 nor 1
     *                  -3 => if mode neither -6, 0 nor 6 and cond less than 1
     *                  -4 => if mode equals 6 or -6 and idist not in range 1 to 3
     *                  -7 => if n negative
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void dlatm1(int mode, real cond, int irsign, int idist, int* iseed, real* d, int n,
                       int& info)
    {
        // Decode and Test the input parameters. Initialize flags & seed.
        info = 0;
        // Quick return if possible
        if (n==0)
        {
            return;
        }
        // Set info if an error
        if (mode<-6 || mode>6)
        {
            info = -1;
        }
        else if ((mode!=-6 && mode!=0 && mode!=6) && (irsign!=0 && irsign!=1))
        {
            info = -2;
        }
        else if ((mode!=-6 && mode!=0 && mode!=6) && cond<ONE)
        {
            info = -3;
        }
        else if ((mode==6 || mode==-6) && (idist<1 || idist>3))
        {
            info = -4;
        }
        else if (n<0)
        {
            info = -7;
        }
        if (info!=0)
        {
            this->xerbla("DLATM1", -info);
            return;
        }
        // Compute d according to cond and mode
        if (mode!=0)
        {
            int i;
            real alpha, temp;
            switch (std::abs(mode))
            {
                case 1:
                    // One large d value:
                    for (i=0; i<n; i++)
                    {
                        d[i] = ONE / cond;
                    }
                    d[0] = ONE;
                    break;
                case 2:
                    // One small d value:
                    for (i=0; i<n; i++)
                    {
                        d[i] = ONE;
                    }
                    d[n-1] = ONE / cond;
                    break;
                case 3:
                    // Exponentially distributed d values:
                    d[0] = ONE;
                    if (n>1)
                    {
                        alpha = std::pow(cond, (-ONE/real(n-1)));
                        for (i=1; i<n; i++)
                        {
                            d[i] = std::pow(alpha, real(i));
                        }
                    }
                    break;
                case 4:
                    // Arithmetically distributed d values:
                    d[0] = ONE;
                    if (n>1)
                    {
                        temp = ONE / cond;
                        alpha = (ONE-temp) / real(n-1);
                        for (i=1; i<n; i++)
                        {
                            d[i] = real(n-1-i)*alpha + temp;
                        }
                    }
                    break;
                case 5:
                    // Randomly distributed d values on (1/cond, 1):
                    alpha = std::log(ONE/cond);
                    for (i=0; i<n; i++)
                    {
                        d[i] = std::exp(alpha*dlaran(iseed));
                    }
                    break;
                case 6:
                    // Randomly distributed d values from idist
                    this->dlarnv(idist, iseed, n, d);
                    break;
            }
            // If mode neither -6 nor 0 nor 6, and IRSIGN = 1, assign random signs to d
            if ((mode!=-6 && mode!=0 && mode!=6) && irsign==1)
            {
                for (i=0; i<n; i++)
                {
                    temp = dlaran(iseed);
                    if (temp>HALF)
                    {
                        d[i] = -d[i];
                    }
                }
            }
            // Reverse if MODE < 0
            if (mode<0)
            {
                for (i=0; i<n/2; i++)
                {
                    temp = d[i];
                    d[i] = d[n-1-i];
                    d[n-1-i] = temp;
                }
            }
        }
    }

    /* dlatms generates random matrices with specified singular values (or symmetric/hermitian with
     * specified eigenvalues) for testing LAPACK programs.
     * dlatms operates by applying the following sequence of operations:
     *     Set the diagonal to d, where d may be input or computed according to mode, cond, dmax,
     *         and sym as described below.
     *     Generate a matrix with the appropriate band structure, by one of two methods:
     *         Method A: Generate a dense m x n matrix by multiplying D on the left and the right
     *                   by random unitary matrices, then:
     *                   Reduce the bandwidth according to kl and ku, using Householder
     *                   transformations.
     *         Method B: Convert the bandwidth-0 (i.e., diagonal) matrix to a bandwidth-1 matrix
     *                   using Givens rotations, "chasing" out-of-band elements back, much as in
     *                   QR; then convert the bandwidth-1 to a bandwidth-2 matrix, etc.
     *                   Note that for reasonably small bandwidths (relative to m and n) this
     *                   requires less storage, as a dense matrix is not generated. Also, for
     *                   symmetric matrices, only one triangle is generated.
     *         Method A is chosen if the bandwidth is a large fraction of the order of the matrix,
     *         and lda is at least m (so a dense matrix can be stored). Method B is chosen if the
     *         bandwidth is small (<1/2 n for symmetric, <0.3 n+m for non-symmetric), or lda is
     *         less than m and not less than the bandwidth.
     *     Pack the matrix if desired. Options specified by pack are:
     *         no packing
     *         zero out upper half (if symmetric)
     *         zero out lower half (if symmetric)
     *         store the upper half columnwise (if symmetric or upper triangular)
     *         store the lower half columnwise (if symmetric or lower triangular)
     *         store the lower triangle in banded format (if symmetric or lower triangular)
     *         store the upper triangle in banded format (if symmetric or upper triangular)
     *         store the entire matrix in banded format
     *         If Method B is chosen, and band format is specified, then the matrix will be
     *         generated in the band format, so no repacking will be necessary.
     * Parameters: m: The number of rows of A.
     *             n: The number of columns of A.
     *             dist: On entry, dist specifies the type of distribution to be used to generate
     *                   the random eigen-/singular values.
     *                   'U' => UNIFORM(0, 1)  ('U' for uniform)
     *                   'S' => UNIFORM(-1, 1) ('S' for symmetric)
     *                   'N' => NORMAL(0, 1)   ('N' for normal)
     *             iseed: an integer array, dimension (4)
     *                    On entry iseed specifies the seed of the random number generator. They
     *                    should lie between 0 and 4095 inclusive, and iseed[3] should be odd. The
     *                    random number generator uses a linear congruential sequence limited to
     *                    small integers, and so should produce machine independent random numbers.
     *                    The values of iseed are changed on exit, and can be used in the next call
     *                    to dlatms to continue the same random number sequence.
     *             sym: If sym=='S' or 'H', the generated matrix is symmetric, with eigenvalues
     *                                      specified by d, cond, mode, and dmax; they may be
     *                                      positive, negative, or zero.
     *                  If sym=='P', the generated matrix is symmetric, with eigenvalues
     *                               (= singular values) specified by d, cond, mode, and dmax; they
     *                               will not be negative.
     *                  If sym=='N', the generated matrix is nonsymmetric, with singular values
     *                               specified by d, cond, mode, and dmax; they will not be
     *                               negative.
     *             d: an array, dimension (MIN(m, n))
     *                This array is used to specify the singular values or eigenvalues of A (see
     *                sym, above). If mode==0, then d is assumed to contain the
     *                singular/eigenvalues, otherwise they will be computed according to mode,
     *                cond and dmax, and placed in d.
     *             mode: On entry this describes how the singular/eigenvalues are to be specified:
     *                   mode==0 means use d as input
     *                   mode==1 sets d[0]=1 and d[1:n-1]=1.0/cond
     *                   mode==2 sets d[0:n-2]=1 and d[n-1]=1.0/cond
     *                   mode==3 sets d[i]=cond**(-(I-1)/(n-1))
     *                   mode==4 sets d[i]=1 - (i-1)/(n-1)*(1 - 1/cond)
     *                   mode==5 sets d to random numbers in the range (1/cond, 1) such that their
     *                           logarithms are uniformly distributed.
     *                   mode==6 set d to random numbers from same distribution as the rest of the
     *                           matrix.
     *                   mode <0 has the same meaning as abs(mode), except that the order of the
     *                           elements of d is reversed.
     *                   Thus if mode is positive, d has entries ranging from 1 to 1/cond, if
     *                       negative, from 1/cond to 1,
     *                   If sym=='S' or 'H', and mode is neither 0, 6, nor -6, then the elements of
     *                       d will also be multiplied by a random sign (i.e., +1 or -1.)
     *             cond: On entry, this is used as described under mode above.
     *                   If used, it must be >=1.
     *             dmax: If mode is neither -6, 0 nor 6, the contents of d, as computed according
     *                   to mode and cond, will be scaled by dmax / max(abs(D[i])); thus, the
     *                   maximum absolute eigen- or singular value (which is to say the norm) will
     *                   be abs(dmax). Note that dmax need not be positive: if dmax is negative
     *                   (or zero), d will be scaled by a negative number (or zero).
     *             kl: This specifies the lower bandwidth of the matrix. For example, kl=0 implies
     *                 upper triangular, kl=1 implies upper Hessenberg, and kl being at least m-1
     *                 means that the matrix has full lower bandwidth. kl must equal ku if the
     *                 matrix is symmetric.
     *             ku: This specifies the upper bandwidth of the matrix. For example, ku=0 implies
     *                 lower triangular, ku=1 implies lower Hessenberg, and ku being at least n-1
     *                 means that the matrix has full upper bandwidth. kl must equal ku if the
     *                 matrix is symmetric.
     *             pack: This specifies packing of matrix as follows:
     *                   'N' => no packing
     *                   'U' => zero out all subdiagonal entries (if symmetric)
     *                   'L' => zero out all superdiagonal entries (if symmetric)
     *                   'C' => store the upper triangle columnwise
     *                          (only if the matrix is symmetric or upper triangular)
     *                   'R' => store the lower triangle columnwise
     *                          (only if the matrix is symmetric or lower triangular)
     *                   'B' => store the lower triangle in band storage scheme
     *                          (only if matrix symmetric or lower triangular)
     *                   'Q' => store the upper triangle in band storage scheme
     *                          (only if matrix symmetric or upper triangular)
     *                   'Z' => store the entire matrix in band storage scheme
     *                          (pivoting can be provided for by using this option to store A in
     *                           the trailing rows of the allocated storage)
     *                   Using these options, the various LAPACK packed and banded storage schemes
     *                   can be obtained:
     *                   GB           - use 'Z'
     *                   PB, SB or TB - use 'B' or 'Q'
     *                   PP, SP or TP - use 'C' or 'R'
     *                   If two calls to dlatms differ only in the pack parameter, they will
     *                   generate mathematically equivalent matrices.
     *             A: an array, dimension (lda, n)
     *                On exit A is the desired test matrix. A is first generated in full (unpacked)
     *                form, and then packed, if so specified by pack. Thus, the first m elements of
     *                the first n columns will always be modified. If pack specifies a packed or
     *                banded storage scheme, all lda elements of the first n columns will be
     *                modified; the elements of the array which do not correspond to elements of
     *                the generated matrix are set to zero.
     *             lda: lda specifies the first dimension of A as declared in the calling program.
     *                  If pack=='N', 'U', 'L', 'C' or 'R', then lda must be at least m.
     *                  If pack=='B' or 'Q', then lda must be at least MIN(kl, m-1) (which is equal
     *                      to MIN(ku,n-1)).
     *                  If pack=='Z', lda must be large enough to hold the packed array:
     *                      MIN(ku, n-1) + MIN(kl, m-1) + 1.
     *             work: an array, dimension (3*MAX(n, m))
     *                   Workspace.
     *             info: Error code. On exit, info will be set to one of the following values:
     *                     0 => normal return
     *                    -1 => m negative or unequal to n and sym=='S', 'H' or 'P'
     *                    -2 => n negative
     *                    -3 => dist illegal string
     *                    -5 => sym illegal string
     *                    -7 => mode not in range -6 to 6
     *                    -8 => cond less than 1.0, and mode neither -6, 0 nor 6
     *                   -10 => kl negative
     *                   -11 => ku negative, or sym=='S' or 'H' and ku not equal to kl
     *                   -12 => pack illegal string, or pack=='U' or 'L', and sym=='N';
     *                          or pack=='C' or 'Q' and sym=='N' and kl is not zero;
     *                          or pack=='R' or 'B' and sym=='N' and ku is not zero;
     *                          or pack=='U', 'L', 'C', 'R', 'B', or 'Q', and m is not n.
     *                   -14 => lda is less than m, or pack=='Z' and lda is less than
     *                          MIN(ku,n-1) + MIN(kl,m-1) + 1.
     *                     1 => Error return from dlatm1
     *                     2 => Cannot scale to dmax (max. sing. value is 0)
     *                     3 => Error return from dlagge or dlagsy
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void dlatms(int m, int n, const char* dist, int* iseed, const char* sym, real* d,
                       int mode, real cond, real dmax, int kl, int ku, char const* pack, real* A,
                       int lda, real* work, int& info)
    {
        //
        // 1)    Decode and Test the input parameters. Initialize flags & seed.
        //
        info = 0;
        // Quick return if possible
        if (m==0 || n==0)
        {
            return;
        }
        // Decode dist
        int idist;
        if (std::toupper(dist[0])=='U')
        {
            idist = 1;
        }
        else if (std::toupper(dist[0])=='S')
        {
            idist = 2;
        }
        else if (std::toupper(dist[0])=='N')
        {
            idist = 3;
        }
        else
        {
            idist = -1;
        }
        // Decode sym
        int irsign=0, isym;
        if (std::toupper(sym[0])=='N')
        {
            isym = 1;
            irsign = 0;
        }
        else if (std::toupper(sym[0])=='P')
        {
            isym = 2;
            irsign = 0;
        }
        else if (std::toupper(sym[0])=='S')
        {
            isym = 2;
            irsign = 1;
        }
        else if (std::toupper(sym[0])=='H')
        {
            isym = 2;
            irsign = 1;
        }
        else
        {
            isym = -1;
        }
        // Decode pack
        char uppack = std::toupper(pack[0]);
        int isympk = 0, ipack;
        switch (uppack)
        {
            case 'N':
                ipack = 0;
                break;
            case 'U':
                ipack = 1;
                isympk = 1;
                break;
            case 'L':
                ipack = 2;
                isympk = 1;
                break;
            case 'C':
                ipack = 3;
                isympk = 2;
                break;
            case 'R':
                ipack = 4;
                isympk = 3;
                break;
            case 'B':
                ipack = 5;
                isympk = 3;
                break;
            case 'Q':
                ipack = 6;
                isympk = 2;
                break;
            case 'Z':
                ipack = 7;
                break;
            default:
                ipack = -1;
                break;
        }
        // Set certain internal parameters
        int mnmin = std::min(m, n);
        int llb = std::min(kl, m-1);
        int uub = std::min(ku, n-1);
        int mr = std::min(m, n+llb);
        int nc = std::min(n, m+uub);
        int minlda;
        if (ipack==5 || ipack==6)
        {
            minlda = uub + 1;
        }
        else if (ipack==7)
        {
            minlda = llb + uub + 1;
        }
        else
        {
            minlda = m;
        }
        // Use Givens rotation method if bandwidth small enough, or if lda is too small to store
        // the matrix unpacked.
        bool givens = false;
        if (isym==1)
        {
            if (real(llb+uub) < real(0.3)*real(std::max(1, mr+nc)))
            {
                givens = true;
            }
        }
        else
        {
            if (2*llb<m)
            {
                givens = true;
            }
        }
        if (lda<m && lda>=minlda)
        {
            givens = true;
        }
        // Set info if an error
        if (m<0)
        {
            info = -1;
        }
        else if (m!=n && isym!=1)
        {
            info = -1;
        }
        else if (n<0)
        {
            info = -2;
        }
        else if (idist==-1)
        {
            info = -3;
        }
        else if (isym==-1)
        {
            info = -5;
        }
        else if (std::abs(mode)>6)
        {
            info = -7;
        }
        else if ((mode!=0 && std::abs(mode)!=6) && cond<ONE)
        {
            info = -8;
        }
        else if (kl<0)
        {
            info = -10;
        }
        else if (ku<0 || (isym!=1 && kl!=ku))
        {
            info = -11;
        }
        else if (ipack==-1 || (isympk==1 && isym==1) || (isympk==2 && isym==1 && kl>0)
                || (isympk==3 && isym==1 && ku>0) || (isympk!=0 && m!=n))
        {
            info = -12;
        }
        else if (lda<1 || lda<minlda)
        {
            info = -14;
        }
        if (info!=0)
        {
            this->xerbla("DLATMS", -info);
            return;
        }
        // Initialize random number generator
        int i;
        for (i=0; i<4; i++)
        {
            iseed[i] = std::abs(iseed[i]) % 4096;
        }
        if ((iseed[3] % 2)!=1)
        {
            iseed[3]++;
        }
        //
        // 2)    Set up d if indicated.
        //
        // Compute d according to cond and mode
        int iinfo;
        this->dlatm1(mode, cond, irsign, idist, iseed, d, mnmin, iinfo);
        if (iinfo!=0)
        {
            info = 1;
            return;
        }
        // Choose Top-Down if d is (apparently) increasing,
        // Bottom-Up if d is (apparently) decreasing.
        bool topdwn;
        if (std::fabs(d[0])<=std::fabs(d[mnmin-1]))
        {
            topdwn = true;
        }
        else
        {
            topdwn = false;
        }
        real alpha, temp;
        if (mode!=0 && std::abs(mode)!=6)
        {
            // Scale by dmax
            temp = std::fabs(d[0]);
            for (i=1; i<mnmin; i++)
            {
                temp = std::max(temp, std::fabs(d[i]));
            }
            if (temp>ZERO)
            {
                alpha = dmax / temp;
            }
            else
            {
                info = 2;
                return;
            }
            Blas<real>::dscal(mnmin, alpha, d, 1);
        }
        //
        // 3)    Generate Banded Matrix using Givens rotations. Also the special case of UUB=LLB=0
        //
        // Compute Addressing constants to cover all storage formats.  Whether GE, SY, GB, or SB,
        // upper or lower triangle or both, the (i,j)-th element is in A[i-j+ioffst-iskew*j,j]
        int ilda, ioffst, iskew;
        if (ipack>4)
        {
            ilda = lda - 1;
            iskew = 0;
            if (ipack>5)
            {
                ioffst = uub;
            }
            else
            {
                ioffst = 0;
            }
        }
        else
        {
            ilda = lda;
            iskew = -1;
            ioffst = 0;
        }
        int ldamiskm1 = lda - iskew - 1;
        // IPACKG is the format that the matrix is generated in. If this is different from IPACK,
        // then the matrix must be repacked at the end. It also signals how to compute the norm,
        // for scaling.
        int ipackg = 0;
        this->dlaset("Full", lda, n, ZERO, ZERO, A, lda);
        bool ilextr, iltemp;
        int icol=0, il, irow=0, itemp1, itemp2, jc, jch, jr;
        real angle, c, dummy, extra, s;
        if (llb==0 && uub==0)
        {
            // Diagonal Matrix: We are done, unless it is to be stored SP/PP/TP (pack=='R' or 'C')
            Blas<real>::dcopy(mnmin, d, 1, &A[ioffst/*+lda*0*/], ilda+1);
            if (ipack<=2 || ipack>=5)
            {
                ipackg = ipack;
            }
        }
        else if (givens)
        {
            // Check whether to use Givens rotations, Householder transformations, or nothing.
            if (isym==1)
            {
                // Non-symmetric -- A = U D V
                if (ipack>4)
                {
                    ipackg = ipack;
                }
                else
                {
                    ipackg = 0;
                }
                Blas<real>::dcopy(mnmin, d, 1, &A[ioffst/*+lda*0*/], ilda+1);
                int ic, ir, jkl, jku;
                if (topdwn)
                {
                    jkl = -1;
                    for (jku=0; jku<uub; jku++)
                    {
                        // Transform from bandwidth jkl+1, jku to jkl+1, jku+1
                        // Last row actually rotated is m-1
                        // Last column actually rotated is MIN(m+jku, n-1)
                        itemp1 = jkl + jku;
                        for (jr=0; jr<=m+itemp1 && jr<n+jkl; jr++)
                        {
                            extra = ZERO,
                            angle = TWOPI * dlarnd(1, iseed);
                            c = std::cos(angle);
                            s = std::sin(angle);
                            icol = std::max(0, jr-jkl-1);
                            if (jr<m-1)
                            {
                                il = std::min(n-1, jr+jku+1) - icol;
                                dlarot(true, jr>jkl, false, il+1, c, s,
                                       &A[jr+ioffst+ldamiskm1*icol], ilda, extra, dummy);
                            }
                            // Chase "EXTRA" back up
                            ir = jr;
                            ic = icol;
                            for (jch=jr-jkl-1; jch>=0; jch-=(itemp1+2))
                            {
                                if (ir<m-1)
                                {
                                    this->dlartg(A[ir+ioffst+1+ldamiskm1*(ic+1)], extra, c, s,
                                                 dummy);
                                }
                                irow = jch - jku - 1;
                                if (irow<0)
                                {
                                    irow = 0;
                                }
                                il = ir + 1 - irow;
                                temp = ZERO;
                                iltemp = (jch>jku);
                                dlarot(false, iltemp, true, il+1, c, -s,
                                       &A[irow+ioffst+ldamiskm1*ic], ilda, temp, extra);
                                if (iltemp)
                                {
                                    this->dlartg(A[irow+ioffst+1+ldamiskm1*(ic+1)], temp, c, s,
                                                 dummy);
                                    icol = jch - itemp1 - 2;
                                    if (icol<0)
                                    {
                                        icol = 0;
                                    }
                                    il = ic + 1 - icol;
                                    extra = ZERO;
                                    dlarot(true, jch>itemp1+1, true, il+1, c, -s,
                                           &A[irow+ioffst+ldamiskm1*icol], ilda, extra, temp);
                                    ic = icol;
                                    ir = irow;
                                }
                            }
                        }
                    }
                    jku = uub-1;
                    for (jkl=0; jkl<llb; jkl++)
                    {
                        // Transform from bandwidth jkl, jku+1 to jkl+1, jku+1
                        itemp1 = jkl + jku;
                        for (jc=0; jc<=n+itemp1 && jc<m+jku; jc++)
                        {
                            extra = ZERO;
                            angle = TWOPI * dlarnd(1, iseed);
                            c = std::cos(angle);
                            s = std::sin(angle);
                            irow = std::max(0, jc-jku-1);
                            if (jc<n-1)
                            {
                                il = std::min(m-1, jc+jkl+1) - irow;
                                dlarot(false, jc>jku, false, il+1, c, s,
                                       &A[irow+ioffst+ldamiskm1*jc], ilda, extra, dummy);
                            }
                            // Chase "EXTRA" back up
                            ic = jc;
                            ir = irow;
                            for (jch=jc-jku-1; jch>=0; jch-=(itemp1+2))
                            {
                                if (ic<n-1)
                                {
                                    this->dlartg(A[ir+1+ioffst+ldamiskm1*(ic+1)], extra, c, s,
                                                 dummy);
                                }
                                icol = jch - jkl - 1;
                                if (icol<0)
                                {
                                    icol = 0;
                                }
                                il = ic + 1 - icol;
                                temp = ZERO;
                                iltemp = (jch>jkl);
                                dlarot(true, iltemp, true, il+1, c, -s,
                                       &A[ir+ioffst+ldamiskm1*icol], ilda, temp, extra);
                                if (iltemp)
                                {
                                    this->dlartg(A[ir+1+ioffst+ldamiskm1*(icol+1)], temp, c, s, dummy);
                                    irow = jch - itemp1 - 2;
                                    if (irow<0)
                                    {
                                        irow = 0;
                                    }
                                    il = ir + 1 - irow;
                                    extra = ZERO;
                                    dlarot(false, jch>itemp1+1, true, il+1, c, -s,
                                           &A[irow+ioffst+ldamiskm1*icol], ilda, extra, temp);
                                    ic = icol;
                                    ir = irow;
                                }
                            }
                        }
                    }
                }
                else
                {
                    // Bottom-Up -- Start at the bottom right.
                    jkl = -1;
                    int iendch;
                    for (jku=0; jku<uub; jku++)
                    {
                        // Transform from bandwidth jkl+1, jku to jkl+1, jku+1
                        // First row actually rotated is m-1
                        // First column actually rotated is MIN(m+jku,n-1)
                        iendch = std::min(m-2, n-1+jkl);
                        for (jc=std::min(m+jku-1, n-2); jc>=-(jkl+1); jc--)
                        {
                            extra = ZERO;
                            angle = TWOPI * dlarnd(1, iseed);
                            c = std::cos(angle);
                            s = std::sin(angle);
                            irow = std::max(0, jc-jku);
                            if (jc>=0)
                            {
                                il = std::min(m-1, jc+jkl+2) - irow;
                                dlarot(false, false, jc+jkl<m-2, il+1, c, s,
                                       &A[irow+ioffst+ldamiskm1*jc], ilda, dummy, extra);
                            }
                            // Chase "EXTRA" back down
                            ic = jc;
                            for (jch=jc+jkl+1; jch<=iendch; jch+=(jkl+jku+2))
                            {
                                ilextr = (ic>=0);
                                if (ilextr)
                                {
                                    this->dlartg(A[jch+ioffst+ldamiskm1*ic], extra, c, s, dummy);
                                }
                                if (0>ic)
                                {
                                    ic = 0;
                                }
                                icol = std::min(n-2, jch+jku+1);
                                iltemp = (jch+jku < n-2);
                                temp = ZERO;
                                dlarot(true, ilextr, iltemp, icol+2-ic, c, s,
                                       &A[jch+ioffst+ldamiskm1*ic], ilda, extra, temp);
                                if (iltemp)
                                {
                                    this->dlartg(A[jch+ioffst+ldamiskm1*icol], temp, c, s, dummy);
                                    il = std::min(iendch-jch+1, 3+jkl+jku);
                                    extra = ZERO;
                                    dlarot(false, true, jch+jkl+jku+1<iendch, il+1, c, s,
                                           &A[jch+ioffst+ldamiskm1*icol], ilda, temp, extra);
                                    ic = icol;
                                }
                            }
                        }
                    }
                    jku = uub-1;
                    for (jkl=0; jkl<llb; jkl++)
                    {
                        // Transform from bandwidth jkl-1, jku to jkl, jku
                        // First row actually rotated is MIN(n+jkl, m-1)
                        // First column actually rotated is n-1
                        iendch = std::min(n-2, m-1+jku);
                        itemp1 = std::min(n+jkl-1, m-2);
                        for (jr=itemp1; jr>=-jku-1; jr--)
                        {
                            extra = ZERO;
                            angle = TWOPI * dlarnd(1, iseed);
                            c = std::cos(angle);
                            s = std::sin(angle);
                            icol = std::max(0, jr-jkl);
                            if (jr>=0)
                            {
                                il = std::min(n-1, jr+jku+2) - icol;
                                dlarot(true, false, jr+jku<n-2, il+1, c, s,
                                       &A[jr+ioffst+ldamiskm1*icol], ilda, dummy, extra);
                            }
                            // Chase "EXTRA" back down
                            ir = jr;
                            for (jch=jr+jku+1; jch<=iendch; jch+=(jkl+jku+2))
                            {
                                ilextr = (ir>=0);
                                if (ilextr)
                                {
                                    this->dlartg(A[ir+ioffst+ldamiskm1*jch], extra, c, s, dummy);
                                }
                                if (0>ir)
                                {
                                    ir = 0;
                                }
                                irow = std::min(m-2, jch+jkl+1);
                                iltemp = ((jch+jkl+2) < m);
                                temp = ZERO;
                                dlarot(false, ilextr, iltemp, irow+2-ir, c, s,
                                       &A[ir+ioffst+ldamiskm1*jch], ilda, extra, temp);
                                if (iltemp)
                                {
                                    this->dlartg(A[irow+ioffst+ldamiskm1*jch], temp, c, s, dummy);
                                    il = std::min(iendch+1-jch, 3+jkl+jku);
                                    extra = ZERO;
                                    dlarot(true, true, jch+jkl+jku+1<iendch, il+1, c, s,
                                           &A[irow+ioffst+ldamiskm1*jch], ilda, temp, extra);
                                    ir = irow;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // Symmetric -- A = U D U'
                ipackg = ipack;
                int ioffg = ioffst;
                int k;
                if (topdwn)
                {
                    // Top-Down -- Generate Upper triangle only
                    if (ipack>=5)
                    {
                        ipackg = 6;
                        ioffg = uub-iskew;
                    }
                    else
                    {
                        ipackg = 1;
                    }
                    Blas<real>::dcopy(mnmin, d, 1, &A[ioffg/*+lda*0*/], ilda+1);
                    for (k=0; k<uub; k++)
                    {
                        itemp1 = k+2;
                        for (jc=0; jc<n-1; jc++)
                        {
                            irow = jc-k-1;
                            if (irow<0)
                            {
                                irow = 0;
                            }
                            il = jc+1;
                            if (il>itemp1)
                            {
                                il = itemp1;
                            }
                            extra = ZERO;
                            temp = A[jc+ioffg+ldamiskm1*(jc+1)];
                            angle = TWOPI * dlarnd(1, iseed);
                            c = std::cos(angle);
                            s = std::sin(angle);
                            dlarot(false, jc>k, true, il+1, c, s, &A[irow+ioffg+ldamiskm1*jc],
                                   ilda, extra, temp);
                            dlarot(true, true, false, std::min(itemp1, n-jc), c, s,
                                   &A[ioffg+(lda-iskew)*jc], ilda, temp, dummy);
                            // Chase EXTRA back up the matrix
                            icol = jc;
                            for (jch=jc-k-1; jch>=0; jch-=(k+1))
                            {
                                this->dlartg(A[jch+1+ioffg+ldamiskm1*(icol+1)], extra, c, s, dummy);
                                temp = A[jch+ioffg+ldamiskm1*(jch+1)];
                                dlarot(true, true, true, k+3, c, -s, &A[ioffg+(lda-iskew)*jch],
                                       ilda, temp, extra);
                                irow = std::max(0, jch-k-1);
                                il = std::min(jch+1, itemp1);
                                extra = ZERO;
                                dlarot(false, jch>k, true, il+1, c, -s,
                                       &A[irow+ioffg+ldamiskm1*jch], ilda, extra, temp);
                                icol = jch;
                            }
                        }
                    }
                    // If we need lower triangle, copy from upper. Note that the order of copying
                    // is chosen to work for 'q' -> 'b'
                    if (ipack!=ipackg && ipack!=3)
                    {
                        for (jc=0; jc<n; jc++)
                        {
                            irow = ioffst - (iskew+1)*jc - 1;
                            itemp1 = 1 + irow + lda*jc;
                            itemp2 = jc + ioffg;
                            for (jr=jc; jr<n && jr<=(jc+uub); jr++)
                            {
                                A[jr+itemp1] = A[itemp2+ldamiskm1*jr];
                            }
                        }
                        if (ipack==5)
                        {
                            for (jc=n-uub; jc<n; jc++)
                            {
                                itemp1 = lda * jc;
                                for (jr=n-jc; jr<=uub; jr++)
                                {
                                    A[jr+itemp1] = ZERO;
                                }
                            }
                        }
                        if (ipackg==6)
                        {
                            ipackg = ipack;
                        }
                        else
                        {
                            ipackg = 0;
                        }
                    }
                }
                else
                {
                    // Bottom-Up -- Generate Lower triangle only
                    if (ipack>=5)
                    {
                        ipackg = 5;
                        if (ipack==6)
                        {
                            ioffg = -iskew;
                        }
                    }
                    else
                    {
                        ipackg = 2;
                    }
                    Blas<real>::dcopy(mnmin, d, 1, &A[ioffg/*+lda*0*/], ilda+1);
                    for (k=0; k<uub; k++)
                    {
                        for (jc=n-2; jc>=0; jc--)
                        {
                            il = std::min(n-jc-1, k+2);
                            extra = ZERO;
                            temp = A[ioffg+1+(lda-iskew)*jc];
                            angle = TWOPI * dlarnd(1, iseed);
                            c = std::cos(angle);
                            s = -std::sin(angle);
                            dlarot(false, true, n-2-jc>k, il+1, c, s, &A[ioffg+(lda-iskew)*jc],
                                   ilda, temp, extra);
                            icol = std::max(0, jc-k);
                            dlarot(true, false, true, jc+2-icol, c, s,
                                   &A[jc+ioffg+ldamiskm1*icol], ilda, dummy, temp);
                            // Chase EXTRA back down the matrix
                            icol = jc;
                            for (jch=jc+k+1; jch<n-1; jch+=(k+1))
                            {
                                this->dlartg(A[jch+ioffg+ldamiskm1*icol], extra, c, s, dummy);
                                temp = A[ioffg+1+(lda-iskew)*jch];
                                dlarot(true, true, true, k+3, c, s, &A[jch+ioffg+ldamiskm1*icol],
                                       ilda, extra, temp);
                                il = std::min(n-1-jch, k+2);
                                extra = ZERO;
                                dlarot(false, true, n-2-jch>k, il+1, c, s,
                                       &A[ioffg+(lda-iskew)*jch], ilda, temp, extra);
                                icol = jch;
                            }
                        }
                    }
                    // If we need upper triangle, copy from lower. Note that the order of copying
                    // is chosen to work for 'b' -> 'q'
                    if (ipack!=ipackg && ipack!=4)
                    {
                        for (jc=n-1; jc>=0; jc--)
                        {
                            irow = ioffst - (iskew+1)*jc - 1;
                            itemp1 = 1 + irow + lda*jc;
                            itemp2 = jc + ioffg;
                            for (jr=jc; jr>=0 && jr>=(jc-uub); jr--)
                            {
                                A[jr+itemp1] = A[itemp2+ldamiskm1*jr];
                            }
                        }
                        if (ipack==6)
                        {
                            for (jc=0; jc<uub; jc++)
                            {
                                itemp1 = lda*jc;
                                for (jr=0; jr<uub-jc; jr++)
                                {
                                    A[jr+itemp1] = ZERO;
                                }
                            }
                        }
                        if (ipackg==5)
                        {
                            ipackg = ipack;
                        }
                        else
                        {
                            ipackg = 0;
                        }
                    }
                }
            }
        }
        else
        {
            //
            // 4)    Generate Banded Matrix by first rotating by random Unitary matrices, then
            //       reducing the bandwidth using Householder transformations.
            //       Note: we should get here only if lda>=n
            //
            if (isym==1)
            {
                // Non-symmetric -- A = U D V
                dlagge(mr, nc, llb, uub, d, A, lda, iseed, work, iinfo);
            }
            else
            {
                // Symmetric -- A = U D U'
                dlagsy(m, llb, d, A, lda, iseed, work, iinfo);
            }
            if (iinfo!=0)
            {
                info = 3;
                return;
            }
        }
        //
        // 5)    Pack the matrix
        //
        int ir1, ir2, j;
        if (ipack!=ipackg)
        {
            if (ipack==1)
            {
                // 'U' -- Upper triangular, not packed
                for (j=0; j<m; j++)
                {
                    itemp1 = lda * j;
                    for (i=j+1; i<m; i++)
                    {
                        A[i+itemp1] = ZERO;
                    }
                }
            }
            else if (ipack==2)
            {
                // 'L' -- Lower triangular, not packed
                for (j=1; j<m; j++)
                {
                    itemp1 = lda * j;
                    for (i=0; i<j; i++)
                    {
                        A[i+itemp1] = ZERO;
                    }
                }
            }
            else if (ipack==3)
            {
                // 'C' -- Upper triangle packed Columnwise.
                icol = 0;
                irow = -1;
                for (j=0; j<m; j++)
                {
                    itemp1 = lda * j;
                    for (i=0; i<=j; i++)
                    {
                        irow++;
                        if (irow>=lda)
                        {
                            irow = 0;
                            icol++;
                        }
                        A[irow+lda*icol] = A[i+itemp1];
                    }
                }
            }
            else if (ipack==4)
            {
                // 'R' -- Lower triangle packed Columnwise.
                icol = 0;
                irow = -1;
                for (j=0; j<m; j++)
                {
                    itemp1 = lda * j;
                    for (i=j; i<m; i++)
                    {
                        irow++;
                        if (irow>=lda)
                        {
                            irow = 0;
                            icol++;
                        }
                        A[irow+lda*icol] = A[i+itemp1];
                    }
                }
            }
            else if (ipack>=5)
            {
                // 'B' -- The lower triangle is packed as a band matrix.
                // 'Q' -- The upper triangle is packed as a band matrix.
                // 'Z' -- The whole matrix is packed as a band matrix.
                if (ipack==5)
                {
                    uub = 0;
                }
                if (ipack==6)
                {
                    llb = 0;
                }
                for (j=0; j<uub; j++)
                {
                    itemp1 = std::min(j+llb, m-1);
                    itemp2 = -j + uub + lda*j;
                    for (i=itemp1; i>=0; i--)
                    {
                        A[i+itemp2] = A[i+lda*j];
                    }
                }
                for (j=uub+1; j<n; j++)
                {
                    itemp1 = lda * j;
                    itemp2 = -j + uub + itemp1;
                    for (i=j-uub; i<=(j+llb) && i<m; i++)
                    {
                        A[i+itemp2] = A[i+itemp1];
                    }
                }
            }
            // If packed, ZERO out extraneous elements.
            // Symmetric/Triangular Packed -- ZERO out everything after A[irow,icol]
            if (ipack==3 || ipack==4)
            {
                for (jc=icol; jc<m; jc++)
                {
                    itemp1 = lda * jc;
                    for (jr=irow+1; jr<lda; jr++)
                    {
                        A[jr+itemp1] = ZERO;
                    }
                    irow = -1;
                }
            }
            else if (ipack>=5)
            {
                // Packed Band --1st row is now in A[UUB-j,j], ZERO above it
                //               m-th row is now in A[m-2+UUB-j,j], ZERO below it
                //               last non-ZERO diagonal is now in A[UUB+LLB,j], ZERO below it, too.
                ir1 = uub + llb + 1;
                ir2 = uub + m + 1;
                for (jc=0; jc<n; jc++)
                {
                    itemp1 = lda * jc;
                    for (jr=0; jr<uub-jc; jr++)
                    {
                        A[jr+itemp1] = ZERO;
                    }
                    itemp2 = ir2 - jc - 1;
                    if (ir1<itemp2)
                    {
                        itemp2 = ir1;
                    }
                    if (itemp2<0)
                    {
                        itemp2 = 0;
                    }
                    for (jr=itemp2; jr<lda; jr++)
                    {
                        A[jr+itemp1] = ZERO;
                    }
                }
            }
        }
    }
};
#endif