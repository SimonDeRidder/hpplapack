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

#include "Blas.hpp"


/*!\class Lapack
 * \brief A template class containing LAPACK routines.
 * Lapack contains the LAPACK routines as static members.
 * Any routine can be called using Lapack<type>::routine(...).
 * The template type is meant to be double, but can be any floating point type
 *
 * \author Simon De Ridder                                                                       */
template<class real>
class Lapack
{
private:
	// constants

	static constexpr real NEGONE= real(-1.0);  //!< A constant negative one       (-1.0)  value
	static constexpr real MEIGTH= real(-0.125);//!< A constant negative one eigth (-0.125)value
	static constexpr real ZERO  = real(0.0);   //!< A constant zero               (0.0)   value
	static constexpr real QURTR = real(0.25);  //!< A constant one quarter        (0.25)  value
	static constexpr real HALF  = real(0.5);   //!< A constant one half           (0.5)   value
	static constexpr real ONE   = real(1.0);   //!< A constant one                (1.0)   value
	static constexpr real TWO   = real(2.0);   //!< A constant two                (2.0)   value
	static constexpr real THREE = real(3.0);   //!< A constant three              (3.0)   value
	static constexpr real FOUR  = real(4.0);   //!< A constant four               (4.0)   value
	static constexpr real EIGHT = real(8.0);   //!< A constant eight              (8.0)   value
	static constexpr real TEN   = real(10.0);  //!< A constant ten                (10.0)  value
	static constexpr real HNDRD = real(100.0); //!< A constant one hundred        (100.0) value

public:
	// LAPACK INSTALL (alphabetically)

	/*! §dlamc3
	 *
	 * §dlamc3 is intended to force §A and §B to be stored prior to doing the addition of §A and
	 * §B, for use in situations where optimizers might hold one of these in a register.
	 * \param[in] A
	 * \param[in] B The values §A and §B.
	 * \return The sum of §A and §B
	 * \authors
	 *     LAPACK is a software package provided by Univ. of Tennessee,
	 *     Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.
	 * \date December 2016                                                                       */
	static real dlamc3(real const A, real const B) /*const*/
	{
		return A + B;
	}

	/*! §dlamch
	 *
	 * §dlamch determines double precision machine parameters.
	 * \param[in] cmach §cmach[0] Specifies the value to be returned by §dlamch: \n
	 *                  'E' or 'e': eps:   Relative machine precision.\n
	 *                  'S' or 's': sfmin: Safe minimum, such that 1/sfmin does not overflow.\n
	 *                  'B' or 'b': base:  Base of the machine (radix).\n
	 *                  'P' or 'p': prec:  $\{eps}\cdot\{base}$.\n
	 *                  'N' or 'n': t:     Number of(base) digits in the mantissa.\n
	 *                  'R' or 'r': rnd:   1.0 when rounding occurs in addition, 0.0 otherwise.\n
	 *                  'M' or 'm': emin:  Minimum exponent before(gradual) underflow.\n
	 *                  'U' or 'u': rmin:  Underflow threshold: $\{base}^{\{emin}-1}$.\n
	 *                  'L' or 'l': emax:  Largest exponent before overflow.\n
	 *                  'O' or 'o': rmax:  Overflow threshold: $(\{base}^\{emax})(1-\{eps})$.
	 * \return The value specified by §cmach, or zero when §cmach is not recognized.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static real dlamch(char const* const cmach) /*const*/
	{
		// Assume rounding, not chopping.Always.
		real rnd = ONE;
		real eps;
		if (ONE==rnd)
		{
			eps = std::numeric_limits<real>::epsilon() * HALF;
		}
		else
		{
			eps = std::numeric_limits<real>::epsilon();
		}
		real sfmin, small;
		switch (std::toupper(cmach[0]))
		{
			case 'E':
				return eps;
			case 'S':
				sfmin = std::numeric_limits<real>::min();
				small = ONE / std::numeric_limits<real>::max();
				if (small>=sfmin)
				{
					//Use SMALL plus a bit, to avoid the possibility of rounding
					// causing overflow when computing  1/sfmin.
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

	/* §dsecnd replaced inline by std::time */

	/*! §ilaver returns the LAPACK version.
	 *
	 * This subroutine returns the LAPACK version.
	 * \param[out] vers_major return the lapack major version
	 * \param[out] vers_minor return the lapack minor version from the major version.
	 * \param[out] vers_patch return the lapack patch version from the minor version
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017                                                                           */
	static void ilaver(int& vers_major, int& vers_minor, int& vers_patch) /*const*/
	{
		vers_major = 3;
		vers_minor = 8;
		vers_patch = 0;
	}

	// LAPACK SRC (alphabetically)

	/*! §dbdsdc
	 *
	 * §dbdsdc computes the singular value decomposition (SVD) of a real §n by §n (upper or lower)
	 * bidiagonal matrix $B$: $B = U S V_T$, using a divide and conquer method, where $S$ is a
	 * diagonal matrix with non-negative diagonal elements (the singular values of $B$), and $U$
	 * and $V_T$ are orthogonal matrices of left and right singular vectors, respectively. §dbdsdc
	 * can be used to compute all singular values, and optionally, singular vectors or singular
	 * vectors in compact form.\n
	 * This code makes very mild assumptions about floating point arithmetic. It will work on
	 * machines with a guard digit in add/subtract, or on those binary machines without guard
	 * digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2. It could
	 * conceivably fail on hexadecimal or decimal machines without guard digits, but we know of
	 * none. See §dlasd3 for details.\n
	 * The code currently calls §dlasdq if singular values only are desired. However, it can be
	 * slightly modified to compute singular values using the divide and conquer method.
	 * \param[in] uplo
	 *     = 'U': $B$ is upper bidiagonal.\n
	 *     = 'L': $B$ is lower bidiagonal.
	 *
	 * \param[in] compq
	 *     Specifies whether singular vectors are to be computed as follows:\n
	 *         = 'N': Compute singular values only;\n
	 *         = 'P': Compute singular values and compute singular vectors in compact form;\n
	 *         = 'I': Compute singular values and singular vectors.
	 *
	 * \param[in]     n The order of the matrix $B$. $\{n}\ge 0$.
	 * \param[in,out] d
	 *     an array, dimension (§n)\n
	 *     On entry, the §n diagonal elements of the bidiagonal matrix $B$.\n
	 *     On exit, if §info = 0, the singular values of $B$.
	 *
	 * \param[in,out] e
	 *     an array, dimension ($\{n}-1$)\n
	 *     On entry, the elements of §e contain the offdiagonal elements of the bidiagonal matrix
	 *     whose SVD is desired.\n
	 *     On exit, §e has been destroyed.
	 *
	 * \param[out] U
	 *     an array, dimension (§ldu,§n)\n.
	 *     If §compq = 'I', then:\n On exit, if §info = 0, §U contains the left singular vectors of
	 *     the bidiagonal matrix.\n For other values of §compq, §U is not referenced.
	 *
	 * \param[in] ldu
	 *     The leading dimension of the array §U. $\{ldu}\ge 1$.\n
	 *     If singular vectors are desired, then $\{ldu}\ge\max(1,\{n})$.
	 *
	 * \param[out] Vt
	 *     an array, dimension (§ldvt,§n)\n
	 *     If §compq = 'I', then:\n On exit, if §info = 0, $\{Vt}^T$ contains the right singular
	 *     vectors of the bidiagonal matrix.\n For other values of §compq, §Vt is not referenced.
	 *
	 * \param[in] ldvt
	 *     The leading dimension of the array §Vt. $\{ldvt}\ge 1$.\n
	 *     If singular vectors are desired, then $\{ldvt}\ge\max(1,\{n})$.
	 *
	 * \param[out] q
	 *     an array, dimension (§ldq)\n
	 *     If §compq = 'P', then:\n
	 *         On exit, if §info = 0, §q and §iq contain the left and right singular vectors in a
	 *         compact form, requiring $\mathcal{O}(\{n}\log\{n})$ space instead of $2\{n}^2$.
	 *         In particular, §q contains all the real data in
	 *         $\{ldq}\ge\{n}*(11+2\{smlsiz}+8\lfloor\log_2(\{n}/(\{smlsiz}+1))\rfloor)$ words of
	 *         memory, where §smlsiz is returned by §ilaenv and is equal to the maximum size of the
	 *         subproblems at the bottom of the computation tree (usually about 25).\n
	 *     For other values of §compq, §q is not referenced.
	 *
	 * \param[out] iq
	 *     an integer array, dimension (§LDIQ)\n
	 *     If §compq = 'P', then:\n
	 *         On exit, if §info = 0, §q and §iq contain the left and right singular vectors in a
	 *         compact form, requiring $\mathcal{O}(\{n}\log\{n})$ space instead of $2\{n}^2$.
	 *         In particular, §iq contains all integer data in
	 *         $\{LDIQ}\ge\{n}(3+3\lfloor\log_2(\{n}/(\{smlsiz}+1))\rfloor)$ words of memory, where
	 *         §smlsiz is returned by §ilaenv and is equal to the maximum size of the subproblems
	 *         at the bottom of the computation tree (usually about 25).\n
	 *     For other values of §compq, §iq is not referenced.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$)\n
	 *         If §compq = 'N' then $\{lwork}\ge(4\{n})$.\n
	 *         If §compq = 'P' then $\{lwork}\ge(6\{n})$.\n
	 *         If §compq = 'I' then $\{lwork}\ge(3\{n}^2+4\{n})$.
	 *
	 * \param[out] iwork an integer array, dimension ($8\{n}$)
	 * \param[out] info
	 *     = 0: successful exit.\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.\n
	 *     > 0: The algorithm failed to compute a singular value.
	 *          The update process of divide and conquer failed.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016
	 * \remark
	 *     Contributors:\n
	 *     Ming Gu and Huan Ren, Computer Science Division,
	 *     University of California at Berkeley, USA                                             */
	static void dbdsdc(char const* const uplo, char const* const compq, int const n, real* const d,
	                   real* const e, real* const U, int const ldu, real* const Vt, int const ldvt,
	                   real* const q, int* const iq, real* const work, int* const iwork, int& info)
	                   /*const*/
	{
		// Test the input parameters.
		info = 0;
		int icompq, iuplo = 0;
		if (std::toupper(uplo[0])=='U')
		{
			iuplo = 1;
		}
		if (std::toupper(uplo[0])=='L')
		{
			iuplo = 2;
		}
		if (std::toupper(compq[0])=='N')
		{
			icompq = 0;
		}
		else if (std::toupper(compq[0])=='P')
		{
			icompq = 1;
		}
		else if (std::toupper(compq[0])=='I')
		{
			icompq = 2;
		}
		else
		{
			icompq = -1;
		}
		if (iuplo==0)
		{
			info = -1;
		}
		else if (icompq<0)
		{
			info = -2;
		}
		else if (n<0)
		{
			info = -3;
		}
		else if (ldu<1 || (icompq==2 && ldu<n))
		{
			info = -7;
		}
		else if (ldvt<1 || (icompq==2 && ldvt<n))
		{
			info = -9;
		}
		if (info!=0)
		{
			xerbla("DBDSDC", -info);
			return;
		}
		// Quick return if possible
		if (n==0)
		{
			return;
		}
		int smlsiz = ilaenv(9, "DBDSDC", " ", 0, 0, 0, 0);
		if (n==1)
		{
			if (icompq==1)
			{
				q[0] = std::copysign(ONE, d[0]);
				q[smlsiz*n] = ONE;
			}
			else if (icompq==2)
			{
				U[0] = std::copysign(ONE, d[0]);
				Vt[0] = ONE;
			}
			d[0] = std::fabs(d[0]);
			return;
		}
		int nm1 = n - 1;
		int nm2 = n - 2;
		// If matrix lower bidiagonal,
		// rotate to be upper bidiagonal by applying Givens rotations on the left
		int wstart = 0;
		int qstart = 2;
		if (icompq==1)
		{
			Blas<real>::dcopy(n,   d, 1, &q[0], 1);
			Blas<real>::dcopy(nm1, e, 1, &q[n], 1);
		}
		int i;
		real r;
		if (iuplo==2)
		{
			qstart = 4;
			if (icompq == 2)
			{
				wstart = 2 * nm1;
			}
			real cs, sn;
			for (i=0; i<nm1; i++)
			{
				dlartg(d[i], e[i], cs, sn, r);
				d[i]   = r;
				e[i]   = sn * d[i+1];
				d[i+1] = cs * d[i+1];
				if (icompq==1)
				{
					q[i+2*n] = cs;
					q[i+3*n] = sn;
				}
				else if (icompq==2)
				{
					work[i]     =  cs;
					work[nm1+i] = -sn;
				}
			}
		}
		// If icompq = 0, use dlasdq to compute the singular values.
		if (icompq==0)
		{
			/* Ignore wstart, instead using work, since the two vectors for cs and -sn above are
			 * added only if icompq == 2, and adding them exceeds documented work size of 4*n.   */
			dlasdq("U", 0, n, 0, 0, 0, d, e, Vt, ldvt, U, ldu, U, ldu, work, info);
		}
		else
		{
			// If n is smaller than the minimum divide size smlsiz,
			// then solve the problem with another solver.
			int iu, ivt=0;
			if (n<=smlsiz)
			{
				if (icompq==2)
				{
					dlaset("A", n, n, ZERO, ONE, U, ldu);
					dlaset("A", n, n, ZERO, ONE, Vt, ldvt);
					dlasdq("U", 0, n, n, n, 0, d, e, Vt, ldvt, U, ldu, U, ldu, &work[wstart],
					       info);
				}
				else if (icompq==1)
				{
					iu = 0;
					ivt = iu + n;
					dlaset("A", n, n, ZERO, ONE, &q[iu+qstart*n], n);
					dlaset("A", n, n, ZERO, ONE, &q[ivt+qstart*n], n);
					dlasdq("U", 0, n, n, n, 0, d, e, &q[ivt+qstart*n], n, &q[iu+qstart*n], n,
					       &q[iu+qstart*n], n, &work[wstart], info);
				}
			}
			else
			{
				if (icompq==2)
				{
					dlaset("A", n, n, ZERO, ONE, U, ldu);
					dlaset("A", n, n, ZERO, ONE, Vt, ldvt);
				}
				// Scale.
				real orgnrm = dlanst("M", n, d, e);
				if (orgnrm==ZERO)
				{
					return;
				}
				int ierr;
				dlascl("G", 0, 0, orgnrm, ONE, n, 1, d, n, ierr);
				dlascl("G", 0, 0, orgnrm, ONE, nm1, 1, e, nm1, ierr);
				real eps = real(0.9) * dlamch("Epsilon");
				int difl=0, difr=0, givcol=0, givnum=0, givptr, ic=0, is=0, k, perm, poles=0, z=0;
				if (icompq==1)
				{
					int mlvl = int(std::log(real(n)/real(smlsiz+1)) / std::log(TWO)) + 1;
					iu     = 0;
					ivt    = smlsiz;
					difl   = ivt + smlsiz + 1;
					difr   = difl + mlvl;
					z      = difr + mlvl*2;
					ic     = z + mlvl;
					is     = ic + 1;
					poles  = is + 1;
					givnum = poles + 2*mlvl;
					k      = 1;
					givptr = 2;
					perm   = 3;
					givcol = perm + mlvl;
				}
				for (i=0; i<n; i++)
				{
					if (std::fabs(d[i])<eps)
					{
						d[i] = std::copysign(eps, d[i]);
					}
				}
				int start = 0;
				int sqre = 0;
				int nsize;
				for (i=0; i<nm1; i++)
				{
					if (std::fabs(e[i])<eps || i==nm2)
					{
						// Subproblem found.
						// First determine its size and then apply divide and conquer on it.
						if (i<nm2)
						{
							// A subproblem with e[i] small for i < n - 2.
							nsize = i - start + 1;
						}
						else if (std::fabs(e[i])>=eps)
						{
							// A subproblem with e[n-2] not too small but i = n - 2.
							nsize = n - start;
						}
						else
						{
							// A subproblem with e[n-2] small. This implies an 1-by-1 subproblem at
							// d[n-1]. Solve this 1-by-1 problem first.
							nsize = i - start + 1;
							if (icompq==2)
							{
								U[nm1+ldu*nm1] = std::copysign(ONE, d[nm1]);
								Vt[nm1+ldvt*nm1] = ONE;
							}else if (icompq==1)
							{
								q[nm1+qstart*n] = std::copysign(ONE, d[nm1]);
								q[nm1+(smlsiz+qstart)*n] = ONE;
							}
							d[nm1] = std::fabs(d[nm1]);
						}
						if (icompq==2)
						{
							dlasd0(nsize, sqre, &d[start], &e[start], &U[start+ldu*start], ldu,
							       &Vt[start+ldvt*start], ldvt, smlsiz, iwork, &work[wstart],
							       info);
						}
						else
						{
							dlasda(icompq, smlsiz, nsize, sqre, &d[start], &e[start],
							       &q[start+(iu+qstart)*n], n, &q[start+(ivt+qstart)*n],
							       &iq[start+k*n], &q[start+(difl+qstart)*n],
							       &q[start+(difr+qstart)*n], &q[start+(z+qstart)*n],
							       &q[start+(poles+qstart)*n], &iq[start+givptr*n],
							       &iq[start+givcol*n], n, &iq[start+perm*n],
							       &q[start+(givnum+qstart)*n], &q[start+(ic+qstart)*n],
							       &q[start+(is+qstart)*n], &work[wstart], iwork, info);
						}
						if (info!=0)
						{
							return;
						}
						start = i + 1;
					}
				}
				// Unscale
				dlascl("G", 0, 0, ONE, orgnrm, n, 1, d, n, ierr);
			}
		}
		// Use Selection Sort to minimize swaps of singular vectors
		int ii, j, kk;
		real p;
		for (ii=1; ii<n; ii++)
		{
			i = ii - 1;
			kk = i;
			p = d[i];
			for (j=ii; j<n; j++)
			{
				if (d[j]>p)
				{
					kk = j;
					p = d[j];
				}
			}
			if (kk!=i)
			{
				d[kk] = d[i];
				d[i] = p;
				if (icompq==1)
				{
					iq[i] = kk+1;
				}
				else if (icompq==2)
				{
					Blas<real>::dswap(n, &U[ldu*i], 1, &U[ldu*kk], 1);
					Blas<real>::dswap(n, &Vt[i], ldvt, &Vt[kk], ldvt);
				}
			}
			else if (icompq==1)
			{
				iq[i] = i+1;
			}
		}
		// If icompq = 1, use iq[n-1] as the indicator for uplo
		if (icompq==1)
		{
			if (iuplo==1)
			{
				iq[nm1] = 1;
			}
			else
			{
				iq[nm1] = 0;
			}
		}
		// If B is lower bidiagonal,
		// update U by those Givens rotations which rotated B to be upper bidiagonal
		if (iuplo==2 && icompq==2)
		{
			dlasr("L", "V", "B", n, n, work, &work[nm1], U, ldu);
		}
	}


	/*! \fn dbdsqr
	 *
	 * \brief §dbdsqr
	 *
	 * \details §dbdsqr computes the singular values and, optionally, the right and/or left
	 * singular vectors from the singular value decomposition (SVD) of a real §n by §n (upper or
	 * lower) bidiagonal matrix $B$ using the implicit zero-shift QR algorithm. The SVD of $B$ has
	 * the form\n
	 *     $B = Q S P^T$\n
	 * where $S$ is the diagonal matrix of singular values, $Q$ is an orthogonal matrix of left
	 * singular vectors, and $P$ is an orthogonal matrix of right singular vectors. If left
	 * singular vectors are requested, this subroutine actually returns $UQ$ instead of $Q$, and,
	 * if right singular vectors are requested, this subroutine returns $P^TV_T$ instead of $P^T$,
	 * for given real input matrices $U$ and $V_T$. When $U$ and $V_T$ are the orthogonal matrices
	 * that reduce a general matrix $A$ to bidiagonal form: $A = U B V_T$, as computed by §dgebrd,
	 * then\n
	 *     $A = (UQ) S (P^TV_T)$\n
	 * is the SVD of $A$. Optionally, the subroutine may also compute $Q^TC$ for a given real input
	 * matrix $C$.n
	 * See "Computing Small Singular Values of Bidiagonal Matrices With Guaranteed High Relative
	 *     Accuracy," by J. Demmel and W. Kahan, LAPACK Working Note #3 (or SIAM J. Sci. Statist.
	 *     Comput. vol. 11, no. 5, pp. 873-912, Sept 1990)\n
	 * and "Accurate singular values and differential qd algorithms," by B. Parlett and
	 *     V. Fernando, Technical Report CPAM-554, Mathematics Department, University of California
	 *     at Berkeley, July 1992\n
	 * for a detailed description of the algorithm.
	 * \param[in] uplo
	 *     'U': $B$ is upper bidiagonal.\n
	 *     'L': $B$ is lower bidiagonal.
	 *
	 * \param[in] n    The order of the matrix $B$. $\{n} \ge 0$.
	 * \param[in] ncvt The number of columns of the matrix $V_T$. $\{ncvt} \ge 0$.
	 * \param[in] nru  The number of rows of the matrix $U$. $\{nru} \ge 0$.
	 * \param[in] ncc  The number of columns of the matrix $C$. $\{ncc} \ge 0$.
	 * \param[in,out] d
	 *     an array, dimension (§n)\n
	 *     On entry, the §n diagonal elements of the bidiagonal matrix $B$.\n
	 *     On exit, if §info = 0, the singular values of $B$ in decreasing order.
	 *
	 * \param[in,out] e
	 *     an array, dimension (§n-1)\n
	 *     On entry, the §n-1 offdiagonal elements of the bidiagonal matrix $B$.\n
	 *     On exit, if §info = 0, §e is destroyed; if §info > 0, §d and §e will contain the
	 *     diagonal and superdiagonal elements of a bidiagonal matrix orthogonally equivalent to
	 *     the one given as input.
	 *
	 * \param[in,out] Vt
	 *     an array, dimension (§ldvt, §ncvt)\n
	 *     On entry, an §n by §ncvt matrix $V_T$.\n
	 *     On exit, §Vt is overwritten by $P^T V_T$.\n
	 *     Not referenced if §ncvt = 0.
	 *
	 * \param[in] ldvt
	 *     The leading dimension of the array §Vt.\n
	 *     $\{ldvt}\ge\max(1,\{n})$ if §ncvt > 0;\n
	 *     $\{ldvt}\ge 1$ if §ncvt = 0.
	 *
	 * \param[in,out] U
	 *     an array, dimension (§ldu, §n)\n
	 *     On entry, an §nru by §n matrix $U$.\n
	 *     On exit, §U is overwritten by $U Q$.\n
	 *     Not referenced if §nru = 0.
	 *
	 * \param[in]     ldu The leading dimension of the array §U. $\{ldu} \ge \max(1,\{nru})$.
	 * \param[in,out] C
	 *     an array, dimension (§ldc, §ncc)\n
	 *     On entry, an §n by §ncc matrix $C$.\n
	 *     On exit, §C is overwritten by $Q^T C$.\n
	 *     Not referenced if §ncc = 0.
	 *
	 * \param[in] ldc
	 *     The leading dimension of the array §C.\n
	 *     $\{ldc}\ge\max(1,\{n})$ if §ncc > 0;\n
	 *     $\{ldc}\ge 1$ if §ncc = 0.
	 *
	 * \param[out] work an array, dimension (4§n)
	 * \param[out] info
	 *     =0: Successful exit.\n
	 *     <0: If §info = -§i, the §i-th argument had an illegal value.\n
	 *     >0:\n
	 *     if §ncvt = §nru = §ncc = 0,
	 *     \li = 1, a split was marked by a positive value in §e.
	 *     \li = 2, current block of Z not diagonalized after 30*§n iterations
	 *              (in inner while loop)
	 *     \li = 3, termination criterion of outer while loop not met
	 *              (program created more than n unreduced blocks)
	 *
	 *     else\n the algorithm did not converge; §d and §e contain the elements of a bidiagonal
	 *            matrix which is orthogonally similar to the input matrix $B$;\n
	 *            if §info = §i, §i elements of §e have not converged to zero.
	 * \remark
	 *     Bug report from Cezary Dendek.\n
	 *     On March 23rd 2017, the integer variable §maxit = $\{MAXITR}\cdot\{n}^2$ is removed
	 *     since it can overflow pretty easily (for §n larger or equal than 18,919). We instead use
	 *     §maxitdivn = $\{MAXITR}\cdot\{n}$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017                                                                            */
	static void dbdsqr(char const* const uplo, int const n, int const ncvt, int const nru,
	                   int const ncc, real* const d, real* const e, real* const Vt, int const ldvt,
	                   real* const U, int const ldu, real* const C, int const ldc,
	                   real* const work, int& info) /*const*/
	{
		const real HNDRTH = real(0.01);
		/* tolmul
		 *
		 * default = max(10, min(100, eps^(-1/8)))\n
		 * §tolmul controls the convergence criterion of the QR loop.\n
		 * If it is positive, §tolmul * eps is the desired relative precision in the computed
		 *     singular values.\n
		 * If it is negative, $|\{tolmul}\ \{eps}\ \{sigma}_\{max}|$ is the desired absolute
		 *     accuracy in the computed singular values (corresponds to relative accuracy
		 *     $|\{tolmul}\ \{eps}|$ in the largest singular value.)\n
		 * $|\{tolmul}|$ should be between 1 and 1/eps, and preferably between 10
		 *     (for fast convergence) and 0.1/eps (for there to be some accuracy in the results).\n
		 * Default is to lose at either one eighth or 2 of the available decimal digits in each
		 * computed singular value (whichever is smaller).                                       */
		real tolmul;
		/* MAXITR
		 *
		 * default = 6\n
		 * §MAXITR controls the maximum number of passes of the algorithm through its inner loop.
		 * The algorithms stops (and so fails to converge) if the number of passes through the
		 * inner loop exceeds $\{MAXITR}\{n}^2$.                                                 */
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
		else if ((ncvt==0 && ldvt<1) || (ncvt>0 && ldvt<std::max(1, n)))
		{
			info = -9;
		}
		else if (ldu<std::max(1, nru))
		{
			info = -11;
		}
		else if ((ncc==0 && ldc<1) || (ncc>0 && ldc<std::max(1, n)))
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
		int nm1 = n - 1;
		if (n!=1)
		{
			// If no singular vectors desired, use qd algorithm
			if (!(ncvt>0 || nru>0 || ncc>0))
			{
				dlasq1(n, d, e, work, info);
				// If info equals 2, dqds didn't finish, try to finish
				if (info!=2)
				{
					return;
				}
				info = 0;
			}
			int nm12 = nm1 + nm1;
			int nm13 = nm12 + nm1;
			int idir = 0;
			// Get machine constants
			real eps = dlamch("Epsilon");
			real unfl = dlamch("Safe minimum");
			// If matrix lower bidiagonal, rotate to be upper bidiagonal by applying Givens
			// rotations on the left
			real cs, sn, r;
			if (lower)
			{
				for (i=0; i<nm1; i++)
				{
					dlartg(d[i], e[i], cs, sn, r);
					d[i] = r;
					e[i]   = sn * d[i+1];
					d[i+1] = cs * d[i+1];
					work[i]     = cs;
					work[nm1+i] = sn;
				}
				// Update singular vectors if desired
				if (nru>0)
				{
					dlasr("R", "V", "F", nru, n, &work[0], &work[nm1], U, ldu);
				}
				if (ncc>0)
				{
					dlasr("L", "V", "F", n, ncc, &work[0], &work[nm1], C, ldc);
				}
			}
			// Compute singular values to relative accuracy tol (By setting tol to be negative,
			// algorithm will compute singular values to absolute accuracy |tol|*||input matrix||
			tolmul = std::max(TEN, std::min(HNDRD, std::pow(eps, MEIGTH)));
			real tol = tolmul * eps;
			// Compute approximate maximum, minimum singular values
			real smax = ZERO;
			for (i=0; i<n; i++)
			{
				smax = std::max(smax, std::fabs(d[i]));
			}
			for (i=0; i<nm1; i++)
			{
				smax = std::max(smax, std::fabs(e[i]));
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
				sminoa /= std::sqrt(real(n));
				thresh = std::max(tol*sminoa, MAXITR*(n*(n*unfl)));
			}
			else
			{
				// Absolute accuracy desired
				thresh = std::max(std::fabs(tol)*smax, MAXITR*(n*(n*unfl)));
			}
			// Prepare for main iteration loop for the singular values (MAXIT is the maximum number
			// of passes through the inner loop permitted before nonconvergence signalled.)
			int maxitdivn = MAXITR * n;
			int iterdivn = 0;
			int iter = -1;
			int oldll = -2;
			int oldm = -2;
			// m points to last element of unconverged part of matrix
			int m = nm1;
			// Begin main iteration loop
			int ll, lll;
			real abse, abss, cosl, cosr, f, g, h, oldcs, oldsn, shift, sigmn, sigmx, sinl, sinr,
			     sll, temp;
			bool breakloop;
			while (m>0)
			{
				// Check for exceeding iteration count
				if (iter>=n)
				{
					iter -= n;
					iterdivn++;
					if (iterdivn>=maxitdivn)
					{
						// Maximum number of iterations exceeded, failure to converge
						info = 0;
						for (i=0; i<nm1; i++)
						{
							if (e[i]!=ZERO)
							{
								info++;
							}
						}
						return;
					}
				}
				// Find diagonal block of matrix to work on
				if (tol<ZERO && std::fabs(d[m])<=thresh)
				{
					d[m] = ZERO;
				}
				smax = std::fabs(d[m]);
				smin = smax;
				breakloop = false;
				for (lll=0; lll<m; lll++)
				{
					ll = m - lll - 1;
					abss = std::fabs(d[ll]);
					abse = std::fabs(e[ll]);
					if (tol<ZERO && abss<=thresh)
					{
						d[ll] = ZERO;
					}
					if (abse<=thresh)
					{
						breakloop = true;
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
				if (breakloop)
				{
					e[ll] = ZERO;
					// Matrix splits since e[ll] = 0
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
				// e[ll] through e[m-1] are nonzero, e[ll-1] is zero
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
						Blas<real>::drot(ncvt, &Vt[m-1], ldvt, &Vt[m], ldvt, cosr, sinr);
					}
					if (nru>0)
					{
						Blas<real>::drot(nru, &U[ldu*(m-1)], 1, &U[ldu*m], 1, cosl, sinl);
					}
					if (ncc>0)
					{
						Blas<real>::drot(ncc, &C[m-1], ldc, &C[m], ldc, cosl, sinl);
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
						idir = 1;
					}
					else
					{
						// Chase bulge from bottom (big end) to top (small end)
						idir = 2;
					}
				}
				// Apply convergence tests
				if (idir==1)
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
						breakloop = false;
						for (lll=ll; lll<m; lll++)
						{
							if (std::fabs(e[lll])<=tol*mu)
							{
								e[lll] = ZERO;
								breakloop = true;
								break;
							}
							mu = std::fabs(d[lll+1]) * (mu/(mu+std::fabs(e[lll])));
							if (mu<sminl)
							{
								sminl = mu;
							}
						}
						if (breakloop)
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
						breakloop = false;
						for (lll=m-1; lll>=ll; lll--)
						{
							if (std::fabs(e[lll])<=tol*mu)
							{
								e[lll] = ZERO;
								breakloop = true;
								break;
							}
							mu = std::fabs(d[lll]) * (mu/(mu+std::fabs(e[lll])));
							if (mu<sminl)
							{
								sminl = mu;
							}
						}
						if (breakloop)
						{
							continue;
						}
					}
				}
				oldll = ll;
				oldm = m;
				// Compute shift. First, test if shifting would ruin relative accuracy,
				// and if so set the shift to zero.
				if (tol>=ZERO && n*tol*(sminl/smax)<=std::max(eps, HNDRTH*tol))
				{
					// Use a zero shift to avoid loss of relative accuracy
					shift = ZERO;
				}
				else
				{
					// Compute the shift from 2-by-2 block at end of matrix
					if (idir==1)
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
					if (idir==1)
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
								e[i-1] = oldsn * r;
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
							dlasr("L", "V", "F", m-ll+1, ncvt, &work[0], &work[nm1], &Vt[ll],
							      ldvt);
						}
						if (nru>0)
						{
							dlasr("R", "V", "F", nru, m-ll+1, &work[nm12], &work[nm13], &U[ldu*ll],
							      ldu);
						}
						if (ncc>0)
						{
							dlasr("L", "V", "F", m-ll+1, ncc, &work[nm12], &work[nm13], &C[ll],
							      ldc);
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
								e[i] = oldsn * r;
							}
							dlartg(oldcs*r, d[i-1]*sn, oldcs, oldsn, d[i]);
							work[i-ll-1] = cs;
							work[i-ll-1+nm1] = -sn;
							work[i-ll-1+nm12] = oldcs;
							work[i-ll-1+nm13] = -oldsn;
						}
						h     = d[ll] * cs;
						d[ll] = h * oldcs;
						e[ll] = h * oldsn;
						// Update singular vectors
						if (ncvt>0)
						{
							dlasr("L", "V", "B", m-ll+1, ncvt, &work[nm12], &work[nm13], &Vt[ll],
							     ldvt);
						}
						if (nru>0)
						{
							dlasr("R", "V", "B", nru, m-ll+1, &work[0], &work[nm1], &U[ldu*ll],
							      ldu);
						}
						if (ncc>0)
						{
							dlasr("L", "V", "B", m-ll+1, ncc, &work[0], &work[nm1], &C[ll], ldc);
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
					if (idir==1)
					{
						// Chase bulge from top to bottom
						// Save cosines and sines for later singular vector updates
						f = (std::fabs(d[ll])-shift) * (std::copysign(ONE, d[ll])+shift/d[ll]);
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
							g      = sinr * d[i+1];
							d[i+1] = cosr * d[i+1];
							dlartg(f, g, cosl, sinl, r);
							d[i] = r;
							f      = cosl*e[i]   + sinl*d[i+1];
							d[i+1] = cosl*d[i+1] - sinl*e[i];
							if (i<m-1)
							{
								g      = sinl * e[i+1];
								e[i+1] = cosl * e[i+1];
							}
							work[i-ll]      = cosr;
							work[i-ll+nm1]  = sinr;
							work[i-ll+nm12] = cosl;
							work[i-ll+nm13] = sinl;
						}
						e[m-1] = f;
						// Update singular vectors
						if (ncvt>0)
						{
							dlasr("L", "V", "F", m-ll+1, ncvt, &work[0], &work[nm1], &Vt[ll],
							      ldvt);
						}
						if (nru>0)
						{
							dlasr("R", "V", "F", nru, m-ll+1, &work[nm12], &work[nm13], &U[ldu*ll],
							      ldu);
						}
						if (ncc>0)
						{
							dlasr("L", "V", "F", m-ll+1, ncc, &work[nm12], &work[nm13], &C[ll],
							      ldc);
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
						f = (std::fabs(d[m])-shift) * (std::copysign(ONE, d[m])+shift/d[m]);
						g = e[m-1];
						for (i=m; i>ll; i--)
						{
							dlartg(f, g, cosr, sinr, r);
							if (i<m)
							{
								e[i] = r;
							}
							f      = cosr*d[i]   + sinr*e[i-1];
							e[i-1] = cosr*e[i-1] - sinr*d[i];
							g      = sinr * d[i-1];
							d[i-1] = cosr * d[i-1];
							dlartg(f, g, cosl, sinl, r);
							d[i] = r;
							f      = cosl*e[i-1] + sinl*d[i-1];
							d[i-1] = cosl*d[i-1] - sinl*e[i-1];
							if (i>ll+1)
							{
								g      = sinl * e[i-2];
								e[i-2] = cosl * e[i-2];
							}
							work[i-ll-1]      =  cosr;
							work[i-ll-1+nm1]  = -sinr;
							work[i-ll-1+nm12] =  cosl;
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
							dlasr("L", "V", "B", m-ll+1, ncvt, &work[nm12], &work[nm13], &Vt[ll],
							      ldvt);
						}
						if (nru>0)
						{
							dlasr("R", "V", "B", nru, m-ll+1, &work[0], &work[nm1], &U[ldu*ll],
							      ldu);
						}
						if (ncc>0)
						{
							dlasr("L", "V", "B", m-ll+1, ncc, &work[0], &work[nm1], &C[ll], ldc);
						}
					}
				}
				// QR iteration finished, go back and check convergence
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
					Blas<real>::dscal(ncvt, NEGONE, &Vt[i], ldvt);
				}
			}
		}
		// Sort the singular values into decreasing order (insertion sort on singular values,
		// but only one transposition per singular vector)
		int isub, j;
		for (i=0; i<nm1; i++)
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
			if (isub!=nm1-i)
			{
				// Swap singular values and vectors
				d[isub] = d[nm1-i];
				d[nm1-i] = smin;
				if (ncvt>0)
				{
					Blas<real>::dswap(ncvt, &Vt[isub], ldvt, &Vt[nm1-i], ldvt);
				}
				if (nru>0)
				{
					Blas<real>::dswap(nru, &U[ldu*isub], 1, &U[ldu*(nm1-i)], 1);
				}
				if (ncc>0)
				{
					Blas<real>::dswap(ncc, &C[isub], ldc, &C[nm1-i], ldc);
				}
			}
		}
	}

	/*! §dbdsvdx
	 *
	 * §dbdsvdx computes the singular value decomposition (SVD) of a real §n by §n (upper or lower)
	 * bidiagonal matrix $B$, $B = U S V^T$, where $S$ is a diagonal matrix with non-negative
	 * diagonal elements (the singular values of $B$), and $U$ and $V^T$ are orthogonal matrices of
	 * left and right singular vectors, respectively.\n
	 * Given an upper bidiagonal $B$ with diagonal $d = [d_0~d_1~\ldots~d_{\{n}-1}]$ and
	 * superdiagonal $e = [e_0~e_1~\ldots~e_{\{n}-2}]$, §DBDSVDX computes the singular value
	 * decompositon of $B$ through the eigenvalues and eigenvectors of the $2\{n}$ by $2\{n}$
	 * tridiagonal matrix\n
	 *     $\{TGK}=\b{bm}  0  & d_0 &     &      &      \\
	 *                    d_0 &  0  & e_0 &      &      \\
	 *                        & e_0 &  0  & d_1  &      \\
	 *                        &     & d_1 &\ddots&\ddots\\
	 *                        &     &     &\ddots&\ddots\e{bm}$\n
	 * If $(s,u,v)$ is a singular triplet of $B$ with $\|u\|=\|v\|=1$, then $(\pm s,q)$, $\|q\|=1$,
	 * are eigenpairs of §TGK, with
	 *     $q=\frac{P\b{bm}u^T&\pm v^T\e{bm}}{\sqrt{2}}$
	 *     $\,=\frac{\b{bm}v_0&u_0&v_1&u_1&\ldots&v_{\{n}-1}&u_{\{n}-1}\e{bm}}{\sqrt{2}}$, and\n
	 *     $P=\b{bm}e_\{n}&e_0&e_{\{n}+1}&e_1&\ldots\e{bm}$.\n
	 * Given a §TGK matrix, one can either
	 *     - a) compute $-s$,$-v$ and change signs so that the singular values (and corresponding
	 *          vectors) are already in descending order (as in §dgesvd or §dgesdd), or
	 *     - b) compute $s$,$v$ and reorder the values (and corresponding vectors).
	 *
	 * §dbdsvdx implements a) by calling §dstevx (bisection plus inverse iteration, to be replaced
	 * with a version of the Multiple Relative Robust Representation algorithm. (See P. Willems and
	 * B. Lang, A framework for the MR^3 algorithm: theory and implementation,
	 * SIAM J. Sci. Comput., 35:740-766, 2013.)
	 * \param[in] uplo
	 *     ='U': $B$ is upper bidiagonal;\n
	 *     ='L': $B$ is lower bidiagonal.
	 *
	 * \param[in] jobz
	 *     ='N': Compute singular values only;\n
	 *     ='V': Compute singular values and singular vectors.
	 *
	 * \param[in] range
	 *     ='A': all singular values will be found.\n
	 *     ='V': all singular values in the half-open interval $[\{vl},\{vu}[$ will be found.\n
	 *     ='I': the §il -th through §iu -th singular values will be found.
	 *
	 * \param[in]      n The order of the bidiagonal matrix. $\{n}\ge 0$.
	 * \param[in, out] d
	 *     an array, dimension (§n)\n The §n diagonal elements of the bidiagonal matrix $B$.
	 *
	 * \param[in, out] e
	 *     an array, dimension ($\max(1,\{n}-1)$)\n
	 *     The $\{n}-1$ superdiagonal elements of the bidiagonal matrix $B$ in elements 0 to
	 *     $\{n}-2$.
	 *
	 * \param[in] vl
	 *     If §range ='V', the lower bound of the interval to be searched for singular values.
	 *     $\{vu}>\{vl}$.\n Not referenced if §range ='A' or 'I'.
	 *
	 * \param[in] vu
	 *     If §range ='V', the upper bound of the interval to be searched for singular values.
	 *     $\{vu}>\{vl}$.\n Not referenced if §range ='A' or 'I'.
	 *
	 * \param[in] il
	 *     If §range ='I', the index of the smallest singular value to be returned.
	 *     $0\le\{il}\le\{iu}<\min(\{M},\{n})$, if $\min(\{M},\{n})>0$.\n
	 *     Not referenced if §range ='A' or 'V'.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in] iu
	 *     If §range ='I', the index of the largest singular value to be returned.
	 *     $0\le\{il}\le\{iu}<\min(\{M},\{n})$, if $\min(\{M},\{n})>0$.\n
	 *     Not referenced if §range ='A' or 'V'.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[out] ns
	 *     The total number of singular values found. $0\le\{ns}\le\{n}$.
	 *     If §range ='A', $\{ns}=\{n}$, and if §range ='I', $\{ns}=\{iu}-\{il}+1$.
	 *
	 * \param[out] s
	 *     an array, dimension (§n)\n
	 *     The first §ns elements contain the selected singular values in ascending order.
	 *
	 * \param[out] Z
	 *     an array, dimension (§LDZ,§K)\n
	 *     - If §jobz ='V', then if $\{info}=0$ the first §ns columns of §Z contain the singular
	 *       vectors of the matrix $B$ corresponding to the selected singular values, with $U$ in
	 *       rows 0 to $\{n}-1$ and $V$ in rows §n to $2\{n}-1$, i.e.\n
	 *       $\{Z}=\b{bm} U \\ V \e{bm}$
	 *     - If §jobz ='N', then §Z is not referenced.
	 *     .
	 *     Note: The user must ensure that at least $K=\{ns}+1$ columns are supplied in the array
	 *           §Z; if §range ='V', the exact value of §ns is not known in advance and an upper
	 *           bound must be used.
	 *
	 * \param[in] ldz
	 *     The leading dimension of the array §Z.
	 *     $\{ldz}\ge 1$, and if §jobz ='V', $\{ldz}\ge\max(2,2\{n})$.
	 *
	 * \param[out] work  an array, dimension ($14\{n}$)
	 * \param[out] iwork
	 *     an integer array, dimension ($12\{n}$)\n
	 *     If §jobz ='V', then if $\{info}=0$, the first §ns elements of §iwork are zero.
	 *     If $\{info}>0$, then §iwork contains the indices of the eigenvectors that failed to
	 *     converge in §dstevx.
	 *
	 * \param[out] info
	 *     =0: successful exit.\n
	 *     <0: if $\{info}=-i$, the $i$-th argument had an illegal value\n
	 *     >0: if $\{info}=i$, then $i$ eigenvectors failed to converge in §dstevx. The indices of
	 *         the eigenvectors (as returned by §dstevx) are stored in the array §iwork.
	 *         if $\{info}=2\{n}+1$, an internal error occurred.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016                                                                           */
	static void dbdsvdx(char const* const uplo, char const* const jobz, char const* const range,
	                    int const n, real* const d, real* const e, real const vl, real const vu,
	                    int const il, int const iu, int& ns, real* const s, real* const Z,
	                    int const ldz, real* const work, int* const iwork, int& info) /*const*/
	{
		real const FUDGE = real(2.0);
		// Test the input parameters.
		bool allsv = (std::toupper(range[0])=='A');
		bool valsv = (std::toupper(range[0])=='V');
		bool indsv = (std::toupper(range[0])=='I');
		bool wantz = (std::toupper(jobz[0])=='V');
		bool lower = (std::toupper(uplo[0])=='L');
		info = 0;
		if (std::toupper(uplo[0])!='U' && !lower)
		{
			info = -1;
		}
		else if (std::toupper(jobz[0])!='N' && !wantz)
		{
			info = -2;
		}
		else if (!(allsv || valsv || indsv))
		{
			info = -3;
		}
		else if (n<0)
		{
			info = -4;
		}
		else if (n>0)
		{
			if (valsv)
			{
				if (vl<ZERO)
				{
					info = -7;
				}
				else if (vu<=vl)
				{
					info = -8;
				}
			}
			else if (indsv)
			{
				if (il<0 || il>std::max(0, n-1))
				{
					info = -9;
				}
				else if (iu<std::min(n-1, il) || iu>=n)
				{
					info = -10;
				}
			}
		}
		if (info==0)
		{
			if (ldz<1 || (wantz && ldz<n*2))
			{
				info = -14;
			}
		}
		if (info!=0)
		{
			xerbla("DBDSVDX", -info);
			return;
		}
		// Quick return if possible (n<=1)
		ns = 0;
		if (n==0)
		{
			return;
		}
		if (n==1)
		{
			if (allsv || indsv)
			{
				ns = 1;
				s[0] = std::fabs(d[0]);
			}
			else
			{
				if (vl<std::fabs(d[0]) && vu>=std::fabs(d[0]))
				{
					ns = 1;
					s[0] = std::fabs(d[0]);
				}
			}
			if (wantz)
			{
				Z[0] = std::copysign(ONE, d[0]);
				Z[1] = ONE;
			}
			return;
		}
		real abstol = 2 * dlamch("Safe Minimum");
		real ulp = dlamch("Precision");
		real eps = dlamch("Epsilon");
		real sqrt2 = std::sqrt(TWO);
		real ortol = std::sqrt(ulp);
		/* Criterion for splitting is taken from dbdsqr when singular values are computed to
		 * relative accuracy tol. (See J. Demmel and W. Kahan, Accurate singular values of
		 * bidiagonal matrices, SIAM J. Sci. and Stat. Comput., 11:873-912, 1990.)               */
		real tol = std::max(TEN, std::min(HNDRD, std::pow(eps, MEIGTH))) * eps;
		// Compute approximate maximum, minimum singular values.
		int i = Blas<real>::idamax(n, d, 1);
		real smax = std::fabs(d[i]);
		i = Blas<real>::idamax(n-1, e, 1);
		smax = std::max(smax, std::fabs(e[i]));
		// Compute threshold for neglecting d's and e's.
		real smin = std::fabs(d[0]);
		if (smin!=ZERO)
		{
			real mu = smin;
			for (i=1; i<n; i++)
			{
				mu = std::fabs(d[i]) * (mu/(mu+std::fabs(e[i-1])));
				smin = std::min(smin, mu);
				if (smin==ZERO)
				{
					break;
				}
			}
		}
		smin /= std::sqrt(real(n));
		real thresh = tol * smin;
		// Check for zeros in d and e (splits), i.e. submatrices.
		for (i=0; i<n-1; i++)
		{
			if (std::fabs(d[i])<=thresh)
			{
				d[i] = ZERO;
			}
			if (std::fabs(e[i])<=thresh)
			{
				e[i] = ZERO;
			}
		}
		if (std::fabs(d[n-1])<=thresh)
		{
			d[n-1] = ZERO;
		}
		// Pointers for arrays used by DSTEVX.
		int idtgk  = 0;
		int ietgk  = idtgk + n*2;
		int itemp  = ietgk + n*2;
		int iifail = 0;
		int iiwork = iifail + n*2;
		// Set rngvx, which corresponds to range for DSTEVX in TGK mode. vl,vu or il,iu are
		// redefined to conform to implementation a) described in the leading comments.
		int iltgk  = -1;
		int iutgk  = -1;
		real vltgk = ZERO;
		real vutgk = ZERO;
		char rngvx[2];
		rngvx[1] = '\0';
		if (allsv)
		{
			/* All singular values will be found. We aim at -s (see leading comments) with
			 * NGVX = 'I'. il and iu are set later (as iltgk and iutgk) according to the dimension
			 * of the active submatrix.                                                          */
			rngvx[0] = 'I';
			if (wantz)
			{
				dlaset("F", n*2, n+1, ZERO, ZERO, Z, ldz);
			}
		}
		else if (valsv)
		{
			/* Find singular values in a half-open interval. We aim at -s (see leading comments)
			 * and we swap vl and vu (as vutgk and vltgk), changing their signs.                 */
			rngvx[0] = 'V';
			vltgk = -vu;
			vutgk = -vl;
			for (i=idtgk; i<idtgk+2*n; i++)
			{
				work[i] = ZERO;
			}
			Blas<real>::dcopy(n,   d, 1, &work[ietgk],   2);
			Blas<real>::dcopy(n-1, e, 1, &work[ietgk+1], 2);
			dstevx("N", "V", n*2, &work[idtgk], &work[ietgk], vltgk, vutgk, iltgk, iltgk, abstol,
			       ns, s, Z, ldz, &work[itemp], &iwork[iiwork], &iwork[iifail], info);
			if (ns==0)
			{
				return;
			}
			else
			{
				if (wantz)
				{
					dlaset("F", n*2, ns, ZERO, ZERO, Z, ldz);
				}
			}
		}
		else if (indsv)
		{
			/* Find the il-th through the iu-th singular values. We aim at -s (see leading
			 * comments) and indices are mapped into values, therefore mimicking DSTEBZ, where
			 *     GL = GL - FUDGE*TNORM*ulp*n - FUDGE*TWO*PIVMIN
			 *     GU = GU + FUDGE*TNORM*ulp*n + FUDGE*PIVMIN                                    */
			iltgk = il;
			iutgk = iu;
			rngvx[0] = 'V';
			for (i=idtgk; i<idtgk+2*n; i++)
			{
				work[i] = ZERO;
			}
			Blas<real>::dcopy(n,   d, 1, &work[ietgk],   2);
			Blas<real>::dcopy(n-1, e, 1, &work[ietgk+1], 2);
			dstevx("N", "I", n*2, &work[idtgk], &work[ietgk], vltgk, vltgk, iltgk, iltgk, abstol,
			       ns, s, Z, ldz, &work[itemp], &iwork[iiwork], &iwork[iifail], info);
			vltgk = s[0] - FUDGE*smax*ulp*n;
			for (i=idtgk; i<idtgk+2*n; i++)
			{
				work[i] = ZERO;
			}
			Blas<real>::dcopy(n,   d, 1, &work[ietgk],   2);
			Blas<real>::dcopy(n-1, e, 1, &work[ietgk+1], 2);
			dstevx("N", "I", n*2, &work[idtgk], &work[ietgk], vutgk, vutgk, iutgk, iutgk, abstol,
			       ns, s, Z, ldz, &work[itemp], &iwork[iiwork], &iwork[iifail], info);
			vutgk = s[0] + FUDGE*smax*ulp*n;
			vutgk = std::min(vutgk, ZERO);
			// If vltgk=vutgk, DSTEVX returns an error message,
			// so if needed we change vutgk slightly.
			if (vltgk==vutgk)
			{
				vltgk = vltgk - tol;
			}
			if (wantz)
			{
				dlaset("F", n*2, iu-il+1, ZERO, ZERO, Z, ldz);
			}
		}
		/* Initialize variables and pointers for s, Z, and work.
		 * nru, nrv: number of rows in U and V for the active submatrix
		 * idbeg, isbeg: offsets for the entries of d and s
		 * irowz, icolz: offsets for the rows and columns of Z
		 * irowu, irowv: offsets for the rows of U and V                                         */
		ns = 0;
		int nru = 0;
		int nrv = 0;
		int idbeg = 0;
		int isbeg = 0;
		int irowz = 0;
		int icolz = 0;
		int irowu = 1;
		int irowv = 0;
		bool split = false;
		bool sveq0 = false;
		// Form the tridiagonal TGK matrix.
		for (i=0; i<n; i++)
		{
			s[i] = ZERO;
		}
		work[ietgk+2*n-1] = ZERO;
		for (i=idtgk; i<idtgk+2*n; i++)
		{
			work[i] = ZERO;
		}
		Blas<real>::dcopy(n,   d, 1, &work[ietgk],   2);
		Blas<real>::dcopy(n-1, e, 1, &work[ietgk+1], 2);
		// Check for splits in two levels, outer level in e and inner level in d.
		int idend, idptr, ieptr, isplt, j, ntgk=0, nsl, zind1, zind2;
		real emin, nrmu, nrmv, zjtji;
		for (ieptr=1; ieptr<n*2; ieptr+=2)
		{
			if (work[ietgk+ieptr]==ZERO)
			{
				// Split in e (this piece of B is square) or bottom of the (input bidiagonal)
				// matrix.
				isplt = idbeg;
				idend = ieptr - 1;
				for (idptr=idbeg; idptr<=idend; idptr+=2)
				{
					if (work[ietgk+idptr]==ZERO)
					{
						// Split in d (rectangular submatrix). Set the number of rows in U and V
						// (nru and nrv) accordingly.
						if (idptr==idbeg)
						{
							// d=0 at the top.
							sveq0 = true;
							if (idbeg==idend)
							{
								nru = 1;
								nrv = 1;
							}
						}
						else if (idptr==idend)
						{
							// d=0 at the bottom.
							sveq0 = true;
							nru = (idend-isplt)/2 + 1;
							nrv = nru;
							if (isplt!=idbeg)
							{
								nru++;
							}
						}
						else
						{
							if (isplt==idbeg)
							{
								// Split: top rectangular submatrix.
								nru = (idptr-idbeg)/2;
								nrv = nru + 1;
							}
							else
							{
								// Split: middle square submatrix.
								nru = (idptr-isplt)/2 + 1;
								nrv = nru;
							}
						}
					}
					else if (idptr==idend)
					{
						// Last entry of d in the active submatrix.
						if (isplt==idbeg)
						{
							// No split (trivial case).
							nru = (idend-idbeg)/2 + 1;
							nrv = nru;
						}
						else
						{
							// Split: bottom rectangular submatrix.
							nrv = (idend-isplt)/2 + 1;
							nru = nrv + 1;
						}
					}
					ntgk = nru + nrv;
					if (ntgk>0)
					{
						/* Compute eigenvalues/vectors of the active submatrix according to range:
						 * if range='A' (allsv) then rngvx = 'I'
						 * if range='V' (valsv) then rngvx = 'V'
						 * if range='I' (indsv) then rngvx = 'V'                                 */
						iltgk = 0;
						iutgk = ntgk/2 - 1;
						if (allsv || vutgk==ZERO)
						{
							if (sveq0 || smin<eps || (ntgk%2)>0)
							{
								// Special case: eigenvalue equal to zero or very small,
								// additional eigenvector is needed.
								iutgk++;
							}
						}
						// Workspace needed by DSTEVX:
						//  work[itemp:]: 2*5*ntgk
						// iwork[0:]    : 2*6*ntgk
						dstevx(jobz, rngvx, ntgk, &work[idtgk+isplt], &work[ietgk+isplt], vltgk,
						       vutgk, iltgk, iutgk, abstol, nsl, &s[isbeg], &Z[irowz+ldz*icolz],
						       ldz, &work[itemp], &iwork[iiwork], &iwork[iifail], info);
						if (info!=0)
						{
							// Exit with the error code from dstevx.
							return;
						}
						if (nsl==0)
						{
							emin = dlamch("O");
						}
						else
						{
							emin = std::fabs(*std::max_element(&s[isbeg],&s[isbeg+nsl]));
						}
						if (nsl>0 && wantz)
						{
							/* Normalize u=Z[[1,3,...],:] and v=Z[[0,2,...],:], changing the sign
							 * of v as discussed in the leading comments. The norms of u and v may
							 * be (slightly) different from 1/sqrt(2) if the corresponding
							 * eigenvalues are very small or too close. We check those norms and,
							 * if needed, reorthogonalize the vectors.                           */
							if (nsl>1 && vutgk==ZERO && (ntgk%2)==0 && emin==ZERO && !split)
							{
								/* d=0 at the top or bottom of the active submatrix: one eigenvalue
								 * is equal to zero; concatenate the eigenvectors corresponding to
								 * the two smallest eigenvalues.                                 */
								zind1 = ldz * (icolz+nsl-2);
								zind2 = zind1 + ldz;
								for (i=irowz; i<irowz+ntgk; i++)
								{
									Z[i+zind1] = Z[i+zind1] + Z[i+zind2];
									Z[i+zind2] = ZERO;
								}
								//if (iutgk*2>=ntgk-1)
								//{
								//    // Eigenvalue equal to zero or very small.
								//    nsl--;
								//}
							}
							zind1 = irowu + ldz*icolz;
							for (i=0; i<std::min(nsl, nru); i++)
							{
								zind2 = zind1 + ldz*i;
								nrmu = Blas<real>::dnrm2(nru, &Z[zind2], 2);
								if (nrmu==ZERO)
								{
									info = n*2 + 1;
									return;
								}
								Blas<real>::dscal(nru, ONE/nrmu, &Z[zind2], 2);
								if (nrmu!=ONE && std::fabs(nrmu-ortol)*sqrt2>ONE)
								{
									for (j=0; j<i; j++)
									{
										zjtji = -Blas<real>::ddot(nru, &Z[zind1+ldz*j], 2,
										                          &Z[zind2], 2);
										Blas<real>::daxpy(nru, zjtji, &Z[zind1+ldz*j], 2,
										                  &Z[zind2], 2);
									}
									nrmu = Blas<real>::dnrm2(nru, &Z[zind2], 2);
									Blas<real>::dscal(nru, ONE/nrmu, &Z[zind2], 2);
								}
							}
							zind1 = irowv + ldz*icolz;
							for (i=0; i<std::min(nsl, nrv); i++)
							{
								zind2 = zind1 + ldz*i;
								nrmv = Blas<real>::dnrm2(nrv, &Z[zind2], 2);
								if (nrmv==ZERO)
								{
									info = n*2 + 1;
									return;
								}
								Blas<real>::dscal(nrv, -ONE/nrmv, &Z[zind2], 2);
								if (nrmv!=ONE && std::fabs(nrmv-ortol)*sqrt2>ONE)
								{
									for (j=0; j<i; j++)
									{
										zjtji = -Blas<real>::ddot(nrv, &Z[zind1+ldz*j], 2,
										                          &Z[zind2], 2);
										Blas<real>::daxpy(nru, zjtji, &Z[zind1+ldz*j], 2,
										                  &Z[zind2], 2);
									}
									nrmv = Blas<real>::dnrm2(nrv, &Z[zind2], 2);
									Blas<real>::dscal(nrv, ONE/nrmv, &Z[zind2], 2);
								}
							}
							if (vutgk==ZERO && idptr<idend && (ntgk%2)>0)
							{
								/* d=0 in the middle of the active submatrix (one eigenvalue is
								 * equal to zero): save the corresponding eigenvector for later use
								 * (when bottom of the active submatrix is reached).             */
								split = true;
								zind1 = ldz * n;
								zind2 = ldz * (ns+nsl-1);
								for (i=irowz; i<irowz+ntgk; i++)
								{
									Z[i+zind1] = Z[i+zind2];
									Z[i+zind2] = ZERO;
								}
							}
						}// if wantz
						nsl = std::min(nsl, nru);
						sveq0 = false;
						// Absolute values of the eigenvalues of TGK.
						for (i=0; i<nsl; i++)
						{
							s[isbeg+i] = std::fabs(s[isbeg+i]);
						}
						// Update pointers for TGK, s and Z.
						isbeg += nsl;
						irowz += ntgk;
						icolz += nsl;
						irowu = irowz;
						irowv = irowz + 1;
						isplt = idptr + 1;
						ns += nsl;
						nru = 0;
						nrv = 0;
					}// ntgk>0
					if (irowz<n*2-1 && wantz)
					{
						zind1 = ldz * icolz;
						for (i=0; i<irowz; i++)
						{
							Z[i+zind1] = ZERO;
						}
					}
				}// idptr loop
				if (split && wantz)
				{
					// Bring back eigenvector corresponding to eigenvalue equal to zero.
					zind1 = ldz * (isbeg-1);
					zind2 = ldz * n;
					for (i=idbeg; i<=idend-ntgk+1; i++)
					{
						Z[i+zind1] = Z[i+zind1] + Z[i+zind2];
						Z[i+zind2] = 0;
					}
				}
				irowv--;
				irowu++;
				idbeg = ieptr + 1;
				sveq0 = false;
				split = false;
			}// Check for split in e
		}// ieptr loop
		// Sort the singular values into decreasing order (insertion sort on singular values, but
		// only one transposition per singular vector)
		int k;
		zind1 = ldz * (ns-1);
		for (i=0; i<ns-1; i++)
		{
			k = 0;
			smin = s[0];
			for (j=1; j<ns-i; j++)
			{
				if (s[j]<=smin)
				{
					k = j;
					smin = s[j];
				}
			}
			if (k!=ns-1-i)
			{
				s[k] = s[ns-1-i];
				s[ns-1-i] = smin;
				if (wantz)
				{
					Blas<real>::dswap(n*2, &Z[ldz*k], 1, &Z[zind1-ldz*i], 1);
				}
			}
		}
		// If range='I', check for singular values/vectors to be discarded.
		if (indsv)
		{
			k = iu - il + 1;
			if (k<ns-1)
			{
				for (i=k; i<ns; i++)
				{
					s[i] = ZERO;
				}
				if (wantz)
				{
					dlaset("All", 2*n, ns-k, ZERO, ZERO, &Z[ldz*k], ldz);
				}
				ns = k;
			}
		}
		// Reorder Z: U=Z[0:n-1,0:ns-1], V=Z[n:n*2-1,0:ns-1].
		// If B is a lower diagonal, swap U and V.
		if (wantz)
		{
			for (i=0; i<ns; i++)
			{
				zind1 = ldz * i;
				Blas<real>::dcopy(n*2, &Z[zind1], 1, work, 1);
				if (lower)
				{
					Blas<real>::dcopy(n, &work[1], 2, &Z[n+zind1], 1);
					Blas<real>::dcopy(n, &work[0], 2, &Z[zind1], 1);
				}
				else
				{
					Blas<real>::dcopy(n, &work[1], 2, &Z[zind1], 1);
					Blas<real>::dcopy(n, &work[0], 2, &Z[n+zind1], 1);
				}
			}
		}
	}

	/*! §dgebak
	 *
	 * §dgebak forms the right or left eigenvectors of a real general matrix by backward
	 * transformation on the computed eigenvectors of the balanced matrix output by §dgebal.
	 * \param[in] job
	 *     Specifies the type of backward transformation required:\n
	 *      = 'N', do nothing, return immediately;\n
	 *      = 'P', do backward transformation for permutation only;\n
	 *      = 'S', do backward transformation for scaling only;\n
	 *      = 'B', do backward transformations for both permutation and scaling.\n
	 *      §job must be the same as the argument §job supplied to §dgebal.
	 *
	 * \param[in] side
	 *     = 'R': §V contains right eigenvectors;\n
	 *     = 'L': §V contains left eigenvectors.
	 *
	 * \param[in] n   The number of rows of the matrix §V. $\{n}\ge 0$.
	 * \param[in] ilo, ihi
	 *     The integers §ilo and §ihi determined by §dgebal.\n
	 *     $0\le\{ilo}<=\{ihi}<\{n}$, if $\{n}>0$; $\{ilo}=0$ and $\{ihi}=-1$, if $\{n}=0$.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in] scale
	 *     an array, dimension (§n)\n
	 *     Details of the permutation and scaling factors, as returned by §dgebal.
	 *
	 * \param[in]     m The number of columns of the matrix §V. $\{m}\ge 0$.
	 * \param[in,out] V
	 *     an array, dimension (§ldv,§m)\n
	 *     On entry, the matrix of right or left eigenvectors to be transformed, as returned by
	 *     §dhsein or §dtrevc.\n
	 *     On exit, §V is overwritten by the transformed eigenvectors.
	 *
	 * \param[in]  ldv  The leading dimension of the array §V. $\{ldv}\ge\max(1,\{n})$.
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dgebak(char const* const job, char const* const side, int const n, int const ilo,
	                   int const ihi, real const* const scale, int const m, real* const V,
	                   int const ldv, int& info) /*const*/
	{
		// Decode and Test the input parameters
		bool rightv = (std::toupper(side[0])=='R');
		bool leftv  = (std::toupper(side[0])=='L');
		info = 0;
		char upjob = std::toupper(job[0]);
		if (upjob!='N' && upjob!='P' && upjob!='S' && upjob!='B')
		{
			info = -1;
		}
		else if (!rightv && !leftv)
		{
			info = -2;
		}
		else if (n<0)
		{
			info = -3;
		}
		else if (ilo<0 || ilo>=std::max(1, n))
		{
			info = -4;
		}
		else if (ihi<std::min(ilo, n-1) || ihi>=n)
		{
			info = -5;
		}
		else if (m<0)
		{
			info = -7;
		}
		else if (ldv<std::max(1, n))
		{
			info = -9;
		}
		if (info!=0)
		{
			xerbla("DGEBAK", -info);
			return;
		}
		// Quick return if possible
		if (n==0)
		{
			return;
		}
		if (m==0)
		{
			return;
		}
		if (upjob=='N')
		{
			return;
		}
		int i;
		real s;
		if (ilo!=ihi)
		{
			// Backward balance
			if (upjob=='S' || upjob=='B')
			{
				if (rightv)
				{
					for (i=ilo; i<=ihi; i++)
					{
						s = scale[i];
						Blas<real>::dscal(m, s, &V[i], ldv);
					}
				}
				if (leftv)
				{
					for (i=ilo; i<=ihi; i++)
					{
						s = ONE / scale[i];
						Blas<real>::dscal(m, s, &V[i], ldv);
					}
				}
			}
		}
		// Backward permutation
		// for i=ilo-1; i>=0; i--
		//       ihi+1; i<n;  i++
		int ii, k;
		if (upjob=='P' || upjob=='B')
		{
			if (rightv)
			{
				for (ii=0; ii<n; ii++)
				{
					i = ii;
					if (i>=ilo && i<=ihi)
					{
						continue;
					}
					if (i<ilo)
					{
						i = ilo - ii - 1;
					}
					k = scale[i] - 1;
					if (k==i)
					{
						continue;
					}
					Blas<real>::dswap(m, &V[i], ldv, &V[k], ldv);
				}
			}
			if (leftv)
			{
				for (ii=0; ii<n; ii++)
				{
					i = ii;
					if (i>=ilo && i<=ihi)
					{
						continue;
					}
					if (i<ilo)
					{
						i = ilo - ii - 1;
					}
					k = scale[i] - 1;
					if (k==i)
					{
						continue;
					}
					Blas<real>::dswap(m, &V[i], ldv, &V[k], ldv);
				}
			}
		}
	}

	/*! §dgebal
	 *
	 * §dgebal balances a general real matrix $A$. This involves, first, permuting $A$ by a
	 * similarity transformation to isolate eigenvalues in the first 0 to §ilo-1 and last §ihi+1 to
	 * §n elements on the diagonal; and second, applying a diagonal similarity transformation to
	 * rows and columns §ilo to §ihi to make the rows and columns as close in norm as possible.
	 * Both steps are optional. Balancing may reduce the 1-norm of the matrix, and improve the
	 * accuracy of the computed eigenvalues and/or eigenvectors.
	 * \param[in] job
	 *     Specifies the operations to be performed on A:\n
	 *     'N': none: simply set §ilo = 0, §ihi = n-1, §scale[§i] = 1.0 for §i = 0,...,§n-1;\n
	 *     'P': permute only;\n
	 *     'S': scale only;\n
	 *     'B': both permute and scale.
	 *
	 * \param[in]     n The order of the matrix $A$. $\{n} \ge 0$.
	 * \param[in,out] A
	 *     an array of dimension (§lda,§n)\n
	 *     On entry, the input matrix $A$.\n
	 *     On exit, §A is overwritten by the balanced matrix.\n
	 *     If §job = 'N', §A is not referenced. See remark.
	 *
	 * \param[in]  lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{n})$.
	 * \param[out] ilo
	 * \param[out] ihi §ilo and §ihi are set to integers such that on exit §A[§i,§j] = 0 if §i > §j
	 *             and §j = 0,...,§ilo-1 or §i = §ihi+1,...,§n-1. If §job = 'N' or 'S', §ilo = 0
	 *             and §ihi = §n-1.\n
	 *             NOTE: These are zero-based indices!
	 *
	 * \param[out] scale
	 *     an array of dimension (§n)
	 *     Details of the permutations and scaling factors applied to §A. If §P[§j] is the
	 *     one-based(!) index of the row and column interchanged with row and column §j and §D[§j]
	 *     is the scaling factor applied to row and column §j, then\n
	 *                   §scale[§j] = §P[§j]    for §j = 0,...,§ilo-1 \n
	 *     &emsp;&emsp;&emsp;&emsp; = §D[§j]    for §j = §ilo,...,§ihi \n
	 *     &emsp;&emsp;&emsp;&emsp; = §P[§j]    for §j = §ihi+1,...,§n-1.\n
	 *     The order in which the interchanges are made is §n-1 to §ihi+1, then 0 to §ilo-1.
	 *
	 * \param[out] info
	 *     = 0: Successful exit.
	 *     < 0:  if §info = -§i, the §i-th argument had an illegal value.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017
	 * \remark
	 *     The permutations consist of row and column interchanges which put the matrix in the
	 *     form\n
	 *     $P A P =
	 *      \b{bm} T_1 & X &  Y  \\
	 *              0  & B &  Z  \\
	 *              0  & 0 & T_2 \e{bm}$\n
	 *     where $T_1$ and $T_2$ are upper triangular matrices whose eigenvalues lie along the
	 *     diagonal. The column indices §ilo and §ihi mark the starting and ending columns of the
	 *     submatrix $B$. Balancing consists of applying a diagonal similarity transformation
	 *     $D^{-1} B D$ to make the 1-norms of each row of $B$ and its corresponding column nearly
	 *     equal. The output matrix is\n
	 *     $\b{bm} T_1  &    XD    &    Y    \\
	 *              0   & D^{-1}BD & D^{-1}Z \\
	 *              0   &    0     &   T_2   \e{bm}$.\n
	 *     Information about the permutations $P$ and the diagonal matrix $D$ is returned in the
	 *     vector scale.\n
	 *     This subroutine is based on the EISPACK routine §BALANC.\n
	 *     Modified by Tzu-Yi Chen, Computer Science Division, University of California at
	 *     Berkeley, USA                                                                         */
	static void dgebal(char const* const job, int const n, real* const A, int const lda, int& ilo,
	                   int& ihi, real* const scale, int& info) /*const*/
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

	/*! §dgebd2 reduces a general matrix to bidiagonal form using an unblocked algorithm.
	 *
	 * §dgebd2 reduces a real general §m by §n matrix $A$ to upper or lower bidiagonal form $B$ by
	 * an orthogonal transformation: $Q^T A P = B$.\n
	 * If $\{m}\ge\{n}$, $B$ is upper bidiagonal; if $\{m}<\{n}$, $B$ is lower bidiagonal.
	 * \param[in]     m The number of rows in the matrix $A$. $\{m} \ge 0$.
	 * \param[in]     n The number of columns in the matrix $A$. $\{n} \ge 0$.
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the §m by §n general matrix to be reduced.\n
	 *     On exit,
	 *     \li if $\{m}\ge\{n}$, the diagonal and the first superdiagonal are overwritten with the
	 *         upper bidiagonal matrix $B$; the elements below the diagonal, with the array §tauq,
	 *         represent the orthogonal matrix $Q$ as a product of elementary reflectors, and the
	 *         elements above the first superdiagonal, with the array §taup, represent the
	 *         orthogonal matrix $P$ as a product of elementary reflectors;
	 *     \li if $\{m}<\{n}$, the diagonal and the first subdiagonal are overwritten with the
	 *         lower bidiagonal matrix $B$; the elements below the first subdiagonal, with the
	 *         array §tauq, represent the orthogonal matrix $Q$ as a product of elementary
	 *         reflectors, and the elements above the diagonal, with the array §taup, represent the
	 *         orthogonal matrix $P$ as a product of elementary reflectors.
	 *
	 *     See Further Details.
	 *
	 * \param[in]  lda The leading dimension of the array §A. $\{lda} \ge \max(1,\{m})$.
	 * \param[out] d
	 *     an array, dimension ($\min(\{m},\{n})$)\n
	 *     The diagonal elements of the bidiagonal matrix $B$: §d[§i] = §A[§i,§i].
	 *
	 * \param[out] e
	 *     an array, dimension ($\min(\{m},\{n})-1$)\n
	 *     The off-diagonal elements of the bidiagonal matrix $B$:\n
	 *     if $\{m}\ge\{n}$, §e[§i] = §A[§i,§i+1] for §i = 0,1,...,§n-2;\n
	 *     if $\{m} < \{n}$, §e[§i] = §A[§i+1,§i] for §i = 0,1,...,§m-2.
	 *
	 * \param[out] tauq
	 *     an array, dimension ($\min(\{m},\{n})$)\n
	 *     The scalar factors of the elementary reflectors which represent the orthogonal matrix
	 *     $Q$. See Further Details.
	 *
	 * \param[out] taup
	 *     an array, dimension ($\min(\{m},\{n})$)\n
	 *     The scalar factors of the elementary reflectors which represent the orthogonal matrix
	 *     $P$. See Further Details.
	 *
	 * \param[out] work an array, dimension ($\max(\{m},\{n})$)
	 * \param[out] info
	 *     =0: successful exit.\n
	 *     <0: if §info = -§i, the §i-th argument had an illegal value.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017
	 * \remark
	 *     The matrices $Q$ and $P$ are represented as products of elementary reflectors:
	 *     \li If $\{m} \ge \{n}$,\n
	 *         $Q = H(0) H(1) \ldots H(\{n}-1)$ and $P = G(0) G(1) \ldots G(\{n}-2)$\n
	 *         Each $H(i)$ and $G(i)$ has the form:\n
	 *         $H(i) = I - \{tauq}[i] v v^T$  and $G(i) = I - \{taup}[i] u u^T$\n
	 *         where v and u are real vectors;\n
	 *         $v[0:i-1]=0$,       $v[i]=1$,&emsp;&nbsp; and $v[i+1:\{m}-1]$ is stored on exit
	 *         in §A$[i+1:\{m}-1,i]$;\n
	 *         $u[0:i]=0$,&emsp;&nbsp; $u[i+1]=1$,       and $u[i+2:\{n}-1]$ is stored on exit
	 *         in §A$[i,i+2:\{n}-1]$.
	 *     \li If $\{m} < \{n}$,
	 *         $Q = H(0) H(1) \ldots H(\{m}-2)$ and $P = G(0) G(1) \ldots G(\{m}-1)$.\n
	 *         Each $H(i)$ and $G(i)$ has the form:
	 *         $H(i) = I - \{tauq}[i] v v^T$ and $G(i) = I - \{taup}[i] u u^T$\n
	 *         where v and u are real vectors;\n
	 *         $v[0:i]=0$, $v[i+1]=1$, and $v[i+2:\{m}-1]$ is stored on exit in §A[§i+2:§m-1,§i];\n
	 *         $u[0:i-1]=0$, $u[i]=1$, and $u[i+1:\{n}-1]$ is stored on exit in §A[§i,§i+1:§n-1].
	 *
	 *     The contents of §A on exit are illustrated by the following examples:
	 *     \li §m = 6 and §n = 5 (§m > §n):
	 *         $\b{bm}  d  &  e  & u_1 & u_1 & u_1 \\
	 *                 v_1 &  d  &  e  & u_2 & u_2 \\
	 *                 v_1 & v_2 &  d  &  e  & u_3 \\
	 *                 v_1 & v_2 & v_3 &  d  &  e  \\
	 *                 v_1 & v_2 & v_3 & v_4 &  d  \\
	 *                 v_1 & v_2 & v_3 & v_4 & v_5 \e{bm}$
	 *     \li §m = 5 and §n = 6 (§m < §n):
	 *         $\b{bm}  d  & u_1 & u_1 & u_1 & u_1 & u_1 \\
	 *                  e  &  d  & u_2 & u_2 & u_2 & u_2 \\
	 *                 v_1 &  e  &  d  & u_3 & u_3 & u_3 \\
	 *                 v_1 & v_2 &  e  &  d  & u_4 & u_4 \\
	 *                 v_1 & v_2 & v_3 &  e  &  d  & u_5 \e{bm}$
	 *
	 *     where $d$ and $e$ denote diagonal and off-diagonal elements of $B$, $v_i$ denotes an
	 *     element of the vector defining $H(i)$, and $u_i$ an element of the vector defining
	 *     $G(i)$.                                                                               */
	static void dgebd2(int const m, int const n, real* const A, int const lda, real* const d,
	                   real* const e, real* const tauq, real* const taup, real* const work,
	                   int& info) /*const*/
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

	/*! §dgebrd
	 *
	 * §dgebrd reduces a general real §m by §n matrix $A$ to upper or lower bidiagonal form $B$ by
	 * an orthogonal transformation: $Q^T A P = B$.\n
	 * If $\{m}\ge\{n}$, $B$ is upper bidiagonal; if $\{m}<\{n}$, $B$ is lower bidiagonal.
	 * \param[in]     m The number of rows in the matrix $A$. $\{m}\ge 0$.
	 * \param[in]     n The number of columns in the matrix $A$. $\{n}\ge 0$.
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the §m by §n general matrix to be reduced.\n
	 *     On exit,
	 *     \li if $\{m}\ge\{n}$, the diagonal and the first superdiagonal are overwritten with the
	 *         upper bidiagonal matrix $B$; the elements below the diagonal, with the array §tauq,
	 *         represent the orthogonal matrix $Q$ as a product of elementary reflectors, and the
	 *         elements above the first superdiagonal, with the array §taup, represent the
	 *         orthogonal matrix $P$ as a product of elementary reflectors;
	 *     \li if $\{m}<\{n}$, the diagonal and the first subdiagonal are overwritten with the
	 *         lower bidiagonal matrix $B$; the elements below the first subdiagonal, with the
	 *         array §tauq, represent the orthogonal matrix $Q$ as a product of elementary
	 *         reflectors, and the elements above the diagonal, with the array §taup, represent the
	 *         orthogonal matrix $P$ as a product of elementary reflectors.
	 *
	 *     See Remark.
	 *
	 * \param[in]  lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[out] d
	 *     an array, dimension ($\min(\{m},\{n})$)\n
	 *     The diagonal elements of the bidiagonal matrix $B$: $\{d}[i] = \{A}[i,i]$.
	 *
	 * \param[out] e
	 *     an array, dimension ($\min(\{m},\{n})-1$)\n
	 *     The off-diagonal elements of the bidiagonal matrix $B$:\n
	 *     \li if $\{m}\ge\{n}$, $\{e}[i] = \{A}[i,i+1]$ for $i=0,1,\ldots,\{n}-2$;
	 *     \li if $\{m}<\{n}$,   $\{e}[i] = \{A}[i+1,i]$ for $i=0,1,\ldots,\{m}-2$.
	 *
	 * \param[out] tauq
	 *     an array, dimension ($\min(\{m},\{n})$)\n
	 *     The scalar factors of the elementary reflectors which represent the orthogonal matrix
	 *     $Q$. See Remark.
	 *
	 * \param[out] taup
	 *     an array, dimension ($\min(\{m},\{n})$)\n
	 *     The scalar factors of the elementary reflectors which represent the orthogonal matrix
	 *     $P$. See Remark.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$)\n
	 *     On exit, if $\{info}=0$, $\{work}[0]$ returns the optimal §lwork.
	 *
	 * \param[in] lwork
	 *     The length of the array §work. $\{lwork}\ge\max(1,\{m},\{n})$.\n
	 *     For optimum performance $\{lwork}\ge(\{m}+\{n})\{nb}$, where §nb is the optimal
	 *     blocksize.\n
	 *     If $\{lwork}=-1$, then a workspace query is assumed; the routine only calculates the
	 *     optimal size of the §work array, returns this value as the first entry of the §work
	 *     array, and no error message related to §lwork is issued by §xerbla.
	 *
	 * \param[out] info
	 *     =0: successful exit\n
	 *     <0: if $\{info}=-i$, the $i$-th argument had an illegal value.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date November 2017
	 * \remark
	 *     The matrices $Q$ and $P$ are represented as products of elementary reflectors:\n
	 *     If $\{m}\ge\{n}$,\n
	 *         $Q = H(0) H(1) \ldots H(\{n}-1)$ and $P = G(0) G(1) \ldots G(\{n}-2)$\n
	 *     Each $H(i)$ and $G(i)$ has the form:\n
	 *         $H(i) = I - \tau_q v v^T$ and $G(i) = I - \tau_p u u^T$\n
	 *     where $\tau_q$ and $\tau_p$ are real scalars, and $v$ and $u$ are real vectors;\n
	 *     $v[0:i-1]=0$, $v[i]=1$, and $v[i+1:m-1]$ is stored on exit in $\{A}[i+1:m-1,i]$;\n
	 *     $u[0:i]=0$, $u[i+1]=1$, and $u[i+2:n-1]$ is stored on exit in $\{A}[i,i+2:n-1]$;\n
	 *     $\tau_q$ is stored in $\{tauq}[i]$ and $\tau_p$ in $\{taup}[i]$.\n
	 *     \n
	 *     If $\{m}<\{n}$,\n
	 *         $Q = H(0) H(1) \ldots H(\{m}-2)$ and $P = G(0) G(1) \ldots G(\{m}-1)$\n
	 *     Each $H(i)$ and $G(i)$ has the form:\n
	 *         $H(i) = I - \tau_q v v^T$ and $G(i) = I - \tau_p u u^T$\n
	 *     where $\tau_q$ and $\tau_p$ are real scalars, and $v$ and $u$ are real vectors;\n
	 *     $v[0:i]=0$, $v[i+1]=1$, and $v[i+2:m-1]$ is stored on exit in $\{A}[i+2:m-1,i]$;\n
	 *     $u[0:i-1]=0$, $u[i]=1$, and $u[i+1:n-1]$ is stored on exit in $\{A}[i,i+1:n-1]$;\n
	 *     $\tau_q$ is stored in $\{tauq}[i]$ and $\tau_p$ in $\{taup}[i]$.\n
	 *     The contents of §A on exit are illustrated by the following examples:\n
	 *     \li $\{m}=6$ and $\{n}=5$ ($\{m}>\{n}$):\n
	 *         $\b{bm} d  &  e  & u_1 & u_1 & u_1 \\
	 *                v_1 &  d  &  e  & u_2 & u_2 \\
	 *                v_1 & v_2 &  d  &  e  & u_3 \\
	 *                v_1 & v_2 & v_3 &  d  &  e  \\
	 *                v_1 & v_2 & v_3 & v_4 &  d  \\
	 *                v_1 & v_2 & v_3 & v_4 & v_5 \e{bm}$
	 *     \li $\{m}=5$ and $\{n}=6$ ($\{m}<\{n}$):\n
	 *         $\b{bm} d  & u_1 & u_1 & u_1 & u_1 & u_1 \\
	 *                 e  &  d  & u_2 & u_2 & u_2 & u_2 \\
	 *                v_1 &  e  &  d  & u_3 & u_3 & u_3 \\
	 *                v_1 & v_2 &  e  &  d  & u_4 & u_4 \\
	 *                v_1 & v_2 & v_3 &  e  &  d  & u_5 \e{bm}$
	 *
	 *     where $d$ and $e$ denote diagonal and off-diagonal elements of $B$, $v_i$ denotes an
	 *     element of the vector defining $H(i)$, and $u_i$ an element of the vector defining
	 *     $G(i)$.                                                                               */
	static void dgebrd(int const m, int const n, real* const A, int const lda, real* const d,
	                   real* const e, real* const tauq, real* const taup, real* const work,
	                   int const lwork, int& info) /*const*/
	{
		// Test the input parameters
		info = 0;
		int nb = std::max(1, ilaenv(1, "DGEBRD", " ", m, n, -1, -1));
		int lwkopt = (m+n) * nb;
		work[0] = real(lwkopt);
		bool lquery = (lwork==-1);
		if (m<0)
		{
			info = -1;
		}
		else if (n<0)
		{
			info = -2;
		}
		else if (lda<std::max(1, m))
		{
			info = -4;
		}
		else if ((lwork<1 || lwork<std::max(m, n)) && !lquery)
		{
			info = -10;
		}
		if (info<0)
		{
			xerbla("DGEBRD", -info);
			return;
		}
		else if (lquery)
		{
			return;
		}
		// Quick return if possible
		int minmn = std::min(m, n);
		if (minmn==0)
		{
			work[0] = 1;
			return;
		}
		int ws = std::max(m, n);
		int ldwrkx = m;
		int ldwrky = n;
		int nbmin, nx;
		if (nb>1 && nb<minmn)
		{
			// Set the crossover point nx.
			nx = std::max(nb, ilaenv(3, "DGEBRD", " ", m, n, -1, -1));
			// Determine when to switch from blocked to unblocked code.
			if (nx<minmn)
			{
				ws = (m+n) * nb;
				if (lwork<ws)
				{
					// Not enough work space for the optimal nb,
					// consider using a smaller block size.
					nbmin = ilaenv(2, "DGEBRD", " ", m, n, -1, -1);
					if (lwork>=(m+n)*nbmin)
					{
						nb = lwork / (m+n);
					}
					else
					{
						nb = 1;
						nx = minmn;
					}
				}
			}
		}
		else
		{
			nx = minmn;
		}
		int i, j, ildai, inbldainb, jldaj;
		for (i=0; i<minmn-nx; i+=nb)
		{
			ildai = i + lda*i;
			inbldainb = nb + ildai + lda*nb;
			// Reduce rows and columns i:i+nb-1 to bidiagonal form and return the matrices X and Y
			// which are needed to update the unreduced part of the matrix
			dlabrd(m-i, n-i, nb, &A[ildai], lda, &d[i], &e[i], &tauq[i], &taup[i], work, ldwrkx,
			       &work[ldwrkx*nb], ldwrky);
			// Update the trailing submatrix A[i+nb:m-1,i+nb:n-1],
			// using an update of the form A := A - V*Y^T - X*U^T
			Blas<real>::dgemm("No transpose", "Transpose", m-i-nb, n-i-nb, nb, -ONE,
			                  &A[nb+ildai], lda, &work[ldwrkx*nb+nb], ldwrky, ONE, &A[inbldainb],
			                  lda);
			Blas<real>::dgemm("No transpose", "No transpose", m-i-nb, n-i-nb, nb, -ONE,
			                  &work[nb], ldwrkx, &A[ildai+lda*nb], lda, ONE, &A[inbldainb], lda);
			// Copy diagonal and off-diagonal elements of B back into A
			if (m>=n)
			{
				for (j=i; j<i+nb; j++)
				{
					jldaj = j + lda*j;
					A[jldaj]     = d[j];
					A[jldaj+lda] = e[j];
				}
			}
			else
			{
				for (j=i; j<i+nb; j++)
				{
					jldaj = j + lda*j;
					A[jldaj]   = d[j];
					A[1+jldaj] = e[j];
				}
			}
		}
		// Use unblocked code to reduce the remainder of the matrix
		int iinfo;
		dgebd2(m-i, n-i, &A[i+lda*i], lda, &d[i], &e[i], &tauq[i], &taup[i], work, iinfo);
		work[0] = ws;
	}

	/*! §dgeev computes the eigenvalues and, optionally, the left and/or right eigenvectors for GE
	 *  matrices
	 *
	 * §dgeev computes for an §N by §N real nonsymmetric matrix $A$, the eigenvalues and,
	 * optionally, the left and/or right eigenvectors.\n
	 * The right eigenvector $v_j$ of $A$ satisfies\n
	 *     $A v_j = \lambda_j v_j$\n
	 * where $\lambda_j$ is its eigenvalue.\n
	 * The left eigenvector $u_j$ of $A$ satisfies\n
	 *     $u_j^H A = \lambda_j u_j^H$\n
	 * where $u_j^H$ denotes the conjugate-transpose of $u_j$.\n
	 * The computed eigenvectors are normalized to have Euclidean norm equal to 1 and largest
	 * component real.
	 * \param[in] jobvl
	 *     ='N': left eigenvectors of $A$ are not computed;\n
	 *     ='V': left eigenvectors of $A$ are computed.
	 *
	 * \param[in] jobvr
	 *     = 'N': right eigenvectors of $A$ are not computed;\n
	 *     = 'V': right eigenvectors of $A$ are computed.
	 *
	 * \param[in]     n The order of the matrix $A$. $\{n}\ge 0$.
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the §n by §n matrix $A$.\n On exit, §A has been overwritten.
	 *
	 * \param[in]  lda    The leading dimension of the array §A. $\{lda}\ge\max(1,\{n})$.
	 * \param[out] wr, wi
	 *     arrays, dimension (§n)\n
	 *     §wr and §wi contain the real and imaginary parts, respectively, of the computed
	 *     eigenvalues. Complex conjugate pairs of eigenvalues appear consecutively with the
	 *     eigenvalue having the positive imaginary part first.
	 *
	 * \param[out] Vl
	 *     an array, dimension (§ldvl,§n)\n
	 *     If §jobvl ='V', the left eigenvectors $u_j$ are stored one after another in the columns
	 *     of §Vl, in the same order as their eigenvalues.\n
	 *     If §jobvl ='N', §Vl is not referenced.\n
	 *     If the $j$-th eigenvalue is real, then $u_j=\{Vl}[:,j]$, the $j$-th column of §Vl. \n
	 *     If the $j$-th and $j+1$-st eigenvalues form a complex conjugate pair, then
	 *     $u_j = \{Vl}[:,j] + i\{Vl}[:,j+1]$ and $u_{j+1} = \{Vl}[:,j] - i\{Vl}[:,j+1]$.
	 *
	 * \param[in] ldvl
	 *     The leading dimension of the array §Vl. $\{ldvl}\ge 1$;
	 *     if §jobvl ='V', $\{ldvl}\ge\{n}$.
	 *
	 * \param[out] Vr
	 *     an array, dimension (§ldvr,§n)\n
	 *     If §jobvr ='V', the right eigenvectors $v_j$ are stored one after another in the columns
	 *     of §Vr, in the same order as their eigenvalues.\n
	 *     If §jobvr ='N', §Vr is not referenced.\n
	 *     If the $j$-th eigenvalue is real, then $v_j=\{Vr}[:,j]$, the $j$-th column of §Vr. \n
	 *     If the $j$-th and $j+1$-st eigenvalues form a complex conjugate pair, then
	 *     $v_j = \{Vr}[:,j] + i\{Vr}[:,j+1]$ and $v_{j+1} = \{Vr}[:,j] - i\{Vr}[:,j+1]$.
	 *
	 * \param[in] ldvr
	 *     The leading dimension of the array §Vr. $\{ldvr}\ge 1$;
	 *     if §jobvr ='V', $\{ldvr}\ge\{n}$.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$)\n
	 *     On exit, if $\{info}=0$, $\{work}[0]$ returns the optimal §lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work. $\{lwork}\ge\max(1,3\{n})$, and if §jobvl ='V' or
	 *     §jobvr ='V', $\{lwork}\ge 4\{n}$. For good performance, §lwork must generally be larger.
	 *     \n
	 *     If $\{lwork}=-1$, then a workspace query is assumed; the routine only calculates the
	 *     optimal size of the §work array, returns this value as the first entry of the §work
	 *     array, and no error message related to §lwork is issued by §XERBLA.
	 *
	 * \param[out] info
	 *     =0: successful exit\n
	 *     <0: if $\{info}=-i$, the $i$-th argument had an illegal value.\n.
	 *     >0: if $\{info}=i$, the QR algorithm failed to compute all the eigenvalues, and no
	 *         eigenvectors have been computed; elements $i:\{n}-1$ of §wr and §wi contain
	 *         eigenvalues which have converged.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016                                                                           */
	static void dgeev(char const* const jobvl, char const* const jobvr, int const n, real* const A,
	                  int const lda, real* const wr, real* const wi, real* const Vl,
	                  int const ldvl, real* const Vr, int const ldvr, real* const work,
	                  int const lwork, int& info) /*const*/
	{
		// Test the input arguments
		info = 0;
		bool lquery = (lwork==-1);
		bool wantvl = (std::toupper(jobvl[0])=='V');
		bool wantvr = (std::toupper(jobvr[0])=='V');
		if (!wantvl && std::toupper(jobvl[0])!='N')
		{
			info = -1;
		}
		else if (!wantvr && std::toupper(jobvr[0])!='N')
		{
			info = -2;
		}
		else if (n<0)
		{
			info = -3;
		}
		else if (lda<std::max(1, n))
		{
			info = -5;
		}
		else if (ldvl<1 || (wantvl && ldvl<n))
		{
			info = -9;
		}
		else if (ldvr<1 || (wantvr && ldvr<n))
		{
			info = -11;
		}
		/* Compute workspace
		 * (Note: Comments in the code beginning "Workspace:" describe the minimal amount of
		 *  workspace needed at that point in the code, as well as the preferred amount for good
		 *  performance. NB refers to the optimal block size for the immediately following
		 *  subroutine, as returned by ilaenv. hswork refers to the workspace preferred by dhseqr,
		 *  as calculated below. hswork is computed assuming ilo=0 and ihi=n-1, the worst case.) */
		int ierr, maxwrk, nout;
		if (info==0)
		{
			int minwrk;
			if (n==0)
			{
				minwrk = 1;
				maxwrk = 1;
			}
			else
			{
				maxwrk = 2*n + n*ilaenv(1, "DGEHRD", " ", n, 1, n, 0);
				int hswork, lwork_trevc;
				if (wantvl)
				{
					minwrk = 4 * n;
					maxwrk = std::max(maxwrk, 2*n+(n-1)*ilaenv(1, "DORGHR", " ", n, 1, n, -1));
					dhseqr("S", "V", n, 0, n-1, A, lda, wr, wi, Vl, ldvl, work, -1, info);
					hswork = int(work[0]);
					maxwrk = std::max(maxwrk, std::max(n+1, n+hswork));
					dtrevc3("L", "B", nullptr, n, A, lda, Vl, ldvl, Vr, ldvr, n, nout, work, -1,
					        ierr);
					lwork_trevc = int(work[0]);
					maxwrk = std::max(maxwrk, n+lwork_trevc);
					maxwrk = std::max(maxwrk, 4*n);
				}
				else if (wantvr)
				{
					minwrk = 4*n;
					maxwrk = std::max(maxwrk, 2*n+(n-1)*ilaenv(1, "DORGHR", " ", n, 1, n, -1));
					dhseqr("S", "V", n, 0, n-1, A, lda, wr, wi, Vr, ldvr, work, -1, info);
					hswork = int(work[0]);
					maxwrk = std::max(maxwrk, std::max(n+1, n+hswork));
					dtrevc3("R", "B", nullptr, n, A, lda, Vl, ldvl, Vr, ldvr, n, nout, work, -1,
					        ierr);
					lwork_trevc = int(work[0]);
					maxwrk = std::max(maxwrk, n+lwork_trevc);
					maxwrk = std::max(maxwrk, 4*n);
				}
				else
				{
					minwrk = 3 * n;
					dhseqr("E", "N", n, 0, n-1, A, lda, wr, wi, Vr, ldvr, work, -1, info);
					hswork = int(work[0]);
					maxwrk = std::max(maxwrk, std::max(n+1, n+hswork));
				}
				maxwrk = std::max(maxwrk, minwrk);
			}
			work[0] = maxwrk;
			if (lwork<minwrk && !lquery)
			{
				info = -13;
			}
		}
		if (info!=0)
		{
			xerbla("DGEEV ", -info);
			return;
		}
		else if (lquery)
		{
			return;
		}
		// Quick return if possible
		if (n==0)
		{
			return;
		}
		// Get machine constants
		real eps    = dlamch("P");
		real smlnum = dlamch("S");
		real bignum = ONE / smlnum;
		dlabad(smlnum, bignum);
		smlnum = std::sqrt(smlnum) / eps;
		bignum = ONE / smlnum;
		// Scale A if max element outside range [smlnum,bignum]
		real anrm = dlange("M", n, n, A, lda, nullptr);
		bool scalea = false;
		real cscale;
		if (anrm>ZERO && anrm<smlnum)
		{
			scalea = true;
			cscale = smlnum;
		}
		else if (anrm>bignum)
		{
			scalea = true;
			cscale = bignum;
		}
		if (scalea)
		{
			dlascl("G", 0, 0, anrm, cscale, n, n, A, lda, ierr);
		}
		// Balance the matrix
		// (Workspace: need n)
		int ibal = 0;
		int ihi, ilo;
		dgebal("B", n, A, lda, ilo, ihi, &work[ibal], ierr);
		// Reduce to upper Hessenberg form
		// (Workspace: need 3*n, prefer 2*n+n*NB)
		int itau = ibal + n;
		int iwrk = itau + n;
		dgehrd(n, ilo, ihi, A, lda, &work[itau], &work[iwrk], lwork-iwrk, ierr);
		char side[2];
		side[1] = '\0';
		if (wantvl)
		{
			// Want left eigenvectors
			// Copy Householder vectors to Vl
			side[0] = 'L';
			dlacpy("L", n, n, A, lda, Vl, ldvl);
			// Generate orthogonal matrix in Vl
			// (Workspace: need 3*n-1, prefer 2*n+(n-1)*NB)
			dorghr(n, ilo, ihi, Vl, ldvl, &work[itau], &work[iwrk], lwork-iwrk, ierr);
			// Perform QR iteration, accumulating Schur vectors in Vl
			// (Workspace: need n+1, prefer n+hswork (see comments))
			iwrk = itau;
			dhseqr("S", "V", n, ilo, ihi, A, lda, wr, wi, Vl, ldvl, &work[iwrk], lwork-iwrk, info);
			if (wantvr)
			{
				// Want left and right eigenvectors
				// Copy Schur vectors to Vr
				side[0] = 'B';
				dlacpy("F", n, n, Vl, ldvl, Vr, ldvr);
			}
		}
		else if (wantvr)
		{
			// Want right eigenvectors
			// Copy Householder vectors to Vr
			side[0] = 'R';
			dlacpy("L", n, n, A, lda, Vr, ldvr);
			// Generate orthogonal matrix in Vr
			// (Workspace: need 3*n-1, prefer 2*n+(n-1)*NB)
			dorghr(n, ilo, ihi, Vr, ldvr, &work[itau], &work[iwrk], lwork-iwrk, ierr);
			// Perform QR iteration, accumulating Schur vectors in Vr
			// (Workspace: need n+1, prefer n+hswork (see comments))
			iwrk = itau;
			dhseqr("S", "V", n, ilo, ihi, A, lda, wr, wi, Vr, ldvr, &work[iwrk], lwork-iwrk, info);
		}
		else
		{
			// Compute eigenvalues only
			// (Workspace: need n+1, prefer n+hswork (see comments))
			iwrk = itau;
			dhseqr("E", "N", n, ilo, ihi, A, lda, wr, wi, Vr, ldvr, &work[iwrk], lwork-iwrk, info);
		}
		// If info != 0 from dhseqr, then quit
		if (info==0)
		{
			if (wantvl || wantvr)
			{
				// Compute left and/or right eigenvectors
				// (Workspace: need 4*n, prefer n + n + 2*n*NB)
				dtrevc3(side, "B", nullptr, n, A, lda, Vl, ldvl, Vr, ldvr, n, nout, &work[iwrk],
				        lwork-iwrk, ierr);
			}
			int i, k, vi, vip;
			real cs, r, scl, sn, temp1, temp2;
			if (wantvl)
			{
				// Undo balancing of left eigenvectors
				// (Workspace: need n)
				dgebak("B", "L", n, ilo, ihi, &work[ibal], n, Vl, ldvl, ierr);
				// Normalize left eigenvectors and make largest component real
				for (i=0; i<n; i++)
				{
					vi = ldvl * i;
					if (wi[i]==ZERO)
					{
						scl = ONE / Blas<real>::dnrm2(n, &Vl[vi], 1);
						Blas<real>::dscal(n, scl, &Vl[vi], 1);
					}
					else if (wi[i]>ZERO)
					{
						vip = vi + ldvl;
						scl = ONE / dlapy2(Blas<real>::dnrm2(n, &Vl[vi],  1),
						                   Blas<real>::dnrm2(n, &Vl[vip], 1));
						Blas<real>::dscal(n, scl, &Vl[vi],  1);
						Blas<real>::dscal(n, scl, &Vl[vip], 1);
						for (k=0; k<n; k++)
						{
							temp1 = Vl[k+vi];
							temp2 = Vl[k+vip];
							work[iwrk+k] = temp1*temp1 + temp2*temp2;
						}
						k = Blas<real>::idamax(n, &work[iwrk], 1);
						dlartg(Vl[k+vi], Vl[k+vip], cs, sn, r);
						Blas<real>::drot(n, &Vl[vi], 1, &Vl[vip], 1, cs, sn);
						Vl[k+vip] = ZERO;
					}
				}
			}
			if (wantvr)
			{
				// Undo balancing of right eigenvectors
				// (Workspace: need n)
				dgebak("B", "R", n, ilo, ihi, &work[ibal], n, Vr, ldvr, ierr);
				// Normalize right eigenvectors and make largest component real
				for (i=0; i<n; i++)
				{
					vi = ldvr * i;
					if (wi[i]==ZERO)
					{
						scl = ONE / Blas<real>::dnrm2(n, &Vr[vi], 1);
						Blas<real>::dscal(n, scl, &Vr[vi], 1);
					}
					else if (wi[i]>ZERO)
					{
						vip = vi + ldvr;
						scl = ONE / dlapy2(Blas<real>::dnrm2(n, &Vr[vi],  1),
						                   Blas<real>::dnrm2(n, &Vr[vip], 1));
						Blas<real>::dscal(n, scl, &Vr[vi],  1);
						Blas<real>::dscal(n, scl, &Vr[vip], 1);
						for (k=0; k<n; k++)
						{
							temp1 = Vr[k+vi];
							temp2 = Vr[k+vip];
							work[iwrk+k] = temp1*temp1 + temp2*temp2;
						}
						k = Blas<real>::idamax(n, &work[iwrk], 1);
						dlartg(Vr[k+vi], Vr[k+vip], cs, sn, r);
						Blas<real>::drot(n, &Vr[vi], 1, &Vr[vip], 1, cs, sn);
						Vr[k+vip] = ZERO;
					}
				}
			}
		}
		// Undo scaling if necessary
		if (scalea)
		{
			dlascl("G", 0, 0, cscale, anrm, n-info, 1, &wr[info], std::max(n-info, 1), ierr);
			dlascl("G", 0, 0, cscale, anrm, n-info, 1, &wi[info], std::max(n-info, 1), ierr);
			if (info>0)
			{
				dlascl("G", 0, 0, cscale, anrm, ilo, 1, wr, n, ierr);
				dlascl("G", 0, 0, cscale, anrm, ilo, 1, wi, n, ierr);
			}
		}
		work[0] = maxwrk;
	}

	/*! §dgehd2 reduces a general square matrix to upper Hessenberg form using an unblocked
	 *  algorithm.
	 *
	 * §dgehd2 reduces a real general matrix $A$ to upper Hessenberg form $H$ by an orthogonal
	 * similarity transformation: $Q^T A Q = H$.
	 * \param[in] n    The order of the matrix $A$. $\{n} \ge 0$.
	 * \param[in] ilo,
	 *            ihi
	 *     It is assumed that $A$ is already upper triangular in rows and columns §0:§ilo-1 and
	 *     §ihi+1:§n-1. §ilo and §ihi are normally set by a previous call to §dgebal; otherwise they
	 *     should be set to 0 and §n-1 respectively. See Remark.\n
	 *     $0 \le \{ilo} \le \{ihi} < \max(1,\{n})$.\n
	 *     Note: zero-based indices!
	 *
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the §n by §n general matrix to be reduced.\n
	 *     On exit, the upper triangle and the first subdiagonal of §A are overwritten with the
	 *     upper Hessenberg matrix $H$, and the elements below the first subdiagonal, with the
	 *     array §tau, represent the orthogonal matrix $Q$ as a product of elementary reflectors.
	 *     See Remark.
	 *
	 * \param[in]  lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{n})$.
	 * \param[out] tau
	 *     an array, dimension (§n-1)\n
	 *     The scalar factors of the elementary reflectors (see Remark).
	 *
	 * \param[out] work an array, dimension (§n)
	 * \param[out] info
	 *     = 0: successful exit.\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     The matrix $Q$ is represented as a product of (§ihi-§ilo) elementary reflectors\n
	 *         $Q = H(\{ilo}) H(\{ilo}+1) \ldots H(\{ihi}-1)$.\n
	 *     Each $H(i)$ has the form\n
	 *         $H(i) = I - \tau v v^T$\n
	 *     where $\tau$ is a real scalar, and $v$ is a real vector with $v[0:i] = 0$, $v[i+1] = 1$
	 *     and $v[\{ihi}+1:\{n}-1] = 0$; $v[i+2:\{ihi}]$ is stored on exit in $\{A}[i+2:\{ihi},i]$,
	 *     and $\tau$ in $\{tau}[i]$.\n
	 *     The contents of §A are illustrated by the following example, with $\{n} = 7$,
	 *     $\{ilo} = 1$ and $\{ihi} = 5$:\n
	 *     on entry,\n
	 *         $\b{bm} a & a & a & a & a & a & a \\
	 *                   & a & a & a & a & a & a \\
	 *                   & a & a & a & a & a & a \\
	 *                   & a & a & a & a & a & a \\
	 *                   & a & a & a & a & a & a \\
	 *                   & a & a & a & a & a & a \\
	 *                   &   &   &   &   &   & a \e{bm}$\n
	 *     on exit,\n
	 *         $\b{bm} a &  a  &  h  &  h  & h & h & a \\
	 *                   &  a  &  h  &  h  & h & h & a \\
	 *                   &  h  &  h  &  h  & h & h & h \\
	 *                   & v_1 &  h  &  h  & h & h & h \\
	 *                   & v_1 & v_2 &  h  & h & h & h \\
	 *                   & v_1 & v_2 & v_3 & h & h & h \\
	 *                   &     &     &     &   &   & a \e{bm}$\n
	 *     where $a$ denotes an element of the original matrix $A$, $h$ denotes a modified element
	 *     of the upper Hessenberg matrix $H$, and $v_i$ denotes an element of the vector defining
	 *     $H(i)$.                                                                               */
	static void dgehd2(int const n, int const ilo, int const ihi, real* const A, int const lda,
	                   real* const tau, real* const work, int& info) /*const*/
	{
		// Test the input parameters
		info = 0;
		if (n<0)
		{
			info = -1;
		}
		else if (ilo<0 || ilo>=std::max(1, n))
		{
			info = -2;
		}
		else if (ihi<std::min(ilo, n-1) || ihi>=n)
		{
			info = -3;
		}
		else if (lda<std::max(1, n))
		{
			info = -5;
		}
		if (info!=0)
		{
			xerbla("DGEHD2", -info);
			return;
		}
		real aii;
		int acoli, acolip, ip1i;
		for (int i=ilo; i<ihi; i++)
		{
			acoli = lda * i;
			acolip = acoli + lda;
			ip1i = i + 1 + acoli;
			// Compute elementary reflector H(i) to annihilate A[i+2:ihi,i]
			dlarfg(ihi-i, A[ip1i], &A[std::min(i+2, n-1)+acoli], 1, tau[i]);
			aii = A[ip1i];
			A[ip1i] = ONE;
			// Apply H(i) to A[0:ihi,i+1:ihi] from the right
			dlarf("Right", ihi+1, ihi-i, &A[ip1i], 1, tau[i], &A[acolip], lda, work);
			// Apply H(i) to A[i+1:ihi,i+1:n-1] from the left
			dlarf("Left", ihi-i, n-1-i, &A[ip1i], 1, tau[i], &A[i+1+acolip], lda, work);
			A[ip1i] = aii;
		}
	}

	/*! §dgehrd
	 *
	 * §dgehrd reduces a real general matrix $A$ to upper Hessenberg form $H$ by an orthogonal
	 * similarity transformation: $Q^T A Q = H$.
	 * \param[in] n        The order of the matrix $A$. $\{n} \ge 0$.
	 * \param[in] ilo, ihi
	 *     It is assumed that $A$ is already upper triangular in rows and columns $0:\{ilo}-1$ and
	 *     $\{ihi}+1:\{n}-1$. §ilo and §ihi are normally set by a previous call to §dgebal;
	 *     otherwise they should be set to 0 and $\{n}-1$ respectively.\n
	 *     See Remark.\n
	 *     $0\le\{ilo}\le\{ihi}<\{n}$, if $\{n}>0$;\n
	 *     $\{ilo}=0$ and $\{ihi}=-1$, if $\{n}=0$.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the §n by §n general matrix to be reduced.\n
	 *     On exit, the upper triangle and the first subdiagonal of §A are overwritten with the
	 *     upper Hessenberg matrix $H$, and the elements below the first subdiagonal, with the
	 *     array §tau, represent the orthogonal matrix $Q$ as a product of elementary reflectors.\n
	 *     See Remark.
	 *
	 * \param[in]  lda The leading dimension of the array §A. $\{lda} \ge \max(1,\{n})$.
	 * \param[out] tau
	 *     an array, dimension (§n-1)\n
	 *     The scalar factors of the elementary reflectors (see Remark).
	 *     Elements $0:\{ilo}-1$ and $\{ihi}:\{n}-2$ of §tau are set to zero.
	 *
	 * \param[out] work
	 *     an array, dimension (§lwork)\n
	 *     On exit, if $\{info}=0$, $\{work}[0]$ returns the optimal §lwork.
	 *
	 * \param[in] lwork
	 *     The length of the array §work. $\{lwork}\ge\max(1,\{n})$.\n
	 *     For good performance, §lwork should generally be larger.\n
	 *     If §lwork = -1, then a workspace query is assumed; the routine only calculates the
	 *     optimal size of the work array, returns this value as the first entry of the §work
	 *     array, and no error message related to §lwork is issued by §xerbla.
	 *
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     The matrix $Q$ is represented as a product of (§ihi-§ilo) elementary reflectors\n
	 *         $Q = H(\{ilo}) H(\{ilo}+1) \ldots H(\{ihi}-1)$.\n
	 *     Each $H(i)$ has the form\n
	 *         $H(i) = I - \tau v v^T$\n
	 *     where $\tau$ is a real scalar, and $v$ is a real vector with $v[0:i]=0$, $v[i+1]=1$
	 *     and $v[\{ihi}+1:\{n}-1]=0$; $v[i+2:\{ihi}-1]$ is stored on exit in
	 *     $\{A}[i+2:\{ihi},i]$, and $\tau$ in $\{tau}[i]$.\n
	 *     The contents of §A are illustrated by the following example, with $\{n}=7$, $\{ilo}=1$
	 *     and $\{ihi}=5$:\n
	 *         on entry,\n
	 *         $\b{bm} a & a & a & a & a & a & a \\
	 *                   & a & a & a & a & a & a \\
	 *                   & a & a & a & a & a & a \\
	 *                   & a & a & a & a & a & a \\
	 *                   & a & a & a & a & a & a \\
	 *                   & a & a & a & a & a & a \\
	 *                   &   &   &   &   &   & a \e{bm}$\n
	 *         on exit,\n
	 *         $\b{bm} a &  a  &  h  &  h  & h & h & a \\
	 *                   &  a  &  h  &  h  & h & h & a \\
	 *                   &  h  &  h  &  h  & h & h & h \\
	 *                   & v_2 &  h  &  h  & h & h & h \\
	 *                   & v_2 & v_3 &  h  & h & h & h \\
	 *                   & v_2 & v_3 & v_4 & h & h & h \\
	 *                   &     &     &     &   &   & a \e{bm}$\n
	 *     where $a$ denotes an element of the original matrix $A$, $h$ denotes a modified element
	 *     of the upper Hessenberg matrix $H$, and $v_i$ denotes an element of the vector defining
	 *     $H(i)$.\n
	 *     This file is a slight modification of LAPACK-3.0's §dgehrd subroutine incorporating
	 *     improvements proposed by Quintana-Orti and Van de Geijn (2006). (See §dlahr2.)        */
	static void dgehrd(int const n, int const ilo, int const ihi, real* const A, int const lda,
	                   real* const tau, real* const work, int const lwork, int& info) /*const*/
	{
		const int NBMAX = 64;
		const int LDT = NBMAX + 1;
		const int TSIZE = LDT * NBMAX;
		// Test the input parameters
		info = 0;
		bool lquery = (lwork==-1);
		if (n<0)
		{
			info = -1;
		}
		else if (ilo<0 || ilo>std::max(0, n-1))
		{
			info = -2;
		}
		else if (ihi<std::min(ilo, n-1) || ihi>=n)
		{
			info = -3;
		}
		else if (lda<std::max(1, n))
		{
			info = -5;
		}
		else if (lwork<std::max(1, n) && !lquery)
		{
			info = -8;
		}
		int lwkopt=0, nb;
		if (info==0)
		{
			// Compute the workspace requirements
			nb = std::min(NBMAX, ilaenv(1, "DGEHRD", " ", n, ilo+1, ihi+1, -1));
			lwkopt = n*nb + TSIZE;
			work[0] = lwkopt;
		}
		if (info!=0)
		{
			xerbla("DGEHRD", -info);
			return;
		}
		else if (lquery)
		{
			return;
		}
		int i;
		// Set elements 0:ilo-1 and ihi:n-1 of tau to zero
		for (i=0; i<ilo; i++)
		{
			tau[i] = ZERO;
		}
		for (i=std::max(0, ihi); i<n-1; i++)
		{
			tau[i] = ZERO;
		}
		// Quick return if possible
		int nh = ihi - ilo + 1;
		if (nh<=1)
		{
			work[0] = 1;
			return;
		}
		// Determine the block size
		nb = std::min(NBMAX, ilaenv(1, "DGEHRD", " ", n, ilo+1, ihi+1, -1));
		int nbmin = 2, nx = 0;
		if (nb>1 && nb<nh)
		{
			// Determine when to cross over from blocked to unblocked code
			// (last block is always handled by unblocked code)
			nx = std::max(nb, ilaenv(3, "DGEHRD", " ", n, ilo+1, ihi+1, -1));
			if (nx<nh)
			{
				// Determine if workspace is large enough for blocked code
				if (lwork<n*nb+TSIZE)
				{
					// Not enough workspace to use optimal nb: determine the minimum value of nb,
					// and reduce nb or force use of unblocked code
					nbmin = std::max(2, ilaenv(2, "DGEHRD", " ", n, ilo+1, ihi+1, -1));
					if (lwork>=(n*nbmin + TSIZE))
					{
						nb = (lwork-TSIZE) / n;
					}
					else
					{
						nb = 1;
					}
				}
			}
		}
		int ldwork = n;
		if (nb<nbmin || nb>=nh)
		{
			// Use unblocked code below
			i = ilo;
		}
		else
		{
			// Use blocked code
			int iwt = n * nb;
			int ib, j;
			real ei;
			for (i=ilo; i<ihi-nx; i+=nb)//40
			{
				ib = std::min(nb, ihi-i);
				// Reduce columns i:i+ib-1 to Hessenberg form, returning the matrices V and T of
				// the block reflector H = I - V*T*V^T which performs the reduction, and also the
				// matrix Y = A*V*T
				dlahr2(ihi+1, i+1, ib, &A[lda*i], lda, &tau[i], &work[iwt], LDT, work, ldwork);
				// Apply the block reflector H to A[0:ihi,i+ib:ihi] from the right,
				// computing  A := A - Y * V^T. V[i+ib,ib-2] must be set to 1
				ei = A[i+ib+lda*(i+ib-1)];
				A[i+ib+lda*(i+ib-1)] = ONE;
				Blas<real>::dgemm("No transpose", "Transpose", ihi+1, ihi-i-ib+1, ib, -ONE, work,
				                  ldwork, &A[i+ib+lda*i], lda, ONE, &A[lda*(i+ib)], lda);
				A[i+ib+lda*(i+ib-1)] = ei;
				// Apply the block reflector H to A[0:i,i+1:i+ib-1] from the right
				Blas<real>::dtrmm("Right", "Lower", "Transpose", "Unit", i+1, ib-1, ONE,
				                  &A[i+1+lda*i], lda, work, ldwork);
				for (j=0; j<ib-1; j++)
				{
					Blas<real>::daxpy(i+1, -ONE, &work[ldwork*j], 1, &A[lda*(i+1+j)], 1);
				}
				// Apply the block reflector H to A[i+1:ihi,i+ib:n-1] from the left
				dlarfb("Left", "Transpose", "Forward", "Columnwise", ihi-i, n-i-ib, ib,
				       &A[i+1+lda*i], lda, &work[iwt], LDT, &A[i+1+lda*(i+ib)], lda, work, ldwork);
			}
		}
		// Use unblocked code to reduce the rest of the matrix
		int iinfo;
		dgehd2(n, i, ihi, A, lda, tau, work, iinfo);
		work[0] = lwkopt;
	}

	/*! §dgeqp3
	 *
	 * §dgeqp3 computes a QR factorization with column pivoting of a matrix $A$:
	 * $A P = Q R$
	 * using Level 3 BLAS.
	 * \param[in]     m The number of rows of the matrix $A$. $\{m} \ge 0$.
	 * \param[in]     n The number of columns of the matrix $A$. $\{n} \ge 0$.
	 * \param[in,out] A
	 *     an array, dimension (§lda, §n)\n
	 *     On entry, the §m by §n matrix $A$.\n
	 *     On exit, the upper triangle of the array contains the $\min(\{m},\{n})$ by §n upper
	 *              trapezoidal matrix $R$;\n&emsp;&emsp;
	 *              the elements below the diagonal, together with the array §tau, represent the
	 *              orthogonal matrix $Q$ as a product of $\min(\{m},\{n})$ elementary reflectors.
	 *
	 * \param[in]     lda The leading dimension of the array §A. $\{lda} \ge \max(1,\{m})$.
	 * \param[in,out] jpvt
	 *     an integer array, dimension (§n)\n
	 *     On entry,
	 *     \li if $\{jpvt}[j]\ne -1$, the $j$-th column of §A is permuted to the front of $AP$
	 *                                (a leading column);
	 *     \li if $\{jpvt}[j]= -1$,   the $j$-th column of §A is a free column.
	 *
	 *     On exit, if $\{jpvt}[j] = \{k}$, then the $j$-th column of $AP$ was the the §k -th
	 *                                      column of §A.\n
	 *     Note: this array contains zero-based indices!
	 *
	 * \param[out] tau
	 *     an array, dimension ($\min(\{m},\{n})$)\n
	 *     The scalar factors of the elementary reflectors.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$)\n
	 *     On exit, if §info = 0, §work[0] returns the optimal §lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work. $\{lwork} \ge 3\{n}+1$.\n
	 *     For optimal performance $\{lwork} \ge 2\{n}+(\{n}+1)\{nb}$, where §nb is the optimal
	 *     blocksize. If §lwork = -1, then a workspace query is assumed; the routine only
	 *     calculates the optimal size of the work array, returns this value as the first entry of
	 *     the work array, and no error message related to §lwork is issued by §xerbla.
	 *
	 * \param[out] info
	 *     =0: successful exit.\n
	 *     <0: if §info = -§i, the §i-th argument had an illegal value.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     The matrix $Q$ is represented as a product of elementary reflectors\n
	 *         $Q = H(0) H(1) \ldots H(\{k}-1)$, where $\{k} = \min(\{m},\{n})$.\n
	 *     Each $H(i)$ has the form\n
	 *         $H(i) = I - \{tau}[i] v v^T$\n
	 *     where $v$ is a real/complex vector with $v[0:i-1]=0$ and $v[i]=1$;
	 *     $v[i+1:\{m}-1]$ is stored on exit in §A$[i+1:\{m}-1,i]$.                              */
	static void dgeqp3(int const m, int const n, real* const A, int const lda, int* const jpvt,
	                   real* const tau, real* const work, int const lwork, int& info) /*const*/
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

	/*! §dgeqr2 computes the QR factorization of a general rectangular matrix using an unblocked
	 *  algorithm.
	 *
	 * §dgeqr2 computes a QR factorization of a real §m by §n matrix $A$: $A = Q R$.
	 * \param[in]     m The number of rows of the matrix $A$. $\{m} \ge 0$.
	 * \param[in]     n The number of columns of the matrix $A$. $\{n} \ge 0$.
	 * \param[in,out] A
	 *     an array, dimension (§lda, §n)\n
	 *     On entry, the §m by §n matrix $A$.
	 *     On exit, the elements on and above the diagonal of the array contain the
	 *     $\min(\{m},\{n})$ by §n upper trapezoidal matrix $R$ ($R$ is upper triangular if
	 *     $\{m}\ge\{n}$);\n
	 *     the elements below the diagonal, with the array §tau, represent the orthogonal matrix
	 *     $Q$ as a product of elementary reflectors (see remark).
	 *
	 * \param[in]  lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[out] tau
	 *     an array, dimension ($\min(\{m},\{n})$)\n
	 *     The scalar factors of the elementary reflectors (see remark).
	 *
	 * \param[out] work an array, dimension (§n)
	 * \param[out] info
	 *     =0: successful exit\n
	 *     <0: if §info = -§i, the §i -th argument had an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     The matrix $Q$ is represented as a product of elementary reflectors\n
	 *         $Q = H(1) H(2) \ldots H(k)$, where $\{k}=\min(\{m},\{n})$.\n
	 *     Each $H(i)$ has the form\n
	 *         $H(i) = I - \{tau}[i] v v^T$\n
	 *     where v is a real vector with $v[0:i-1]=0$ and $v[i]=1$;
	 *     $v[i+1:\{m}-1]$ is stored on exit in §A$[i+1:\{m}-1,i]$.                              */
	static void dgeqr2(int const m, int const n, real* const A, int const lda, real* const tau,
	                   real* const work, int& info) /*const*/
	{
		int i, k, coli, icoli;
		real aii;
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
			icoli = i+coli;
			// Generate elementary reflector H[i] to annihilate A[i+1:m-1, i]
			dlarfg(m-i, A[icoli], &A[std::min(i+1,m-1)+coli], 1, tau[i]);
			if (i<(n-1))
			{
				// Apply H[i] to A[i:m-1, i+1:n-1] from the left
				aii = A[icoli];
				A[icoli] = ONE;
				dlarf("Left", m-i, n-i-1, &A[icoli], 1, tau[i], &A[icoli+lda], lda, work);
				A[icoli] = aii;
			}
		}
	}

	/*! dgeqrf
	 *
	 * §dgeqrf computes a QR factorization of a real §m by §n matrix $A$:\n
	 * $A = Q R$.
	 * \param[in]     m The number of rows of the matrix $A$. $\{m} \ge 0$.
	 * \param[in]     n The number of columns of the matrix $A$. $\{n} \ge 0$.
	 * \param[in,out] A
	 *     an array, dimension (§lda, §n)\n
	 *     On entry, the §m by §n matrix $A$.\n
	 *     On exit, the elements on and above the diagonal of the array contain the
	 *         $\min(\{m},\{n})$ by §n upper trapezoidal matrix $R$ ($R$ is upper triangular
	 *         if $\{m}\ge\{n})$; the elements below the diagonal, with the array §tau, represent
	 *         the orthogonal matrix $Q$ as a product of $\min(\{m},\{n})$ elementary reflectors
	 *         (see remark).
	 *
	 * \param[in]  lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[out] tau
	 *     an array, dimension ($\min(\{m},\{n})$)\n
	 *     The scalar factors of the elementary reflectors (see remark).
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$)\n
	 *     On exit, if §info = 0, §work[0] returns the optimal lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work. $\{lwork}\ge\max(1,\{n})$.\n
	 *     For optimum performance $\{lwork}\ge\{n}\{nb}$, where §nb is the optimal blocksize.\n
	 *     If §lwork = -1, then a workspace query is assumed; the routine only calculates the
	 *     optimal size of the work array, returns this value as the first entry of the work array,
	 *     and no error message related to §lwork is issued by §xerbla.
	 *
	 * \param[out] info
	 *     =0: successful exit\n
	 *     <0: If §info = -§i, the §i -th argument had an illegal value.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     The matrix $Q$ is represented as a product of elementary reflectors\n
	 *         $Q = H(1) H(2) \ldots H(k)$, where $\{k}=\min(\{m},\{n})$.\n
	 *     Each $H(i)$ has the form\n
	 *         $H(i) = I - \{tau}[i] v v^T$\n
	 *     where v is a real vector with $v[0:i-1]=0$ and $v[i]=1$;
	 *     $v[i+1:\{m}-1]$ is stored on exit in §A$[i+1:\{m}-1,i]$.                              */
	static void dgeqrf(int const m, int const n, real* const A, int const lda, real* const tau,
	                   real* const work, int const lwork, int& info) /*const*/
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
			nx = std::max(0, ilaenv(3, "DGEQRF", " ", m, n, -1, -1));
			if (nx<k)
			{
				// Determine if workspace is large enough for blocked code.
				ldwork = n;
				iws = ldwork*nb;
				if (lwork<iws)
				{
					//Not enough workspace to use optimal nb:
					// reduce nb and determine the minimum value of nb.
					nb = lwork / ldwork;
					nbmin = std::max(2, ilaenv(2, "DGEQRF", " ", m, n, -1, -1));
				}
			}
		}
		int i, ib, iinfo, aind;
		if (nb>=nbmin && nb<k && nx<k)
		{
			// Use blocked code initially
			for (i=0; i<(k-nx); i+=nb)
			{
				ib = std::min(k-i, nb);
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

	/*! §dhseqr
	 *
	 * §dhseqr computes the eigenvalues of a Hessenberg matrix $H$ and, optionally, the matrices
	 * $T$ and $Z$ from the Schur decomposition $H=ZTZ^T$, where $T$ is an upper quasi-triangular
	 * matrix (the Schur form), and $Z$ is the orthogonal matrix of Schur vectors.\n
	 * Optionally $Z$ may be postmultiplied into an input orthogonal matrix $Q$ so that this
	 * routine can give the Schur factorization of a matrix $A$ which has been reduced to the
	 * Hessenberg form $H$ by the orthogonal matrix $Q$: $A=QHQ^T=(QZ)T(QZ)^T$.
	 * \param[in] job
	 *     ='E': compute eigenvalues only;\n
	 *     ='S': compute eigenvalues and the Schur form $T$.
	 *
	 * \param[in] compz
	 *     = 'N': no Schur vectors are computed;\n
	 *     = 'I': $Z$ is initialized to the unit matrix and the matrix $Z$ of Schur vectors of $H$
	 *            is returned;\n
	 *     = 'V': $Z$ must contain an orthogonal matrix $Q$ on entry, and the product $QZ$ is
	 *            returned.
	 *
	 * \param[in] n The order of the matrix $H$. $n\ge 0$.
	 * \param[in] ilo, ihi
	 *     It is assumed that §H is already upper triangular in rows and columns $0:\{ilo}-1$ and
	 *     $\{ihi}+1:\{n}-1$. §ilo and §ihi are normally set by a previous call to §dgebal, and
	 *     then passed to §dgehrd when the matrix output by §dgebal is reduced to Hessenberg form.
	 *     Otherwise §ilo and §ihi should be set to 0 and $\{n}-1$ respectively.\n If $\{n}>0$,
	 *     then $0\le\{ilo}\le\{ihi}<\{n}$.\n If $\{n}=0$, then $\{ilo}=0$ and $\{ihi}=-1$.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in,out] H
	 *     an array, dimension (§ldh,§n)\n
	 *     On entry, the upper Hessenberg matrix $H$.\n
	 *     On exit, if $\{info}=0$ and §job ='S', then §H contains the upper quasi-triangular
	 *     matrix $T$ from the Schur decomposition (the Schur form); 2 by 2 diagonal blocks
	 *     (corresponding to complex conjugate pairs of eigenvalues) are returned in standard form,
	 *     with $\{H}[i,i]=\{H}[i+1,i+1]$ and $\{H}[i+1,i]\{H}[i,i+1]<0$. If $\{info}=0$ and
	 *     §job ='E', the contents of §H are unspecified on exit. (The output value of §H when
	 *     $\{info}>0$ is given under the description of §info below.)\n
	 *     Unlike earlier versions of §dhseqr, this subroutine may explicitly set $\{H}[i,j]=0$ for
	 *     $i>j$ and $j=0,1,\ldots\{ilo}-1$ or $j=\{ihi}+1,\{ihi}+2,\ldots\{n}-1$.
	 *
	 * \param[in]  ldh    The leading dimension of the array §H. $\{ldh}\ge\max(1,\{n})$.
	 * \param[out] wr, wi
	 *     arrays, dimension (§n)\n
	 *     The real and imaginary parts, respectively, of the computed eigenvalues. If two
	 *     eigenvalues are computed as a complex conjugate pair, they are stored in consecutive
	 *     elements of §wr and §wi, say the $i$-th and $i+1$-th, with $\{wi}[i]>0$ and
	 *     $\{wi}[i+1]<0$. If §job ='S', the eigenvalues are stored in the same order as on the
	 *     diagonal of the Schur form returned in §H, with $\{wr}[i]=\{H}[i,i]$ and, if
	 *     $\{H}[i:i+1,i:i+1]$ is a 2 by 2 diagonal block,
	 *     $\{wi}[i]=\sqrt{-\{H}[i+1,i]\{H}[i,i+1]}$ and $\{wi}[i+1]=-\{wi}[i]$.
	 *
	 * \param[in,out] Z
	 *     an array, dimension (§ldz,§n)\n
	 *     If §compz ='N', §Z is not referenced.\n
	 *     If §compz ='I', on entry §Z need not be set and on exit, if $\{info}=0$, §Z contains the
	 *     orthogonal matrix $Z$ of the Schur vectors of $H$.\n
	 *     If §compz ='V', on entry §Z must contain an §n by §n matrix $Q$, which is assumed to be
	 *     equal to the unit matrix except for the submatrix $\{Z}[\{ilo}:\{ihi},\{ilo}:\{ihi}]$.
	 *     On exit, if $\{info}=0$, §Z contains $QZ$.\n
	 *     Normally $Q$ is the orthogonal matrix generated by §dorghr after the call to §dgehrd
	 *     which formed the Hessenberg matrix $H$. (The output value of §Z when $\{info}>0$ is
	 *     given under the description of §info below.)
	 *
	 * \param[in] ldz
	 *     The leading dimension of the array §Z. if §compz ='I' or §compz ='V', then
	 *     $\{ldz}\ge\max(1,\{n})$. Otherwize, $\{ldz}\ge 1$.
	 *
	 * \param[out] work
	 *     an array, dimension (§lwork)\n
	 *     On exit, if $\{info}=0$, $\{work}[0]$ returns an estimate of the optimal value for
	 *     §lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work. $\{lwork}\ge\max(1,\{n})$ is sufficient and delivers
	 *     very good and sometimes optimal performance. However, §lwork as large as $11\{n}$ may be
	 *     required for optimal performance. A workspace query is recommended to determine the
	 *     optimal workspace size.\n
	 *     If $\{lwork}=-1$, then §dhseqr does a workspace query. In this case, §dhseqr checks the
	 *     input parameters and estimates the optimal workspace size for the given values of §n,
	 *     §ilo and §ihi. The estimate is returned in $\{work}[0]$. No error message related to
	 *     §lwork is issued by §xerbla. Neither §H nor §Z are accessed.
	 *
	 * \param[out] info
	 *     - =0: successful exit
	 *     - <0: if $\{info}=-i$, the $i$-th argument had an illegal value
	 *     - >0: if $\{info}=i$, §dhseqr failed to compute all of the eigenvalues. Elements
	 *           $0:\{ilo}-1$ and $i:\{n}-1$ of §wr and §wi contain those eigenvalues which have
	 *           been successfully computed. (Failures are rare.)
	 *           - If §job ='E', then on exit, the remaining unconverged eigenvalues are the
	 *             eigenvalues of the upper Hessenberg matrix rows and columns §ilo through
	 *             $\{info}-1$ of the final, output value of §H.
	 *           - If §job ='S', then on exit\n
	 *                 $(\text{initial value of }\{H})U = U(\text{final value of }\{H})$&emsp;(*)\n
	 *             where $U$ is an orthogonal matrix. The final value of §H is upper Hessenberg and
	 *             quasi-triangular in rows and columns §info through §ihi.
	 *           - If §compz ='V', then on exit\n
	 *                 $(\text{final value of }\{Z}) = (\text{initial value of }\{Z})U$\n
	 *             where $U$ is the orthogonal matrix in (*) (regard less of the value of §job.)
	 *           - If §compz ='I', then on exit\n
	 *                 $(\text{final value of }Z) = U$\n
	 *             where $U$ is the orthogonal matrix in (*) (regardless of the value of §job.)
	 *           - If §compz ='N', then §Z is not accessed.
	 *
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Contributors:\n
	 *         Karen Braman and Ralph Byers, Department of Mathematics, University of Kansas, USA
	 *     \n\n
	 *     Further Details:\n
	 *         Default values supplied by
	 *             $\{ilaenv(ISPEC,"DHSEQR",job[0]+compz[0],n,ilo+1,ihi+1,lwork)}$.\n
	 *         It is suggested that these defaults be adjusted in order to attain best performance
	 *         in each particular computational environment.
	 *         - §ISPEC=12: The §dlahqr vs §dlaqr0 crossover point.
	 *                      Default: 75. (Must be at least 11.)
	 *         - §ISPEC=13: Recommended deflation window size. This depends on §ilo, §ihi and §NS.
	 *                      §NS is the number of simultaneous shifts returned by
	 *                      $\{ilaenv(ISPEC=15)}$.  (See §ISPEC=15 below.)\n
	 *                      The default for $(\{ihi}-\{ilo}+1)\le 500$ is §NS.\n
	 *                      The default for $(\{ihi}-\{ilo}+1)>500$ is $3\{NS}/2$.
	 *         - §ISPEC=14: Nibble crossover point. (See §iparmq for details.)
	 *                      Default: 14% of deflation window size.
	 *         - §ISPEC=15: Number of simultaneous shifts in a multishift QR iteration.
	 *                      If $\{ihi}-\{ilo}+1$ is ...\n
	 *                      $\begin{tabular}{rrl}
	 *                           greater than or equal to & but less than & the default is \\
	 *                                                  1 &            30 & \{NS} = 2(+)   \\
	 *                                                 30 &            60 & \{NS} = 4(+)   \\
	 *                                                 60 &           150 & \{NS} = 10(+)  \\
	 *                                                150 &           590 & \{NS} = **     \\
	 *                                                590 &          3000 & \{NS} = 64     \\
	 *                                               3000 &          6000 & \{NS} = 128    \\
	 *                                               6000 &      infinity & \{NS} = 256
	 *                       \end{tabular}$
	 *                      - (+) By default some or all matrices of this order are passed to the
	 *                            implicit double shift routine §dlahqr and this parameter is
	 *                            ignored. See §ISPEC=12 above and comments in §iparmq for details.
	 *                      - (**) The asterisks (**) indicate an ad-hoc function of §n increasing
	 *                             from 10 to 64.
	 *                      .
	 *         - §ISPEC=16: Select structured matrix multiply. If the number of simultaneous shifts
	 *                      (specified by §ISPEC=15) is less than 14, then the default for
	 *                      §ISPEC=16 is 0. Otherwise the default for §ISPEC=16 is 2.
	 *
	 *     \n\n
	 *     References:\n
	 *         K. Braman, R. Byers and R. Mathias, The Multi-Shift QR Algorithm Part I: Maintaining
	 *         Well Focused Shifts, and Level 3 Performance, SIAM Journal of Matrix Analysis,
	 *         volume 23, pages 929--947, 2002.\n
	 *         \n
	 *         K. Braman, R. Byers and R. Mathias, The Multi-Shift QR Algorithm Part II: Aggressive
	 *         Early Deflation, SIAM Journal of Matrix Analysis, volume 23, pages 948--973, 2002.*/
	static void dhseqr(char const* const job, char const* const compz, int const n, int const ilo,
	                   int const ihi, real* const H, int const ldh, real* const wr, real* const wi,
	                   real* const Z, int const ldz, real* const work, int const lwork, int& info)
	                   /*const*/
	{
		/* Matrices of order NTINY or smaller must be processed by §dlahqr because of insufficient
		 * subdiagonal scratch space. (This is a hard limit.)                                    */
		int const NTINY = 11;
		/* NL allocates some local workspace to help small matrices through a rare §dlahqr failure.
		 * NL > NTINY = 11 is required and NL <= nmin = ilaenv(ISPEC=12,...) is recommended.
		 * (The default value of nmin is 75.)  Using NL = 49 allows up to six simultaneous shifts
		 * and a 16 by 16 deflation window.                                                      */
		int const NL = 49;
		// Decode and check the input parameters.
		bool wantt = (std::toupper(job[0])=='S');
		bool initz = (std::toupper(compz[0])=='I');
		bool wantz = (initz || (std::toupper(compz[0])=='V'));
		work[0] = real(std::max(1, n));
		bool lquery = (lwork==-1);
		info = 0;
		if (!(std::toupper(job[0])=='E') && !wantt)
		{
			info = -1;
		}
		else if (!(std::toupper(compz[0])=='N') && !wantz)
		{
			info = -2;
		}
		else if (n<0)
		{
			info = -3;
		}
		else if (ilo<0 || ilo>std::max(0, n-1))
		{
			info = -4;
		}
		else if (ihi<std::min(ilo, n-1) || ihi>=n)
		{
			info = -5;
		}
		else if (ldh<std::max(1, n))
		{
			info = -7;
		}
		else if (ldz<1 || (wantz && ldz<std::max(1, n)))
		{
			info = -11;
		}
		else if (lwork<std::max(1, n) && !lquery)
		{
			info = -13;
		}
		if (info!=0)
		{
			// Quick return in case of invalid argument.
			xerbla("DHSEQR", -info);
			return;
		}
		else if (n==0)
		{
			// Quick return in case n = 0; nothing to do.
			return;
		}
		else if (lquery)
		{
			// Quick return in case of a workspace query
			dlaqr0(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, ilo, ihi, Z, ldz, work, lwork, info);
			// Ensure reported workspace size is backward-compatible with previous LAPACK versions.
			work[0] = std::max(real(std::max(1, n)), work[0]);
			return;
		}
		else
		{
			// copy eigenvalues isolated by dgebal
			int i;
			for (i=0; i<ilo; i++)
			{
				wr[i] = H[i+ldh*i];
				wi[i] = ZERO;
			}
			for (i=ihi+1; i<n; i++)
			{
				wr[i] = H[i+ldh*i];
				wi[i] = ZERO;
			}
			// Initialize Z, if requested
			if (initz)
			{
				dlaset("A", n, n, ZERO, ONE, Z, ldz);
			}
			// Quick return if possible
			if (ilo==ihi)
			{
				wr[ilo] = H[ilo+ldh*ilo];
				wi[ilo] = ZERO;
				return;
			}
			// dlahqr/dlaqr0 crossover point
			char concat[3];
			concat[0] = job[0];
			concat[1] = compz[0];
			concat[2] = '\0';
			int nmin = std::max(NTINY, ilaenv(12, "DHSEQR", concat, n, ilo+1, ihi+1, lwork));
			// dlaqr0 for big matrices; dlahqr for small ones
			if (n>nmin)
			{
				dlaqr0(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, ilo, ihi, Z, ldz, work, lwork,
				       info);
			}
			else
			{
				// Small matrix
				dlahqr(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, ilo, ihi, Z, ldz, info);
				if (info>0)
				{
					// A rare dlahqr failure!  dlaqr0 sometimes succeeds when dlahqr fails.
					int kbot = info - 1;
					if (n>=NL)
					{
						// Larger matrices have enough subdiagonal scratch space to call dlaqr0
						// directly.
						dlaqr0(wantt, wantz, n, ilo, kbot, H, ldh, wr, wi, ilo, ihi, Z, ldz, work,
						       lwork, info);
					}
					else
					{
						// Tiny matrices don't have enough subdiagonal scratch space to benefit
						// from dlaqr0. Hence, tiny matrices must be copied into a larger array
						// before calling dlaqr0.
						real Hl[NL*NL];
						real workl[NL];
						dlacpy("A", n, n, H, ldh, Hl, NL);
						Hl[n+NL*(n-1)] = ZERO;
						dlaset("A", NL, NL-n, ZERO, ZERO, &Hl[NL*n], NL);
						dlaqr0(wantt, wantz, NL, ilo, kbot, Hl, NL, wr, wi, ilo, ihi, Z, ldz,
						       workl, NL, info);
						if (wantt || info!=0)
						{
							dlacpy("A", n, n, Hl, NL, H, ldh);
						}
					}
				}
			}
			// Clear out the trash, if necessary.
			if ((wantt || info!=0) && n>2)
			{
				dlaset("L", n-2, n-2, ZERO, ZERO, &H[2], ldh);
			}
			// Ensure reported workspace size is backward-compatible with previous LAPACK versions.
			work[0] = std::max(real(std::max(1, n)), work[0]);
		}
	}

	/* disnan replaced by std::isnan */

	/*! §dlabad
	 *
	 * §dlabad takes as input the values computed by §dlamch for underflow and overflow, and
	 * returns the square root of each of these values if the log of §large is sufficiently large.
	 * This subroutine is intended to identify machines with a large exponent range, such as the
	 * Crays, and redefine the underflow and overflow limits to be the square roots of the values
	 * computed by §dlamch. This subroutine is needed because dlamch does not compensate for poor
	 * arithmetic in the upper half of the exponent range, as is found on a Cray.
	 * \param[in,out] small
	 *     On entry, the underflow threshold as computed by §dlamch.\n
	 *     On exit, if $\log_{10}(\{large})$ is sufficiently large, the square root of §small,
	 *              otherwise unchanged.
	 *
	 * \param[in,out] large
	 *     On entry, the overflow threshold as computed by §dlamch.\n
	 *     On exit, if $\log_{10}(\{large})$ is sufficiently large, the square root of large,
	 *              otherwise unchanged.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlabad(real& small, real& large) /*const*/
	{
		// If it looks like we're on a Cray, take the square root of small and large to avoid
		// overflow and underflow problems.
		if (std::log10(large)>real(2000.0))
		{
			small = std::sqrt(small);
			large = std::sqrt(large);
		}
	}

	/*! §dlabrd reduces the first nb rows and columns of a general matrix to a bidiagonal form.
	 *
	 * §dlabrd reduces the first §nb rows and columns of a real general §m by §n matrix $A$ to
	 * upper or lower bidiagonal form by an orthogonal transformation $Q^T A P$, and returns the
	 * matrices $X$ and $Y$ which are needed to apply the transformation to the unreduced part of
	 * $A$.\n
	 * If $\{m}\ge\{n}$, $A$ is reduced to upper bidiagonal form;\n if $\{m}<\{n}$, to lower
	 * bidiagonal form.\n
	 * This is an auxiliary routine called by §dgebrd
	 * \param[in]     m  The number of rows in the matrix $A$.
	 * \param[in]     n  The number of columns in the matrix $A$.
	 * \param[in]     nb The number of leading rows and columns of $A$ to be reduced.
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the §m by §n general matrix to be reduced.\n
	 *     On exit, the first §nb rows and columns of the matrix are overwritten; the rest of the
	 *     array is unchanged.
	 *     \li If $\{m}\ge\{n}$, elements on and below the diagonal in the first §nb columns, with
	 *         the array §tauq, represent the orthogonal matrix $Q$ as a product of elementary
	 *         reflectors; and elements above the diagonal in the first §nb rows, with the array
	 *         §taup, represent the orthogonal matrix $P$ as a product of elementary reflectors.
	 *     \li If $\{m}<\{n}$, elements below the diagonal in the first §nb columns, with the array
	 *         §tauq, represent the orthogonal matrix $Q$ as a product of elementary reflectors,
	 *         and elements on and above the diagonal in the first §nb rows, with the array §taup,
	 *         represent the orthogonal matrix $P$ as a product of elementary reflectors.
	 *
	 *     See Remark.
	 *
	 * \param[in]  lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[out] d
	 *     an array, dimension (§nb)\n
	 *     The diagonal elements of the first §nb rows and columns of the reduced matrix.
	 *     $\{d}[i] = A[i,i]$.
	 *
	 * \param[out] e
	 *     an array, dimension (§nb)\n
	 *     The off-diagonal elements of the first §nb rows and columns of the reduced matrix.
	 *
	 * \param[out] tauq
	 *     an  array, dimension (§nb)\n
	 *     The scalar factors of the elementary reflectors which represent the orthogonal matrix
	 *     $Q$. See Remark.
	 *
	 * \param[out] taup
	 *     an array, dimension (§nb)\n
	 *     The scalar factors of the elementary reflectors which represent the orthogonal matrix
	 *     $P$. See Remark.
	 *
	 * \param[out] X
	 *     an array, dimension (§ldx,§nb)\n
	 *     The §m by §nb matrix $X$ required to update the unreduced part of $A$.
	 *
	 * \param[in]  ldx The leading dimension of the array §X. $\{ldx}\ge\max(1,\{m})$.
	 * \param[out] Y
	 *     an array, dimension (§ldy,§nb)\n
	 *     The §n by §nb matrix $Y$ required to update the unreduced part of $A$.
	 *
	 * \param[in] ldy The leading dimension of the array §Y. $\{ldy}\ge\max(1,\{n})$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017
	 * \remark
	 *     The matrices $Q$ and $P$ are represented as products of elementary reflectors:\n
	 *         $Q = H(0) H(1) \ldots H(nb-1)$ and $P = G(0) G(1) \ldots G(nb-1)$\n
	 *     Each $H(i)$ and $G(i)$ has the form:\n
	 *         $H(i) = I - \tau_q v v^T$  and $G(i) = I - \tau_p u u^T$\n
	 *     where $\tau_q$ and $\tau_p$ are real scalars, and $v$ and $u$ are real vectors.
	 *     \li If $\{m}\ge\{n}$,\n $v[0:i-1] = 0$, $v[i] = 1$, and $v[i:\{m}-1]$ is stored on exit
	 *         in $\{A}[i:\{m}-1,i]$;\n $u[0:i] = 0$, $u[i+1] = 1$, and $u[i+1:\{n}-1]$ is stored
	 *         on exit in $\{A}[i,i+1:\{n}-1]$;\n $\tau_q$ is stored in $\{tauq}[i]$ and $\tau_p$
	 *         in $\{taup}[i]$.
	 *     \li If $\{m}<\{n}$,\n $v[0:i] = 0$, $v[i+1] = 1$, and $v[i+1:\{m}-1]$ is stored on exit
	 *         in $\{A}[i+2:\{m}-1,i]$;\n $u[0:i-1] = 0$, $u[i] = 1$, and $u[i:\{n}-1]$ is stored
	 *         on exit in $\{A}[i,i+1:\{n}-1]$;\n $\tau_q$ is stored in $\{tauq}[i]$ and $\tau_p$
	 *         in $\{taup}[i]$.
	 *
	 *     The elements of the vectors $v$ and $u$ together form the §m by §nb matrix $V$ and the
	 *     §nb by §n matrix $U^T$ which are needed, with $X$ and $Y$, to apply the transformation
	 *     to the unreduced part of the matrix, using a block update of the form:
	 *         $A = A - V Y^T - X U^T$.\n
	 *     The contents of §A on exit are illustrated by the following examples with $\{nb} = 2$:\n
	 *     $\{m} = 6$ and $\{n} = 5$ ($\{m}>\{n}$):\n
	 *         $\b{bm} 1  &  1  & u_1 & u_1 & u_1 \\
	 *                v_1 &  1  &  1  & u_2 & u_2 \\
	 *                v_1 & v_2 &  a  &  a  &  a  \\
	 *                v_1 & v_2 &  a  &  a  &  a  \\
	 *                v_1 & v_2 &  a  &  a  &  a  \\
	 *                v_1 & v_2 &  a  &  a  &  a  \e{bm}$\n
	 *     $\{m} = 5$ and $\{n} = 6$ ($\{m}<\{n}$):\n
	 *         $\b{bm} 1  & u_1 & u_1 & u_1 & u_1 & u_1 \\
	 *                 1  &  1  & u_2 & u_2 & u_2 & u_2 \\
	 *                v_1 &  1  &  a  &  a  &  a  &  a  \\
	 *                v_1 & v_2 &  a  &  a  &  a  &  a  \\
	 *                v_1 & v_2 &  a  &  a  &  a  &  a  \e{bm}$\n
	 *     where $a$ denotes an element of the original matrix which is unchanged, $v_i$ denotes an
	 *     element of the vector defining $H(i)$, and $u_i$ an element of the vector defining
	 *     $G(i)$.                                                                               */
	static void dlabrd(int const m, int const n, int const nb, real* const A, int const lda,
	                   real* const d, real* const e, real* const tauq, real* const taup,
	                   real* const X, int const ldx, real* const Y, int const ldy) /*const*/
	{
		// Quick return if possible
		if (m<=0 || n<=0)
		{
			return;
		}
		int i, ildai, ip, ipldxi, ipldyi, ldai, ldaip, ldxi, ldyi, mmip, nmip;
		int nm = n - 1;
		int mm = m - 1;
		if (m>=n)
		{
			int ildaip, mmi;
			// Reduce to upper bidiagonal form
			for (i=0; i<nb; i++)
			{
				ldai   = lda * i;
				ildai  =  i  + ldai;
				ip     =  i  + 1;
				ldyi   = ldy * i;
				ipldyi = ip  + ldyi;
				mmi    =  m  - i;
				// Update A[i:m-1,i]
				Blas<real>::dgemv("No transpose", mmi, i, -ONE, &A[i], lda, &Y[i], ldy, ONE,
				                  &A[ildai], 1);
				Blas<real>::dgemv("No transpose", mmi, i, -ONE, &X[i], ldx, &A[ldai], 1, ONE,
				                  &A[ildai], 1);
				// Generate reflection Q(i) to annihilate A[i+1:m-1,i]
				dlarfg(mmi, A[ildai], &A[std::min(ip, mm)+ldai], 1, tauq[i]);
				d[i] = A[ildai];
				if (i<nm)
				{
					ldaip  = lda * ip;
					ildaip =  i  + ldaip;
					ldxi   = ldx * i;
					ipldxi = ip  + ldxi;
					mmip   = mm  - i;
					nmip   = nm  - i;
					A[ildai] = ONE;
					// Compute Y[i+1:n-1,i]
					Blas<real>::dgemv("Transpose", mmi, nmip, ONE, &A[ildaip], lda, &A[ildai], 1,
					                  ZERO, &Y[ipldyi], 1);
					Blas<real>::dgemv("Transpose", mmi, i, ONE, &A[i], lda, &A[ildai], 1, ZERO,
					                  &Y[ldyi], 1);
					Blas<real>::dgemv("No transpose", nmip, i, -ONE, &Y[ip], ldy, &Y[ldyi], 1, ONE,
					                  &Y[ipldyi], 1);
					Blas<real>::dgemv("Transpose", mmi, i, ONE, &X[i], ldx, &A[ildai], 1, ZERO,
					                  &Y[ldyi], 1);
					Blas<real>::dgemv("Transpose", i, nmip, -ONE, &A[ldaip], lda, &Y[ldyi], 1, ONE,
					                  &Y[ipldyi], 1);
					Blas<real>::dscal(nmip, tauq[i], &Y[ipldyi], 1);
					// Update A[i,i+1:n-1]
					Blas<real>::dgemv("No transpose", nmip, ip, -ONE, &Y[ip], ldy, &A[i], lda, ONE,
					                  &A[ildaip], lda);
					Blas<real>::dgemv("Transpose", i, nmip, -ONE, &A[ldaip], lda, &X[i], ldx, ONE,
					                  &A[ildaip], lda);
					// Generate reflection P(i) to annihilate A[i,i+2:n-1]
					dlarfg(nmip, A[ildaip], &A[i+lda*std::min(i+2, nm)], lda, taup[i]);
					e[i] = A[ildaip];
					A[ildaip] = ONE;
					// Compute X[i+1:m-1,i]
					Blas<real>::dgemv("No transpose", mmip, nmip, ONE, &A[1+ildaip], lda,
					                  &A[ildaip], lda, ZERO, &X[ipldxi], 1);
					Blas<real>::dgemv("Transpose", nmip, ip, ONE, &Y[ip], ldy, &A[ildaip], lda,
					                  ZERO, &X[ldxi], 1);
					Blas<real>::dgemv("No transpose", mmip, ip, -ONE, &A[ip], lda, &X[ldxi], 1,
					                  ONE, &X[ipldxi], 1);
					Blas<real>::dgemv("No transpose", i, nmip, ONE, &A[ldaip], lda, &A[ildaip],
					                  lda, ZERO, &X[ldxi], 1);
					Blas<real>::dgemv("No transpose", mmip, i, -ONE, &X[ip], ldx, &X[ldxi], 1, ONE,
					                  &X[ipldxi], 1);
					Blas<real>::dscal(mmip, taup[i], &X[ipldxi], 1);
				}
			}
		}
		else
		{
			int ipldai, nmi;
			// Reduce to lower bidiagonal form
			for (i=0; i<nb; i++)
			{
				ldai  = lda * i;
				ildai =  i  + ldai;
				ip    =  i  + 1;
				nmi   =  n  - i;
				// Update A[i,i:n-1]
				Blas<real>::dgemv("No transpose", nmi, i, -ONE, &Y[i], ldy, &A[i], lda, ONE,
				                  &A[ildai], lda);
				Blas<real>::dgemv("Transpose", i, nmi, -ONE, &A[ldai], lda, &X[i], ldx, ONE,
				                  &A[ildai], lda);
				// Generate reflection P(i) to annihilate A[i,i+1:n-1]
				dlarfg(nmi, A[ildai], &A[i+lda*std::min(ip, nm)], lda, taup[i]);
				d[i] = A[ildai];
				if (i<mm)
				{
					ipldai = ip  + ldai;
					ldaip  = lda * ip;
					ldxi   = ldx * i;
					ldyi   = ldy * i;
					ipldxi = ip  + ldxi;
					ipldyi = ip  + ldyi;
					mmip   = mm  - i;
					nmip   = nm  - i;
					A[ildai] = ONE;
					// Compute X[i+1:m-1,i]
					Blas<real>::dgemv("No transpose", mmip, nmi, ONE, &A[ipldai], lda, &A[ildai],
					                  lda, ZERO, &X[ipldxi], 1);
					Blas<real>::dgemv("Transpose", nmi, i, ONE, &Y[i], ldy, &A[ildai], lda, ZERO,
					                  &X[ldxi], 1);
					Blas<real>::dgemv("No transpose", mmip, i, -ONE, &A[ip], lda, &X[ldxi], 1, ONE,
					                  &X[ipldxi], 1);
					Blas<real>::dgemv("No transpose", i, nmi, ONE, &A[ldai], lda, &A[ildai], lda,
					                  ZERO, &X[ldxi], 1);
					Blas<real>::dgemv("No transpose", mmip, i, -ONE, &X[ip], ldx, &X[ldxi], 1, ONE,
					                  &X[ipldxi], 1);
					Blas<real>::dscal(mmip, taup[i], &X[ipldxi], 1);
					// Update A[i+1:m-1,i]
					Blas<real>::dgemv("No transpose", mmip, i, -ONE, &A[ip], lda, &Y[i], ldy, ONE,
					                  &A[ipldai], 1);
					Blas<real>::dgemv("No transpose", mmip, ip, -ONE, &X[ip], ldx, &A[ldai], 1,
					                  ONE, &A[ipldai], 1);
					// Generate reflection Q(i) to annihilate A[i+2:m-1,i]
					dlarfg(mmip, A[ipldai], &A[std::min(i+2, mm)+ldai], 1, tauq[i]);
					e[i] = A[ipldai];
					A[ipldai] = ONE;
					// Compute Y[i+1:n-1,i]
					Blas<real>::dgemv("Transpose", mmip, nmip, ONE, &A[ip+ldaip], lda, &A[ipldai],
					                  1, ZERO, &Y[ipldyi], 1);
					Blas<real>::dgemv("Transpose", mmip, i, ONE, &A[ip], lda, &A[ipldai], 1, ZERO,
					                  &Y[ldyi], 1);
					Blas<real>::dgemv("No transpose", nmip, i, -ONE, &Y[ip], ldy, &Y[ldyi], 1, ONE,
					                  &Y[ipldyi], 1);
					Blas<real>::dgemv("Transpose", mmip, ip, ONE, &X[ip], ldx, &A[ipldai], 1, ZERO,
					                  &Y[ldyi], 1);
					Blas<real>::dgemv("Transpose", ip, nmip, -ONE, &A[ldaip], lda, &Y[ldyi], 1,
					                  ONE, &Y[ipldyi], 1);
					Blas<real>::dscal(nmip, tauq[i], &Y[ipldyi], 1);
				}
			}
		}
	}

	/*! §dlacpy copies all or part of one two-dimensional array to another.
	 *
	 * §dlacpy copies all or part of a two-dimensional matrix $A$ to another matrix $B$.
	 * \param[in] uplo
	 *     §uplo[0] specifies the part of the matrix $A$ to be copied to $B$.\n
	 *     'U' Upper triangular part\n
	 *     'L' Lower triangular part\n
	 *     Otherwise: All of the matrix $A$
	 *
	 * \param[in] m The number of rows of the matrix $A$. $\{m}\ge 0$.
	 * \param[in] n The number of columns of the matrix $A$. $\{n}\ge 0$.
	 * \param[in] A
	 *     an array, dimension (§lda,§n)\n
	 *     The §m by §n matrix $A$.\n
	 *     If §uplo = 'U', only the upper triangle or trapezoid is accessed;\n
	 *     if §uplo = 'L', only the lower triangle or trapezoid is accessed.
	 *
	 * \param[in] lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[in] B
	 *     an array, dimension (§ldb,§n)\n
	 *     On exit, §B = §A in the locations specified by uplo.
	 *
	 * \param[in] ldb The leading dimension of the array §B. $\{ldb}\ge\max(1,\{m})$.
	 *              otherwise unchanged.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlacpy(char const* const uplo, int const m, int const n, real const* const A,
	                   int const lda, real* const B, int const ldb) /*const*/
	{
		int i, j, ldaj, ldbj;
		if (std::toupper(uplo[0])=='U')
		{
			for (j=0; j<n; j++)
			{
				ldaj = lda*j;
				ldbj = ldb*j;
				for (i=0; i<=j && i<m; i++)
				{
					B[i+ldbj] = A[i+ldaj];
				}
			}
		}
		else if (std::toupper(uplo[0])=='L')
		{
			for (j=0; j<n; j++)
			{
				ldaj = lda*j;
				ldbj = ldb*j;
				for (i=j; i<m; i++)
				{
					B[i+ldbj] = A[i+ldaj];
				}
			}
		}
		else
		{
			for (j=0; j<n; j++)
			{
				ldaj = lda*j;
				ldbj = ldb*j;
				for (i=0; i<m; i++)
				{
					B[i+ldbj] = A[i+ldaj];
				}
			}
		}
	}

	/*! §dladiv performs complex division in real arithmetic, avoiding unnecessary overflow.
	 *
	 * §dladiv performs complex division in  real arithmetic\n
	 *     $p+iq = \frac{a+ib}{c+id}$\n
	 * The algorithm is due to Michael Baudin and Robert L. Smith and can be found in the paper
	 * "A Robust Complex Division in Scilab"
	 * \param[in]  a, b, c, d The scalars $a$, $b$, $c$, and $d$ in the above expression.
	 * \param[out] p, q The scalars $p$ and $q$ in the above expression.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date January 2013                                                                        */
	static void dladiv(real a, real b, real c, real d, real& p, real& q) /*const*/
	{
		const real BS = TWO;
		real ab = std::max(std::fabs(a), std::fabs(b));
		real cd = std::max(std::fabs(c), std::fabs(d));
		real s = ONE;
		real ov  = dlamch("Overflow threshold");
		real un  = dlamch("Safe minimum");
		real eps = dlamch("Epsilon");
		real be = BS / (eps*eps);
		if (ab >= HALF*ov)
		{
			a *= HALF;
			b *= HALF;
			s *= TWO;
		}
		if (cd >= HALF*ov)
		{
			c *= HALF;
			d *= HALF;
			s *= HALF;
		}
		if (ab <= un*BS/eps)
		{
			a *= be;
			b *= be;
			s /= be;
		}
		if (cd <= un*BS/eps)
		{
			c *= be;
			d *= be;
			s *= be;
		}
		if (std::fabs(d)<=std::fabs(c))
		{
			dladiv1(a, b, c, d, p, q);
		}
		else
		{
			dladiv1(b, a, d, c, p, q);
			q = -q;
		}
		p *= s;
		q *= s;
	}

	/*! §dladiv1
	 *
	 * Auxiliary routine to §dladiv                                                              */
	static void dladiv1(real& a, real const b, real const c, real const d, real& p, real& q)
	                    /*const*/
	{
		real r = d / c;
		real t = ONE / (c + d*r);
		p = dladiv2(a, b, c, d, r, t);
		a = -a;
		q = dladiv2(b, a, c, d, r, t);
	}

	/*! §dladiv2
	 *
	 * Auxiliary routine to §dladiv1.                                                            */
	static real dladiv2(real const a, real const b, real const c, real const d, real const r,
	                    real const t) /*const*/
	{
		if (r!=ZERO)
		{
			real br = b * r;
			if (br!=ZERO)
			{
				return (a + br) * t;
			}
			else
			{
				return a * t + (b*t) * r;
			}
		}
		else
		{
			return (a + d*(b/c)) * t;
		}
	}

	/*! §dlae2 computes the eigenvalues of a 2 by 2 symmetric matrix.
	 *
	 * §dlae2  computes the eigenvalues of a 2 by 2 symmetric matrix\n
	 *     $\b{bm} \{a} & \{b} \\
	 *             \{b} & \{c} \e{bm}$.\n
	 * On return, §rt1 is the eigenvalue of larger absolute value, and §rt2 is the eigenvalue of
	 * smaller absolute value.
	 * \param[in]  a   The [0,0] element of the 2 by 2 matrix.
	 * \param[in]  b   The [0,1] and [1,0] elements of the 2 by 2 matrix.
	 * \param[in]  c   The [1,1] element of the 2 by 2 matrix.
	 * \param[out] rt1 The eigenvalue of larger absolute value.
	 * \param[out] rt2 The eigenvalue of smaller absolute value.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     §rt1 is accurate to a few ulps barring over/underflow.\n
	 *     §rt2 may be inaccurate if there is massive cancellation in the determinant
	 *     $\{a}\{c}-\{b}\{b}$; higher precision or correctly rounded or correctly truncated
	 *     arithmetic would be needed to compute §rt2 accurately in all cases.\n
	 *     Overflow is possible only if §rt1 is within a factor of 5 of overflow. Underflow is
	 *     harmless if the input data is 0 or exceeds $\{underflow\_threshold}/\{macheps}.$      */
	static void dlae2(real const a, real const b, real const c, real& rt1, real& rt2) /*const*/
	{
		real acmn, acmx, rt;
		{
			// Compute the eigenvalues
			real adf = std::fabs(a-c);
			real ab  = std::fabs(b+b);
			if (std::fabs(a)>std::fabs(c))
			{
				acmx = a;
				acmn = c;
			}
			else
			{
				acmx = c;
				acmn = a;
			}
			{
				real temp;
				if (adf>ab)
				{
					temp = ab / adf;
					rt = adf * std::sqrt(ONE+temp*temp);
				}
				else if (adf<ab)
				{
					temp = adf / ab;
					rt = ab  * std::sqrt(ONE+temp*temp);
				}
				else
				{
					// Includes case ab=adf=0
					rt = ab * std::sqrt(TWO);
				}
			}
		}
		real sm = a + c;
		if (sm<ZERO)
		{
			rt1 = HALF*(sm-rt);
			// Order of execution important.
			// To get fully accurate smaller eigenvalue, the next line needs to be executed in
			// higher precision.
			rt2 = (acmx/rt1)*acmn - (b/rt1)*b;
		}
		else if (sm>ZERO)
		{
			rt1 = HALF * (sm+rt);
			// Order of execution important.
			// To get fully accurate smaller eigenvalue, the next line needs to be executed in
			// higher precision.
			rt2 = (acmx/rt1)*acmn - (b/rt1)*b;
		}
		else
		{
			// Includes case rt1 = rt2 = 0
			rt1 =  HALF * rt;
			rt2 = -HALF * rt;
		}
}

	/*! §dlaebz computes the number of eigenvalues of a real symmetric tridiagonal matrix which are
	 *  less than or equal to a given value, and performs other tasks required by the routine
	 *  §dstebz.
	 *
	 * §dlaebz contains the iteration loops which compute and use the function $N(w)$, which is the
	 * count of eigenvalues of a symmetric tridiagonal matrix $T$ less than or equal to its
	 * argument $w$. It performs a choice of two types of loops:
	 * \li ijob=1, followed by
	 * \li ijob=2: It takes as input a list of intervals and returns a list of sufficiently small
	 *             intervals whose union contains the same eigenvalues as the union of the original
	 *             intervals. The input intervals are $]\{Ab}[j,0],\{Ab}[j,1]]$,
	 *             $j=0,\ldots,\{minp}-1$.
	 *             The output interval $]\{Ab}[j,0],\{Ab}[j,1]]$ will contain eigenvalues
	 *             $\{Nab}[j,0]+1,\ldots,\{Nab}[j,1]$, where $0 \le j < \{mout}$.
	 * \li ijob=3: It performs a binary search in each input interval $]\{Ab}[j,0],\{Ab}[j,1]]$ for
	 *             a point $w(j)$ such that $N(w(j))=\{nval}[j]$, and uses $c[j]$ as the starting
	 *             point of the search. If such a $w(j)$ is found, then on output
	 *             $\{Ab}[j,0]=\{Ab}[j,1]=w$. If no such $w(j)$ is found, then on output
	 *             $]\{Ab}[j,0],\{Ab}[j,1]]$ will be a small interval containing the point where
	 *             $N(w)$ jumps through $\{nval}[j]$, unless that point lies outside the initial
	 *             interval.
	 *
	 * Note that the intervals are in all cases half-open intervals, i.e., of the form $]a,b]$,
	 * which includes $b$ but not $a$.\n
	 * To avoid underflow, the matrix should be scaled so that its largest element is no greater
	 * than $\sqrt{\{overflow}\sqrt{\{underflow}}}$ in absolute value. To assure the
	 * most accurate computation of small eigenvalues, the matrix should be scaled to be not much
	 * smaller than that, either.\n
	 * See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal Matrix", Report CS41, Computer
	 * Science Dept., Stanford University, July 21, 1966\n
	 * Note: the arguments are, in general, *not* checked for unreasonable values.
	 * \param[in] ijob
	 *     Specifies what is to be done:\n
	 *         =1: Compute §Nab for the initial intervals.\n
	 *         =2: Perform bisection iteration to find eigenvalues of $T$.\n
	 *         =3: Perform bisection iteration to invert $N(w)$, i.e., to find a point which has a
	 *             specified number of eigenvalues of $T$ to its left.\n
	 *         Other values will cause §dlaebz to return with $\{info}=-1$.
	 *
	 * \param[in] nitmax
	 *     The maximum number of "levels" of bisection to be performed, i.e., an interval of width
	 *     $W$ will not be made smaller than $2^{-\{nitmax}}W$. If not all intervals have converged
	 *     after §nitmax iterations, then §info is set to the number of non-converged intervals.
	 *
	 * \param[in] n    The dimension §n of the tridiagonal matrix $T$. It must be at least 1.
	 * \param[in] mmax
	 *     The maximum number of intervals. If more than §mmax intervals are generated, then
	 *     §dlaebz will quit with $\{info}=\{mmax}+1$.
	 *
	 * \param[in] minp  The initial number of intervals. It may not be greater than §mmax.
	 * \param[in] nbmin
	 *     The smallest number of intervals that should be processed using a vector loop. If zero,
	 *     then only the scalar loop will be used.
	 *
	 * \param[in] abstol
	 *     The minimum (absolute) width of an interval. When an interval is narrower than §abstol,
	 *     or than §reltol times the larger (in magnitude) endpoint, then it is considered to be
	 *     sufficiently small, i.e., converged. This must be at least zero.
	 *
	 * \param[in] reltol
	 *     The minimum relative width of an interval.  When an interval is narrower than §abstol,
	 *     or than §reltol times the larger (in magnitude) endpoint, then it is considered to be
	 *     sufficiently small, i.e., converged. Note: this should always be at least
	 *     $\{radix}*\{machine epsilon}$.
	 *
	 * \param[in] pivmin
	 *     The minimum absolute value of a "pivot" in the Sturm sequence loop.
	 *     This must be at least $\max_j{\left|\{e}[j]^2\right|}\{safemin}$ and at least §safemin,
	 *     where §safemin is at least the smallest number that can divide one without overflow.
	 *
	 * \param[in] d an array, dimension (§n)\n The diagonal elements of the tridiagonal matrix $T$.
	 * \param[in] e
	 *     an array, dimension (§n)\n
	 *     The offdiagonal elements of the tridiagonal matrix $T$ in positions 0 through $\{n}-2$.
	 *     $\{e}[\{n}-1]$ is arbitrary.
	 *
	 * \param[in] e2
	 *     an array, dimension (§n)\n
	 *     The squares of the offdiagonal elements of the tridiagonal matrix $T$.
	 *     $\{e2}[\{n}-1]$ is ignored.
	 *
	 * \param[in,out] nval
	 *     an integer array, dimension (§minp)\n
	 *     If $\{ijob}=1$ or $2$, not referenced.\n
	 *     If $\{ijob}=3$, the desired values of $N(w)$. The elements of §nval will be reordered to
	 *     correspond with the intervals in §Ab. Thus, $\{nval}[j]$ on output will not, in general
	 *     be the same as $\{nval}[j]$ on input, but it will correspond with the interval
	 *     $]\{Ab}[j,0],\{Ab}[j,1]]$ on output.
	 *
	 * \param[in,out] Ab
	 *     an array, dimension (§mmax,2)\n
	 *     The endpoints of the intervals. $\{Ab}[j,0]$ is $a[j]$, the left endpoint of the $j$-th
	 *     interval, and $\{Ab}[j,1]$ is $b[j]$, the right endpoint of the $j$-th interval. The
	 *     input intervals will, in general, be modified, split, and reordered by the calculation.
	 *
	 * \param[in,out] c
	 *     an array, dimension (§mmax)\n
	 *     If $\{ijob}=1$, ignored.\n
	 *     If $\{ijob}=2$, workspace.\n
	 *     If $\{ijob}=3$, then on input $c[j]$ should be initialized to the first search point in
	 *                     the binary search.
	 *
	 * \param[out] mout
	 *     If $\{ijob}=1$, the number of eigenvalues in the intervals.\n
	 *     If $\{ijob}=2$ or $3$, the number of intervals output.\n
	 *     If $\{ijob}=3$, §mout will equal §minp.
	 *
	 * \param[in,out] Nab
	 *     an integer array, dimension (§mmax,2)\n
	 *     If $\{ijob}=1$, then on output $\{Nab}[i,j]$ will be set to $N(\{Ab}[i,j])$.\n\n
	 *     If $\{ijob}=2$, then on input, $\{Nab}[i,j]$ should be set. It must satisfy the
	 *                     condition:\n
	 *                         $N(\{Ab}[i,0]) \le \{Nab}[i,0] \le \{Nab}[i,1] \le N(\{Ab}[i,1])$,\n
	 *                     which means that in interval $i$ only eigenvalues
	 *                     $\{Nab}[i,0]+1,\ldots,\{Nab}[i,1]$ will be considered. Usually,
	 *                     $\{Nab}[i,j]=N(\{Ab}[i,j])$, from a previous call to §dlaebz with
	 *                     $\{ijob}=1$.\n
	 *                     On output, $\{Nab}[i,j]$ will contain
	 *                     $\max\left(na[k],\min\left(nb[k],N(\{Ab}[i,j])\right)\right)$, where $k$
	 *                     is the index of the input interval that the output interval
	 *                     $]\{Ab}[j,0],\{Ab}[j,1]]$ came from, and $na[k]$ and $nb[k]$ are the the
	 *                     input values of $\{Nab}[k,0]$ and $\{Nab}[k,1]$.\n\n
	 *     If $\{ijob}=3$, then on output, $\{Nab}[i,j]$ contains $N(\{Ab}[i,j])$, unless
	 *                     $N(w)>\{nval}[i]$ for all search points $w$, in which case $\{Nab}[i,0]$
	 *                     will not be modified, i.e., the output value will be the same as the
	 *                     input value (modulo reorderings -- see §nval and §Ab), or unless
	 *                     $N(w)<\{nval}[i]$ for all search points $w$, in which case $\{Nab}[i,1]$
	 *                     will not be modified. Normally, §Nab should be set to some distinctive
	 *                     value(s) before §dlaebz is called.
	 *
	 * \param[out] work  an array,         dimension (§mmax)\n Workspace.
	 * \param[out] iwork an integer array, dimension (§mmax)\n Workspace.
	 * \param[out] info
	 *     $=0$:        All intervals converged.\n
	 *     $=1-\{mmax}$: The last §info intervals did not converge.\n
	 *     $=\{mmax}+1$:  More than mmax intervals were generated.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     This routine is intended to be called only by other LAPACK routines, thus the interface
	 *     is less user-friendly. It is intended for two purposes:
	 *      -# finding eigenvalues.\n In this case, §dlaebz should have one or more initial
	 *         intervals set up in §Ab, and §dlaebz should be called with $\{ijob}=1$. This sets up
	 *         §Nab, and also counts the eigenvalues. Intervals with no eigenvalues would usually
	 *         be thrown out at this point. Also, if not all the eigenvalues in an interval $i$ are
	 *         desired, $\{Nab}[i,0]$ can be increased or $\{Nab}[i,1]$ decreased. For example, set
	 *         $\{Nab}[i,0]=\{Nab}[i,1]-1$ to get the largest eigenvalue. §dlaebz is then called
	 *         with $\{ijob}=2$ and §mmax no smaller than the value of §mout returned by the call
	 *         with $\{ijob}=1$. After this ($\{ijob}=2$) call, eigenvalues $\{Nab}[i,0]+1$ through
	 *         $\{Nab}[i,1]$ are approximately $\{Ab}[i,0]$ (or $\{Ab}[i,1]$) to the tolerance
	 *         specified by §abstol and §reltol.
	 *      -# finding an interval $]a',b']$ containing eigenvalues $w(f),\ldots,w(l)$.\n
	 *         In this case, start with a Gershgorin interval $]a,b[$. Set up §Ab to contain 2
	 *         search intervals, both initially $]a,b[$. One §nval element should contain $f-1$ and
	 *         the other should contain $l$, while §c should contain $a$ and $b$, resp.
	 *         $\{Nab}[i,0]$ should be $-1$ and $\{Nab}[i,1]$ should be $\{n}+1$, to flag an error
	 *         if the desired interval does not lie in $]a,b[$. §dlaebz is then called with
	 *         $\{ijob}=3$. On exit, if $w(f-1)<w(f)$, then one of the intervals --$j$-- will have
	 *         $\{Ab}[j,0]=\{Ab}[j,1]$ and $\{Nab}[j,0]=\{Nab}[j,1]=f-1$, while if, to the
	 *         specified tolerance, $w(f-k)=\ldots=w(f+r)$, $k>0$ and $r\ge 0$, then the interval
	 *         will have $N(\{Ab}[j,0])=\{Nab}[j,0]=f-k$ and $N(\{Ab}[j,1])=\{Nab}[j,1]=f+r$. The
	 *         cases $w(l)<w(l+1)$ and $w(l-r)=\ldots=w(l+k)$ are handled similarly.             */
	static void dlaebz(int const ijob, int const nitmax, int const n, int const mmax,
	                   int const minp, int const nbmin, real const abstol, real const reltol,
	                   real const pivmin, real const* const d, real const* const e,
	                   real const* const e2, int* const nval, real* const Ab, real* const c,
	                   int& mout, int* const Nab, real* const work, int* const iwork, int& info)
	                   /*const*/
	{
		// Check for Errors
		info = 0;
		if (ijob<1 || ijob>3)
		{
			info = -1;
			return;
		}
		// Initialize Nab
		int j, ji;
		real tmp1;
		if (ijob==1)
		{
			// Compute the number of eigenvalues in the initial intervals.
			mout = 0;
			int jp, abi;
			for (ji=0; ji<minp; ji++)
			{
				for (jp=0; jp<=mmax; jp+=mmax)
				{
					abi = ji + jp;
					tmp1 = d[0] - Ab[abi];
					if (std::fabs(tmp1)<pivmin)
					{
						tmp1 = -pivmin;
					}
					Nab[abi] = 0;
					if (tmp1<=ZERO)
					{
						Nab[abi] = 1;
					}
					for (j=1; j<n; j++)
					{
						tmp1 = d[j] - e2[j-1]/tmp1 - Ab[abi];
						if (std::fabs(tmp1)<pivmin)
						{
							tmp1 = -pivmin;
						}
						if (tmp1<=ZERO)
						{
							Nab[abi]++;
						}
					}
				}
				mout += Nab[ji+mmax] - Nab[ji];
			}
			return;
		}
		// Initialize for loop
		// kf and kl have the following meaning:
		//     Intervals 0,...,kf-1 have converged.
		//     Intervals kf,...,kl  still need to be refined.
		int kf = 0;
		int kl = minp - 1;
		// If ijob=2, initialize c.
		// If ijob=3, use the user-supplied starting point.
		if (ijob==2)
		{
			for (ji=0; ji<minp; ji++)
			{
				c[ji] = HALF * (Ab[ji]+Ab[ji+mmax]);
			}
		}
		// Iteration loop
		int itmp1, itmp2, jip, jit, kfnew, klnew;
		real tmp2;
		for (jit=0; jit<nitmax; jit++)
		{
			// Loop over intervals
			if (kl-kf+1>=nbmin && nbmin>0)
			{
				// Begin of Parallel Version of the loop
				for (ji=kf; ji<=kl; ji++)
				{
					// Compute N(c), the number of eigenvalues less than c
					work[ji] = d[0] - c[ji];
					iwork[ji] = 0;
					if (work[ji]<=pivmin)
					{
						iwork[ji] = 1;
						work[ji] = std::min(work[ji], -pivmin);
					}
					for (j=1; j<n; j++)
					{
						work[ji] = d[j] - e2[j-1]/work[ji] - c[ji];
						if (work[ji]<=pivmin)
						{
							iwork[ji]++;
							work[ji] = std::min(work[ji], -pivmin);
						}
					}
				}
				if (ijob<=2)
				{
					// ijob=2: Choose all intervals containing eigenvalues.
					klnew = kl;
					for (ji=kf; ji<=kl; ji++)
					{
						jip = ji + mmax;
						// Ensure that N(w) is monotone
						iwork[ji] = std::min(Nab[jip], std::max(Nab[ji], iwork[ji]));
						// Update the Queue -- add intervals if both halves contain eigenvalues.
						if (iwork[ji]==Nab[jip])
						{
							// No eigenvalue in the upper interval: just use the lower interval.
							Ab[jip] = c[ji];
						}
						else if (iwork[ji]==Nab[ji])
						{
							// No eigenvalue in the lower interval: just use the upper interval.
							Ab[ji] = c[ji];
						}
						else
						{
							klnew++;
							if (klnew<mmax)
							{
								// Eigenvalue in both intervals -- add upper to queue.
								Ab[ klnew+mmax] = Ab[ jip];
								Nab[klnew+mmax] = Nab[jip];
								Ab[ klnew] =     c[ji];
								Nab[klnew] = iwork[ji];
								Ab[ jip] =     c[ji];
								Nab[jip] = iwork[ji];
							}
							else
							{
								info = mmax + 1;
							}
						}
					}
					if (info!=0)
					{
						return;
					}
					kl = klnew;
				}
				else
				{
					// ijob=3: Binary search. Keep only the interval containing w s.t. N(w) = nval
					for (ji=kf; ji<=kl; ji++)
					{
						if (iwork[ji]<=nval[ji])
						{
							Ab[ ji] =     c[ji];
							Nab[ji] = iwork[ji];
						}
						if (iwork[ji]>=nval[ji])
						{
							jip = ji + mmax;
							Ab[ jip] =     c[ji];
							Nab[jip] = iwork[ji];
						}
					}
				}
				// End of Parallel Version of the loop
			}
			else
			{
				// Begin of Serial Version of the loop
				klnew = kl;
				for (ji=kf; ji<=kl; ji++)
				{
					// Compute N(w), the number of eigenvalues less than w
					tmp1  = c[ji];
					tmp2  = d[0] - tmp1;
					itmp1 = 0;
					if (tmp2<=pivmin)
					{
						itmp1 = 1;
						tmp2  = std::min(tmp2, -pivmin);
					}
					for (j=1; j<n; j++)
					{
						tmp2 = d[j] - e2[j-1]/tmp2 - tmp1;
						if (tmp2<=pivmin)
						{
							itmp1++;
							tmp2 = std::min(tmp2, -pivmin);
						}
					}
					if (ijob<=2)
					{
						jip = ji + mmax;
						// ijob=2: Choose all intervals containing eigenvalues.
						// Ensure that N(w) is monotone
						itmp1 = std::min(Nab[jip], std::max(Nab[ji], itmp1));
						// Update the Queue -- add intervals if both halves contain eigenvalues.
						if (itmp1==Nab[jip])
						{
							// No eigenvalue in the upper interval: just use the lower interval.
							Ab[jip] = tmp1;
						}
						else if (itmp1==Nab[ji])
						{
							// No eigenvalue in the lower interval: just use the upper interval.
							Ab[ji] = tmp1;
						}
						else if (klnew<mmax-1)
						{
							// Eigenvalue in both intervals -- add upper to queue.
							klnew++;
							Ab[ klnew+mmax] =  Ab[jip];
							Nab[klnew+mmax] = Nab[jip];
							Ab[ klnew] = tmp1;
							Nab[klnew] = itmp1;
							Ab[ jip] = tmp1;
							Nab[jip] = itmp1;
						}
						else
						{
							info = mmax + 1;
							return;
						}
					}
					else
					{
						// ijob=3: Binary search.
						// Keep only the interval containing w s.t. N(w) = nval
						if (itmp1<=nval[ji])
						{
							Ab[ ji] = tmp1;
							Nab[ji] = itmp1;
						}
						if (itmp1>=nval[ji])
						{
							jip = ji + mmax;
							Ab[ jip] = tmp1;
							Nab[jip] = itmp1;
						}
					}
				}
				kl = klnew;
			}
			// Check for convergence
			kfnew = kf;
			for (ji=kf; ji<=kl; ji++)
			{
				jip = ji + mmax;
				tmp1 = std::fabs(Ab[jip]-Ab[ji]);
				tmp2 = std::max(std::fabs(Ab[jip]), std::fabs(Ab[ji]));
				if (tmp1<std::max(abstol, std::max(pivmin, reltol*tmp2)) || Nab[ji]>=Nab[jip])
				{
					// Converged -- Swap with position kfnew, then increment kfnew
					if (ji>kfnew)
					{
						tmp1  =  Ab[ji];
						tmp2  =  Ab[jip];
						itmp1 = Nab[ji];
						itmp2 = Nab[jip];
						Ab[ji]   =  Ab[kfnew];
						Ab[jip]  =  Ab[kfnew+mmax];
						Nab[ji]  = Nab[kfnew];
						Nab[jip] = Nab[kfnew+mmax];
						Ab[ kfnew]      = tmp1;
						Ab[ kfnew+mmax] = tmp2;
						Nab[kfnew]      = itmp1;
						Nab[kfnew+mmax] = itmp2;
						if (ijob==3)
						{
							itmp1       = nval[ji];
							nval[ji]    = nval[kfnew];
							nval[kfnew] = itmp1;
						}
					}
					kfnew++;
				}
			}
			kf = kfnew;
			// Choose Midpoints
			for (ji=kf; ji<=kl; ji++)
			{
				c[ji] = HALF * (Ab[ji]+Ab[ji+mmax]);
			}
			// If no more intervals to refine, quit.
			if (kf>kl)
			{
				break;
			}
		}
		// Converged
		info = std::max(kl-kf+1, 0);
		mout = kl + 1;
	}

	/*! §dlaed6 used by §sstedc. Computes one Newton step in solution of the secular equation.
	 *
	 * §dlaed6 computes the positive or negative root (closest to the origin) of\n
	 * $f(x) = \{rho}+\frac{\{z[0]}}{\{d[0]}-x}
	 *               +\frac{\{z[1]}}{\{d[1]}-x}
	 *               +\frac{\{z[2]}}{\{d[2]}-x}$\n
	 * It is assumed that
	 *     \li if §orgati = true the root is between §d[1] and §d[2];
	 *     \li otherwise it is between §d[0] and §d[1]
	 *
	 * This routine will be called by §dlaed4 when necessary. In most cases, the root sought is the
	 * smallest in magnitude, though it might not be in some extremely rare situations.
	 * \param[in] kniter
	 *     Refer to §dlaed4 for its significance.\n
	 *     NOTE: zero-based!
	 *
	 * \param[in] orgati
	 *     If orgati is true, the needed root is between §d[1] and §d[2]; otherwise it is between
	 *     §d[0] and §d[1]. See §dlaed4 for further details.
	 *
	 * \param[in] rho Refer to the equation $f(x)$ above.
	 * \param[in] d
	 *     an array, dimension (3)\n
	 *     §d satisfies §d[0] < §d[1] < §d[2].
	 *
	 * \param[in] z
	 *     an array, dimension (3)\n
	 *     Each of the elements in §z must be positive.
	 *
	 * \param[in] finit
	 *     The value of $f$ at 0. It is more accurate than the one evaluated inside this routine
	 *     (if someone wants to do so).
	 *
	 * \param[out] tau The root of the equation $f(x)$.
	 * \param[out] info
	 *     =0: successful exit\n
	 *     >0: if §info = 1, failure to converge
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 * \li 10/02/03: This version has a few statements commented out for thread safety
	 *               (machine parameters are computed on each entry). SJH.
	 * \li 05/10/06: Modified from a new version of Ren-Cang Li, use Gragg-Thornton-Warner cubic
	 *               convergent scheme for better stability.
	 *
	 * Contributors:
	 *     Ren-Cang Li, Computer Science Division, University of California at Berkeley, USA     */
	static void dlaed6(int const kniter, bool const orgati, real const rho, real const* const d,
	                   real const* const z, real const finit, real& tau, int& info) /*const*/
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
				// Scale up by power of radix nearest 1/safmin^(2/3)
				sclfac = sminv2;
				sclinv = small2;
			}
			else
			{
				// Scale up by power of radix nearest 1/safmin^(1/3)
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

	/*! §dlaev2 computes the eigenvalues and eigenvectors of a 2 by 2 symmetric/Hermitian matrix.
	 *
	 * §dlaev2 computes the eigendecomposition of a 2 by 2 symmetric matrix\n
	 *     $\b{bm} \{a} & \{b} \\
	 *             \{b} & \{c} \e{bm}$\n
	 * On return, §rt1 is the eigenvalue of the larger absolute value, §rt2 is the eigenvalue of
	 * the smaller absolute value, and (§cs1,§sn1) is the unit right eigenvector for §rt1, giving
	 * the decomposition\n
	 *     $\b{bm}\{cs1} & \{sn1}\\
	 *           -\{sn1} & \{cs1}\e{bm}\b{bm}\{a} & \{b}\\
	 *                                       \{b} & \{c}\e{bm}\b{bm}\{cs1} & -\{sn1}\\
	 *                                                              \{sn1} &  \{cs1}\e{bm}
	 *      = \b{bm}\{rt1} &   0   \\
	 *                 0   & \{rt2}\e{bm}$.
	 * \param[in] a The [0,0] element of the 2 by 2 matrix.
	 * \param[in] b
	 *     The [0,1] element and the conjugate of the [1,0] element of the 2 by 2 matrix.
	 *
	 * \param[in]  c        The [1,1] element of the 2 by 2 matrix.
	 * \param[out] rt1      The eigenvalue of larger absolute value.
	 * \param[out] rt2      The eigenvalue of smaller absolute value.
	 * \param[out] cs1, sn1 The vector [§cs1,§sn1] is a unit right eigenvector for §rt1.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     §rt1 is accurate to a few ulps barring over/underflow.\n
	 *     §rt2 may be inaccurate if there is massive cancellation in the determinant
	 *     $\{a}\{c}-\{b}\{b}$; higher precision or correctly rounded or correctly truncated
	 *     arithmetic would be needed to compute §rt2 accurately in all cases.\n
	 *     §cs1 and §sn1 are accurate to a few ulps barring over/underflow.\n
	 *     Overflow is possible only if §rt1 is within a factor of 5 of overflow.\n
	 *     Underflow is harmless if the input data is 0 or exceeds
	 *     $\{underflow\_threshold}/\{macheps}$.                                                 */
	static void dlaev2(real const a, real const b, real const c, real& rt1, real& rt2, real& cs1,
	                   real &sn1) /*const*/
	{
		// Compute the eigenvalues
		real sm  = a + c;
		real df  = a - c;
		real adf = std::fabs(df);
		real tb  = b + b;
		real ab  = std::fabs(tb);
		real acmn, acmx;
		if (std::fabs(a)>std::fabs(c))
		{
			acmx = a;
			acmn = c;
		}
		else
		{
			acmx = c;
			acmn = a;
		}
		real rt, tn;
		if (adf>ab)
		{
			tn = ab  / adf;
			rt = adf * std::sqrt(ONE+tn*tn);
		}
		else if (adf<ab)
		{
			tn = adf / ab;
			rt = ab  * std::sqrt(ONE+tn*tn);
		}
		else
		{
			// Includes case ab=adf=0
			rt = ab * std::sqrt(TWO);
		}
		int sgn1;
		if (sm<ZERO)
		{
			rt1 = HALF * (sm-rt);
			sgn1 = -1;
			// Order of execution important. To get fully accurate smaller eigenvalue,
			// next line needs to be executed in higher precision.
			rt2 = (acmx/rt1)*acmn - (b/rt1)*b;
		}
		else if (sm>ZERO)
		{
			rt1 = HALF * (sm+rt);
			sgn1 = 1;
			// Order of execution important. To get fully accurate smaller eigenvalue,
			// next line needs to be executed in higher precision.
			rt2 = (acmx/rt1)*acmn - (b/rt1)*b;
		}
		else
		{
			// Includes case rt1 = rt2 = 0
			rt1  =  HALF * rt;
			rt2  = -HALF * rt;
			sgn1 = 1;
		}
		// Compute the eigenvector
		int sgn2;
		real cs;
		if (df>=ZERO)
		{
			cs   = df + rt;
			sgn2 = 1;
		}
		else
		{
			cs   = df - rt;
			sgn2 = -1;
		}
		if (std::fabs(cs)>ab)
		{
			real ct = -tb / cs;
			sn1     = ONE / std::sqrt(ONE+ct*ct);
			cs1     = ct  * sn1;
		}
		else
		{
			if (ab==ZERO)
			{
				cs1 = ONE;
				sn1 = ZERO;
			}
			else
			{
				tn  = -cs / tb;
				cs1 = ONE / std::sqrt(ONE+tn*tn);
				sn1 = tn  * cs1;
			}
		}
		if (sgn1==sgn2)
		{
			tn  =  cs1;
			cs1 = -sn1;
			sn1 =  tn;
		}
	}

	/*! §dlaexc swaps adjacent diagonal blocks of a real upper quasi-triangular matrix in Schur
	 *  canonical form, by an orthogonal similarity transformation.
	 *
	 * §dlaexc swaps adjacent diagonal blocks $T_{00}$ and $T_{11}$ of order 1 or 2 in an upper
	 * quasi-triangular matrix $T$ by an orthogonal similarity transformation.\n
	 * $T$ must be in Schur canonical form, that is, block upper triangular with 1-by-1 and 2-by-2
	 * diagonal blocks; each 2-by-2 diagonal block has its diagonal elemnts equal and its
	 * off-diagonal elements of opposite sign.
	 * \param[in] wantq
	 *     = §true: accumulate the transformation in the matrix $Q$;\n
	 *     = §false: do not accumulate the transformation.
	 *
	 * \param[in]     n The order of the matrix $T$. $\{n}\ge 0$.
	 * \param[in,out] T
	 *     an array, dimension (§ldt,§n)\n
	 *     On entry, the upper quasi-triangular matrix $T$, in Schur canonical form.\n
	 *     On exit, the updated matrix $T$, again in Schur canonical form.
	 *
	 * \param[in]     ldt The leading dimension of the array $T$. $\{ldt}\ge\max(1,\{n})$.
	 * \param[in,out] Q
	 *     an array, dimension (§ldq,§n)\n
	 *     On entry, if §wantq is §true, the orthogonal matrix $Q$.\n
	 *     On exit, if §wantq is §true, the updated matrix $Q$.\n &emsp;&emsp;&emsp;
	 *              If §wantq is §false, $Q$ is not referenced.
	 *
	 * \param[in] ldq
	 *     The leading dimension of the array $Q$. $\{ldq}\ge 1$;
	 *     and if §wantq is §true, $\{ldq}\ge\{n}$.
	 *
	 * \param[in] j0
	 *     The index of the first row of the first block $T_{00}$.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in]  n0   The order of the first block $T_{00}$. $\{n0}=0$, 1 or 2.
	 * \param[in]  n1   The order of the second block $T_{11}$. $\{n1}=0$, 1 or 2.
	 * \param[out] work an array, dimension (§n)
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     = 1: the transformed matrix $T$ would be too far from Schur form;
	 *          the blocks are not swapped and T and Q are unchanged.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlaexc(bool const wantq, int const n, real* const T, int const ldt, real* const Q,
	                   int const ldq, int const j0, int const n0, int const n1, real* const work,
	                   int& info) /*const*/
	{
		const int LDD = 4;
		const int LDX = 2;
		info = 0;
		// Quick return if possible
		if (n==0 || n0==0 || n1==0)
		{
			return;
		}
		int j1 = j0 + 1;
		int j2 = j0 + 2;
		int j3 = j0 + 3;
		if (j1+n0>=n)
		{
			return;
		}
		int tj0 = ldt * j0;
		int tj1 = ldt * j1;
		int tj2 = ldt * j2;
		int qj0 = ldq * j0;
		real cs, sn, t11, temp;
		if (n0==1 && n1==1)
		{
			real t22;
			// Swap two 1-by-1 blocks.
			t11 = T[j0+tj0];
			t22 = T[j1+tj1];
			// Determine the transformation to perform the interchange.
			dlartg(T[j0+tj1], t22-t11, cs, sn, temp);
			// Apply transformation to the matrix T.
			if (j2<n)
			{
				Blas<real>::drot(n-j2, &T[j0+tj2], ldt, &T[j1+tj2], ldt, cs, sn);
			}
			Blas<real>::drot(j0, &T[tj0], 1, &T[tj1], 1, cs, sn);
			T[j0+tj0] = t22;
			T[j1+tj1] = t11;
			if (wantq)
			{
				// Accumulate transformation in the matrix Q.
				Blas<real>::drot(n, &Q[qj0], 1, &Q[ldq*j1], 1, cs, sn);
			}
		}
		else
		{
			int ierr;
			real scale, xnorm;
			real D[LDD*4], u[3], u1[3], u2[3], X[LDX*2];
			// Swapping involves at least one 2-by-2 block.
			// Copy the diagonal block of order n0+n1 to the local array D and compute its norm.
			int nd = n0 + n1;
			dlacpy("Full", nd, nd, &T[j0+tj0], ldt, D, LDD);
			real dnorm = dlange("Max", nd, nd, D, LDD, work);
			// Compute machine-dependent threshold for test for accepting swap.
			real eps    = dlamch("P");
			real smlnum = dlamch("S") / eps;
			real thresh = std::max(TEN*eps*dnorm, smlnum);
			// Solve t11*X - X*t22 = scale*T12 for X.
			dlasy2(false, false, -1, n0, n1, D, LDD, &D[n0+LDD*n0], LDD, &D[LDD*n0], LDD, scale, X,
			       LDX, xnorm, ierr);
			// Swap the adjacent diagonal blocks.
			real tau;
			int k = n0 + n0 + n1 - 3;
			switch (k)
			{
				case 1:
					// n0 = 1, n1 = 2: generate elementary reflector H so that:
					// (scale, X11, X12) H = (0, 0, *)
					u[0] = scale;
					u[1] = X[0];
					u[2] = X[LDX];
					dlarfg(3, u[2], u, 1, tau);
					u[2] = ONE;
					t11  = T[j0+tj0];
					// Perform swap provisionally on diagonal block in D.
					dlarfx("L", 3, 3, u, tau, D, LDD, work);
					dlarfx("R", 3, 3, u, tau, D, LDD, work);
					// Test whether to reject swap.
					if (std::max(std::max(std::fabs(D[2]), std::fabs(D[2+LDD*1])),
					             std::fabs(D[2+LDD*2]-t11)) > thresh)
					{
						// Exit with info = 1 if swap was rejected.
						info = 1;
						return;
					}
					// Accept swap: apply transformation to the entire matrix T.
					dlarfx("L", 3,  n-j0, u, tau, &T[j0+tj0], ldt, work);
					dlarfx("R", j2, 3,    u, tau, &T[tj0],    ldt, work);
					T[j2+tj0] = ZERO;
					T[j2+tj1] = ZERO;
					T[j2+tj2] = t11;
					if (wantq)
					{
						// Accumulate transformation in the matrix Q.
						dlarfx("R", n, 3, u, tau, &Q[qj0], ldq, work);
					}
					break;
				case 2:
					// n0 = 2, n1 = 1: generate elementary reflector H so that:
					// H (- X11) = (*)
					//   (- X21) = (0)
					//   (scale) = (0)
					u[0] = -X[0];
					u[1] = -X[1];
					u[2] = scale;
					dlarfg(3, u[0], &u[1], 1, tau);
					u[0] = ONE;
					{
						real t33 = T[j2+tj2];
						// Perform swap provisionally on diagonal block in D.
						dlarfx("L", 3, 3, u, tau, D, LDD, work);
						dlarfx("R", 3, 3, u, tau, D, LDD, work);
						// Test whether to reject swap.
						if (std::max(std::max(std::fabs(D[1]), std::fabs(D[2])),
						             std::fabs(D[0]-t33)) > thresh)
						{
							// Exit with info = 1 if swap was rejected.
							info = 1;
							return;
						}
						// Accept swap: apply transformation to the entire matrix T.
						dlarfx("R", j3, 3,    u, tau, &T[tj0],    ldt, work);
						dlarfx("L", 3,  n-j1, u, tau, &T[j0+tj1], ldt, work);
						T[j0+tj0] = t33;
					}
					T[j1+tj0] = ZERO;
					T[j2+tj0] = ZERO;
					if (wantq)
					{
						// Accumulate transformation in the matrix Q.
						dlarfx("R", n, 3, u, tau, &Q[qj0], ldq, work);
					}
					break;
				case 3:
					// n0 = 2, n1 = 2: generate elementary reflectors H(1) and H(2) so that:
					// H(2) H(1) (-X11  -X12) = (*  *)
					//           (-X21  -X22)   (0  *)
					//           (scale    0)   (0  0)
					//           (0    scale)   (0  0)
					real tau1, tau2;
					u1[0] = -X[0];
					u1[1] = -X[1];
					u1[2] = scale;
					dlarfg(3, u1[0], &u1[1], 1, tau1);
					u1[0] = ONE;
					temp  = -tau1 * (X[LDX]+u1[1]*X[1+LDX]);
					u2[0] = -temp*u1[1] - X[1+LDX];
					u2[1] = -temp*u1[2];
					u2[2] = scale;
					dlarfg(3, u2[0], &u2[1], 1, tau2);
					u2[0] = ONE;
					// Perform swap provisionally on diagonal block in D.
					dlarfx("L", 3, 4, u1, tau1, D,       LDD, work);
					dlarfx("R", 4, 3, u1, tau1, D,       LDD, work);
					dlarfx("L", 3, 4, u2, tau2, &D[1],   LDD, work);
					dlarfx("R", 4, 3, u2, tau2, &D[LDD], LDD, work);
					// Test whether to reject swap.
					if (std::max(std::max(std::fabs(D[2]), std::fabs(D[2+LDD])),
					             std::max(std::fabs(D[3]), std::fabs(D[3+LDD])))>thresh)
					{
						// Exit with info = 1 if swap was rejected.
						info = 1;
						return;
					}
					// Accept swap: apply transformation to the entire matrix T.
					dlarfx("L", 3,    n-j0, u1, tau1, &T[j0+tj0], ldt, work);
					dlarfx("R", j3+1, 3,    u1, tau1, &T[tj0],    ldt, work);
					dlarfx("L", 3,    n-j0, u2, tau2, &T[j1+tj0], ldt, work);
					dlarfx("R", j3+1, 3,    u2, tau2, &T[tj1],    ldt, work);
					T[j2+tj0] = ZERO;
					T[j2+tj1] = ZERO;
					T[j3+tj0] = ZERO;
					T[j3+tj1] = ZERO;
					if (wantq)
					{
						// Accumulate transformation in the matrix Q.
						dlarfx("R", n, 3, u1, tau1, &Q[qj0], ldq, work);
						dlarfx("R", n, 3, u2, tau2, &Q[ldq*j1], ldq, work);
					}
					break;
			}
			real wi1, wi2, wr1, wr2;
			if (n1==2)
			{
				// Standardize new 2-by-2 block t00
				dlanv2(T[j0+tj0], T[j0+tj1], T[j1+tj0], T[j1+tj1], wr1, wi1, wr2, wi2, cs, sn);
				Blas<real>::drot(n-j0, &T[j0+tj2], ldt, &T[j1+tj2], ldt, cs, sn);
				Blas<real>::drot(j0, &T[tj0], 1, &T[tj1], 1, cs, sn);
				if (wantq)
				{
					Blas<real>::drot(n, &Q[qj0], 1, &Q[ldq*j1], 1, cs, sn);
				}
			}
			if (n0==2)
			{
				// Standardize new 2-by-2 block t11
				j2 = j0 + n1;
				j3 = j2 + 1;
				int j4 = j2 + 2;
				tj2 = ldt * j2;
				int tj3 = ldt * j3;
				dlanv2(T[j2+tj2], T[j2+tj3], T[j3+tj2], T[j3+tj3], wr1, wi1, wr2, wi2, cs, sn);
				if (j4<n)
				{
					Blas<real>::drot(n-j4, &T[j2+ldt*j4], ldt, &T[j3+ldt*j4], ldt, cs, sn);
				}
				Blas<real>::drot(j2, &T[tj2], 1, &T[tj3], 1, cs, sn);
				if (wantq)
				{
					Blas<real>::drot(n, &Q[ldq*j2], 1, &Q[ldq*j3], 1, cs, sn);
				}
			}
		}
	}

	/*! §dlagtf computes an LU factorization of a matrix $T-\lambda I$, where $T$ is a general
	 *  tridiagonal matrix, and $\lambda$ a scalar, using partial pivoting with row interchanges.
	 *
	 * §dlagtf factorizes the matrix $T-\lambda I$, where $T$ is an §n by §n tridiagonal matrix and
	 * $\lambda$ is a scalar, as\n
	 *     $T - \lambda I = P L U$,\n
	 * where $P$ is a permutation matrix, $L$ is a unit lower tridiagonal matrix with at most one
	 * non-zero sub-diagonal element per column and $U$ is an upper triangular matrix with at most
	 * two non-zero super-diagonal elements per column.\n
	 * The factorization is obtained by Gaussian elimination with partial pivoting and implicit row
	 * scaling.\n
	 * The parameter §lambda is included in the routine so that §dlagtf may be used, in conjunction
	 * with §dlagts, to obtain eigenvectors of $T$ by inverse iteration.
	 * \param[in]     n The order of the matrix $T$.
	 * \param[in,out] a
	 *     an array, dimension (§n)\n On entry, §a must contain the diagonal elements of $T$.\n
	 *     On exit, §a is overwritten by the §n diagonal elements of the upper triangular matrix
	 *              $U$ of the factorization of $T$.
	 *
	 * \param[in]     lambda On entry, the scalar $\lambda$.
	 * \param[in,out] b
	 *     an array, dimension ($\{n}-1$)\n
	 *     On entry, §b must contain the $\{n}-1$ super-diagonal elements of $T$.\n
	 *     On exit, §b is overwritten by the $\{n}-1$ super-diagonal elements of the matrix $U$
	 *              of the factorization of $T$.
	 *
	 * \param[in,out] c
	 *     an array, dimension ($\{n}-1$)\n
	 *     On entry, §c must contain the $\{n}-1$ sub-diagonal elements of $T$.\n
	 *     On exit, §c is overwritten by the $\{n}-1$ sub-diagonal elements of the matrix $L$ of
	 *              the factorization of $T$.
	 *
	 * \param[in] tol
	 *     On entry, a relative tolerance used to indicate whether or not the matrix $T-\lambda I$
	 *     is nearly singular. §tol should normally be chosen as approximately the largest relative
	 *     error in the elements of $T$. For example, if the elements of $T$ are correct to about 4
	 *     significant figures, then §tol should be set to about $5*10^{-4}$. If §tol is supplied
	 *     as less than §eps, where §eps is the relative machine precision, then the value §eps is
	 *     used in place of §tol.
	 *
	 * \param[out] d
	 *     an array, dimension ($\{n}-2$)\n
	 *     On exit, §d is overwritten by the $\{n}-2$ second super-diagonal elements of the matrix
	 *              $U$ of the factorization of $T$.
	 *
	 * \param[out] in
	 *     an integer array, dimension (§n)\n
	 *     On exit, §in contains details of the permutation matrix $P$. If an interchange occurred
	 *     at the $k$th step of the elimination, then $\{in}[k]=1$, otherwise $\{in}[k]=0$. The
	 *     element $\{in}[\{n}-1]$ returns the smallest positive integer $j$ such that\n
	 *         $|U[j,j]|\le\|(T-\lambda I)[j]\|*\{tol}$,\n
	 *     where $\|A[j]\|$ denotes the sum of the absolute values of the $j$th row of the matrix
	 *     $A$. If no such $j$ exists then $\{in}[\{n}-1]$ is returned as -1. If $\{in}[\{n}-1]$
	 *     is returned as positive, then a diagonal element of $U$ is small, indicating that
	 *     ($T-\lambda I$) is singular or nearly singular.\n
	 *     NOTE: $\{in}[\{n}-1]$ is a zero-based index!
	 *
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if $\{info}=-k$, the $k$th argument had an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlagtf(int const n, real* const a, real const lambda, real* const b, real* const c,
	                   real const tol, real* const d, int* const in, int& info) /*const*/
	{
		info = 0;
		if (n<0)
		{
			info = -1;
			xerbla("DLAGTF", -info);
			return;
		}
		if (n==0)
		{
			return;
		}
		a[0] -= lambda;
		int nm = n - 1;
		in[nm] = -1;
		if (n==1)
		{
			if (a[0]==ZERO)
			{
				in[0] = 1;
			}
			return;
		}
		real eps = dlamch("Epsilon");
		real tl = std::max(tol, eps);
		real scale1 = std::fabs(a[0]) + std::fabs(b[0]);
		int k, kp;
		real mult, piv1, piv2, scale2, temp;
		for (k=0; k<nm; k++)
		{
			kp = k + 1;
			a[kp] -= lambda;
			scale2 = std::fabs(c[k]) + std::fabs(a[kp]);
			if (kp<nm)
			{
				scale2 += std::fabs(b[kp]);
			}
			if (a[k]==ZERO)
			{
				piv1 = ZERO;
			}
			else
			{
				piv1 = std::fabs(a[k]) / scale1;
			}
			if (c[k]==ZERO)
			{
				in[k]  = 0;
				piv2   = ZERO;
				scale1 = scale2;
				if (kp<nm)
				{
					d[k] = ZERO;
				}
			}
			else
			{
				piv2 = std::fabs(c[k]) / scale2;
				if (piv2<=piv1)
				{
					in[k]  = 0;
					scale1 = scale2;
					c[k]  /= a[k];
					a[kp] -= c[k] * b[k];
					if (kp<nm)
					{
						d[k] = ZERO;
					}
				}
				else
				{
					in[k] = 1;
					mult  = a[k] / c[k];
					a[k]  = c[k];
					temp  = a[kp];
					a[kp] = b[k] - mult*temp;
					if (kp<nm)
					{
						d[k]  = b[kp];
						b[kp] = -mult * d[k];
					}
					b[k] = temp;
					c[k] = mult;
				}
			}
			if (std::max(piv1, piv2)<=tl && in[nm]==-1)
			{
				in[nm] = k;
			}
		}
		if (std::fabs(a[nm])<=scale1*tl && in[nm]==-1)
		{
			in[nm] = nm;
		}
	}

	/*! §dlagts solves the system of equations $(T-\lambda I)x = y$ or $(T-\lambda I)^Tx = y$,
	 *  where $T$ is a general tridiagonal matrix and $\lambda$ a scalar, using the LU
	 *  factorization computed by §dlagtf.
	 *
	 * §dlagts may be used to solve one of the systems of equations\n
	 *     $(T - \lambda I) x = y$ or $(T - \lambda I)^T x = y$,\n
	 * where $T$ is an §n by §n tridiagonal matrix, for $x$, following the factorization of
	 * $(T - \lambda I)$ as\n
	 *     $(T - \lambda I) = P L U$,\n
	 * by routine §dlagtf. The choice of equation to be solved is controlled by the argument §job,
	 * and in each case there is an option to perturb zero or very small diagonal elements of $U$,
	 * this option being intended for use in applications such as inverse iteration.
	 * \param[in] job
	 *     Specifies the job to be performed by §dlagts as follows:\n
	 *     = 1: The equations $(T - \lambda I)x = y$ are to be solved,
	 *          but diagonal elements of $U$ are not to be perturbed.\n
	 *     =-1: The equations $(T - \lambda I)x = y$ are to be solved and, if
	 *          overflow would otherwise occur, the diagonal elements of $U$ are to be perturbed.
	 *          See argument §tol below.\n
	 *     = 2: The equations $(T - \lambda I)^Tx = y$ are to be solved,
	 *          but diagonal elements of $U$ are not to be perturbed.\n
	 *     =-2: The equations $(T - \lambda I)^Tx = y$ are to be solved and, if
	 *          overflow would otherwise occur, the diagonal elements of $U$ are to be perturbed.
	 *          See argument §tol below.
	 *
	 * \param[in] n The order of the matrix $T$.
	 * \param[in] a
	 *     an array, dimension (§n)\n
	 *     On entry, §a must contain the diagonal elements of $U$ as returned from §dlagtf.
	 *
	 * \param[in] b
	 *     an array, dimension ($\{n}-1$)\n
	 *     On entry, §b must contain the first super-diagonal elements of $U$ as returned from
	 *               §dlagtf.
	 *
	 * \param[in] c
	 *     an array, dimension ($\{n}-1$)\n
	 *     On entry, §c must contain the sub-diagonal elements of $L$ as returned from §dlagtf.
	 *
	 * \param[in] d
	 *     an array, dimension ($\{n}-2$)\n
	 *     On entry, §d must contain the second super-diagonal elements of $U$ as returned from
	 *               §dlagtf.
	 *
	 * \param[in] in
	 *     an integer array, dimension (§n)\n
	 *     On entry, §in must contain details of the matrix $P$ as returned from §dlagtf.\n
	 *     NOTE: $\{in}[\{n}-1]$ must be a zero-based index!
	 *
	 * \param[in,out] y
	 *     an array, dimension (§n)\n
	 *     On entry, the right hand side vector $y$.\n
	 *     On exit, §y is overwritten by the solution vector $x$.
	 *
	 * \param[in,out] tol
	 *     On entry, with $\{job}<0$, §tol should be the minimum perturbation to be made to very
	 *               small diagonal elements of $U$. §tol should normally be chosen as about
	 *               $\{eps}\|U\|$, where §eps is the relative machine precision, but if §tol is
	 *               supplied as non-positive, then it is reset to $\{eps}\,\max(|U[i,j]|)$.
	 *               If $\{job}>0$ then §tol is not referenced.\n
	 *     On exit, §tol is changed as described above, only if §tol is non-positive on entry.
	 *              Otherwise §tol is unchanged.
	 *
	 * \param[out] info
	 *     =0: successful exit\n
	 *     <0: if $\{info}=-i$, the $i$-th argument had an illegal value\n
	 *     >0: overflow would occur when computing the $(\{info}-1)$th element of the solution
	 *         vector $x$. This can only occur when §job is supplied as positive and either means that a
	 *         diagonal element of $U$ is very small, or that the elements of the right-hand side
	 *         vector $y$ are very large.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlagts(int const job, int const n, real const* const a, real const* const b,
	                   real const* const c, real const* const d, int const* const in,
	                   real* const y, real& tol, int& info) /*const*/
	{
		info = 0;
		if ((std::fabs(job)>2) || (job==0))
		{
			info = -1;
		}
		else if (n<0)
		{
			info = -2;
		}
		if (info!=0)
		{
			xerbla("DLAGTS", -info);
			return;
		}
		if (n==0)
		{
			return;
		}
		real eps = dlamch("Epsilon");
		real sfmin = dlamch("Safe minimum");
		real bignum = ONE / sfmin;
		int k;
		if (job<0)
		{
			if (tol<=ZERO)
			{
				tol = std::fabs(a[0]);
				if (n>1)
				{
					tol = std::max(tol, std::max(std::fabs(a[1]), std::fabs(b[0])));
				}
				for (k=2; k<n; k++)
				{
					tol = std::max(std::max(tol,               std::fabs(a[k])),
					               std::max(std::fabs(b[k-1]), std::fabs(d[k-2])));
				}
				tol *= eps;
				if (tol==ZERO)
				{
					tol = eps;
				}
			}
		}
		real absak, ak, pert, temp;
		if (std::fabs(job)==1)
		{
			for (k=1; k<n; k++)
			{
				if (in[k-1]==0)
				{
					y[k] -= c[k-1] * y[k-1];
				}
				else
				{
					temp   = y[k-1];
					y[k-1] = y[k];
					y[k]   = temp - c[k-1]*y[k];
				}
			}
			if (job==1)
			{
				for (k=n-1; k>=0; k--)
				{
					if (k<n-2)
					{
						temp = y[k] - b[k]*y[k+1] - d[k]*y[k+2];
					}
					else if (k==n-2)
					{
						temp = y[k] - b[k]*y[k+1];
					}
					else
					{
						temp = y[k];
					}
					ak = a[k];
					absak = std::fabs(ak);
					if (absak<ONE)
					{
						if (absak<sfmin)
						{
							if (absak==ZERO || std::fabs(temp)*sfmin>absak)
							{
								info = k + 1;
								return;
							}
							else
							{
								temp *= bignum;
								ak   *= bignum;
							}
						}
						else if (std::fabs(temp)>absak*bignum)
						{
							info = k + 1;
							return;
						}
					}
					y[k] = temp / ak;
				}
			}
			else
			{
				for (k=n-1; k>=0; k--)
				{
					if (k<n-2)
					{
						temp = y[k] - b[k]*y[k+1] - d[k]*y[k+2];
					}
					else if (k==n-2)
					{
						temp = y[k] - b[k]*y[k+1];
					}
					else
					{
						temp = y[k];
					}
					ak = a[k];
					pert = std::copysign(tol, ak);
					while (true)
					{
						absak = std::fabs(ak);
						if (absak<ONE)
						{
							if (absak<sfmin)
							{
								if (absak==ZERO || std::fabs(temp)*sfmin>absak)
								{
									ak += pert;
									pert *= 2;
									continue;
								}
								else
								{
									temp *= bignum;
									ak  *= bignum;
								}
							}
							else if (std::fabs(temp)>absak*bignum)
							{
								ak   += pert;
								pert *= 2;
								continue;
							}
						}
						break;
					}
					y[k] = temp / ak;
				}
			}
		}
		else
		{
			// Come to here if job = 2 or -2
			if (job==2)
			{
				for (k=0; k<n; k++)
				{
					if (k>=2)
					{
						temp = y[k] - b[k-1]*y[k-1] - d[k-2]*y[k-2];
					}
					else if (k==1)
					{
						temp = y[k] - b[k-1]*y[k-1];
					}
					else
					{
						temp = y[k];
					}
					ak = a[k];
					absak = std::fabs(ak);
					if (absak<ONE)
					{
						if (absak<sfmin)
						{
							if (absak==ZERO || std::fabs(temp)*sfmin>absak)
							{
								info = k + 1;
								return;
							}
							else
							{
								temp *= bignum;
								ak   *= bignum;
							}
						}
						else if (std::fabs(temp)>absak*bignum)
						{
							info = k + 1;
							return;
						}
					}
					y[k] = temp / ak;
				}
			}
			else
			{
				for (k=0; k<n; k++)
				{
					if (k>=2)
					{
						temp = y[k] - b[k-1]*y[k-1] - d[k-2]*y[k-2];
					}
					else if (k==1)
					{
						temp = y[k] - b[k-1]*y[k-1];
					}
					else
					{
						temp = y[k];
					}
					ak = a[k];
					pert = std::copysign(tol, ak);
					while (true)
					{
						absak = std::fabs(ak);
						if (absak<ONE)
						{
							if (absak<sfmin)
							{
								if (absak==ZERO || std::fabs(temp)*sfmin>absak)
								{
									ak   += pert;
									pert *= 2;
									continue;
								}
								else
								{
									temp *= bignum;
									ak   *= bignum;
								}
							}
							else if (std::fabs(temp)>absak*bignum)
							{
								ak   += pert;
								pert *= 2;
								continue;
							}
						}
						break;
					}
					y[k] = temp / ak;
				}
			}
			for (k=n-1; k>=1; k--)
			{
				if (in[k-1]==0)
				{
					y[k-1] -= c[k-1]*y[k];
				}
				else
				{
					temp   = y[k-1];
					y[k-1] = y[k];
					y[k]   = temp - c[k-1]*y[k];
				}
			}
		}
	}

	/*! §dlahqr computes the eigenvalues and Schur factorization of an upper Hessenberg matrix,
	 *  using the double-shift/single-shift QR algorithm.
	 *
	 * §dlahqr is an auxiliary routine called by §dhseqr to update the eigenvalues and Schur
	 * decomposition already computed by §dhseqr, by dealing with the Hessenberg submatrix in rows
	 * and columns §ilo to §ihi.
	 * \param[in] wantt
	 *     = true:  the full Schur form $T$ is required;
	 *     = false: only eigenvalues are required.
	 *
	 * \param[in] wantz
	 *     = true:  the matrix of Schur vectors $Z$ is required;
	 *     = false: Schur vectors are not required.
	 *
	 * \param[in] n The order of the matrix $H$. $\{n}\ge 0$.
	 * \param[in] ilo, ihi
	 *     It is assumed that §H is already upper quasi-triangular in rows and columns
	 *     $\{ihi}+1:\{n}-1$, and that $\{H}[\{ilo},\{ilo}-1] = 0$ (unless $\{ilo} = 0$).
	 *     §dlahqr works primarily with the Hessenberg submatrix in rows and columns §ilo to §ihi,
	 *     but applies transformations to all of §H if §wantt is true.\n
	 *     $0\le\{ilo}\le\max(0,\{ihi})$; $\{ihi}<\{n}$.\n
	 *     NOTE: Zero-based indices!
	 *
	 * \param[in,out] H
	 *     an array, dimension (§ldh,§n)\n
	 *     On entry, the upper Hessenberg matrix $H$.\n
	 *     On exit, if §info is zero and if §wantt is true, §H is upper quasi-triangular in rows
	 *     and columns $\{ilo}:\{ihi}$, with any 2 by 2 diagonal blocks in standard form. If
	 *     §info is zero and §wantt is false, the contents of §H are unspecified on exit. The
	 *     output state of §H if §info is nonzero is given below in the description of §info.
	 *
	 * \param[in]  ldh The leading dimension of the array §H. $\{ldh}\ge\max(1,\{n})$.
	 * \param[out] wr, wi
	 *     arrays, dimension (§n)\n
	 *     The real and imaginary parts, respectively, of the computed eigenvalues §ilo to §ihi are
	 *     stored in the corresponding elements of §wr and §wi. If two eigenvalues are computed as
	 *     a complex conjugate pair, they are stored in consecutive elements of §wr and §wi, say
	 *     the $i$-th and $(i+1)$-th, with $\{wi}[i]>0$ and $\{wi}[i+1]<0$. If §wantt is true, the
	 *     eigenvalues are stored in the same order as on the diagonal of the Schur form returned
	 *     in §H, with $\{wr}[i]=\{H}[i,i]$, and, if $\{H}[i:i+1,i:i+1]$ is a 2 by 2 diagonal
	 *     block, $\{wi}[i]=\sqrt{\{H}[i+1,i]\{H}[i,i+1]}$ and $\{wi}[i+1]=-\{wi}[i]$.
	 *
	 * \param[in] iloz, ihiz
	 *     Specify the rows of $Z$ to which transformations must be applied if §wantz is true.
	 *     $0 \le \{iloz} \le \{ilo}$; $\{ihi} \le \{ihiz} < \{n}$.\n
	 *     NOTE: Zero-based indices!
	 *
	 * \param[in,out] Z
	 *     an array, dimension (§ldz,§n)\n
	 *     If §wantz is true, on entry §Z must contain the current matrix $Z$ of transformations
	 *     accumulated by §dhseqr, and on exit §Z has been updated; transformations are applied
	 *     only to the submatrix $\{Z}[\{iloz}:\{ihiz},\{ilo}:\{ihi}]$.\n
	 *     If §wantz is false, §Z is not referenced.
	 *
	 * \param[in]  ldz  The leading dimension of the array §Z. $\{ldz}\ge\max(1,\{n})$.
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     > 0: If $\{info}=I$, §dlahqr failed to compute all the eigenvalues §ilo to §ihi in a
	 *          total of 30 iterations per eigenvalue; elements $I:\{ihi}$ of §wr and §wi contain
	 *          those eigenvalues which have been successfully computed.\n
	 *    &emsp;If $\{info}>0$ and §wantt is false, then on exit, the remaining unconverged
	 *          eigenvalues are the eigenvalues of the upper Hessenberg matrix rows and columns
	 *          §ilo through $\{info}-1$ of the final, output value of §H.\n
	 *    &emsp;If $\{info}>0$ and §wantt is true, then on exit\n
	 *    &emsp;(*)&emsp;&emsp;$(\text{initial value of }\{H})U = U(\text{final value of }\{H})$\n
	 *    &emsp;where $U$ is an orthognal matrix. The final value of §H is upper Hessenberg and
	 *          triangular in rows and columns §info through §ihi.\n
	 *    &emsp;If $\{info}>0$ and §wantz is true, then on exit
	 *          $(\text{final value of }\{Z}) = (\text{initial value of }\{Z}) U$ where $U$ is the
	 *          orthogonal matrix in (*) (regardless of the value of §wantt.)
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     02-96 Based on modifications by David Day, Sandia National Laboratory, USA\n
	 *     12-04 Further modifications by Ralph Byers, University of Kansas, USA\n
	 *     This is a modified version of §dlahqr from LAPACK version 3.0.\n
	 *     It is \li more robust against overflow and underflow and \li adopts the more
	 *     conservative Ahues & Tisseur stopping criterion (LAWN 122, 1997).                     */
	static void dlahqr(bool const wantt, bool const wantz, int const n, int const ilo,
	                   int const ihi, real* const H, int const ldh, real* const wr, real* const wi,
	                   int const iloz, int const ihiz, real* const Z, int const ldz, int& info)
	                   /*const*/
	{
		const real DAT1 = THREE / FOUR;
		const real DAT2 = real(-0.4375);
		info = 0;
		// Quick return if possible
		if (n==0)
		{
			return;
		}
		if (ilo==ihi)
		{
			wr[ilo] = H[ilo+ldh*ilo];
			wi[ilo] = ZERO;
			return;
		}
		// clear out the trash
		int j;
		for (j=ilo; j<ihi-2; j++)
		{
			H[j+2+ldh*j] = ZERO;
			H[j+3+ldh*j] = ZERO;
		}
		if (ilo<=ihi-2)
		{
			H[ihi+ldh*(ihi-2)] = ZERO;
		}
		int nh = ihi - ilo + 1;
		int nz = ihiz - iloz + 1;
		// Set machine-dependent constants for the stopping criterion.
		real safmin = dlamch("SAFE MINIMUM");
		real safmax = ONE / safmin;
		dlabad(safmin, safmax);
		real ulp    = dlamch("PRECISION");
		real smlnum = safmin * (real(nh)/ulp);
		// i1 and i2 are the indices of the first row and last column of H to which transformations
		// must be applied. If eigenvalues only are being computed, i1 and i2 are set inside the
		// main loop.
		int i1, i2=-1;
		if (wantt)
		{
			i1 = 0;
			i2 = n - 1;
		}
		// itmax is the total number of QR iterations allowed.
		int itmax = 30 * std::max(10, nh);
		// The main loop begins here. i is the loop index and decreases from ihi+1 to ilo+1 in steps of
		// 1 or 2. Each iteration of the loop works with the active submatrix in rows and columns
		// l to i. Eigenvalues i+1 to ihi have already converged. Either l = ilo or
		// H[l,l-1] is negligible so that the matrix splits.
		real aa, ab, ba, bb, cs, det, h11, h12, h21, h21s, h22, rt1i, rt1r, rt2i, rt2r, rtdisc, s,
			 sn, sum, t1, t2, t3, tr, tst, v2, v3;
		int hind1, hind2, its, k, l, m, nr, zind;
		real v[3];
		int i = ihi;
		bool conv;
		while (i>=ilo)
		{
			l = ilo;
			// Perform QR iterations on rows and columns ilo to i until a submatrix of order 1
			// or 2 splits off at the bottom because a subdiagonal element has become negligible.
			conv = false;
			for (its=0; its<=itmax; its++)
			{
				// Look for a single small subdiagonal element.
				for (k=i; k>l; k--)
				{
					hind1 = k + ldh*k;
					hind2 = hind1 - ldh;
					if (std::fabs(H[hind2])<=smlnum)
					{
						break;
					}
					tst = std::fabs(H[hind2-1]) + std::fabs(H[hind1]);
					if (tst==ZERO)
					{
						if (k-2>=ilo)
						{
							tst += std::fabs(H[hind2-1-ldh]);
						}
						if (k<ihi)
						{
							tst += std::fabs(H[hind1+1]);
						}
					}
					// The following is a conservative small subdiagonal deflation criterion due to
					// Ahues & Tisseur (LAWN 122, 1997). It has better mathematical foundation and
					// improves accuracy in some cases.
					if (std::fabs(H[hind2])<=ulp*tst)
					{
						ab = std::max(std::fabs(H[hind2]), std::fabs(H[hind1-1]));
						ba = std::min(std::fabs(H[hind2]), std::fabs(H[hind1-1]));
						aa = std::max(std::fabs(H[hind1]), std::fabs(H[hind2-1]-H[hind1]));
						bb = std::min(std::fabs(H[hind1]), std::fabs(H[hind2-1]-H[hind1]));
						s = aa + ab;
						if (ba*(ab/s) <= std::max(smlnum, ulp*(bb*(aa/s))))
						{
							break;
						}
					}
				}
				l = k;
				if (l>ilo)
				{
					// H[l,l-1] is negligible
					H[l+ldh*(l-1)] = ZERO;
				}
				// Exit from loop if a submatrix of order 1 or 2 has split off.
				if (l+1>=i)
				{
					conv = true;
					break;
				}
				// Now the active submatrix is in rows and columns l to i. If eigenvalues only are
				// being computed, only the active submatrix need be transformed.
				if (!wantt)
				{
					i1 = l;
					i2 = i;
				}
				if (its==10)
				{
					// Exceptional shift.
					s = std::fabs(H[l+1+ldh*l]) + std::fabs(H[l+2+ldh*(l+1)]);
					h11 = DAT1*s + H[l+ldh*l];
					h12 = DAT2*s;
					h21 = s;
					h22 = h11;
				}
				else if (its==20)
				{
					// Exceptional shift.
					s = std::fabs(H[i+ldh*(i-1)]) + std::fabs(H[i-1+ldh*(i-2)]);
					h11 = DAT1*s + H[i+ldh*i];
					h12 = DAT2*s;
					h21 = s;
					h22 = h11;
				}
				else
				{
					// Prepare to use Francis' double shift
					// (i.e. 2nd degree generalized Rayleigh quotient)
					h11 = H[i-1+ldh*(i-1)];
					h21 = H[i  +ldh*(i-1)];
					h12 = H[i-1+ldh*i];
					h22 = H[i  +ldh*i];
				}
				s = std::fabs(h11) + std::fabs(h12) + std::fabs(h21) + std::fabs(h22);
				if (s==ZERO)
				{
					rt1r = ZERO;
					rt1i = ZERO;
					rt2r = ZERO;
					rt2i = ZERO;
				}
				else
				{
					h11 /= s;
					h21 /= s;
					h12 /= s;
					h22 /= s;
					tr = (h11+h22) / TWO;
					det = (h11-tr)*(h22-tr) - h12*h21;
					rtdisc = std::sqrt(std::fabs(det));
					if (det>=ZERO)
					{
						// complex conjugate shifts
						rt1r = tr * s;
						rt2r = rt1r;
						rt1i = rtdisc * s;
						rt2i = -rt1i;
					}
					else
					{
						// real shifts (use only one of them)
						rt1r = tr + rtdisc;
						rt2r = tr - rtdisc;
						if (std::fabs(rt1r-h22)<=std::fabs(rt2r-h22))
						{
							rt1r *= s;
							rt2r  = rt1r;
						}
						else
						{
							rt2r *= s;
							rt1r = rt2r;
						}
						rt1i = ZERO;
						rt2i = ZERO;
					}
				}
				// Look for two consecutive small subdiagonal elements.
				for (m=i-2; m>=l; m--)
				{
					hind1 = m + ldh*m;
					// Determine the effect of starting the double-shift QR iteration at row m,
					// and see if this would make H[m,m-1] negligible.
					// (The following uses scaling to avoid overflows and most underflows.)
					h21s = H[hind1+1];
					s = std::fabs(H[hind1]-rt2r) + std::fabs(rt2i) + std::fabs(h21s);
					h21s = H[hind1+1] / s;
					v[0] = h21s*H[hind1+ldh] + (H[hind1]-rt1r)*((H[hind1]-rt2r)/s) - rt1i*(rt2i/s);
					v[1] = h21s * (H[hind1]+H[hind1+1+ldh]-rt1r-rt2r);
					v[2] = h21s * H[hind1+2+ldh];
					s = std::fabs(v[0]) + std::fabs(v[1]) + std::fabs(v[2]);
					v[0] /= s;
					v[1] /= s;
					v[2] /= s;
					if (m==l)
					{
						break;
					}
					if (std::fabs(H[hind1-ldh]) * (std::fabs(v[1])+std::fabs(v[2]))
					    <= ulp * std::fabs(v[0])
					       * (std::fabs(H[hind1-1-ldh])+std::fabs(H[hind1])
					          +std::fabs(H[hind1+1+ldh])))
					{
						break;
					}
				}
				// Double-shift QR step
				for (k=m; k<i; k++)
				{
					hind1 = k + ldh*(k-1);
					// The first iteration of this loop determines a reflection G from the vector v
					// and applies it from left and right to H, thus creating a nonzero bulge below
					// the subdiagonal.
					// Each subsequent iteration determines a reflection G to restore the
					// Hessenberg form in the (k-1)-th column, and thus chases the bulge one step
					// toward the bottom of the active submatrix. nr is the order of G.
					nr = std::min(3, i-k+1);
					if (k>m)
					{
						Blas<real>::dcopy(nr, &H[hind1], 1, v, 1);
					}
					dlarfg(nr, v[0], &v[1], 1, t1);
					if (k>m)
					{
						H[hind1]   = v[0];
						H[hind1+1] = ZERO;
						if (k<i-1)
						{
							H[hind1+2] = ZERO;
						}
					}
					else if (m>l)
					{
						// Use the following instead of H[k,k-1] = -H[k,k-1] to avoid a bug when
						// v[1] and v[2] underflow.
						H[hind1] *= ONE - t1;
					}
					v2 = v[1];
					t2 = t1 * v2;
					hind1 = ldh * k;
					hind2 = hind1 + ldh;
					zind = ldz * k;
					if (nr==3)
					{
						v3 = v[2];
						t3 = t1 * v3;
						// Apply G from the left to transform the rows of the matrix
						// in columns k to i2.
						for (j=k; j<=i2; j++)
						{
							sum = H[k+ldh*j] + v2*H[k+1+ldh*j] + v3*H[k+2+ldh*j];
							H[k+ldh*j]   -= sum * t1;
							H[k+1+ldh*j] -= sum * t2;
							H[k+2+ldh*j] -= sum * t3;
						}
						// Apply G from the right to transform the columns of the matrix
						// in rows i1 to min(k+3,i).
						for (j=i1; j<=std::min(k+3, i); j++)
						{
							sum = H[j+hind1] + v2*H[j+hind2] + v3*H[j+hind2+ldh];
							H[j+hind1]     -= sum * t1;
							H[j+hind2]     -= sum * t2;
							H[j+hind2+ldh] -= sum * t3;
						}
						if (wantz)
						{
							// Accumulate transformations in the matrix Z
							for (j=iloz; j<=ihiz; j++)
							{
								sum = Z[j+zind] + v2*Z[j+zind+ldz] + v3*Z[j+zind+2*ldz];
								Z[j+zind]       -= sum * t1;
								Z[j+zind+ldz]   -= sum * t2;
								Z[j+zind+2*ldz] -= sum * t3;
							}
						}
					}
					else if (nr==2)
					{
						// Apply G from the left to transform the rows of the matrix in columns
						// k to i2.
						for (j=k; j<=i2; j++)
						{
							sum = H[k+ldh*j] + v2*H[k+1+ldh*j];
							H[k+ldh*j]   -= sum * t1;
							H[k+1+ldh*j] -= sum * t2;
						}
						// Apply G from the right to transform the columns of the matrix in rows
						// i1 to min(k+3,i).
						for (j=i1; j<=i; j++)
						{
							sum = H[j+hind1] + v2*H[j+hind2];
							H[j+hind1] -= sum * t1;
							H[j+hind2] -= sum * t2;
						}
						if (wantz)
						{
							// Accumulate transformations in the matrix Z
							for (j=iloz; j<=ihiz; j++)
							{
								sum = Z[j+ldz*k] + v2*Z[j+zind+ldz];
								Z[j+zind]     -= sum * t1;
								Z[j+zind+ldz] -= sum * t2;
							}
						}
					}
				}
			}
			if (!conv)
			{
				// Failure to converge in remaining number of iterations
				info = i+1;
				return;
			}
			hind1 = i + ldh*i;
			if (l==i)
			{
				// H[i,i-1] is negligible: one eigenvalue has converged.
				wr[i] = H[hind1];
				wi[i] = ZERO;
			}
			else if (l==i-1)
			{
				// H[i-1,i-2] is negligible: a pair of eigenvalues have converged.
				// Transform the 2 by 2 submatrix to standard Schur form, and compute and store the
				// eigenvalues.
				dlanv2(H[hind1-1-ldh], H[hind1-1], H[hind1-ldh], H[hind1], wr[i-1], wi[i-1], wr[i],
				       wi[i], cs, sn);
				if (wantt)
				{
					// Apply the transformation to the rest of H.
					if (i2>i)
					{
						Blas<real>::drot(i2-i, &H[hind1-1+ldh], ldh, &H[hind1+ldh], ldh, cs, sn);
					}
					Blas<real>::drot(i-i1-1, &H[i1+ldh*(i-1)], 1, &H[i1+ldh*i], 1, cs, sn);
				}
				if (wantz)
				{
					// Apply the transformation to Z.
					Blas<real>::drot(nz, &Z[iloz+ldz*(i-1)], 1, &Z[iloz+ldz*i], 1, cs, sn);
				}
			}
			// return to start of the main loop with new value of i.
			i = l - 1;
		}
	}

	/*! §dlahr2 reduces the specified number of first columns of a general rectangular matrix $A$
	 *  so that elements below the specified subdiagonal are zero, and returns auxiliary matrices
	 *  which are needed to apply the transformation to the unreduced part of $A$.
	 *
	 * §dlahr2 reduces the first §nb columns of a real general §n by (§n-k+1) matrix $A$ so that
	 * elements below the §k -th subdiagonal are zero. The reduction is performed by an orthogonal
	 * similarity transformation $Q^T A Q$. The routine returns the matrices $V$ and $T$ which
	 * determine $Q$ as a block reflector $I - V T V^T$, and also the matrix $Y = A V T$.\n
	 * This is an auxiliary routine called by §dgehrd.
	 * \param[in] n The order of the matrix $A$.
	 * \param[in] k
	 *     The offset for the reduction. Elements below the §k -th subdiagonal in the first §nb
	 *     columns are reduced to zero. $\{k} < \{n}$.
	 *
	 * \param[in]     nb The number of columns to be reduced.
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n-k+1)\n
	 *     On entry, the §n by (§n-k+1) general matrix $A$.\n
	 *     On exit, the elements on and above the §k -th subdiagonal in the first §nb columns are
	 *     overwritten with the corresponding elements of the reduced matrix; the elements below
	 *     the §k -th subdiagonal, with the array §tau, represent the matrix $Q$ as a product of
	 *     elementary reflectors. The other columns of §A are unchanged. See Remark.
	 *
	 * \param[in]  lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{n})$.
	 * \param[out] tau
	 *     an array, dimension (§nb)\n
	 *     The scalar factors of the elementary reflectors. See Remark.
	 *
	 * \param[out] T   an array, dimension (§ldt,§nb)\n The upper triangular matrix $T$.
	 * \param[in]  ldt The leading dimension of the array $T$. $\{ldt}\ge\{nb}$.
	 * \param[out] Y   an array, dimension (§ldy,§nb)\n The §n by §nb matrix $Y$.
	 * \param[in]  ldy The leading dimension of the array §Y. $\{ldy}\ge\{n}$.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     The matrix $Q$ is represented as a product of §nb elementary reflectors\n
	 *         $Q = H(0) H(1) \ldots H(nb-1)$.\n
	 *     Each $H(i)$ has the form\n
	 *         $H(i) = I - \tau v v^T$\n.
	 *     where $\tau$ is a real scalar, and $v$ is a real vector with $v[0:i+k-1] = 0$,
	 *     $v[i+k] = 1$; $v[i+k+1:n-1]$ is stored on exit in $\{A}[i+k+1:n-1,i]$, and $\tau$ in
	 *     $\{tau}[i]$.\n
	 *     The elements of the vectors $v$ together form the (§n-k+1) by §nb matrix $V$ which is
	 *     needed, with $T$ and $Y$, to apply the transformation to the unreduced part of the
	 *     matrix, using an update of the form:\n
	 *         $A = (I - V T V^T) (A - Y V^T)$.\n
	 *     The contents of §A on exit are illustrated by the following example with $\{n} = 7$,
	 *     $\{k} = 3$ and $\{nb} = 2$:\n
	 *         $\b{bm} a  &  a  & a & a & a \\
	 *                 a  &  a  & a & a & a \\
	 *                 a  &  a  & a & a & a \\
	 *                 h  &  h  & a & a & a \\
	 *                v_1 &  h  & a & a & a \\
	 *                v_1 & v_2 & a & a & a \\
	 *                v_1 & v_2 & a & a & a \e{bm}$\n
	 *     where $a$ denotes an element of the original matrix $A$, $h$ denotes a modified element
	 *     of the upper Hessenberg matrix $H$, and $v_i$ denotes an element of the vector defining
	 *     $H(i)$.\n
	 *     This subroutine is a slight modification of LAPACK-3.0's §dlahrd incorporating
	 *     improvements proposed by Quintana-Orti and Van de Gejin. Note that the entries of
	 *     $\{A}[0:k-1,1:\{nb}-1]$ differ from those returned by the original LAPACK-3.0's §dlahrd
	 *     routine. (This subroutine is not backward compatible with LAPACK-3.0's §dlahrd.)\n\n
	 *     References:\n
	 *         Gregorio Quintana-Orti and Robert van de Geijn, "Improving the performance of
	 *         reduction to Hessenberg form," ACM Transactions on Mathematical Software,
	 *         32(2):180-194, June 2006.                                                         */
	static void dlahr2(int const n, int const k, int const nb, real* const A, int const lda,
	                   real* const tau, real* const T, int const ldt, real* const Y, int const ldy)
	                   /*const*/
	{
		// Quick return if possible
		if (n<=1)
		{
			return;
		}
		real ei=ZERO;
		int nbm = nb - 1;
		int tnbm = ldt * nbm;
		int nmk = n - k;
		int kpi, ldai, nmki, kcoli, kpicoli, ldti, yind;
		for (int i=0; i<nb; i++)
		{
			kpi = k + i;
			ldai = lda * i;
			kpicoli = kpi + ldai;
			nmki = nmk - i;
			if (i>0)
			{
				kcoli = k + ldai;
				// Update A[k:n-1,i]
				// Update i-th column of A - Y * V^T
				Blas<real>::dgemv("NO TRANSPOSE", nmk, i, -ONE, &Y[k], ldy, &A[kpi-1], lda, ONE,
				                  &A[kcoli], 1);
				// Apply I - V * T^T * V^T to this column (call it b) from the left,
				// using the last column of T as workspace
				// Let  V = (V1)   and   b = (b1)   (first i rows)
				//          (V2)             (b2)
				// where V1 is unit lower triangular
				// w := V1^T * b1
				Blas<real>::dcopy(i, &A[kcoli], 1, &T[tnbm], 1);
				Blas<real>::dtrmv("Lower", "Transpose", "UNIT", i, &A[k], lda, &T[tnbm], 1);
				// w := w + V2^T * b2
				Blas<real>::dgemv("Transpose", nmki, i, ONE, &A[kpi], lda, &A[kpicoli], 1, ONE,
				                  &T[tnbm], 1);
				// w := T^T * w
				Blas<real>::dtrmv("Upper", "Transpose", "NON-UNIT", i, T, ldt, &T[tnbm], 1);
				// b2 := b2 - V2*w
				Blas<real>::dgemv("NO TRANSPOSE", nmki, i, -ONE, &A[kpi], lda, &T[tnbm], 1, ONE,
				                  &A[kpicoli], 1);
				// b1 := b1 - V1*w
				Blas<real>::dtrmv("Lower", "NO TRANSPOSE", "UNIT", i, &A[k], lda, &T[tnbm], 1);
				Blas<real>::daxpy(i, -ONE, &T[tnbm], 1, &A[kcoli], 1);
				A[kpi-1+ldai-lda] = ei;
			}
			// Generate the elementary reflector H(i) to annihilate A[k+i+1:n-1,i]
			dlarfg(nmki, A[kpicoli], &A[std::min(kpi+1, n-1)+ldai], 1, tau[i]);
			ei = A[kpicoli];
			A[kpicoli] = ONE;
			// Compute  Y[k:n-1,i]
			yind = k + ldy*i;
			ldti = ldt * i;
			Blas<real>::dgemv("NO TRANSPOSE", nmk, nmki, ONE, &A[k+ldai+lda], lda, &A[kpicoli], 1,
			                  ZERO, &Y[yind], 1);
			Blas<real>::dgemv("Transpose", nmki, i, ONE, &A[kpi], lda, &A[kpicoli], 1, ZERO,
			                  &T[ldti], 1);
			Blas<real>::dgemv("NO TRANSPOSE", nmk, i, -ONE, &Y[k], ldy, &T[ldti], 1, ONE, &Y[yind],
			                  1);
			Blas<real>::dscal(nmk, tau[i], &Y[yind], 1);
			// Compute T[0:i,i]
			Blas<real>::dscal(i, -tau[i], &T[ldti], 1);
			Blas<real>::dtrmv("Upper", "No Transpose", "NON-UNIT", i, T, ldt, &T[ldti], 1);
			T[i+ldti] = tau[i];
		}
		A[k+nbm+lda*nbm] = ei;
		// Compute Y[0:k-1,0:nb-1]
		dlacpy("ALL", k, nb, &A[lda], lda, Y, ldy);
		Blas<real>::dtrmm("RIGHT", "Lower", "NO TRANSPOSE", "UNIT", k, nb, ONE, &A[k], lda, Y,
		                  ldy);
		if (n>k+nb)
		{
			Blas<real>::dgemm("NO TRANSPOSE", "NO TRANSPOSE", k, nb, nmk-nb, ONE, &A[lda*(nb+1)],
			                  lda, &A[k+nb], lda, ONE, Y, ldy);
		}
		Blas<real>::dtrmm("RIGHT", "Upper", "NO TRANSPOSE", "NON-UNIT", k, nb, ONE, T, ldt, Y,
		                  ldy);
	}

	/*! §dlaln2 solves a 1 by 1 or 2 by 2 linear system of equations of the specified form.
	 *
	 * §dlaln2 solves a system of the form  $(c A - w D) X = s B$ or $(c A^T - w D) X = s B$ with
	 * possible scaling ($s$) and perturbation of $A$. ($A^T$ means $A$-transpose.)\n
	 * $A$ is an §na by §na real matrix, $c$ is a real scalar, $D$ is an §na by §na real diagonal
	 * matrix, $w$ is a real or complex value, and $X$ and $B$ are §na by 1 matrices -- real if §w
	 * is real, complex if §w is complex. §na may be 1 or 2.\n
	 * If $w$ is complex, $X$ and $B$ are represented as §na by 2 matrices, the first column of
	 * each being the real part and the second being the imaginary part.\n
	 * $s$ is a scaling factor ($\le 1$), computed by §dlaln2, which is so chosen that $X$ can be
	 * computed without overflow. $X$ is further scaled if necessary to assure that
	 * $\on{norm}(c A - w D)\on{norm}(X)$ is less than overflow.\n
	 * If both singular values of $(c A - w D)$ are less than §smin, $\{smin}\cdot\{Identity}$
	 * will be used instead of $(c A - w D)$. If only one singular value is less than §smin, one
	 * element of $(c A - w D)$ will be perturbed enough to make the smallest singular value
	 * roughly §smin. If both singular values are at least §smin, $(c A - w D)$ will not be
	 * perturbed. In any case, the perturbation will be at most some small multiple of
	 * $\max(\{smin},\{ulp}\cdot\on{norm}(c A - w D))$. The singular values are computed by
	 * infinity-norm approximations, and thus will only be correct to a factor of 2 or so.\n
	 * Note: all input quantities are assumed to be smaller than overflow by a reasonable factor.
	 * (See §bignum.)
	 * \param[in] ltrans
	 *     = true:  $A$-transpose will be used.\n
	 *     = false: $A$ will be used (not transposed.)
	 *
	 * \param[in] na   The size of the matrix $A$.\n It may (only) be 1 or 2.
	 * \param[in] nw   1 if $w$ is real, 2 if $w$ is complex.\n It may only be 1 or 2.
	 * \param[in] smin
	 *     The desired lower bound on the singular values of $A$. This should be a safe distance
	 *     away from underflow or overflow, say, between (underflow/machine precision) and
	 *     (machine precision * overflow). (See §bignum and §ulp.)
	 *
	 * \param[in] ca  The coefficient $c$, which $A$ is multiplied by.
	 * \param[in] A   an array, dimension (§lda,§na)\n The §na by §na matrix $A$.
	 * \param[in] lda The leading dimension of §A. It must be at least §na.
	 * \param[in] d1  The 0,0 element in the diagonal matrix $D$.
	 * \param[in] d2  The 1,1 element in the diagonal matrix $D$. Not used if §na = 1.
	 * \param[in] B
	 *     an array, dimension (§ldb,§nw)\n
	 *     The §na by §nw matrix $B$ (right-hand side). If §nw = 2 ($w$ is complex), column 0
	 *     contains the real part of $B$ and column 1 contains the imaginary part.
	 *
	 * \param[in]  ldb The leading dimension of $B$. It must be at least §na.
	 * \param[in]  wr  The real part of the scalar $w$.
	 * \param[in]  wi The imaginary part of the scalar $w$. Not used if §nw = 1.
	 * \param[out] X
	 *     an array, dimension (§ldx,§nw)\n
	 *     The §na by &nw matrix $X$ (unknowns), as computed by §dlaln2.\n
	 *     If §nw = 2 ($w$ is complex), on exit, column 1 will contain the real part of $X$ and
	 *     column 2 will contain the imaginary part.
	 *
	 * \param[in]  ldx   The leading dimension of §X. It must be at least §na.
	 * \param[out] scale
	 *     The scale factor that $B$ must be multiplied by to insure that overflow does not occur
	 *     when computing $X$. Thus, $(c A - w D) X$  will be $\{scale}\ B$, not $B$ (ignoring
	 *     perturbations of $A$.) It will be at most 1.
	 *
	 * \param[out] xnorm
	 *     The infinity-norm of $X$, when $X$ is regarded as an §na by §nw real matrix.
	 *
	 * \param[out] info
	 *     An error flag. It will be set to zero if no error occurs, a negative number if an
	 *     argument is in error, or a positive number if $c A - w D$ had to be perturbed.
	 *     The possible values are:\n
	 *         = 0: No error occurred, and $(c A - w D)$ did not have to be perturbed.\n
	 *         = 1: $(c A - w D)$ had to be perturbed to make its smallest (or only) singular value
	 *              greater than §smin. \n
	 *     NOTE: In the interests of speed, this routine does not check the inputs for errors.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlaln2(bool const ltrans, int const na, int const nw, real smin, real const ca,
	                   real const* const A, int const lda, real const d1, real const d2,
	                   real const* const B, int const ldb, real const wr, real const wi,
	                   real* const X, int const ldx, real& scale, real& xnorm, int& info) /*const*/
	{
		const bool ZSWAP[4] = {false, false, true,  true};
		const bool RSWAP[4] = {false, true,  false, true};
		const int IPIVOT[16] = {0, 1, 2, 3,
		                        1, 0, 3, 2,
		                        2, 3, 0, 1,
		                        3, 2, 1, 0};
		// Compute bignum
		real smlnum = TWO * dlamch("Safe minimum");
		real bignum = ONE / smlnum;
		if (smin<smlnum)
		{
			smin = smlnum;
		}
		// Don't check for input errors
		info = 0;
		// Standard Initializations
		scale = ONE;
		real bnorm;
		if (na==1)
		{
			real cnorm, csr;
			// 1 by 1 (i.e., scalar) system C X = B
			if (nw==1)
			{
				// Real 1 by 1 system.
				// C = ca A - w D
				csr = ca*A[0] - wr*d1;
				cnorm = std::fabs(csr);
				// If |C| < smin, use C = smin
				if (cnorm<smin)
				{
					csr   = smin;
					cnorm = smin;
					info = 1;
				}
				// Check scaling for X = B / C
				bnorm = std::fabs(B[0]);
				if (cnorm<ONE && bnorm>ONE)
				{
					if (bnorm>bignum*cnorm)
					{
						scale = ONE / bnorm;
					}
				}
				// Compute X
				X[0] = (B[0]*scale) / csr;
				xnorm = std::fabs(X[0]);
			}
			else
			{
				// Complex 1 by 1 system ($w$ is complex) C = ca A - w D
				csr = ca*A[0] - wr*d1;
				real csi =    -wi * d1;
				cnorm = std::fabs(csr) + std::fabs(csi);
				// If |C| < smin, use C = smin
				if (cnorm<smin)
				{
					csr   = smin;
					csi   = ZERO;
					cnorm = smin;
					info = 1;
				}
				// Check scaling for X = B / C
				bnorm = std::fabs(B[0]) + std::fabs(B[ldb]);
				if (cnorm<ONE && bnorm>ONE)
				{
					if (bnorm>bignum*cnorm)
					{
						scale = ONE / bnorm;
					}
				}
				// Compute X
				dladiv(scale*B[0], scale*B[ldb], csr, csi, X[0], X[ldx]);
				xnorm = std::fabs(X[0]) + std::fabs(X[ldx]);
			}
		}
		else
		{
			int icmax, j;
			real bbnd, br1, br2, cmax, cr21, cr22, lr21, temp, ur11, ur11r, ur12, ur22, xr1, xr2;
			real crv[4];
			// 2 by 2 System
			// Compute the real part of  C = ca A - w D  (or  ca A^T - w D)
			crv[0] = ca*A[0]     - wr*d1;//CR[0,0]
			crv[3] = ca*A[1+lda] - wr*d2;//CR[1,1]
			if (ltrans)
			{
				crv[2] = ca*A[1];  //CR[0,1]
				crv[1] = ca*A[lda];//CR[1,0]
			}
			else
			{
				crv[1] = ca*A[1];  //CR[1,0]
				crv[2] = ca*A[lda];//CR[0,1]
			}
			if (nw==1)
			{
				// Real 2 by 2 system (w is real)
				// Find the largest element in C
				cmax = ZERO;
				icmax = -1;
				for (j=0; j<4; j++)
				{
					if (std::fabs(crv[j])>cmax)
					{
						cmax = std::fabs(crv[j]);
						icmax = j;
					}
				}
				// If norm(C) < smin, use smin*identity.
				if (cmax<smin)
				{
					bnorm = std::max(std::fabs(B[0]), std::fabs(B[1]));
					if (smin<ONE && bnorm>ONE)
					{
						if (bnorm>bignum*smin)
						{
							scale = ONE / bnorm;
						}
					}
					temp = scale / smin;
					X[0]  = temp*B[0];
					X[1]  = temp*B[1];
					xnorm = temp*bnorm;
					info = 1;
					return;
				}
				// Gaussian elimination with complete pivoting.
				ur11 = crv[icmax];
				cr21 = crv[IPIVOT[1+4*icmax]];//IPIVOT[1,icmax]
				ur12 = crv[IPIVOT[2+4*icmax]];//IPIVOT[2,icmax]
				cr22 = crv[IPIVOT[3+4*icmax]];//IPIVOT[3,icmax]
				ur11r = ONE / ur11;
				lr21 = ur11r * cr21;
				ur22 = cr22 - ur12*lr21;
				// If smaller pivot < smin, use smin
				if (std::fabs(ur22)<smin)
				{
					ur22 = smin;
					info = 1;
				}
				if (RSWAP[icmax])
				{
					br1 = B[1];
					br2 = B[0];
				}
				else
				{
					br1 = B[0];
					br2 = B[1];
				}
				br2 -= lr21 * br1;
				bbnd = std::max(std::fabs(br1*(ur22*ur11r)), std::fabs(br2));
				if (bbnd>ONE && std::fabs(ur22)<ONE)
				{
					if (bbnd>=bignum*std::fabs(ur22))
					{
						scale = ONE / bbnd;
					}
				}
				xr2 = (br2*scale) / ur22;
				xr1 = (scale*br1)*ur11r - xr2*(ur11r*ur12);
				if (ZSWAP[icmax])
				{
					X[0] = xr2;
					X[1] = xr1;
				}
				else
				{
					X[0] = xr1;
					X[1] = xr2;
				}
				xnorm = std::max(std::fabs(xr1), std::fabs(xr2));
				// Further scaling if norm(A) norm(X) > overflow
				if (xnorm>ONE && cmax>ONE)
				{
					if (xnorm > bignum/cmax)
					{
						temp = cmax / bignum;
						X[0]  *= temp;
						X[1]  *= temp;
						xnorm *= temp;
						scale *= temp;
					}
				}
			}
			else
			{
				real bi1, bi2, ci21, ci22, li21, u22abs, ui11, ui11r, ui12, ui12s, ui22, ur12s,
				     xi1, xi2;
				real civ[4];
				// Complex 2 by 2 system (w is complex)
				// Find the largest element in C
				civ[0] = -wi * d1;//CI[0,0]
				civ[1] = ZERO;    //CI[1,0]
				civ[2] = ZERO;    //CI[0,1]
				civ[3] = -wi * d2;//CI[1,1]
				cmax = ZERO;
				icmax = -1;
				for (j=0; j<4; j++)
				{
					if (std::fabs(crv[j])+std::fabs(civ[j])>cmax)
					{
						cmax = std::fabs(crv[j]) + std::fabs(civ[j]);
						icmax = j;
					}
				}
				// If norm(C) < smin, use smin*identity.
				if (cmax<smin)
				{
					bnorm = std::max(std::fabs(B[0])+std::fabs(B[ldb]),
					                 std::fabs(B[1])+std::fabs(B[1+ldb]));
					if (smin<ONE && bnorm>ONE)
					{
						if (bnorm>bignum*smin)
						{
							scale = ONE / bnorm;
						}
					}
					temp = scale / smin;
					X[0]     = temp*B[0];
					X[1]     = temp*B[1];
					X[ldx]   = temp*B[ldb];
					X[1+ldx] = temp*B[1+ldb];
					xnorm = temp*bnorm;
					info = 1;
					return;
				}
				// Gaussian elimination with complete pivoting.
				ur11 = crv[icmax];
				ui11 = civ[icmax];
				cr21 = crv[IPIVOT[1+4*icmax]];//IPIVOT[1,icmax]
				ci21 = civ[IPIVOT[1+4*icmax]];//IPIVOT[1,icmax]
				ur12 = crv[IPIVOT[2+4*icmax]];//IPIVOT[2,icmax]
				ui12 = civ[IPIVOT[2+4*icmax]];//IPIVOT[2,icmax]
				cr22 = crv[IPIVOT[3+4*icmax]];//IPIVOT[3,icmax]
				ci22 = civ[IPIVOT[3+4*icmax]];//IPIVOT[3,icmax]
				if (icmax==0 || icmax==3)
				{
					// Code when off-diagonals of pivoted C are real
					if (std::fabs(ur11)>std::fabs(ui11))
					{
						temp  = ui11 / ur11;
						ur11r = ONE / (ur11*(ONE+temp*temp));
						ui11r = -temp * ur11r;
					}
					else
					{
						temp  = ur11 / ui11;
						ui11r = -ONE / (ui11*(ONE+temp*temp));
						ur11r = -temp * ui11r;
					}
					lr21  = cr21 * ur11r;
					li21  = cr21 * ui11r;
					ur12s = ur12 * ur11r;
					ui12s = ur12 * ui11r;
					ur22  = cr22 - ur12*lr21;
					ui22  = ci22 - ur12*li21;
				}
				else
				{
					// Code when diagonals of pivoted C are real
					ur11r = ONE / ur11;
					ui11r = ZERO;
					lr21  = cr21 * ur11r;
					li21  = ci21 * ur11r;
					ur12s = ur12 * ur11r;
					ui12s = ui12 * ur11r;
					ur22  = cr22 - ur12*lr21 + ui12*li21;
					ui22  = -ur12*li21 - ui12*lr21;
				}
				u22abs = std::fabs(ur22) + std::fabs(ui22);
				// If smaller pivot < smin, use smin
				if (u22abs<smin)
				{
					ur22 = smin;
					ui22 = ZERO;
					info = 1;
				}
				if (RSWAP[icmax])
				{
					br2 = B[0];
					br1 = B[1];
					bi2 = B[ldb];
					bi1 = B[1+ldb];
				}
				else
				{
					br1 = B[0];
					br2 = B[1];
					bi1 = B[ldb];
					bi2 = B[1+ldb];
				}
				br2 = br2 - lr21*br1 + li21*bi1;
				bi2 = bi2 - li21*br1 - lr21*bi1;
				bbnd = std::max((std::fabs(br1)+std::fabs(bi1))
				                * (u22abs*(std::fabs(ur11r)+std::fabs(ui11r))),
				                std::fabs(br2)+std::fabs(bi2));
				if (bbnd>ONE && u22abs<ONE)
				{
					if (bbnd>=bignum*u22abs)
					{
						scale = ONE / bbnd;
						br1 *= scale;
						bi1 *= scale;
						br2 *= scale;
						bi2 *= scale;
					}
				}
				dladiv(br2, bi2, ur22, ui22, xr2, xi2);
				xr1 = ur11r*br1 - ui11r*bi1 - ur12s*xr2 + ui12s*xi2;
				xi1 = ui11r*br1 + ur11r*bi1 - ui12s*xr2 - ur12s*xi2;
				if (ZSWAP[icmax])
				{
					X[0]     = xr2;
					X[1]     = xr1;
					X[ldx]   = xi2;
					X[1+ldx] = xi1;
				}
				else
				{
					X[0]     = xr1;
					X[1]     = xr2;
					X[ldx]   = xi1;
					X[1+ldx] = xi2;
				}
				xnorm = std::max(std::fabs(xr1)+std::fabs(xi1), std::fabs(xr2)+std::fabs(xi2));
				// Further scaling if  norm(A) norm(X) > overflow
				if (xnorm>ONE && cmax>ONE)
				{
					if (xnorm > bignum/cmax)
					{
						temp = cmax / bignum;
						X[0]     *= temp;
						X[1]     *= temp;
						X[ldx]   *= temp;
						X[1+ldx] *= temp;
						xnorm    *= temp;
						scale    *= temp;
					}
				}
			}
		}
	}

	/*! §dlamrg creates a permutation list to merge the entries of two independently sorted sets
	 *  into a single set sorted in ascending order.
	 *
	 * §dlamrg will create a permutation list which will merge the elements of a (which is composed
	 * of two independently sorted sets) into a single set which is sorted in ascending order.
	 * \param[in] n1
	 * \param[in] n2
	 *     These arguments contain the respective lengths of the two sorted lists to be merged.
	 *
	 * \param[in] a
	 *     an array, dimension (§n1+§n2) \n
	 *     The first §n1 elements of §a contain a list of numbers which are sorted in either
	 *     ascending or descending order. Likewise for the final §n2 elements.
	 *
	 * \param[in] dtrd1
	 * \param[in] dtrd2
	 *     These are the strides to be taken through the array §a. Allowable strides are 1 and -1.
	 *     They indicate whether a subset of §a is sorted in ascending (§dtrdx = 1) or descending
	 *     (§DTRDx = -1) order.
	 *
	 * \param[out] index
	 *     an integer array, dimension (§n1+§n2) \n
	 *     On exit this array will contain a permutation such that if §b[§i] = §a[§index[§i]] for
	 *     §i = 0, ... , §n1+§n2-1, then §b will be sorted in ascending order.\n
	 *     NOTE: Zero-based indices!
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016                                                                           */
	static void dlamrg(int n1, int n2, real const* const a, int const dtrd1, int const dtrd2,
	                   int* const index) /*const*/
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

	/*! §dlangb returns the value of the 1-norm, Frobenius norm, infinity-norm, or the largest
	 *  absolute value of any element of general band matrix.
	 *
	 * §dlangb returns the value of the one-norm, or the Frobenius norm, or the infinity-norm, or
	 * the element of largest absolute value of an §n by §n band matrix $A$, with §kl sub-diagonals
	 * and §ku super-diagonals.
	 * \param[in] norm Specifies the value to be returned by §dlangb as described below.
	 * \param[in] n
	 *     The order of the matrix $A$. $\{n}\ge 0$. When $\{n}=0$, §dlangb returns zero.
	 *
	 * \param[in] kl The number of sub-diagonals of the matrix $A$. $\{kl}\ge 0$.
	 * \param[in] ku The number of super-diagonals of the matrix $A$. $\{ku}\ge 0$.
	 * \param[in] Ab
	 *     an array, dimension (§ldab,§n)\n
	 *     The band matrix $A$, stored in rows 0 to $\{kl}+\{ku}$. The $j$-th column of $A$ is
	 *     stored in the $j$-th column of the array §Ab as follows:\n
	 *         $\{Ab}[\{ku}+i-j,j]=A[i,j]$ for $\max(0,j-\{ku})\le i\le\min(\{n}-1,j+\{kl})$.
	 *
	 * \param[in] ldab The leading dimension of the array §Ab. $\{ldab}\ge\{kl}+\{ku}+1$.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$),\n
	 *     where $\{lwork}\ge\{n}$ when §norm ='I'; otherwise, §work is not referenced.
	 *
	 * \return
	 * $\left(\begin{tabular}{ll}
	 *     \(\max(|\{A}[i,j]|)\), & \{norm} = 'M' or 'm'          \\
	 *     \(\on{norm1}(\{A})\),  & \{norm} = '1', 'O' or 'o'     \\
	 *     \(\on{normI}(\{A})\),  & \{norm} = 'I' or 'i'          \\
	 *     \(\on{normF}(\{A})\),  & \{norm} = 'F', 'f', 'E' or 'e'
	 *  \end{tabular}\right.$\n
	 * where $\on{norm1}$ denotes the       one-norm of a matrix (maximum column sum),
	 *       $\on{normI}$ denotes the  infinity-norm of a matrix (maximum row sum) and
	 *       $\on{normF}$ denotes the Frobenius norm of a matrix (square root of sum of squares).\n
	 *       Note that $\max(|\{A}[i,j]|)$ is not a consistent matrix norm.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static real dlangb(char const* const norm, int const n, int const kl, int const ku,
	                   real const* const Ab, int const ldab, real* const work) /*const*/
	{
		int i, j, abj, stop;
		real sum, value=ZERO, temp;
		char upnorm = std::toupper(norm[0]);
		if (n==0)
		{
			value = ZERO;
		}
		else if (upnorm=='M')
		{
			// Find max(|A[i,j]|).
			value = ZERO;
			for (j=0; j<n; j++)
			{
				abj = ldab * j;
				stop = std::min(n+ku-j, kl+ku+1);
				for (i=std::max(ku-j, 0); i<stop; i++)
				{
					temp = std::fabs(Ab[i+abj]);
					if (value<temp || std::isnan(temp))
					{
						value = temp;
					}
				}
			}
		}
		else if (upnorm=='O' || norm[0]=='1')
		{
			// Find norm1(A).
			value = ZERO;
			for (j=0; j<n; j++)
			{
				abj = ldab * j;
				sum = ZERO;
				stop = std::min(n+ku-j, kl+ku+1);
				for (i=std::max(ku-j, 0); i<stop; i++)
				{
					sum += std::fabs(Ab[i+abj]);
				}
				if (value<sum || std::isnan(sum))
				{
					value = sum;
				}
			}
		}
		else if (upnorm=='I')
		{
			// Find normI(A).
			for (i=0; i<n; i++)
			{
				work[i] = ZERO;
			}
			for (j=0; j<n; j++)
			{
				abj = ku - j + ldab*j;
				stop = std::min(n, j+kl+1);
				for (i=std::max(0, j-ku); i<stop; i++)
				{
					work[i] += std::fabs(Ab[i+abj]);
				}
			}
			value = ZERO;
			for (i=0; i<n; i++)
			{
				temp = work[i];
				if (value<temp || std::isnan(temp))
				{
					value = temp;
				}
			}
		}
		else if (upnorm=='F' || upnorm=='E')
		{
			// Find normF(A).
			real scale = ZERO;
			sum = ONE;
			int l;
			for (j=0; j<n; j++)
			{
				l = std::max(0, j-ku);
				dlassq(std::min(n, j+kl+1)-l, &Ab[ku-j+l+ldab*j], 1, scale, sum);
			}
			value = scale * std::sqrt(sum);
		}
		return value;
	}

	/*! §dlange returns the value of the 1-norm, Frobenius norm, infinity-norm, or the largest
	 *  absolute value of any element of a general rectangular matrix.
	 *
	 * §dlange returns the value of the 1-norm, or the Frobenius norm, or the infinity-norm, or the
	 * largest absolute value of any element of a general real rectangular matrix $A$.
	 * \param[in] norm Specifies the value to be returned by §dlange as described below.
	 * \param[in] m
	 *     The number of rows of the matrix $A$. $\{m}\ge 0$. When $\{m}=0$, §dlange returns zero.
	 *
	 * \param[in] n
	 *     The number of columns of the matrix $A$. $\{n}\ge 0$.
	 *     When $\{n}=0$, §dlange returns zero.
	 *
	 * \param[in] A
	 *     an array, dimension (§lda,§n)\n
	 *     The §m by §n matrix $A$.
	 *
	 * \param[in] lda The leading dimension of the array §A. $\{lda}\ge\max(\{m},1)$.
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$), where $\{lwork}\ge\{m}$ when §norm = 'I';\n
	 *     otherwise, §work is not referenced.
	 *
	 * \return
	 * $\{dlange} = \left(\begin{array}{ll}
	 *     \max\left(\left|A[i,j]\right|\right), & \{norm}=\text{'M' or 'm'}           \\
	 *     \{norm1}(A),                          & \{norm}=\text{'1', 'O' or 'o'}      \\
	 *     \{normI}(A),                          & \{norm}=\text{'I' or 'i'}           \\
	 *     \{normF}(A),                          & \{norm}=\text{'F', 'f', 'E' or 'e'} \\
	 * \end{array} \right.$\n
	 * where §norm1 denotes the  one norm of a matrix (maximum column sum), §normI denotes the
	 * infinity norm of a matrix (maximum row sum) and §normF denotes the Frobenius norm of a
	 * matrix (square root of sum of squares). Note that $\max\left(\left|A[i,j]\right|\right)$ is
	 * not a consistent matrix norm.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static real dlange(char const* const norm, int const m, int const n, real const* const A,
	                   int const lda, real* const work) /*const*/
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
			//Find max(abs(A[i,j])).
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
			for (i=0; i<m; i++)
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

	/*! §dlansb returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or
	 *  the element of largest absolute value of a symmetric band matrix.
	 *
	 * §dlansb returns the value of the one-norm, or the Frobenius norm, or the infinity-norm, or
	 * the element of largest absolute value of an §n by §n symmetric band matrix $A$, with §k
	 * super-diagonals.
	 * \param[in] norm Specifies the value to be returned by §dlansb as described above.
	 * \param[in] uplo
	 *     Specifies whether the upper or lower triangular part of the band matrix $A$ is supplied.
	 *     \n ='U': Upper triangular part is supplied
	 *     \n ='L': Lower triangular part is supplied
	 *
	 * \param[in] n The order of the matrix $A$. $\{n}\ge 0$. When $\{n}=0$, §dlansb returns zero.
	 * \param[in] k
	 *     The number of super-diagonals or sub-diagonals of the band matrix $A$. $\{k}\ge 0$.
	 *
	 * \param[in] Ab
	 *     an array, dimension (§ldab,§n)\n
	 *     The upper or lower triangle of the symmetric band matrix $A$, stored in the first
	 *     $\{k}+1$ rows of §Ab. The $j$-th column of $A$ is stored in the $j$-th column of the
	 *     array §Ab as follows:\n
	 *         if §uplo ='U', $\{Ab}[\{k}+i-j,j] = A[i,j]$ for $\max(0,j-\{k})\le i\le j$;\n
	 *         if §uplo ='L', $\{Ab}[i-j,j]      = A[i,j]$ for $j\le i\le\min(\{n}-1,j+\{k})$.
	 *
	 * \param[in]  ldab The leading dimension of the array §Ab. $\{ldab}\ge\{k}+1$.
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$),\n
	 *     where $\{lwork}\ge\{n}$ when §norm ='I' or '1' or 'O';
	 *     Otherwise, §work is not referenced.
	 *
	 * \return
	 * $\left(\begin{tabular}{ll}
	 *     \(\max(|\{A}[i,j]|)\), & \{norm} = 'M' or 'm'          \\
	 *     \(\on{norm1}(\{A})\),  & \{norm} = '1', 'O' or 'o'     \\
	 *     \(\on{normI}(\{A})\),  & \{norm} = 'I' or 'i'          \\
	 *     \(\on{normF}(\{A})\),  & \{norm} = 'F', 'f', 'E' or 'e'
	 *  \end{tabular}\right.$\n
	 * where $\on{norm1}$ denotes the       one-norm of a matrix (maximum column sum),
	 *       $\on{normI}$ denotes the  infinity-norm of a matrix (maximum row sum) and
	 *       $\on{normF}$ denotes the Frobenius norm of a matrix (square root of sum of squares).\n
	 *       Note that $\max(|\{A}[i,j]|)$ is not a consistent matrix norm.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static real dlansb(char const* const norm, char const* const uplo, int const n, int const k,
	                   real const* const Ab, int const ldab, real* const work) /*const*/
	{
		int i, j, l, abj, stop;
		real sum, value=ZERO;
		char upnorm = std::toupper(norm[0]);
		char upuplo = std::toupper(uplo[0]);
		if (n==0)
		{
			value = ZERO;
		}
		else if (upnorm=='M')
		{
			// Find max(abs(A(i,j))).
			value = ZERO;
			if (upuplo=='U')
			{
				for (j=0; j<n; j++)
				{
					abj = ldab * j;
					for (i=std::max(k-j, 0); i<=k; i++)
					{
						sum = std::fabs(Ab[i+abj]);
						if (value < sum || std::isnan(sum))
						{
							value = sum;
						}
					}
				}
			}
			else
			{
				for (j=0; j<n; j++)
				{
					abj = ldab * j;
					stop = std::min(n-j, k+1);
					for (i=0; i<stop; i++)
					{
						sum = std::fabs(Ab[i+abj]);
						if (value < sum || std::isnan(sum))
						{
							value = sum;
						}
					}
				}
			}
		}
		else if (upnorm=='I' || upnorm=='O' || norm[0]=='1')
		{
			// Find normI(A) (= norm1(A), since A is symmetric).
			int labj;
			real absa;
			value = ZERO;
			if (upuplo=='U')
			{
				for (j=0; j<n; j++)
				{
					abj = ldab * j;
					sum = ZERO;
					l = k - j;
					labj = l + abj;
					for (i=std::max(0, j-k); i<j; i++)
					{
						absa = std::fabs(Ab[i+labj]);
						sum += absa;
						work[i] += absa;
					}
					work[j] = sum + std::fabs(Ab[k+abj]);
				}
				for (i=0; i<n; i++)
				{
					sum = work[i];
					if (value < sum || std::isnan(sum))
					{
						value = sum;
					}
				}
			}
			else
			{
				for (i=0; i<n; i++)
				{
					work[i] = ZERO;
				}
				for (j=0; j<n; j++)
				{
					abj = ldab * j;
					sum = work[j] + std::fabs(Ab[abj]);
					l = -j;
					labj = l + abj;
					stop = std::min(n-1, j+k);
					for (i=j+1; i<=stop; i++)
					{
						absa = std::fabs(Ab[i+labj]);
						sum += absa;
						work[i] += absa;
					}
					if (value < sum || std::isnan(sum))
					{
						value = sum;
					}
				}
			}
		}
		else if (upnorm=='F' || upnorm=='E')
		{
			// Find normF(A).
			real scale = ZERO;
			sum = ONE;
			if (k>0)
			{
				if (upuplo=='U')
				{
					for (j=1; j<n; j++)
					{
						dlassq(std::min(j, k), &Ab[std::max(k-j, 0)+ldab*j], 1, scale, sum);
					}
					l = k;
				}
				else
				{
					for (j=0; j<n-1; j++)
					{
						dlassq(std::min(n-1-j, k), &Ab[1+ldab*j], 1, scale, sum);
					}
					l = 0;
				}
				sum *= TWO;
			}
			else
			{
				l = 0;
			}
			dlassq(n, &Ab[l], ldab, scale, sum);
			value = scale * std::sqrt(sum);
		}
		return value;
	}

	/*! §dlansp returns the value of the 1-norm, or the Frobenius norm, or the infinity-norm, or
	 *  the element of largest absolute value of a symmetric matrix supplied in packed form.
	 *
	 * §dlansp returns the value of the one-norm, or the Frobenius norm, or the infinity-norm, or
	 * the element of largest absolute value of a real symmetric matrix $A$, supplied in packed
	 * form.
	 * \param[in] norm Specifies the value to be returned by §dlansp as described above.
	 * \param[in] uplo
	 *     Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is
	 *     supplied.\n
	 *     ='U': Upper triangular part of $A$ is supplied\n
	 *     ='L': Lower triangular part of $A$ is supplied
	 *
	 * \param[in] n  The order of the matrix $A$. $\{n}\ge 0$. When $\{n}=0$, §dlansp returns zero.
	 * \param[in] ap
	 *     an array, dimension ($\{n}(\{n}+1)/2$)\n
	 *     The upper or lower triangle of the symmetric matrix $A$, packed columnwise in a linear
	 *     array. The $j$-th column of $A$ is stored in the array §ap as follows:\n
	 *     if §uplo ='U', $\{ap}[i+j(j+1)/2]       = A[i,j]$ for $0\le i\le j$;\n
	 *     if §uplo ='L', $\{ap}[i+j(2\{n}-j-1)/2] = A[i,j]$ for $j\le i<\{n}$.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$),\n
	 *     where $\{lwork}\ge\{n}$ when §norm ='I' or '1' or 'O';
	 *     otherwise, §work is not referenced.
	 *
	 * \return
	 * $\left(\begin{tabular}{ll}
	 *     \(\max(|\{A}[i,j]|)\), & \{norm} = 'M' or 'm'          \\
	 *     \(\on{norm1}(\{A})\),  & \{norm} = '1', 'O' or 'o'     \\
	 *     \(\on{normI}(\{A})\),  & \{norm} = 'I' or 'i'          \\
	 *     \(\on{normF}(\{A})\),  & \{norm} = 'F', 'f', 'E' or 'e'
	 *  \end{tabular}\right.$\n
	 * where $\on{norm1}$ denotes the       one-norm of a matrix (maximum column sum),
	 *       $\on{normI}$ denotes the  infinity-norm of a matrix (maximum row sum) and
	 *       $\on{normF}$ denotes the Frobenius norm of a matrix (square root of sum of squares).\n
	 *       Note that $\max(|\{A}[i,j]|)$ is not a consistent matrix norm.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static real dlansp(char const* const norm, char const* const uplo, int const n,
	                   real const* const ap, real* const work) /*const*/
	{
		int i, j, k, stop;
		real absa, sum, value=ZERO;
		char upnorm = std::toupper(norm[0]);
		char upuplo = std::toupper(uplo[0]);
		if (n==0)
		{
			value = ZERO;
		}
		else if (upnorm=='M')
		{
			// Find max(abs(A(i,j))).
			value = ZERO;
			if (upuplo=='U')
			{
				k = 0;
				for (j=0; j<n; j++)
				{
					stop = k + j;
					for (i=k; i<=stop; i++)
					{
						sum = std::fabs(ap[i]);
						if (value < sum || std::isnan(sum))
						{
							value = sum;
						}
					}
					k += j + 1;
				}
			}
			else
			{
				k = 0;
				for (j=0; j<n; j++)
				{
					stop = k + n - j;
					for (i=k; i<stop; i++)
					{
						sum = std::fabs(ap[i]);
						if (value < sum || std::isnan(sum))
						{
							value = sum;
						}
					}
					k += n - j;
				}
			}
		}
		else if (upnorm=='I' || upnorm=='O' || norm[0]=='1')
		{
			// Find normI(A) (= norm1(A), since A is symmetric).
			value = ZERO;
			k = 0;
			if (upuplo=='U')
			{
				for (j=0; j<n; j++)
				{
					sum = ZERO;
					for (i=0; i<j; i++)
					{
						absa = std::fabs(ap[k]);
						sum     += absa;
						work[i] += absa;
						k++;
					}
					work[j] = sum + std::fabs(ap[k]);
					k++;
				}
				for (i=0; i<n; i++)
				{
					sum = work[i];
					if (value < sum || std::isnan(sum))
					{
						value = sum;
					}
				}
			}
			else
			{
				for (i=0; i<n; i++)
				{
					work[i] = ZERO;
				}
				for (j=0; j<n; j++)
				{
					sum = work[j] + std::fabs(ap[k]);
					k++;
					for (i=j+1; i<n; i++)
					{
						absa = std::fabs(ap[k]);
						sum     += absa;
						work[i] += absa;
						k++;
					}
					if (value < sum || std::isnan(sum))
					{
						value = sum;
					}
				}
			}
		}
		else if (upnorm=='F' || upnorm=='E')
		{
			// Find normF(A).
			real scale = ZERO;
			sum = ONE;
			k = 1;
			if (upuplo=='U')
			{
				for (j=1; j<n; j++)
				{
					dlassq(j, &ap[k], 1, scale, sum);
					k += j + 1;
				}
			}
			else
			{
				for (j=0; j<n-1; j++)
				{
					dlassq(n-j-1, &ap[k], 1, scale, sum);
					k += n - j;
				}
			}
			sum *= TWO;
			k = 0;
			real temp;
			for (i=0; i<n; i++)
			{
				if (ap[k]!=ZERO)
				{
					absa = std::fabs(ap[k]);
					if (scale<absa)
					{
						temp = scale / absa;
						sum = ONE + sum*temp*temp;
						scale = absa;
					}
					else
					{
						temp = absa / scale;
						sum += temp * temp;
					}
				}
				if (upuplo=='U')
				{
					k += i + 2;
				}
				else
				{
					k += n - i;
				}
			}
			value = scale * std::sqrt(sum);
		}
		return value;
	}

	/*! §dlanst returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or
	 *  the element of largest absolute value of a real symmetric tridiagonal matrix.
	 *
	 * §dlanst returns the value of the one norm, or the Frobenius norm, or the  infinity norm, or
	 * the element of largest absolute value of a real symmetric tridiagonal matrix $A$.\n
	 *     $\{dlanst}=\left(\begin{tabular}{ll}
	 *         \(\max(|A[i,j]|)\), & \{norm} = 'M' or 'm'           \\
	 *         \(\on{norm1}(A)\),  & \{norm} = '1', 'O' or 'o'      \\
	 *         \(\on{normI}(A)\),  & \{norm} = 'I' or 'i'           \\
	 *         \(\on{normF}(A)\),  & \{norm} = 'F', 'f', 'E' or 'e'
	 *     \end{tabular}\right.$\n
	 * where $\on{norm1}$ denotes the one norm of a matrix (maximum column sum),
	 * $\on{normI}$ denotes the infinity norm  of a matrix (maximum row sum) and
	 * $\on{normF}$ denotes the Frobenius norm of a matrix (square root of sum of
	 * squares).\n Note that $\max(|A[i,j]|)$ is not a consistent matrix norm.
	 * \param[in] norm Specifies the value to be returned in §dlanst as described above.
	 * \param[in] n
	 *     The order of the matrix $A$. $\{n}\ge 0$. When §n = 0, §dlanst is set to zero.
	 *
	 * \param[in] d an array, dimension (§n)\n The diagonal elements of $A$.
	 * \param[in] e
	 *     an array, dimension ($\{n}-1$)\n
	 *     The ($\{n}-1$) sub-diagonal or super-diagonal elements of $A$.
	 * \return The norm of $A$ specified by §norm.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static real dlanst(char const* const norm, int const n, real const* const d,
	                   real const* const e) /*const*/
	{
		int i;
		real anorm=ZERO, sum;
		char upnorm = std::toupper(norm[0]);
		if (n<=0)
		{
			anorm = ZERO;
		}
		else if (upnorm=='M')
		{
			// Find max(abs(A[i,j])).
			anorm = std::fabs(d[n-1]);
			for (i=0; i<n-1; i++)
			{
				sum = std::fabs(d[i]);
				if (anorm < sum || std::isnan(sum))
				{
					anorm = sum;
				}
				sum = std::fabs(e[i]);
				if (anorm < sum || std::isnan(sum))
				{
					anorm = sum;
				}
			}
		}
		else if (upnorm=='O' || norm[0]=='1' || upnorm=='I')
		{
			// Find norm1(A).
			if (n==1)
			{
				anorm = std::fabs(d[0]);
			}
			else
			{
				anorm = std::fabs(d[0])   + std::fabs(e[0]);
				sum   = std::fabs(e[n-2]) + std::fabs(d[n-1]);
				if (anorm < sum || std::isnan(sum))
				{
					anorm = sum;
				}
				for (i=1; i<n-1; i++)
				{
					sum = std::fabs(d[i]) + std::fabs(e[i]) + std::fabs(e[i-1]);
					if (anorm < sum || std::isnan(sum))
					{
						anorm = sum;
					}
				}
			}
		}
		else if (upnorm=='F' || upnorm=='E')
		{
			// Find normF(A).
			real scale = ZERO;
			sum = ONE;
			if (n>1)
			{
				dlassq(n-1, e, 1, scale, sum);
				sum *= TWO;
			}
			dlassq(n, d, 1, scale, sum);
			anorm = scale * std::sqrt(sum);
		}
		return anorm;
	}

	/*! §dlansy returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or
	 *  the element of largest absolute value of a real symmetric matrix.
	 *
	 * §dlansy returns the value of the one-norm, or the Frobenius norm, or the infinity-norm, or
	 * the element of largest absolute value of a real symmetric matrix $A$.
	 * \param[in] norm Specifies the value to be returned in §dlansy as described below.
	 * \param[in] uplo
	 *     Specifies whether the upper or lower triangular part of the symmetric matrix $A$ is to
	 *     be referenced.\n
	 *         ='U': Upper triangular part of $A$ is referenced\n
	 *         ='L': Lower triangular part of $A$ is referenced
	 *
	 * \param[in] n
	 *     The order of the matrix $A$. $\{n}\ge 0$. When $\{n}=0$, §dlansy is set to zero.
	 *
	 * \param[in] A
	 *     an array, dimension (§lda,§n)\n
	 *     The symmetric matrix $A$.\n
	 *     If §uplo ='U', the leading §n by §n upper triangular part of §A contains the upper
	 *     triangular part of the matrix $A$, and the strictly lower triangular part of §A is not
	 *     referenced.\n
	 *     If §uplo ='L', the leading §n by §n lower triangular part of §A contains the lower
	 *     triangular part of the matrix $A$, and the strictly upper triangular part of §A is not
	 *     referenced.
	 *
	 * \param[in] lda The leading dimension of the array §A. $\{lda}\ge\max(\{n},1)$.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$),\n
	 *     where $\{lwork}\ge\{n}$ when §norm ='I' or '1' or 'O';
	 *     otherwise, §work is not referenced.
	 *
	 * \return
	 * $\left(\begin{tabular}{ll}
	 *     \(\max(|\{A}[i,j]|)\), & \{norm} = 'M' or 'm'          \\
	 *     \(\on{norm1}(\{A})\),  & \{norm} = '1', 'O' or 'o'     \\
	 *     \(\on{normI}(\{A})\),  & \{norm} = 'I' or 'i'          \\
	 *     \(\on{normF}(\{A})\),  & \{norm} = 'F', 'f', 'E' or 'e'
	 *  \end{tabular}\right.$\n
	 * where $\on{norm1}$ denotes the one-norm of a matrix      (maximum column sum),
	 *       $\on{normI}$ denotes the infinity-norm of a matrix (maximum row sum) and
	 *       $\on{normF}$ denotes the Frobenius norm of a matrix (square root of sum of squares).\n
	 *       Note that $\max(|\{A}[i,j]|)$ is not a consistent matrix norm.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static real dlansy(char const* const norm, char const* const uplo, int const n,
	                   real const* const A, int const lda, real* const work) /*const*/
	{
		int i, j, aj;
		real sum, value=ZERO;
		char upnorm = std::toupper(norm[0]);
		char upuplo = std::toupper(uplo[0]);
		if (n==0)
		{
			value = ZERO;
		}
		else if (upnorm=='M')
		{
			// Find max(abs(A[i,j])).
			value = ZERO;
			if (upuplo=='U')
			{
				for (j=0; j<n; j++)
				{
					aj = lda * j;
					for (i=0; i<=j; i++)
					{
						sum = std::fabs(A[i+aj]);
						if (value<sum || std::isnan(sum))
						{
							value = sum;
						}
					}
				}
			}
			else
			{
				for (j=0; j<n; j++)
				{
					aj = lda * j;
					for (i=j; i<n; i++)
					{
						sum = std::fabs(A[i+aj]);
						if (value<sum || std::isnan(sum))
						{
							value = sum;
						}
					}
				}
			}
		}
		else if (upnorm=='I' || upnorm=='O' || norm[0]=='1')
		{
			// Find normI(A) (= norm1(A), since A is symmetric).
			real absa;
			value = ZERO;
			if (upuplo=='U')
			{
				for (j=0; j<n; j++)
				{
					aj = lda * j;
					sum = ZERO;
					for (i=0; i<j; i++)
					{
						absa = std::fabs(A[i+aj]);
						sum += absa;
						work[i] += absa;
					}
					work[j] = sum + std::fabs(A[j+aj]);
				}
				for (i=0; i<n; i++)
				{
					sum = work[i];
					if (value<sum || std::isnan(sum))
					{
						value = sum;
					}
				}
			}
			else
			{
				for (i=0; i<n; i++)
				{
					work[i] = ZERO;
				}
				for (j=0; j<n; j++)
				{
					aj = lda * j;
					sum = work[j] + std::fabs(A[j+aj]);
					for (i=j+1; i<n; i++)
					{
						absa = std::fabs(A[i+aj]);
						sum += absa;
						work[i] += absa;
					}
					if (value<sum || std::isnan(sum))
					{
						value = sum;
					}
				}
			}
		}
		else if (upnorm=='F' || upnorm=='E')
		{
			// Find normF(A).
			real scale = ZERO;
			sum = ONE;
			if (upuplo=='U')
			{
				for (j=1; j<n; j++)
				{
					dlassq(j, &A[lda*j], 1, scale, sum);
				}
			}
			else
			{
				for (j=0; j<n-1; j++)
				{
					dlassq(n-j-1, &A[j+1+lda*j], 1, scale, sum);
				}
			}
			sum *= TWO;
			dlassq(n, A, lda+1, scale, sum);
			value = scale * std::sqrt(sum);
		}
		return value;
	}

	/*! §dlanv2 computes the Schur factorization of a real 2 by 2 nonsymmetric matrix in standard
	 *  form.
	 *
	 * §dlanv2 computes the Schur factorization of a real 2 by 2 nonsymmetric matrix in standard
	 * form:\n
	 *     $\b{bm} \{a} & \{b} \\
	 *             \{c} & \{d} \e{bm} = \b{bm} \{cs} & -\{sn} \\
	 *                                         \{sn} &  \{cs} \e{bm}
	 *                                     \b{bm} aa & bb \\
	 *                                            cc & dd \e{bm}\b{bm} \{cs} & \{sn} \\
	 *                                                                -\{sn} & \{cs} \e{bm}$\n
	 * where either
	 * \li $cc = 0$ so that $aa$ and $dd$ are real eigenvalues of the matrix, or
	 * \li $aa = dd$ and $bb * cc < 0$, so that $aa \pm \sqrt{bb * cc}$ are complex conjugate
	 *     eigenvalues.
	 *
	 * \param[in,out] a, b, c, d
	 *     On entry, the elements of the input matrix.\n
	 *     On exit, they are overwritten by the elements of the standardised Schur form.
	 *
	 * \param[out] rt1r, rt1i, rt2r, rt2i
	 *     The real and imaginary parts of the eigenvalues.
	 *     If the eigenvalues are a complex conjugate pair, $\{rt1i}>0$.
	 *
	 * \param[out] cs, sn Parameters of the rotation matrix.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Modified by V. Sima, Research Institute for Informatics, Bucharest, Romania, to reduce
	 *     the risk of cancellation errors, when computing real eigenvalues, and to ensure,
	 *     if possible, that $|\{rt1r}| \ge |\{rt2r}|$.                                          */
	static void dlanv2(real& a, real& b, real& c, real& d, real& rt1r, real& rt1i, real& rt2r,
	                   real& rt2i, real& cs, real& sn) /*const*/
	{
		const real MULTPL = FOUR;
		real aa, bb, bcmax, bcmis, cc, cs1, dd, eps, p, sab, sac, scale, sigma, sn1, tau, temp, z;
		eps = dlamch("P");
		if (c==ZERO)
		{
			cs = ONE;
			sn = ZERO;
		}
		else if (b==ZERO)
		{
			// Swap rows and columns
			cs   = ZERO;
			sn   = ONE;
			temp = d;
			d    = a;
			a    = temp;
			b    = -c;
			c    = ZERO;
		}
		else if ((a-d)==ZERO && std::copysign(ONE, b)!=std::copysign(ONE, c))
		{
			cs = ONE;
			sn = ZERO;
		}
		else
		{
			temp  = a - d;
			p     = HALF * temp;
			bcmax = std::max(std::fabs(b), std::fabs(c));
			bcmis = std::min(std::fabs(b), std::fabs(c))
			        *std::copysign(ONE, b) * std::copysign(ONE, c);
			scale = std::max(std::fabs(p), bcmax);
			z     = (p/scale)*p + (bcmax/scale)*bcmis;
			// If z is of the order of the machine accuracy,
			// postpone the decision on the nature of eigenvalues
			if (z>=MULTPL*eps)
			{
				// Real eigenvalues. Compute a and d.
				z  = p + std::copysign(std::sqrt(scale)*std::sqrt(z), p);
				a  = d + z;
				d -= (bcmax/z) * bcmis;
				// Compute b and the rotation matrix
				tau = dlapy2(c, z);
				cs  = z / tau;
				sn  = c / tau;
				b  -= c;
				c   = ZERO;
			}
			else
			{
				// Complex eigenvalues, or real (almost) equal eigenvalues.
				// Make diagonal elements equal.
				sigma = b + c;
				tau   = dlapy2(sigma, temp);
				cs    = std::sqrt(HALF*(ONE+std::fabs(sigma)/tau));
				sn    = -(p/(tau*cs)) * std::copysign(ONE, sigma);
				// Compute [ aa  bb ] = [ a  b ] [ cs -sn ]
				//         [ cc  dd ]   [ c  d ] [ sn  cs ]
				aa =  a*cs + b*sn;
				bb = -a*sn + b*cs;
				cc =  c*cs + d*sn;
				dd = -c*sn + d*cs;
				// Compute [ a  b ] = [ cs  sn ] [ aa  bb ]
				//         [ c  d ]   [-sn  cs ] [ cc  dd ]
				a =  aa*cs + cc*sn;
				b =  bb*cs + dd*sn;
				c = -aa*sn + cc*cs;
				d = -bb*sn + dd*cs;
				temp = HALF * (a+d);
				a    = temp;
				d    = temp;
				if (c!=ZERO)
				{
					if (b!=ZERO)
					{
						if (std::copysign(ONE, b)==std::copysign(ONE, c))
						{
							// Real eigenvalues: reduce to upper triangular form
							sab  = std::sqrt(std::fabs(b));
							sac  = std::sqrt(std::fabs(c));
							p    = std::copysign(sab*sac, c);
							tau  = ONE / std::sqrt(std::fabs(b+c));
							a    = temp + p;
							d    = temp - p;
							b   -= c;
							c    = ZERO;
							cs1  = sab * tau;
							sn1  = sac * tau;
							temp = cs*cs1 - sn*sn1;
							sn   = cs*sn1 + sn*cs1;
							cs   = temp;
						}
					}
					else
					{
						b    = -c;
						c    = ZERO;
						temp = cs;
						cs   = -sn;
						sn   = temp;
					}
				}
			}
		}
		// Store eigenvalues in (rt1r,rt1i) and (rt2r,rt2i).
		rt1r = a;
		rt2r = d;
		if (c==ZERO)
		{
			rt1i = ZERO;
			rt2i = ZERO;
		}
		else
		{
			rt1i = std::sqrt(std::fabs(b))*std::sqrt(std::fabs(c));
			rt2i = -rt1i;
		}
	}

	/*! §dlapy2 returns $\sqrt{x^2+y^2}$.
	 *
	 * §dlapy2 returns $\sqrt{x^2+y^2}$, taking care not to cause unnecessary overflow.
	 * \param[in] x, y §x and §y specify the values $x$ and $y$.
	 * \return $\sqrt{x^2+y^2}$, or NaN if either §x or §y is NaN.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017                                                                           */
	static real dlapy2(real const x, real const y) /*const*/
	{
		if (std::isnan(x))
		{
			return x;
		}
		else if (std::isnan(y))
		{
			return y;
		}
		else
		{
			real w, xabs, yabs, z;
			xabs = std::fabs(x);
			yabs = std::fabs(y);
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
				real temp = z / w;
				return w * std::sqrt(ONE + temp*temp);
			}
		}
	}

	/*! §dlaqp2 computes a QR factorization with column pivoting of the matrix block.
	 *
	 * §dlaqp2 computes a QR factorization with column pivoting of the block
	 * $A[\{offset}:\{m}-1,0:\{n}-1]$. The block $A[0:\{offset}-1,0:\{n}-1]$ is accordingly
	 * pivoted, but not factorized.
	 * \param[in] m      The number of rows of the matrix $A$. $\{m}\ge 0$.
	 * \param[in] n      The number of columns of the matrix $A$. $\{n}\ge 0$.
	 * \param[in] offset The number of rows of the matrix $A$ that must be pivoted but not
	 *                     factorized. offset>=0.
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the §m by §n matrix $A$.\n
	 *     On exit, the upper triangle of block $A[\{offset}:\{m}-1,0:\{n}-1]$ is the triangular
	 *              factor obtained; the elements in block $A[\{offset}:\{m}-1,0:\{n}-1]$ below the
	 *              diagonal, together with the array §tau, represent the orthogonal matrix $Q$ as
	 *              a product of elementary reflectors. Block $A[0:\{offset}-1,0:\{n}-1]$ has been
	 *              accordingly pivoted,but not factorized.
	 *
	 * \param[in]     lda  The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[in,out] jpvt
	 *     an integer array, dimension (§n)\n
	 *     On entry,    if $\{jpvt}[i]\ne -1$, the $i$-th column of $A$ is permuted to the front of
	 *                  $AP$ (a leading column);\n
	 *     &emsp;&emsp; if $\{jpvt}[i]=-1$, the $i$-th column of $A$ is a free column.\n
	 *     On exit, if $\{jpvt}[i]=k$, then the $i$-th column of $AP$ was the $k$-th column of
	 *              $A$.\n
	 *     Note: This array contains zero-based indices.
	 *
	 * \param[out] tau
	 *     an array, dimension ($\min(\{m},\{n})$)\n
	 *     The scalar factors of the elementary reflectors.
	 *
	 * \param[in,out] vn1
	 *     an array, dimension (§n)\n
	 *     The vector with the partial column norms.
	 *
	 * \param[in,out] vn2
	 *     an array, dimension (§n)\n
	 *     The vector with the exact column norms.
	 *
	 * \param[out] work an array, dimension (§n)
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 * Contributors:\n
	 *     G. Quintana-Orti, Depto. de Informatica, Universidad Jaime I, Spain X. Sun, Computer
	 *     Science Dept., Duke University, USA\n
	 *     Partial column norm updating strategy modified on April 2011 Z. Drmac and Z. Bujanovic,
	 *     Dept. of Mathematics, University of Zagreb, Croatia.\n\n
	 * References:\n
	 *     LAPACK Working Note 176
	 *     <a href="http://www.netlib.org/lapack/lawnspdf/lawn176.pdf">[PDF]</a>                 */
	static void dlaqp2(int const m, int const n, int const offset, real* const A, int const lda,
	                   int* const jpvt, real* const tau, real* const vn1, real* const vn2,
	                   real* const work) /*const*/
	{
		int mn = std::min(m-offset, n);
		real tol3z = std::sqrt(dlamch("Epsilon"));
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
				itemp     = jpvt[pvt];
				jpvt[pvt] = jpvt[i];
				jpvt[i]   = itemp;
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
			if (i<n-1)
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
					// NOTE: The following 6 lines follow from the analysis in
					// Lapack Working Note 176.
					temp = std::fabs(A[offpi+lda*j]) / vn1[j];
					temp = ONE - temp*temp;
					temp = ((temp>ZERO) ? temp : ZERO);
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
						vn1[j] *= std::sqrt(temp);
					}
				}
			}
		}
	}

	/*! §dlaqps computes a step of QR factorization with column pivoting of a real §m by §n matrix
	 *  $A$ by using BLAS level 3.
	 *
	 * §dlaqps computes a step of QR factorization with column pivoting of a real §m by §n matrix
	 * $A$ by using Blas-3. It tries to factorize §nb columns from $A$ starting from the row
	 * §offset+1, and updates all of the matrix with Blas-3 §xgemm. In some cases, due to
	 * catastrophic cancellations, it cannot factorize §nb columns. Hence, the actual number of
	 * factorized columns is returned in §kb. Block $A[0:\{offset}-1,0:\{n}-1]$ is accordingly
	 * pivoted, but not factorized.
	 * \param[in]     m      The number of rows of the matrix $A$. $\{m}\ge 0$.
	 * \param[in]     n      The number of columns of the matrix $A$. $\{n}\ge 0$.
	 * \param[in]     offset The number of rows of $A$ that have been factorized in previous steps.
	 * \param[in]     nb     The number of columns to factorize.
	 * \param[out]    kb     The number of columns actually factorized.
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the §m by §n matrix $A$.\n
	 *     On exit, block $A[\{offset}:\{m}-1,0:\{kb}-1]$ is the triangular factor obtained and
	 *              block $A[0:\{offset}-1,0:\{n}-1]$ has been accordingly pivoted,
	 *              but not factorized. The rest of the matrix, block
	 *              $A[\{offset}:\{m}-1, \{kb}:\{n}-1]$ has been updated.
	 *
	 * \param[in]     lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[in,out] jpvt
	 *     an integer array, dimension (§n)
	 *     $\{jpvt}[i]=k \Leftrightarrow$ Column $k$ of the full matrix $A$ has been permuted into
	 *     position $i$ in $AP$.\n
	 *     Note: this array contains zero-based indices
	 *
	 * \param[out] tau
	 *     an array, dimension (§kb)\n
	 *     The scalar factors of the elementary reflectors.
	 *
	 * \param[in,out] vn1 an array, dimension (§n)\n The vector with the partial column norms.
	 * \param[in,out] vn2 an array, dimension (§n)\n The vector with the exact column norms.
	 * \param[in,out] auxv an array, dimension (§nb)\n Auxiliary vector.
	 * \param[in,out] F an array, dimension (§ldf,§nb)\n Matrix $F^T = L Y^T A$.
	 * \param[in]     ldf The leading dimension of the array §F. $\{ldf}\ge\max(1,\{n})$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 * Contributors:\n
	 *     G. Quintana-Orti, Depto. de Informatica, Universidad Jaime I, Spain X. Sun, Computer
	 *     Science Dept., Duke University, USA\n
	 *     Partial column norm updating strategy modified on April 2011 Z. Drmac and Z. Bujanovic,
	 *     Dept. of Mathematics, University of Zagreb, Croatia.\n\n
	 * References:\n
	 *     LAPACK Working Note 176
	 *     <a href="http://www.netlib.org/lapack/lawnspdf/lawn176.pdf">[PDF]</a>                 */
	static void dlaqps(int const m, int const n, int const offset, int const nb, int& kb,
	                   real* const A, int const lda, int* const jpvt, real* const tau,
	                   real* const vn1, real* const vn2, real* const auxv, real* const F,
	                   int const ldf) /*const*/
	{
		int lastrk = std::min(m, n+offset) - 1;
		int lsticc = -1;
		int k = -1;
		real tol3z = std::sqrt(dlamch("Epsilon"));
		// Beginning of while loop.
		int itemp, j, pvt, rk, acolk;
		real akk, temp, temp2;
		while (k<nb-1 && lsticc==-1)
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
				itemp     = jpvt[pvt];
				jpvt[pvt] = jpvt[k];
				jpvt[k]   = itemp;
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
			if (rk<lastrk)
			{
				for (j=k+1; j<n; j++)
				{
					if (vn1[j]!=ZERO)
					{
						// NOTE: The following 6 lines follow from the analysis in Lapack Working
						//       Note 176.
						temp = std::fabs(A[rk+lda*j]) / vn1[j];
						temp = (ONE+temp)*(ONE-temp);
						temp = ((ZERO>temp) ? ZERO : temp);
						temp2 = vn1[j] / vn2[j];
						temp2 = temp * temp2 * temp2;
						if (temp2<=tol3z)
						{
							vn2[j] = real(lsticc+1);
							lsticc = j;
						}
						else
						{
							vn1[j] *= std::sqrt(temp);
						}
					}
				}
			}
			A[rk+acolk] = akk;
		}
		kb = k + 1;
		rk = offset + kb;
		// Apply the block reflector to the rest of the matrix:
		// A[offset+kb:m-1, kb:n-1] -= A[offset+kb:m-1, 0:kb-1] * F[kb:n-1, 0:kb-1]^T.
		if (kb < std::min(n, m-offset))
		{
			Blas<real>::dgemm("No transpose", "Transpose", m-rk, n-kb, kb, -ONE, &A[rk], lda,
			                  &F[kb], ldf, ONE, &A[rk+lda*kb], lda);
		}
		// Recomputation of difficult columns.
		while (lsticc>=0)
		{
			itemp = std::round(vn2[lsticc]) - 1;
			vn1[lsticc] = Blas<real>::dnrm2(m-rk, &A[rk+lda*lsticc], 1);
			// NOTE: The computation of vn1[lsticc] relies on the fact that dnrm2 does not fail on
			// vectors with norm below the value of sqrt(dlamch("S"))
			vn2[lsticc] = vn1[lsticc];
			lsticc = itemp;
		}
	}

	/*! §dlaqr0 computes the eigenvalues of a Hessenberg matrix, and optionally the matrices from
	 *  the Schur decomposition.
	 *
	 * §dlaqr0 computes the eigenvalues of a Hessenberg matrix $H$ and, optionally, the matrices
	 * $T$ and $Z$ from the Schur decomposition $H=ZTZ^T$, where $T$ is an upper quasi-triangular
	 * matrix (the Schur form), and $Z$ is the orthogonal matrix of Schur vectors.\n
	 * Optionally $Z$ may be postmultiplied into an input orthogonal matrix $Q$ so that this
	 * routine can give the Schur factorization of a matrix $A$ which has been reduced to the
	 * Hessenberg form $H$ by the orthogonal matrix $Q$: $A=QHQ^T=(QZ)T(QZ)^T$.
	 * \param[in] wantt
	 *     = §true: the full Schur form $T$ is required;\n
	 *     = §false: only eigenvalues are required.
	 *
	 * \param[in] wantz
	 *     = §true: the matrix of Schur vectors $Z$ is required;\n
	 *     = §false: Schur vectors are not required.
	 *
	 * \param[in] n        The order of the matrix $H$. $\{n}\ge 0$.
	 * \param[in] ilo, ihi
	 *     It is assumed that §H is already upper triangular in rows and columns $0:\{ilo}-1$ and
	 *     $\{ihi}+1:{n}-1$ and, if $\{ilo}>0$, $\{H}[\{ilo},\{ilo}-1]$ is zero. §ilo and §ihi are
	 *     normally set by a previous call to §dgebal, and then passed to §dgehrd when the matrix
	 *     output by §dgebal is reduced to Hessenberg form. Otherwise, §ilo and §ihi should be set
	 *     to 0 and $\{n}-1$, respectively. If $\{n}>0$, then $0\le\{ilo}\le\{ihi}<\{n}$.
	 *     If $\{n}=0$, then $\{ilo}=0$ and $\{ihi}=-1$.\n
	 *     NOTE: Zero-based indices!
	 *
	 * \param[in,out] H
	 *     an array, dimension (§ldh,§n)\n
	 *     On entry, the upper Hessenberg matrix $H$.\n
	 *     On exit, if $\{info}=0$ and §wantt is §true, then §H contains the upper quasi-triangular
	 *     matrix $T$ from the Schur decomposition (the Schur form); 2 by 2 diagonal blocks
	 *     (corresponding to complex conjugate pairs of eigenvalues) are returned in standard form,
	 *     with $\{H}[i,i]=\{H}[i+1,i+1]$ and $\{H}[i+1,i]\{H}[i,i+1]<0$. If $\{info}=0$ and §wantt
	 *     is §false, then the contents of §H are unspecified on exit. (The output value of §H when
	 *     $\{info}>0$ is given under the description of §info below.)\n
	 *     This subroutine may explicitly set $\{H}[i,j]=0$ for $i>j$ and $j=0,1,\ldots,\{ilo}-1$
	 *     or $j=\{ihi}+1,\{ihi}+2,\ldots,\{n}-1$.
	 *
	 * \param[in]  ldh    The leading dimension of the array §H. $\{ldh}\ge\max(1,\{n})$.
	 * \param[out] wr, wi
	 *     arrays, dimension ($\{ihi}+1$)\n
	 *     The real and imaginary parts, respectively, of the computed eigenvalues of
	 *     $\{H}[\{ilo}:\{ihi},\{ilo}:\{ihi}]$ are stored in $\{wr}[\{ilo}:\{ihi}]$ and
	 *     $\{wi}[\{ilo}:\{ihi}]$. If two eigenvalues are computed as a complex conjugate pair,
	 *     they are stored in consecutive elements of §wr and §wi, say the $i$-th and $i+1$-th,
	 *     with $\{wi}[i]>0$ and $\{wi}[i+1]<0$. If §wantt is §true, then the eigenvalues are
	 *     stored in the same order as on the diagonal of the Schur form returned in §H, with
	 *     $\{wr}[i]=\{H}[i,i]$ and, if $\{H}[i:i+1,i:i+1]$ is a 2 by 2 diagonal block,
	 *     $\{wi}[i]=\sqrt{-\{H}[i+1,i]\{H}[i,i+1]}$ and $\{wi}[i+1]=-\{wi}[i]$.
	 *
	 * \param[in] iloz, ihiz
	 *     Specify the rows of §Z to which transformations must be applied if §wantz is §true.\n
	 *     $0\le\{iloz}\le\{ilo}$; $\{ihi}\le\{ihiz}<\{n}$.\n
	 *     NOTE: Zero-based indices!
	 *
	 * \param[in,out] Z
	 *     an array, dimension (§ldz,$\{ihi}+1$)\n
	 *     If §wantz is §false, then §Z is not referenced.\
	 *     If §wantz is §true, then $\{Z}[\{ilo}:\{ihi},\{iloz}:\{ihiz}]$ is replaced by
	 *     $\{Z}[\{ilo}:\{ihi},\{iloz}:\{ihiz}]U$ where $U$ is the orthogonal Schur factor of
	 *     $\{H}[\{ilo}:\{ihi},\{ilo}:\{ihi}]$. (The output value of §Z when $\{info}>0$ is given
	 *     under the description of §info below.)
	 *
	 * \param[in] ldz
	 *     The leading dimension of the array §Z.
	 *     If §wantz is §true then $\{ldz}>\max(0,\{ihiz})$. Otherwize, $\{ldz}\ge 1$.
	 *
	 * \param[out] work
	 *     an array, dimension §lwork \n
	 *     On exit, if $\{lwork}=-1$, $\{work}[0]$ returns an estimate of the optimal value for
	 *     §lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work. $\{lwork}\ge\max(1,\{n})$ is sufficient, but §lwork
	 *     typically as large as $6\{n}$ may be required for optimal performance. A workspace query
	 *     to determine the optimal workspace size is recommended.\n
	 *     If $\{lwork}=-1$, then §dlaqr0 does a workspace query. In this case, §dlaqr0 checks the
	 *     input parameters and estimates the optimal workspace size for the given values of §n,
	 *     §ilo and §ihi. The estimate is returned in $\{work}[0]$. No error message related to
	 *     §lwork is issued by §xerbla. Neither §H nor §Z are accessed.
	 *
	 * \param[out] info
	 *     - =0: successful exit
	 *     - >0: - if $\{info}=i$, §dlaqr0 failed to compute all of the eigenvalues. Elements
	 *             $0:\{ilo}-1$ and $i:\{n}-1$ of §wr and §wi contain those eigenvalues which have
	 *             been successfully computed. (Failures are rare.)
	 *           - If $\{info}>0$ and §wantt is §false, then on exit, the remaining unconverged
	 *             eigenvalues are the eigenvalues of the upper Hessenberg matrix rows and columns
	 *             §ilo through $\{info}-1$ of the final, output value of §H.
	 *           - If $\{info}>0$ and §wantt is §true, then on exit\n
	 *                 $(\text{initial value of }\{H})U=U(\text{final value of }\{H})$    (*)\n
	 *             where $U$ is an orthogonal matrix. The final value of §H is upper Hessenberg and
	 *             quasi-triangular in rows and columns §info through §ihi.
	 *           - If $\{info}>0$ and §wantz is §true, then on exit\n
	 *                 $(\text{final value of }\{Z}[\{ilo}:\{ihi},\{iloz}:\{ihiz}])
	 *                           =(\text{initial value of }\{Z}[\{ilo}:\{ihi},\{iloz}:\{ihiz}])U$\n
	 *             where $U$ is the orthogonal matrix in (*) (regardless of the value of §wantt.)
	 *           - If $\{info}>0$ and §wantz is §false, then §Z is not accessed.
	 *
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Contributors:\n
	 *         Karen Braman and Ralph Byers, Department of Mathematics, University of Kansas, USA\n
	 *     \n References:\n
	 *         K. Braman, R. Byers and R. Mathias, The Multi-Shift QR Algorithm Part I: Maintaining
	 *         Well Focused Shifts, and Level 3 Performance, SIAM Journal of Matrix Analysis,
	 *         volume 23, pages 929--947, 2002.\n
	 *         K. Braman, R. Byers and R. Mathias, The Multi-Shift QR Algorithm Part II: Aggressive
	 *         Early Deflation, SIAM Journal of Matrix Analysis, volume 23, pages 948--973, 2002.*/
	static void dlaqr0(bool const wantt, bool const wantz, int const n, int const ilo,
	                   int const ihi, real* const H, int const ldh, real* const wr, real* const wi,
	                   int const iloz, int const ihiz, real* const Z, int const ldz,
	                   real* const work, int const lwork, int& info) /*const*/
	{
		// Matrices of order NTINY or smaller must be processed by dlahqr because of insufficient
		// subdiagonal scratch space. (This is a hard limit.)
		int const NTINY = 11;
		// Exceptional deflation windows: try to cure rare slow convergence by varying the size of
		// the deflation window after KEXNW iterations.
		int const KEXNW = 5;
		// Exceptional shifts: try to cure rare slow convergence with ad-hoc exceptional shifts
		// every KEXSH iterations.
		int const KEXSH = 6;
		// The constants WILK1 and WILK2 are used to form the exceptional shifts.
		real const WILK1 = real(0.75);
		real const WILK2 = real(-0.4375);
		info = 0;
		// Quick return for n = 0: nothing to do.
		if (n==0)
		{
			work[0] = ONE;
			return;
		}
		int lwkopt;
		if (n<=NTINY)
		{
			// Tiny matrices must use dlahqr.
			lwkopt = 1;
			if (lwork!=-1)
			{
				dlahqr(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, iloz, ihiz, Z, ldz, info);
			}
		}
		else
		{
			// Use small bulge multi-shift QR with aggressive early deflation on larger-than-tiny
			// matrices.
			// Hope for the best.
			info = 0;
			//Set up job flags for ilaenv.
			char jbcmpz[3];
			if (wantt)
			{
				jbcmpz[0] = 'S';
			}
			else
			{
				jbcmpz[0] = 'E';
			}
			if (wantz)
			{
				jbcmpz[1] = 'V';
			}
			else
			{
				jbcmpz[1] = 'N';
			}
			jbcmpz[2] = '\0';
			/* nwr = recommended deflation window size. At this point, n > NTINY = 11, so there is
			 * enough subdiagonal workspace for nwr>=2 as required. (In fact, there is enough
			 * subdiagonal space for nwr>=3.)                                                    */
			int nwr = ilaenv(13, "DLAQR0", jbcmpz, n, ilo+1, ihi+1, lwork);
			nwr = std::max(2, nwr);
			nwr = std::min(ihi-ilo+1, std::min((n-1)/3, nwr));
			/* nsr = recommended number of simultaneous shifts. At this point n > NTINY = 11, so
			 * there is at least enough subdiagonal workspace for nsr to be even and greater than
			 * or equal to two as required.                                                      */
			int nsr = ilaenv(15, "DLAQR0", jbcmpz, n, ilo+1, ihi+1, lwork);
			nsr = std::min(nsr, std::min((n+6)/9, ihi-ilo));
			nsr = std::max(2, nsr-(nsr%2));
			// Estimate optimal workspace
			// Workspace query call to dlaqr3
			int ld, ls;
			dlaqr3(wantt, wantz, n, ilo, ihi, nwr+1, H, ldh, iloz, ihiz, Z, ldz, ls, ld, wr, wi, H,
			       ldh, n, H, ldh, n, H, ldh, work, -1);
			// Optimal workspace = std::max(dlaqr5, dlaqr3) ====
			lwkopt = std::max(3*nsr/2, int(work[0]));
			// Quick return in case of workspace query.
			if (lwork==-1)
			{
				work[0] = real(lwkopt);
				return;
			}
			// dlahqr/dlaqr0 crossover point
			int nmin = std::max(NTINY, ilaenv(12, "DLAQR0", jbcmpz, n, ilo+1, ihi+1, lwork));
			// Nibble crossover point
			int nibble = std::max(0, ilaenv(14, "DLAQR0", jbcmpz, n, ilo+1, ihi+1, lwork));
			// Accumulate reflections during ttswp?
			// Use block 2 by 2 structure during matrix-matrix multiply?
			int kacc22 = std::max(0, ilaenv(16, "DLAQR0", jbcmpz, n, ilo+1, ihi+1, lwork));
			kacc22 = std::min(2, kacc22);
			// nwmax = the largest possible deflation window for which there is sufficient
			// workspace.
			int nwmax = std::min((n-1)/3, lwork/2);
			int nw = nwmax;
			// nsmax = the Largest number of simultaneous shifts for which there is sufficient
			//         workspace.
			int nsmax = std::min((n+6)/9, 2*lwork/3);
			nsmax -= (nsmax % 2);
			// ndfl: an iteration count restarted at deflation.
			int ndfl = 1;
			// itmax = iteration limit
			int itmax = std::max(30, 2*KEXSH) * std::max(10, (ihi-ilo+1));
			// Last row and column in the active block
			int kbot = ihi;
			// Main Loop
			real aa, bb, cc, cs, dd, sn, ss, swap;
			int i, im, im2, inf, ip, k, kdu, ks, kt, ktop, ku, kv, kwh, kwtop, kwv, ndec, nh, nho,
			    ns, nve, nwupbd;
			bool sorted;
			bool done = false;
			for (int it=0; it<itmax; it++)
			{
				// Done when kbot falls below ilo
				if (kbot<ilo)
				{
					done = true;
					break;
				}
				// Locate active block
				for (k=kbot; k>ilo; k--)
				{
					if (H[k+ldh*(k-1)]==ZERO)
					{
						break;
					}
				}
				ktop = k;
				/* Select deflation window size:
				 *     Typical Case:
				 *         If possible and advisable, nibble the entire active block. If not, use
				 *         size min(nwr,nwmax) or min(nwr+1,nwmax) depending upon which has the
				 *         smaller corresponding subdiagonal entry (a heuristic).
				 *     Exceptional Case:
				 *         If there have been no deflations in KEXNW or more iterations, then vary
				 *         the deflation window size. At first, because, larger windows are, in
				 *         general, more powerful than smaller ones, rapidly increase the window to
				 *         the maximum possible. Then, gradually reduce the window size.         */
				nh = kbot - ktop + 1;
				nwupbd = std::min(nh, nwmax);
				if (ndfl<KEXNW)
				{
					nw = std::min(nwupbd, nwr);
				}
				else
				{
					nw = std::min(nwupbd, 2*nw);
				}
				if (nw<nwmax)
				{
					if (nw>=nh-1)
					{
						nw = nh;
					}
					else
					{
						kwtop = kbot - nw + 1;
						if (std::fabs(H[kwtop+ldh*(kwtop-1)])
						    > std::fabs(H[kwtop-1+ldh*(kwtop-2)]))
						{
							nw++;
						}
					}
				}
				if (ndfl<KEXNW)
				{
					ndec = -1;
				}
				else if (ndec>=0 || nw>=nwupbd)
				{
					ndec++;
					if (nw-ndec<2)
					{
						ndec = 0;
					}
					nw -= ndec;
				}
				/* Aggressive early deflation: split workspace under the subdiagonal into
				 *     - an nw by nw work array V in the lower left-hand-corner,
				 *     - an nw by at-least-nw-but-more-is-better (nw by nho) horizontal work array
				 *       along the bottom edge,
				 *     - an at-least-nw-but-more-is-better by nw (NHV by nw) vertical work array
				 *       along the left-hand-edge.                                               */
				kv  = n - nw;
				kt  = nw;
				nho = (n-nw-1) - kt;
				kwv = nw + 1;
				nve = (n-nw) - kwv;
				// Aggressive early deflation
				dlaqr3(wantt, wantz, n, ktop, kbot, nw, H, ldh, iloz, ihiz, Z, ldz, ls, ld, wr, wi,
				       &H[kv], ldh, nho, &H[kv+ldh*kt], ldh, nve, &H[kwv], ldh, work, lwork);
				// Adjust kbot accounting for new deflations.
				kbot -= ld;
				// ks points to the shifts.
				ks = kbot - ls + 1;
				/* Skip an expensive QR sweep if there is a (partly heuristic) reason to expect
				 * that many eigenvalues will deflate without it. Here, the QR sweep is skipped if
				 * many eigenvalues have just been deflated or if the remaining active block is
				 * small.                                                                        */
				if (ld==0 || (100*ld<=nw*nibble && (kbot-ktop+1>std::min(nmin, nwmax))))
				{
					// ns = nominal number of simultaneous shifts.
					// This may be lowered (slightly) if dlaqr3 did not provide that many shifts.
					ns = std::min(std::min(nsmax, nsr), std::max(2, kbot-ktop));
					ns -= ns % 2;
					/* If there have been no deflations in a multiple of KEXSH iterations, then try
					 * exceptional shifts. Otherwise use shifts provided by DLAQR3 above or from
					 * the eigenvalues of a trailing principal submatrix.                        */
					if ((ndfl%KEXSH) == 0)
					{
						ks = kbot - ns + 1;
						for (i=kbot; i>std::max(ks, ktop+1); i-=2)
						{
							ss = std::fabs(H[i+ldh*(i-1)]) + std::fabs(H[i-1+ldh*(i-2)]);
							aa = WILK1*ss + H[i+ldh*i];
							bb = ss;
							cc = WILK2 * ss;
							dd = aa;
							dlanv2(aa, bb, cc, dd, wr[i-1], wi[i-1], wr[i], wi[i], cs, sn);
						}
						if (ks==ktop)
						{
							wr[ks+1] = H[ks+1+ldh*(ks+1)];
							wi[ks+1] = ZERO;
							wr[ks] = wr[ks+1];
							wi[ks] = wi[ks+1];
						}
					}
					else
					{
						/* Got ns/2 or fewer shifts? Use dlaqr4 or dlahqr on a trailing principal
						 * submatrix to get more. (Since ns<=nsmax<=(n+6)/9, there is enough space
						 * below the subdiagonal to fit an ns by ns scratch array.)              */
						if (kbot-ks+1 <= ns/2)
						{
							ks = kbot - ns + 1;
							kt = n - ns;
							dlacpy("A", ns, ns, &H[ks+ldh*ks], ldh, &H[kt], ldh);
							if (ns>nmin)
							{
								dlaqr4(false, false, ns, 0, ns-1, &H[kt], ldh, &wr[ks], &wi[ks], 0,
								       0, nullptr, 1, work, lwork, inf);
							}
							else
							{
								dlahqr(false, false, ns, 0, ns-1, &H[kt], ldh, &wr[ks], &wi[ks], 0,
								       0, nullptr, 1, inf);
							}
							ks += inf;
							// In case of a rare QR failure use eigenvalues of the trailing 2 by 2
							// principal submatrix.
							if (ks>=kbot)
							{
								aa = H[kbot-1+ldh*(kbot-1)];
								cc = H[kbot  +ldh*(kbot-1)];
								bb = H[kbot-1+ldh* kbot];
								dd = H[kbot  +ldh* kbot];
								dlanv2(aa, bb, cc, dd, wr[kbot-1], wi[kbot-1], wr[kbot], wi[kbot],
								       cs, sn);
								ks = kbot - 1;
							}
						}
						if (kbot-ks+1>ns)
						{
							// Sort the shifts (Helps a little)
							// Bubble sort keeps complex conjugate pairs together.
							sorted = false;
							for (k=kbot; k>ks; k--)
							{
								if (sorted)
								{
									break;
								}
								sorted = true;
								for (i=ks; i<k; i++)
								{
									ip = i + 1;
									if (std::fabs(wr[i])+std::fabs(wi[i])
									    < std::fabs(wr[ip])+std::fabs(wi[ip]))
									{
										sorted = false;
										swap   = wr[i];
										wr[i]  = wr[ip];
										wr[ip] = swap;
										swap   = wi[i];
										wi[i]  = wi[ip];
										wi[ip] = swap;
									}
								}
							}
						}
						/* Shuffle shifts into pairs of real shifts and pairs of complex conjugate
						 * shifts assuming complex conjugate shifts are already adjacent to one
						 * another. (Yes, they are.)                                             */
						for (i=kbot; i>=ks+2; i-=2)
						{
							im = i - 1;
							if (wi[i]!=-wi[im])
							{
								im2 = i - 2;
								swap    = wr[i];
								wr[i]   = wr[im];
								wr[im]  = wr[im2];
								wr[im2] = swap;
								swap    = wi[i];
								wi[i]   = wi[im];
								wi[im]  = wi[im2];
								wi[im2] = swap;
							}
						}
					}
					// If there are only two shifts and both are real, then use only one.
					if (kbot-ks+1==2)
					{
						if (wi[kbot]==ZERO)
						{
							if (std::fabs(wr[kbot]-H[kbot+ldh*kbot])
							    < std::fabs(wr[kbot-1]-H[kbot+ldh*kbot]))
							{
								wr[kbot-1] = wr[kbot];
							}
							else
							{
								wr[kbot] = wr[kbot-1];
							}
						}
					}
					// Use up to ns of the the smallest magnatiude shifts. If there aren't ns
					// shifts available, then use them all, possibly dropping one to make the
					// number of shifts even.
					ns = std::min(ns, kbot-ks+1);
					ns -= ns % 2;
					ks = kbot - ns + 1;
					/* Small-bulge multi-shift QR sweep: split workspace under the subdiagonal into
					 *     - a kdu by kdu work array U in the lower left-hand-corner,
					 *     - a kdu by at-least-kdu-but-more-is-better (kdu by nho) horizontal work
					 *       array WH along the bottom edge,
					 *     - and an at-least-kdu-but-more-is-better by kdu (nve by kdu) vertical
					 *       work array WV along the left-hand edge.                             */
					kdu = 3*ns - 3;
					ku  = n - kdu;
					kwh = kdu;
					nho = (n-kdu+1-4) - (kdu+1) + 1;
					kwv = kdu + 3;
					nve = n - kdu - kwv;
					// Small-bulge multi-shift QR sweep
					dlaqr5(wantt, wantz, kacc22, n, ktop, kbot, ns, &wr[ks], &wi[ks], H, ldh, iloz,
					       ihiz, Z, ldz, work, 3, &H[ku], ldh, nve, &H[kwv], ldh, nho,
					       &H[ku+ldh*kwh], ldh);
				}
				// Note progress (or the lack of it).
				if (ld>0)
				{
					ndfl = 1;
				}
				else
				{
					ndfl++;
				}
				// End of main loop
			}
			if (!done)
			{
				// Iteration limit exceeded. Set info to show where the problem occurred and exit.
				info = kbot + 1;
			}
		}
		// Return the optimal value of lwork.
		work[0] = real(lwkopt);
	}

	/*! §dlaqr1 sets a scalar multiple of the first column of the product of a 2 by 2 or 3 by 3
	 *  matrix H and specified shifts.
	 *
	 * Given a 2 by 2 or 3 by 3 matrix $H$, §dlaqr1 sets §v to a scalar multiple of the first
	 * column of the product\n
	 *     (*)&emsp;&emsp; $K = (H - (\{sr1} + i*\{si1})I) (H - (\{sr2} + i\{si2})I)$\n
	 * scaling to avoid overflows and most underflows. It is assumed that either\n
	 *     1) $\{sr1}=\{sr2}$ and $\{si1}=-\{si2}$\n
	 * or\n
	 *     2) $\{si1}=\{si2}=0$.\n
	 * This is useful for starting double implicit shift bulges in the QR algorithm.
	 * \param[in] n   Order of the matrix $H$. §n must be either 2 or 3.
	 * \param[in] H   an array, dimension (§ldh,§n)\n The 2 by 2 or 3 by 3 matrix $H$ in (*).
	 * \param[in] ldh
	 *     The leading dimension of §H as declared in the calling procedure. $\{ldh}\ge\{n}$
	 *
	 * \param[in]  sr1, si1, sr2, si2 The shifts in (*).
	 * \param[out] v
	 *     an array, dimension (§n)\n
	 *     A scalar multiple of the first column of the matrix $K$ in (*).
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017
	 * \remark
	 *     Contributors:\n
	 *     Karen Braman and Ralph Byers, Department of Mathematics, University of Kansas, USA    */
	static void dlaqr1(int const n, real const* const H, int const ldh, real const sr1,
	                   real const si1, real const sr2, real const si2, real* const v) /*const*/
	{
		real H21S, H31S, S;
		if (n==2)
		{
			S = std::fabs(H[0]-sr2) + std::fabs(si2) + std::fabs(H[1]);
			if (S==ZERO)
			{
				v[0] = ZERO;
				v[1] = ZERO;
			}
			else
			{
				H21S = H[1] / S;
				v[0] = H21S*H[ldh] + (H[0]-sr1)*((H[0]-sr2)/S) - si1*(si2/S);
				v[1] = H21S * (H[0]+H[1+ldh]-sr1-sr2);
			}
		}
		else
		{
			S = std::fabs(H[0]-sr2) + std::fabs(si2) + std::fabs(H[1]) + std::fabs(H[2]);
			if (S==ZERO)
			{
				v[0] = ZERO;
				v[1] = ZERO;
				v[2] = ZERO;
			}
			else
			{
				H21S = H[1] / S;
				H31S = H[2] / S;
				v[0] = (H[0]-sr1)*((H[0]-sr2)/S) - si1*(si2/S) + H[ldh]*H21S + H[ldh*2]*H31S;
				v[1] = H21S*(H[0]+H[1+ldh]-sr1-sr2) + H[1+ldh*2]*H31S;
				v[2] = H31S*(H[0]+H[2+ldh*2]-sr1-sr2) + H21S*H[2+ldh];
			}
		}
	}

	/*! §dlaqr2 performs the orthogonal similarity transformation of a Hessenberg matrix to detect
	 *  and deflate fully converged eigenvalues from a trailing principal submatrix (aggressive
	 *  early deflation).
	 *
	 * §dlaqr2 is identical to §dlaqr3 except that it avoids recursion by calling §dlahqr instead
	 * of §dlaqr4.\n
	 * Aggressive early deflation:\n
	 * This subroutine accepts as input an upper Hessenberg matrix $H$ and performs an orthogonal
	 * similarity transformation designed to detect and deflate fully converged eigenvalues from a
	 * trailing principal submatrix.  On output $H$ has been overwritten by a new Hessenberg matrix
	 * that is a perturbation of an orthogonal similarity transformation of $H$. It is to be hoped
	 * that the final version of $H$ has many zero subdiagonal entries.
	 * \param[in] wantt
	 *     If §true, then the Hessenberg matrix $H$ is fully updated so that the quasi-triangular
	 *     Schur factor may be computed (in cooperation with the calling subroutine).\n
	 *     If §false, then only enough of $H$ is updated to preserve the eigenvalues.
	 *
	 * \param[in] wantz
	 *     If §true, then the orthogonal matrix $Z$ is updated so so that the orthogonal Schur
	 *     factor may be computed (in cooperation with the calling subroutine).\n
	 *     If §false, then §Z is not referenced.
	 *
	 * \param[in] n
	 *     The order of the matrix $H$ and (if §wantz is §true) the order of the orthogonal matrix
	 *     $Z$.
	 *
	 * \param[in] ktop
	 *     It is assumed that either $\{ktop}=0$ or $H[\{ktop},\{ktop}-1]=0$.\n
	 *     §kbot and §ktop together determine an isolated block along the diagonal of the
	 *     Hessenberg matrix.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in] kbot
	 *     It is assumed without a check that either $\{kbot}=\{n}-1$ or $H[\{kbot}+1,\{kbot}]=0$.
	 *     \n §kbot and §ktop together determine an isolated block along the diagonal of the
	 *     Hessenberg matrix.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in]     nw Deflation window size. $1\le\{nw}\le(\{kbot}-\{ktop}+1)$.
	 * \param[in,out] H
	 *     an array, dimension (§ldh,§n)\n
	 *     On input the initial §n by §n section of §H stores the Hessenberg matrix undergoing
	 *     aggressive early deflation.\n
	 *     On output §H has been transformed by an orthogonal similarity transformation, perturbed,
	 *     and then returned to Hessenberg form that (it is to be hoped) has some zero subdiagonal
	 *     entries.
	 *
	 * \param[in] ldh
	 *     Leading dimension of §H just as declared in the calling subroutine. $\{n}\le\{ldh}$
	 *
	 * \param[in] iloz, ihiz
	 *     Specify the rows of $Z$ to which transformations must be applied if §wantz is §true.
	 *     $0\le\{iloz}\le\{ihiz}<\{n}$.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in,out] Z
	 *     an array, dimension (§ldz,§n)\n
	 *     If §wantz is §true, then on output, the orthogonal similarity transformation mentioned
	 *     above has been accumulated into $\{Z}[\{iloz}:\{ihiz},\{iloz}:\{ihiz}]$ from the right.
	 *     \n If §wantz is §false, then §Z is unreferenced.
	 *
	 * \param[in] ldz
	 *     The leading dimension of §Z just as declared in the calling subroutine. $1\le\{ldz}$.
	 *
	 * \param[out] ns
	 *     The number of unconverged (ie approximate) eigenvalues returned in §sr and §si that may
	 *     be used as shifts by the calling subroutine.
	 *
	 * \param[out] nd The number of converged eigenvalues uncovered by this subroutine.
	 *
	 * \param[out] sr, si
	 *     arrays, dimension (§kbot)\n
	 *     On output, the real and imaginary parts of approximate eigenvalues that may be used for
	 *     shifts are stored in $\{sr}[\{kbot}-\{nd}-\{ns}+1]$ through $\{sr}[\{kbot}-\{nd}]$ and
	 *     $\{si}[\{kbot}-\{nd}-\{ns}+1]$ through $\{si}[\{kbot}-\{nd}]$, respectively.\n.
	 *     The real and imaginary parts of converged eigenvalues are stored in
	 *     $\{sr}[\{kbot}-\{nd}+1]$ through $\{sr}[\{kbot}]$ and $\{si}[\{kbot}-\{nd}-1]$ through
	 *     $\{si}[\{kbot}]$, respectively.
	 *
	 * \param[out] V   an array, dimension (§ldv,§nw)\n An §nw by §nw work array.
	 * \param[in]  ldv
	 *     The leading dimension of §V just as declared in the calling subroutine. $\{nw}\le\{ldv}$
	 *
	 * \param[in]  nh  The number of columns of §T. $\{nh}\ge\{nw}$.
	 * \param[out] T   an array, dimension (§ldt,§nw)
	 * \param[in]  ldt
	 *     The leading dimension of §T just as declared in the calling subroutine. $\{nw}\le\{ldt}$
	 *
	 * \param[in] nv
	 *     The number of rows of work array §Wv available for workspace. $\{nv}\ge\{nw}$.
	 *
	 * \param[out] Wv   an array, dimension (§ldwv,§nw)
	 * \param[in]  ldwv
	 *     The leading dimension of §Wv just as declared in the calling subroutine.
	 *     $\{nw}\le\{ldv}$.
	 *
	 * \param[out] work
	 *     an array, dimension (§lwork)\n On exit, $\{work}[0]$ is set to an estimate of the
	 *     optimal value of §lwork for the given values of §n, §nw, §ktop and §kbot.
	 *
	 * \param[in] lwork
	 *     The dimension of the work array §work. $\{lwork}=2\{nw}$ suffices, but greater
	 *     efficiency may result from larger values of §lwork.\n
	 *     If $\{lwork}=-1$, then a workspace query is assumed; §dlaqr2 only estimates the optimal
	 *     workspace size for the given values of §n, §nw, §ktop and §kbot. The estimate is
	 *     returned in $\{work}[0]$. No error message related to §lwork is issued by §xerbla.
	 *     Neither §H nor §Z are accessed.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017
	 * \remark
	 *     Contributors:\n
	 *         Karen Braman and Ralph Byers, Department of Mathematics, University of Kansas, USA*/
	static void dlaqr2(bool const wantt, bool const wantz, int const n, int const ktop,
	                   int const kbot, int const nw, real* const H, int const ldh, int const iloz,
	                   int const ihiz, real* const Z, int const ldz, int& ns, int& nd,
	                   real* const sr, real* const si, real* const V, int const ldv, int const nh,
	                   real* const T, int const ldt, int const nv, real* const Wv, int const ldwv,
	                   real* const work, int const lwork) /*const*/
	{
		// Estimate optimal workspace.
		int jw = std::min(nw, kbot-ktop+1);
		int info, lwkopt;
		if (jw<=2)
		{
			lwkopt = 1;
		}
		else
		{
			// Workspace query call to DGEHRD
			dgehrd(jw, 0, jw-2, T, ldt, work, work, -1, info);
			int lwk1 = int(work[0]);
			// Workspace query call to DORMHR
			dormhr("R", "N", jw, jw, 0, jw-2, T, ldt, work, V, ldv, work, -1, info);
			int lwk2 = int(work[0]);
			// Optimal workspace
			lwkopt = jw + std::max(lwk1, lwk2);
		}
		// Quick return in case of workspace query.
		if (lwork==-1)
		{
			work[0] = real(lwkopt);
			return;
		}
		// Nothing to do for an empty active block ...
		ns = 0;
		nd = 0;
		work[0] = ONE;
		if (ktop>kbot)
		{
			return;
		}
		// ... nor for an empty deflation window.
		if (nw<1)
		{
			return;
		}
		// Machine constants
		real safmin = dlamch("SAFE MINIMUM");
		real safmax = ONE / safmin;
		dlabad(safmin, safmax);
		real ulp = dlamch("PRECISION");
		real smlnum = safmin * (real(n)/ulp);
		// Setup deflation window
		jw = std::min(nw, kbot-ktop+1);
		int kwtop = kbot - jw + 1;
		real s;
		if (kwtop==ktop)
		{
			s = ZERO;
		}
		else
		{
			s = H[kwtop+ldh*(kwtop-1)];
		}
		if (kbot==kwtop)
		{
			// 1-by-1 deflation window: not much to do
			sr[kwtop] = H[kwtop+ldh*kwtop];
			si[kwtop] = ZERO;
			ns = 1;
			nd = 0;
			if (std::fabs(s)<=std::max(smlnum, ulp*std::fabs(H[kwtop+ldh*kwtop])))
			{
				ns = 0;
				nd = 1;
				if (kwtop>ktop)
				{
					H[kwtop+ldh*(kwtop-1)] = ZERO;
				}
			}
			work[0] = ONE;
			return;
		}
		// Convert to spike-triangular form. (In case of a rare QR failure, this routine continues
		// to do aggressive early deflation using that part of the deflation window that converged
		// using infqr here and there to keep track.)
		dlacpy("U", jw, jw, &H[kwtop+ldh*kwtop], ldh, T, ldt);
		Blas<real>::dcopy(jw-1, &H[kwtop+1+ldh*kwtop], ldh+1, &T[1], ldt+1);
		dlaset("A", jw, jw, ZERO, ONE, V, ldv);
		int infqr;
		dlahqr(true, true, jw, 0, jw-1, T, ldt, &sr[kwtop], &si[kwtop], 0, jw-1, V, ldv, infqr);
		// DTREXC needs a clean margin near the diagonal
		int i;
		for (i=0; i<jw-3; i++)
		{
			T[i+2+ldt*i] = ZERO;
			T[i+3+ldt*i] = ZERO;
		}
		if (jw>2)
		{
			T[jw-1+ldt*(jw-3)] = ZERO;
		}
		// Deflation detection loop
		ns = jw;
		real foo;
		int ifst;
		bool bulge;
		int ilst = infqr;
		int nsm = ns - 1;
		while (ilst<ns)
		{
			if (ns==1)
			{
				bulge = false;
			}
			else
			{
				bulge = (T[nsm+ldt*(nsm-1)]!=ZERO);
			}
			// Small spike tip test for deflation
			if (!bulge)
			{
				// Real eigenvalue
				foo = std::fabs(T[nsm+ldt*nsm]);
				if (foo==ZERO)
				{
					foo = std::fabs(s);
				}
				if (std::fabs(s*V[ldv*nsm])<=std::max(smlnum, ulp*foo))
				{
					// Deflatable
					ns--;
					nsm = ns - 1;
				}
				else
				{
					// Undeflatable. Move it up out of the way. (DTREXC can not fail in this case.)
					ifst = nsm;
					dtrexc("V", jw, T, ldt, V, ldv, ifst, ilst, work, info);
					ilst++;
				}
			}
			else
			{
				// Complex conjugate pair
				foo = std::fabs(T[nsm+ldt*nsm]) + std::sqrt(std::fabs(T[nsm+ldt*(nsm-1)]))
				                                 *std::sqrt(std::fabs(T[nsm-1+ldt*nsm]));
				if (foo==ZERO)
				{
					foo = std::fabs(s);
				}
				if (std::max(std::fabs(s*V[ldv*nsm]), std::fabs(s*V[ldv*(nsm-1)]))
				    <= std::max(smlnum, ulp*foo))
				{
					// Deflatable
					ns -= 2;
					nsm = ns - 1;
				}
				else
				{
					// Undeflatable. Move them up out of the way. Fortunately, dtrexc does the
					// right thing with ilst in case of a rare exchange failure.
					ifst = nsm;
					dtrexc("V", jw, T, ldt, V, ldv, ifst, ilst, work, info);
					ilst += 2;
				}
			}
		} // End deflation detection loop
		// Return to Hessenberg form
		if (ns==0)
		{
			s = ZERO;
		}
		if (ns<jw)
		{
			// sorting diagonal blocks of T improves accuracy for graded matrices.
			// Bubble sort deals well with exchange failures.
			bool sorted = false;
			i = ns;
			real evi, evk;
			int k, kend, ti, tk;
			while (!sorted)
			{
				sorted = true;
				kend = i - 1;
				i = infqr;
				ti = i + ldt*i;
				if (i==ns-1)
				{
					k = i + 1;
				}
				else if (T[1+ti]==ZERO)
				{
					k = i + 1;
				}
				else
				{
					k = i + 2;
				}
				tk = k + ldt*k;
				while (k<=kend)
				{
					if (k==i+1)
					{
						evi = std::fabs(T[ti]);
					}
					else
					{
						evi = std::fabs(T[ti])
						      + std::sqrt(std::fabs(T[1+ti]))*std::sqrt(std::fabs(T[ti+ldt]));
					}
					if (k==kend)
					{
						evk = std::fabs(T[tk]);
					}
					else if (T[1+tk]==ZERO)
					{
						evk = std::fabs(T[tk]);
					}
					else
					{
						evk = std::fabs(T[tk])
						      + std::sqrt(std::fabs(T[1+tk]))*std::sqrt(std::fabs(T[tk+ldt]));
					}
					if (evi>=evk)
					{
						i = k;
					}
					else
					{
						sorted = false;
						ifst = i;
						ilst = k;
						dtrexc("V", jw, T, ldt, V, ldv, ifst, ilst, work, info);
						if (info==0)
						{
							i = ilst;
						}
						else
						{
							i = k;
						}
					}
					ti = i + ldt*i;
					if (i==kend)
					{
						k = i + 1;
					}
					else if (T[1+ti]==ZERO)
					{
						k = i + 1;
					}
					else
					{
						k = i + 2;
					}
					tk = k + ldt*k;
				}
			}
		}
		// Restore shift/eigenvalue array from T
		i = jw - 1;
		real aa, bb, cc, cs, dd, sn;
		while (i>=infqr)
		{
			if (i==infqr)
			{
				sr[kwtop+i] = T[i+ldt*i];
				si[kwtop+i] = ZERO;
				i--;
			}
			else if (T[i+ldt*(i-1)]==ZERO)
			{
				sr[kwtop+i] = T[i+ldt*i];
				si[kwtop+i] = ZERO;
				i--;
			}
			else
			{
				aa = T[i-1+ldt*(i-1)];
				cc = T[i+ldt*(i-1)];
				bb = T[i-1+ldt*i];
				dd = T[i+ldt*i];
				dlanv2(aa, bb, cc, dd, sr[kwtop+i-1], si[kwtop+i-1], sr[kwtop+i], si[kwtop+i], cs,
				       sn);
				i -= 2;
			}
		}
		if (ns<jw || s==ZERO)
		{
			if (ns>1 && s!=ZERO)
			{
				// Reflect spike back into lower triangle
				Blas<real>::dcopy(ns, V, ldv, work, 1);
				real beta = work[0];
				real tau;
				dlarfg(ns, beta, &work[1], 1, tau);
				work[0] = ONE;
				dlaset("L", jw-2, jw-2, ZERO, ZERO, &T[2], ldt);
				dlarf("L", ns, jw, work, 1, tau, T, ldt, &work[jw]);
				dlarf("R", ns, ns, work, 1, tau, T, ldt, &work[jw]);
				dlarf("R", jw, ns, work, 1, tau, V, ldv, &work[jw]);
				dgehrd(jw, 0, ns-1, T, ldt, work, &work[jw], lwork-jw, info);
			}
			// Copy updated reduced window into place
			if (kwtop>0)
			{
				H[kwtop+ldh*(kwtop-1)] = s * V[0];
			}
			dlacpy("U", jw, jw, T, ldt, &H[kwtop+ldh*kwtop], ldh);
			Blas<real>::dcopy(jw-1, &T[1], ldt+1, &H[kwtop+1+ldh*kwtop], ldh+1);
			// Accumulate orthogonal matrix in order update H and Z, if requested.
			if (ns>1 && s!=ZERO)
			{
				dormhr("R", "N", jw, ns, 0, ns-1, T, ldt, work, V, ldv, &work[jw], lwork-jw, info);
			}
			// Update vertical slab in H
			int ltop;
			if (wantt)
			{
				ltop = 0;
			}
			else
			{
				ltop = ktop;
			}
			int kln, krow;
			int htop = ldh * kwtop;
			for (krow=ltop; krow<kwtop; krow+=nv)
			{
				kln = std::min(nv, kwtop-krow-1);
				Blas<real>::dgemm("N", "N", kln, jw, jw, ONE, &H[krow+htop], ldh, V, ldv, ZERO, Wv,
				                  ldwv);
				dlacpy("A", kln, jw, Wv, ldwv, &H[krow+htop], ldh);
			}
			// Update horizontal slab in H
			if (wantt)
			{
				for (int kcol=kbot+1; kcol<n; kcol+=nh)
				{
					kln = std::min(nh, n-kcol);
					Blas<real>::dgemm("C", "N", jw, kln, jw, ONE, V, ldv, &H[kwtop+ldh*kcol], ldh,
					                  ZERO, T, ldt);
					dlacpy("A", jw, kln, T, ldt, &H[kwtop+ldh*kcol], ldh);
				}
			}
			// Update vertical slab in Z
			if (wantz)
			{
				int ztop = ldz * kwtop;
				for (krow=iloz; krow<=ihiz; krow+=nv)
				{
					kln = std::min(nv, ihiz-krow+1);
					Blas<real>::dgemm("N", "N", kln, jw, jw, ONE, &Z[krow+ztop], ldz, V, ldv, ZERO,
					                  Wv, ldwv);
					dlacpy("A", kln, jw, Wv, ldwv, &Z[krow+ztop], ldz);
				}
			}
		}
		// Return the number of deflations ...
		nd = jw - ns;
		// ... and the number of shifts. (Subtracting infqr from the spike length takes care of the
		// case of a rare QR failure while calculating eigenvalues of the deflation window.)
		ns -= infqr;
		// Return optimal workspace.
		work[0] = real(lwkopt);
	}

	/*! §dlaqr3 performs the orthogonal similarity transformation of a Hessenberg matrix to detect
	 *  and deflate fully converged eigenvalues from a trailing principal submatrix (aggressive
	 *  early deflation).
	 *
	 * Aggressive early deflation:\n
	 * §dlaqr3 accepts as input an upper Hessenberg matrix $H$ and performs an orthogonal
	 * similarity transformation designed to detect and deflate fully converged eigenvalues from a
	 * trailing principal submatrix. On output $H$ has been overwritten by a new Hessenberg matrix
	 * that is a perturbation of an orthogonal similarity transformation of $H$. It is to be hoped
	 * that the final version of $H$ has many zero subdiagonal entries.
	 * \param[in] wantt
	 *     If §true, then the Hessenberg matrix $H$ is fully updated so that the quasi-triangular
	 *     Schur factor may be computed (in cooperation with the calling subroutine).\n
	 *     If §false, then only enough of $H$ is updated to preserve the eigenvalues.
	 *
	 * \param[in] wantz
	 *     If §true, then the orthogonal matrix $Z$ is updated so that the orthogonal Schur factor
	 *     may be computed (in cooperation with the calling subroutine).\n
	 *     If §false, then $Z$ is not referenced.
	 *
	 * \param[in] n
	 *     The order of the matrix $H$ and (if §wantz is §true) the order of the orthogonal matrix
	 *     $Z$.
	 *
	 * \param[in] ktop
	 *     It is assumed that either $\{ktop}=0$ or $H[\{ktop},\{ktop}-1]=0$. §kbot and §ktop
	 *     together determine an isolated block along the diagonal of the Hessenberg matrix.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in] kbot
	 *     It is assumed without a check that either $\{kbot}=\{n}-1$ or $H[\{kbot}+1,\{kbot}]=0$.
	 *     §kbot and §ktop together determine an isolated block along the diagonal of the
	 *     Hessenberg matrix.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in] nw Deflation window size. $1\le\{nw}\le(\{kbot}-\{ktop}+1)$.
	 *
	 * \param[in,out] H
	 *     an array, dimension (§ldh,§n)\n
	 *     On input the initial §n by §n section of §H stores the Hessenberg matrix undergoing
	 *     aggressive early deflation.\n
	 *     On output §H has been transformed by an orthogonal similarity transformation, perturbed,
	 *     and then returned to Hessenberg form that (it is to be hoped) has some zero subdiagonal
	 *     entries.
	 *
	 * \param[in] ldh
	 *     Leading dimension of §H just as declared in the calling subroutine. $\{n}\le\{ldh}$.
	 *
	 * \param[in] iloz, ihiz
	 *     Specify the rows of §Z to which transformations must be applied if §wantz is §true.
	 *     $0\le\{iloz}\le\{ihiz}<\{n}$.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in,out] Z
	 *     an array, dimension (§ldz,§n)\n
	 *     If §wantz is §true, then on output, the orthogonal similarity transformation mentioned
	 *     above has been accumulated into $\{Z}[\{iloz}:\{ihiz},\{iloz}:\{ihiz}]$ from the right.
	 *     \n If §wantz is §false, then §Z is unreferenced.
	 *
	 * \param[in] ldz
	 *     The leading dimension of §Z just as declared in the calling subroutine. $1\le\{ldz}$.
	 *
	 * \param[out] ns
	 *     The number of unconverged (ie approximate) eigenvalues returned in §sr and §si that may
	 *     be used as shifts by the calling subroutine.
	 *
	 * \param[out] nd     The number of converged eigenvalues uncovered by this subroutine.
	 * \param[out] sr, si
	 *     arrays, dimension ($\{kbot}+1$)\n
	 *     On output, the real and imaginary parts of approximate eigenvalues that may be used for
	 *     shifts are stored in $\{sr}[\{kbot}-\{nd}-\{ns}+1]$ through $\{sr}[\{kbot}-\{nd}]$ and
	 *     $\{si}[\{kbot}-\{nd}-\{ns}+1]$ through $\{si}[\{kbot}-\{nd}]$, respectively.\n
	 *     The real and imaginary parts of converged eigenvalues are stored in
	 *     $\{sr}[\{kbot}-\{nd}+1]$ through $\{sr}[\{kbot}]$ and $\{si}[\{kbot}-\{nd}+1]$ through
	 *     $\{si}[\{kbot}]$, respectively.
	 *
	 * \param[out] V   an array, dimension (§ldv,§nw)\n An §nw by §nw work array.
	 * \param[in]  ldv
	 *     The leading dimension of §V just as declared in the calling subroutine.
	 *     $\{nw}\le\{ldv}$.
	 *
	 * \param[in]  nh  The number of columns of §T. $\{nh}\ge\{nw}$.
	 * \param[out] T   an array, dimension (§ldt,§nw)
	 * \param[in]  ldt
	 *     The leading dimension of §T just as declared in the calling subroutine.
	 *     $\{nw}\le\{ldt}$.
	 *
	 * \param[in] nv
	 *     The number of rows of work array §Wv available for workspace. $\{nv}\ge\{nw}$.
	 *
	 * \param[out] Wv   an array, dimension (§ldwv,§nw)
	 * \param[in]  ldwv
	 *     The leading dimension of §W just as declared in the calling subroutine.
	 *     $\{nw}\le\{ldwv}$.
	 *
	 * \param[out] work
	 *     an array, dimension (§lwork)\n
	 *     On exit, $\{work}[0]$ is set to an estimate of the optimal value of §lwork for the given
	 *     values of §n, §nw, §ktop and §kbot.
	 *
	 * \param[in] lwork
	 *     The dimension of the work array §work. $\{lwork}=2\{nw}$ suffices, but greater
	 *     efficiency may result from larger values of §lwork.\n
	 *     If $\{lwork}=-1$, then a workspace query is assumed; §dlaqr3 only estimates the optimal
	 *     workspace size for the given values of §n, §nw, §ktop and §kbot. The estimate is
	 *     returned in $\{work}[0]$. No error message related to §lwork is issued by §XERBLA.
	 *     Neither §H nor §Z are accessed.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016
	 * \remark
	 *     Contributors:\n
	 *         Karen Braman and Ralph Byers, Department of Mathematics, University of Kansas, USA*/
	static void dlaqr3(bool const wantt, bool const wantz, int const n, int const ktop,
	                   int const kbot, int const nw, real* const H, int const ldh, int const iloz,
	                   int const ihiz, real* const Z, int const ldz, int& ns, int& nd,
	                   real* const sr, real* const si, real* const V, int const ldv, int const nh,
	                   real* const T, int const ldt, int const nv, real* const Wv, int const ldwv,
	                   real* const work, int const lwork) /*const*/
	{
		// Estimate optimal workspace.
		int jw = std::min(nw, kbot-ktop+1);
		int info, infqr, lwkopt;
		if (jw<=2)
		{
			lwkopt = 1;
		}
		else
		{
			// Workspace query call to dgehrd
			dgehrd(jw, 0, jw-2, T, ldt, work, work, -1, info);
			int lwk1 = int(work[0]);
			// Workspace query call to dormhr
			dormhr("R", "N", jw, jw, 0, jw-2, T, ldt, work, V, ldv, work, -1, info);
			int lwk2 = int(work[0]);
			// Workspace query call to dlaqr4
			dlaqr4(true, true, jw, 0, jw-1, T, ldt, sr, si, 0, jw-1, V, ldv, work, -1, infqr);
			int lwk3 = int(work[0]);
			// Optimal workspace
			lwkopt = std::max(jw+std::max(lwk1, lwk2), lwk3);
		}
		// Quick return in case of workspace query.
		if (lwork==-1)
		{
			work[0] = real(lwkopt);
			return;
		}
		// Nothing to do for an empty active block ...
		ns = 0;
		nd = 0;
		work[0] = ONE;
		if (ktop>kbot)
		{
			return;
		}
		// ... nor for an empty deflation window.
		if (nw<1)
		{
			return;
		}
		// Machine constants
		real safmin = dlamch("SAFE MINIMUM");
		real safmax = ONE / safmin;
		dlabad(safmin, safmax);
		real ulp = dlamch("PRECISION");
		real smlnum = safmin * (real(n)/ulp);
		// Setup deflation window
		jw = std::min(nw, kbot-ktop+1);
		int kwtop = kbot - jw + 1;
		real s;
		if (kwtop==ktop)
		{
			s = ZERO;
		}
		else
		{
			s = H[kwtop+ldh*(kwtop-1)];
		}
		if (kbot==kwtop)
		{
			// 1 by 1 deflation window: not much to do
			sr[kwtop] = H[kwtop+ldh*kwtop];
			si[kwtop] = ZERO;
			ns = 1;
			nd = 0;
			if (std::fabs(s) <= std::max(smlnum, ulp*std::fabs(H[kwtop+ldh*kwtop])))
			{
				ns = 0;
				nd = 1;
				if (kwtop>ktop)
				{
					H[kwtop+ldh*(kwtop-1)] = ZERO;
				}
			}
			work[0] = ONE;
			return;
		}
		/* Convert to spike-triangular form. (In case of a rare QR failure, this routine continues
		 * to do aggressive early deflation using that part of the deflation window that converged
		 * using infqr here and there to keep track.)                                            */
		dlacpy("U", jw, jw, &H[kwtop+ldh*kwtop], ldh, T, ldt);
		Blas<real>::dcopy(jw-1, &H[kwtop+1+ldh*kwtop], ldh+1, &T[1], ldt+1);
		dlaset("A", jw, jw, ZERO, ONE, V, ldv);
		int nmin = ilaenv(12, "DLAQR3", "SV", jw, 1, jw, lwork);
		if (jw>nmin)
		{
			dlaqr4(true, true, jw, 0, jw-1, T, ldt, &sr[kwtop], &si[kwtop], 0, jw-1, V, ldv, work,
			       lwork, infqr);
		}
		else
		{
			dlahqr(true, true, jw, 0, jw-1, T, ldt, &sr[kwtop], &si[kwtop], 0, jw-1, V, ldv,
			       infqr);
		}
		// dtrexc needs a clean margin near the diagonal
		for (int j=0; j<jw-3; j++)
		{
			T[j+2+ldt*j] = ZERO;
			T[j+3+ldt*j] = ZERO;
		}
		if (jw>2)
		{
			T[jw-1+ldt*(jw-3)] = ZERO;
		}
		// Deflation detection loop
		ns = jw;
		int ifst, ilst = infqr;
		real foo;
		bool bulge;
		while (ilst<ns)
		{
			if (ns==1)
			{
				bulge = false;
			}
			else
			{
				bulge = (T[ns-1+ldt*(ns-2)]!=ZERO);
			}
			// Small spike tip test for deflation
			if (!bulge)
			{
				// Real eigenvalue
				foo = std::fabs(T[ns-1+ldt*(ns-1)]);
				if (foo==ZERO)
				{
					foo = std::fabs(s);
				}
				if (std::fabs(s*V[ldv*(ns-1)]) <= std::max(smlnum, ulp*foo))
				{
					// Deflatable
					ns--;
				}
				else
				{
					// Undeflatable. Move it up out of the way. (DTREXC cannot fail in this case.)
					ifst = ns - 1;
					dtrexc("V", jw, T, ldt, V, ldv, ifst, ilst, work, info);
					ilst++;
				}
			}
			else
			{
				// Complex conjugate pair
				foo = std::fabs(T[ns-1+ldt*(ns-1)])
				      + std::sqrt(std::fabs(T[ns-1+ldt*(ns-2)]))*std::sqrt(std::fabs(T[ns-2+ldt*(ns-1)]));
				if (foo==ZERO)
				{
					foo = std::fabs(s);
				}
				if (std::max(std::fabs(s*V[ldv*(ns-1)]), std::fabs(s*V[ldv*(ns-2)]))
					<= std::max(smlnum, ulp*foo))
				{
					// Deflatable
					ns -= 2;
				}
				else
				{
					/* Undeflatable. Move them up out of the way. Fortunately, DTREXC does the
					 * right thing with ilst in case of a rare exchange failure.                 */
					ifst = ns - 1;
					dtrexc("V", jw, T, ldt, V, ldv, ifst, ilst, work, info);
					ilst += 2;
				}
			}
		} // End deflation detection loop
		// Return to Hessenberg form
		if (ns==0)
		{
			s = ZERO;
		}
		int i;
		if (ns<jw)
		{
			// sorting diagonal blocks of T improves accuracy for graded matrices.
			// Bubble sort deals well with exchange failures.
			real evi, evk;
			int k, kend;
			bool sorted = false;
			i = ns;
			while (!sorted)
			{
				sorted = true;
				kend = i - 1;
				i = infqr;
				if (i==ns-1)
				{
					k = i + 1;
				}
				else if (T[i+1+ldt*i]==ZERO)
				{
					k = i + 1;
				}
				else
				{
					k = i + 2;
				}
				while (k<=kend)
				{
					if (k==i+1)
					{
						evi = std::fabs(T[i+ldt*i]);
					}
					else
					{
						evi = std::fabs(T[i+ldt*i])
						      + std::sqrt(std::fabs(T[i+1+ldt*i]))*std::sqrt(std::fabs(T[i+ldt*(i+1)]));
					}
					if (k==kend)
					{
						evk = std::fabs(T[k+ldt*k]);
					}
					else if (T[k+1+ldt*k]==ZERO)
					{
						evk = std::fabs(T[k+ldt*k]);
					}
					else
					{
						evk = std::fabs(T[k+ldt*k])
						      + std::sqrt(std::fabs(T[k+1+ldt*k]))*std::sqrt(std::fabs(T[k+ldt*(k+1)]));
					}
					if (evi>=evk)
					{
						i = k;
					}
					else
					{
						sorted = false;
						ifst = i;
						ilst = k;
						dtrexc("V", jw, T, ldt, V, ldv, ifst, ilst, work, info);
						if (info==0)
						{
							i = ilst;
						}
						else
						{
							i = k;
						}
					}
					if (i==kend)
					{
						k = i + 1;
					}
					else if (T[i+1+ldt*i]==ZERO)
					{
						k = i + 1;
					}
					else
					{
						k = i + 2;
					}
				}
			}
		}
		// Restore shift/eigenvalue array from T
		real aa, bb, cc, dd, cs, sn;
		i = jw - 1;
		while (i>=infqr)
		{
			if (i==infqr)
			{
				sr[kwtop+i] = T[i+ldt*i];
				si[kwtop+i] = ZERO;
				i--;
			}
			else if (T[i+ldt*(i-1)]==ZERO)
			{
				sr[kwtop+i] = T[i+ldt*i];
				si[kwtop+i] = ZERO;
				i--;
			}
			else
			{
				aa = T[i-1+ldt*(i-1)];
				cc = T[i+ldt*(i-1)];
				bb = T[i-1+ldt*i];
				dd = T[i+ldt*i];
				dlanv2(aa, bb, cc, dd, sr[kwtop+i-1], si[kwtop+i-1], sr[kwtop+i], si[kwtop+i],
				       cs, sn);
				i -= 2;
			}
		}
		if (ns<jw || s==ZERO)
		{
			if (ns>1 && s!=ZERO)
			{
				// Reflect spike back into lower triangle
				Blas<real>::dcopy(ns, V, ldv, work, 1);
				real beta = work[0];
				real tau;
				dlarfg(ns, beta, &work[1], 1, tau);
				work[0] = ONE;
				dlaset("L", jw-2, jw-2, ZERO, ZERO, &T[2], ldt);
				dlarf("L", ns, jw, work, 1, tau, T, ldt, &work[jw]);
				dlarf("R", ns, ns, work, 1, tau, T, ldt, &work[jw]);
				dlarf("R", jw, ns, work, 1, tau, V, ldv, &work[jw]);
				dgehrd(jw, 1-1, ns-1, T, ldt, work, &work[jw], lwork-jw, info);
			}
			// Copy updated reduced window into place
			if (kwtop>0)
			{
				H[kwtop+ldh*(kwtop-1)] = s * V[0];
			}
			dlacpy("U", jw, jw, T, ldt, &H[kwtop+ldh*kwtop], ldh);
			Blas<real>::dcopy(jw-1, &T[1], ldt+1, &H[kwtop+1+ldh*kwtop], ldh+1);
			// Accumulate orthogonal matrix in order update H and Z, if requested.
			if (ns>1 && s!=ZERO)
			{
				dormhr("R", "N", jw, ns, 0, ns-1, T, ldt, work, V, ldv, &work[jw], lwork-jw,
				       info);
			}
			// Update vertical slab in H
			int ltop;
			if (wantt)
			{
				ltop = 0;
			}
			else
			{
				ltop = ktop;
			}
			int kln, krow;
			for (krow=ltop; krow<kwtop; krow+=nv)
			{
				kln = std::min(nv, kwtop-krow);
				Blas<real>::dgemm("N", "N", kln, jw, jw, ONE, &H[krow+ldh*kwtop], ldh, V, ldv, ZERO,
				                  Wv, ldwv);
				dlacpy("A", kln, jw, Wv, ldwv, &H[krow+ldh*kwtop], ldh);
			}
			// Update horizontal slab in H
			if (wantt)
			{
				for (int kcol=kbot+1; kcol<n; kcol+=nh)
				{
					kln = std::min(nh, n-kcol);
					Blas<real>::dgemm("C", "N", jw, kln, jw, ONE, V, ldv, &H[kwtop+ldh*kcol], ldh,
					                  ZERO, T, ldt);
					dlacpy("A", jw, kln, T, ldt, &H[kwtop+ldh*kcol], ldh);
				}
			}
			// Update vertical slab in Z
			if (wantz)
			{
				for (krow=iloz; krow<=ihiz; krow+=nv)
				{
					kln = std::min(nv, ihiz-krow+1);
					Blas<real>::dgemm("N", "N", kln, jw, jw, ONE, &Z[krow+ldz*kwtop], ldz, V, ldv,
					                  ZERO, Wv, ldwv);
					dlacpy("A", kln, jw, Wv, ldwv, &Z[krow+ldz*kwtop], ldz);
				}
			}
		}
		// Return the number of deflations ...
		nd = jw - ns;
		// ... and the number of shifts. (Subtracting infqr from the spike length takes care of the
		// case of a rare QR failure while calculating eigenvalues of the deflation window.)
		ns -= infqr;
		// Return optimal workspace.
		work[0] = real(lwkopt);
	}

	/*! §dlaqr4 computes the eigenvalues of a Hessenberg matrix, and optionally the matrices from
	 *  the Schur decomposition.
	 *
	 * §dlaqr4 implements one level of recursion for §dlaqr0. It is a complete implementation of
	 * the small bulge multi-shift QR algorithm. It may be called by §dlaqr0 and, for large enough
	 * deflation window size, it may be called by §dlaqr3. This subroutine is identical to §dlaqr0
	 * except that it calls §dlaqr2 instead of §dlaqr3.\n
	 * §dlaqr4 computes the eigenvalues of a Hessenberg matrix $H$ and, optionally, the matrices
	 * $T$ and $Z$ from the Schur decomposition $H = Z T Z^T$, where $T$ is an upper quasi-
	 * triangular matrix (the Schur form), and $Z$ is the orthogonal matrix of Schur vectors.\n
	 * Optionally $Z$ may be postmultiplied into an input orthogonal matrix $Q$ so that this
	 * routine can give the Schur factorization of a matrix $A$ which has been reduced to the
	 * Hessenberg form $H$ by the orthogonal matrix $Q$: $A = Q H Q^T = (QZ) T (QZ)^T$.
	 * \param[in] wantt
	 *     = §true:  the full Schur form $T$ is required.\n
	 *     = §false: only eigenvalues are required.
	 *
	 * \param[in] wantz
	 *     = §true: the matrix of Schur vectors $Z$ is required.\n
	 *     = §false: Schur vectors are not required.
	 *
	 * \param[in] n        The order of the matrix $H$. $\{n}\ge 0$.
	 * \param[in] ilo, ihi
	 *     It is assumed that $H$ is already upper triangular in rows and columns $0:\{ilo}-1$ and
	 *     $\{ihi}+1:\{n}-1$ and, if $\{ilo}>0$, $H[\{ilo},\{ilo}-1]$ is zero. §ilo and §ihi are
	 *     normally set by a previous call to §dgebal, and then passed to §dgehrd when the matrix
	 *     output by §dgebal is reduced to Hessenberg form. Otherwise, §ilo and §ihi should be set
	 *     to §0 and $\{n}-1$, respectively. If $\{n}>0$, then $0\le\{ilo}\le\{ihi}<\{n}$.\n
	 *     If $\{n}=0$, then $\{ilo}=0$ and $\{ihi}=-1$.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in,out] H
	 *     an array, dimension (§ldh,§n)\n
	 *     On entry, the upper Hessenberg matrix $H$.\n
	 *     On exit, if $\{info}=0$ and §wantt is §true, then §H contains the upper quasi-triangular
	 *     matrix $T$ from the Schur decomposition (the Schur form); 2 by 2 diagonal blocks
	 *     (corresponding to complex conjugate pairs of eigenvalues) are returned in standard form,
	 *     with $\{H}[i,i]=\{H}[i+1,i+1]$ and $\{H}[i+1,i]\{H}[i,i+1]<0$. If $\{info}=0$ and §wantt
	 *     is §false, then the contents of §H are unspecified on exit.\n
	 *     (The output value of §H when $\{info}>0$ is given under the description of §info below.)
	 *     \n This subroutine may explicitly set $\{H}[i,j]=0$ for $i>j$ and
	 *     $j=0,1,\ldots,\{ilo}-1$ or $j=\{ihi}+1,\{ihi}+2,\ldots,\{n}-1$.
	 *
	 * \param[in]  ldh    The leading dimension of the array §H. $\{ldh}\ge\max(1,\{n})$.
	 * \param[out] wr, wi
	 *     arrays, dimension ($\{ihi}+1$)\n
	 *     The real and imaginary parts, respectively, of the computed eigenvalues of
	 *     $\{H}[\{ilo}:\{ihi},\{ilo}:\{ihi}]$ are stored in $\{wr}[\{ilo}:\{ihi}]$ and
	 *     $\{wi}[\{ilo}:\{ihi}]$. If two eigenvalues are computed as a complex conjugate pair,
	 *     they are stored in consecutive elements of §wr and §wi, say the $i$-th and $(i+1)$th,
	 *     with $\{wi}[i]>0$ and $\{wi}[i+1]<0$. If §wantt is §true, then the eigenvalues are
	 *     stored in the same order as on the diagonal of the Schur form returned in §H, with
	 *     $\{wr}[i]=\{H}[i,i]$ and, if $\{H}[i:i+1,i:i+1]$ is a 2 by 2 diagonal block,
	 *     $\{wi}[i]=\sqrt{-\{H}[i+1,i]\{H}[i,i+1]}$ and $\{wi}[i+1]=-\{wi}[i]$.
	 *
	 * \param[in] iloz, ihiz
	 *     Specify the rows of §Z to which transformations must be applied if §wantz is §true.\n
	 *     $0\le\{iloz}\le\{ilo}$; $\{ihi}\le\{ihiz}<\{n}$.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in,out] Z
	 *     an array, dimension (§ldz,$\{ihi}+1$)\n
	 *     If §wantz is §false, then §Z is not referenced.\n
	 *     If §wantz is §true, then $\{Z}[\{ilo}:\{ihi},\{iloz}:\{ihiz}]$ is replaced by
	 *     $\{Z}[\{ilo}:\{ihi},\{iloz}:\{ihiz}]U$ where $U$ is the orthogonal Schur factor of
	 *     $\{H}[\{ilo}:\{ihi},\{ilo}:\{ihi}]$. (The output value of §Z when $\{info}>0$ is given
	 *     under the description of §info below.)
	 *
	 * \param[in] ldz
	 *     The leading dimension of the array §Z.\n
	 *     If §wantz is §true then $\{ldz}\ge\max(1,\{ihiz}+1)$. Otherwize, $\{ldz}\ge 1$.
	 *
	 * \param[out] work
	 *     an array, dimension (§lwork)\n
	 *     On exit, if $\{lwork}=-1$, $\{work}[0]$ returns an estimate of the optimal value for
	 *     §lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work. $\{lwork}\ge\max(1,\{n})$ is sufficient, but §lwork
	 *     typically as large as $6\{n}$ may be required for optimal performance. A workspace query
	 *     to determine the optimal workspace size is recommended.\n
	 *     If $\{lwork}=-1$, then §dlaqr4 does a workspace query. In this case, §dlaqr4 checks the
	 *     input parameters and estimates the optimal workspace size for the given values of §n,
	 *     §ilo and §ihi. The estimate is returned in $\{work}[0]$. No error message related to
	 *     §lwork is issued by §xerbla. Neither §H nor §Z are accessed.
	 *
	 * \param[out] info
	 *     \li =0: successful exit
	 *     \li >0: \n
	 *             - if $\{info}=i$, §dlaqr4 failed to compute all of the eigenvalues. Elements
	 *               $0:\{ilo}-1$ and $i:\{n}-1$ of §wr and §wi contain those eigenvalues which
	 *               have been successfully computed. (Failures are rare.)
	 *             - If $\{info}>0$ and §wantt is §false, then on exit, the remaining unconverged
	 *               eigenvalues are the eigenvalues of the upper Hessenberg matrix rows and
	 *               columns §ilo through $\{info}-1$ of the final, output value of §H.
	 *             - If $\{info}>0$ and §wantt is §true, then on exit\n
	 *                 $(\text{initial value of }\{H})U = U(\text{final value of }\{H})$&emsp;(*)\n
	 *               where $U$ is a orthogonal matrix. The final value of §H is upper Hessenberg
	 *               and triangular in rows and columns §info through §ihi.
	 *             - If $\{info}>0$ and §wantz is §true, then on exit\n
	 *                 $(\text{final value of }\{Z}[\{ilo}:\{ihi},\{iloz}:\{ihiz}])$
	 *                 $=(\text{initial value of }\{Z}[\{ilo}:\{ihi},\{iloz}:\{ihiz}])U$\n
	 *               where $U$ is the orthogonal matrix in (*) (regardless of the value of §wantt.)
	 *             - If $\{info}>0$ and §wantz is §false, then §Z is not accessed.
	 *             .
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Contributors:\n
	 *         Karen Braman and Ralph Byers, Department of Mathematics, University of Kansas, USA\n
	 *     References:\n
	 *         K. Braman, R. Byers and R. Mathias, The Multi-Shift QR Algorithm Part I: Maintaining
	 *         Well Focused Shifts, and Level 3 Performance, SIAM Journal of Matrix Analysis,
	 *         volume 23, pages 929--947, 2002.\n\n
	 *     K. Braman, R. Byers and R. Mathias, The Multi-Shift QR Algorithm Part II: Aggressive
	 *     Early Deflation, SIAM Journal of Matrix Analysis, volume 23, pages 948--973, 2002.    */
	static void dlaqr4(bool const wantt, bool const wantz, int const n, int const ilo,
	                   int const ihi, real* const H, int const ldh, real* const wr, real* const wi,
	                   int const iloz, int const ihiz, real* const Z, int const ldz,
	                   real* const work, int const lwork, int& info) /*const*/
	{
		// Matrices of order NTINY or smaller must be processed by dlahqr because of insufficient
		// subdiagonal scratch space. (This is a hard limit.)
		const int NTINY = 11;
		// Exceptional deflation windows: try to cure rare slow convergence by varying the size of
		// the deflation window after KEXNW iterations.
		const int KEXNW = 5;
		// Exceptional shifts: try to cure rare slow convergence with ad-hoc exceptional shifts
		// every KEXSH iterations.
		const int KEXSH = 6;
		// The constants WILK1 and WILK2 are used to form the exceptional shifts.
		const real WILK1 = real(0.75);
		const real WILK2 = real(-0.4375);
		info = 0;
		// Quick return for n = 0: nothing to do.
		if (n==0)
		{
			work[0] = ONE;
			return;
		}
		int lwkopt;
		if (n<=NTINY)
		{
			// Tiny matrices must use dlahqr.
			lwkopt = 1;
			if (lwork!=-1)
			{
				dlahqr(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi, iloz, ihiz, Z, ldz, info);
			}
		}
		else
		{
			// Use small bulge multi-shift QR with aggressive early deflation on larger-than-tiny
			// matrices.
			// Hope for the best.
			info = 0;
			// Set up job flags for ilaenv.
			char jbcmpz[3];
			if (wantt)
			{
				jbcmpz[0] = 'S';
			}
			else
			{
				jbcmpz[0] = 'E';
			}
			if (wantz)
			{
				jbcmpz[1] = 'V';
			}
			else
			{
				jbcmpz[1] = 'N';
			}
			jbcmpz[2] = '\0';
			// nwr = recommended deflation window size. At this point, n>NTINY=11, so there is
			// enough subdiagonal workspace for nwr>=2 as required.
			// (In fact, there is enough subdiagonal space for nwr>=3.)
			int nwr;
			nwr = ilaenv(13, "DLAQR4", jbcmpz, n, ilo+1, ihi+1, lwork);
			nwr = std::max(2, nwr);
			nwr = std::min(ihi-ilo+1, std::min((n-1)/3, nwr));
			// nsr = recommended number of simultaneous shifts. At this point n>NTINY=11, so there
			// is enough subdiagonal workspace for nsr to be even and greater than or equal to two
			// as required.
			int nsr;
			nsr = ilaenv(15, "DLAQR4", jbcmpz, n, ilo+1, ihi+1, lwork);
			nsr = std::min(nsr, std::min((n+6)/9, ihi-ilo));
			nsr = std::max(2, nsr-nsr%2);
			// Estimate optimal workspace
			// Workspace query call to dlaqr2
			int ld, ls;
			dlaqr2(wantt, wantz, n, ilo, ihi, nwr+1, H, ldh, iloz, ihiz, Z, ldz, ls, ld, wr,
			       wi, H, ldh, n, H, ldh, n, H, ldh, work, -1);
			// Optimal workspace = std::max(dlaqr5, dlaqr2)
			lwkopt = std::max(3*nsr/2, int(work[0]));
			// Quick return in case of workspace query.
			if (lwork==-1)
			{
				work[0] = real(lwkopt);
				return;
			}
			// dlahqr/dlaqr0 crossover point
			int nmin = std::max(NTINY, ilaenv(12, "DLAQR4", jbcmpz, n, ilo+1, ihi+1, lwork));
			// Nibble crossover point
			int nibble = std::max(0, ilaenv(14, "DLAQR4", jbcmpz, n, ilo+1, ihi+1, lwork));
			// Accumulate reflections during ttswp? Use block 2 by 2 structure during matrix-matrix
			// multiply?
			int kacc22 = std::max(0, ilaenv(16, "DLAQR4", jbcmpz, n, ilo+1, ihi+1, lwork));
			kacc22 = std::min(2, kacc22);
			// nwmax = the largest possible deflation window for which there is sufficient
			// workspace.
			int nwmax = std::min((n-1)/3, lwork/2);
			int nw = nwmax;
			// nsmax = the Largest number of simultaneous shifts for which there is sufficient
			// workspace.
			int nsmax = std::min((n+6)/9, 2*lwork/3);
			nsmax -= nsmax % 2;
			// ndfl: an iteration count restarted at deflation.
			int ndfl = 1;
			// itmax = iteration limit
			int itmax = std::max(30, 2*KEXSH) * std::max(10, (ihi-ilo+1));
			// Last row and column in the active block
			int kbot = ihi;
			// Main Loop
			real aa, bb, cc, cs, dd, sn, ss, swap;
			int i, inf, it, k, kdu, ks, kt, ktop, ku, kv, kwh, kwtop, kwv, ndec, nh, nho, ns, nve,
				nwupbd;
			bool sorted;
			for (it=0; it<itmax; it++)
			{
				// Done when kbot falls below ilo
				if (kbot<ilo)
				{
					break;
				}
				// Locate active block
				for (k=kbot; k>ilo; k--)
				{
					if (H[k+ldh*(k-1)]==ZERO)
					{
						break;
					}
				}
				ktop = k;
				/* Select deflation window size:
				 *     Typical Case:
				 *         If possible and advisable, nibble the entire active block. If not, use
				 *         size min(nwr,nwmax) or min(nwr+1,nwmax) depending upon which has the
				 *         smaller corresponding subdiagonal entry (a heuristic).
				 *     Exceptional Case:
				 *         If there have been no deflations in KEXNW or more iterations, then vary
				 *         the deflation window size. At first, because, larger windows are, in
				 *         general, more powerful than smaller ones, rapidly increase the window to
				 *         the maximum possible. Then, gradually reduce the window size.         */
				nh = kbot - ktop + 1;
				nwupbd = std::min(nh, nwmax);
				if (ndfl<KEXNW)
				{
					nw = std::min(nwupbd, nwr);
				}
				else
				{
					nw = std::min(nwupbd, 2*nw);
				}
				if (nw<nwmax)
				{
					if (nw>=nh-1)
					{
						nw = nh;
					}
					else
					{
						kwtop = kbot - nw + 1;
						if (std::fabs(H[kwtop+ldh*(kwtop-1)])
						    > std::fabs(H[kwtop-1+ldh*(kwtop-2)]))
						{
							nw++;
						}
					}
				}
				if (ndfl<KEXNW)
				{
					ndec = -1;
				}
				else if (ndec>=0 || nw>=nwupbd)
				{
					ndec++;
					if (nw-ndec<2)
					{
						ndec = 0;
					}
					nw -= ndec;
				}
				/* Aggressive early deflation:
				 *   split workspace under the subdiagonal into
				 *     - an nw by nw work array V in the lower left-hand-corner,
				 *     - an nw by at-least-nw-but-more-is-better (nw by nho) horizontal work array
				 *       along the bottom edge,
				 *     - an at-least-nw-but-more-is-better by nw (nho by nw) vertical work array
				 *       along the left-hand-edge.                                               */
				kv  = n - nw;
				kt  = nw;
				nho = (n-nw-1) - kt;
				kwv = nw + 1;
				nve = (n-nw) - kwv;
				// Aggressive early deflation
				dlaqr2(wantt, wantz, n, ktop, kbot, nw, H, ldh, iloz, ihiz, Z, ldz, ls, ld, wr, wi,
				       &H[kv], ldh, nho, &H[kv+ldh*kt], ldh, nve, &H[kwv], ldh, work, lwork);
				// Adjust kbot accounting for new deflations.
				kbot -= ld;
				// ks points to the shifts.
				ks = kbot - ls + 1;
				/* Skip an expensive QR sweep if there is a (partly heuristic) reason to expect
				 * that many eigenvalues will deflate without it. Here, the QR sweep is skipped if
				 * many eigenvalues have just been deflated or if the remaining active block is
				 * small.                                                                        */
				if (ld==0 || (100*ld<=nw*nibble && kbot-ktop+1>std::min(nmin, nwmax)))
				{
					/* ns = nominal number of simultaneous shifts. This may be lowered (slightly)
					 * if dlaqr2 did not provide that many shifts.                               */
					ns = std::min(std::min(nsmax, nsr), std::max(2, kbot-ktop));
					ns -= ns % 2;
					/* If there have been no deflations in a multiple of KEXSH iterations, then try
					 * exceptional shifts. Otherwise use shifts provided by dlaqr2 above or from
					 * the eigenvalues of a trailing principal submatrix.                        */
					if ((ndfl%KEXSH)==0)
					{
						ks = kbot - ns + 1;
						for (i=kbot; i>std::max(ks, ktop+1); i-=2)
						{
							ss = std::fabs(H[i+ldh*(i-1)]) + std::fabs(H[i-1+ldh*(i-2)]);
							aa = WILK1 * ss + H[i+ldh*i];
							bb = ss;
							cc = WILK2 * ss;
							dd = aa;
							dlanv2(aa, bb, cc, dd, wr[i-1], wi[i-1], wr[i], wi[i], cs, sn);
						}
						if (ks==ktop)
						{
							wr[ks+1] = H[ks+1+ldh*(ks+1)];
							wi[ks+1] = ZERO;
							wr[ks]   = wr[ks+1];
							wi[ks]   = wi[ks+1];
						}
					}
					else
					{
						/* Got ns/2 or fewer shifts? Use dlahqr on a trailing principal submatrix
						 * to get more. (Since ns<=nsmax<=(n+6)/9, there is enough space below the
						 * subdiagonal to fit an ns-by-ns scratch array.)                        */
						if (kbot-ks+1<=ns/2)
						{
							ks = kbot - ns + 1;
							kt = n - ns;
							dlacpy("A", ns, ns, &H[ks+ldh*ks], ldh, &H[kt], ldh);
							dlahqr(false, false, ns, 0, ns-1, &H[kt], ldh, &wr[ks], &wi[ks], 0, 0,
							       nullptr, 1, inf);
							ks += inf;
							/* In case of a rare QR failure use eigenvalues of the trailing 2 by 2
							 * principal submatrix.                                              */
							if (ks>=kbot)
							{
								aa = H[kbot-1+ldh*(kbot-1)];
								cc = H[kbot  +ldh*(kbot-1)];
								bb = H[kbot-1+ldh*kbot];
								dd = H[kbot  +ldh*kbot];
								dlanv2(aa, bb, cc, dd, wr[kbot-1], wi[kbot-1], wr[kbot], wi[kbot],
									   cs, sn);
								ks = kbot - 1;
							}
						}
						if (kbot-ks+1>ns)
						{
							// Sort the shifts (Helps a little)
							// Bubble sort keeps complex conjugate pairs together.
							sorted = false;
							for (k=kbot; k>ks; k--)
							{
								if (sorted)
								{
									break;
								}
								sorted = true;
								for (i=ks; i<k; i++)
								{
									if (std::fabs(wr[i])  +std::fabs(wi[i])
									  < std::fabs(wr[i+1])+std::fabs(wi[i+1]))
									{
										sorted = false;
										swap    = wr[i];
										wr[i]   = wr[i+1];
										wr[i+1] = swap;
										swap    = wi[i];
										wi[i]   = wi[i+1];
										wi[i+1] = swap;
									}
								}
							}
						}
						/* Shuffle shifts into pairs of real shifts and pairs of complex conjugate
						 * shifts assuming complex conjugate shifts are already adjacent to one
						 * another. (Yes, they are.)                                             */
						for (i=kbot; i>ks+1; i-=2)
						{
							if (wi[i]!=-wi[i-1])
							{
								swap    = wr[i];
								wr[i]   = wr[i-1];
								wr[i-1] = wr[i-2];
								wr[i-2] = swap;
								swap    = wi[i];
								wi[i]   = wi[i-1];
								wi[i-1] = wi[i-2];
								wi[i-2] = swap;
							}
						}
					}
					// If there are only two shifts and both are real, then use only one.
					if (kbot-ks+1==2)
					{
						if (wi[kbot]==ZERO)
						{
							if (std::fabs(wr[kbot]  -H[kbot+ldh*kbot])
							  < std::fabs(wr[kbot-1]-H[kbot+ldh*kbot]))
							{
								wr[kbot-1] = wr[kbot];
							}
							else
							{
								wr[kbot] = wr[kbot-1];
							}
						}
					}
					/* Use up to ns of the smallest magnitude shifts. If there aren't ns shifts
					 * available, then use them all, possibly dropping one to make the number of
					 * shifts even.                                                              */
					ns = std::min(ns, kbot-ks+1);
					ns -= ns % 2;
					ks = kbot - ns + 1;
					/* Small-bulge multi-shift QR sweep
					 *   split workspace under the subdiagonal into
					 *     - a kdu by kdu work array U in the lower left-hand-corner,
					 *     - a kdu by at-least-kdu-but-more-is-better (kdu by nho) horizontal work
					 *       array WH along the bottom edge,
					 *     - and an at-least-kdu-but-more-is-better by kdu (nve by kdu) vertical
					 *       work WV arrow along the left-hand-edge.                             */
					kdu = 3*ns - 3;
					ku  = n - kdu;
					kwh = kdu;
					nho = (n-kdu-4) - kdu + 1;
					kwv = kdu + 3;
					nve = n - kdu - kwv;
					// Small-bulge multi-shift QR sweep
					dlaqr5(wantt, wantz, kacc22, n, ktop, kbot, ns, &wr[ks], &wi[ks], H, ldh, iloz,
					       ihiz, Z, ldz, work, 3, &H[ku], ldh, nve, &H[kwv], ldh, nho,
					       &H[ku+ldh*kwh], ldh);
				}
				// Note progress (or the lack of it).
				if (ld>0)
				{
					ndfl = 1;
				}
				else
				{
					ndfl++;
				}
				// End of main loop
			}
			if (it>=itmax)
			{
				// Iteration limit exceeded. Set info to show where the problem occurred and exit.
				info = kbot + 1;
			}
		}
		// Return the optimal value of lwork.
		work[0] = real(lwkopt);
	}

	/*! §dlaqr5 performs a single small-bulge multi-shift QR sweep.
	 *
	 * §dlaqr5, called by §dlaqr0, performs a single small-bulge multi-shift QR sweep.
	 * \param[in] wantt
	 *     §wantt = §true if the quasi-triangular Schur factor is being computed.\n
	 *     §wantt is set to §false otherwise.
	 *
	 * \param[in] wantz
	 *     §wantz = §true if the orthogonal Schur factor is being computed.\n
	 *     §wantz is set to §false otherwise.
	 *
	 * \param[in] kacc22
	 *     integer with value 0, 1, or 2.\n
	 *     Specifies the computation mode of far-from-diagonal orthogonal updates.\n
	 *     =0: §dlaqr5 does not accumulate reflections and does not use matrix-matrix multiply to
	 *         update far-from-diagonal matrix entries.\n
	 *     =1: §dlaqr5 accumulates reflections and uses matrix-matrix multiply to update the
	 *         far-from-diagonal matrix entries.\n
	 *     =2: §dlaqr5 accumulates reflections, uses matrix-matrix multiply to update the
	 *         far-from-diagonal matrix entries, and takes advantage of 2-by-2 block structure
	 *         during matrix multiplies.
	 *
	 * \param[in] n
	 *     §n is the order of the Hessenberg matrix $H$ upon which this subroutine operates.
	 *
	 * \param[in] ktop, kbot
	 *     These are the first and last rows and columns of an isolated diagonal block upon which
	 *     the QR sweep is to be applied. It is assumed without a check that\n
	 *     &emsp;  either $\{ktop}=0$      or $H[\{ktop},\{ktop}-1]=0$\n
	 *     and\n
	 *     &emsp;  either $\{kbot}=\{n}-1$ or $H[\{kbot}+1,\{kbot}]=0$.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in] nshfts
	 *     §nshfts gives the number of simultaneous shifts. §nshfts must be positive and even.
	 *
	 * \param[in,out] sr, si
	 *     arrays, dimension (§nshfts)\n
	 *     §sr contains the real parts and §si contains the imaginary parts of the §nshfts shifts
	 *     of origin that define the multi-shift QR sweep.\n
	 *     On output §sr and §si may be reordered.
	 *
	 * \param[in,out] H
	 *     an array, dimension (§ldh,§n)\n
	 *     On input §H contains a Hessenberg matrix.\n
	 *     On output a multi-shift QR sweep with shifts $\{sr}[j]+i\{si}[j]$ is applied to the
	 *     isolated diagonal block in rows and columns §ktop through §kbot.
	 *
	 * \param[in] ldh
	 *     §ldh is the leading dimension of §H just as declared in the calling procedure.
	 *     $\{ldh}\ge\max(1,\{n})$.
	 *
	 * \param[in] iloz, ihiz
	 *     Specify the rows of §Z to which transformations must be applied if §wantz is §true. \n
	 *     $0\le\{iloz}\le\{ihiz}<\{n}$\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in,out] Z
	 *     an array, dimension (§ldz,§ihiz)\n
	 *     If §wantz = §true, then the QR Sweep orthogonal similarity transformation is accumulated
	 *     into $\{Z}[\{iloz}:\{ihiz},\{iloz}:\{ihiz}]$ from the right.\n
	 *     If §wantz = §false, then §Z is unreferenced.
	 *
	 * \param[in] ldz
	 *     §ldz is the leading dimension of §Z just as declared in the calling procedure.
	 *     $\{ldz}\ge\{n}$.
	 *
	 * \param[out] V   an array, dimension (§ldv,$\{nshfts}/2$)
	 * \param[in]  ldv
	 *     §ldv is the leading dimension of §V as declared in the calling procedure. $\{ldv}\ge 3$.
	 *
	 * \param[out] U   an array, dimension (§ldu,$3\{nshfts}-3$)
	 * \param[in]  ldu
	 *     §ldu is the leading dimension of §U just as declared in the in the calling subroutine.
	 *     $\{ldu}\ge 3\{nshfts}-3$.
	 *
	 * \param[in]  nv   §nv is the number of rows in §Wv agailable for workspace. $\{nv}\ge 1$.
	 * \param[out] Wv   an array, dimension (§ldwv,$3\{nshfts}-3$)
	 * \param[in]  ldwv
	 *     &ldwv is the leading dimension of §Wv as declared in the in the calling subroutine.
	 *     $\{ldwv}\ge\{nv}$.
	 *
	 * \param[in] nh
	 *     §nh is the number of columns in array §Wh available for workspace. $\{nh}\ge 1$.
	 *
	 * \param[out] Wh   an array, dimension (§ldwh,§nh)
	 * \param[in]  ldwh
	 *     Leading dimension of §Wh just as declared in the calling procedure.
	 *     $\{ldwh}\ge 3\{nshfts}-3$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016
	 * \remark
	 *     Contributors:\n
	 *     Karen Braman and Ralph Byers, Department of Mathematics, University of Kansas, USA\n\n
	 *     References:\n
	 *     K. Braman, R. Byers and R. Mathias, The Multi-Shift QR Algorithm Part I: Maintaining
	 *     Well Focused Shifts, and Level 3 Performance, SIAM Journal of Matrix Analysis,
	 *     volume 23, pages 929--947, 2002.                                                      */
	static void dlaqr5(bool const wantt, bool const wantz, int const kacc22, int const n,
	                   int const ktop, int const kbot, int const nshfts, real* const sr,
	                   real* const si, real* const H, int const ldh, int const iloz,
	                   int const ihiz, real* const Z, int const ldz, real* const V, int const ldv,
	                   real* const U, int const ldu, int const nv, real* const Wv, int const ldwv,
	                   int const nh, real* const Wh, int const ldwh) /*const*/
	{
		// If there are no shifts, then there is nothing to do.
		if (nshfts<2)
		{
			return;
		}
		// If the active block is empty or 1 by 1, then there is nothing to do.
		if (ktop>=kbot)
		{
			return;
		}
		// Shuffle shifts into pairs of real shifts and pairs of complex conjugate shifts assuming
		// complex conjugate shifts are already adjacent to one another.
		real swap;
		for (int i=0; i<nshfts-2; i+=2)
		{
			if (si[i]!=-si[i+1])
			{
				swap    = sr[i];
				sr[i]   = sr[i+1];
				sr[i+1] = sr[i+2];
				sr[i+2] = swap;
				swap    = si[i];
				si[i]   = si[i+1];
				si[i+1] = si[i+2];
				si[i+2] = swap;
			}
		}
		// nshfts is supposed to be even, but if it is odd, then simply reduce it by one. The
		// shuffle above ensures that the dropped shift is real and that the remaining shifts are
		// paired.
		int ns = nshfts - (nshfts%2);
		// Machine constants for deflation
		real safmin = dlamch("SAFE MINIMUM");
		real safmax = ONE / safmin;
		dlabad(safmin, safmax);
		real ulp = dlamch("PRECISION");
		real smlnum = safmin * (real(n)/ulp);
		// Use accumulated reflections to update far-from-diagonal entries?
		bool accum = (kacc22==1) || (kacc22==2);
		// If so, exploit the 2 by 2 block structure?
		bool blk22 = (ns>2) && (kacc22==2);
		// clear trash
		int hktop = ktop + ldh*ktop;
		if (ktop+2<=kbot)
		{
			H[hktop+2] = ZERO;
		}
		// nbmps = number of 2-shift bulges in the chain
		int nbmps = ns / 2;
		// kdu = width of slab
		int kdu = 6*nbmps - 3;
		// Create and chase chains of nbmps bulges
		real alpha, beta, h11, h12, h21, h22, refsum, scl, tst1, tst2;
		int hj, hk, hk1, hk2, hk3, i2, i4, incol, j, j2, j4, jbot, jcol, jlen, jrow, jtop, k, k1,
		    kms, knz, krcol, kzs, m, m22, mbot, mend, mstart, mtop, ndcol, nu, uind1, uind2, uind3,
		    vm, vm22, wvi2, wvkzs, zind1, zind2, zind3;
		bool bmp22;
		real vt[3];
		for (incol=3*(1-nbmps)+ktop; incol<=kbot+1; incol+=3*nbmps-2)
		{
			ndcol = incol + kdu - 1;
			if (accum)
			{
				dlaset("ALL", kdu, kdu, ZERO, ONE, U, ldu);
			}
			/* Near-the-diagonal bulge chase. The following loop performs the near-the-diagonal
			 * part of a small bulge multi-shift QR sweep. Each 6*nbmps-2 column diagonal chunk
			 * extends from column incol to column ndcol (including both column incol and column
			 * ndcol). The following loop chases a 3*nbmps column long chain of nbmps bulges
			 * 3*nbmps-2 columns to the right. (incol may be less than ktop and and ndcol may be
			 * greater than kbot indicating phantom columns from which to chase bulges before they
			 * are actually introduced or to which to chase bulges beyond column kbot.)          */
			for (krcol=incol-1; krcol<std::min(incol+3*nbmps-3, kbot-1); krcol++)
			{
				/* Bulges number mtop to mbot are active double implicit shift bulges. There may or
				 * may not also be small 2 by 2 bulge, if there is room. The inactive bulges
				 * (if any) must wait until the active bulges have moved down the diagonal to make
				 * room. The phantom matrix paradigm described above helps keep track.           */
				mtop = std::max(0, (ktop-krcol+1)/3);
				mbot = std::min(nbmps, (kbot-krcol)/3) - 1;
				m22 = mbot + 1;
				bmp22 = ((mbot<nbmps-1) && (krcol+3*m22)==(kbot-2));
				// Generate reflections to chase the chain right one column.
				// (The minimum value of k is ktop-1.)
				for (m=mtop; m<=mbot; m++)
				{
					k = krcol + 3*m;
					vm = ldv * m;
					if (k==ktop-1)
					{
						dlaqr1(3, &H[hktop], ldh, sr[2*m], si[2*m], sr[2*m+1], si[2*m+1], &V[vm]);
						alpha = V[vm];
						dlarfg(3, alpha, &V[1+vm], 1, V[vm]);
					}
					else
					{
						hk = k + ldh*k;
						hk1 = k + ldh*(k+1);
						hk2 = hk1 + ldh;
						beta    = H[hk+1];
						V[1+vm] = H[hk+2];
						V[2+vm] = H[hk+3];
						dlarfg(3, beta, &V[1+vm], 1, V[vm]);
						// A Bulge may collapse because of vigilant deflation or destructive
						// underflow.  In the underflow case, try the two-small-subdiagonals trick
						// to try to reinflate the bulge.
						if (H[hk+3]!=ZERO || H[3+hk1]!= ZERO || H[3+hk2]==ZERO)
						{
							// Typical case: not collapsed (yet).
							H[hk+1] = beta;
							H[hk+2] = ZERO;
							H[hk+3] = ZERO;
						}
						else
						{
							// Atypical case: collapsed. Attempt to reintroduce ignoring H[k+1,k]
							// and H[k+2,k]. If the fill resulting from the new reflector is too
							// large, then abandon it. Otherwise, use the new one.
							dlaqr1(3, &H[1+hk1], ldh, sr[2*m], si[2*m], sr[2*m+1], si[2*m+1], vt);
							alpha = vt[0];
							dlarfg(3, alpha, &vt[1], 1, vt[0]);
							refsum = vt[0] * (H[hk+1]+vt[1]*H[hk+2]);
							if (std::fabs(H[hk+2]-refsum*vt[1])+std::fabs(refsum*vt[2])
							    > ulp*(std::fabs(H[hk])+std::fabs(H[1+hk1])+std::fabs(H[2+hk2])))
							{
								// Starting a new bulge here would create non-negligible fill.
								// Use the old one with trepidation.
								H[hk+1] = beta;
								H[hk+2] = ZERO;
								H[hk+3] = ZERO;
							}
							else
							{
								// Stating a new bulge here would create only negligible fill.
								// Replace the old reflector with the new one.
								H[hk+1] -= refsum;
								H[hk+2] = ZERO;
								H[hk+3] = ZERO;
								V[vm]   = vt[0];
								V[1+vm] = vt[1];
								V[2+vm] = vt[2];
							}
						}
					}
				}
				// Generate a 2 by 2 reflection, if needed.
				k = krcol + 3*m22;
				vm22 = ldv * m22;
				if (bmp22)
				{
					if (k==ktop-1)
					{
						dlaqr1(2, &H[k+1+ldh*(k+1)], ldh, sr[2*m22], si[2*m22], sr[2*m22+1],
						       si[2*m22+1], &V[vm22]);
						beta = V[vm22];
						dlarfg(2, beta, &V[1+vm22], 1, V[vm22]);
					}
					else
					{
						hk = k + ldh*k;
						beta      = H[hk+1];
						V[1+vm22] = H[hk+2];
						dlarfg(2, beta, &V[1+vm22], 1, V[vm22]);
						H[hk+1] = beta;
						H[hk+2] = ZERO;
					}
				}
				// Multiply H by reflections from the left
				if (accum)
				{
					jbot = std::min(ndcol, kbot);
				}
				else if (wantt)
				{
					jbot = n - 1;
				}
				else
				{
					jbot = kbot;
				}
				for (j=std::max(ktop, krcol); j<=jbot; j++)
				{
					mend = std::min(mbot, (j-krcol-1)/3);
					hj = ldh * j;
					for (m=mtop; m<=mend; m++)
					{
						k = krcol + 3*m;
						vm = ldv * m;
						refsum = V[vm] * (H[k+1+hj]+V[1+vm]*H[k+2+hj]+V[2+vm]*H[k+3+hj]);
						H[k+1+hj] -= refsum;
						H[k+2+hj] -= refsum*V[1+vm];
						H[k+3+hj] -= refsum*V[2+vm];
					}
				}
				if (bmp22)
				{
					k = krcol + 3*m22;
					for (j=std::max(k+1, ktop); j<=jbot; j++)
					{
						hj = k + ldh*j;
						refsum = V[vm22] * (H[1+hj]+V[1+vm22]*H[2+hj]);
						H[1+hj] -= refsum;
						H[2+hj] -= refsum*V[1+vm22];
					}
				}
				// Multiply H by reflections from the right. Delay filling in the last row until
				// the vigilant deflation check is complete.
				if (accum)
				{
					jtop = std::max(ktop, incol-1);
				}
				else if (wantt)
				{
					jtop = 0;
				}
				else
				{
					jtop = ktop;
				}
				for (m=mtop; m<=mbot; m++)
				{
					vm = ldv * m;
					if (V[vm]!=ZERO)
					{
						k = krcol + 3*m;
						hk1 = ldh * (k+1);
						hk2 = hk1 + ldh;
						hk3 = hk2 + ldh;
						for (j=jtop; j<=std::min(kbot, k+3); j++)
						{
							refsum = V[vm] * (H[j+hk1]+V[1+vm]*H[j+hk2]+V[2+vm]*H[j+hk3]);
							H[j+hk1] -= refsum;
							H[j+hk2] -= refsum*V[1+vm];
							H[j+hk3] -= refsum*V[2+vm];
						}
						if (accum)
						{
							// Accumulate U. (If necessary, update U later with with an efficient
							// matrix-matrix multiply.)
							kms = k - incol;
							uind1 = ldu * (kms+1);
							uind2 = uind1 + ldu;//ldu * (kms+2);
							uind3 = uind2 + ldu;//ldu * (kms+3);
							for (j=std::max(0, ktop-incol); j<kdu; j++)
							{
								refsum = V[vm]
								         * (U[j+uind1]+V[1+vm]*U[j+uind2]+V[2+vm]*U[j+uind3]);
								U[j+uind1] -= refsum;
								U[j+uind2] -= refsum*V[1+vm];
								U[j+uind3] -= refsum*V[2+vm];
							}
						}
						else if (wantz)
						{
							// U is not accumulated, so update Z now by multiplying by reflections
							// from the right.
							zind1 = ldz * (k+1);
							zind2 = zind1 + ldz;//ldz * (k+2);
							zind3 = zind2 + ldz;//ldz * (k+3);
							for (j=iloz; j<=ihiz; j++)
							{
								refsum = V[vm]
								         * (Z[j+zind1]+V[1+vm]*Z[j+zind2]+V[2+vm]*Z[j+zind3]);
								Z[j+zind1] -= refsum;
								Z[j+zind2] -= refsum*V[1+vm];
								Z[j+zind3] -= refsum*V[2+vm];
							}
						}
					}
				}
				// Special case: 2 by 2 reflection (if needed)
				k = krcol + 3*m22;
				hk1 = ldh * (k+1);
				if (bmp22)
				{
					if (V[vm22]!=ZERO)
					{
						hk2 = hk1 + ldh;
						for (j=jtop; j<=std::min(kbot, k+3); j++)
						{
							refsum = V[vm22] * (H[j+hk1]+V[1+vm22]*H[j+hk2]);
							H[j+hk1] -= refsum;
							H[j+hk2] -= refsum*V[1+vm22];
						}
						if (accum)
						{
							kms = k - incol;
							uind1 = ldu * (kms+1);
							uind2 = uind1 + ldu;//ldu * (kms+2);
							for (j=std::max(0, ktop-incol); j<kdu; j++)
							{
								refsum = V[vm22] * (U[j+uind1]+V[1+vm22]*U[j+uind2]);
								U[j+uind1] -= refsum;
								U[j+uind2] -= refsum*V[1+vm22];
							}
						}
						else if (wantz)
						{
							zind1 = ldz * (k+1);
							zind2 = zind1 + ldz;//ldz * (k+2);
							for (j=iloz; j<=ihiz; j++)
							{
								refsum = V[vm22] * (Z[j+zind1]+V[1+vm22]*Z[j+zind2]);
								Z[j+zind1] -= refsum;
								Z[j+zind2] -= refsum*V[1+vm22];
							}
						}
					}
				}
				// Vigilant deflation check
				mstart = mtop;
				if (krcol+3*mstart<ktop)
				{
					mstart++;
				}
				mend = mbot;
				if (bmp22)
				{
					mend++;
				}
				if (krcol==kbot-2)
				{
					mend++;
				}
				for (m=mstart; m<=mend; m++)
				{
					k = std::min(kbot-1, krcol+3*m);
					hk = k + ldh*k;
					hk1 = k + ldh*(k+1);
					/* The following convergence test requires that the tradition small-compared-
					 * to-nearby-diagonals criterion and the Ahues & Tisseur (LAWN 122, 1997)
					 * criteria both be satisfied. The latter improves accuracy in some examples.
					 * Falling back on an alternate convergence criterion when tst1 or tst2 is zero
					 * (as done here) is traditional but probably unnecessary.                   */
					if (H[hk+1]!=ZERO)
					{
						tst1 = std::fabs(H[hk]) + std::fabs(H[1+hk1]);
						if (tst1==ZERO)
						{
							if (k>=ktop+1)
							{
								tst1 += std::fabs(H[k+ldh*(k-1)]);
							}
							if (k>=ktop+2)
							{
								tst1 += std::fabs(H[k+ldh*(k-2)]);
							}
							if (k>=ktop+3)
							{
								tst1 += std::fabs(H[k+ldh*(k-3)]);
							}
							if (k<=kbot-2)
							{
								tst1 += std::fabs(H[2+hk1]);
							}
							if (k<=kbot-3)
							{
								tst1 += std::fabs(H[3+hk1]);
							}
							if (k<=kbot-4)
							{
								tst1 += std::fabs(H[4+hk1]);
							}
						}
						if (std::fabs(H[hk+1])<=std::max(smlnum, ulp*tst1))
						{
							h12 = std::max(std::fabs(H[1+hk]),  std::fabs(H[hk1]));
							h21 = std::min(std::fabs(H[1+hk]),  std::fabs(H[hk1]));
							h11 = std::max(std::fabs(H[1+hk1]), std::fabs(H[hk]-H[1+hk1]));
							h22 = std::min(std::fabs(H[1+hk1]), std::fabs(H[hk]-H[1+hk1]));
							scl = h11 + h12;
							tst2 = h22 * (h11/scl);
							if (tst2==ZERO || h21*(h12/scl)<=std::max(smlnum, ulp*tst2))
							{
								H[1+hk] = ZERO;
							}
						}
					}
				}
				// Fill in the last row of each bulge.
				mend = std::min(nbmps, (kbot-krcol-1)/3) - 1;
				for (m=mtop; m<=mend; m++)
				{
					k = krcol + 3*m;
					hk1 = k + 4 + ldh*(k+1);
					hk2 = hk1 + ldh;
					hk3 = hk2 + ldh;
					vm = ldv * m;
					refsum = V[vm] * V[2+vm] * H[hk3];
					H[hk1] = -refsum;
					H[hk2] = -refsum*V[1+vm];
					H[hk3] -= refsum*V[2+vm];
				}
				// End of near-the-diagonal bulge chase.
			}
			// Use U (if accumulated) to update far-from-diagonal entries in H.
			// If required, use U to update Z as well.
			if (accum)
			{
				if (wantt)
				{
					jtop = 0;
					jbot = n - 1;
				}
				else
				{
					jtop = ktop;
					jbot = kbot;
				}
				if ((!blk22) || (incol<=ktop) || (ndcol>kbot) || (ns<=2))
				{
					/* Updates not exploiting the 2 by 2 block structure of U. k1 and nu keep track
					 * of the location and size of U in the special cases of introducing bulges and
					 * chasing bulges off the bottom. In these special cases and in case the number
					 * of shifts is ns = 2, there is no 2 by 2 block structure to exploit.       */
					k1 = std::max(0, ktop-incol);
					uind1 = k1 + ldu*k1;
					nu = (kdu-std::max(0, ndcol-kbot)) - k1;
					// Horizontal Multiply
					hk1 = incol + k1;
					for (jcol=std::min(ndcol, kbot)+1; jcol<=jbot; jcol+=nh)
					{
						hk2 = hk1 + ldh*jcol;
						jlen = std::min(nh, jbot-jcol+1);
						Blas<real>::dgemm("C", "N", nu, jlen, nu, ONE, &U[uind1], ldu, &H[hk2],
						                  ldh, ZERO, Wh, ldwh);
						dlacpy("ALL", nu, jlen, Wh, ldwh, &H[hk2], ldh);
					}
					// Vertical multiply
					hk1 = ldh * (incol+k1);
					for (jrow=jtop; jrow<std::max(ktop, incol-1); jrow+=nv)
					{
						hk2 = jrow + hk1;
						jlen = std::min(nv, std::max(ktop, incol-1)-jrow);
						Blas<real>::dgemm("N", "N", jlen, nu, nu, ONE, &H[hk2], ldh, &U[uind1],
						                  ldu, ZERO, Wv, ldwv);
						dlacpy("ALL", jlen, nu, Wv, ldwv, &H[hk2], ldh);
					}
					// Z multiply (also vertical)
					if (wantz)
					{
						zind1 = ldz * (incol+k1);
						for (jrow=iloz; jrow<=ihiz; jrow+=nv)
						{
							zind2 = jrow + zind1;
							jlen = std::min(nv, ihiz-jrow+1);
							Blas<real>::dgemm("N", "N", jlen, nu, nu, ONE, &Z[zind2], ldz,
							                  &U[uind1], ldu, ZERO, Wv, ldwv);
							dlacpy("ALL", jlen, nu, Wv, ldwv, &Z[zind2], ldz);
						}
					}
				}
				else
				{
					// Updates exploiting U's 2 by 2 block structure.
					// (i2-1, i4-1, j2-1, j4-1 are the last rows and columns of the blocks.)
					i2 = (kdu+1) / 2;
					i4 = kdu;
					j2 = i4 - i2;
					j4 = kdu;
					// kzs and knz deal with the band of zeros along the diagonal of one of the
					// triangular blocks.
					kzs = (j4-j2) - (ns+1);
					knz = ns + 1;
					uind1 = j2 + ldu*kzs;
					uind2 = ldu * i2;
					uind3 = j2 + uind2;
					wvi2  = ldwv * i2;
					wvkzs = ldwv * kzs;
					// Horizontal multiply
					for (jcol=std::min(ndcol, kbot)+1; jcol<=jbot; jcol+=nh)
					{
						hk1 = incol + ldh*jcol;
						jlen = std::min(nh, jbot-jcol+1);
						// Copy bottom of H to top+kzs of scratch
						// (The first kzs rows get multiplied by zero.)
						dlacpy("ALL", knz, jlen, &H[j2+hk1], ldh, &Wh[kzs], ldwh);
						// Multiply by U21^T
						dlaset("ALL", kzs, jlen, ZERO, ZERO, Wh, ldwh);
						Blas<real>::dtrmm("L", "U", "C", "N", knz, jlen, ONE, &U[uind1], ldu,
						                  &Wh[kzs], ldwh);
						// Multiply top of H by U11^T
						Blas<real>::dgemm("C", "N", i2, jlen, j2, ONE, U, ldu, &H[hk1], ldh, ONE,
						                  Wh, ldwh);
						// Copy top of H to bottom of Wh
						dlacpy("ALL", j2, jlen, &H[hk1], ldh, &Wh[i2], ldwh);
						// Multiply by U21^T
						Blas<real>::dtrmm("L", "L", "C", "N", j2, jlen, ONE, &U[uind2], ldu,
						                  &Wh[i2], ldwh);
						// Multiply by U22
						Blas<real>::dgemm("C", "N", i4-i2, jlen, j4-j2, ONE, &U[uind3], ldu,
						                  &H[j2+hk1], ldh, ONE, &Wh[i2], ldwh);
						// Copy it back
						dlacpy("ALL", kdu, jlen, Wh, ldwh, &H[hk1], ldh);
					}
					// Vertical multiply
					hk1 = ldh * (incol+j2);
					hk2 = ldh * incol;
					for (jrow=jtop; jrow<std::max(incol-1, ktop); jrow+=nv)
					{
						hk3 = jrow + hk2;
						jlen = std::min(nv, std::max(incol-1, ktop)-jrow);
						// Copy right of H to scratch
						// (the first kzs columns get multiplied by zero)
						dlacpy("ALL", jlen, knz, &H[jrow+hk1], ldh, &Wv[wvkzs], ldwv);
						// Multiply by U21
						dlaset("ALL", jlen, kzs, ZERO, ZERO, Wv, ldwv);
						Blas<real>::dtrmm("R", "U", "N", "N", jlen, knz, ONE, &U[uind1], ldu,
						                  &Wv[wvkzs], ldwv);
						// Multiply by U11
						Blas<real>::dgemm("N", "N", jlen, i2, j2, ONE, &H[hk3], ldh, U, ldu, ONE,
						                  Wv, ldwv);
						// Copy left of H to right of scratch
						dlacpy("ALL", jlen, j2, &H[hk3], ldh, &Wv[wvi2], ldwv);
						// Multiply by U21
						Blas<real>::dtrmm("R", "L", "N", "N", jlen, i4-i2, ONE, &U[uind2], ldu,
						                  &Wv[wvi2], ldwv);
						// Multiply by U22
						Blas<real>::dgemm("N", "N", jlen, i4-i2, j4-j2, ONE, &H[jrow+hk1], ldh,
						                  &U[uind3], ldu, ONE, &Wv[wvi2], ldwv);
						// Copy it back
						dlacpy("ALL", jlen, kdu, Wv, ldwv, &H[hk3], ldh);
					}
					// Multiply Z (also vertical)
					if (wantz)
					{
						zind1 = ldz * (incol+j2);
						zind2 = ldz * incol;
						for (jrow=iloz; jrow<=ihiz; jrow+=nv)
						{
							zind3 = jrow + zind2;
							jlen = std::min(nv, ihiz-jrow+1);
							// Copy right of Z to left of scratch
							// (first kzs columns get multiplied by zero)
							dlacpy("ALL", jlen, knz, &Z[jrow+zind1], ldz, &Wv[wvkzs], ldwv);
							// Multiply by U12
							dlaset("ALL", jlen, kzs, ZERO, ZERO, Wv, ldwv);
							Blas<real>::dtrmm("R", "U", "N", "N", jlen, knz, ONE, &U[uind1], ldu,
							                  &Wv[wvkzs], ldwv);
							// Multiply by U11
							Blas<real>::dgemm("N", "N", jlen, i2, j2, ONE, &Z[zind3], ldz, U, ldu,
							                  ONE, Wv, ldwv);
							// Copy left of Z to right of scratch
							dlacpy("ALL", jlen, j2, &Z[zind3], ldz, &Wv[wvi2], ldwv);
							// Multiply by U21
							Blas<real>::dtrmm("R", "L", "N", "N", jlen, i4-i2, ONE, &U[uind2],
							                  ldu, &Wv[wvi2], ldwv);
							// Multiply by U22
							Blas<real>::dgemm("N", "N", jlen, i4-i2, j4-j2, ONE, &Z[jrow+zind1],
							                  ldz, &U[uind3], ldu, ONE, &Wv[wvi2], ldwv);
							// Copy the result back to Z
							dlacpy("ALL", jlen, kdu, Wv, ldwv, &Z[zind3], ldz);
						}
					}
				}
			}
		}
	}

	/*! §dlarf applies an elementary reflector to a general rectangular matrix.
	 *
	 * §dlarf applies a real elementary reflector $H$ to a real §m by §n matrix $C$, from either
	 * the left or the right. $H$ is represented in the form\n
	 *     $H = I - \tau v v^T$\n
	 * where $\tau$ is a real scalar and $v$ is a real vector.
	 * If $\tau=0$, then $H$ is taken to be the unit matrix.
	 * \param[in] side
	 *     'L': form $H C$\n
	 *     'R': form $C H$
	 *
	 * \param[in] m The number of rows of the matrix $C$.
	 * \param[in] n The number of columns of the matrix $C$.
	 * \param[in] v
	 *     an array, dimension ($1+(\{m}-1)*|\{incv}|$) if §side = 'L'\n
	 *                      or ($1+(\{n}-1)*|\{incv}|$) if §side = 'R'\n
	 *     The vector $v$ in the representation of $H$. §v is not used if §tau = 0.
	 *
	 * \param[in]     incv The increment between elements of §v. $\{incv}\ne 0$.
	 * \param[in]     tau  The value $\tau$ in the representation of $H$.
	 * \param[in,out] C
	 *     an array, dimension (§ldc,§n)\n
	 *     On entry, the §m by §n matrix $C$.\n
	 *     On exit, §C is overwritten by the matrix\n $H C$ if §side = 'L',
	 *                                           or\n $C H$ if §side = 'R'.
	 *
	 * \param[in]  ldc  The leading dimension of the array §C. $\{ldc}\ge\max(1,\{m})$.
	 * \param[out] work
	 *     an array, dimension (§n) if §side = 'L'\n
	 *     &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;or (§m) if §side = 'R'
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlarf(char const* const side, int const m, int const n, real const* const v,
	                  int const incv, real const tau, real* const C, int const ldc,
	                  real* const work) /*const*/
	{
		int i, lastv=0, lastc=0;
		bool applyleft = (std::toupper(side[0])=='L');
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
				// Scan for the last non-zero column in C[0:lastv-1,:].
				lastc = iladlc(lastv, n, C, ldc) + 1;
			}
			else
			{
				// Scan for the last non-zero row in C[:,0:lastv-1].
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

	/*! §dlarfb applies a block reflector or its transpose to a general rectangular matrix.
	 *
	 * §dlarfb applies a real block reflector $H$ or its transpose $H^T$ to a real §m by §n matrix
	 * $C$, from either the left or the right.
	 * \param[in] side
	 *     'L': apply $H$ or $H^T$ from the Left\n
	 *     'R': apply $H$ or $H^T$ from the Right
	 *
	 * \param[in] trans
	 *     'N': apply $H$ (No transpose)\n
	 *     'T': apply $H^T$ (Transpose)
	 *
	 * \param[in] direct
	 *     Indicates how $H$ is formed from a product of elementary reflectors\n
	 *     'F': $H = H(0) H(1) \ldots H(k-1)$ (Forward)\n
	 *     'B': $H = H(k-1) \ldots H(1) H(0)$ (Backward)
	 *
	 * \param[in] storev
	 *     Indicates how the vectors which define the elementary reflectors are stored:\n
	 *     'C': Columnwise\n 'R': Rowwise
	 *
	 * \param[in] m The number of rows of the matrix $C$.
	 * \param[in] n The number of columns of the matrix $C$.
	 * \param[in] k
	 *     The order of the matrix $T$ (= the number of elementary reflectors whose product defines
	 *     the block reflector).
	 *
	 * \param[in] V
	 *     an array, dimension
	 *         - (§ldv,§k) if §storev = 'C'
	 *         - (§ldv,§m) if §storev = 'R' and §side = 'L'
	 *         - (§ldv,§n) if §storev = 'R' and §side = 'R'
	 *         .
	 *     The matrix $V$. See remarks.
	 *
	 * \param[in] ldv
	 *     The leading dimension of the array §V.\n
	 *     If §storev = 'C' and §side = 'L', $\{ldv}\ge\max(1,\{m})$;\n
	 *     if §storev = 'C' and §side = 'R', $\{ldv}\ge\max(1,\{n})$;\n
	 *     if §storev = 'R', $\{ldv} \ge \{k}$.
	 *
	 * \param[in] T
	 *     an array, dimension (§ldt,§k)\n
	 *     The triangular §k by §k matrix $T$ in the representation of the block reflector.
	 *
	 * \param[in]     ldt The leading dimension of the array §T. $\{ldt}\ge\{k}$.
	 * \param[in,out] C
	 *     an array, dimension (§ldc,§n)\n
	 *     On entry, the §m by §n matrix $C$.\n
	 *     On exit, §C is overwritten by $H C$ or $H^T C$ or $C H$ or $C H^T$.
	 *
	 * \param[in]  ldc    The leading dimension of the array §C. $\{ldc}\ge\max(1,\{m})$.
	 * \param[out] Work   an array, dimension (§ldwork,§k)
	 * \param[in]  ldwork
	 *     The leading dimension of the array §Work.\n
	 *     If §side = 'L', $\{ldwork}\ge\max(1,\{n})$;
	 *     if §side = 'R', $\{ldwork}\ge\max(1,\{m})$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2013
	 * \remark
	 *     The shape of the matrix $V$ and the storage of the vectors which define the $H(i)$ is
	 *     best illustrated by the following example with §n = 5 and §k = 3. The elements equal to
	 *     1 are not stored; the corresponding array elements are modified but restored on exit.
	 *     The rest of the array is not used.\n
	 *     §direct = 'F' and §storev = 'C':\n
	 *         $V=\b{bm}  1  &     &     \\
	 *                   v_1 &  1  &     \\
	 *                   v_1 & v_2 &  1  \\
	 *                   v_1 & v_2 & v_3 \\
	 *                   v_1 & v_2 & v_3 \e{bm}$\n\n
	 *     §direct = 'F' and §storev = 'R':\n
	 *         $V=\b{bm} 1 & v_1 & v_1 & v_1 & v_1 \\
	 *                   1 & v_2 & v_2 & v_2 &     \\
	 *                   1 & v_3 & v_3 &     &     \e{bm}$\n\n
	 *     §direct = 'B' and §storev = 'C':\n
	 *         $V=\b{bm} v_1 & v_2 & v_3 \\
	 *                   v_1 & v_2 & v_3 \\
	 *                    1  & v_2 & v_3 \\
	 *                    1  & v_3 &     \\
	 *                    1  &     &     \e{bm}$\n\n
	 *     §direct = 'B' and §storev = 'R':\n
	 *         $V=\b{bm} v_1 & v_1 &  1  &     &   \\
	 *                   v_2 & v_2 & v_2 & 1   &   \\
	 *                   v_3 & v_3 & v_3 & v_3 & 1 \e{bm}$                                       */
	static void dlarfb(char const* const side, char const* const trans, char const* const direct,
	                   char const* const storev, int const m, int const n, int const k,
	                   real const* const V, int const ldv, real const* const T, int const ldt,
	                   real* const C, int const ldc, real* const Work, int const ldwork) /*const*/
	{
		// Quick return if possible
		if (m<=0 || n<=0)
		{
			return;
		}
		char const* transt;
		if (std::toupper(trans[0])=='N')
		{
			transt = "Transpose";
		}
		else
		{
			transt = "No transpose";
		}
		int i, j, ccol, workcol;
		char upstorev = std::toupper(storev[0]);
		char updirect = std::toupper(direct[0]);
		char upside   = std::toupper(side[0]);
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
					// W = W * V1
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
					ccol = m - k;
					for (j=0; j<k; j++)
					{
						Blas<real>::dcopy(n, &C[ccol+j], ldc, &Work[ldwork*j], 1);
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

	/*! §dlarfg generates an elementary reflector (Householder matrix).
	 *
	 * §dlarfg generates a real elementary reflector $H$ of order §n, such that\n
	 *     $H \b{bm} \alpha \\
	 *                 x    \e{bm} = \b{bm} \beta \\
	 *                                        0   \e{bm},\hspace{20pt}
	 *      H^T H = I$.\n
	 * where $\alpha$ and $\beta$ are scalars, and $x$ is an (§n-1)-element real vector.
	 * $H$ is represented in the form\n
	 *     $H = I - \tau \b{bm} 1 \\
	 *                          v \e{bm}\b{bm} 1 & v^T \e{bm}$,
	 * where $\tau$ is a real scalar and $v$ is a real (§n-1)-element vector.
	 * If the elements of $x$ are all zero, then $\tau=0$ and $H$ is taken to be the unit matrix.
	 * Otherwise $1 \le \tau \le 2$.
	 * \param[in]     n     The order of the elementary reflector.
	 * \param[in,out] alpha
	 *     On entry, the value $\alpha$.\n
	 *     On exit, it is overwritten with the value $\beta$.
	 *
	 * \param[in,out] x
	 *     array, dimension ($1+(\{n}-2)*|\{incx}|$)\n
	 *     On entry, the vector $x$.\n
	 *     On exit, it is overwritten with the vector $v$.
	 *
	 * \param[in]  incx The increment between elements of §x. $\{incx}>0$.
	 * \param[out] tau  The value $\tau$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date November 2017                                                                       */
	static void dlarfg(int const n, real& alpha, real* const x, int const incx, real& tau)/*const*/
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
			if (std::fabs(beta)<safmin)
			{
				// xnorm, beta may be inaccurate; scale x and recompute them
				rsafmin = ONE / safmin;
				do
				{
					knt++;
					Blas<real>::dscal(n-1, rsafmin, x, incx);
					beta *= rsafmin;
					alpha *= rsafmin;
				} while (std::fabs(beta)<safmin && knt<20);
				// New beta is at most 1, at least safmin
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

	/*! §dlarft forms the triangular factor $T$ of a block reflector $H = I - VTV^H$
	 *
	 * §dlarft forms the triangular factor $T$ of a real block reflector $H$ of order §n,
	 * which is defined as a product of §k elementary reflectors.\n
	 *     If §direct = 'F', $H = H(0) H(1) \ldots H(k-1)$ and $T$ is upper triangular;\n
	 *     If §direct = 'B', $H = H(k-1) \ldots H(1) H(0)$ and $T$ is lower triangular.\n
	 *     If §storev = 'C', the vector which defines the elementary reflector $H(i)$ is stored in
	 *                       the $i$-th column of the array §V, and $H = I - V T V^T$.\n
	 *     If §storev = 'R', the vector which defines the elementary reflector $H(i)$ is stored in
	 *                       the $i$-th row of the array §V, and $H = I - V^T T V$.
	 * \param[in] direct
	 *     Specifies the order in which the elementary reflectors are multiplied to form the block
	 *     reflector:\n
	 *         'F': $H = H(0) H(1) \ldots H(k-1)$ (Forward)\n
	 *         'B': $H = H(k-1) \ldots H(1) H(0)$ (Backward)\n
	 *
	 * \param[in] storev
	 *     Specifies how the vectors which define the elementary reflectors are stored
	 *     (see also remarks):\n 'C': columnwise\n 'R': rowwise
	 *
	 * \param[in] n The order of the block reflector $H$. $\{n} \ge 0$.
	 * \param[in] k
	 *     The order of the triangular factor $T$(= the number of elementary reflectors).
	 *     $\{k}\ge 1$.
	 *
	 * \param[in] V
	 *     an array, dimension\n (§ldv,§k) if §storev = 'C'\n
	 *                           (§ldv,§n) if §storev = 'R'
	 *
	 * \param[in] ldv
	 *     The leading dimension of the array §V.\n
	 *     If §storev = 'C', $\{ldv}\ge\max(1,\{n})$;\n if §storev = 'R', $\{ldv}\ge\{k}$.
	 *
	 * \param[in] tau
	 *     an array, dimension (§k)\n
	 *     §tau[$i$] must contain the scalar factor of the elementary reflector $H(i)$.
	 *
	 * \param[out] T
	 *     an array, dimension (§ldt,§k)\n
	 *     The §k by §k triangular factor $T$ of the block reflector.\n
	 *     If §direct = 'F', $T$ is upper triangular;\n
	 *     if §direct = 'B', $T$ is lower triangular. The rest of the array is not used.
	 *
	 * \param[in] ldt The leading dimension of the array §T. $\{ldt}\ge\{k}$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     The shape of the matrix $V$ and the storage of the vectors which define the $H(i)$ is
	 *     best illustrated by the following example with §n = 5 and §k = 3. The elements equal to
	 *     1 are not stored.\n
	 *     §direct = 'F' and §storev = 'C':\n
	 *         $V = \b{bm}  1  &     &     \\
	 *                     v_1 &  1  &     \\
	 *                     v_1 & v_2 &  1  \\
	 *                     v_1 & v_2 & v_3 \\
	 *                     v_1 & v_2 & v_3 \e{bm}$\n\n
	 *     §direct = 'F' and §storev = 'R':\n
	 *         $V = \b{bm} 1 & v_1 & v_1 & v_1 & v_1 \\
	 *                       &  1  & v_2 & v_2 & v_2 \\
	 *                       &     &  1  & v_3 & v_3 \e{bm}$\n\n
	 *     §direct = 'B' and §storev = 'C':\n
	 *         $V = \b{bm} v_1 & v_2 & v_3 \\
	 *                     v_1 & v_2 & v_3 \\
	 *                      1  & v_2 & v_3 \\
	 *                         &  1  & v_3 \\
	 *                         &     &  1  \e{bm}$\n\n
	 *     §direct = 'B' and §storev = 'R':\n
	 *         $V = \b{bm} v_1 & v_1 &  1  &     &   \\
	 *                     v_2 & v_2 & v_2 &  1  &   \\
	 *                     v_3 & v_3 & v_3 & v_3 & 1 \e{bm}$                                     */
	static void dlarft(char const* const direct, char const* const storev, int const n,
	                   int const k, real const* const V, int const ldv, real const* const tau,
	                   real* const T, int const ldt) /*const*/
	{
		// Quick return if possible
		if (n==0)
		{
			return;
		}
		char updirect = std::toupper(direct[0]);
		char upstorev = std::toupper(storev[0]);
		int i, j, prevlastv, lastv, tcoli, vcol;
		if (updirect=='F')
		{
			prevlastv = n-1;
			for (i=0; i<k; i++)
			{
				tcoli = ldt * i;
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
					Blas<real>::dtrmv("Upper", "No transpose", "Non-unit", i, T, ldt, &T[tcoli],
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
				tcoli = ldt * i;
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
							vcol = n - k - i;
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
						                  &T[i+1+ldt*(i+1)], ldt, &T[i+1+tcoli], 1);
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

	/*! §dlarfx applies an elementary reflector to a general rectangular matrix, with loop
	 *  unrolling when the reflector has order $\le 10$.
	 *
	 * §dlarfx applies a real elementary reflector $H$ to a real §m by §n matrix $C$, from either
	 * the left or the right. $H$ is represented in the form\n
	 *     $H = I - \tau v v^T$\n
	 * where $\tau$ is a real scalar and $v$ is a real vector.\n
	 * If $\tau=0$, then $H$ is taken to be the unit matrix.\n
	 * This version uses inline code if $H$ has order $<11$.
	 * \param[in] side
	 *     = 'L': form $H C$\n
	 *     = 'R': form $C H$
	 *
	 * \param[in] m The number of rows of the matrix $C$.
	 * \param[in] n The number of columns of the matrix $C$.
	 * \param[in] v
	 *     an array, dimension\n &emsp; (§m) if §side = 'L'\n
	 *                               or (§n) if §side = 'R'\n
	 *     The vector $v$ in the representation of $H$.
	 *
	 * \param[in]     tau The value $\tau$ in the representation of $H$.
	 * \param[in,out] C
	 *     an array, dimension (§ldc,§n)\n
	 *     On entry, the §m by §n matrix $C$.\n
	 *     On exit, §C is overwritten by the matrix $H C$ if §side = 'L', or $C H$ if §side = 'R'.
	 *
	 * \param[in]  ldc  The leading dimension of the array §C. $\{ldc}\ge\max(1,\{m})$.
	 * \param[out] work
	 *     an array, dimension\n &emsp; (§n) if §side = 'L'\n
	 *                               or (§m) if §side = 'R'\n
	 *     §work is not referenced if $H$ has order $<11$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlarfx(char const* const side, int const m, int const n, real const* const v,
	                   real const tau, real* const C, int const ldc, real* const work) /*const*/
	{
		if (tau==ZERO)
		{
			return;
		}
		int j;
		real sum, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9;
		if (std::toupper(side[0])=='L')
		{
			int cj;
			// Form H * C, where H has order m.
			switch (m)
			{
				default:
					// Code for general m
					dlarf(side, m, n, v, 1, tau, C, ldc, work);
					break;
				case 1:
					// Special code for 1 x 1 Householder
					t0 = ONE - tau*v[0]*v[0];
					for (j=0; j<n; j++)
					{
						C[ldc*j] *= t0;
					}
					break;
				case 2:
					// Special code for 2 x 2 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					for (j=0; j<n; j++)
					{
						cj = ldc * j;
						sum = v0*C[cj] + v1*C[1+cj];
						C[  cj] -= sum * t0;
						C[1+cj] -= sum * t1;
					}
					break;
				case 3:
					// Special code for 3 x 3 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					for (j=0; j<n; j++)
					{
						cj = ldc * j;
						sum = v0*C[cj] + v1*C[1+cj] + v2*C[2+cj];
						C[  cj] -= sum * t0;
						C[1+cj] -= sum * t1;
						C[2+cj] -= sum * t2;
					}
					break;
				case 4:
					// Special code for 4 x 4 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					for (j=0; j<n; j++)
					{
						cj = ldc * j;
						sum = v0*C[cj] + v1*C[1+cj] + v2*C[2+cj] + v3*C[3+cj];
						C[  cj] -= sum * t0;
						C[1+cj] -= sum * t1;
						C[2+cj] -= sum * t2;
						C[3+cj] -= sum * t3;
					}
					break;
				case 5:
					// Special code for 5 x 5 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					for (j=0; j<n; j++)
					{
						cj = ldc * j;
						sum = v0*C[cj] + v1*C[1+cj] + v2*C[2+cj] + v3*C[3+cj] + v4*C[4+cj];
						C[  cj] -= sum * t0;
						C[1+cj] -= sum * t1;
						C[2+cj] -= sum * t2;
						C[3+cj] -= sum * t3;
						C[4+cj] -= sum * t4;
					}
					break;
				case 6:
					// Special code for 6 x 6 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					v5 = v[5];
					t5 = tau * v5;
					for (j=0; j<n; j++)
					{
						cj = ldc * j;
						sum = v0*C[  cj] + v1*C[1+cj] + v2*C[2+cj] + v3*C[3+cj] + v4*C[4+cj]
						    + v5*C[5+cj];
						C[  cj] -= sum * t0;
						C[1+cj] -= sum * t1;
						C[2+cj] -= sum * t2;
						C[3+cj] -= sum * t3;
						C[4+cj] -= sum * t4;
						C[5+cj] -= sum * t5;
					}
					break;
				case 7:
					// Special code for 7 x 7 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					v5 = v[5];
					t5 = tau * v5;
					v6 = v[6];
					t6 = tau * v6;
					for (j=0; j<n; j++)
					{
						cj = ldc * j;
						sum = v0*C[  cj] + v1*C[1+cj] + v2*C[2+cj] + v3*C[3+cj] + v4*C[4+cj]
						    + v5*C[5+cj] + v6*C[6+cj];
						C[  cj] -= sum * t0;
						C[1+cj] -= sum * t1;
						C[2+cj] -= sum * t2;
						C[3+cj] -= sum * t3;
						C[4+cj] -= sum * t4;
						C[5+cj] -= sum * t5;
						C[6+cj] -= sum * t6;
					}
					break;
				case 8:
					// Special code for 8 x 8 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					v5 = v[5];
					t5 = tau * v5;
					v6 = v[6];
					t6 = tau * v6;
					v7 = v[7];
					t7 = tau * v7;
					for (j=0; j<n; j++)
					{
						cj = ldc * j;
						sum = v0*C[  cj] + v1*C[1+cj] + v2*C[2+cj] + v3*C[3+cj] + v4*C[4+cj]
						    + v5*C[5+cj] + v6*C[6+cj] + v7*C[7+cj];
						C[  cj] -= sum * t0;
						C[1+cj] -= sum * t1;
						C[2+cj] -= sum * t2;
						C[3+cj] -= sum * t3;
						C[4+cj] -= sum * t4;
						C[5+cj] -= sum * t5;
						C[6+cj] -= sum * t6;
						C[7+cj] -= sum * t7;
					}
					break;
				case 9:
					// Special code for 9 x 9 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					v5 = v[5];
					t5 = tau * v5;
					v6 = v[6];
					t6 = tau * v6;
					v7 = v[7];
					t7 = tau * v7;
					v8 = v[8];
					t8 = tau * v8;
					for (j=0; j<n; j++)
					{
						cj = ldc * j;
						sum = v0*C[  cj] + v1*C[1+cj] + v2*C[2+cj] + v3*C[3+cj] + v4*C[4+cj]
						    + v5*C[5+cj] + v6*C[6+cj] + v7*C[7+cj] + v8*C[8+cj];
						C[  cj] -= sum * t0;
						C[1+cj] -= sum * t1;
						C[2+cj] -= sum * t2;
						C[3+cj] -= sum * t3;
						C[4+cj] -= sum * t4;
						C[5+cj] -= sum * t5;
						C[6+cj] -= sum * t6;
						C[7+cj] -= sum * t7;
						C[8+cj] -= sum * t8;
					}
					break;
				case 10:
					// Special code for 10 x 10 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					v5 = v[5];
					t5 = tau * v5;
					v6 = v[6];
					t6 = tau * v6;
					v7 = v[7];
					t7 = tau * v7;
					v8 = v[8];
					t8 = tau * v8;
					v9 = v[9];
					t9 = tau * v9;
					for (j=0; j<n; j++)
					{
						cj = ldc * j;
						sum = v0*C[  cj] + v1*C[1+cj] + v2*C[2+cj] + v3*C[3+cj] + v4*C[4+cj]
						    + v5*C[5+cj] + v6*C[6+cj] + v7*C[7+cj] + v8*C[8+cj] + v9*C[9+cj];
						C[  cj] -= sum * t0;
						C[1+cj] -= sum * t1;
						C[2+cj] -= sum * t2;
						C[3+cj] -= sum * t3;
						C[4+cj] -= sum * t4;
						C[5+cj] -= sum * t5;
						C[6+cj] -= sum * t6;
						C[7+cj] -= sum * t7;
						C[8+cj] -= sum * t8;
						C[9+cj] -= sum * t9;
					}
					break;
			}
		}
		else
		{
			int c1=0, c2=0, c3, c4, c5, c6, c7, c8, c9;
			switch (n)
			{
				case 10:
					c9 = ldc * 9;
					[[fallthrough]];
				case 9:
					c8 = ldc * 8;
					[[fallthrough]];
				case 8:
					c7 = ldc * 7;
					[[fallthrough]];
				case 7:
					c6 = ldc * 6;
					[[fallthrough]];
				case 6:
					c5 = ldc * 5;
					[[fallthrough]];
				case 5:
					c4 = ldc * 4;
					[[fallthrough]];
				case 4:
					c3 = ldc * 3;
					[[fallthrough]];
				case 3:
					c2 = ldc * 2;
					[[fallthrough]];
				case 2:
					c1 = ldc;
			}
			// Form  C * H, where H has order n.
			switch (n)
			{
				default:
					// Code for general n
					dlarf(side, m, n, v, 1, tau, C, ldc, work);
					break;
				case 1:
					// Special code for 1 x 1 Householder
					t0 = ONE - tau*v[0]*v[0];
					for (j=0; j<n; j++)
					{
						C[j] *= t0;
					}
					break;
				case 2:
					// Special code for 2 x 2 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					for (j=0; j<n; j++)
					{
						sum = v0*C[j] + v1*C[j+c1];
						C[j   ] -= sum * t0;
						C[j+c1] -= sum * t1;
					}
					break;
				case 3:
					// Special code for 3 x 3 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					for (j=0; j<n; j++)
					{
						sum = v0*C[j] + v1*C[j+c1] + v2*C[j+c2];
						C[j   ] -= sum * t0;
						C[j+c1] -= sum * t1;
						C[j+c2] -= sum * t2;
					}
					break;
				case 4:
					// Special code for 4 x 4 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					for (j=0; j<n; j++)
					{
						sum = v0*C[j] + v1*C[j+c1] + v2*C[j+c2] + v3*C[j+c3];
						C[j   ] -= sum * t0;
						C[j+c1] -= sum * t1;
						C[j+c2] -= sum * t2;
						C[j+c3] -= sum * t3;
					}
					break;
				case 5:
					// Special code for 5 x 5 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					for (j=0; j<n; j++)
					{
						sum = v0*C[j] + v1*C[j+c1] + v2*C[j+c2] + v3*C[j+c3] + v4*C[j+c4];
						C[j   ] -= sum * t0;
						C[j+c1] -= sum * t1;
						C[j+c2] -= sum * t2;
						C[j+c3] -= sum * t3;
						C[j+c4] -= sum * t4;
					}
					break;
				case 6:
					// Special code for 6 x 6 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					v5 = v[5];
					t5 = tau * v5;
					for (j=0; j<n; j++)
					{
						sum = v0*C[j   ] + v1*C[j+c1] + v2*C[j+c2] + v3*C[j+c3] + v4*C[j+c4]
						    + v5*C[j+c5];
						C[j   ] -= sum * t0;
						C[j+c1] -= sum * t1;
						C[j+c2] -= sum * t2;
						C[j+c3] -= sum * t3;
						C[j+c4] -= sum * t4;
						C[j+c5] -= sum * t5;
					}
					break;
				case 7:
					// Special code for 7 x 7 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					v5 = v[5];
					t5 = tau * v5;
					v6 = v[6];
					t6 = tau * v6;
					for (j=0; j<n; j++)
					{
						sum = v0*C[j   ] + v1*C[j+c1] + v2*C[j+c2] + v3*C[j+c3] + v4*C[j+c4]
						    + v5*C[j+c5] + v6*C[j+c6];
						C[j   ] -= sum * t0;
						C[j+c1] -= sum * t1;
						C[j+c2] -= sum * t2;
						C[j+c3] -= sum * t3;
						C[j+c4] -= sum * t4;
						C[j+c5] -= sum * t5;
						C[j+c6] -= sum * t6;
					}
					break;
				case 8:
					// Special code for 8 x 8 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					v5 = v[5];
					t5 = tau * v5;
					v6 = v[6];
					t6 = tau * v6;
					v7 = v[7];
					t7 = tau * v7;
					for (j=0; j<n; j++)
					{
						sum = v0*C[j   ] + v1*C[j+c1] + v2*C[j+c2] + v3*C[j+c3] + v4*C[j+c4]
						    + v5*C[j+c5] + v6*C[j+c6] + v7*C[j+c7];
						C[j   ] -= sum * t0;
						C[j+c1] -= sum * t1;
						C[j+c2] -= sum * t2;
						C[j+c3] -= sum * t3;
						C[j+c4] -= sum * t4;
						C[j+c5] -= sum * t5;
						C[j+c6] -= sum * t6;
						C[j+c7] -= sum * t7;
					}
					break;
				case 9:
					// Special code for 9 x 9 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					v5 = v[5];
					t5 = tau * v5;
					v6 = v[6];
					t6 = tau * v6;
					v7 = v[7];
					t7 = tau * v7;
					v8 = v[8];
					t8 = tau * v8;
					for (j=0; j<n; j++)
					{
						sum = v0*C[j   ] + v1*C[j+c1] + v2*C[j+c2] + v3*C[j+c3] + v4*C[j+c4]
						    + v5*C[j+c5] + v6*C[j+c6] + v7*C[j+c7] + v8*C[j+c8];
						C[j   ] -= sum * t0;
						C[j+c1] -= sum * t1;
						C[j+c2] -= sum * t2;
						C[j+c3] -= sum * t3;
						C[j+c4] -= sum * t4;
						C[j+c5] -= sum * t5;
						C[j+c6] -= sum * t6;
						C[j+c7] -= sum * t7;
						C[j+c8] -= sum * t8;
					}
					break;
				case 10:
					// Special code for 10 x 10 Householder
					v0 = v[0];
					t0 = tau * v0;
					v1 = v[1];
					t1 = tau * v1;
					v2 = v[2];
					t2 = tau * v2;
					v3 = v[3];
					t3 = tau * v3;
					v4 = v[4];
					t4 = tau * v4;
					v5 = v[5];
					t5 = tau * v5;
					v6 = v[6];
					t6 = tau * v6;
					v7 = v[7];
					t7 = tau * v7;
					v8 = v[8];
					t8 = tau * v8;
					v9 = v[9];
					t9 = tau * v9;
					for (j=0; j<n; j++)
					{
						sum = v0*C[j   ] + v1*C[j+c1] + v2*C[j+c2] + v3*C[j+c3] + v4 *C[j+c4]
						    + v5*C[j+c5] + v6*C[j+c6] + v7*C[j+c7] + v8*C[j+c8] + v9*C[j+c9];
						C[j   ] -= sum * t0;
						C[j+c1] -= sum * t1;
						C[j+c2] -= sum * t2;
						C[j+c3] -= sum * t3;
						C[j+c4] -= sum * t4;
						C[j+c5] -= sum * t5;
						C[j+c6] -= sum * t6;
						C[j+c7] -= sum * t7;
						C[j+c8] -= sum * t8;
						C[j+c9] -= sum * t9;
					}
					break;
			}
		}
	}

	/*! §dlarnv returns a vector of random numbers from a uniform or normal distribution.
	 *
	 * §dlarnv returns a vector of §n random real numbers from a uniform or normal distribution.
	 * \param[in] idist
	 *     Specifies the distribution of the random numbers:\n
	 *         = 1: uniform (0,1)\n
	 *         = 2: uniform (-1,1)\n
	 *         = 3: normal (0,1)
	 *
	 * \param[in,out] iseed
	 *     an integer array, dimension (4)\n
	 *     On entry, the seed of the random number generator; the array elements must be between
	 *               0 and 4095, and §iseed[3] must be odd.\n
	 *     On exit, the seed is updated.
	 *
	 * \param[in]  n The number of random numbers to be generated.
	 * \param[out] x
	 *     an array, dimension (§n)\n
	 *     The generated random numbers.
	 *
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     This routine calls the auxiliary routine §dlaruv to generate random real numbers from a
	 *     uniform (0,1) distribution, in batches of up to 128 using vectorisable code. The Box-
	 *     Muller method is used to transform numbers from a uniform to a normal distribution.   */
	static void dlarnv(int const idist, int* const iseed, int const n, real* const x) /*const*/
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

	/*! §dlartg generates a plane rotation with real cosine and real sine.
	 *
	 * §dlartg generates a plane rotation so that\n
	 *     $\b{bm}  \{cs} & \{sn} \\
	 *             -\{sn} & \{cs} \e{bm}\b{bm} \{f} \\
	 *                                         \{g} \e{bm} = \b{bm} \{r} \\
	 *                                                                0 \e{bm}$
	 *     where $\{cs}^2 + \{sn}^2 = 1$.\n
	 * This is a slower, more accurate version of the BLAS1 routine §drotg,
	 * with the following other differences:
	 * \li §f and §g are unchanged on return.
	 * \li If §g = 0, then §cs = 1 and §sn = 0.
	 * \li If §f = 0 and $\{g}\ne 0$, then §cs = 0 and §sn = 1 without doing any floating point
	 *     operations (saves work in §dbdsqr when there are zeros on the diagonal).
	 *
	 * If §f exceeds §g in magnitude, §cs will be positive.
	 * \param[in]  f  The first component of vector to be rotated.
	 * \param[in]  g  The second component of vector to be rotated.
	 * \param[out] cs The cosine of the rotation.
	 * \param[out] sn The sine of the rotation.
	 * \param[out] r  The nonzero component of the rotated vector.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 * This version has a few statements commented out for thread safety (machine parameters are
	 * computed on each entry). 10 feb 03, SJH.                                                  */
	static void dlartg(real const f, real const g, real& cs, real& sn, real& r) /*const*/
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

	/*! §dlaruv returns a vector of §n random real numbers from a uniform distribution.
	 *
	 * §dlaruv returns a vector of n random real numbers from a uniform (0,1) distribution
	 * ($\{n}\le 128$).\n
	 * This is an auxiliary routine called by §dlarnv and §zlarnv.
	 * \param[in,out] iseed
	 *     an integer array, dimension (4)\n
	 *     On entry, the seed of the random number generator; the array elements must be between 0
	 *     and 4095, and §iseed[3] must be odd.\n
	 *     On exit, the seed is updated.
	 *
	 * \param[in]  n The number of random numbers to be generated. $\{n}\le 128$.
	 * \param[out] x an array, dimension (§n)\n  The generated random numbers.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *    This routine uses a multiplicative congruential method with modulus $2^48$ and multiplier
	 *    33952834046453 (see G.S.Fishman, 'Multiplicative congruential random number generators
	 *    with modulus 2^b: an exhaustive analysis for b = 32 and a partial analysis for b = 48',
	 *    Math. Comp. 189, pp 331-344, 1990).
	 *    48-bit integers are stored in 4 integer array elements with 12 bits per element. Hence
	 *    the routine is portable across machines with integers of 32 bits or more.              */
	static void dlaruv(int* const iseed, int const n, real* const x) /*const*/
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
					// integer above happen to be all 1 (which will occur about once every 2^n
					// calls), then X[i] will be rounded to exactly 1.0.
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

	/*! §dlas2 computes singular values of a 2-by-2 triangular matrix.
	 *
	 * §dlas2 computes the singular values of the 2-by-2 matrix\n
	 *     $\b{bm} \{f} & \{g} \\
	 *               0  & \{h} \e{bm}$.\n
	 * On return, §ssmin is the smaller singular value and §ssmax is the larger singular value.
	 * \param[in]  f     The [0,0] element of the 2-by-2 matrix.
	 * \param[in]  g     The [0,1] element of the 2-by-2 matrix.
	 * \param[in]  h     The [1,1] element of the 2-by-2 matrix.
	 * \param[out] ssmin The smaller singular value.
	 * \param[out] ssmax The larger singular value.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Barring over/underflow, all output quantities are correct to within a few units in the
	 *     last place (ulps), even in the absence of a guard digit in addition/subtraction.
	 *     In §IEEE arithmetic, the code works correctly if one matrix element is infinite.
	 *     Overflow will not occur unless the largest singular value itself overflows, or is within
	 *     a few ulps of overflow. (On machines with partial overflow, like the Cray, overflow may
	 *     occur if the largest singular value is within a factor of 2 of overflow.)
	 *     Underflow is harmless if underflow is gradual. Otherwise, results may correspond to a
	 *     matrix modified by perturbations of size near the underflow threshold.                */
	static void dlas2(real const f, real const g, real const h, real& ssmin, real& ssmax) /*const*/
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

	/*! §dlascl multiplies a general rectangular matrix by a real scalar defined as §cto/§cfrom.
	 *
	 * §dlascl multiplies the §m by §n real matrix $A$ by the real scalar §cto/§cfrom. This is done
	 * without over/underflow as long as the final result $\{cto}A[i,j]/\{cfrom}$ does not
	 * over/underflow. §type specifies that $A$ may be full, upper triangular, lower triangular,
	 * upper Hessenberg, or banded.
	 * \param[in] type
	 *     indicates the storage type of the input matrix.\n
	 *         ='G': $A$ is a full matrix.\n
	 *         ='L': $A$ is a lower triangular matrix.\n
	 *         ='U': $A$ is an upper triangular matrix.\n
	 *         ='H': $A$ is an upper Hessenberg matrix.\n
	 *         ='B': $A$ is a symmetric band matrix with lower bandwidth §kl and upper bandwidth
	 *               §ku and with the only the lower half stored.\n
	 *         ='Q': $A$ is a symmetric band matrix with lower bandwidth §kl and upper bandwidth
	 *               §ku and with the only the upper half stored.\n
	 *         ='Z': $A$ is a band matrix with lower bandwidth §kl and upper bandwidth §ku.
	 *               See §dgbtrf for storage details.
	 *
	 * \param[in] kl    The lower bandwidth of $A$. Referenced only if §type = 'B', 'Q' or 'Z'.
	 * \param[in] ku    The upper bandwidth of $A$. Referenced only if §type = 'B', 'Q' or 'Z'.
	 * \param[in] cfrom
	 * \param[in] cto
	 *     The matrix $A$ is multiplied by §cto/§cfrom. $A[i,j]$ is computed without
	 *                  over/underflow if the final result $\{cto}A[i,j]/\{cfrom}$ can be
	 *                  represented without over/underflow. §cfrom must be nonzero.
	 *
	 * \param[in]     m The number of rows of the matrix $A$. $\{m}\ge 0$.
	 * \param[in]     n The number of columns of the matrix $A$. $\{n}\ge 0$.
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     The matrix to be multiplied by §cto/§cfrom. See §type for the storage type.
	 *
	 * \param[in] lda
	 *     The leading dimension of the array §A.\n
	 *         If §type = 'G', 'L', 'U', 'H': $\{lda} \ge \max(1,\{m})$;\n
	 *         if §type = 'B'               : $\{lda} \ge \{kl}+1$;\n
	 *         if §type = 'Q'               : $\{lda} \ge \{ku}+1$;\n
	 *         if §type = 'Z'               : $\{lda} \ge 2*\{kl}+\{ku}+1$.
	 *
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.
	 *
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016                                                                           */
	static void dlascl(char const* const type, int const kl, int const ku, real const cfrom,
	                   real const cto, int const m, int const n, real* const A, int const lda,
	                   int& info) /*const*/
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

	/*! §dlasd0 computes the singular values of a real upper bidiagonal §n by §m matrix $B$ with
	 *  diagonal §d and off-diagonal §e. Used by §sbdsdc.
	 *
	 * Using a divide and conquer approach, §dlasd0 computes the singular value decomposition (SVD)
	 * of a real upper bidiagonal §n by §m matrix $B$ with diagonal §d and offdiagonal §e, where
	 * §m = §n+§sqre. The algorithm computes orthogonal matrices $U$ and $V_T$ such that
	 * $B = U S V_T$. The singular values $S$ are overwritten on §d.\n
	 * A related subroutine, §dlasda, computes only the singular values, and optionally, the
	 * singular vectors in compact form.
	 * \param[in] n
	 *     On entry, the row dimension of the upper bidiagonal matrix.\n
	 *     This is also the dimension of the main diagonal array §d.
	 *
	 * \param[in] sqre
	 *     Specifies the column dimension of the bidiagonal matrix.\n
	 *     = 0: The bidiagonal matrix has column dimension §m = §n;\n
	 *     = 1: The bidiagonal matrix has column dimension §m = §n+1;
	 *
	 * \param[in,out] d
	 *     an array, dimension (§n)\n
	 *     On entry §d contains the main diagonal of the bidiagonal matrix.\n
	 *     On exit §d, if §info = 0, contains its singular values.
	 *
	 * \param[in,out] e
	 *     an array, dimension (§m-1)\n
	 *     Contains the subdiagonal entries of the bidiagonal matrix.
	 *     On exit, §e has been destroyed.
	 *
	 * \param[out] U
	 *     an array, dimension (§ldu,§n)\n
	 *     On exit, §U contains the left singular vectors.
	 *
	 * \param[in]  ldu On entry, leading dimension of §U.
	 * \param[out] Vt
	 *     an array, dimension (§ldvt,§m)\n
	 *     On exit, $\{Vt}^T$ contains the right singular vectors.
	 *
	 * \param[in] ldvt On entry, leading dimension of §Vt.
	 * \param[in] smlsiz
	 *     On entry, maximum size of the subproblems at the bottom of the computation tree.
	 *
	 * \param[out] iwork an integer array, dimension ($8\{n}$)
	 * \param[out] work  an array, dimension ($3\{m}^2+2\{m}$)
	 * \param[out] info
	 *     = 0: successful exit.\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.\n
	 *     > 0: if §info = 1, a singular value did not converge
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017
	 * \remark
	 *     Contributors:\n
	 *     Ming Gu and Huan Ren, Computer Science Division,
	 *     University of California at Berkeley, USA                                             */
	static void dlasd0(int const n, int const sqre, real* const d, real* const e, real* const U,
	                   int const ldu, real* const Vt, int const ldvt, int const smlsiz,
	                   int* const iwork, real* const work, int& info) /*const*/
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

	/*! §dlasd1 computes the SVD of an upper bidiagonal matrix $B$ of the specified size.
	 *  Used by §sbdsdc.
	 *
	 * §dlasd1 computes the SVD of an upper bidiagonal $n$ by $m$ matrix $B$, where
	 * $n=\{nl}+\{nr}+1$ and $m=n+\{sqre}$. §dlasd1 is called from §dlasd0.\n
	 * A related subroutine §dlasd7 handles the case in which the singular values (and the singular
	 * vectors in factored form) are desired.\n
	 * §dlasd1 computes the SVD as follows:
	 *     \f{align*}{
	 *         B &= U^\{in} \b{bm} d_1^\{in} & 0 &     0     & 0 \\
	 *                               Z_1^T   & a &   Z_2^T   & b \\
	 *                                 0     & 0 & d_2^\{in} & 0 \e{bm} V_T^\{in}\\
	 *           &= U^\{out} \b{bm} d^\{out} & 0 \e{bm} V_T^\{out}
	 *     \f}
	 * where $Z^T = (Z_1^T a Z_2^T b) = u^T V_T^T$, and $u$ is a vector of dimension $m$ with
	 * §alpha and §beta in the §nl+1 and §nl+2 -th entries and zeros elsewhere; and the entry $b$
	 * is empty if §sqre = 0.\n
	 * The left singular vectors of the original matrix are stored in §U, and the transpose of the
	 * right singular vectors are stored in §Vt, and the singular values are in §d. The algorithm
	 * consists of three stages:
	 * \li The first stage consists of deflating the size of the problem when there are multiple
	 *     singular values or when there are zeros in the $Z$ vector. For each such occurrence the
	 *     dimension of the secular equation problem is reduced by one. This stage is performed by
	 *     the routine §dlasd2.
	 * \li The second stage consists of calculating the updated singular values. This is done by
	 *     finding the square roots of the roots of the secular equation via the routine §dlasd4
	 *     (as called by §dlasd3). This routine also calculates the singular vectors of the current
	 *     problem.
	 * \li The final stage consists of computing the updated singular vectors directly using the
	 *     updated singular values. The singular vectors for the current problem are multiplied
	 *     with the singular vectors from the overall problem.
	 *
	 * \param[in] nl   The row dimension of the upper block. $\{nl}\ge 1$.
	 * \param[in] nr   The row dimension of the lower block. $\{nr}\ge 1$.
	 * \param[in] sqre
	 *     = 0: the lower block is an §nr by §nr square matrix.\n
	 *     = 1: the lower block is an §nr by (§nr+1) rectangular matrix.\n
	 *     The bidiagonal matrix has row dimension $n=\{nl}+\{nr}+1$, and column dimension
	 *     $m=n+\{sqre}$.
	 *
	 * \param[in,out] d
	 *     an array, dimension ($n=\{nl}+\{nr}+1$).\n
	 *     On entry $\{d}[0:\{nl}-1,0:\{nl}-1]$ contains the singular values of the upper block;
	 *     and $\{d}[\{nl}+1:n-1]$ contains the singular values of the lower block.\n
	 *     On exit $\{d}[0:n-1]$ contains the singular values of the modified matrix.
	 *
	 * \param[in,out] alpha Contains the diagonal element associated with the added row.
	 * \param[in,out] beta  Contains the off-diagonal element associated with the added row.
	 * \param[in,out] U
	 *     an array, dimension (§ldu,$n$)\n
	 *     On entry $\{U}[0:\{nl}-1,0:\{nl}-1]$ contains the left singular vectors of the upper
	 *     block; $\{U}[\{nl}+1:n-1,\{nl}+1:n-1]$ contains the left singular vectors of the lower
	 *     block.\n
	 *     On exit §U contains the left singular vectors of the bidiagonal matrix.
	 *
	 * \param[in]     ldu The leading dimension of the array §U. $\{ldu}\ge\max(1,n)$.
	 * \param[in,out] Vt
	 *     an array, dimension (§ldvt,$m$) where $m=n+\{sqre}$.\n
	 *     On entry $\{Vt}[0:\{nl},0:\{nl}]^T$ contains the right singular vectors of the upper
	 *     block; $\{Vt}[\{nl}+1:m-1,\{nl}+1:m-1]^T$ contains the right singular vectors of the
	 *     lower block.\n
	 *     On exit $\{Vt}^T$ contains the right singular vectors of the bidiagonal matrix.
	 *
	 * \param[in]     ldvt The leading dimension of the array §Vt. $\{ldvt}\ge\max(1,m)$.
	 * \param[in,out] idxq
	 *      an integer array, dimension ($n$)\n
	 *      This contains the permutation which will reintegrate the subproblem just solved back
	 *      into sorted order, i.e. $\{d}[\{idxq}[0:n-1]]$ will be in ascending order.\n
	 *      NOTE: zero-based indexing!
	 *
	 * \param[out] iwork an integer array, dimension ($4n$)
	 * \param[out] work  an array, dimension ($3m^2 + 2m$)
	 * \param[out] info
	 *     = 0: successful exit.\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.\n
	 *     > 0: if §info = 1, a singular value did not converge
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016
	 * \remark
	 *     Contributors:\n
	 *     Ming Gu and Huan Ren, Computer Science Division,
	 *     University of California at Berkeley, USA                                             */
	static void dlasd1(int const nl, int const nr, int const sqre, real* const d, real& alpha,
	                   real& beta, real* const U, int const ldu, real* const Vt, int const ldvt,
	                   int* const idxq, int* const iwork, real* const work, int& info) /*const*/
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

	/*! §dlasd2 merges the two sets of singular values together into a single sorted set.
	 *  Used by §sbdsdc.
	 *
	 * §dlasd2 merges the two sets of singular values together into a single sorted set. Then it
	 * tries to deflate the size of the problem. There are two ways in which deflation can occur:
	 * when two or more singular values are close together or if there is a tiny entry in the §z
	 * vector. For each such occurrence the order of the related secular equation problem is
	 * reduced by one.\n
	 * §dlasd2 is called from §dlasd1.
	 * \param[in] nl   The row dimension of the upper block. $\{nl}\ge 1$.
	 * \param[in] nr   The row dimension of the lower block. $\{nr}\ge 1$.
	 * \param[in] sqre
	 *     = 0: the lower block is an §nr by §nr square matrix.\n
	 *     = 1: the lower block is an §nr by (§nr+1) rectangular matrix.\n
	 *     The bidiagonal matrix has $n = \{nl}+\{nr}+1$ rows and $m = n+\{sqre} \ge n$ columns.
	 *
	 * \param[out] k
	 *     Contains the dimension of the non-deflated matrix, This is the order of the related
	 *     secular equation. $1 \le \{k} \le n$.
	 *
	 * \param[in,out] d
	 *     an array, dimension ($n$)\n
	 *     On entry §d contains the singular values of the two submatrices to be combined.\n
	 *     On exit §d contains the trailing ($n-\{k}$) updated singular values (those which were
	 *     deflated) sorted into increasing order.
	 *
	 * \param[out] z
	 *     an array, dimension ($n$)\n
	 *     On exit §z contains the updating row vector in the secular equation.
	 *
	 * \param[in]     alpha Contains the diagonal element associated with the added row.
	 * \param[in]     beta  Contains the off-diagonal element associated with the added row.
	 * \param[in,out] U
	 *     an array, dimension (§ldu,$n$)\n
	 *     On entry §U contains the left singular vectors of two submatrices in the two square
	 *     blocks with corners at [0,0], [§nl-1,§nl-1], and [§nl+1,§nl+1], [$n$-1,$n$-1].\n
	 *     On exit §U contains the trailing ($n-\{k}$) updated left singular vectors (those which
	 *     were deflated) in its last $n-\{k}$ columns.
	 *
	 * \param[in]     ldu The leading dimension of the array §U. $\{ldu}\ge n$.
	 * \param[in,out] Vt
	 *     an array, dimension (§ldvt,$m$)\n
	 *     On entry $\{Vt}^T$ contains the right singular vectors of two submatrices in the two
	 *     square blocks with corners at [0,0], [§nl,§nl], and [§nl+1,§nl+1], [$m$-1,$m$-1].\n
	 *     On exit $\{Vt}^T$ contains the trailing ($n-\{k}$) updated right singular vectors (those
	 *     which were deflated) in its last $n-\{k}$ columns.\n
	 *     In case §sqre = 1, the last row of §Vt spans the right null space.
	 *
	 * \param[in]  ldvt The leading dimension of the array §Vt. $\{ldvt}\ge m$.
	 * \param[out] dsigma
	 *     an array, dimension ($n$)\n
	 *     Contains a copy of the diagonal elements (§k-1 singular values and one zero) in the
	 *     secular equation.
	 *
	 * \param[out] U2
	 *     an array, dimension (§ldu2,$n$)\n
	 *     Contains a copy of the first §k-1 left singular vectors which will be used by §dlasd3 in
	 *     a matrix multiply (§dgemm) to solve for the new left singular vectors. §U2 is arranged
	 *     into four blocks. The first block contains a column with 1 at §nl and zero everywhere
	 *     else; the second block contains non-zero entries only at and above §nl-1; the third
	 *     contains non-zero entries only below §nl; and the fourth is dense.
	 *
	 * \param[in]  ldu2 The leading dimension of the array §U2. $\{ldu2}\ge n$.
	 * \param[out] Vt2
	 *     an array, dimension (§ldvt2,$n$)\n
	 *     $\{Vt2}^T$ contains a copy of the first §k right singular vectors which will be used by
	 *     §dlasd3 in a matrix multiply (§dgemm) to solve for the new right singular vectors. §Vt2
	 *     is arranged into three blocks. The first block contains a row that corresponds to the
	 *     special 0 diagonal element in §dsigma; the second block contains non-zeros only at and
	 *     before §nl; the third block contains non-zeros only at and after §nl+1.
	 *
	 * \param[in]  ldvt2 The leading dimension of the array §Vt2. $\{ldvt2}\ge m$.
	 * \param[out] idxp
	 *     an integer array, dimension ($n$)\n
	 *     This will contain the permutation used to place deflated values of §d at the end of the
	 *     array. On output $\{idxp}[1:\{k}-1]$ points to the nondeflated §d values and
	 *     $\{idxp}[\{k}:n-1]$ points to the deflated singular values.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] idx
	 *     an integer array, dimension ($n$)\n
	 *     This will contain the permutation used to sort the contents of §d into ascending
	 *     order.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] idxc
	 *     an integer array, dimension ($n$)\n
	 *     This will contain the permutation used to arrange the columns of the deflated §U matrix
	 *     into three groups: the first group contains non-zero entries only at and above §nl-1,
	 *     the second contains non-zero entries only below §nl+1, and the third is dense.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in,out] idxq
	 *     an integer array, dimension ($n$)\n
	 *     This contains the permutation which separately sorts the two sub-problems in §d into
	 *     ascending order. Note that entries in the first half of this permutation must first be
	 *     moved one position backward; and entries in the second half must first have §nl added
	 *     to their values.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] coltyp
	 *     an integer array, dimension ($n$)\n
	 *     As workspace, this will contain a label which will indicate which of the following types
	 *     a column in the §U2 matrix or a row in the §Vt2 matrix is:\n
	 *         0: non-zero in the upper half only\n
	 *         1: non-zero in the lower half only\n
	 *         2: dense\n
	 *         3: deflated\n
	 *     On exit, it is an array of dimension 4, with §coltyp[$i$] being the dimension of the
	 *     $i$-th type columns.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] info
	 *     = 0: successful exit.\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017
	 * \remark
	 *     Contributors:\n
	 *     Ming Gu and Huan Ren, Computer Science Division,
	 *     University of California at Berkeley, USA                                             */
	static void dlasd2(int const nl, int const nr, int const sqre, int& k, real* const d,
	                   real* const z, real const alpha, real const beta, real* const U,
	                   int const ldu, real* const Vt, int const ldvt, real* const dsigma,
	                   real* const U2, int const ldu2, real* const Vt2, int const ldvt2,
	                   int* const idxp, int* const idx, int* const idxc, int* const idxq,
	                   int* const coltyp, int& info) /*const*/
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
		int j, jprev=-1;
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
		for (j=0; j<4; j++)
		{
			ctot[j] = 0;
		}
		for (j=1; j<n; j++)
		{
			ctot[coltyp[j]]++;
		}
		// psm[*] = Position in SubMatrix (of types 0 through 3) (zero-based!)
		psm[0] = 1;
		psm[1] = 1 + ctot[0];
		psm[2] = psm[1] + ctot[1];
		psm[3] = psm[2] + ctot[2];
		// Fill out the idxc array so that the permutation which it induces will place all type-0
		// columns first, all type-1 columns next, then all type-2's, and finally all type-3's,
		// starting from the second column. This applies similarly to the rows of Vt.
		int jp, ct;
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
				Vt2[ldvt2*i] =  c*Vt[nl+tind];
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
			Blas<real>::dcopy(n-k, &dsigma[k], 1, &d[k], 1);
			dlacpy("A", n, n-k, &U2[ldu2*k], ldu2, &U[ldu*k], ldu);
			dlacpy("A", n-k, m, &Vt2[k], ldvt2, &Vt[k], ldvt);
		}
		// Copy ctot into coltyp for referencing in dlasd3.
		for (j=0; j<4; j++)
		{
			coltyp[j] = ctot[j];
		}
	}

	/*! §dlasd3 finds all square roots of the roots of the secular equation, as defined by the
	 *  values in §d and §z, and then updates the singular vectors by matrix multiplication.
	 *  Used by §sbdsdc.
	 *
	 * §dlasd3 finds all the square roots of the roots of the secular equation, as defined by the
	 * values in §d and §z. It makes the appropriate calls to §dlasd4 and then updates the singular
	 * vectors by matrix multiplication.\n
	 * This code makes very mild assumptions about floating point arithmetic. It will work on
	 * machines with a guard digit in add/subtract, or on those binary machines without guard
	 * digits which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2. It could
	 * conceivably fail on hexadecimal or decimal machines without guard digits, but we know of
	 * none.\n
	 * §dlasd3 is called from §dlasd1.
	 * \param[in] nl   The row dimension of the upper block. $\{nl}\ge 1$.
	 * \param[in] nr   The row dimension of the lower block. $\{nr}\ge 1$.
	 * \param[in] sqre
	 *     = 0: the lower block is an §nr by §nr square matrix.\n
	 *     = 1: the lower block is an §nr by (§nr+1) rectangular matrix.\n
	 *     The bidiagonal matrix has $n = \{nl}+\{nr}+1$ rows and $m = n+\{sqre} \ge n$ columns.
	 *
	 * \param[in]  k The size of the secular equation, $1 \le \{k} \le n$.
	 * \param[out] d
	 *     an array, dimension (§k)\n
	 *     On exit the square roots of the roots of the secular equation, in ascending order.
	 *
	 * \param[out]    Q      an array, dimension (§ldq,§k)
	 * \param[in]     ldq    The leading dimension of the array §Q. $\{ldq}\ge\{k}$.
	 * \param[in,out] dsigma
	 *     an array, dimension (§k)\n
	 *     The first §k elements of this array contain the old roots of the deflated updating
	 *     problem. These are the poles of the secular equation.
	 *
	 * \param[out] U
	 *     an array, dimension (§ldu,$n$)\n
	 *     The last $n-\{k}$ columns of this matrix contain the deflated left singular vectors.
	 *
	 * \param[in] ldu The leading dimension of the array §U. $\{ldu}\ge n$.
	 * \param[in] U2
	 *     an array, dimension (§ldu2,$n$)\n
	 *     The first §k columns of this matrix contain the non-deflated left singular vectors for
	 *     the split problem.
	 *
	 * \param[in]  ldu2 The leading dimension of the array §U2. $\{ldu2}\ge n$.
	 * \param[out] Vt
	 *     an array, dimension (§ldvt,$m$)\n
	 *     The last $m-\{k}$ columns of $\{Vt}^T$ contain the deflated right singular vectors.
	 *
	 * \param[in]     ldvt The leading dimension of the array §Vt. $\{ldvt}\ge n$.
	 * \param[in,out] Vt2
	 *     an array, dimension (§ldvt2,$n$)\n
	 *     The first §k columns of $\{Vt2}^T$ contain the non-deflated right singular vectors for
	 *     the split problem.
	 *
	 * \param[in] ldvt2 The leading dimension of the array §Vt2. $\{ldvt2}\ge n$.
	 * \param[in] idxc
	 *     an integer array, dimension ($n$)\n
	 *     The permutation used to arrange the columns of §U (and rows of §Vt) into three groups:
	 *     the first group contains non-negative entries only at and above (or before) §nl; the
	 *     second contains non-negative entries only at and below (or after) §nl+1; and the third
	 *     is dense. The first column of §U and the row of §Vt are treated separately, however.\n
	 *     The rows of the singular vectors found by §dlasd4 must be likewise permuted before the
	 *     matrix multiplies can take place.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in] ctot
	 *     an integer array, dimension (4)\n
	 *     A count of the total number of the various types of columns in §U (or rows in §Vt), as
	 *     described in §idxc. The fourth column type is any column which has been deflated.
	 *
	 * \param[in,out] z
	 *     an array, dimension (§k)\n
	 *     The first §k elements of this array contain the components of the deflation-adjusted
	 *     updating row vector.
	 *
	 * \param[out] info
	 *     = 0: successful exit.\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.\n
	 *     > 0: if §info = 1, a singular value did not converge
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017
	 * \remark
	 *     Contributors:\n
	 *     Ming Gu and Huan Ren, Computer Science Division,
	 *     University of California at Berkeley, USA                                             */
	static void dlasd3(int const nl, int const nr, int const sqre, int const k, real* const d,
	                   real* const Q, int const ldq, real* const dsigma, real* const U,
	                   int const ldu, real const* const U2, int const ldu2, real* const Vt,
	                   int const ldvt, real* const Vt2, int const ldvt2, int const* const idxc,
	                   int const* const ctot, real* const z, int& info) /*const*/
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
					U[i] = -U2[i];
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
			dlasd4(k, j, dsigma, z, &U[ldu*j], rho, d[j], &Vt[ldvt*j], info);
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
				z[i] *= (U[i+ldu*j]*Vt[i+ldvt*j] / (dsigma[i]-dsigma[j+1])
				                                 / (dsigma[i]+dsigma[j+1]));
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
				Blas<real>::dgemm("N", "N", nl, k, ctot[2], ONE, &U2[ldu2*ktempm1], ldu2,
				                  &Q[ktempm1], ldq, ZERO, U, ldu);
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
			temp = Blas<real>::dnrm2(k, &Vt[ldvti], 1);
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
			Blas<real>::dgemm("N", "N", k, nlp1, ctot[2], ONE, &Q[ldq*ktempm1], ldq, &Vt2[ktempm1],
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
		Blas<real>::dgemm("N", "N", k, nrp1, ctemp, ONE, &Q[qind], ldq, &Vt2[ktempm1+ldvt2*nlp1],
		                  ldvt2, ZERO, &Vt[ldvt*nlp1], ldvt);
	}

	/*! §dlasd4 computes the square root of the §i -th updated eigenvalue of a positive symmetric
	 *  rank-one modification to a positive diagonal matrix. Used by §dbdsdc.
	 *
	 * §dlasd4: This subroutine computes the square root of the §i -th updated eigenvalue of a
	 * positive symmetric rank-one modification to a positive diagonal matrix whose entries are
	 * given as the squares of the corresponding entries in the array §d, and that\n
	 *     $0 \le \{d}[i] < \{d}[j]$ for $i < j$\n
	 * and that $\{rho} > 0$. This is arranged by the calling routine, and is no loss in
	 * generality. The rank-one modified system is thus\n
	 *     $\on{diag}(\{d})*\on{diag}(\{d}) + \{rho}\ \{z}\ \{z}^T$.\n
	 * where we assume the Euclidean norm of $z$ is 1.\n
	 * The method consists of approximating the rational functions in the secular equation by
	 * simpler interpolating rational functions.
	 * \param[in] n The length of all arrays.
	 * \param[in] i
	 *     The index of the eigenvalue to be computed. $0\le\{i}<\{n}$.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in] d
	 *     an array, dimension (§n)\n
	 *     The original eigenvalues. It is assumed that they are in order,
	 *     $0 \le \{d}[i] < \{d}[j]$ for $i < j$.
	 *
	 * \param[in] z
	 *     an array, dimension (§n)\n
	 *     The components of the updating vector.
	 *
	 * \param[out] delta
	 *     an array, dimension (§n)\n
	 *     If $\{n}\ne 1$, §delta contains ($\{d}[j]-\sigma_i$) in its $j$-th component.\n
	 *     If $\{n} = 1$, then §delta[0] = 1. The vector §delta contains the information necessary
	 *     to construct the (singular) eigenvectors.
	 *
	 * \param[in]  rho   The scalar in the symmetric updating formula.
	 * \param[out] sigma The computed $\sigma_i$, the $i$-th updated eigenvalue.
	 * \param[out] work
	 *     an array, dimension (§n)\n
	 *     If $\{n}\ne 1$, §work contains ($\{d}[j]+\sigma_i$) in its $j$-th component.\n
	 *     If $\{n} = 1$, then §work[0] = 1.
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     > 0: if §info = 1, the updating process failed.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Contributors:\n
	 *     Ren-Cang Li, Computer Science Division, University of California at Berkeley, USA     */
	static void dlasd4(int const n, int const i, real const* const d, real const* const z,
	                   real* const delta, real const rho, real& sigma, real* const work, int& info)
	                   /*const*/
	{
		/* orgati
		 *
		 * Logical variable §orgati (origin-at-i?) is used for distinguishing whether §d[i] or
		 * §d[i+1] is treated as the origin.\n
		 *     orgati==true    origin at i\n
		 *     orgati==false   origin at i+1                                                     */
		bool orgati;
		/* swtch3
		 *
		 * Logical variable §swtch3 (switch-for-3-poles?) is for noting if we are working with
		 * THREE poles!*/
		bool swtch3;
		// MAXIT is the maximum number of iterations allowed for each eigenvalue.
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
			swtch3 = false;
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

	/*! §dlasd5 computes the square root of the §i -th eigenvalue of a positive symmetric rank-one
	 *  modification of a 2-by-2 diagonal matrix. Used by §sbdsdc.
	 *
	 * §dlasd5: This subroutine computes the square root of the §i -th eigenvalue of a positive
	 * symmetric rank-one modification of a 2-by-2 diagonal matrix\n
	 *     $\on{diag}(\{d}) \on{diag}(\{d}) + \{rho}\ \{z}\ \{z}^T$.\n
	 * The diagonal entries in the array §d are assumed to satisfy\n
	 *     $0 \le \{d}[i] < \{d}[j]$ for $i < j$.\n
	 * We also assume $\{rho} > 0$ and that the Euclidean norm of the vector §z is one.
	 * \param[in] i
	 *     The index of the eigenvalue to be computed. $\{i}=0$ or $\{i}=1$.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in] d
	 *     an array, dimension (2)\n
	 *     The original eigenvalues. We assume $0 \le \{d}[0] < \{d}[1]$.
	 *
	 * \param[in] z
	 *     an array, dimension (2)\n
	 *     The components of the updating vector.
	 *
	 * \param[out] delta
	 *     an array, dimension (2)\n
	 *     Contains ($\{d}[j]-\sigma_\{i}$) in its $j$-th component. The vector §delta contains the
	 *     information necessary to construct the eigenvectors.
	 *
	 * \param[in]  rho    The scalar in the symmetric updating formula.
	 * \param[out] dsigma The computed $\sigma_\{i}$, the §i -th updated eigenvalue.
	 * \param[out] work
	 *     an array, dimension (2)\n
	 *     §work contains ($\{d}[j]+\sigma_\{i}$) in its $j$-th component.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Contributors:\n
	 *     Ren-Cang Li, Computer Science Division, University of California at Berkeley, USA     */
	static void dlasd5(int const i, real const* const d, real const* const z, real* const delta,
	                   real const rho, real& dsigma, real* const work) /*const*/
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

	/*! §dlasd6 computes the SVD of an updated upper bidiagonal matrix obtained by merging two
	 *  smaller ones by appending a row. Used by §sbdsdc.
	 *
	 * §dlasd6 computes the SVD of an updated upper bidiagonal matrix $B$ obtained by merging two
	 * smaller ones by appending a row. This routine is used only for the problem which requires
	 * all singular values and optionally singular vector matrices in factored form. $B$ is an
	 * $n$ by $m$ matrix with $n=\{nl}+\{nr}+1$ and $m=n+\{sqre}$. A related subroutine, §dlasd1,
	 * handles the case in which all singular values and singular vectors of the bidiagonal matrix
	 * are desired.\n
	 * §dlasd6 computes the SVD as follows:\n
	 *
	 *     \f{align}{
	 *         B &= U^\{in} \b{bm} \{d}_1^\{in} & 0 &      0       & 0 \\
	 *                               \{z}_1^T   & a &   \{z}_2^T   & b \\
	 *                                  0       & 0 & \{d}_2^\{in} & 0 \e{bm} V_T^\{in}\\
	 *
	 *           &= U^\{out} \b{bm} \{d}^\{out} & 0 \e{bm} V_T^\{out}
	 *     \f}\n
	 *
	 * where $\{z}^T = \b{bm} \{z}_1^T & a & \{z}_2^T & b \e{bm} = u^T V_T^T$, and $u$ is a vector
	 * of dimension §m with §alpha and §beta at indices §nl and §nl+1 and zeros elsewhere; and the
	 * entry $b$ is empty if §sqre = 0.\n
	 * The singular values of $B$ can be computed using $\{d}_1$, $\{d}_2$, the first components of
	 * all the right singular vectors of the lower block, and the last components of all the right
	 * singular vectors of the upper block. These components are stored and updated in §vf and §vl,
	 * respectively, in §dlasd6. Hence $U$ and $V_T$ are not explicitly referenced.\n
	 * The singular values are stored in §d. The algorithm consists of two stages:
	 * \li The first stage consists of deflating the size of the problem when there are multiple
	 *     singular values or if there is a zero in the §z vector. For each such occurrence the
	 *     dimension of the secular equation problem is reduced by one. This stage is performed by
	 *     the routine §dlasd7.
	 * \li The second stage consists of calculating the updated singular values. This is done by
	 *     finding the roots of the secular equation via the routine §dlasd4 (as called by
	 *     §dlasd8). This routine also updates §vf and §vl and computes the distances between the
	 *     updated singular values and the old singular values.
	 *
	 * §dlasd6 is called from §dlasda.
	 * \param[in] icompq
	 *     Specifies whether singular vectors are to be computed in factored form:\n
	 *     = 0: Compute singular values only.\n
	 *     = 1: Compute singular vectors in factored form as well.
	 * \param[in] nl   The row dimension of the upper block. $\{nl}\ge 1$.
	 * \param[in] nr   The row dimension of the lower block. $\{nr}\ge 1$.
	 * \param[in] sqre
	 *     = 0: the lower block is an §nr by §nr square matrix.\n
	 *     = 1: the lower block is an §nr by (§nr+1) rectangular matrix.\n
	 *     The bidiagonal matrix has row dimension $n=\{nl}+\{nr}+1$, and column dimension
	 *     $m=n+\{sqre}$.
	 *
	 * \param[in,out] d
	 *     an array, dimension ($\{nl}+\{nr}+1$).\n
	 *     On entry $\{d}[0:\{nl}-1]$ contains the singular values of the upper block, and
	 *     $\{d}[\{nl}+1:n-1]$ contains the singular values of the lower block.\n
	 *     On exit $\{d}[0:n-1]$ contains the singular values of the modified matrix.
	 *
	 * \param[in,out] vf
	 *     an array, dimension ($m$)\n
	 *     On entry, $\{vf}[0:\{nl}]$ contains the first components of all right singular vectors
	 *     of the upper block; and $\{vf}[\{nl}+1:m-1]$ contains the first components of all right
	 *     singular vectors of the lower block.\n
	 *     On exit, §vf contains the first components of all right singular vectors of the
	 *     bidiagonal matrix.
	 *
	 * \param[in,out] vl
	 *     an array, dimension ($m$)\n
	 *     On entry, $\{vl}[0:\{nl}]$ contains the last components of all right singular vectors of
	 *     the upper block; and $\{vl}[\{nl}+1:m-1]$ contains the last components of all right
	 *     singular vectors of the lower block.\n
	 *     On exit, §vl contains the last components of all right singular vectors of the
	 *     bidiagonal matrix.
	 *
	 * \param[in,out] alpha Contains the diagonal element associated with the added row.
	 * \param[in,out] beta  Contains the off-diagonal element associated with the added row.
	 * \param[in,out] idxq
	 *     an integer array, dimension ($n$)\n
	 *     This contains the permutation which will reintegrate the subproblem just solved back
	 *     into sorted order, i.e. $\{d}[\{idxq}[0:n-1]]$ will be in ascending order.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] perm
	 *     an integer array, dimension ($n$)\n
	 *     The permutations (from deflation and sorting) to be applied to each block.\n
	 *     Not referenced if §icompq = 0.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] givptr
	 *     The number of Givens rotations which took place in this subproblem.\n
	 *     Not referenced if §icompq = 0.
	 *
	 * \param[out] Givcol
	 *     an integer array, dimension (§ldgcol, 2)\n
	 *     Each pair of numbers indicates a pair of columns to take place in a Givens rotation.\n
	 *     Not referenced if §icompq = 0.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in]  ldgcol leading dimension of §Givcol, must be at least $n$.
	 * \param[out] Givnum
	 *     an array, dimension (§ldgnum, 2)\n
	 *     Each number indicates the $c$ or $s$ value to be used in the corresponding Givens
	 *     rotation.\n Not referenced if §icompq = 0.
	 *
	 * \param[in]  ldgnum The leading dimension of §Givnum and §Poles, must be at least $n$.
	 * \param[out] Poles
	 *     an array, dimension (§ldgnum, 2)\n
	 *     On exit, $\{Poles}[:,0]$ is an array containing the new singular values obtained from
	 *     solving the secular equation, and $\{Poles}[:,1]$ is an array containing the poles in
	 *     the secular equation.\n  Not referenced if §icompq = 0.
	 *
	 * \param[out] difl
	 *     an array, dimension ($n$)\n
	 *     On exit, $\{difl}[i]$ is the distance between $i$-th updated (undeflated) singular value
	 *     and the $i$-th (undeflated) old singular value.
	 *
	 * \param[out] Difr
	 *     an array,\n dimension (§ldgnum, 2) if §icompq = 1 and\n
	 *                 dimension (§k)         if §icompq = 0.\n
	 *     On exit, $\{Difr}[i,0] = \{d}[i]-\{dsigma}[i+1]$, $\{Difr}[\{k}-1,0]$ is not defined and
	 *     will not be referenced.\n
	 *     If §icompq = 1, $\{Difr}[0:\{k}-1,1]$ is an array containing the normalizing factors for
	 *     the right singular vector matrix.\n See §dlasd8 for details on §difl and §Difr.
	 *
	 * \param[out] z
	 *     an array, dimension ($m$)\n
	 *     The first elements of this array contain the components of the deflation-adjusted
	 *     updating row vector.
	 *
	 * \param[out] k
	 *     Contains the dimension of the non-deflated matrix,
	 *     This is the order of the related secular equation. $1 \le \{k} \le n$.
	 *
	 * \param[out] c
	 *     §c contains garbage if §sqre = 0 and the $c$-value of a Givens rotation related to the
	 *     right null space if §sqre = 1.
	 *
	 * \param[out] s
	 *     §s contains garbage if §sqre = 0 and the $s$-value of a Givens rotation related to the
	 *     right null space if §sqre = 1.
	 *
	 * \param[out] work an array, dimension ($4m$)
	 * \param[out] iwork: an integer array, dimension ($3n$)
	 * \param[out] info
	 *     = 0: Successful exit.\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.\n
	 *     > 0: if §info = 1, a singular value did not converge
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016
	 * \remark
	 *     Contributors:\n
	 *     Ming Gu and Huan Ren, Computer Science Division,
	 *     University of California at Berkeley, USA                                             */
	static void dlasd6(int const icompq, int const nl, int const nr, int const sqre, real* const d,
	                   real* const vf, real* const vl, real& alpha, real& beta, int* const idxq,
	                   int* const perm, int& givptr, int* const Givcol, int const ldgcol,
	                   real* const Givnum, int const ldgnum, real* const Poles, real* const difl,
	                   real* const Difr, real* const z, int& k, real& c, real& s, real* const work,
	                   int* const iwork, int& info) /*const*/
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
		for (int i=0; i<n; i++)
		{
			if (std::fabs(d[i])>orgnrm)
			{
				orgnrm = std::fabs(d[i]);
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

	/*! §dlasd7 merges the two sets of singular values together into a single sorted set.
	 *  Then it tries to deflate the size of the problem. Used by §sbdsdc.
	 *
	 * §dlasd7 merges the two sets of singular values together into a single sorted set. Then it
	 * tries to deflate the size of the problem. There are two ways in which deflation can occur:
	 * when two or more singular values are close together or if there is a tiny entry in the §z
	 * vector. For each such occurrence the order of the related secular equation problem is
	 * reduced by one.\n
	 * §dlasd7 is called from §dlasd6.
	 * \param[in] icompq
	 *     Specifies whether singular vectors are to be computed in compact form, as follows:\n
	 *     0: Compute singular values only.\n
	 *     1: Compute singular vectors of upper bidiagonal matrix in compact form.
	 * \param[in] nl   The row dimension of the upper block. $\{nl}\ge 1$.
	 * \param[in] nr   The row dimension of the lower block. $\{nr}\ge 1$.
	 * \param[in] sqre
	 *     = 0: the lower block is an §nr by §nr square matrix.\n
	 *     = 1: the lower block is an §nr by (§nr+1) rectangular matrix.\n
	 *     The bidiagonal matrix has $n=\{nl}+\{nr}+1$ rows and $m=n+\{sqre}\ge n$ columns.
	 *
	 * \param[out] k
	 *     Contains the dimension of the non-deflated matrix, this is the order of the related
	 *     secular equation. $1 \le \{k} \le n$.
	 *
	 * \param[in,out] d
	 *     an array, dimension ($n$)\n
	 *     On entry §d contains the singular values of the two submatrices to be combined. On exit
	 *     §d contains the trailing ($n-\{k}$) updated singular values (those which were deflated)
	 *     sorted into increasing order.
	 *
	 * \param[out] z
	 *     an array, dimension ($m$)\n
	 *     On exit §z contains the updating row vector in the secular equation.
	 *
	 * \param[out]    zw an array, dimension ($m$)\n Workspace for §z.
	 * \param[in,out] vf
	 *     an array, dimension ($m$)\n
	 *     On entry, $\{vf}[0:\{nl}]$ contains the first components of all right singular vectors
	 *     of the upper block; and $\{vf}[\{nl}+1:m-1]$ contains the first components of all right
	 *     singular vectors of the lower block.\n
	 *     On exit, §vf contains the first components of all right singular vectors of the
	 *     bidiagonal matrix.
	 *
	 * \param[out]    vfw an array, dimension ($m$)\n Workspace for §vf.
	 * \param[in,out] vl
	 *     an array, dimension ($m$)\n
	 *     On entry, $\{vl}[0:\{nl}]$ contains the last components of all right singular vectors of
	 *     the upper block; and $\{vl}[\{nl}+1:m-1]$ contains the last components of all right
	 *     singular vectors of the lower block.\n
	 *     On exit, §vl contains the last components of all right singular vectors of the
	 *     bidiagonal matrix.
	 *
	 * \param[out] vlw    an array, dimension ($m$)\n Workspace for §vl.
	 * \param[in]  alpha  Contains the diagonal element associated with the added row.
	 * \param[in]  beta   Contains the off-diagonal element associated with the added row.
	 * \param[out] dsigma
	 *     an array, dimension ($n$)\n
	 *     Contains a copy of the diagonal elements ($\{k}-1$ singular values and one zero) in the
	 *     secular equation.
	 *
	 * \param[out] idx
	 *     an integer array, dimension ($n$)\n
	 *     This will contain the permutation used to sort the contents of §d into ascending
	 *     order.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] idxp
	 *     an integer array, dimension ($n$)\n
	 *     This will contain the permutation used to place deflated values of §d at the end of the
	 *     array. On output $\{idxp}[1:\{k}-1]$ points to the nondeflated §d values and
	 *     $\{idxp}[\{k}:n-1]$ points to the deflated singular values.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in] idxq
	 *     an array, dimension ($n$)\n
	 *     This contains the permutation which separately sorts the two sub-problems in §d into
	 *     ascending order. Note that entries in the first half of this permutation must first be
	 *     moved one position backward; and entries in the second half must first have nl+1 added
	 *     to their values.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] perm
	 *     an integer array, dimension ($n$)\n
	 *     The permutations (from deflation and sorting) to be applied to each singular block.\n
	 *     Not referenced if §icompq = 0.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] givptr
	 *     The number of Givens rotations which took place in this subproblem.\n
	 *     Not referenced if §icompq = 0.
	 *
	 * \param[out] Givcol
	 *     an integer array, dimension (§ldgcol,2)\n
	 *     Each pair of numbers indicates a pair of columns to take place in a Givens rotation.\n
	 *     Not referenced if §icompq = 0.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in]  ldgcol The leading dimension of §Givcol, must be at least $n$.
	 * \param[out] Givnum
	 *     an array, dimension (§ldgnum, 2)\n
	 *     Each number indicates the $c$ or $s$ value to be used in the corresponding Givens
	 *     rotation.\n Not referenced if §icompq = 0.
	 *
	 * \param[in] ldgnum The leading dimension of §Givnum, must be at least $n$.
	 * \param[out] c
	 *     contains garbage if §sqre = 0 and the $c$-value of a Givens rotation related to the
	 *     right null space if §sqre = 1.
	 *
	 * \param[out] s
	 *     contains garbage if §sqre = 0 and the $s$-value of a Givens rotation related to the
	 *     right null space if §sqre = 1.
	 *
	 * \param[out] info
	 *     = 0: successful exit.\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Contributors:\n
	 *     Ming Gu and Huan Ren, Computer Science Division,
	 *     University of California at Berkeley, USA                                             */
	static void dlasd7(int const icompq, int const nl, int const nr, int const sqre, int& k,
	                   real* const d, real* const z, real* const zw, real* const vf,
	                   real* const vfw, real* const vl, real* const vlw, real const alpha,
	                   real const beta, real* const dsigma, int* const idx, int* const idxp,
	                   int* const idxq, int* const perm, int& givptr, int* const Givcol,
	                   int const ldgcol, real* const Givnum, int const ldgnum, real& c, real& s,
	                   int& info) /*const*/
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
		int j, jprev=-1, km1;
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

	/*! §dlasd8 finds the square roots of the roots of the secular equation, and stores, for each
	 *  element in §d, the distance to its two nearest poles. Used by §sbdsdc.
	 *
	 * §dlasd8 finds the square roots of the roots of the secular equation, as defined by the
	 * values in §dsigma and §z. It makes the appropriate calls to dlasd4, and stores, for each
	 * element in §d, the distance to its two nearest poles (elements in §dsigma). It also updates
	 * the arrays §vf and §vl, the first and last components of all the right singular vectors of
	 * the original bidiagonal matrix.\n
	 * §dlasd8 is called from §dlasd6.
	 * \param[in] icompq
	 *     Specifies whether singular vectors are to be computed in factored form in the calling
	 *     routine:\n
	 *         = 0: Compute singular values only.\n
	 *         = 1: Compute singular vectors in factored form as well.
	 *
	 * \param[in] k
	 *     The number of terms in the rational function to be solved by §dlasd4. $\{k}\ge 1$.
	 *
	 * \param[out] d
	 *     an array, dimension (§k)\n
	 *     On output, §d contains the updated singular values.
	 *
	 * \param[in,out] z
	 *     an array, dimension (§k)\n
	 *     On entry, the first §k elements of this array contain the components of the
	 *     deflation-adjusted updating row vector.\n
	 *     On exit, §z is updated.
	 *
	 * \param[in,out] vf
	 *     an array, dimension (§k)\n
	 *     On entry, §vf contains information passed through §dbede8.\n
	 *     On exit, §vf contains the first §k components of the first components of all right
	 *     singular vectors of the bidiagonal matrix.
	 *
	 * \param[in,out] vl
	 *     an array, dimension (§k)\n
	 *     On entry, §vl contains information passed through §dbede8.\n
	 *     On exit, §vl contains the first §k components of the last components of all right
	 *     singular vectors of the bidiagonal matrix.
	 *
	 * \param[out] difl
	 *     an array, dimension (§k)\n
	 *     On exit, $\{difl}[i] = \{d}[i] - \{dsigma}[i]$.
	 *
	 * \param[out] Difr
	 *     an array,\n dimension (§lddifr, 2) if §icompq = 1 and\n
	 *                 dimension (§k) if §icompq = 0.\n
	 *     On exit, $\{Difr}[i,0] = \{d}[i] - \{dsigma}[i+1]$, $\{Difr}[\{k}-1,0]$ is not defined
	 *     and will not be referenced.\n
	 *     If §icompq = 1, $\{Difr}[0:\{k}-1,1]$ is an array containing the normalizing factors for
	 *     the right singular vector matrix.
	 *
	 * \param[in]     lddifr The leading dimension of §Difr, must be at least §k.
	 * \param[in,out] dsigma
	 *     an array, dimension (§k)\n
	 *     On entry, the first §k elements of this array contain the old roots of the deflated
	 *     updating problem. These are the poles of the secular equation.\n
	 *     On exit, the elements of §dsigma may be very slightly altered in value.
	 *
	 * \param[out] work an array, dimension ($3\{k}$)
	 * \param[out] info
	 *     = 0: successful exit.\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.\n
	 *     > 0: if §info = 1, a singular value did not converge
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017
	 * \remark
	 *     Contributors:\n
	 *     Ming Gu and Huan Ren, Computer Science Division,
	 *     University of California at Berkeley, USA                                             */
	static void dlasd8(int const icompq, int const k, real* const d, real* const z, real* const vf,
	                   real* const vl, real* const difl, real* const Difr, int const lddifr,
	                   real* const dsigma, real* const work, int& info) /*const*/
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
		real diflj, difrj=ZERO, dj, dsigj, dsigjp=ZERO, temp;
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

	/*! §dlasda computes the singular value decomposition (SVD) of a real upper bidiagonal matrix
	 *  with diagonal §d and off-diagonal §e. Used by §sbdsdc.
	 *
	 * Using a divide and conquer approach, §dlasda computes the singular value decomposition (SVD)
	 * of a real upper bidiagonal §n by §m matrix $B$ with diagonal $D$ and offdiagonal $E$, where
	 * §m = §n + §sqre. The algorithm computes the singular values in the SVD $B = U S V_T$. The
	 * orthogonal matrices $U$ and $V_T$ are optionally computed in compact form.\n
	 * A related subroutine, §dlasd0, computes the singular values and the singular vectors in
	 * explicit form.
	 * \param[in] icompq
	 *     Specifies whether singular vectors are to be computed in compact form, as follows\n
	 *     = 0: Compute singular values only.\n
	 *     = 1: Compute singular vectors of upper bidiagonal matrix in compact form.
	 *
	 * \param[in] smlsiz The maximum size of the subproblems at the bottom of the computation tree.
	 * \param[in] n
	 *     The row dimension of the upper bidiagonal matrix.
	 *     This is also the dimension of the main diagonal array §D.
	 *
	 * \param[in] sqre
	 *     Specifies the column dimension of the bidiagonal matrix.\n
	 *     = 0: The bidiagonal matrix has column dimension §m = §n; \n
	 *     = 1: The bidiagonal matrix has column dimension §m = §n + 1.
	 *
	 * \param[in,out] d
	 *     an array, dimension (§n)\n
	 *     On entry §d contains the main diagonal of the bidiagonal matrix.\n
	 *     On exit §d, if §info = 0, contains its singular values.
	 *
	 * \param[in] e
	 *     an array, dimension (§m-1)\n
	 *     Contains the subdiagonal entries of the bidiagonal matrix.\n
	 *     On exit, §e has been destroyed.
	 *
	 * \param[out] U
	 *     an array, dimension (§ldu,§smlsiz) if §icompq = 1, and not referenced if §icompq = 0.\n
	 *     If §icompq = 1, on exit, §U contains the left singular vector matrices of all
	 *     subproblems at the bottom level.
	 *
	 * \param[in] ldu
	 *     $\{ldu}\ge\{n}$.\n
	 *     The leading dimension of arrays §U, §Vt, §Difl, §Difr, §Poles, §Givnum, and §Z.
	 *
	 * \param[out] Vt
	 *     an array, dimension (§ldu,$\{smlsiz}+1$) if §icompq = 1, and not referenced if
	 *     §icompq = 0.\n If §icompq = 1, on exit, $\{Vt}^T$ contains the right singular vector
	 *     matrices of all subproblems at the bottom level.
	 *
	 * \param[out] k
	 *     an integer array, dimension (§n) if §icompq = 1 and dimension 1 if §icompq = 0.\n
	 *     If §icompq = 1, on exit, $\{k}[i]$ is the dimension of the $i$-th secular equation on
	 *     the computation tree.
	 *
	 * \param[out] Difl
	 *     an array, dimension (§ldu,§nlvl), where $\{nlvl}=\lfloor\log_2(\{n}/\{smlsiz})\rfloor$.
	 *
	 * \param[out] Difr
	 *     an array,\n dimension (§ldu, $2\{nlvl}$) if §icompq = 1 and\n
	 *                 dimension (§n) if §icompq = 0.\n
	 *     If §icompq = 1, on exit, $\{Difl}[0:\{n}-1,i]$ and $\{Difr}[0:\{n}-1,2i]$ record
	 *     distances between singular values on the $i$-th level and singular values on the
	 *     $(i-1)$-th level, and $\{Difr}[0:\{n}-1,2i+1]$ contains the normalizing factors for the
	 *     right singular vector matrix. See §dlasd8 for details.
	 *
	 * \param[out] Z
	 *     an array,\n dimension (§ldu,§nlvl) if §icompq = 1 and\n
	 *                 dimension (§n) if §icompq = 0.\n
	 *     The first §k elements of $\{Z}[0,i]$ contain the components of the deflation-adjusted
	 *     updating row vector for subproblems on the $i$-th level.
	 *
	 * \param[out] Poles
	 *     an array, dimension (§ldu,$2\{nlvl}$) if §icompq = 1, and not referenced if §icompq = 0.
	 *     \n If §icompq = 1, on exit, $\{Poles}[:,2i]$ and $\{Poles}[:,2i+1]$ contain the new and
	 *     old singular values involved in the secular equations on the $i$-th level.
	 *
	 * \param[out] Givptr
	 *     an integer array, dimension (§n) if §icompq = 1, and not referenced if §icompq = 0.\n
	 *     If §icompq = 1, on exit, $\{Givptr}[i]$ records the number of Givens rotations performed
	 *     on the $i$-th problem on the computation tree.
	 *
	 * \param[out] Givcol
	 *     an integer array, dimension (§ldgcol,$2\{nlvl}$) if §icompq = 1, and not referenced if
	 *     §icompq = 0.\n If §icompq = 1, on exit, for each $i$, $\{Givcol}[:,2i]$ and
	 *     $\{Givcol}[:,2i+1]$ record the locations of Givens rotations performed on the $i$-th
	 *     level on the computation tree.\n
	 *     NOTE: zero-base indices!
	 *
	 * \param[in]  ldgcol $\{ldgcol}\ge\{n}$. The leading dimension of arrays §Givcol and §Perm.
	 * \param[out] Perm
	 *     an integer array, dimension (§ldgcol,§nlvl) if §icompq = 1, and not referenced if
	 *     §icompq = 0.\n If §icompq = 1, on exit, $\{Perm}[:,i]$ records permutations done on the
	 *     $i$-th level of the computation tree.\n
	 *     NOTE: zero-base indices!
	 *
	 * \param[out] Givnum
	 *     an array, dimension (§ldu,$2\{nlvl}$) if §icompq = 1, and not referenced if §icompq = 0.
	 *     \n If §icompq = 1, on exit, for each $i$, $\{Givnum}[0,2i]$ and $\{Givnum}[0,2i+1]$
	 *     record the $c$- and $s$-values of Givens rotations performed on the $i$-th level on the
	 *     computation tree.
	 *
	 * \param[out] c
	 *     an array, dimension (§n) if §icompq = 1, and dimension 1 if §icompq = 0.\n
	 *     If §icompq = 1 and the $i$-th subproblem is not square, on exit, $\{c}[i]$ contains the
	 *     $c$-value of a Givens rotation related to the right null space of the $i$-th subproblem.
	 *
	 * \param[out] s
	 *     an array, dimension (§n) if §icompq = 1, and dimension 1 if §icompq = 0.\n
	 *     If §icompq = 1 and the $i$-th subproblem is not square, on exit, $\{s}[i]$ contains the
	 *     $s$-value of a Givens rotation related to the right null space of the $i$-th subproblem.
	 *
	 * \param[out] work  an array, dimension ($6\{n}+(\{smlsiz}+1)^2$).
	 * \param[out] iwork an integer array, dimension ($7\{n}$)
	 * \param[out] info
	 *     = 0: successful exit.\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value.
	 *     > 0: if §info = 1, a singular value did not converge
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017
	 * \remark
	 *     Contributors:\n
	 *     Ming Gu and Huan Ren, Computer Science Division,
	 *     University of California at Berkeley, USA                                             */
	static void dlasda(int const icompq, int const smlsiz, int const n, int const sqre,
	                   real* const d, real* const e, real* const U, int const ldu, real* const Vt,
	                   int* const k, real* const Difl, real* const Difr, real* const Z,
	                   real* const Poles, int* const Givptr, int* const Givcol, int const ldgcol,
	                   int* const Perm, real* const Givnum, real* const c, real* const s,
	                   real* const work, int* const iwork, int& info) /*const*/
	{
		// Test the input parameters.
		info = 0;
		if (icompq<0 || icompq>1)
		{
			info = -1;
		}
		else if (smlsiz<3)
		{
			info = -2;
		}
		else if (n<0)
		{
			info = -3;
		}
		else if (sqre<0 || sqre>1)
		{
			info = -4;
		}
		else if (ldu<(n+sqre))
		{
			info = -8;
		}
		else if (ldgcol<n)
		{
			info = -17;
		}
		if (info!=0)
		{
			xerbla("DLASDA", -info);
			return;
		}
		int m = n + sqre;
		// If the input matrix is too small, call dlasdq to find the SVD.
		if (n<=smlsiz)
		{
			if (icompq==0)
			{
				dlasdq("U", sqre, n, 0, 0, 0, d, e, Vt, ldu, U, ldu, U, ldu, work, info);
			}
			else
			{
				dlasdq("U", sqre, n, m, n, 0, d, e, Vt, ldu, U, ldu, U, ldu, work, info);
			}
			return;
		}
		// Book-keeping and set up the computation tree.
		int inode = 0;
		int ndiml = inode + n;
		int ndimr = ndiml + n;
		int idxq  = ndimr + n;
		int iwk   = idxq  + n;
		int ncc = 0;
		int nru = 0;
		int smlszp = smlsiz + 1;
		int vf = 0;
		int vl = vf + m;
		int nwork1 = vl + m;
		int nwork2 = nwork1 + smlszp*smlszp;
		int nlvl, nd;
		dlasdt(n, nlvl, nd, &iwork[inode], &iwork[ndiml], &iwork[ndimr], smlsiz);
		// for the nodes on bottom level of the tree, solve their subproblems by dlasdq.
		int ndb1 = (nd-1) / 2;
		int i, ic, idxqi, itemp, j, nl, nlf, nlp1, nr, nrf, nrp1, sqrei, vfi, vli;
		for (i=ndb1; i<nd; i++)
		{
			// ic : center row of each node
			// nl : number of rows of left  subproblem
			// nr : number of rows of right subproblem
			// nlf: starting row of the left  subproblem
			// nrf: starting row of the right subproblem
			ic = iwork[inode+i];
			nl = iwork[ndiml+i];
			nlp1 = nl + 1;
			nr = iwork[ndimr+i];
			nlf = ic - nl;
			nrf = ic + 1;
			idxqi = idxq + nlf;
			vfi = vf + nlf;
			vli = vl + nlf;
			sqrei = 1;
			if (icompq==0)
			{
				dlaset("A", nlp1, nlp1, ZERO, ONE, &work[nwork1], smlszp);
				dlasdq("U", sqrei, nl, nlp1, nru, ncc, &d[nlf], &e[nlf], &work[nwork1], smlszp,
				       &work[nwork2], nl, &work[nwork2], nl, &work[nwork2], info);
				itemp = nwork1 + nl*smlszp;
				Blas<real>::dcopy(nlp1, &work[nwork1], 1, &work[vfi], 1);
				Blas<real>::dcopy(nlp1, &work[itemp], 1, &work[vli], 1);
			}
			else
			{
				dlaset("A", nl, nl, ZERO, ONE, &U[nlf], ldu);
				dlaset("A", nlp1, nlp1, ZERO, ONE, &Vt[nlf], ldu);
				dlasdq("U", sqrei, nl, nlp1, nl, ncc, &d[nlf], &e[nlf], &Vt[nlf], ldu, &U[nlf],
				       ldu, &U[nlf], ldu, &work[nwork1], info);
				Blas<real>::dcopy(nlp1, &Vt[nlf], 1, &work[vfi], 1);
				Blas<real>::dcopy(nlp1, &Vt[nlf+ldu*nl], 1, &work[vli], 1);
			}
			if (info!=0)
			{
				return;
			}
			for (j=0; j<nl; j++)
			{
				iwork[idxqi+j] = j;
			}
			if (i==(nd-1) && sqre==0)
			{
				sqrei = 0;
			}
			else
			{
				sqrei = 1;
			}
			idxqi += nlp1;
			vfi += nlp1;
			vli += nlp1;
			nrp1 = nr + sqrei;
			if (icompq==0)
			{
				dlaset("A", nrp1, nrp1, ZERO, ONE, &work[nwork1], smlszp);
				dlasdq("U", sqrei, nr, nrp1, nru, ncc, &d[nrf], &e[nrf], &work[nwork1], smlszp,
				       &work[nwork2], nr, &work[nwork2], nr, &work[nwork2], info);
				itemp = nwork1 + (nrp1-1)*smlszp;
				Blas<real>::dcopy(nrp1, &work[nwork1], 1, &work[vfi], 1);
				Blas<real>::dcopy(nrp1, &work[itemp], 1, &work[vli], 1);
			}
			else
			{
				dlaset("A", nr, nr, ZERO, ONE, &U[nrf], ldu);
				dlaset("A", nrp1, nrp1, ZERO, ONE, &Vt[nrf], ldu);
				dlasdq("U", sqrei, nr, nrp1, nr, ncc, &d[nrf], &e[nrf], &Vt[nrf], ldu, &U[nrf],
				       ldu, &U[nrf], ldu, &work[nwork1], info);
				Blas<real>::dcopy(nrp1, &Vt[nrf], 1, &work[vfi], 1);
				Blas<real>::dcopy(nrp1, &Vt[nrf+ldu*(nrp1-1)], 1, &work[vli], 1);
			}
			if (info!=0)
			{
				return;
			}
			for (j=0; j<nr; j++)
			{
				iwork[idxqi+j] = j;
			}
		}
		// Now conquer each subproblem bottom-up.
		j = (1 << nlvl) - 1; // 2^nlvl - 1
		int lf, ll, lvl, lvl2, ldgl, ldgl2, ldul, ldul2;
		real alpha, beta;
		for (lvl=nlvl-1; lvl>=0; lvl--)
		{
			lvl2 = lvl * 2;
			// Find the first node lf and last node ll on the current level lvl.
			if (lvl==0)
			{
				lf = 0;
				ll = 0;
			}
			else
			{
				lf = (1 << lvl) - 1; // 2^lvl - 1
				ll = 2*lf;
			}
			if (icompq!=0)
			{
				ldgl  = ldgcol * lvl;
				ldgl2 = ldgcol * lvl2;
				ldul  = ldu    * lvl;
				ldul2 = ldu    * lvl2;
			}
			for (i=lf; i<=ll; i++)
			{
				ic = iwork[inode+i];
				nl = iwork[ndiml+i];
				nr = iwork[ndimr+i];
				nlf = ic - nl;
				//nrf = ic + 1;
				if (i==ll)
				{
					sqrei = sqre;
				}
				else
				{
					sqrei = 1;
				}
				vfi   = vf   + nlf;
				vli   = vl   + nlf;
				idxqi = idxq + nlf;
				alpha = d[ic];
				beta  = e[ic];
				if (icompq==0)
				{
					dlasd6(icompq, nl, nr, sqrei, &d[nlf], &work[vfi], &work[vli], alpha, beta,
					       &iwork[idxqi], Perm, Givptr[0], Givcol, ldgcol, Givnum, ldu, Poles,
					       Difl, Difr, Z, k[0], c[0], s[0], &work[nwork1], &iwork[iwk], info);
				}
				else
				{
					j--;
					dlasd6(icompq, nl, nr, sqrei, &d[nlf], &work[vfi], &work[vli], alpha, beta,
					       &iwork[idxqi], &Perm[nlf+ldgl], Givptr[j], &Givcol[nlf+ldgl2], ldgcol,
					       &Givnum[nlf+ldul2], ldu, &Poles[nlf+ldul2], &Difl[nlf+ldul],
					       &Difr[nlf+ldul2], &Z[nlf+ldul], k[j], c[j], s[j], &work[nwork1],
					       &iwork[iwk], info);
				}
				if (info!=0)
				{
					return;
				}
			}
		}
	}

	/*! §dlasdq computes the SVD of a real bidiagonal matrix with diagonal §d and off-diagonal §e.
	 *  Used by §sbdsdc.
	 *
	 * §dlasdq computes the singular value decomposition (SVD) of a real (upper or lower)
	 * bidiagonal matrix with diagonal §d and offdiagonal §e, accumulating the transformations if
	 * desired. Letting $B$ denote the input bidiagonal matrix, the algorithm computes orthogonal
	 * matrices $Q$ and $P$ such that $B = Q S P^T$ ($P^T$ denotes the transpose of $P$). The
	 * singular values $S$ are overwritten on §d.\n
	 * The input matrix §U  is changed to $\{U} Q$    if desired.\n
	 * The input matrix §Vt is changed to $P^T \{Vt}$ if desired.\n
	 * The input matrix §C  is changed to $Q^T \{C}$  if desired.\n
	 * See "Computing  Small Singular Values of Bidiagonal Matrices With Guaranteed High Relative
	 * Accuracy," by J. Demmel and W. Kahan, LAPACK Working Note #3, for a detailed description of
	 * the algorithm.
	 * \param[in] uplo
	 *     On entry, uplo specifies whether the input bidiagonal matrix is upper or lower
	 *     bidiagonal, and whether it is square or not.\n
	 *         §uplo = 'U' or 'u': $B$ is upper bidiagonal.\n
	 *         §uplo = 'L' or 'l': $B$ is lower bidiagonal.
	 *
	 * \param[in] sqre
	 *     = 0: then the input matrix is §n by §n.\n
	 *     = 1: then the input matrix is §n by (§n+1) if §uplo = 'U' and (§n+1) by §n if
	 *     §uplo = 'L'.\n
	 *     The bidiagonal matrix has $\{n}=\{nl}+\{nr}+1$ rows and $m=\{n}+\{sqre}\ge\{n}$ columns.
	 *
	 * \param[in] n
	 *     On entry, §n specifies the number of rows and columns in the matrix.
	 *     §n must be at least 0.
	 *
	 * \param[in] ncvt
	 *     On entry, §ncvt specifies the number of columns of the matrix §Vt.
	 *     §ncvt must be at least 0.
	 *
	 * \param[in] nru
	 *     On entry, §nru specifies the number of rows of the matrix §U.
	 *     §nru must be at least 0.
	 *
	 * \param[in] ncc
	 *     On entry, §ncc specifies the number of columns of the matrix §C.
	 *     §ncc must be at least 0.
	 *
	 * \param[in,out] d
	 *     an array, dimension (§n)\n
	 *     On entry, §d contains the diagonal entries of the bidiagonal matrix whose SVD is
	 *     desired.\n
	 *     On normal exit, §d contains the singular values in ascending order.
	 *
	 * \param[in,out] e
	 *     an array. dimension (§n-1) if §sqre = 0 and §n if §sqre = 1.\n
	 *     On entry, the entries of §e contain the offdiagonal entries of the bidiagonal matrix
	 *     whose SVD is desired.\n
	 *     On normal exit, §e will contain 0. If the algorithm does not converge, §d and §e will
	 *     contain the diagonal and superdiagonal entries of a bidiagonal matrix orthogonally
	 *     equivalent to the one given as input.
	 *
	 * \param[in,out] Vt
	 *     an array, dimension (§ldvt, §ncvt)\n
	 *     On entry, contains a matrix which on exit has been premultiplied by $P^T$, dimension
	 *     §n by §ncvt if §sqre = 0 and (§n+1) by §ncvt if §sqre = 1 (not referenced if §ncvt = 0).
	 *
	 * \param[in] ldvt
	 *     On entry, §ldvt specifies the leading dimension of §Vt as declared in the calling
	 *     (sub)program. §ldvt must be at least 1. If §ncvt is nonzero §ldvt must also be at least
	 *     §n.
	 *
	 * \param[in,out] U
	 *     an array, dimension (§ldu,§n)\n
	 *     On entry, contains a matrix which on exit has been postmultiplied by $Q$, dimension
	 *     §nru by §n if §sqre = 0 and §nru by (§n+1) if §sqre = 1 (not referenced if §nru = 0).
	 *
	 * \param[in] ldu
	 *     On entry, §ldu specifies the leading dimension of §U as declared in the calling
	 *     (sub)program. §ldu must be at least $\max(1,\{nru})$.
	 *
	 * \param[in,out] C
	 *     an array, dimension (§ldc,§ncc)\n
	 *     On entry, contains an §n by §ncc matrix which on exit has been premultiplied by $Q^T$
	 *     dimension §n by §ncc if §sqre = 0 and (§n+1) by §ncc if §sqre = 1
	 *     (not referenced if §ncc = 0).
	 *
	 * \param[in] ldc
	 *     On entry, §ldc specifies the leading dimension of §C as declared in the calling
	 *     (sub)program. §ldc must be at least 1. If §ncc is nonzero, §ldc must also be at least
	 *     §n.
	 *
	 * \param[out] work
	 *     an array, dimension ($4\{n}$)\n
	 *     Workspace. Only referenced if one of §ncvt, §nru, or §ncc is nonzero, and if §n is at
	 *     least 2.
	 *
	 * \param[out] info
	 *     On exit, a value of 0 indicates a successful exit.\n
	 *     If §info < 0, argument number -§info is illegal.\n
	 *     If §info > 0, the algorithm did not converge, and info specifies how many superdiagonals
	 *                   did not converge.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016
	 * \remark
	 *     Contributors:\n
	 *     Ming Gu and Huan Ren, Computer Science Division,
	 *     University of California at Berkeley, USA                                             */
	static void dlasdq(char const* const uplo, int sqre, int const n, int const ncvt,
	                   int const nru, int const ncc, real* const d, real* const e, real* const Vt,
	                   int const ldvt, real* const U, int const ldu, real* const C, int const ldc,
	                   real* const work, int& info) /*const*/
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

	/*! §dlasdt creates a tree of subproblems for bidiagonal divide and conquer. Used by §sbdsdc.
	 *
	 * §dlasdt creates a tree of subproblems for bidiagonal divide and sconquer.
	 * \param[in]  n     On entry, the number of diagonal elements of the bidiagonal matrix.
	 * \param[out] lvl   On exit, the number of levels on the computation tree.
	 * \param[out] nd    On exit, the number of nodes on the tree.
	 * \param[out] inode
	 *     an integer array, dimension (§n)\n On exit, centers of subproblems.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] ndiml
	 *     an integer array, dimension (§n)\n On exit, row dimensions of left children.
	 *
	 * \param[out] ndimr
	 *     an integer array, dimension (§n)\n On exit, row dimensions of right children.
	 *
	 * \param[in] msub
	 *     On entry, the maximum row dimension each subproblem at the bottom of the tree can be of.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Contributors:\n
	 *     Ming Gu and Huan Ren, Computer Science Division,
	 *     University of California at Berkeley, USA                                             */
	static void dlasdt(int const n, int& lvl, int& nd, int* const inode, int* const ndiml,
	                   int* const ndimr, int const msub) /*const*/
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

	/*! §dlaset initializes the off-diagonal elements and the diagonal elements of a matrix to
	 *  given values.
	 *
	 * §dlaset initializes an §m by §n matrix $A$ to §beta on the diagonal and §alpha on the
	 * offdiagonals.
	 * \param[in] uplo
	 *     Specifies the part of the matrix $A$ to be set.\n
	 *     ='U': Upper triangular part is set; the strictly lower triangular part of §A is not
	 *           changed.\n
	 *     ='L': Lower triangular part is set; the strictly upper triangular part of §A is not
	 *           changed.\n
	 *     Otherwise: All of the matrix $A$ is set.
	 *
	 * \param[in]  m     The number of rows of the matrix $A$. $\{m}\ge 0$.
	 * \param[in]  n     The number of columns of the matrix $A$. $\{n}\ge 0$.
	 * \param[in]  alpha The constant to which the offdiagonal elements are to be set.
	 * \param[in]  beta  The constant to which the diagonal elements are to be set.
	 * \param[out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On exit, the leading §m by §n submatrix of §A is set as follows:\n
	 *     if §uplo = 'U',       $\{A}[i,j]=\{alpha}$, $0  \le i<j$,   $0\le j<\{n}$,\n
	 *     if §uplo = 'L',       $\{A}[i,j]=\{alpha}$, $j+1\le i<\{m}$,$0\le j<\{n}$,\n
	 *     otherwise,&emsp;&emsp;$\{A}[i,j]=\{alpha}$, $0  \le i<\{m}$,$0\le j<\{n}$, $i\ne j$,\n
	 *     and, for all §uplo,   $\{A}[i,i]=\{beta}$,  $0  \le i<\min(\{m},\{n})$.
	 *
	 * \param[in] lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                        */
	static void dlaset(char const* const uplo, int const m, int const n, real const alpha,
	                   real const beta, real* const A, int const lda) /*const*/
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

	/*! §dlasq1 computes the singular values of a real square bidiagonal matrix. Used by §sbdsqr.
	 *
	 * §dlasq1 computes the singular values of a real §n by §n bidiagonal matrix with diagonal §d
	 * and off-diagonal §e. The singular values are computed to high relative accuracy, in the
	 * absence of denormalization, underflow and overflow. The algorithm was first presented in\n
	 *     "Accurate singular values and differential qd algorithms" by K. V. Fernando and
	 *     B. N. Parlett, Numer. Math., Vol-67, No. 2, pp. 191-230, 1994,\n
	 * and the present implementation is described in
	 *     "An implementation of the dqds Algorithm (Positive Case)", LAPACK Working Note.
	 * \param[in]     n The number of rows and columns in the matrix. $\{n}\ge 0$.
	 * \param[in,out] d
	 *     an array, dimension (§n)\n
	 *     On entry, §d contains the diagonal elements of the bidiagonal matrix whose SVD is
	 *     desired.\n
	 *     On normal exit, §d contains the singular values in decreasing order.
	 *
	 * \param[in,out] e
	 *     an array, dimension (§n)\n
	 *     On entry, elements $\{e}[0:\{n}-2]$ contain the off-diagonal elements of the bidiagonal
	 *     matrix whose SVD is desired.\n
	 *     On exit, §e is overwritten.
	 *
	 * \param[out] work an array, dimension ($4\{n}$)
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value\n
	 *     > 0: the algorithm failed
	 *          \li =1, a split was marked by a positive value in §e
	 *          \li =2, current block of Z not diagonalized after $100\{n}$ iterations (in inner
	 *                  while loop). On exit §d and §e represent a matrix with the same singular
	 *                  values which the calling subroutine could use to finish the computation, or
	 *                  even feed back into §dlasq1
	 *          \li =3, termination criterion of outer while loop not met
	 *                  (program created more than §n unreduced blocks)
	 *
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlasq1(int const n, real* const d, real* const e, real* const work, int& info)
	                   /*const*/
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
		// Early return if sigmx is zero (matrix is already diagonal).
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

	/*! §dlasq2 computes all the eigenvalues of the symmetric positive definite tridiagonal matrix
	 *  associated with the qd Array §Z to high relative accuracy. Used by §sbdsqr and §sstegr.
	 *
	 * §dlasq2 computes all the eigenvalues of the symmetric positive definite tridiagonal matrix
	 * associated with the qd array §Z to high relative accuracy, in the absence of
	 * denormalization, underflow and overflow.\n
	 * To see the relation of §Z to the tridiagonal matrix, let $L$ be a unit lower bidiagonal
	 * matrix with subdiagonals $\{Z}[1,3,5,\ldots]$ and let $U$ be an upper bidiagonal matrix with
	 * 1's above and diagonal $\{Z}[0,2,4,\ldots]$. The tridiagonal is $LU$ or, if you prefer, the
	 * symmetric tridiagonal to which it is similar.\n
	 * Note: §dlasq2 defines a logical variable, §IEEE, which is true on machines which follow
	 * ieee-754 floating-point standard in their handling of infinities and NaNs, and false
	 * otherwise. This variable is passed to §dlasq3.
	 * \param[in]     n The number of rows and columns in the matrix. $\{n}ge 0$.
	 * \param[in,out] Z
	 *     an array, dimension ($4\{n}$)\n
	 *     On entry §Z holds the qd array.\n
	 *     On exit, entries 0 to §n-1 hold the eigenvalues in decreasing order, $\{Z}[2\{n}]$ holds
	 *     the trace, and $\{Z}[2\{n}+1]$ holds the sum of the eigenvalues. If $\{n}>2$, then
	 *     $\{Z}[2\{n}+2]$ holds the iteration count, $\{Z}[2\{n}+3]$ holds $\{ndivs}/\{nin}^2$,
	 *     and $\{Z}[2\{n}+4]$ holds the percentage of shifts that failed.
	 *
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if the $i$-th argument is a scalar and had an illegal value, then §info = $-i$, if
	 *          the $i$-th argument is an array and the $j$-entry had an illegal value, then
	 *          §info = $-(100i+j)$\n
	 *     > 0: the algorithm failed
	 *          \li = 1, a split was marked by a positive value in §Z
	 *          \li = 2, current block of §Z not diagonalized after $100\{n}$ iterations (in inner
	 *                   while loop).\n&emsp;&emsp; On exit §Z holds a qd array with the same
	 *                   eigenvalues as the given §Z.
	 *          \li = 3, termination criterion of outer while loop not met
	 *                   (program created more than §n unreduced blocks)
	 *
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlasq2(int const n, real* const Z, int& info) /*const*/
	{
		/* Local Variables: §i0:§n0 defines a current unreduced segment of §Z. The shifts are
		 * accumulated in §sigma. Iteration count is in §iter. Ping-pong is controlled by §pp
		 * (alternates between 0 and 1).                                                         */
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
			// e[n0] holds the value of sigma when submatrix in i0:n0 splits from the rest of the
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
			// Maximum number of iterations exceeded, restore the shift sigma and place the new
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

	/*! §dlasq3 checks for deflation, computes a shift and calls §dqds. Used by §sbdsqr.
	 *
	 * §dlasq3 checks for deflation, computes a shift (§tau) and calls §dqds. In case of failure it
	 * changes shifts, and tries again until output is positive.
	 * \param[in]     i0 First index. (NOTE: zero-based index!)
	 * \param[in,out] n0 Last index. (NOTE: zero-based index!)
	 * \param[in,out] Z  an array, dimension ($4\{n0}+4$)\n §Z holds the qd array.
	 * \param[in,out] pp
	 *     §pp = 0 for ping, §pp = 1 for pong.\n
	 *     §pp = 2 indicates that flipping was applied to the §Z array and that the initial tests
	 *             for deflation should not be performed.
	 *
	 * \param[out]    dmin  Minimum value of §d.
	 * \param[out]    sigma Sum of shifts used in current segment.
	 * \param[in,out] desig Lower order part of §sigma
	 * \param[in]     qmax  Maximum value of §q.
	 * \param[in,out] nfail Increment §nfail by 1 each time the shift was too big.
	 * \param[in,out] iter  Increment §iter by 1 for each iteration.
	 * \param[in,out] ndiv  Increment §ndiv by 1 for each division.
	 * \param[in]     ieee  Flag for IEEE or non IEEE arithmetic (passed to §dlasq5).
	 * \param[in,out] ttype Shift type.
	 * \param[in,out] dmin1,
	 *                dmin2,
	 *                dn,
	 *                dn1,
	 *                dn2,
	 *                g,
	 *                tau
	 *     These are passed as arguments in order to save their values between calls to §dlasq3.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016                                                                           */
	static void dlasq3(int const i0, int& n0, real* const Z, int& pp, real& dmin, real& sigma,
	                   real& desig, real& qmax, int& nfail, int& iter, int& ndiv, bool const ieee,
	                   int& ttype, real& dmin1, real& dmin2, real& dn, real& dn1, real& dn2,
	                   real& g, real& tau) /*const*/
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

	/*! §dlasq4 computes an approximation to the smallest eigenvalue using values of §d from the
	 *  previous transform. Used by §sbdsqr.
	 *
	 * §dlasq4 computes an approximation §tau to the smallest eigenvalue using values of §d from
	 * the previous transform.
	 * \param[in]     i0    First index. (note: zero-based index!)
	 * \param[in]     n0    Last index. (note: zero-based index!)
	 * \param[in]     Z     an array, dimension ($4(\{n0}+1)$)\n §Z holds the qd array.
	 * \param[in]     pp    §pp = 0 for ping, §pp = 1 for pong.
	 * \param[in]     n0in  The value of §n0 at start of §eigtest. (note: zero-based index!)
	 * \param[in]     dmin  Minimum value of §d.
	 * \param[in]     dmin1 Minimum value of §d, excluding $\{d}[\{n0}]$.
	 * \param[in]     dmin2 Minimum value of §d, excluding $\{d}[\{n0}]$ and $\{d}[\{n0}-1]$.
	 * \param[in]     dn    $\{d}[N]$
	 * \param[in]     dn1   $\{d}[N-1]$
	 * \param[in]     dn2   $\{d}[N-2]$
	 * \param[out]    tau   This is the shift.
	 * \param[out]    ttype Shift type.
	 * \param[in,out] g
	 *     §g is passed as an argument in order to save its value between calls to §dlasq4.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016                                                                           */
	static void dlasq4(int const i0, int const n0, real const* const Z, int const pp,
	                   int const n0in, real const dmin, real const dmin1, real const dmin2,
	                   real const dn, real const dn1, real const dn2, real& tau, int& ttype,
	                   real& g) /*const*/
	{
		const real CNST1 = real(0.563); // CNST1 = 9/16
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

	/*! §dlasq5 computes one dqds transform in ping-pong form. Used by §sbdsqr and §sstegr.
	 *
	 * §dlasq5 computes one dqds transform in ping-pong form, one version for ieee machines another
	 * for non ieee machines.
	 * \param[in] i0 First index. (note: zero-based index!)
	 * \param[in] n0 Last index. (note: zero-based index!)
	 * \param[in] Z
	 *     an array, dimension ($4N$)\n
	 *     holds the qd array. §emin is stored in $\{Z}[4(n0+1)]$ to avoid an extra argument.
	 *
	 * \param[in]  pp    §pp = 0 for ping, §pp = 1 for pong.
	 * \param[in]  tau   This is the shift.
	 * \param[in]  sigma This is the accumulated shift up to this step.
	 * \param[out] dmin  Minimum value of §d.
	 * \param[out] dmin1 Minimum value of §d, excluding $\{d}[\{n0}]$.
	 * \param[out] dmin2 Minimum value of §d, excluding $\{d}[\{n0}]$ and $\{d}[\{n0}-1]$.
	 * \param[out] dn    $\{d}[\{n0}]$, the last value of §d.
	 * \param[out] dnm1  $\{d}[\{n0}-1]$.
	 * \param[out] dnm2  $\{d}[\{n0}-2]$.
	 * \param[in]  ieee  Flag for ieee or non ieee arithmetic.
	 * \param[in]  eps   This is the value of epsilon used.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017                                                                           */
	static void dlasq5(int const i0, int const n0, real* const Z, int const pp, real tau,
	                   real const sigma, real& dmin, real& dmin1, real& dmin2, real& dn,
	                   real& dnm1, real& dnm2, bool const ieee, real const eps) /*const*/
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

	/*! §dlasq6 computes one dqd transform in ping-pong form. Used by §sbdsqr and §sstegr.
	 *
	 * §dlasq6 computes one dqd (shift equal to zero) transform in ping-pong form, with protection
	 * against underflow and overflow.
	 * \param[in] i0 First index. (note: zero-based index!)
	 * \param[in] n0 Last index. (note: zero-based index!)
	 * \param[in] Z
	 *     an array, dimension ($4N$)\n
	 *     §Z holds the qd array. §emin is stored in $\{Z}[4\{n0}+3]$ to avoid an extra argument.
	 * \param[in]  pp    §pp = 0 for ping, §pp = 1 for pong.
	 * \param[out] dmin  Minimum value of §d.
	 * \param[out] dmin1 Minimum value of §d, excluding $\{d}[\{n0}]$.
	 * \param[out] dmin2 Minimum value of §d, excluding $\{d}[\{n0}]$ and $\{d}[\{n0}-1]$.
	 * \param[out] dn    $\{d}[\{n0}]$, the last value of §d.
	 * \param[out] dnm1  $\{d}[\{n0}-1]$.
	 * \param[out] dnm2  $\{d}[\{n0}-2]$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlasq6(int const i0, int const n0, real* const Z, int const pp, real& dmin,
	                   real& dmin1, real& dmin2, real& dn, real& dnm1, real& dnm2) /*const*/
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

	/*! §dlasr applies a sequence of plane rotations to a general rectangular matrix.
	 *
	 * §dlasr applies a sequence of plane rotations to a real matrix $A$, from either the left or
	 * the right.\n
	 * When §side = 'L', the transformation takes the form\n
	 *     $A = P A$\n
	 * and when §side = 'R', the transformation takes the form\n
	 *     $A = A P^T$\n
	 * where $P$ is an orthogonal matrix consisting of a sequence of $z$ plane rotations, with
	 * $z=\{m}$ when §side = 'L' and $z=\{n}$ when §side = 'R', and $P^T$ is the transpose of
	 * $P$.\n
	 * When §direct = 'F' (Forward sequence), then\n
	 *     $P = P(z-2) \ldots P(1) P(0)$\n
	 * and when §direct = 'B' (Backward sequence), then\n
	 *     $P = P(0) P(1) \ldots P(z-2)$\n
	 * where $P(k)$ is a plane rotation matrix defined by the 2 by 2 rotation\n
	 *     $R(k) = \b{bm}  c(k) & s(k) \\
	 *                    -s(k) & c(k) \e{bm}$.\n
	 * When §pivot = 'V' (Variable pivot), the rotation is performed for the plane $(k,k+1)$,
	 * i.e., $P(k)$ has the form\n
	 *     $P(k)=\b{bm}  1  &        &     &       &      &     &        &     \\
	 *                      & \ldots &     &       &      &     &        &     \\
	 *                      &        &  1  &       &      &     &        &     \\
	 *                      &        &     &  c(k) & s(k) &     &        &     \\
	 *                      &        &     & -s(k) & c(k) &     &        &     \\
	 *                      &        &     &       &      &  1  &        &     \\
	 *                      &        &     &       &      &     & \ldots &     \\
	 *                      &        &     &       &      &     &        &  1  \e{bm}$\n
	 * where $R(k)$ appears as a rank-2 modification to the identity matrix in rows and columns
	 * $k$ and $k+1$.\n
	 * When §pivot = 'T' (Top pivot), the rotation is performed for the plane $(1,k+1)$,
	 * so $P(k)$ has the form\n
	 *     $P(k)=\b{bm}  c(k) &     &        &     & s(k) &     &        &     \\
	 *                        &  1  &        &     &      &     &        &     \\
	 *                        &     & \ldots &     &      &     &        &     \\
	 *                        &     &        &  1  &      &     &        &     \\
	 *                  -s(k) &     &        &     & c(k) &     &        &     \\
	 *                        &     &        &     &      &  1  &        &     \\
	 *                        &     &        &     &      &     & \ldots &     \\
	 *                        &     &        &     &      &     &        &  1  \e{bm}$\n
	 * where $R(k)$ appears in rows and columns $1$ and $k+1$.\n
	 * Similarly, when §pivot = 'B' (Bottom pivot), the rotation is performed for the plane
	 * $(k,z)$, giving $P(k)$ the form
	 *     $P(k)=\b{bm}  1  &        &     &       &     &        &     &      \\
	 *                      & \ldots &     &       &     &        &     &      \\
	 *                      &        &  1  &       &     &        &     &      \\
	 *                      &        &     &  c(k) &     &        &     & s(k) \\
	 *                      &        &     &       &  1  &        &     &      \\
	 *                      &        &     &       &     & \ldots &     &      \\
	 *                      &        &     &       &     &        &  1  &      \\
	 *                      &        &     & -s(k) &     &        &     & c(k) \e{bm}$\n
	 * where $R(k)$ appears in rows and columns $k$ and $z$. The rotations are performed without
	 * ever forming $P(k)$ explicitly.
	 * \param[in] side
	 *     Specifies whether the plane rotation matrix $P$ is applied to $A$ on the left or the
	 *     right.\n    = 'L': Left, compute $A = P A$\n
	 *                 = 'R': Right, compute $A = A P^T$
	 *
	 * \param[in] pivot
	 *     Specifies the plane for which $P(k)$ is a plane rotation matrix.\n
	 *         = 'V': Variable pivot, the plane $(k,k+1)$\n
	 *         = 'T': Top pivot, the plane $(1,k+1)$\n
	 *         = 'B': Bottom pivot, the plane $(k,z)$
	 *
	 * \param[in] direct
	 *     Specifies whether $P$ is a forward or backward sequence of plane rotations.\n
	 *         = 'F': Forward, $P = P(z-2) \ldots P(1) P(0)$\n
	 *         = 'B': Backward, $P = P(0) P(1) \ldots P(z-2)$
	 *
	 * \param[in] m
	 *     The number of rows of the matrix $A$. If $\{m}\le 1$, an immediate return is effected.
	 *
	 * \param[in] n
	 *     The number of columns of the matrix $A$.
	 *     If $\{n}\le 1$, an immediate return is effected.
	 *
	 * \param[in] c
	 *     an array, dimension\n ($\{m}-1$) if §side = 'L'\n
	 *                           ($\{n}-1$) if §side = 'R'\n
	 *     The cosines $c(k)$ of the plane rotations.
	 *
	 * \param[in] s
	 *     an array, dimension\n ($\{m}-1$) if §side = 'L'\n
	 *                           ($\{n}-1$) if §side = 'R'\n
	 *     The sines $s(k)$ of the plane rotations.\n
	 *     The 2 by 2 plane rotation part of the matrix $P(k)$, $R(k)$, has the form\n
	 *         $R(k)=\b{bm}  c(k) & s(k) \\
	 *                      -s(k) & c(k) \e{bm}$.
	 *
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     The §m by §n matrix $A$.\n
	 *     On exit, §A is overwritten by $P A$ if §side = 'R' or by $A P^T$ if §side = 'L'.
	 *
	 * \param[in] lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlasr(char const* const side, char const* const pivot, char const* const direct,
	                  int const m, int const n, real const* const c, real const* const s,
	                  real* const A, int const lda) /*const*/
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

	/*! §dlasrt sorts numbers in increasing or decreasing order.
	 *
	 * §dlasrt sorts the numbers in §d in increasing order (if §id = 'I') or in decreasing order
	 * (if §id = 'D').\n
	 * Use Quick Sort, reverting to Insertion sort on arrays of size$\le 20$. Dimension of 'stack'
	 * limits §n to about $2^32$.
	 * \param[in] id
	 *     = 'I': sort §d in increasing order;\n
	 *     = 'D': sort §d in decreasing order.
	 *
	 * \param[in]     n The length of the array §d.
	 * \param[in,out] d
	 *     an array, dimension (§n)\n
	 *     On entry, the array to be sorted.\n
	 *     On exit, §d has been sorted into increasing order ($\{d}[0]\le\ldots\le\{d}[\{n}-1]$) or
	 *              into decreasing order ($\{d}[0]\ge\ldots\ge\{d}[\{n}-1]$), depending on §id.
	 *
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016                                                                           */
	static void dlasrt(char const* const id, int const n, real* const d, int& info) /*const*/
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
				// Do Insertion sort on d[start:endd]
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
				// Partition d[start:endd] and stack parts, largest one first
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

	/*! §dlassq updates a sum of squares represented in scaled form.
	 *
	 * §dlassq returns the values $scl$ and $smsq$ such that\n
	 *     $scl^2 smsq = \{x}(0)^2 + \ldots + \{x}(\{n}-1)^2 + \{scale}^2\{sumsq}$,\n
	 * where $\{x}(i) = \{x}[i*\{incx}]$. The value of §sumsq is assumed to be non-negative and
	 * $scl$ returns the value\n
	 *     $scl = \max(\{scale}, |x(i)|)$.\n
	 * $scl$ and $smsq$ are overwritten on §scale and §sumsq respectively.\n
	 * The routine makes only one pass through the vector §x.
	 * \param[in] n The number of elements to be used from the vector §x.
	 * \param[in] x
	 *     an array, dimension (§n)\n
	 *     The vector for which a scaled sum of squares is computed.\n
	 *         $\{x}(i) = \{x}[i*\{incx}]$, $0 \le i < \{n}$.
	 *
	 * \param[in]     incx The increment between successive values of the vector §x. $\{incx} > 0$.
	 * \param[in,out] scale
	 *     On entry, the value §scale in the equation above.\n
	 *     On exit, §scale is overwritten with $scl$, the scaling factor for the sum of squares.
	 *
	 * \param[in,out] sumsq
	 *     On entry, the value §sumsq in the equation above.\n
	 *     On exit, §sumsq is overwritten with $smsq$,
	 *              the basic sum of squares from which $scl$ has been factored out.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dlassq(int const n, real const* const x, int const incx, real& scale, real& sumsq)
	                   /*const*/
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

	/*! §dlasv2 computes the singular value decomposition of a 2-by-2 triangular matrix.
	 *
	 * §dlasv2 computes the singular value decomposition of a 2-by-2 triangular matrix\n
	 *     $\b{bm} \{f} & \{g} \\
	 *               0  & \{h} \e{bm}$.\n
	 * On return, $|\{ssmax}|$ is the larger singular value, $|\{ssmin}|$ is the smaller singular
	 * value, and $\b{bm} \{csl} & \{snl} \e{bm}$ and $\b{bm} \{csr} & \{snr} \e{bm}$ are the left
	 * and right singular vectors for $|\{ssmax}|$, giving the decomposition\n
	 *     $\b{bm} \{csl} & \{snl} \\
	 *            -\{snl} & \{csl} \e{bm} \b{bm} \{f} & \{g} \\
	 *                                             0  & \{h} \e{bm}
	 *      \b{bm} \{csr} & -\{snr} \\
	 *             \{snr} &  \{csr} \e{bm} = \b{bm} \{ssmax} &     0    \\
	 *                                                 0     & \{ssmin} \e{bm}$.
	 * \param[in]  f     The [0,0] element of the 2 by 2 matrix.
	 * \param[in]  g     The [0,1] element of the 2 by 2 matrix.
	 * \param[in]  h     The [1,1] element of the 2 by 2 matrix.
	 * \param[out] ssmin $|\{ssmin}|$ is the smaller singular value.
	 * \param[out] ssmax $|\{ssmax}|$ is the larger singular value.
	 * \param[out] snl,
	 *             csl
	 *     The vector $\b{bm} \{csl} & \{snl} \e{bm}$ is a unit left singular vector for the
	 *     singular value $|\{ssmax}|$.
	 * \param[out] snr,
	 *             csr
	 *     The vector $\b{bm} \{csr} & \{snr} \e{bm}$ is a unit right singular vector for the
	 *     singular value $|\{ssmax}|$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Any input parameter may be aliased with any output parameter.\n
	 *     Barring over/underflow and assuming a guard digit in subtraction, all output quantities
	 *         are correct to within a few units in the last place (ulps).\n
	 *     In IEEE arithmetic, the code works correctly if one matrix element is infinite.\n
	 *     Overflow will not occur unless the largest singular value itself overflows or is within
	 *         a few ulps of overflow. (On machines with partial overflow, like the Cray, overflow
	 *         may occur if the largest singular value is within a factor of 2 of overflow.)\n
	 *     Underflow is harmless if underflow is gradual. Otherwise, results may correspond to a
	 *     matrix modified by perturbations of size near the underflow threshold.                */
	static void dlasv2(real const f, real const g, real const h, real& ssmin, real& ssmax,
	                   real& snr, real& csr, real& snl, real& csl) /*const*/
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
		tsign = ONE;
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

	/*! §dlasy2 solves the Sylvester matrix equation where the matrices are of order 1 or 2.
	 *
	 * §dlasy2 solves for the §n1 by §n2 matrix §X, $1\le\{n1}$, $\{n2}\le 2$, in\n
	 *     $\operatorname{op}(\{Tl})\{X}+\{isgn}\:\{X}\operatorname{op}(\{Tr})=\{scale}\,\{B}$,\n
	 * where §Tl is §n1 by §n1, §Tr is §n2 by §n2, §B is §n1 by §n2, and $\{isgn}=1$ or $-1$.
	 * $\operatorname{op}(T)=T$ or $T^T$, where $T^T$ denotes the transpose of $T$.
	 * \param[in] ltranl
	 *     On entry, §ltranl specifies the $\operatorname{op}(\{Tl})$:\n
	 *     = false, $\operatorname{op}(\{Tl})=\{Tl}$,\n
	 *     = true,  $\operatorname{op}(\{Tl})=\{Tl}^T$.
	 *
	 * \param[in] ltranr
	 *     On entry, §ltranr specifies the $\operatorname{op}(\{Tr})$:\n
	 *     = false, $\operatorname{op}(\{Tr})=\{Tr}$,\n
	 *     = true,  $\operatorname{op}(\{Tr})=\{Tr}^T$.
	 *
	 * \param[in] isgn
	 *     On entry, §isgn specifies the sign of the equation as described before.
	 *     §isgn may only be 1 or -1.
	 *
	 * \param[in] n1 On entry, §n1 specifies the order of matrix §Tl. §n1 may only be 0, 1 or 2.
	 * \param[in] n2 On entry, §n2 specifies the order of matrix §Tr. §n2 may only be 0, 1 or 2.
	 * \param[in] Tl   an array, dimension (§ldtl,2)\n On entry, §Tl contains an §n1 by §n1 matrix.
	 * \param[in] ldtl The leading dimension of the matrix §Tl. $\{ldtl}\ge\max(1,\{n1})$.
	 * \param[in] Tr   an array, dimension (§ldtr,2)\n On entry, §Tr contains an §n2 by §n2 matrix.
	 * \param[in] ldtr The leading dimension of the matrix §Tr. $\{ldtr}\ge\max(1,\{n2})$.
	 * \param[in] B
	 *     an array, dimension (§ldb,2)\n
	 *     On entry, the §n1 by §n2 matrix §B contains the right-hand side of the equation.
	 *
	 * \param[in]  ldb   The leading dimension of the matrix §B. $\{ldb}\ge\max(1,\{n1})$.
	 * \param[out] scale
	 *     On exit, §scale contains the scale factor.
	 *     §scale is chosen less than or equal to 1 to prevent the solution overflowing.
	 *
	 * \param[out] X
	 *     an array, dimension (§ldx,2)\n On exit, §X contains the §n1 by §n2 solution.
	 * \param[in]  ldx   The leading dimension of the matrix §X. $\{ldx}\ge\max(1,\{n1})$.
	 * \param[out] xnorm On exit, §xnorm is the infinity-norm of the solution.
	 * \param[out] info
	 *     On exit, §info is set to\n
	 *     0: successful exit.\n
	 *     1: §Tl and §Tr have too close eigenvalues, so §Tl or §Tr is perturbed to get a
	 *        nonsingular equation.\n
	 *     NOTE: In the interests of speed, this routine does not check the inputs for errors.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016                                                                           */
	static void dlasy2(bool const ltranl, bool const ltranr, int const isgn, int const n1,
	                   int const n2, real const* const Tl, int const ldtl, real const* const Tr,
	                   int const ldtr, real const* const B, int const ldb, real& scale,
	                   real* const X, int const ldx, real& xnorm, int& info) /*const*/
	{
		const bool BSWPIV[4] = {false, true, false, true};
		const bool XSWPIV[4] = {false, false, true, true};
		const int LOCL21[4] = {1, 0, 3, 2};
		const int LOCU12[4] = {2, 3, 0, 1};
		const int LOCU22[4] = {3, 2, 1, 0};
		// Do not check the input parameters for errors
		info = 0;
		// Quick return if possible
		if (n1==0 || n2==0)
		{
			return;
		}
		// Set constants to control overflow
		real eps    = dlamch("P");
		real smlnum = dlamch("S") / eps;
		real sgn    = isgn;
		int k = n1 + n1 + n2 - 2;
		real smin, temp;
		real btmp[4], tmp[4];
		if (k<4)
		{
			if (k==1)
			{
				// 1 by 1: TL11*X + sgn*X*TR11 = B11
				real tau1 = Tl[0] + sgn*Tr[0];
				real bet  = std::fabs(tau1);
				if (bet<=smlnum)
				{
					tau1 = smlnum;
					bet  = smlnum;
					info = 1;
				}
				scale = ONE;
				real gam = std::fabs(B[0]);
				if (smlnum*gam>bet)
				{
					scale = ONE / gam;
				}
				X[0] = (B[0]*scale) / tau1;
				xnorm = std::fabs(X[0]);
				return;
			}
			else
			{
				if (k==2)
				{
					// 1 by 2: TL11*[X11 X12] + isgn*[X11 X12]*op[TR11 TR12]  = [B11 B12]
					//                                           [TR21 TR22]
					smin = std::max(eps*std::max(std::max(std::fabs(Tl[0]), std::fabs(Tr[0])),
					                             std::max(std::max(std::fabs(Tr[ldtr]),
					                                               std::fabs(Tr[1])),
					                                      std::fabs(Tr[1+ldtr]))), smlnum);
					tmp[0] = Tl[0] + sgn*Tr[0];
					tmp[3] = Tl[0] + sgn*Tr[1+ldtr];
					if (ltranr)
					{
						tmp[1] = sgn*Tr[1];
						tmp[2] = sgn*Tr[ldtr];
					}
					else
					{
						tmp[1] = sgn*Tr[ldtr];
						tmp[2] = sgn*Tr[1];
					}
					btmp[0] = B[0];
					btmp[1] = B[ldb];
				}
				else // k==3
				{
					// 2 by 1: op[TL11 TL12]*[X11] + isgn* [X11]*TR11  = [B11]
					//           [TL21 TL22] [X21]         [X21]         [B21]
					smin = std::max(eps*std::max(std::max(std::fabs(Tr[0]), std::fabs(Tl[0])),
					                             std::max(std::max(std::fabs(Tl[ldtl]),
					                                               std::fabs(Tl[1])),
					                                      std::fabs(Tl[1+ldtl]))), smlnum);
					tmp[0] = Tl[0]      + sgn*Tr[0];
					tmp[3] = Tl[1+ldtl] + sgn*Tr[0];
					if (ltranl)
					{
						tmp[1] = Tl[ldtl];
						tmp[2] = Tl[1];
					}
					else
					{
						tmp[1] = Tl[1];
						tmp[2] = Tl[ldtl];
					}
					btmp[0] = B[0];
					btmp[1] = B[1];
				}
				// Solve 2 by 2 system using complete pivoting.
				// Set pivots less than smin to smin.
				int ipiv = Blas<real>::idamax(4, tmp, 1);
				real u11 = tmp[ipiv];
				if (std::fabs(u11)<=smin)
				{
					info = 1;
					u11 = smin;
				}
				real u12 = tmp[LOCU12[ipiv]];
				real l21 = tmp[LOCL21[ipiv]] / u11;
				real u22 = tmp[LOCU22[ipiv]] - u12*l21;
				if (std::fabs(u22)<=smin)
				{
					info = 1;
					u22  = smin;
				}
				if (BSWPIV[ipiv])
				{
					temp    = btmp[1];
					btmp[1] = btmp[0] - l21*temp;
					btmp[0] = temp;
				}
				else
				{
					btmp[1] -= l21 * btmp[0];
				}
				scale = ONE;
				if ((TWO*smlnum)*std::fabs(btmp[1])>std::fabs(u22)
				 || (TWO*smlnum)*std::fabs(btmp[0])>std::fabs(u11))
				{
					scale = HALF / std::max(std::fabs(btmp[0]), std::fabs(btmp[1]));
					btmp[0] *= scale;
					btmp[1] *= scale;
				}
				real x2[2];
				x2[1] = btmp[1] / u22;
				x2[0] = btmp[0] / u11 - (u12/u11)*x2[1];
				if (XSWPIV[ipiv])
				{
					temp  = x2[1];
					x2[1] = x2[0];
					x2[0] = temp;
				}
				X[0] = x2[0];
				if (n1==1)
				{
					X[ldx] = x2[1];
					xnorm  = std::fabs(X[0]) + std::fabs(X[ldx]);
				}
				else
				{
					X[1]  = x2[1];
					xnorm = std::max(std::fabs(X[0]), std::fabs(X[1]));
				}
				return;
			}
		}
		// 2 by 2: op[TL11 TL12]*[X11 X12] +isgn* [X11 X12]*op[TR11 TR12] = [B11 B12]
		//           [TL21 TL22] [X21 X22]        [X21 X22]   [TR21 TR22]   [B21 B22]
		// Solve equivalent 4 by 4 system using complete pivoting.
		// Set pivots less than smin to smin.
		smin = std::max(std::max(std::fabs(Tr[0]), std::fabs(Tr[ldtr])),
						std::max(std::fabs(Tr[1]), std::fabs(Tr[1+ldtr])));
		smin = std::max(smin, std::max(std::max(std::fabs(Tl[0]), std::fabs(Tl[ldtl])),
									   std::max(std::fabs(Tl[1]), std::fabs(Tl[1+ldtl]))));
		smin = std::max(eps*smin, smlnum);
		btmp[0] = ZERO;
		real t16[4*4];
		Blas<real>::dcopy(16, btmp, 0, t16, 1);
		t16[0]  = Tl[0]      + sgn*Tr[0];      // t16[0,0]
		t16[5]  = Tl[1+ldtl] + sgn*Tr[0];      // t16[1,1]
		t16[10] = Tl[0]      + sgn*Tr[1+ldtr]; // t16[2,2]
		t16[15] = Tl[1+ldtl] + sgn*Tr[1+ldtr]; // t16[3,3]
		if (ltranl)
		{
			t16[4]  = Tl[1];    // t16[0,1]
			t16[1]  = Tl[ldtl]; // t16[1,0]
			t16[14] = Tl[1];    // t16[2,3]
			t16[11] = Tl[ldtl]; // t16[3,2]
		}
		else
		{
			t16[4]  = Tl[ldtl]; // t16[0,1]
			t16[1]  = Tl[1];    // t16[1,0]
			t16[14] = Tl[ldtl]; // t16[2,3]
			t16[11] = Tl[1];    // t16[3,2]
		}
		if (ltranr)
		{
			t16[8]  = sgn*Tr[ldtr]; // t16[0,2]
			t16[13] = sgn*Tr[ldtr]; // t16[1,3]
			t16[2]  = sgn*Tr[1];    // t16[2,0]
			t16[7]  = sgn*Tr[1];    // t16[3,1]
		}
		else
		{
			t16[8]  = sgn*Tr[1];    // t16[0,2]
			t16[13] = sgn*Tr[1];    // t16[1,3]
			t16[2]  = sgn*Tr[ldtr]; // t16[2,0]
			t16[7]  = sgn*Tr[ldtr]; // t16[3,1]
		}
		btmp[0] = B[0];
		btmp[1] = B[1];
		btmp[2] = B[ldb];
		btmp[3] = B[1+ldb];
		// Perform elimination
		int i, ip, ipsv, j, jp, jpsv=0;
		real xmax;
		int jpiv[4];
		for (i=0; i<3; i++)
		{
			xmax = ZERO;
			for (ip=i; ip<4; ip++)
			{
				for (jp=i; jp<4; jp++)
				{
					if (std::fabs(t16[ip+4*jp])>=xmax) // t16[ip,jp]
					{
						xmax = std::fabs(t16[ip+4*jp]);
						ipsv = ip;
						jpsv = jp;
					}
				}
			}
			if (ipsv!=i)
			{
				Blas<real>::dswap(4, &t16[ipsv], 4, &t16[i], 4); // t16[ipsv,0], t16[i,0]
				temp       = btmp[i];
				btmp[i]    = btmp[ipsv];
				btmp[ipsv] = temp;
			}
			if (jpsv!=i)
			{
				Blas<real>::dswap(4, &t16[4*jpsv], 1, &t16[4*i], 1); // t16[0,jpsv], t16[0,i]
			}
			jpiv[i] = jpsv;
			if (std::fabs(t16[i+4*i])<smin) // t16[i,i]
			{
				info = 1;
				t16[i+4*i] = smin; // t16[i,i]
			}
			for (j=i+1; j<4; j++)
			{
				t16[j+4*i] /= t16[i+4*i]; // t16[j,i], t16[i,i]
				btmp[j]    -= t16[j+4*i]*btmp[i]; // t16[j,i]
				for (k=i+1; k<4; k++)
				{
					t16[j+4*k] -= t16[j+4*i] * t16[i+4*k]; // t16[j,k], t16[j,i], t16[i,k]
				}
			}
		}
		if (std::fabs(t16[15])<smin) // t16[3,3]
		{
			info = 1;
			t16[15] = smin; // t16[3,3]
		}
		scale = ONE;
		if ((EIGHT*smlnum)*std::fabs(btmp[0])>std::fabs(t16[0])   // t16[0,0]
		 || (EIGHT*smlnum)*std::fabs(btmp[1])>std::fabs(t16[5])   // t16[1,1]
		 || (EIGHT*smlnum)*std::fabs(btmp[2])>std::fabs(t16[10])  // t16[2,2]
		 || (EIGHT*smlnum)*std::fabs(btmp[3])>std::fabs(t16[15])) // t16[3,3]
		{
			scale = (ONE/EIGHT) / std::max(std::max(std::fabs(btmp[0]), std::fabs(btmp[1])),
										   std::max(std::fabs(btmp[2]), std::fabs(btmp[3])));
			btmp[0] *= scale;
			btmp[1] *= scale;
			btmp[2] *= scale;
			btmp[3] *= scale;
		}
		for (i=0; i<4; i++)
		{
			k = 3 - i;
			temp   = ONE / t16[k+4*k]; // t16[k,k]
			tmp[k] = btmp[k] * temp;
			for (j=k+1; j<4; j++)
			{
				tmp[k] -= (temp*t16[k+4*j]) * tmp[j]; // t16[k,j]
			}
		}
		for (i=0; i<3; i++)
		{
			if (jpiv[2-i]!=2-i)
			{
				temp           = tmp[2-i];
				tmp[2-i]       = tmp[jpiv[2-i]];
				tmp[jpiv[2-i]] = temp;
			}
		}
		X[0]     = tmp[0];
		X[1]     = tmp[1];
		X[ldx]   = tmp[2];
		X[1+ldx] = tmp[3];
		xnorm = std::max(std::fabs(tmp[0])+std::fabs(tmp[2]), std::fabs(tmp[1])+std::fabs(tmp[3]));
	}

	/*! §dorg2r generates all or part of the orthogonal matrix $Q$ from a QR factorization
	 *  determined by §sgeqrf (unblocked algorithm).
	 *
	 * §dorg2r generates an §m by §n real matrix $Q$ with orthonormal columns, which is defined as
	 * the first §n columns of a product of §k elementary reflectors of order §m \n
	 *     $Q = H(0) H(1) \ldots H(k-1)$\n
	 * as returned by §dgeqrf.
	 * \param[in] m The number of rows of the matrix $Q$. $\{m}\ge 0$.
	 * \param[in] n The number of columns of the matrix $Q$. $\{m}\ge\{n}\ge 0$.
	 * \param[in] k
	 *     The number of elementary reflectors whose product defines the matrix $Q$.
	 *     $\{n}\ge\{k}\ge 0$.
	 *
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the $i$-th column must contain the vector which defines the elementary
	 *     reflector $H(i)$, for $i = 0, 1, \ldots, \{k}-1$, as returned by §dgeqrf in the first
	 *     §k columns of its array argument §A. \n
	 *     On exit, the §m by §n matrix $Q$.
	 *
	 * \param[in] lda The first dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[in] tau
	 *     an array, dimension (§k)\n
	 *     $\{tau}[i]$ must contain the scalar factor of the elementary reflector $H(i)$,
	 *      as returned by §dgeqrf.
	 *
	 * \param[out] work an array, dimension (§n)
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if §info = $-i$, the $i$-th argument has an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dorg2r(int const m, int const n, int const k, real* const A, int const lda,
	                   real const* const tau, real* const work, int& info) /*const*/
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
		if (n<=0)
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
			// Set A[0:i-1,i] to zero
			for (j=0; j<i; j++)
			{
				A[j+acoli] = ZERO;
			}
		}
	}

	/*! §dorgbr
	 *
	 * §dorgbr generates one of the real orthogonal matrices $Q$ or $P^T$ determined by §dgebrd
	 * when reducing a real matrix $A$ to bidiagonal form: $A = Q B P^T$. $Q$ and $P^T$ are defined
	 * as products of elementary reflectors $H(i)$ or $G(i)$ respectively.
	 * \li If §vect = 'Q', $A$ is assumed to have been an §m by §k matrix, and $Q$ is of order
	 *     §m: \n
	 *         if $\{m}\ge\{k}$, $Q = H(0) H(1) \ldots H(\{k}-1)$ and §dorgbr returns the first §n
	 *         columns of $Q$, where $\{m}\ge\{n}\ge\{k}$;\n
	 *         if $\{m}<\{k}$, $Q = H(0) H(1) \ldots H(\{m}-2)$ and §dorgbr returns $Q$ as an
	 *         §m by §m matrix.\n
	 * \li If §vect = 'P', $A$ is assumed to have been a §k by §n matrix, and $P^T$ is of order §n:
	 *         \n if $\{k}<\{n}$, $P^T = G(\{k}-1) \ldots G(1) G(0)$ and §dorgbr returns the first
	 *         §m rows of $P^T$, where $\{n}\ge\{m}\ge\{k}$;\n
	 *         if $\{k}\ge\{n}$, $P^T = G(\{n}-2) \ldots G(1) G(0)$ and §dorgbr returns $P^T$ as an
	 *         §n by §n matrix.
	 * \param[in] vect
	 *     Specifies whether the matrix $Q$ or the matrix $P^T$ is required, as defined in the
	 *     transformation applied by §dgebrd: \n
	 *     = 'Q': generate $Q$; \n
	 *     = 'P': generate $P^T$.
	 *
	 * \param[in] m The number of rows of the matrix $Q$ or $P^T$ to be returned. $\{m}\ge 0$.
	 * \param[in] n
	 *     The number of columns of the matrix $Q$ or $P^T$ to be returned. $\{n}\ge 0$.\n
	 *     If §vect = 'Q', $\{m}\ge\{n}\ge\min(\{m},\{k})$;\n
	 *     if §vect = 'P', $\{n}\ge\{m}\ge\min(\{n},\{k})$.
	 *
	 * \param[in] k
	 *     If §vect = 'Q', the number of columns in the original §m by §k matrix reduced by
	 *         §dgebrd.\n
	 *     If §vect = 'P', the number of rows in the original §k by §n matrix reduced by §dgebrd.\n
	 *     $\{k}\ge 0$.
	 *
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the vectors which define the elementary reflectors, as returned by §dgebrd.\n
	 *     On exit, the §m by §n matrix $Q$ or $P^T$.
	 *
	 * \param[in] lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[in] tau
	 *     an array, dimension\n $\min(\{m},\{k})$ if §vect = 'Q'\n
	 *                           $\min(\{n},\{k})$ if §vect = 'P'\n
	 *     $\{tau}[i]$ must contain the scalar factor of the elementary reflector $H(i)$ or $G(i)$,
	 *     which determines $Q$ or $P^T$, as returned by §dgebrd in its array argument §tauq or
	 *     §taup.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$)\n
	 *     On exit, if $\{info}=0$, $\{work}[0]$ returns the optimal §lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work. $\{lwork}\ge\max(1,\min(\{m},\{n}))$.\n
	 *     For optimum performance $\{lwork}\ge\min(\{m},\{n})\{nb}$, where §nb is the optimal
	 *     blocksize.\n\n
	 *     If $\{lwork}=-1$, then a workspace query is assumed; the routine only calculates the
	 *     optimal size of the §work array, returns this value as the first entry of the §work
	 *     array, and no error message related to §lwork is issued by §xerbla.
	 *
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if $\{info}=-i$, the $i$-th argument had an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date April 2012                                                                          */
	static void dorgbr(char const* const vect, int const m, int const n, int const k,
	                   real* const A, int const lda, real const* const tau, real* const work,
	                   int const lwork, int& info) /*const*/
	{
		// Test the input arguments
		info = 0;
		bool wantq = std::toupper(vect[0])=='Q';
		int mn = std::min(m, n);
		bool lquery = (lwork==-1);
		if (!wantq && std::toupper(vect[0])!='P')
		{
			info = -1;
		}
		else if (m<0)
		{
			info = -2;
		}
		else if (n<0 || (wantq && (n>m || n<std::min(m, k)))
		             || (!wantq && (m>n || m<std::min(n, k))))
		{
			info = -3;
		}
		else if (k<0)
		{
			info = -4;
		}
		else if (lda<std::max(1, m))
		{
			info = -6;
		}
		else if (lwork<std::max(1, mn) && !lquery)
		{
			info = -9;
		}
		int iinfo, lwkopt;
		if (info==0)
		{
			work[0] = 1;
			if (wantq)
			{
				if (m>=k)
				{
					dorgqr(m, n, k, A, lda, tau, work, -1, iinfo);
				}
				else
				{
					if (m>1)
					{
						dorgqr(m-1, m-1, m-1, &A[1+lda], lda, tau, work, -1, iinfo);
					}
				}
			}
			else
			{
				if (k<n)
				{
					dorglq(m, n, k, A, lda, tau, work, -1, iinfo);
				}
				else
				{
					if (n>1)
					{
						dorglq(n-1, n-1, n-1, &A[1+lda], lda, tau, work, -1, iinfo);
					}
				}
			}
			lwkopt = work[0];
			lwkopt = std::max(lwkopt, mn);
		}
		if (info!=0)
		{
			xerbla("DORGBR", -info);
			return;
		}
		else if (lquery)
		{
			work[0] = lwkopt;
			return;
		}
		// Quick return if possible
		if (m==0 || n==0)
		{
			work[0] = 1;
			return;
		}
		int i, j, ldaj, ldajm;
		if (wantq)
		{
			// Form Q, determined by a call to dgebrd to reduce an m by k matrix
			if (m>=k)
			{
				// If m >= k, assume m >= n >= k
				dorgqr(m, n, k, A, lda, tau, work, lwork, iinfo);
			}
			else
			{
				// If m < k, assume m = n
				// Shift the vectors which define the elementary reflectors one column to the
				// right, and set the first row and column of Q to those of the unit matrix
				for (j=m-1; j>=1; j--)
				{
					ldaj  = lda * j;
					ldajm = ldaj - lda;
					A[ldaj] = ZERO;
					for (i=j+1; i<m; i++)
					{
						A[i+ldaj] = A[i+ldajm];
					}
				}
				A[0] = ONE;
				for (i=1; i<m; i++)
				{
					A[i] = ZERO;
				}
				if (m>1)
				{
					// Form Q(2:m,2:m)
					dorgqr(m-1, m-1, m-1, &A[1+lda], lda, tau, work, lwork, iinfo);
				}
			}
		}
		else
		{
			// Form P^T, determined by a call to dgebrd to reduce a k by n matrix
			if (k<n)
			{
				// If k < n, assume k <= m <= n
				dorglq(m, n, k, A, lda, tau, work, lwork, iinfo);
			}
			else
			{
				// If k >= n, assume m = n
				// Shift the vectors which define the elementary reflectors one row downward, and
				// set the first row and column of P^T to those of the unit matrix
				A[0] = ONE;
				for (i=1; i<n; i++)
				{
					A[i] = ZERO;
				}
				for (j=1; j<n; j++)
				{
					ldaj = lda * j;
					ldajm = ldaj - 1;
					for (i=j-1; i>=1; i--)
					{
						A[i+ldaj] = A[i+ldajm];
					}
					A[ldaj] = ZERO;
				}
				if (n>1)
				{
					// Form P^T(2:n,2:n)
					dorglq(n-1, n-1, n-1, &A[1+lda], lda, tau, work, lwork, iinfo);
				}
			}
		}
		work[0] = lwkopt;
	}

	/*! §dorghr
	 *
	 * §dorghr generates a real orthogonal matrix $Q$ which is defined as the product of
	 * $\{ihi}-\{ilo}$ elementary reflectors of order §n, as returned by §dgehrd: \n
	 *     $Q = H(\{ilo}) H(\{ilo}+1) \ldots H(\{ihi}-1)$.
	 * \param[in] n        The order of the matrix $Q$. $\{n}\ge 0$.
	 * \param[in] ilo, ihi
	 *     §ilo and §ihi must have the same values as in the previous call of §dgehrd. $Q$ is equal
	 *     to the unit matrix except in the submatrix $Q[\{ilo}+1:\{ihi},\{ilo}+1:\{ihi}]$.
	 *     $0\le\{ilo}\le\{ihi}<\{n}$, if $\{n}>0$; $\{ilo}=0$ and $\{ihi}=-1$, if $\{n}=0$.\n
	 *     NOTE: Zero-based indices!
	 *
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the vectors which define the elementary reflectors, as returned by §dgehrd.\n
	 *     On exit, the §n by §n orthogonal matrix $Q$.
	 *
	 * \param[in] lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{n})$.
	 * \param[in] tau
	 *     an array, dimension ($\{n}-1$)\n
	 *     $\{tau}[i]$ must contain the scalar factor of the elementary reflector $H(i)$, as
	 *     returned by §dgehrd.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$)\n
	 *     On exit, if $\{info}=0$, $\{work}[0]$ returns the optimal §lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work. $\{lwork}\ge\{ihi}-\{ilo}$.\n
	 *     For optimum performance $\{lwork}\ge(\{ihi}-\{ilo})\{nb}$, where §nb is the optimal
	 *     blocksize.\n
	 *     If $\{lwork}=-1$, then a workspace query is assumed; the routine only calculates the
	 *     optimal size of the §work array, returns this value as the first entry of the §work
	 *     array, and no error message related to §lwork is issued by §xerbla.
	 *
	 * \param[out] info
	 *     =0: successful exit\n
	 *     <0: if $\{info}=-i$, the $i$-th argument had an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dorghr(int const n, int const ilo, int const ihi, real* const A, int const lda,
	                   real const* const tau, real* const work, int const lwork, int& info)
	                   /*const*/
	{
		// Test the input arguments
		info = 0;
		int nh = ihi - ilo;
		bool lquery = (lwork==-1);
		if (n<0)
		{
			info = -1;
		}
		else if (ilo<0 || ilo>std::max(0, n-1))
		{
			info = -2;
		}
		else if (ihi<std::min(ilo, n-1) || ihi>=n)
		{
			info = -3;
		}
		else if (lda<std::max(1, n))
		{
			info = -5;
		}
		else if (lwork<std::max(1, nh) && !lquery)
		{
			info = -8;
		}
		int lwkopt=0, nb;
		if (info==0)
		{
			nb = ilaenv(1, "DORGQR", " ", nh, nh, nh, -1);
			lwkopt = std::max(1, nh) * nb;
			work[0] = lwkopt;
		}
		if (info!=0)
		{
			xerbla("DORGHR", -info);
			return;
		}
		else if (lquery)
		{
			return;
		}
		// Quick return if possible
		if (n==0)
		{
			work[0] = 1;
			return;
		}
		// Shift the vectors which define the elementary reflectors one column to the right, and
		// set the first ilo+1 and the last n-ihi-1 rows and columns to those of the unit matrix
		int i, j, ldaj;
		for (j=ihi; j>ilo; j--)
		{
			ldaj = lda * j;
			for (i=0; i<j; i++)
			{
				A[i+ldaj] = ZERO;
			}
			for (i=j+1; i<=ihi; i++)
			{
				A[i+ldaj] = A[i+ldaj-lda];
			}
			for (i=ihi+1; i<n; i++)
			{
				A[i+ldaj] = ZERO;
			}
		}
		for (j=0; j<=ilo; j++)
		{
			ldaj = lda * j;
			for (i=0; i<n; i++)
			{
				A[i+ldaj] = ZERO;
			}
			A[j+ldaj] = ONE;
		}
		for (j=ihi+1; j<n; j++)
		{
			ldaj = lda * j;
			for (i=0; i<n; i++)
			{
				A[i+ldaj] = ZERO;
			}
			A[j+ldaj] = ONE;
		}
		if (nh>0)
		{
			// Generate Q[ilo+1:ihi,ilo+1:ihi]
			int iinfo;
			dorgqr(nh, nh, nh, &A[ilo+1+lda*(ilo+1)], lda, &tau[ilo], work, lwork, iinfo);
		}
		work[0] = lwkopt;
	}

	/*! §dorgl2
	 *
	 * §dorgl2 generates an §m by §n real matrix $Q$ with orthonormal rows, which is defined as the
	 * first §m rows of a product of §k elementary reflectors of order §n \n
	 *     $Q = H(k-1) \ldots H(1) H(0)$\n
	 * as returned by §dgelqf.
	 * \param[in] m The number of rows of the matrix $Q$. ${m}\ge 0$.
	 * \param[in] n The number of columns of the matrix $Q$. $\{n}\ge\{m}$.
	 * \param[in] k
	 *     The number of elementary reflectors whose product defines the matrix $Q$.
	 *     $\{m}\ge\{k}\ge 0$.
	 *
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the $i$-th row must contain the vector which defines the elementary reflector
	 *     $H(i)$, for $i = 0,1,\ldots,\{k}$, as returned by §dgelqf in the first §k rows of its
	 *     array argument §A.\n
	 *     On exit, the §m by §n matrix $Q$.
	 *
	 * \param[in] lda The first dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[in] tau
	 *     an array, dimension (§k)\n
	 *     $\{tau}[i]$ must contain the scalar factor of the elementary reflector $H(i)$,
	 *     as returned by §dgelqf.
	 *
	 * \param[out] work an array, dimension (§m)
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if $\{info}=-i$, the $i$-th argument has an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dorgl2(int const m, int const n, int const k, real* const A, int const lda,
	                   real const* const tau, real* const work, int& info) /*const*/
	{
		// Test the input arguments
		info = 0;
		if (m<0)
		{
			info = -1;
		}
		else if (n<m)
		{
			info = -2;
		}
		else if (k<0 || k>m)
		{
			info = -3;
		}
		else if (lda<std::max(1, m))
		{
			info = -5;
		}
		if (info!=0)
		{
			xerbla("DORGL2", -info);
			return;
		}
		// Quick return if possible
		if (m<=0)
		{
			return;
		}
		int km = k - 1;
		int l;
		if (k<m)
		{
			// Initialise rows k:m-1 to rows of the unit matrix
			int ldaj;
			for (int j=0; j<n; j++)
			{
				ldaj = lda * j;
				for (l=k; l<m; l++)
				{
					A[l+ldaj] = ZERO;
				}
				if (j>km && j<m)
				{
					A[j+ldaj] = ONE;
				}
			}
		}
		int ildai;
		int nm = n - 1;
		int mm = m - 1;
		for (int i=km; i>=0; i--)
		{
			ildai = i + lda*i;
			// Apply H(i) to A[i:m-1,i:n-1] from the right
			if (i<nm)
			{
				if (i<mm)
				{
					A[ildai] = ONE;
					dlarf("Right", mm-i, n-i, &A[ildai], lda, tau[i], &A[1+ildai], lda, work);
				}
				Blas<real>::dscal(nm-i, -tau[i], &A[ildai+lda], lda);
			}
			A[ildai] = ONE - tau[i];
			// Set A[i,0:i-1] to zero
			for (l=0; l<i; l++)
			{
				A[i+lda*l] = ZERO;
			}
		}
	}

	/*! §dorglq
	 *
	 * §dorglq generates an §m by §n real matrix $Q$ with orthonormal rows, which is defined as the
	 * first §m rows of a product of §k elementary reflectors of order §n \n
	 *     $Q = H(\{k}-1) \ldots H(1) H(0)$\n
	 * as returned by §dgelqf.
	 * \param[in] m The number of rows of the matrix $Q$. $\{m} \ge 0$.
	 * \param[in] n The number of columns of the matrix $Q$. $\{n}\ge\{m}$.
	 * \param[in] k
	 *     The number of elementary reflectors whose product defines the matrix $Q$.
	 *     $\{m}\ge\{k}\ge 0$.\n
	 *
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the $i$-th row must contain the vector which defines the elementary reflector
	 *     $H(i)$, for $i = 0,1,\ldots,\{k}$, as returned by §dgelqf in the first §k rows of its
	 *     array argument §A.\n
	 *     On exit, the §m by §n matrix $Q$.
	 *
	 * \param[in] lda The first dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[in] tau
	 *     an array, dimension (§k)\n
	 *     $\{tau}[i]$ must contain the scalar factor of the elementary reflector $H(i)$, as
	 *     returned by §dgelqf.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$)\n
	 *     On exit, if $\{info} = 0$, $\{work}[0]$ returns the optimal §lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work. $\{lwork}\ge\max(1,\{m})$.\n
	 *     For optimum performance $\{lwork}\ge\{m}\{nb}$, where §nb is the optimal blocksize.\n
	 *     If $\{lwork}=-1$, then a workspace query is assumed; the routine only calculates the
	 *     optimal size of the §work array, returns this value as the first entry of the §work
	 *     array, and no error message related to §lwork is issued by §xerbla.
	 *
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if $\{info}=-i$, the $i$-th argument has an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dorglq(int const m, int const n, int const k, real* const A, int const lda,
	                   real const* const tau, real* const work, int const lwork, int& info)
	                   /*const*/
	{
		// Test the input arguments
		info = 0;
		int nb = ilaenv(1, "DORGLQ", " ", m, n, k, -1);
		work[0] = std::max(1, m) * nb;
		bool lquery = (lwork==-1);
		if (m<0)
		{
			info = -1;
		}
		else if (n<m)
		{
			info = -2;
		}
		else if (k<0 || k>m)
		{
			info = -3;
		}
		else if (lda<std::max(1, m))
		{
			info = -5;
		}
		else if (lwork<std::max(1, m) && !lquery)
		{
			info = -8;
		}
		if (info!=0)
		{
			xerbla("DORGLQ", -info);
			return;
		}
		else if (lquery)
		{
			return;
		}
		// Quick return if possible
		if (m<=0)
		{
			work[0] = 1;
			return;
		}
		int nbmin = 2;
		int nx = 0;
		int iws = m;
		int ldwork=0;
		if (nb>1 && nb<k)
		{
			// Determine when to cross over from blocked to unblocked code.
			nx = std::max(0, ilaenv(3, "DORGLQ", " ", m, n, k, -1));
			if (nx<k)
			{
				// Determine if workspace is large enough for blocked code.
				ldwork = m;
				iws = ldwork * nb;
				if (lwork<iws)
				{
					// Not enough workspace to use optimal nb:
					// reduce nb and determine the minimum value of nb.
					nb = lwork / ldwork;
					nbmin = std::max(2, ilaenv(2, "DORGLQ", " ", m, n, k, -1));
				}
			}
		}
		int ki, kk, i, j, ldaj;
		if (nb>=nbmin && nb<k && nx<k)
		{
			// Use blocked code after the last block.
			// The first kk rows are handled by the block method.
			ki = ((k-nx-1)/nb) * nb;
			kk = std::min(k, ki+nb);
			// Set A[kk:m-1,0:kk-1] to zero.
			for (j=0; j<kk; j++)
			{
				ldaj = lda * j;
				for (i=kk; i<m; i++)
				{
					A[i+ldaj] = ZERO;
				}
			}
		}
		else
		{
			kk = 0;
		}
		// Use unblocked code for the last or only block.
		int iinfo;
		if (kk<m)
		{
			dorgl2(m-kk, n-kk, k-kk, &A[kk+lda*kk], lda, &tau[kk], work, iinfo);
		}
		if (kk>0)
		{
			// Use blocked code
			int ib, l, ildai;
			for (i=ki; i>=0; i-=nb)
			{
				ib = std::min(nb, k-i);
				ildai = i + lda*i;
				if (i+ib<m)
				{
					// Form the triangular factor of the block reflector
					// H = H(i) H(i+1) . . . H(i+ib-1)
					dlarft("Forward", "Rowwise", n-i, ib, &A[ildai], lda, &tau[i], work, ldwork);
					// Apply H^T to A[i+ib:m-1,i:n-1] from the right
					dlarfb("Right", "Transpose", "Forward", "Rowwise", m-i-ib, n-i, ib, &A[ildai],
					       lda, work, ldwork, &A[ib+ildai], lda, &work[ib], ldwork);
				}
				// Apply H^T to columns i:n-1 of current block
				dorgl2(ib, n-i, ib, &A[ildai], lda, &tau[i], work, iinfo);
				// Set columns 0:i-1 of current block to zero
				for (j=0; j<i; j++)
				{
					ldaj = lda * j;
					for (l=i; l<i+ib; l++)
					{
						A[l+ldaj] = ZERO;
					}
				}
			}
		}
		work[0] = iws;
	}

	/*! §dorgqr
	 *
	 * §dorgqr generates an §m by §n real matrix $Q$ with orthonormal columns, which is defined as
	 * the first §n columns of a product of §k elementary reflectors of order §m \n
	 *     $Q = H(0) H(1) \ldots H(\{k}-1)$\n
	 * as returned by §dgeqrf.
	 * \param[in] m The number of rows of the matrix $Q$. $\{m}\ge 0$.
	 * \param[in] n The number of columns of the matrix $Q$. $\{m}\ge\{n}\ge 0$.
	 * \param[in] k
	 *     The number of elementary reflectors whose product defines the matrix $Q$.
	 *     $\{n}\ge\{k}\ge 0$.
	 *
	 * \param[in,out] A
	 *     an array, dimension (§lda,§n)\n
	 *     On entry, the $i$-th column must contain the vector which defines the elementary
	 *     reflector $H(i)$, for $i=0, 1, \ldots, \{k}-1$, as returned by §dgeqrf in the first §k
	 *     columns of its array argument §A.\n
	 *     On exit, the §m by §n matrix $Q$.
	 *
	 * \param[in] lda The first dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[in] tau
	 *     array, dimension (§k).\n
	 *     $\{tau}[i]$ must contain the scalar factor of the elementary reflector $H(i)$, as
	 *     returned by §dgeqrf.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$)\n
	 *     On exit, if §info = 0, §work[0] returns the optimal §lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work. $\{lwork}\ge\max(1,\{n})$.\n
	 *     For optimum performance $\{lwork}\ge\{n}\{nb}$, where §nb is the optimal blocksize.\n
	 *     If §lwork = -1, then a workspace query is assumed; the routine only calculates the
	 *     optimal size of the §work array, returns this value as the first entry of the §work
	 *     array, and no error message related to §lwork is issued by §xerbla.
	 *
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if §info = $-i$, the $i$-th argument has an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dorgqr(int const m, int const n, int const k, real* const A, int const lda,
	                   real const* const tau, real* const work, int const lwork, int& info)
	                   /*const*/
	{
		// Test the input arguments
		info = 0;
		int nb = ilaenv(1, "DORGQR", " ", m, n, k, -1);
		int lwkopt = std::max(1, n) * nb;
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
			nx = std::max(0, ilaenv(3, "DORGQR", " ", m, n, k, -1));
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
					nbmin = std::max(2, ilaenv(2, "DORGQR", " ", m, n, k, -1));
				}
			}
		}
		int i, j, kk, ki = 0, acol;
		if (nb>=nbmin && nb<k && nx<k)
		{
			// Use blocked code after the last block. The first kk columns are handled by the block method.
			ki = ((k-nx-1)/nb) * nb;
			kk = std::min(k, ki+nb);
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
				ib = std::min(nb, k-i);
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

	/*! §dorm2r multiplies a general matrix by the orthogonal matrix from a QR factorization
	 *  determined by §sgeqrf (unblocked algorithm).
	 *
	 * §dorm2r overwrites the general real §m by §n matrix $C$ with
	 *     \f{align*}{
	 *         Q   C   &\text{ if \tt{side} = 'L' and \tt{trans} = 'N', or} \\
	 *         Q^T C   &\text{ if \tt{side} = 'L' and \tt{trans} = 'T', or} \\
	 *         C   Q   &\text{ if \tt{side} = 'R' and \tt{trans} = 'N', or} \\
	 *         C   Q^T &\text{ if \tt{side} = 'R' and \tt{trans} = 'T',}
	 *     \f}
	 * where $Q$ is a real orthogonal matrix defined as the product of §k elementary reflectors\n
	 *     $Q = H(0) H(1) \ldots H(\{k}-1)$\n
	 * as returned by §dgeqrf. $Q$ is of order §m if §side = 'L' and of order §n if §side = 'R'.
	 * \param[in] side
	 *     'L': apply $Q$ or $Q^T$ from the Left\n
	 *     'R': apply $Q$ or $Q^T$ from the Right
	 *
	 * \param[in] trans
	 *     'N': apply $Q$ (No transpose)\n
	 *     'T': apply $Q^T$ (Transpose)
	 *
	 * \param[in] m The number of rows of the matrix $C$. $\{m}\ge 0$.
	 * \param[in] n The number of columns of the matrix $C$. $\{n}\ge 0$.
	 * \param[in] k
	 *     The number of elementary reflectors whose product defines the matrix $Q$.\n
	 *         If §side = 'L', $\{m}\ge\{k}\ge 0$;\n
	 *         if §side = 'R', $\{n}\ge\{k}\ge 0$.
	 *
	 * \param[in] A
	 *     an array, dimension (§lda,§k)\n
	 *     The $i$-th column must contain the vector which defines the elementary reflector $H(i)$,
	 *     for $i = 0, 1, \ldots, \{k}-1$, as returned by §dgeqrf in the first §k columns of its
	 *     array argument §A.\n §A is modified by the routine but restored on exit.
	 *
	 * \param[in] lda
	 *     The leading dimension of the array §A.\n
	 *         If §side = 'L', $\{lda}\ge\max(1,\{m})$;\n
	 *         if §side = 'R', $\{lda}\ge\max(1,\{n})$.
	 *
	 * \param[in] tau
	 *     an array, dimension (§k)\n
	 *     $\{tau}[i]$ must contain the scalar factor of the elementary reflector $H(i)$, as
	 *     returned by §dgeqrf.
	 *
	 * \param[in,out] C
	 *     an array, dimension (§ldc,§n)\n
	 *     On entry, the §m by §n matrix $C$.\n
	 *     On exit, §C is overwritten by $Q C$ or $Q^T C$ or $C Q^T$ or $C Q$.
	 *
	 * \param[in]  ldc  The leading dimension of the array §C. $\{ldc}\ge\max(1,\{m})$.
	 * \param[out] work an array, dimension\n (§n) if §side = 'L',\n (§m) if §side = 'R'
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dorm2r(char const* const side, char const* const trans, int const m, int const n,
	                   int const k, real* const A, int const lda, real const* const tau,
	                   real* const C, int const ldc, real* const work, int& info) /*const*/
	{
		// Test the input arguments
		bool left   = (std::toupper(side[0]) =='L');
		bool notran = (std::toupper(trans[0])=='N');
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
		if (!left && (std::toupper(side[0])!='R'))
		{
			info = -1;
		}
		else if (!notran && (std::toupper(trans[0])!='T'))
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
				// H(i) is applied to C[i:m-1,0:n-1]
				mi = m - i;
				ic = i;
			}
			else
			{
				// H(i) is applied to C[0:m-1,i:n-1]
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

	/*! §dormhr
	 *
	 * §dormhr overwrites the general real §m by §n matrix $C$ with\n
	 *     $\begin{tabular}{lll} & \{side} = `L' & \{side} = `R' \\
	 *           \{trans} = `N': & \(Q C\)       & \(C Q\)       \\
	 *           \{trans} = `T': & \(Q^T C\)     & \(C Q^T\)     \end{tabular}$\n
	 * where $Q$ is a real orthogonal matrix of order $n_q$, with $n_q=\{m}$ if §side = 'L' and
	 * $n_q = \{n}$ if §side = 'R'. $Q$ is defined as the product of $\{ihi}-\{ilo}$ elementary
	 * reflectors, as returned by §dgehrd: \n
	 *     $Q = H(\{ilo}) H(\{ilo}+1) \ldots H(\{ihi}-1)$.
	 * \param[in] side
	 *     = 'L': apply $Q$ or $Q^T$ from the Left;\n
	 *     = 'R': apply $Q$ or $Q^T$ from the Right.
	 *
	 * \param[in] trans
	 *     = 'N':  No transpose, apply $Q$;\n
	 *     = 'T':  Transpose, apply $Q^T$.
	 *
	 * \param[in] m        The number of rows of the matrix $C$. $\{m}\ge 0$.
	 * \param[in] n        The number of columns of the matrix $C$. $\{n}\ge 0$.
	 * \param[in] ilo, ihi
	 *     §ilo and §ihi must have the same values as in the previous call of §dgehrd. $Q$ is equal
	 *     to the unit matrix except in the submatrix $Q[\{ilo}+1:\{ihi},\{ilo}+1:\{ihi}]$.\n
	 *     If §side = 'L', then $0\le\{ilo}\le\{ihi}<\{m}$, if $\{m}>0$, and $\{ilo}=0$ and
	 *     $\{ihi}=-1$, if $\{m}=0$;\n
	 *     if §side = 'R', then $0\le\{ilo}\le\{ihi}<\{n}$, if $\{n}>0$, and $\{ilo}=0$ and
	 *     $\{ihi}=-1$, if $\{n}=0$.\n
	 *     NOTE: zero-base indices!
	 *
	 * \param[in] A
	 *     an array, dimension\n (§lda,§m) if §side = 'L'\n
	 *                           (§lda,§n) if §side = 'R'\n
	 *     The vectors which define the elementary reflectors, as returned by §dgehrd. \n
	 *     §A may be modified by the routine but is restored on exit.
	 *
	 * \param[in] lda
	 *     The leading dimension of the array §A.\n
	 *     $\{lda}\ge\max(1,\{m})$ if §side = 'L';\n $\{lda}\ge\max(1,\{n})$ if §side = 'R'.
	 *
	 * \param[in] tau
	 *     an array, dimension\n ($\{m}-1$) if §side = 'L'\n
	 *                           ($\{n}-1$) if §side = 'R'\n
	 *     $\{tau}[i]$ must contain the scalar factor of the elementary reflector $H(i)$, as
	 *     returned by §dgehrd.
	 *
	 * \param[in,out] C
	 *     an array, dimension (§ldc,§n)\n
	 *     On entry, the §m by §n matrix $C$.\n
	 *     On exit, §C is overwritten by $QC$ or $Q^TC$ or $CQ^T$ or $CQ$.
	 *
	 * \param[in]  ldc The leading dimension of the array §C. $\{ldc}\ge\max(1,\{m})$.
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$)\n
	 *     On exit, if §info = 0, $\{work}[0]$ returns the optimal §lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work.\n
	 *     If §side = 'L', $\{lwork}\ge\max(1,\{n})$;\n
	 *     if §side = 'R', $\{lwork}\ge\max(1,\{m})$.\n
	 *     For optimum performance\n $\{lwork}\ge\{n}\,\{nb}$ if §side = 'L', and\n
	 *     $\{lwork}\ge\{m}\,\{nb}$ if §side = 'R', where §nb is the optimal blocksize.\n
	 *     If $\{lwork}=-1$, then a workspace query is assumed; the routine only calculates the
	 *     optimal size of the §work array, returns this value as the first entry of the §work
	 *     array, and no error message related to §lwork is issued by §xerbla.
	 *
	 * \param[out] info
	 *     = 0:  successful exit\n
	 *     < 0:  if $\{info}=-i$, the $i$-th argument had an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dormhr(char const* const side, char const* const trans, int const m, int const n,
	                   int const ilo, int const ihi, real* const A, int const lda,
	                   real const* const tau, real* const C, int const ldc, real* const work,
	                   int const lwork, int& info) /*const*/
	{
		// Test the input arguments
		info = 0;
		int nh = ihi - ilo;
		bool left = (std::toupper(side[0])=='L');
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
		if (!left && std::toupper(side[0])!='R')
		{
			info = -1;
		}
		else if (std::toupper(trans[0])!='N' && std::toupper(trans[0])!='T')
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
		else if (ilo<0 || ilo>std::max(0, nq-1))
		{
			info = -5;
		}
		else if (ihi<std::min(ilo, nq-1) || ihi>=nq)
		{
			info = -6;
		}
		else if (lda<std::max(1, nq))
		{
			info = -8;
		}
		else if (ldc<std::max(1, m))
		{
			info = -11;
		}
		else if (lwork<std::max(1, nw) && !lquery)
		{
			info = -13;
		}
		int lwkopt=0, nb;
		if (info==0)
		{
			char concat[3];
			concat[0] = std::toupper(side[0]);
			concat[1] = std::toupper(trans[0]);
			concat[2] = '\0';
			if (left)
			{
				nb = ilaenv(1, "DORMQR", concat, nh, n, nh, -1);
			}
			else
			{
				nb = ilaenv(1, "DORMQR", concat, m, nh, nh, -1);
			}
			lwkopt = std::max(1, nw) * nb;
			work[0] = lwkopt;
		}
		if (info!=0)
		{
			xerbla("DORMHR", -info);
			return;
		}
		else if (lquery)
		{
			return;
		}
		// Quick return if possible
		if (m==0 || n==0 || nh==0)
		{
			work[0] = 1;
			return;
		}
		int i1, i2, iinfo, mi, ni;
		if (left)
		{
			mi = nh;
			ni = n;
			i1 = ilo + 1;
			i2 = 0;
		}
		else
		{
			mi = m;
			ni = nh;
			i1 = 0;
			i2 = ilo + 1;
		}
		dormqr(side, trans, mi, ni, nh, &A[ilo+1+lda*ilo], lda, &tau[ilo], &C[i1+ldc*i2], ldc,
		       work, lwork, iinfo);
		work[0] = lwkopt;
	}

	/*! §dormqr
	 *
	 * §dormqr overwrites the general real §m by §n matrix $C$ with
	 *     \f[\begin{tabular}{lll}
	 *                         &  \{side} = 'L' & \{side} = 'R' \\
	 *         \{trans} = 'N': &    \(Q C\)     &    \(C Q\)   \\
	 *         \{trans} = 'T': &   \(Q^T C\)    &   \(C Q^T\)
	 *     \end{tabular}\f]
	 * where $Q$ is a real orthogonal matrix defined as the product of §k elementary reflectors\n
	 *     $Q = H(0) H(1) \ldots H(\{k}-1)$\n
	 * as returned by §dgeqrf. $Q$ is of order §m if §side = 'L' and of order §n if §side = 'R'.
	 * \param[in] side
	 *     'L': apply $Q$ or $Q^T$ from the Left;\n
	 *     'R': apply $Q$ or $Q^T$ from the Right.
	 *
	 * \param[in] trans
	 *     'N': No transpose, apply $Q$;\n 'T': Transpose, apply $Q^T$.
	 *
	 * \param[in] m The number of rows of the matrix $C$. $\{m}\ge 0$.
	 * \param[in] n The number of columns of the matrix $C$. $\{n}\ge 0$.
	 * \param[in] k
	 *     The number of elementary reflectors whose product defines the matrix $Q$.\n
	 *         If §side = 'L', $\{m}\ge\{k}\ge 0$;\n
	 *         if §side = 'R', $\{n}\ge\{k}\ge 0$.
	 *
	 * \param[in] A
	 *     an array, dimension (§lda,§k)\n
	 *     The $i$-th column must contain the vector which defines the elementary reflector $H(i)$,
	 *     for $i=0, 1, \ldots, \{k}-1$, as returned by §dgeqrf in the first §k columns of its
	 *     array argument §A. §A may be modified by the routine but is restored on exit.
	 *
	 * \param[in] lda
	 *     The leading dimension of the array §A.\n
	 *         If §side = 'L', $\{lda}\ge\max(1,\{m})$;\n
	 *         if §side = 'R', $\{lda}\ge\max(1,\{n})$.
	 *
	 * \param[in] tau
	 *     array, dimension (§k)\n
	 *     $\{tau}[i]$ must contain the scalar factor of the elementary reflector $H(i)$,
	 *     as returned by §dgeqrf.
	 *
	 * \param[in,out] C
	 *     an array, dimension (§ldc,§n)\n
	 *     On entry, the §m by §n matrix $C$.\n
	 *     On exit, §C is overwritten by $Q C$ or $Q^T C$ or $C Q^T$ or $C Q$.
	 *
	 * \param[in]  ldc  The leading dimension of the array §C. $\{ldc}\ge\max(1,\{m})$.
	 * \param[out] work
	 *     an array, dimension ($\max(1,\{lwork})$)\n
	 *     On exit, if §info = 0, $\{work}[0]$ returns the optimal §lwork.
	 *
	 * \param[in] lwork
	 *     The dimension of the array §work.\n
	 *     If §side = 'L', $\{lwork}\ge\max(1,\{n})$;\n
	 *     if §side = 'R', $\{lwork}\ge\max(1,\{m})$.\n
	 *     For good performance, §lwork should generally be larger.\n
	 *     If §lwork = -1, then a workspace query is assumed; the routine only calculates the
	 *     optimal size of the §work array, returns this value as the first entry of the §work
	 *     array, and no error message related to §lwork is issued by §xerbla.
	 *
	 * \param[in] info
	 *     = 0: successful exit\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dormqr(char const* const side, char const* const trans, int const m, int const n,
	                   int const k, real* const A, int const lda, real const* const tau,
	                   real* const C, int const ldc, real* const work, int const lwork, int& info)
	                   /*const*/
	{
		const int NBMAX = 64;
		const int LDT   = NBMAX + 1;
		const int TSIZE = LDT*NBMAX;
		// Test the input arguments
		info = 0;
		char upside = std::toupper(side[0]);
		char uptrans = std::toupper(trans[0]);
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
		if (!left && upside!='R')
		{
			info = -1;
		}
		else if (!notran && uptrans!='T')
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
		int lwkopt=0, nb;
		if (info==0)
		{
			// Compute the workspace requirements
			nb = std::min(NBMAX, ilaenv(1, "DORMQR", opts, m, n, k, -1));
			lwkopt = std::max(1, nw)*nb + TSIZE;
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
		if (nb>1 && nb<k)
		{
			if (lwork<nw*nb+TSIZE)
			{
				nb = (lwork-TSIZE) / ldwork;
				nbmin = std::max(2, ilaenv(2, "DORMQR", opts, m, n, k, -1));
			}
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
			int iwt = nw*nb;
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
				ib = std::min(nb, k-i);
				// Form the triangular factor of the block reflector
				//     H = H(i) H(i + 1) . ..H(i + ib - 1)
				aind = i + lda*i;
				dlarft("Forward", "Columnwise", nq-i, ib, &A[aind], lda, &tau[i], &work[iwt], LDT);
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
				dlarfb(side, trans, "Forward", "Columnwise", mi, ni, ib, &A[aind], lda, &work[iwt],
				       LDT, &C[ic+ldc*jc], ldc, work, ldwork);
			}
		}
		work[0] = lwkopt;
	}

	/*! §dstebz
	 *
	 * §dstebz computes the eigenvalues of a symmetric tridiagonal matrix $T$. The user may ask for
	 * all eigenvalues, all eigenvalues in the half-open interval $]\{vl},\{vu}]$, or the §il -th
	 * through §iu -th eigenvalues.\n
	 * To avoid overflow, the matrix must be scaled so that its largest element is no greater than
	 * $\sqrt{\{overflow}\sqrt{\{underflow}}}$ in absolute value, and for greatest accuracy, it
	 * should not be much smaller than that.\n
	 * See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal Matrix", Report CS41,
	 * Computer Science Dept., Stanford University, July 21, 1966.
	 * \param[in] range
	 *     = 'A': ("All") &emsp; all eigenvalues will be found.\n
	 *     = 'V': ("Value") all eigenvalues in the half-open interval $]\{vl},\{vu}]$ will be
	 *                      found.\n
	 *     = 'I': ("Index") the §il -th through §iu -th eigenvalues (of the entire matrix) will be
	 *                      found.
	 *
	 * \param[in] order
	 *     = 'B': ("By Block") &emsp; the eigenvalues will be grouped by split-off block (see
	 *                              §iblock, §isplit) and ordered from smallest to largest within
	 *                              the block.\n
	 *     = 'E': ("Entire matrix") the eigenvalues for the entire matrix will be ordered from
	 *                              smallest to largest.
	 *
	 * \param[in] n The order of the tridiagonal matrix $T$. $\{n}\ge 0$.
	 * \param[in] vl
	 *     If §range='V', the lower bound of the interval to be searched for eigenvalues.
	 *     Eigenvalues less than or equal to §vl, or greater than §vu, will not be returned.
	 *     $\{vl}<\{vu}$.\n
	 *     Not referenced if §range = 'A' or 'I'.
	 *
	 * \param[in] vu
	 *     If §range='V', the upper bound of the interval to be searched for eigenvalues.
	 *     Eigenvalues less than or equal to §vl, or greater than §vu, will not be returned.
	 *     $\{vl}<\{vu}$.\n
	 *     Not referenced if §range = 'A' or 'I'.
	 *
	 * \param[in] il
	 *     If §range='I', the index of the smallest eigenvalue to be returned.\n
	 *     $0\le\{il}\le\{iu}<\{n}$, if $\{n}>0$; $\{il}=0$ and $\{iu}=0$ if $\{n}=0$.\n
	 *     Not referenced if §range = 'A' or 'V'.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in] iu
	 *     If §range='I', the index of the largest eigenvalue to be returned.\n
	 *     $0\le\{il}\le\{iu}<\{n}$, if $\{n}>0$; $\{il}=1$ and $\{iu}=-1$ if $\{n}=0$.\n
	 *     Not referenced if §range = 'A' or 'V'.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in] abstol
	 *     The absolute tolerance for the eigenvalues. An eigenvalue (or cluster) is considered to
	 *     be located if it has been determined to lie in an interval whose width is §abstol or
	 *     less. If §abstol is less than or equal to zero, then $\{ulp}|T|$ will be used, where
	 *     $|T|$ means the 1-norm of $T$.\n.
	 *     Eigenvalues will be computed most accurately when §abstol is set to twice the underflow
	 *     threshold $2\,\{dlamch}('S')$, not zero.
	 *
	 * \param[in] d
	 *     an array, dimension (§n)\n The §n diagonal elements of the tridiagonal matrix $T$.
	 *
	 * \param[in] e
	 *     an array, dimension ($\{n}-1$)\n
	 *     The $\{n}-1$ off-diagonal elements of the tridiagonal matrix $T$.
	 *
	 * \param[out] m
	 *     The actual number of eigenvalues found. $0\le\{m}\le\{n}$.\n
	 *     (See also the description of §info=2,3.)
	 *
	 * \param[out] nsplit The number of diagonal blocks in the matrix $T$. $1\le\{nsplit}\le\{n}$.
	 * \param[out] w
	 *     an array, dimension (§n)\n
	 *     On exit, the first §m elements of §w will contain the eigenvalues.\n
	 *     (§dstebz may use the remaining $\{n}-\{m}$ elements as workspace.)
	 *
	 * \param[out] iblock
	 *     an integer array, dimension (§n)\n
	 *     At each row/column $j$ where $\{e}[j]$ is zero or small, the matrix $T$ is considered to
	 *     split into a block diagonal matrix. On exit, if $\{info}=0$, $\{iblock}[i]$ specifies to
	 *     which block (from 0 to the number of blocks - 1) the eigenvalue $\{w}[i]$ belongs.
	 *     (§dstebz may use the remaining $\{n}-\{m}$ elements as workspace.)\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] isplit
	 *     an integer array, dimension (§n)\n
	 *     The splitting points, at which $T$ breaks up into submatrices. The first submatrix
	 *     consists of rows/columns $0$ to $\{isplit}[0]$, the second of rows/columns
	 *     $\{isplit}[0]+1$ through $\{isplit}[1]$, etc., and the $\{nsplit}-1$-th consists of
	 *     rows/columns $\{isplit}[\{nsplit}-2]+1$ through $\{isplit}[\{nsplit}-1]=\{n}-1$.\n
	 *     (Only the first §nsplit elements will actually be used, but since the user cannot know a
	 *     priori what value §nsplit will have, §n elements must be reserved for §isplit.)\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] work an array, dimension ($4\{n}$)
	 * \param[out] iwork an integer array, dimension ($3\{n}$)
	 * \param[out] info
	 *     - =0: successful exit
	 *     - <0: if $\{info}=-i$, the $i$-th argument had an illegal value
	 *     - >0: some or all of the eigenvalues failed to converge or were not computed:
	 *         - =1 or 3: Bisection failed to converge for some eigenvalues; these eigenvalues are
	 *                    flagged by a negative block number. The effect is that the eigenvalues
	 *                    may not be as accurate as the absolute and relative tolerances. This is
	 *                    generally caused by unexpectedly inaccurate arithmetic.
	 *         - =2 or 3: §range='I' only: Not all of the eigenvalues $\{il}:\{iu}$ were found.
	 *                    - Effect: $\{m}<\{iu}+1-\{il}$
	 *                    - Cause:  non-monotonic arithmetic, causing the Sturm sequence to be
	 *                              non-monotonic.
	 *                    - Cure:   recalculate, using §range='A', and pick out eigenvalues
	 *                              $\{il}:\{iu}$. In some cases, increasing the internal constant
	 *                              §FUDGE may make things work.
	 *                    .
	 *         - = 4:     §range='I', and the Gershgorin interval initially used was too small. No
	 *                    eigenvalues were computed.
	 *                    - Probable cause: your machine has sloppy floating-point arithmetic.
	 *                    - Cure: Increase the internal constant §FUDGE, recompile, and try again.
	 *
	 * \remark:
	 *     Internal Constants:\n\n
	 *         §RELFAC default = 2.0\n
	 *             The relative tolerance. An interval $]a,b]$ lies within "relative tolerance" if
	 *             $b-a < \{RELFAC}*\{ulp}*\max\left(|a|,|b|\right)$, where §ulp is the machine
	 *             precision (distance from 1 to the next larger floating point number.)\n\n
	 *         §FUDGE default = 2.1\n
	 *             A "fudge factor" to widen the Gershgorin intervals. Ideally, a value of 1 should
	 *             work, but on machines with sloppy arithmetic, this needs to be larger. The
	 *             default for publicly released versions should be large enough to handle the
	 *             worst machine around. Note that this has no effect on accuracy of the solution.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016                                                                           */
	static void dstebz(char const* const range, char const* const order, int const n,
	                   real const vl, real const vu, int const il, int const iu, real const abstol,
	                   real const* const d, real const* const e, int& m, int& nsplit,
	                   real* const w, int* const iblock, int* const isplit, real* const work,
	                   int* const iwork, int& info) /*const*/
	{
		const real FUDGE = 2.1;
		const real RELFAC = 2.0;
		info = 0;
		// Decode range
		int irange;
		char uprange = std::toupper(range[0]);
		if (uprange=='A')
		{
			irange = 1;
		}
		else if (uprange=='V')
		{
			irange = 2;
		}
		else if (uprange=='I')
		{
			irange = 3;
		}
		else
		{
			irange = 0;
		}
		// Decode order
		int iorder;
		char uporder = std::toupper(order[0]);
		if (uporder=='B')
		{
			iorder = 2;
		}
		else if (uporder=='E')
		{
			iorder = 1;
		}
		else
		{
			iorder = 0;
		}
		// Check for Errors
		if (irange<=0)
		{
			info = -1;
		}
		else if (iorder<=0)
		{
			info = -2;
		}
		else if (n<0)
		{
			info = -3;
		}
		else if (irange==2)
		{
			if (vl>=vu)
			{
				info = -5;
			}
		}
		else if (irange==3 && (il<0 || il>std::max(0, n-1)))
		{
			info = -6;
		}
		else if (irange==3 && (iu<std::min(n-1, il) || iu>=n))
		{
			info = -7;
		}
		if (info!=0)
		{
			xerbla("DSTEBZ", -info);
			return;
		}
		// Initialize error flags
		info = 0;
		bool ncnvrg = false;
		bool toofew = false;
		// Quick return if possible
		m = 0;
		if (n==0)
		{
			return;
		}
		// Simplifications:
		if (irange==3 && il==0 && iu==n-1)
		{
			irange = 1;
		}
		// Get machine constants
		// nb is the minimum vector length for vector bisection, or 0 if only scalar is to be done.
		real safemn = dlamch("S");
		real ulp = dlamch("P");
		real rtoli = ulp * RELFAC;
		int nb = ilaenv(1, "DSTEBZ", " ", n, -1, -1, -1);
		if (nb<=1)
		{
			nb = 0;
		}
		// Special Case when n=1
		if (n==1)
		{
			nsplit = 1;
			isplit[0] = 0;
			if (irange==2 && (vl>=d[0] || vu<d[0]))
			{
				m = 0;
			}
			else
			{
				w[0] = d[0];
				iblock[0] = 0;
				m = 1;
			}
			return;
		}
		// Compute Splitting Points
		nsplit = 1;
		work[n-1] = ZERO;
		real pivmin = ONE, tmp1;
		int j;
		for (j=1; j<n; j++)
		{
			tmp1 = e[j-1] * e[j-1];
			if (std::fabs(d[j]*d[j-1])*ulp*ulp+safemn > tmp1)
			{
				isplit[nsplit-1] = j - 1;
				nsplit++;
				work[j-1] = ZERO;
			}
			else
			{
				work[j-1] = tmp1;
				pivmin = std::max(pivmin, tmp1);
			}
		}
		isplit[nsplit-1] = n - 1;
		pivmin *= safemn;
		int iinfo, iout, itmax, nwl, nwu;
		real atoli, gl, gu, tmp2, tnorm, wl, wlu, wu, wul;
		// Compute Interval and atoli
		if (irange==3)
		{
			// range='I': Compute the interval containing eigenvalues il through iu.
			// Compute Gershgorin interval for entire (split) matrix and use it as the initial
			// interval
			gu = d[0];
			gl = d[0];
			tmp1 = ZERO;
			for (j=0; j<n-1; j++)
			{
				tmp2 = std::sqrt(work[j]);
				gu   = std::max(gu, d[j]+tmp1+tmp2);
				gl   = std::min(gl, d[j]-tmp1-tmp2);
				tmp1 = tmp2;
			}
			gu    = std::max(gu, d[n-1]+tmp1);
			gl    = std::min(gl, d[n-1]-tmp1);
			tnorm = std::max(std::fabs(gl), std::fabs(gu));
			gl -= FUDGE*tnorm*ulp*n - FUDGE*TWO*pivmin;
			gu += FUDGE*tnorm*ulp*n + FUDGE*pivmin;
			// Compute Iteration parameters
			itmax = int((std::log(tnorm+pivmin)-std::log(pivmin))/std::log(TWO)) + 2;
			if (abstol<=ZERO)
			{
				atoli = ulp * tnorm;
			}
			else
			{
				atoli = abstol;
			}
			work[n]   = gl;
			work[n+1] = gl;
			work[n+2] = gu;
			work[n+3] = gu;
			work[n+4] = gl;
			work[n+5] = gu;
			iwork[0] = -1;
			iwork[1] = -1;
			iwork[2] = n + 1;
			iwork[3] = n + 1;
			iwork[4] = il;
			iwork[5] = iu + 1;
			dlaebz(3, itmax, n, 2, 2, nb, atoli, rtoli, pivmin, d, e, work, &iwork[4], &work[n],
			       &work[n+4], iout, iwork, w, iblock, iinfo);
			if (iwork[5]==iu+1)
			{
				wl  = work[n];
				wlu = work[n+2];
				nwl = iwork[0];
				wu  = work[n+3];
				wul = work[n+1];
				nwu = iwork[3];
			}
			else
			{
				wl  = work[n+1];
				wlu = work[n+3];
				nwl = iwork[1];
				wu  = work[n+2];
				wul = work[n];
				nwu = iwork[2];
			}
			if (nwl<0 || nwl>=n || nwu<1 || nwu>n)
			{
				info = 4;
				return;
			}
		}
		else
		{
			// range='A' or 'V' -- Set atoli
			tnorm = std::max(std::fabs(d[0])+std::fabs(e[0]), std::fabs(d[n-1])+std::fabs(e[n-2]));
			for (j=1; j<n-1; j++)
			{
				tnorm = std::max(tnorm, std::fabs(d[j])+std::fabs(e[j-1])+std::fabs(e[j]));
			}
			if (abstol<=ZERO)
			{
				atoli = ulp * tnorm;
			}
			else
			{
				atoli = abstol;
			}
			if (irange==2)
			{
				wl = vl;
				wu = vu;
			}
			else
			{
				wl = ZERO;
				wu = ZERO;
			}
		}
		// Find Eigenvalues -- Loop Over Blocks and recompute nwl and nwu.
		// nwl accumulates the number of eigenvalues <= wl,
		// nwu accumulates the number of eigenvalues <= wu
		m = 0;
		int ibegin, iend = -1;
		info = 0;
		nwl  = 0;
		nwu  = 0;
		int ib, im, in, ioff, iwoff, je;
		real bnorm;
		for (int jb=0; jb<nsplit; jb++)
		{
			ioff = iend;
			ibegin = ioff + 1;
			iend = isplit[jb];
			in = iend - ioff;
			if (in==1)
			{
				// Special Case -- in=1
				if (irange==1 || wl>=d[ibegin]-pivmin)
				{
					nwl++;
				}
				if (irange==1 || wu>=d[ibegin]-pivmin)
				{
					nwu++;
				}
				if (irange==1 || (wl<d[ibegin]-pivmin && wu>=d[ibegin]-pivmin))
				{
					m++;
					w[m-1] = d[ibegin];
					iblock[m-1] = jb;
				}
			}
			else
			{
				// General Case -- in > 1
				// Compute Gershgorin Interval and use it as the initial interval
				gu = d[ibegin];
				gl = d[ibegin];
				tmp1 = ZERO;
				for (j=ibegin; j<iend; j++)
				{
					tmp2 = std::fabs(e[j]);
					gu = std::max(gu, d[j]+tmp1+tmp2);
					gl = std::min(gl, d[j]-tmp1-tmp2);
					tmp1 = tmp2;
				}
				gu = std::max(gu, d[iend]+tmp1);
				gl = std::min(gl, d[iend]-tmp1);
				bnorm = std::max(std::fabs(gl), std::fabs(gu));
				gl -= FUDGE*bnorm*ulp*in - FUDGE*pivmin;
				gu += FUDGE*bnorm*ulp*in + FUDGE*pivmin;
				// Compute atoli for the current submatrix
				if (abstol<=ZERO)
				{
					atoli = ulp * std::max(std::fabs(gl), std::fabs(gu));
				}
				else
				{
					atoli = abstol;
				}
				if (irange>1)
				{
					if (gu<wl)
					{
						nwl += in;
						nwu += in;
						continue;
					}
					gl = std::max(gl, wl);
					gu = std::min(gu, wu);
					if (gl>=gu)
					{
						continue;
					}
				}
				// Set Up Initial Interval
				work[n]    = gl;
				work[n+in] = gu;
				dlaebz(1, 0, in, in, 1, nb, atoli, rtoli, pivmin, &d[ibegin], &e[ibegin],
				       &work[ibegin], nullptr, &work[n], &work[n+2*in], im, iwork, &w[m],
				       &iblock[m], iinfo);
				nwl += iwork[0];
				nwu += iwork[in];
				iwoff = m - iwork[0];
				// Compute Eigenvalues
				itmax = int((std::log(gu-gl+pivmin)-std::log(pivmin))/std::log(TWO)) + 2;
				dlaebz(2, itmax, in, in, 1, nb, atoli, rtoli, pivmin, &d[ibegin], &e[ibegin],
				       &work[ibegin], nullptr, &work[n], &work[n+2*in], iout, iwork, &w[m],
				       &iblock[m], iinfo);
				// Copy eigenvalues into w and iblock
				// Use -jb-2 for block number for unconverged eigenvalues.
				for (j=0; j<iout; j++)
				{
					tmp1 = HALF * (work[j+n]+work[j+in+n]);
					// Flag non-convergence.
					if (j>=iout-iinfo)
					{
						ncnvrg = true;
						ib = -jb - 2;
					}
					else
					{
						ib = jb;
					}
					for (je=iwork[j]+iwoff; je<iwork[j+in]+iwoff; je++)
					{
						w[je] = tmp1;
						iblock[je] = ib;
					}
				}
				m += im;
			}
		}
		// If range='I', then (wl,wu) contains eigenvalues nwl+1,...,nwu
		// If nwl < il or nwu > iu+1, discard extra eigenvalues.
		if (irange==3)
		{
			im = -1;
			int idiscl = il - nwl;
			int idiscu = nwu - iu - 1;
			if (idiscl>0 || idiscu>0)
			{
				for (je=0; je<m; je++)
				{
					if (w[je]<=wlu && idiscl>0)
					{
						idiscl--;
					}
					else if (w[je]>=wul && idiscu>0)
					{
						idiscu--;
					}
					else
					{
						im++;
						w[im]      = w[je];
						iblock[im] = iblock[je];
					}
				}
				m = im + 1;
			}
			if (idiscl>0 || idiscu>0)
			{
				/* Code to deal with effects of bad arithmetic:
				 * Some low eigenvalues to be discarded are not in ]wl,wlu], or high eigenvalues to
				 * be discarded are not in ]wul,wu] so just kill off the smallest idiscl/largest
				 * idiscu eigenvalues, by simply finding the smallest/largest eigenvalue(s).
				 * (If N(w) is monotone non-decreasing, this should never happen.)               */
				int iw, jdisc;
				real wkill;
				if (idiscl>0)
				{
					wkill = wu;
					for (jdisc=0; jdisc<idiscl; jdisc++)
					{
						iw = -1;
						for (je=0; je<m; je++)
						{
							if (iblock[je]!=-1 && (w[je]<wkill || iw==-1))
							{
								iw    = je;
								wkill = w[je];
							}
						}
						iblock[iw] = -1;
					}
				}
				if (idiscu>0)
				{
					wkill = wl;
					for (jdisc=0; jdisc<idiscu; jdisc++)
					{
						iw = -1;
						for (je=0; je<m; je++)
						{
							if (iblock[je]!=-1 && (w[je]>wkill || iw==-1))
							{
								iw    = je;
								wkill = w[je];
							}
						}
						iblock[iw] = -1;
					}
				}
				im = -1;
				for (je=0; je<m; je++)
				{
					if (iblock[je]!=-1)
					{
						im++;
						w[im]      = w[je];
						iblock[im] = iblock[je];
					}
				}
				m = im + 1;
			 }
			 if (idiscl<0 || idiscu<0)
			 {
				toofew = true;
			 }
		}
		// If order='B', do nothing -- the eigenvalues are already sorted by block.
		// If order='E', sort the eigenvalues from smallest to largest
		if (iorder==1 && nsplit>1)
		{
			int ie, itmp1;
			for (je=0; je<m-1; je++)
			{
				ie = -1;
				tmp1 = w[je];
				for (j=je+1; j<m; j++)
				{
					if (w[j]<tmp1)
					{
						ie   = j;
						tmp1 = w[j];
					}
				}
				if (ie!=-1)
				{
					itmp1      = iblock[ie];
					w[ie]      = w[je];
					iblock[ie] = iblock[je];
					w[je]      = tmp1;
					iblock[je] = itmp1;
				}
			}
		}
		info = 0;
		if (ncnvrg)
		{
			info += 1;
		}
		if (toofew)
		{
			info += 2;
		}
	}

	/*! §dstein
	 *
	 * §dstein computes the eigenvectors of a real symmetric tridiagonal matrix $T$ corresponding
	 * to specified eigenvalues, using inverse iteration.\n
	 * The maximum number of iterations allowed for each eigenvector is specified by an internal
	 * parameter §MAXITS (currently set to 5).
	 * \param[in] n The order of the matrix. $\{n}\ge 0$.
	 * \param[in] d
	 *     an array, dimension (§n)\n The §n diagonal elements of the tridiagonal matrix $T$.
	 *
	 * \param[in] e
	 *     an array, dimension ($\{n}-1$)\n
	 *     The $\{n}-1$ subdiagonal elements of the tridiagonal matrix $T$, in elements 0 to
	 *     $\{n}-2$.
	 *
	 * \param[in] m The number of eigenvectors to be found. $0\le\{m}\le\{n}$.
	 * \param[in] w
	 *     an array, dimension (§n)\n
	 *     The first §m elements of §w contain the eigenvalues for which eigenvectors are to be
	 *     computed. The eigenvalues should be grouped by split-off block and ordered from smallest
	 *     to largest within the block. (The output array §w from §dstebz with §order = 'B' is
	 *     expected here.)
	 *
	 * \param[in] iblock
	 *     an integer array, dimension (§n)\n
	 *     The submatrix indices associated with the corresponding eigenvalues in §w;
	 *     $\{iblock}[i]=0$ if eigenvalue $\{w}[i]$ belongs to the first submatrix from the top,
	 *     $=1$ if $\{w}[i]$ belongs to the second submatrix, etc. (The output array §iblock from
	 *     §dstebz is expected here.)\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[in] isplit
	 *     an integer array, dimension (§n)\n
	 *     The splitting points, at which $T$ breaks up into submatrices.\n
	 *     The first submatrix consists of rows/columns 0 to $\{isplit}[0]$, the second of
	 *     rows/columns $\{isplit}[0]+1$ through $\{isplit}[1]$, etc.\n
	 *     (The output array §isplit from §dstebz is expected here.)\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] Z
	 *     an array, dimension (§ldz,§m)\n
	 *     The computed eigenvectors. The eigenvector associated with the eigenvalue $\{w}[i]$ is
	 *     stored in the $i$-th column of §Z. Any vector which fails to converge is set to its
	 *     current iterate after §MAXITS iterations.
	 *
	 * \param[in]  ldz   The leading dimension of the array §Z. $\{ldz}\ge\max(1,\{n})$.
	 * \param[out] work  an array, dimension ($5\{n}$)
	 * \param[out] iwork an integer array, dimension (§n)
	 * \param[out] ifail
	 *     an integer array, dimension (§m)\n
	 *     On normal exit, all elements of §ifail are $-1$.\n
	 *     If one or more eigenvectors fail to converge after §MAXITS iterations, then their
	 *     indices are stored in array ifail.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] info
	 *     =0: successful exit.\n
	 *     <0: if $\{info}=-i$, the $i$-th argument had an illegal value
	 *     >0: if $\{info}= i$, then $i$ eigenvectors failed to converge in §MAXITS iterations.
	 *         Their indices are stored in array §ifail.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Internal Parameters:\n
	 *         MAXITS, integer, default=5: The maximum number of iterations performed.\n
	 *         EXTRA,  integer, default=2: The number of iterations performed after norm growth
	 *                                     criterion is satisfied, should be at least 1.         */
	static void dstein(int const n, real const* const d, real const* const e, int const m,
	                   real const* const w, int const* const iblock, int const* const isplit,
	                   real* const Z, int const ldz, real* const work, int* const iwork,
	                   int* const ifail, int& info) /*const*/
	{
		const real ODM3 = real(1.0e-3);
		const real ODM1 = real(1.0e-1);
		const int MAXITS = 5;
		const int EXTRA = 2;
		int i, j;
		// Test the input parameters.µ
		info = 0;
		for (i=0; i<m; i++)
		{
			ifail[i] = -1;
		}
		if (n<0)
		{
			info = -1;
		}
		else if (m<0 || m>n)
		{
			info = -4;
		}
		else if (ldz<std::max(1, n))
		{
			info = -9;
		}
		else
		{
			for (j=1; j<m; j++)
			{
				if (iblock[j]<iblock[j-1])
				{
					info = -6;
					break;
				}
				if (iblock[j]==iblock[j-1] && w[j]<w[j-1])
				{
					info = -5;
					break;
				}
			}
		}
		if (info!=0)
		{
			xerbla("DSTEIN", -info);
			return;
		}
		// Quick return if possible
		if (n==0 || m==0)
		{
			return;
		}
		else if (n==1)
		{
			Z[0] = ONE;
			return;
		}
		// Get machine constants.
		real eps = dlamch("Precision");
		// Initialize seed for random number generator DLARNV.
		int iseed[4];
		for (i=0; i<4; i++)
		{
			iseed[i] = 1;
		}
		// Initialize pointers.
		int indrv1 = 0;
		int indrv2 = indrv1 + n;
		int indrv3 = indrv2 + n;
		int indrv4 = indrv3 + n;
		int indrv5 = indrv4 + n;
		// Compute eigenvectors of matrix blocks.
		int j1 = 0;
		int b1, blksiz, bn, gpind=-1, iinfo, its, jblk, jmax, nblk, nrmchk, zind;
		real dtpcrt, eps1, nrm, onenrm, ortol, pertol, scl, sep, tol, xj, xjm=ZERO, ztr;
		for (nblk=0; nblk<=iblock[m-1]; nblk++)
		{
			// Find starting and ending indices of block nblk.
			if (nblk==0)
			{
				b1 = 0;
			}
			else
			{
				b1 = isplit[nblk-1] + 1;
			}
			bn = isplit[nblk];
			blksiz = bn - b1 + 1;
			if (blksiz!=1)
			{
				gpind = j1;
				// Compute reorthogonalization criterion and stopping criterion.
				onenrm = std::fabs(d[b1]) + std::fabs(e[b1]);
				onenrm = std::max(onenrm, std::fabs(d[bn])+std::fabs(e[bn-1]));
				for (i=b1+1; i<bn; i++)
				{
					onenrm = std::max(onenrm, std::fabs(d[i])+std::fabs(e[i-1])+std::fabs(e[i]));
				}
				ortol = ODM3 * onenrm;
				dtpcrt = std::sqrt(ODM1/blksiz);
			}
			// Loop through eigenvalues of block nblk.
			jblk = -1;
			for (j=j1; j<m; j++)
			{
				if (iblock[j]!=nblk)
				{
					j1 = j;
					break;
				}
				jblk++;
				xj = w[j];
				// Skip all the work if the block size is one.
				if (blksiz==1)
				{
					work[indrv1] = ONE;
				}
				else
				{
					// If eigenvalues j and j-1 are too close, add a relatively small perturbation.
					if (jblk>0)
					{
						eps1 = std::fabs(eps*xj);
						pertol = TEN * eps1;
						sep = xj - xjm;
						if (sep<pertol)
						{
							xj = xjm + pertol;
						}
					}
					its = 0;
					nrmchk = 0;
					// Get random starting vector.
					dlarnv(2, iseed, blksiz, &work[indrv1]);
					// Copy the matrix T so it won't be destroyed in factorization.
					Blas<real>::dcopy(blksiz,   &d[b1], 1, &work[indrv4],   1);
					Blas<real>::dcopy(blksiz-1, &e[b1], 1, &work[indrv2+1], 1);
					Blas<real>::dcopy(blksiz-1, &e[b1], 1, &work[indrv3],   1);
					// Compute LU factors with partial pivoting  (PT = LU)
					tol = ZERO;
					dlagtf(blksiz, &work[indrv4], xj, &work[indrv2+1], &work[indrv3], tol,
					       &work[indrv5], iwork, iinfo);
					// Update iteration count.
					do
					{
						its++;
						if (its>MAXITS)
						{
							// If stopping criterion was not satisfied, update info and store
							// eigenvector number in array ifail.
							info++;
							ifail[info-1] = j;
							break;
						}
						// Normalize and scale the righthand side vector Pb.
						jmax = Blas<real>::idamax(blksiz, &work[indrv1], 1);
						scl = blksiz * onenrm * std::max(eps, std::fabs(work[indrv4+blksiz-1]))
							  / std::fabs(work[indrv1+jmax]);
						Blas<real>::dscal(blksiz, scl, &work[indrv1], 1);
						// Solve the system LU = Pb.
						dlagts(-1, blksiz, &work[indrv4], &work[indrv2+1], &work[indrv3],
							   &work[indrv5], iwork, &work[indrv1], tol, iinfo);
						//Reorthogonalize by modified Gram-Schmidt if eigenvalues are close enough.
						if (jblk!=0)
						{
							if (std::fabs(xj-xjm)>ortol)
							{
								gpind = j;
							}
							if (gpind!=j)
							{
								for (i=gpind; i<j; i++)
								{
									zind = b1 + ldz*i;
									ztr = -Blas<real>::ddot(blksiz, &work[indrv1], 1, &Z[zind], 1);
									Blas<real>::daxpy(blksiz, ztr, &Z[zind], 1, &work[indrv1], 1);
								}
							}
						}
						// Check the infinity norm of the iterate.
						jmax = Blas<real>::idamax(blksiz, &work[indrv1], 1);
						nrm = std::fabs(work[indrv1+jmax]);
						//Continue for additional iterations after norm reaches stopping criterion.
						if (nrm<dtpcrt)
						{
							continue;
						}
						nrmchk++;
					} while (nrmchk<EXTRA+1);
					// Accept iterate as jth eigenvector.
					scl = ONE / Blas<real>::dnrm2(blksiz, &work[indrv1], 1);
					jmax = Blas<real>::idamax(blksiz, &work[indrv1], 1);
					if (work[indrv1+jmax]<ZERO)
					{
						scl = -scl;
					}
					Blas<real>::dscal(blksiz, scl, &work[indrv1], 1);
				}
				zind = ldz * j;
				for (i=0; i<n; i++)
				{
					Z[i+zind] = ZERO;
				}
				zind += b1;
				for (i=0; i<blksiz; i++)
				{
					Z[i+zind] = work[indrv1+i];
				}
				// Save the shift to check eigenvalue spacing at next iteration.
				xjm = xj;
			}
		}
	}

	/*! §dsteqr
	 *
	 * §dsteqr computes all eigenvalues and, optionally, eigenvectors of a symmetric tridiagonal
	 * matrix using the implicit QL or QR method. The eigenvectors of a full or band symmetric
	 * matrix can also be found if §dsytrd or §dsptrd or §dsbtrd has been used to reduce this
	 * matrix to tridiagonal form.
	 * \param[in] compz
	 *     ='N': Compute eigenvalues only.\n
	 *     ='V': Compute eigenvalues and eigenvectors of the original symmetric matrix. On entry,
	 *           $Z$ must contain the orthogonal matrix used to reduce the original matrix to
	 *           tridiagonal form.
	 *     ='I': Compute eigenvalues and eigenvectors of the tridiagonal matrix. $Z$ is initialized
	 *           to the identity matrix.
	 *
	 * \param[in]     n The order of the matrix. $\{n}\ge 0$.
	 * \param[in,out] d
	 *     an array, dimension (§n)\n
	 *     On entry, the diagonal elements of the tridiagonal matrix.\n
	 *     On exit, if $\{info}=0$, the eigenvalues in ascending order.
	 *
	 * \param[in,out] e
	 *     an array, dimension ($\{n}-1$)\n
	 *     On entry, the $\{n}-1$ subdiagonal elements of the tridiagonal matrix.\n
	 *     On exit, §e has been destroyed.
	 *
	 * \param[in,out] Z
	 *     an array, dimension (§ldz,§n)\n
	 *     On entry, if §compz ='V', then §Z contains the orthogonal matrix used in the reduction
	 *     to tridiagonal form.\n
	 *     On exit, if $\{info}=0$, then if §compz ='V', §Z contains the orthonormal eigenvectors
	 *     of the original symmetric matrix, and if §compz ='I', §Z contains the orthonormal
	 *     eigenvectors of the symmetric tridiagonal matrix.
	 *     If §compz ='N', then §Z is not referenced.
	 *
	 * \param[in] ldz
	 *     The leading dimension of the array §Z. $\{ldz}\ge 1$, and if eigenvectors are desired,
	 *     then $\{ldz}\ge\max(1,\{n})$.
	 *
	 * \param[out] work
	 *     an array, dimension ($\max(1,2\{n}-2)$)\n
	 *     If §compz ='N', then §work is not referenced.
	 *
	 * \param[out] info
	 *     =0: successful exit\n
	 *     <0: if $\{info}=-i$, the $i$-th argument had an illegal value.\n
	 *     >0: the algorithm has failed to find all the eigenvalues in a total of $30\{n}$
	 *         iterations; if $\{info}=i$, then $i$ elements of §e have not converged to zero;
	 *         on exit, §d and §e contain the elements of a symmetric tridiagonal matrix which is
	 *         orthogonally similar to the original matrix.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dsteqr(char const* const compz, int const n, real* const d, real* const e,
	                   real* const Z, int const ldz, real* const work, int& info) /*const*/
	{
		int const MAXIT = 30;
		// Test the input parameters.
		info = 0;
		int icompz;
		if (std::toupper(compz[0])=='N')
		{
			icompz = 0;
		}
		else if (std::toupper(compz[0])=='V')
		{
			icompz = 1;
		}
		else if (std::toupper(compz[0])=='I')
		{
			icompz = 2;
		}
		else
		{
			icompz = -1;
		}
		if (icompz<0)
		{
			info = -1;
		}
		else if (n<0)
		{
			info = -2;
		}
		else if (ldz<1 || (icompz>0 && ldz<std::max(1, n)))
		{
			info = -6;
		}
		if (info!=0)
		{
			xerbla("DSTEQR", -info);
			return;
		}
		// Quick return if possible
		if (n==0)
		{
			return;
		}
		if (n==1)
		{
			if (icompz==2)
			{
				Z[0] = ONE;
			}
			return;
		}
		// Determine the unit roundoff and over/underflow thresholds.
		real eps = dlamch("E");
		real eps2 = eps * eps;
		real safmin = dlamch("S");
		real safmax = ONE / safmin;
		real ssfmax = std::sqrt(safmax) / THREE;
		real ssfmin = std::sqrt(safmin) / eps2;
		// Compute the eigenvalues and eigenvectors of the tridiagonal matrix.
		if (icompz==2)
		{
			dlaset("Full", n, n, ZERO, ONE, Z, ldz);
		}
		int nmaxit = n * MAXIT;
		int jtot = 0;
		// Determine where the matrix splits and choose QL or QR iteration for each block,
		// according to whether top or bottom diagonal element is smaller.
		int l1 = 0;
		int nm1 = n - 1;
		int i, ii, iscale, j, k, l, lend, lendp1, lendsv, lsv, m, mm1;
		real anorm, b, c, f, g, p, r, rt1, rt2, s, tst;
		do
		{
			if (l1>=n)
			{
				// Order eigenvalues and eigenvectors.
				if (icompz==0)
				{
					// Use Quick Sort
					dlasrt("I", n, d, info);
				}
				else
				{
					// Use Selection Sort to minimize swaps of eigenvectors
					for (ii=1; ii<n; ii++)
					{
						i = ii - 1;
						k = i;
						p = d[i];
						for (j=ii; j<n; j++)
						{
							if (d[j]<p)
							{
								k = j;
								p = d[j];
							}
						}
						if (k!=i)
						{
							d[k] = d[i];
							d[i] = p;
							Blas<real>::dswap(n, &Z[ldz*i], 1, &Z[ldz*k], 1);
						}
					}
				}
				return;
			}
			if (l1>0)
			{
				e[l1-1] = ZERO;
			}
			if (l1<nm1)
			{
				for (m=l1; m<nm1; m++)
				{
					tst = std::fabs(e[m]);
					if (tst==ZERO)
					{
						break;
					}
					if (tst<=(std::sqrt(std::fabs(d[m]))*std::sqrt(std::fabs(d[m+1])))*eps)
					{
						e[m] = ZERO;
						break;
					}
				}
			}
			else
			{
				m = nm1;
			}
			l      = l1;
			lsv    = l;
			lend   = m;
			lendsv = lend;
			l1 = m + 1;
			if (lend==l)
			{
				continue;
			}
			// Scale submatrix in rows and columns l to lend
			anorm = dlanst("M", lend-l+1, &d[l], &e[l]);
			iscale = 0;
			if (anorm==ZERO)
			{
				continue;
			}
			if (anorm>ssfmax)
			{
				iscale = 1;
				dlascl("G", 0, 0, anorm, ssfmax, lend-l+1, 1, &d[l], n, info);
				dlascl("G", 0, 0, anorm, ssfmax, lend-l,   1, &e[l], n, info);
			}
			else if (anorm<ssfmin)
			{
				iscale = 2;
				dlascl("G", 0, 0, anorm, ssfmin, lend-l+1, 1, &d[l], n, info);
				dlascl("G", 0, 0, anorm, ssfmin, lend-l,   1, &e[l], n, info);
			}
			// Choose between QL and QR iteration
			if (std::fabs(d[lend])<std::fabs(d[l]))
			{
				lend = lsv;
				l = lendsv;
			}
			if (lend>l)
			{
				// QL Iteration
				// Look for small subdiagonal element.
				while (true)
				{
					if (l!=lend)
					{
						for (m=l; m<lend; m++)
						{
							tst = std::fabs(e[m]);
							if (tst*tst <= (eps2*std::fabs(d[m]))*std::fabs(d[m+1])+safmin)
							{
								break;
							}
						}
					}
					else
					{
						m = lend;
					}
					if (m<lend)
					{
						e[m] = ZERO;
					}
					p = d[l];
					if (m==l)
					{
						// Eigenvalue found.
						d[l] = p;
						l++;
						if (l<=lend)
						{
							continue;
						}
						break;
					}
					// If remaining matrix is 2 by 2, use dlae2 or dlaev2 to compute its
					// eigensystem.
					if (m==l+1)
					{
						if (icompz>0)
						{
							dlaev2(d[l], e[l], d[l+1], rt1, rt2, c, s);
							work[l] = c;
							work[nm1+l] = s;
							dlasr("R", "V", "B", n, 2, &work[l], &work[nm1+l], &Z[ldz*l], ldz);
						}
						else
						{
							dlae2(d[l], e[l], d[l+1], rt1, rt2);
						}
						d[l] = rt1;
						d[l+1] = rt2;
						e[l] = ZERO;
						l += 2;
						if (l<=lend)
						{
							continue;
						}
						break;
					}
					if (jtot==nmaxit)
					{
						break;
					}
					jtot++;
					// Form shift.
					g = (d[l+1]-p) / (TWO*e[l]);
					r = dlapy2(g, ONE);
					g = d[m] - p + (e[l]/(g+std::copysign(r, g)));
					s = ONE;
					c = ONE;
					p = ZERO;
					// Inner loop
					for (i=m-1; i>=l; i--)
					{
						f = s * e[i];
						b = c * e[i];
						dlartg(g, f, c, s, r);
						if (i!=m-1)
						{
							e[i+1] = r;
						}
						g = d[i+1] - p;
						r = (d[i]-g)*s + TWO*c*b;
						p = s * r;
						d[i+1] = g + p;
						g = c*r - b;
						// If eigenvectors are desired, then save rotations.
						if (icompz>0)
						{
							work[i] = c;
							work[nm1+i] = -s;
						}
					}
					// If eigenvectors are desired, then apply saved rotations.
					if (icompz>0)
					{
						dlasr("R", "V", "B", n, m-l+1, &work[l], &work[nm1+l], &Z[ldz*l], ldz);
					}
					d[l] -= p;
					e[l] = g;
				}
			}
			else
			{
				// QR Iteration
				// Look for small superdiagonal element.
				while (true)
				{
					if (l!=lend)
					{
						lendp1 = lend + 1;
						for (m=l; m>=lendp1; m--)
						{
							mm1 = m - 1;
							tst = std::fabs(e[mm1]);
							if (tst*tst <= (eps2*std::fabs(d[m]))*std::fabs(d[mm1])+safmin)
							{
								break;
							}
						}
					}
					else
					{
						m = lend;
					}
					if (m>lend)
					{
						e[m-1] = ZERO;
					}
					p = d[l];
					if (m==l)
					{
						// Eigenvalue found.
						d[l] = p;
						l--;
						if (l>=lend)
						{
							continue;
						}
						break;
					}
					// If remaining matrix is 2 by 2, use dlae2 or dlaev2 to compute its
					// eigensystem.
					if (m==l-1)
					{
						if (icompz>0)
						{
							dlaev2(d[l-1], e[l-1], d[l], rt1, rt2, c, s);
							work[m] = c;
							work[nm1+m] = s;
							dlasr("R", "V", "F", n, 2, &work[m], &work[nm1+m], &Z[ldz*(l-1)], ldz);
						}
						else
						{
							dlae2(d[l-1], e[l-1], d[l], rt1, rt2);
						}
						d[l-1] = rt1;
						d[l] = rt2;
						e[l-1] = ZERO;
						l -= 2;
						if (l>=lend)
						{
							continue;
						}
						break;
					}
					if (jtot==nmaxit)
					{
						break;
					}
					jtot++;
					// Form shift.
					g = (d[l-1]-p) / (TWO*e[l-1]);
					r = dlapy2(g, ONE);
					g = d[m] - p + (e[l-1]/(g+std::copysign(r, g)));
					s = ONE;
					c = ONE;
					p = ZERO;
					// Inner loop
					for (i=m; i<l; i++)
					{
						f = s * e[i];
						b = c * e[i];
						dlartg(g, f, c, s, r);
						if (i!=m)
						{
							e[i-1] = r;
						}
						g = d[i] - p;
						r = (d[i+1]-g)*s + TWO*c*b;
						p = s * r;
						d[i] = g + p;
						g = c*r - b;
						// If eigenvectors are desired, then save rotations.
						if (icompz>0)
						{
							work[i] = c;
							work[nm1+i] = s;
						}
					}
					// If eigenvectors are desired, then apply saved rotations.
					if (icompz>0)
					{
						dlasr("R", "V", "F", n, l-m+1, &work[m], &work[nm1+m], &Z[ldz*m], ldz);
					}
					d[l] -= p;
					e[l-1] = g;
				}
			}
			// Undo scaling if necessary
			if (iscale==1)
			{
				dlascl("G", 0, 0, ssfmax, anorm, lendsv-lsv+1, 1, &d[lsv], n, info);
				dlascl("G", 0, 0, ssfmax, anorm, lendsv-lsv,   1, &e[lsv], n, info);
			}
			else if (iscale==2)
			{
				dlascl("G", 0, 0, ssfmin, anorm, lendsv-lsv+1, 1, &d[lsv], n, info);
				dlascl("G", 0, 0, ssfmin, anorm, lendsv-lsv,   1, &e[lsv], n, info);
			}
			// Check for no convergence to an eigenvalue after a total of n*MAXIT iterations.
		} while (jtot<nmaxit);
		for (i=0; i<nm1; i++)
		{
			if (e[i]!=ZERO)
			{
				info++;
			}
		}
	}

	/*! §dsterf
	 *
	 * §dsterf computes all eigenvalues of a symmetric tridiagonal matrix using the
	 * Pal-Walker-Kahan variant of the QL or QR algorithm.
	 * \param[in]     n The order of the matrix. $\{n}\ge 0$.
	 * \param[in,out] d
	 *     an array, dimension (§n)\n
	 *     On entry, the §n diagonal elements of the tridiagonal matrix.\n
	 *     On exit, if $\{info}=0$, the eigenvalues in ascending order.
	 *
	 * \param[in,out] e
	 *     an array, dimension ($\{n}-1$)\n
	 *     On entry, the ($\{n}-1$) subdiagonal elements of the tridiagonal matrix.\n
	 *     On exit, §e has been destroyed.
	 *
	 * \param[out] info
	 *     =0: successful exit\n
	 *     <0: if $\{info}=-i$, the $i$-th argument had an illegal value\n
	 *     >0: the algorithm failed to find all of the eigenvalues in a total of $30\{n}$
	 *         iterations; if $\{info}=i$, then $i$ elements of §e have not converged to zero.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dsterf(int const n, real* const d, real* const e, int& info) /*const*/
	{
		int const MAXIT = 30;
		// Test the input parameters.
		info = 0;
		// Quick return if possible
		if (n<0)
		{
			info = -1;
			xerbla("DSTERF", -info);
			return;
		}
		if (n<=1)
		{
			return;
		}
		// Determine the unit roundoff for this environment.
		real eps    = dlamch("E");
		real eps2   = eps * eps;
		real safmin = dlamch("S");
		real safmax = ONE / safmin;
		real ssfmax = std::sqrt(safmax) / THREE;
		real ssfmin = std::sqrt(safmin) / eps2;
		// Compute the eigenvalues of the tridiagonal matrix.
		int nmaxit = n * MAXIT;
		real sigma = ZERO;
		int jtot   = 0;
		/* Determine where the matrix splits and choose QL or QR iteration for each block,
		 * according to whether top or bottom diagonal element is smaller.                       */
		int l1 = 0;
		int i, iscale, l, lend, lendsv, lsv, m;
		real alpha, anorm, bb, c, gamma, oldc, oldgam, p, r, rt1, rt2, rte, s;
		do
		{
			if (l1>=n)
			{
				// Sort eigenvalues in increasing order.
				dlasrt("I", n, d, info);
				return;
			}
			if (l1>0)
			{
				e[l1-1] = ZERO;
			}
			for (m=l1; m<n-1; m++)
			{
				if (std::fabs(e[m])
					<= (std::sqrt(std::fabs(d[m]))*std::sqrt(std::fabs(d[m+1]))) * eps)
				{
					e[m] = ZERO;
					break;
				}
			}
			l      = l1;
			lsv    = l;
			lend   = m;
			lendsv = lend;
			l1     = m + 1;
			if (lend==l)
			{
				continue;
			}
			// Scale submatrix in rows and columns l to lend
			anorm = dlanst("M", lend-l+1, &d[l], &e[l]);
			iscale = 0;
			if (anorm==ZERO)
			{
				continue;
			}
			if (anorm>ssfmax)
			{
				iscale = 1;
				dlascl("G", 0, 0, anorm, ssfmax, lend-l+1, 1, &d[l], n, info);
				dlascl("G", 0, 0, anorm, ssfmax, lend-l,   1, &e[l], n, info);
			}
			else if (anorm<ssfmin)
			{
				iscale = 2;
				dlascl("G", 0, 0, anorm, ssfmin, lend-l+1, 1, &d[l], n, info);
				dlascl("G", 0, 0, anorm, ssfmin, lend-l,   1, &e[l], n, info);
			}
			for (i=l; i<lend; i++)
			{
				e[i] *= e[i];
			}
			// Choose between QL and QR iteration
			if (std::fabs(d[lend]) < std::fabs(d[l]))
			{
				lend = lsv;
				l    = lendsv;
			}
			if (lend>=l)
			{
				// QL Iteration
				// Look for small subdiagonal element.
				while (true)
				{
					if (l!=lend)
					{
						for (m=l; m<lend; m++)
						{
							if (std::fabs(e[m]) <= eps2*std::fabs(d[m]*d[m+1]))
							{
								break;
							}
						}
					}
					else
					{
						m = lend;
					}
					if (m<lend)
					{
						e[m] = ZERO;
					}
					p = d[l];
					if (m==l)
					{
						// Eigenvalue found.
						d[l] = p;
						l++;
						if (l<=lend)
						{
							continue;
						}
						break;
					}
					// If remaining matrix is 2 by 2, use dlae2 to compute its eigenvalues.
					if (m==l+1)
					{
						rte = std::sqrt(e[l]);
						dlae2(d[l], rte, d[l+1], rt1, rt2);
						d[l]   = rt1;
						d[l+1] = rt2;
						e[l]   = ZERO;
						l += 2;
						if (l<=lend)
						{
							continue;
						}
						break;
					}
					if (jtot==nmaxit)
					{
						break;
					}
					jtot++;
					// Form shift.
					rte   = std::sqrt(e[l]);
					sigma = (d[l+1]-p) / (TWO*rte);
					r     = dlapy2(sigma, ONE);
					sigma = p - (rte/(sigma+std::copysign(r, sigma)));
					c = ONE;
					s = ZERO;
					gamma = d[m] - sigma;
					p     = gamma * gamma;
					// Inner loop
					for (i=m-1; i>=l; i--)
					{
						bb = e[i];
						r  = p + bb;
						if (i!=m-1)
						{
							e[i+1] = s * r;
						}
						oldc = c;
						c    = p / r;
						s    = bb / r;
						oldgam = gamma;
						alpha  = d[i];
						gamma  = c*(alpha-sigma) - s*oldgam;
						d[i+1] = oldgam + (alpha-gamma);
						if (c!=ZERO)
						{
							p = (gamma*gamma) / c;
						}
						else
						{
							p = oldc * bb;
						}
					}
					e[l] = s * p;
					d[l] = sigma + gamma;
				}
			}
			else
			{
				// QR Iteration
				// Look for small superdiagonal element.
				while (true)
				{
					for (m=l; m>lend; m--)
					{
						if (std::fabs(e[m-1]) <= eps2*std::fabs(d[m]*d[m-1]))
						{
							break;
						}
					}
					if (m>lend)
					{
						e[m-1] = ZERO;
					}
					p = d[l];
					if (m==l)
					{
						// Eigenvalue found.
						d[l] = p;
						l--;
						if (l>=lend)
						{
							continue;
						}
						break;
					}
					// If remaining matrix is 2 by 2, use dlae2 to compute its eigenvalues.
					if (m==l-1)
					{
						rte = std::sqrt(e[l-1]);
						dlae2(d[l], rte, d[l-1], rt1, rt2);
						d[l]   = rt1;
						d[l-1] = rt2;
						e[l-1] = ZERO;
						l -= 2;
						if (l>=lend)
						{
							continue;
						}
						break;
					}
					if (jtot==nmaxit)
					{
						break;
					}
					jtot++;
					// Form shift.
					rte   = std::sqrt(e[l-1]);
					sigma = (d[l-1]-p) / (TWO*rte);
					r     = dlapy2(sigma, ONE);
					sigma = p - (rte/(sigma+std::copysign(r, sigma)));
					c = ONE;
					s = ZERO;
					gamma = d[m] - sigma;
					p     = gamma * gamma;
					// Inner loop
					for (i=m; i<l; i++)
					{
						bb = e[i];
						r = p + bb;
						if (i!=m)
						{
							e[i-1] = s * r;
						}
						oldc = c;
						c    = p / r;
						s    = bb / r;
						oldgam = gamma;
						alpha  = d[i+1];
						gamma  = c*(alpha-sigma) - s*oldgam;
						d[i]   = oldgam + (alpha-gamma);
						if (c!=ZERO)
						{
							p = (gamma*gamma) / c;
						}
						else
						{
							p = oldc * bb;
						}
					}
					e[l-1] = s * p;
					d[l]   = sigma + gamma;
				}
			}
			// Undo scaling if necessary
			if (iscale==1)
			{
				dlascl("G", 0, 0, ssfmax, anorm, lendsv-lsv+1, 1, &d[lsv], n, info);
			}
			if (iscale==2)
			{
				dlascl("G", 0, 0, ssfmin, anorm, lendsv-lsv+1, 1, &d[lsv], n, info);
			}
			// Check for no convergence to an eigenvalue after a total of n * MAXIT iterations.
		} while (jtot<nmaxit);
		for (i=0; i<n-1; i++)
		{
			if (e[i]!=ZERO)
			{
				info++;
			}
		}
	}

	/*! §dstevx computes the eigenvalues and, optionally, the left and/or right eigenvectors for
	 *  simmetric tridiagonal matrices.
	 *
	 * §dstevx computes selected eigenvalues and, optionally, eigenvectors of a real symmetric
	 * tridiagonal matrix $A$. Eigenvalues and eigenvectors can be selected by specifying either a
	 * range of values or a range of indices for the desired eigenvalues.
	 * \param[in] jobz
	 *     ='N': Compute eigenvalues only;\n
	 *     ='V': Compute eigenvalues and eigenvectors.
	 *
	 * \param[in] range
	 *     ='A': all eigenvalues will be found.\n
	 *     ='V': all eigenvalues in the half-open interval $]\{vl},\{vu}]$ will be found.\n
	 *     ='I': the §il -th through §iu -th eigenvalues will be found.
	 *
	 * \param[in]     n The order of the matrix. $\{n}\ge 0$.
	 * \param[in,out] d
	 *     an array, dimension (§n)\n
	 *     On entry, the §n diagonal elements of the tridiagonal matrix $A$.\n
	 *     On exit, §d may be multiplied by a constant factor chosen to avoid over/underflow in
	 *     computing the eigenvalues.
	 *
	 * \param[in,out] e
	 *     an array, dimension ($\max(1,\{n}-1)$)\n
	 *     On entry, the $\{n}-1$ subdiagonal elements of the tridiagonal matrix $A$ in elements
	 *     0 to $\{n}-2$ of §e. \n
	 *     On exit, §e may be multiplied by a constant factor chosen to avoid over/underflow in
	 *     computing the eigenvalues.
	 *
	 * \param[in] vl
	 *     If §range ='V', the lower bound of the interval to be searched for eigenvalues.
	 *     $\{vl}<\{vu}$.\n Not referenced if §range ='A' or 'I'.
	 *
	 * \param[in] vu
	 *     If §range ='V', the upper bound of the interval to be searched for eigenvalues.
	 *     $\{vl}<\{vu}$.\n Not referenced if §range ='A' or 'I'.
	 *
	 * \param[in] il
	 *     If §range ='I', the index of the smallest eigenvalue to be returned.\n
	 *     $0\le\{il}\le\{iu}<\{n}$, if $\{n}>0$; $\{il}=0$ and $\{iu}=-1$ if $\{n}=0$.\n
	 *     Not referenced if §range ='A' or 'V'.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in] iu
	 *     If §range ='I', the index of the largest eigenvalue to be returned.\n
	 *     $0\le\{il}\le\{iu}<\{n}$, if $\{n}>0$; $\{il}=0$ and $\{iu}=-1$ if $\{n}=0$.\n
	 *     Not referenced if §range ='A' or 'V'.\n
	 *     NOTE: zero-based index!
	 *
	 * \param[in] abstol
	 *     The absolute error tolerance for the eigenvalues. An approximate eigenvalue is accepted
	 *     as converged when it is determined to lie in an interval $[a,b]$ of width less than or
	 *     equal to\n
	 *         $\{abstol}+\{EPS}\max(|a|,|b|)$,\n
	 *     where §EPS is the machine precision. If §abstol is less than or equal to zero, then
	 *     $\{EPS}|T|$ will be used in its place, where $|T|$ is the 1-norm of the tridiagonal
	 *     matrix.\n
	 *     Eigenvalues will be computed most accurately when §abstol is set to twice the underflow
	 *     threshold $2\,\{dlamch}('S')$, not zero. If this routine returns with $\{info}>0$,
	 *     indicating that some eigenvectors did not converge, try setting §abstol to
	 *     $2\,\{dlamch}('S')$.\n
	 *     See "Computing Small Singular Values of Bidiagonal Matrices with Guaranteed High
	 *     Relative Accuracy," by Demmel and Kahan, LAPACK Working Note #3.
	 *
	 * \param[out] m
	 *     The total number of eigenvalues found. $0\le\{m}\le\{n}$.
	 *     If §range ='A', $\{m}=\{n}$, and if §range ='I', $\{m}=\{iu}-\{il}+1$.
	 *
	 * \param[out] w
	 *     an array, dimension (§n)\n
	 *     The first §m elements contain the selected eigenvalues in ascending order.
	 *
	 * \param[out] Z
	 *     an array, dimension (§ldz, $\max(1,\{m})$)\n
	 *     If §jobz ='V', then if $\{info}=0$, the first §m columns of §Z contain the orthonormal
	 *     eigenvectors of the matrix $A$ corresponding to the selected eigenvalues, with the
	 *     $i$-th column of §Z holding the eigenvector associated with $\{w}[i]$. If an eigenvector
	 *     fails to converge ($\{info}>0$), then that column of §Z contains the latest
	 *     approximation to the eigenvector, and the index of the eigenvector is returned in
	 *     §ifail. If §jobz ='N', then §Z is not referenced.\n
	 *     Note: the user must ensure that at least $\max(1,\{m})$ columns are supplied in the
	 *     array §Z; if §range ='V', the exact value of §m is not known in advance and an upper
	 *     bound must be used.
	 *
	 * \param[in] ldz
	 *     The leading dimension of the array §Z.
	 *     $\{ldz}\ge 1$, and if §jobz ='V', $\{ldz}\ge\max(1,\{n})$.
	 *
	 * \param[out] work  an array, dimension ($5\{n}$)
	 * \param[out] iwork an integer array, dimension ($5\{n}$)
	 * \param[out] ifail
	 *     an integer array, dimension (§n)\n
	 *     If §jobz ='V', then if $\{info}=0$, the first §m elements of §ifail are $-1$.
	 *     If $\{info}>0$, then §ifail contains the indices of the eigenvectors that failed to
	 *     converge.\n
	 *     If §jobz ='N', then §ifail is not referenced.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] info
	 *     =0: successful exit.\n
	 *     <0: if $\{info}=-i$, the $i$-th argument had an illegal value.\n
	 *     >0: if $\{info}=i$, then $i$ eigenvectors failed to converge.
	 *         Their indices are stored in array §ifail.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016                                                                           */
	static void dstevx(char const* const jobz, char const* const range, int const n, real* const d,
	                   real* const e, real const vl, real const vu, int const il, int const iu,
	                   real const abstol, int& m, real* const w, real* const Z, int const ldz,
	                   real* const work, int* const iwork, int* const ifail, int& info) /*const*/
	{
		// Test the input parameters.
		bool wantz  = (std::toupper(jobz[0]) =='V');
		bool alleig = (std::toupper(range[0])=='A');
		bool valeig = (std::toupper(range[0])=='V');
		bool indeig = (std::toupper(range[0])=='I');
		info = 0;
		if (!wantz && std::toupper(jobz[0])!='N')
		{
			info = -1;
		}
		else if (!(alleig || valeig || indeig))
		{
			info = -2;
		}
		else if (n<0)
		{
			info = -3;
		}
		else
		{
			if (valeig)
			{
				if (n>0 && vu<=vl)
				{
					info = -7;
				}
			}
			else if (indeig)
			{
				if (il<0 || il>std::max(0, n-1))
				{
					info = -8;
				}
				else if (iu<std::min(n-1, il) || iu>=n)
				{
					info = -9;
				}
			}
		}
		if (info==0){
			if (ldz<1 || (wantz && ldz<n))
			{
				info = -14;
			}
		}
		if (info!=0)
		{
			xerbla("DSTEVX", -info);
			return;
		}
		// Quick return if possible
		m = 0;
		if (n==0)
		{
			return;
		}
		if (n==1)
		{
			if (alleig || indeig)
			{
				m = 1;
				w[0] = d[0];
			}
			else
			{
				if (vl<d[0] && vu>=d[0])
				{
					m = 1;
					w[0] = d[0];
				}
			}
			if (wantz)
			{
				Z[0] = ONE;
			}
			return;
		}
		// Get machine constants.
		real safmin = dlamch("Safe minimum");
		real eps    = dlamch("Precision");
		real smlnum = safmin / eps;
		real bignum = ONE / smlnum;
		real rmin   = std::sqrt(smlnum);
		real rmax   = std::min(std::sqrt(bignum), ONE/std::sqrt(std::sqrt(safmin)));
		// Scale matrix to allowable range, if necessary.
		int iscale = 0;
		real vll, vuu;
		if (valeig)
		{
			vll = vl;
			vuu = vu;
		}
		else
		{
			vll = ZERO;
			vuu = ZERO;
		}
		real tnrm = dlanst("M", n, d, e);
		real sigma;
		if (tnrm>ZERO && tnrm<rmin)
		{
			iscale = 1;
			sigma = rmin / tnrm;
		}
		else if (tnrm>rmax)
		{
			iscale = 1;
			sigma = rmax / tnrm;
		}
		if (iscale==1)
		{
			Blas<real>::dscal(n,   sigma, d, 1);
			Blas<real>::dscal(n-1, sigma, e, 1);
			if (valeig)
			{
				vll = vl * sigma;
				vuu = vu * sigma;
			}
		}
		// If all eigenvalues are desired and abstol is less than zero, then call dsterf or dsteqr.
		// If this fails for some eigenvalue, then try dstebz
		bool test = false;
		if (indeig)
		{
			if (il==0 && iu==n-1)
			{
				test = true;
			}
		}
		int i, indwrk;
		bool err = true;
		if ((alleig || test) && abstol<=ZERO)
		{
			Blas<real>::dcopy(n,   d, 1, w,    1);
			Blas<real>::dcopy(n-1, e, 1, work, 1);
			indwrk = n;
			if (!wantz)
			{
				dsterf(n, w, work, info);
			}
			else
			{
				dsteqr("I", n, w, work, Z, ldz, &work[indwrk], info);
				if (info==0)
				{
					for (i=0; i<n; i++)
					{
						ifail[i] = -1;
					}
				}
			}
			if (info==0)
			{
				m = n;
				err=false;
			}
			info = 0;
		}
		int indibl;
		if (err)
		{
			// call dstebz and, if eigenvectors are desired, dstein.
			char order[2];
			order[1] = '\0';
			if (wantz)
			{
			   order[0] = 'B';
			}
			else
			{
			   order[0] = 'E';
			}
			indwrk = 0;
			indibl = 0;
			int indisp = indibl + n;
			int indiwo = indisp + n;
			int nsplit;
			dstebz(range, order, n, vll, vuu, il, iu, abstol, d, e, m, nsplit, w, &iwork[indibl],
			       &iwork[indisp], &work[indwrk], &iwork[indiwo], info);
			if (wantz)
			{
				dstein(n, d, e, m, w, &iwork[indibl], &iwork[indisp], Z, ldz, &work[indwrk],
				       &iwork[indiwo], ifail, info);
			}
		}
		// If matrix was scaled, then rescale eigenvalues appropriately.
		if (iscale==1)
		{
			int imax;
			if (info==0)
			{
				imax = m;
			}
			else
			{
				imax = info - 1;
			}
			Blas<real>::dscal(imax, ONE/sigma, w, 1);
		}
		// If eigenvalues are not in order, then sort them, along with eigenvectors.
		if (wantz)
		{
			int itmp1, j, jj;
			real tmp1;
			for (j=0; j<m-1; j++)
			{
				i = -1;
				tmp1 = w[j];
				for (jj=j+1; jj<m; jj++)
				{
					if (w[jj]<tmp1)
					{
						i = jj;
						tmp1 = w[jj];
					}
				}
				if (i!=-1)
				{
					itmp1           = iwork[indibl+i];
					w[i] = w[j];
					iwork[indibl+i] = iwork[indibl+j];
					w[j] = tmp1;
					iwork[indibl+j] = itmp1;
					Blas<real>::dswap(n, &Z[ldz*i], 1, &Z[ldz*j], 1);
					if (info!=0)
					{
						itmp1    = ifail[i];
						ifail[i] = ifail[j];
						ifail[j] = itmp1;
					}
				}
			}
		}
	}

	/*! §dtrevc3
	 *
	 * §dtrevc3 computes some or all of the right and/or left eigenvectors of a real upper
	 * quasi-triangular matrix $T$. Matrices of this type are produced by the Schur factorization
	 * of a real general matrix: $A = Q T Q^T$, as computed by §dhseqr.\n
	 * The right eigenvector $x$ and the left eigenvector $y$ of $T$ corresponding to an eigenvalue
	 * $w$ are defined by:\n
	 *     $T x = w x$, &emsp; $y^HT = wy^H$\n
	 * where $y^H$ denotes the conjugate transpose of $y$. The eigenvalues are not input to this
	 * routine, but are read directly from the diagonal blocks of $T$.\n
	 * This routine returns the matrices $X$ and/or $Y$ of right and left eigenvectors of $T$, or
	 * the products $Q X$ and/or $Q Y$, where $Q$ is an input matrix. If $Q$ is the orthogonal
	 * factor that reduces a matrix $A$ to Schur form $T$, then $Q X$ and $Q Y$ are the matrices of
	 * right and left eigenvectors of $A$.\n
	 * This uses a Level 3 BLAS version of the back transformation.
	 * \param[in] side
	 *     = 'R': compute right eigenvectors only;\n
	 *     = 'L': compute left eigenvectors only;\n
	 *     = 'B': compute both right and left eigenvectors.
	 *
	 * \param[in] howmny
	 *     = 'A': compute all right and/or left eigenvectors;\n
	 *     = 'B': compute all right and/or left eigenvectors, backtransformed by the matrices in
	 *            §Vr and/or §Vl;\n
	 *     = 'S': compute selected right and/or left eigenvectors, as indicated by the logical
	 *            array §select.
	 *
	 * \param[in,out] select
	 *     a boolean array, dimension (§n)\n
	 *     If §howmny = 'S', §select specifies the eigenvectors to be computed.
	 *     If $\{w}[j]$ is a real eigenvalue, the corresponding real eigenvector is computed if
	 *     $\{select}[j]$ is §true. If $\{w}[j]$ and $\{w}[j+1]$ are the real and imaginary parts
	 *     of a complex eigenvalue, the corresponding complex eigenvector is computed if either
	 *     $\{select}[j]$ or $\{select}[j+1]$ is §true, and on exit $\{select}[j]$ is set to §true
	 *     and $\{select}[j+1]$ is set to §false. \n Not referenced if §howmny = 'A' or 'B'.
	 *
	 * \param[in] n The order of the matrix $T$. $\{n}\ge 0$.
	 * \param[in] T
	 *      an array, dimension (§ldt,§n)\n
	 *      The upper quasi-triangular matrix $T$ in Schur canonical form.
	 *
	 * \param[in] ldt
	 *     The leading dimension of the array $T$. $\{ldt} \ge max(1,n)$.
	 *
	 * \param[in,out] Vl
	 *     an array, dimension (§ldvl,§mm)\n
	 *     On entry, if §side = 'L' or 'B' and §howmny = 'B', §Vl must contain an §n by §n matrix
	 *     $Q$ (usually the orthogonal matrix $Q$ of Schur vectors returned by §dhseqr).\n
	 *     On exit, if §side = 'L' or 'B', §Vl contains:
	 *     \li if §howmny = 'A', the matrix $Y$ of left eigenvectors of $T$;
	 *     \li if §howmny = 'B', the matrix $Q Y$;
	 *     \li if §howmny = 'S', the left eigenvectors of $T$ specified by §select, stored
	 *                           consecutively in the columns of §Vl, in the same order as their
	 *                           eigenvalues.
	 *
	 *     A complex eigenvector corresponding to a complex eigenvalue is stored in two consecutive
	 *     columns, the first holding the real part, and the second the imaginary part.\n
	 *     Not referenced if §side = 'R'.
	 *
	 * \param[in] ldvl
	 *     The leading dimension of the array §Vl. \n
	 *     $\{ldvl}\ge 1$, and if §side = 'L' or 'B', $\{ldvl}\ge\{n}$.
	 *
	 * \param[in,out] Vr
	 *     an array, dimension (§ldvr,§mm)\n
	 *     On entry, if §side = 'R' or 'B' and §howmny = 'B', §Vr must contain an §n by §n matrix
	 *     $Q$ (usually the orthogonal matrix $Q$ of Schur vectors returned by §dhseqr). \n
	 *     On exit, if §side = 'R' or 'B', §Vr contains:
	 *     \li if §howmny = 'A', the matrix $X$ of right eigenvectors of $T$;
	 *     \li if §howmny = 'B', the matrix $Q X$;
	 *     \li if §howmny = 'S', the right eigenvectors of $T$ specified by §select, stored
	 *                           consecutively in the columns of §Vr, in the same order as their
	 *                           eigenvalues.
	 *
	 *     A complex eigenvector corresponding to a complex eigenvalue is stored in two consecutive
	 *     columns, the first holding the real part and the second the imaginary part.\n
	 *     Not referenced if §side = 'L'.
	 *
	 * \param[in] ldvr
	 *     The leading dimension of the array §Vr.\n
	 *     $\{ldvr}\ge 1$, and if §side = 'R' or 'B', $\{ldvr}\ge\{n}$.
	 *
	 * \param[in] mm
	 *     The number of columns in the arrays §Vl and/or §Vr. $\{mm}\ge\{m}$.
	 *
	 * \param[out] m
	 *     The number of columns in the arrays §Vl and/or §Vr actually used to store the
	 *     eigenvectors. If §howmny = 'A' or 'B', §m is set to §n. Each selected real eigenvector
	 *     occupies one column and each selected complex eigenvector occupies two columns.
	 *
	 * \param[out] work  an array, dimension ($\max(1,\{lwork})$)
	 * \param[in]  lwork
	 *     The dimension of array §work. $\{lwork}\ge\max(1,3\{n})$.\n
	 *     For optimum performance, $\{lwork}\ge\{n}+2\{n}\ \{nb}$, where §nb is the optimal
	 *     blocksize.\n
	 *     If §lwork = -1, then a workspace query is assumed; the routine only calculates the
	 *     optimal size of the §work array, returns this value as the first entry of the §work
	 *     array, and no error message related to §lwork is issued by §xerbla.
	 *
	 * \param[out] info
	 *     = 0: successful exit\n
	 *     < 0: if §info = $-i$, the $i$-th argument had an illegal value
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date November 2017
	 * \remark
	 *     The algorithm used in this program is basically backward (forward) substitution, with
	 *     scaling to make the the code robust against possible overflow.\n
	 *     Each eigenvector is normalized so that the element of largest magnitude has magnitude 1;
	 *     here the magnitude of a complex number ($x$,$y$) is taken to be $|x| + |y|$.          */
	static void dtrevc3(char const* const side, char const* const howmny, bool* const select,
	                    int const n, real const* const T, int const ldt, real* const Vl,
	                    int const ldvl, real* const Vr, int const ldvr, int const mm, int& m,
	                    real* const work, int const lwork, int& info) /*const*/
	{
		const int NBMIN=8, NBMAX=128;
		// Decode and test the input parameters
		char upside = std::toupper(side[0]);
		bool bothv  = (upside=='B');
		bool rightv = (upside=='R' || bothv);
		bool leftv  = (upside=='L' || bothv);
		char uphowmny = std::toupper(howmny[0]);
		bool allv  = (uphowmny=='A');
		bool over  = (uphowmny=='B');
		bool somev = (uphowmny=='S');
		info = 0;
		int nb;
		{
			char def[3];
			def[0] = upside;
			def[1] = uphowmny;
			def[2] = '\0';
			nb = ilaenv(1, "DTREVC", def, n, -1, -1, -1);
		}
		work[0] = real(n + 2*n*nb); // maxwork
		bool lquery = (lwork==-1);
		int i, j;
		if (!rightv && !leftv)
		{
			info = -1;
		}
		else if (!allv && !over && !somev)
		{
			info = -2;
		}
		else if (n<0)
		{
			info = -4;
		}
		else if (ldt<std::max(1, n))
		{
			info = -6;
		}
		else if (ldvl<1 || (leftv && ldvl<n))
		{
			info = -8;
		}
		else if (ldvr<1 || (rightv && ldvr<n))
		{
			info = -10;
		}
		else if (lwork<std::max(1, 3*n) && !lquery)
		{
			info = -14;
		}
		else
		{
			// Set m to the number of columns required to store the selected eigenvectors,
			// standardize the array select if necessary, and test mm.
			if (somev)
			{
				m = 0;
				bool pair = false;
				for (j=0; j<n; j++)
				{
					if (pair)
					{
						pair = false;
						select[j] = false;
					}
					else
					{
						if (j<n-1)
						{
							if (T[j+1+ldt*j]==ZERO)
							{
								if (select[j])
								{
									m++;
								}
							}
							else
							{
								pair = true;
								if (select[j] || select[j+1])
								{
									select[j] = true;
									m += 2;
								}
							}
						}
						else
						{
							if (select[n-1])
							{
								m++;
							}
						}
					}
				}
			}
			else
			{
				m = n;
			}
			if (mm<m)
			{
				info = -11;
			}
		}
		if (info!=0)
		{
			xerbla("DTREVC3", -info);
			return;
		}
		else if (lquery)
		{
			return;
		}
		// Quick return if possible.
		if (n==0)
		{
			return;
		}
		// Use blocked version of back-transformation if sufficient workspace.
		// Zero-out the workspace to avoid potential NaN propagation.
		if (over && lwork >= n + 2*n*NBMIN)
		{
			nb = (lwork-n) / (2*n);
			nb = std::min(nb, NBMAX);
			dlaset("F", n, 1+2*nb, ZERO, ZERO, work, n);
		}
		else
		{
			nb = 1;
		}
		// Set the constants to control overflow.
		real unfl = dlamch("Safe minimum");
		real ovfl = ONE / unfl;
		dlabad(unfl, ovfl);
		real ulp = dlamch("Precision");
		real smlnum = unfl * (n/ulp);
		real bignum = (ONE-ulp) / smlnum;
		// Compute 1-norm of each column of strictly upper triangular part of T to control overflow
		// in triangular solver.
		work[0] = ZERO;
		int tcolki;
		for (j=1; j<n; j++)
		{
			work[j] = ZERO;
			tcolki = ldt * j;
			for (i=0; i<j; i++)
			{
				work[j] += std::fabs(T[i+tcolki]);
			}
		}
		/* Index ip is used to specify the real or complex eigenvalue:
		 * ip = 0, real eigenvalue,
		 *      1, first  of conjugate complex pair: (wr,wi)
		 *     -1, second of conjugate complex pair: (wr,wi)
		 * iscomplex array stores ip for each column in current block.                           */
		int ierr, ii, ip, is, iv, ivpn, j1, j2, jnxt, k, ki, kip1, ki2, tcolj, tcol3, workcol;
		real emax, rec, remax=ONE, scale, smin, wi, wr, xnorm;
		real X[2*2];
		int iscomplex[NBMAX];
		if (rightv)
		{
			/* Compute right eigenvectors.
			 * iv is index of column in current block.
			 * For complex right vector, uses iv for real part and iv+1 for complex part.
			 * Non-blocked version always uses iv=1;
			 * blocked     version starts with iv=nb-1, goes down to 0 or 1.
			 * (Note the "-1-th" column is used for 1-norms computed above.)                     */
			iv = 1;
			if (nb>2)
			{
				iv = nb - 1;
			}
			ip = 0;
			is = m - 1;
			int ivn, jm1, kim1, tcolkim, vrcol, vrcolm;
			ivn = iv * n;
			ivpn = ivn + n;
			for (ki=n-1; ki>=0; ki--)
			{
				kim1 = ki - 1;
				kip1 = ki + 1;
				tcolki = ldt * ki;
				tcolkim = tcolki - ldt;
				if (ip==-1)
				{
					// previous iteration (ki+1) was second of conjugate pair,
					// so this ki is first of conjugate pair; skip to end of loop
					ip = 1;
					continue;
				}
				else if (ki==0)
				{
					// last column, so this ki must be real eigenvalue
					ip = 0;
				}
				else if (T[ki+tcolkim]==ZERO)
				{
					// zero on sub-diagonal, so this ki is real eigenvalue
					ip = 0;
				}
				else
				{
					// non-zero on sub-diagonal, so this ki is second of conjugate pair
					ip = -1;
				}
				if (somev)
				{
					if (ip==0)
					{
						if (!select[ki])
						{
							continue;
						}
					}
					else
					{
						if (!select[kim1])
						{
							continue;
						}
					}
				}
				// Compute the ki-th eigenvalue (wr,wi).
				wr = T[ki+tcolki];
				wi = ZERO;
				if (ip!=0)
				{
					wi = std::sqrt(std::fabs(T[ki+tcolkim]))
					   * std::sqrt(std::fabs(T[kim1+tcolki]));
				}
				smin = std::max(ulp*(std::fabs(wr)+std::fabs(wi)), smlnum);
				if (ip==0)
				{
					// Real right eigenvector
					work[ki+ivpn] = ONE;
					// Form right-hand side.
					for (k=0; k<ki; k++)
					{
						work[k+ivpn] = -T[k+tcolki];
					}
					// Solve upper quasi-triangular system: [T[0:ki-1,0:ki-1] - wr]*X = scale*work.
					jnxt = kim1;
					for (j=kim1; j>=0; j--)
					{
						if (j>jnxt)
						{
							continue;
						}
						j1 = j;
						j2 = j;
						jm1 = j - 1;
						jnxt = jm1;
						tcolj = ldt * j;
						tcol3 = tcolj - ldt;
						if (j>0)
						{
							if (T[j+tcol3]!=ZERO)
							{
								j1   = jm1;
								jnxt = j - 2;
							}
						}
						if (j1==j2)
						{
							// 1-by-1 diagonal block
							dlaln2(false, 1, 1, smin, ONE, &T[j+tcolj], ldt, ONE, ONE,
							       &work[j+ivpn], n, wr, ZERO, X, 2, scale, xnorm, ierr);
							// Scale X[0,0] to avoid overflow when updating the right-hand side.
							if (xnorm>ONE)
							{
								if (work[j] > bignum/xnorm)
								{
									X[0]  /= xnorm;
									scale /= xnorm;
								}
							}
							// Scale if necessary
							if (scale!=ONE)
							{
								Blas<real>::dscal(kip1, scale, &work[ivpn], 1);
							}
							work[j+ivpn] = X[0];
							// Update right-hand side
							Blas<real>::daxpy(j, -X[0], &T[tcolj], 1, &work[ivpn], 1);
						}
						else
						{
							// 2-by-2 diagonal block
							dlaln2(false, 2, 1, smin, ONE, &T[jm1+tcol3], ldt, ONE, ONE,
							       &work[jm1+ivpn], n, wr, ZERO, X, 2, scale, xnorm, ierr);
							// Scale X[0,0] and X[1,0] to avoid overflow when updating the
							// right-hand side.
							if (xnorm>ONE)
							{
								if (std::max(work[jm1], work[j]) > bignum/xnorm)
								{
									X[0]  /= xnorm;
									X[1]  /= xnorm;
									scale /= xnorm;
								}
							}
							// Scale if necessary
							if (scale!=ONE)
							{
								Blas<real>::dscal(kip1, scale, &work[ivpn], 1);
							}
							work[jm1+ivpn] = X[0];
							work[j  +ivpn] = X[1];
							// Update right-hand side
							Blas<real>::daxpy(jm1, -X[0], &T[tcol3], 1, &work[ivpn], 1);
							Blas<real>::daxpy(jm1, -X[1], &T[tcolj],     1, &work[ivpn], 1);
						}
					}
					// Copy the vector x or Q*x to Vr and normalize.
					if (!over)
					{
						// no back-transform: copy x to Vr and normalize.
						vrcol = ldvr * is;
						Blas<real>::dcopy(kip1, &work[ivpn], 1, &Vr[vrcol], 1);
						ii = Blas<real>::idamax(kip1, &Vr[vrcol], 1);
						remax = ONE / std::fabs(Vr[ii+vrcol]);
						Blas<real>::dscal(kip1, remax, &Vr[vrcol], 1);
						for (k=kip1; k<n; k++)
						{
							Vr[k+vrcol] = ZERO;
						}
					}
					else if (nb==1)
					{
						// version 1: back-transform each vector with GEMV, Q*x.
						vrcol = ldvr * ki;
						if (ki>0)
						{
							Blas<real>::dgemv("N", n, ki, ONE, Vr, ldvr, &work[ivpn], 1,
							                  work[ki+ivpn], &Vr[vrcol], 1);
						}
						ii = Blas<real>::idamax(n, &Vr[vrcol], 1);
						remax = ONE / std::fabs(Vr[ii+vrcol]);
						Blas<real>::dscal(n, remax, &Vr[vrcol], 1);
					}
					else
					{
						// version 2: back-transform block of vectors with GEMM
						// zero out below vector
						for (k=kip1; k<n; k++)
						{
							work[k+ivpn] = ZERO;
						}
						iscomplex[iv] = ip;
						// back-transform and normalization is done below
					}
				}
				else
				{
					// Complex right eigenvector.
					// Initial solve
					// [ (T[ki-1,ki-1] T[ki-1,ki]) - (wr + i*wi) ]*X = 0.
					// [ (T[ki,ki-1]   T[ki,ki])                 ]
					if (std::fabs(T[kim1+tcolki]) >= std::fabs(T[ki+tcolkim]))
					{
						work[kim1+ivn]  = ONE;
						work[ki  +ivpn] = wi / T[kim1+tcolki];
					}
					else
					{
						work[kim1+ivn]  = -wi / T[ki+tcolkim];
						work[ki  +ivpn] = ONE;
					}
					work[ki + ivn]  = ZERO;
					work[kim1+ivpn] = ZERO;
					// Form right-hand side.
					for (k=0; k<kim1; k++)
					{
						work[k+ivn]  = -work[kim1+ivn]  * T[k+tcolkim];
						work[k+ivpn] = -work[ki + ivpn] * T[k+tcolki];
					}
					// Solve upper quasi-triangular system:
					// [ T[0:ki-2,0:ki-2] - (wr+i*wi) ]*X = scale*(work+i*WORK2)
					jnxt = ki - 2;
					for (j=ki-2; j>=0; j--)
					{
						if (j>jnxt)
						{
							continue;
						}
						j1   = j;
						j2   = j;
						jm1  = j - 1;
						jnxt = jm1;
						tcolj = ldt * j;
						tcol3 = tcolj - ldt;
						if (j>0)
						{
							if (T[j+tcol3]!=ZERO)
							{
								j1   = jm1;
								jnxt = j - 2;
							}
						}
						if (j1==j2)
						{
							// 1-by-1 diagonal block
							dlaln2(false, 1, 2, smin, ONE, &T[j+tcolj], ldt, ONE, ONE,
							       &work[j+ivn], n, wr, wi, X, 2, scale, xnorm, ierr);
							// Scale X[0,0] and X[0,1] to avoid overflow when updating the
							// right-hand side.
							if (xnorm>ONE)
							{
								if (work[j] > bignum/xnorm)
								{
									X[0]  /= xnorm;
									X[2]  /= xnorm;
									scale /= xnorm;
								}
							}
							// Scale if necessary
							if (scale!=ONE)
							{
								Blas<real>::dscal(kip1, scale, &work[ivn],  1);
								Blas<real>::dscal(kip1, scale, &work[ivpn], 1);
							}
							work[j+ivn]  = X[0];
							work[j+ivpn] = X[2];
							// Update the right-hand side
							Blas<real>::daxpy(j, -X[0], &T[tcolj], 1, &work[ivn],  1);
							Blas<real>::daxpy(j, -X[2], &T[tcolj], 1, &work[ivpn], 1);
						}
						else
						{
							// 2 by 2 diagonal block
							dlaln2(false, 2, 2, smin, ONE, &T[jm1+tcol3], ldt, ONE, ONE,
							       &work[jm1+ivn], n, wr, wi, X, 2, scale, xnorm, ierr);
							// Scale X to avoid overflow when updating the right-hand side.
							if (xnorm>ONE)
							{
								if (std::max(work[jm1], work[j]) > bignum/xnorm)
								{
									rec = ONE / xnorm;
									X[0]  *= rec;
									X[2]  *= rec;
									X[1]  *= rec;
									X[3]  *= rec;
									scale *= rec;
								}
							}
							// Scale if necessary
							if (scale!=ONE)
							{
								Blas<real>::dscal(kip1, scale, &work[ivn], 1);
								Blas<real>::dscal(kip1, scale, &work[ivpn], 1);
							}
							work[jm1+ivn]  = X[0];
							work[j  +ivn]  = X[1];
							work[jm1+ivpn] = X[2];
							work[j  +ivpn] = X[3];
							// Update the right-hand side
							Blas<real>::daxpy(jm1, -X[0], &T[tcol3], 1, &work[ivn],  1);
							Blas<real>::daxpy(jm1, -X[1], &T[tcolj],     1, &work[ivn],  1);
							Blas<real>::daxpy(jm1, -X[2], &T[tcol3], 1, &work[ivpn], 1);
							Blas<real>::daxpy(jm1, -X[3], &T[tcolj],     1, &work[ivpn], 1);
						}
					}
					// Copy the vector x or Q*x to Vr and normalize.
					if (!over)
					{
						// no back-transform: copy x to Vr and normalize.
						vrcol  = ldvr * is;
						vrcolm = vrcol - ldvr;
						Blas<real>::dcopy(kip1, &work[ivn],  1, &Vr[vrcolm], 1);
						Blas<real>::dcopy(kip1, &work[ivpn], 1, &Vr[vrcol],  1);
						emax = ZERO;
						for (k=0; k<=ki; k++)
						{
							emax = std::max(emax, std::fabs(Vr[k+vrcolm])+std::fabs(Vr[k+vrcol]));
						}
						remax = ONE / emax;
						Blas<real>::dscal(kip1, remax, &Vr[vrcolm], 1);
						Blas<real>::dscal(kip1, remax, &Vr[vrcol],  1);
						for (k=kip1; k<n; k++)
						{
							Vr[k+vrcolm] = ZERO;
							Vr[k+vrcol]  = ZERO;
						}
					}
					else if (nb==1)
					{
						// version 1: back-transform each vector with GEMV, Q*x.
						vrcol  = ldvr * ki;
						vrcolm = vrcol - ldvr;
						if (ki>1)
						{
							Blas<real>::dgemv("N", n, kim1, ONE, Vr, ldvr, &work[ivn],  1,
							                  work[kim1+ivn], &Vr[vrcolm], 1);
							Blas<real>::dgemv("N", n, kim1, ONE, Vr, ldvr, &work[ivpn], 1,
							                  work[ki+ivpn],  &Vr[vrcol],  1);
						}
						else
						{
							Blas<real>::dscal(n, work[kim1+ivn],  &Vr[vrcolm], 1);
							Blas<real>::dscal(n, work[ki + ivpn], &Vr[vrcol],  1);
						}
						emax = ZERO;
						for (k=0; k<n; k++)
						{
							emax = std::max(emax, std::fabs(Vr[k+vrcolm])+std::fabs(Vr[k+vrcol]));
						}
						remax = ONE / emax;
						Blas<real>::dscal(n, remax, &Vr[vrcolm], 1);
						Blas<real>::dscal(n, remax, &Vr[vrcol],  1);
					}
					else
					{
						// version 2: back-transform block of vectors with GEMM
						// zero out below vector
						for (k=kip1; k<n; k++)
						{
							work[k+ivn]  = ZERO;
							work[k+ivpn] = ZERO;
						}
						iscomplex[iv-1] = -ip;
						iscomplex[iv]   =  ip;
						iv--;
						ivn  = iv * n;
						ivpn = ivn + n;
						// back-transform and normalization is done below
					}
				}
				if (nb>1)
				{
					// Blocked version of back-transform
					// For complex case, ki2 includes both vectors (ki-1 and ki)
					if (ip==0)
					{
						ki2 = ki;
					}
					else
					{
						ki2 = kim1;
					}
					// Columns iv+1:nb-1 of work are valid vectors.
					// When the number of vectors stored reaches nb-1 or nb, or if this was last
					// vector, do the GEMM
					if (iv<=1 || ki2==0)
					{
						Blas<real>::dgemm("N", "N", n, nb-iv, ki2+nb-iv, ONE, Vr, ldvr,
						                  &work[ivpn], n, ZERO, &work[(nb+iv+1)*n], n);
						// normalize vectors
						for (k=iv; k<nb; k++)
						{
							workcol = (nb+k+1) * n;
							if (iscomplex[k]==0)
							{
								// real eigenvector
								ii = Blas<real>::idamax(n, &work[workcol], 1);
								remax = ONE / std::fabs(work[ii+workcol]);
							}
							else if (iscomplex[k]==1)
							{
								// first eigenvector of conjugate pair
								emax = ZERO;
								for (ii=0; ii<n; ii++)
								{
									emax = std::max(emax, std::fabs(work[ii+workcol])
									                     +std::fabs(work[ii+workcol+n]));
								}
								remax = ONE / emax;
							// }else if (iscomplex[k]==-1){
							// second eigenvector of conjugate pair reuse same remax as previous k
							}
							Blas<real>::dscal(n, remax, &work[workcol], 1);
						}
						dlacpy("F", n, nb-iv, &work[(nb+iv+1)*n], n, &Vr[ldvr*ki2], ldvr);
						iv = nb - 1;
					}
					else
					{
						iv--;
					}
					ivn = iv * n;
					ivpn = ivn + n;
				}// blocked back-transform
				is--;
				if (ip!=0)
				{
					is--;
				}
			}
		}
		if (leftv)
		{
			// Compute left eigenvectors.
			// iv is index of column in current block.
			// For complex left vector, uses iv for real part and iv+1 for complex part.
			// Non-blocked version always uses iv=0;
			// blocked     version starts with iv=0, goes up to nb-2 or nb-1.
			// (Note the "-1-th" column is used for 1-norms computed above.)
			int ivp2n, jp1, kip2, vlcol, vlcolp;
			real temp, vcrit, vmax;
			iv = 0;
			ivpn = n;
			ivp2n = n + n;
			ip = 0;
			is = 0;
			for (ki=0; ki<n; ki++)
			{
				kip1 = ki + 1;
				kip2 = ki + 2;
				tcolki = ldt * ki;
				if (ip==1)
				{
					// previous iteration (ki-1) was first of conjugate pair,
					// so this ki is second of conjugate pair; skip to end of loop
					ip = -1;
					continue;
				}
				else if (ki==n-1)
				{
					// last column, so this ki must be real eigenvalue
					ip = 0;
				}
				else if (T[kip1+tcolki]==ZERO)
				{
					// zero on sub-diagonal, so this ki is real eigenvalue
					ip = 0;
				}
				else
				{
					// non-zero on sub-diagonal, so this ki is first of conjugate pair
					ip = 1;
				}
				if (somev)
				{
					if (!select[ki])
					{
						continue;
					}
				}
				// Compute the ki-th eigenvalue (wr,wi).
				wr = T[ki+tcolki];
				wi = ZERO;
				if (ip!=0)
				{
					wi = std::sqrt(std::fabs(T[ki+tcolki+ldt]))
					   * std::sqrt(std::fabs(T[kip1+tcolki]));
				}
				smin = std::max(ulp*(std::fabs(wr)+std::fabs(wi)), smlnum);
				if (ip==0)
				{
					// Real left eigenvector
					work[ki+ivpn] = ONE;
					// Form right-hand side.
					for (k=kip1; k<n; k++)
					{
						work[k+ivpn] = -T[ki+ldt*k];
					}
					// Solve transposed quasi-triangular system:
					// [ T[ki+1:n-1,ki+1:n-1] - wr ]^T * X = scale*work
					vmax = ONE;
					vcrit = bignum;
					jnxt = kip1;
					for (j=kip1; j<n; j++)
					{
						if (j<jnxt)
						{
							continue;
						}
						j1 = j;
						j2 = j;
						jp1 = j + 1;
						jnxt = jp1;
						tcolj = ldt * j;
						if (j<n-1)
						{
							if (T[jp1+tcolj]!=ZERO)
							{
								j2 = jp1;
								jnxt = j + 2;
							}
						}
						if (j1==j2)
						{
							// 1 by 1 diagonal block
							// Scale if necessary to avoid overflow when forming the right-hand
							// side.
							if (work[j]>vcrit)
							{
								rec = ONE / vmax;
								Blas<real>::dscal(n-ki, rec, &work[ki+ivpn], 1);
								vmax = ONE;
								vcrit = bignum;
							}
							work[j+ivpn] -= Blas<real>::ddot(j-kip1, &T[kip1+tcolj], 1,
							                                 &work[kip1+ivpn], 1);
							// Solve [ T[j,j] - wr ]^T * X = work
							dlaln2(false, 1, 1, smin, ONE, &T[j+tcolj], ldt, ONE, ONE,
							       &work[j+ivpn], n, wr, ZERO, X, 2, scale, xnorm, ierr);
							// Scale if necessary
							if (scale!=ONE)
							{
								Blas<real>::dscal(n-ki, scale, &work[ki+ivpn], 1);
							}
							work[j+ivpn] = X[0];
							vmax = std::max(std::fabs(work[j+ivpn]), vmax);
							vcrit = bignum / vmax;
						}
						else
						{
							// 2 by 2 diagonal block
							// Scale if necessary to avoid overflow when forming the right-hand
							// side.
							if (std::max(work[j], work[jp1]) > vcrit)
							{
								rec = ONE / vmax;
								Blas<real>::dscal(n-ki, rec, &work[ki+ivpn], 1);
								vmax = ONE;
								vcrit = bignum;
							}
							work[j + ivpn] -= Blas<real>::ddot(j-kip1, &T[kip1+tcolj],     1,
							                                   &work[kip1+ivpn], 1);
							work[jp1+ivpn] -= Blas<real>::ddot(j-kip1, &T[kip1+tcolj+ldt], 1,
							                                   &work[kip1+ivpn], 1);
							// Solve
							// [ T[j,j]-wr   T[j,j+1]      ]^T * X = scale*(WORK1)
							// [ T[j+1,j]    T[j+1,j+1]-wr ]               (WORK2)
							dlaln2(true, 2, 1, smin, ONE, &T[j+tcolj], ldt, ONE, ONE,
							       &work[j+ivpn], n, wr, ZERO, X, 2, scale, xnorm, ierr);
							// Scale if necessary
							if (scale!=ONE)
							{
								Blas<real>::dscal(n-ki, scale, &work[ki+ivpn], 1);
							}
							work[j+ivpn] = X[0];
							work[jp1+ivpn] = X[1];
							temp = std::max(std::fabs(work[j+ivpn]), std::fabs(work[jp1+ivpn]));
							if (temp>vmax)
							{
								vmax = temp;
							}
							vcrit = bignum / vmax;
						}
					}
					// Copy the vector x or Q*x to Vl and normalize.
					if (!over)
					{
						// no back-transform: copy x to Vl and normalize.
						vlcol = ldvl * is;
						Blas<real>::dcopy(n-ki, &work[ki+ivpn], 1, &Vl[ki+vlcol], 1);
						ii = Blas<real>::idamax(n-ki, &Vl[ki+vlcol], 1) + ki;
						remax = ONE / std::fabs(Vl[ii+vlcol]);
						Blas<real>::dscal(n-ki, remax, &Vl[ki+vlcol], 1);
						for (k=0; k<ki; k++)
						{
							Vl[k+vlcol] = ZERO;
						}
					}
					else if (nb==1)
					{
						// version 1: back-transform each vector with GEMV, Q*x.
						vlcol = ldvl * ki;
						if (ki<n-1)
						{
							Blas<real>::dgemv("N", n, n-kip1, ONE, &Vl[vlcol+ldvl], ldvl,
							                  &work[kip1+ivpn], 1, work[ki+ivpn], &Vl[vlcol], 1);
						}
						ii = Blas<real>::idamax(n, &Vl[vlcol], 1);
						remax = ONE / std::fabs(Vl[ii+vlcol]);
						Blas<real>::dscal(n, remax, &Vl[vlcol], 1);
					}
					else
					{
						// version 2: back-transform block of vectors with GEMM zero out above
						// vector could go from ki-nv+2 to ki
						for (k=0; k<ki; k++)
						{
							work[k+ivpn] = ZERO;
						}
						iscomplex[iv] = ip;
						// back-transform and normalization is done below
					}
				}
				else
				{
					// Complex left eigenvector.
					// Initial solve:
					// [ (T[ki,ki]   T[ki,ki+1])^T - (wr - i* wi) ]*X = 0.
					// [ (T[ki+1,ki] T[ki+1,ki+1])                ]
					if (std::fabs(T[ki+tcolki+ldt])>=std::fabs(T[kip1+tcolki]))
					{
						work[ki  +ivpn]  = wi / T[ki+tcolki+ldt];
						work[kip1+ivp2n] = ONE;
					}
					else
					{
						work[ki  +ivpn]  = ONE;
						work[kip1+ivp2n] = -wi / T[kip1+tcolki];
					}
					work[kip1+ivpn]  = ZERO;
					work[ki  +ivp2n] = ZERO;
					// Form right-hand side.
					for (k=kip2; k<n; k++)
					{
						work[k+ivpn]  = -work[ki  +ivpn] *T[ki  +ldt*k];
						work[k+ivp2n] = -work[kip1+ivp2n]*T[kip1+ldt*k];
					}
					// Solve transposed quasi-triangular system:
					// [ T[ki+2:n-1,ki+2:n-1]^T - (wr-i*wi) ]*X = WORK1+i*WORK2
					vmax = ONE;
					vcrit = bignum;
					jnxt = kip2;
					for (j=kip2; j<n; j++)
					{
						if (j<jnxt)
						{
							continue;
						}
						j1 = j;
						j2 = j;
						jp1 = j + 1;
						jnxt = jp1;
						tcolj = ldt * j;
						if (j<n-1)
						{
							if (T[jp1+tcolj]!=ZERO)
							{
								j2 = jp1;
								jnxt = j + 2;
							}
						}
						if (j1==j2)
						{
							// 1-by-1 diagonal block
							// Scale if necessary to avoid overflow when forming the right-hand
							// side elements.
							if (work[j]>vcrit)
							{
								rec = ONE / vmax;
								Blas<real>::dscal(n-ki, rec, &work[ki+ivpn], 1);
								Blas<real>::dscal(n-ki, rec, &work[ki+ivp2n], 1);
								vmax = ONE;
								vcrit = bignum;
							}
							work[j+ivpn] -= Blas<real>::ddot(j-kip2, &T[kip2+tcolj], 1,
							                                 &work[kip2+ivpn], 1);
							work[j+ivp2n] -= Blas<real>::ddot(j-kip2, &T[kip2+tcolj], 1,
							                                  &work[kip2+ivp2n], 1);
							// Solve [ T[j,j]-(wr-i*wi) ]*(X11+i*X12)= WK+i*WK2.
							dlaln2(false, 1, 2, smin, ONE, &T[j+tcolj], ldt, ONE, ONE,
							       &work[j+ivpn], n, wr, -wi, X, 2, scale, xnorm, ierr);
							// Scale if necessary
							if (scale!=ONE)
							{
								Blas<real>::dscal(n-ki, scale, &work[ki+ivpn],  1);
								Blas<real>::dscal(n-ki, scale, &work[ki+ivp2n], 1);
							}
							work[j+ivpn]  = X[0];
							work[j+ivp2n] = X[2];
							temp = std::max(std::fabs(work[j+ivpn]), std::fabs(work[j+ivp2n]));
							if (temp>vmax)
							{
								vmax = temp;
							}
							vcrit = bignum / vmax;
						}
						else
						{
							// 2-by-2 diagonal block
							// Scale if necessary to avoid overflow when forming the right-hand
							// side elements.
							if (std::max(work[j], work[jp1]) > vcrit)
							{
								rec = ONE / vmax;
								Blas<real>::dscal(n-ki, rec, &work[ki+ivpn],  1);
								Blas<real>::dscal(n-ki, rec, &work[ki+ivp2n], 1);
								vmax = ONE;
								vcrit = bignum;
							}
							work[j+ivpn]    -= Blas<real>::ddot(j-kip2, &T[kip2+tcolj],     1,
							                                    &work[kip2+ivpn],  1);
							work[j+ivp2n]   -= Blas<real>::ddot(j-kip2, &T[kip2+tcolj],     1,
							                                    &work[kip2+ivp2n], 1);
							work[jp1+ivpn]  -= Blas<real>::ddot(j-kip2, &T[kip2+tcolj+ldt], 1,
							                                    &work[kip2+ivpn],  1);
							work[jp1+ivp2n] -= Blas<real>::ddot(j-kip2, &T[kip2+tcolj+ldt], 1,
							                                    &work[kip2+ivp2n], 1);
							// Solve 2-by-2 complex linear equation
							// [ (T[j,j]   T[j,j+1])^T - (wr-i*wi)*i ]*X = scale*B
							// [ (T[j+1,j] T[j+1,j+1])               ]
							dlaln2(true, 2, 2, smin, ONE, &T[j+tcolj], ldt, ONE, ONE,
							       &work[j+ivpn], n, wr, -wi, X, 2, scale, xnorm, ierr);
							// Scale if necessary
							if (scale!=ONE)
							{
								Blas<real>::dscal(n-ki, scale, &work[ki+ivpn],  1);
								Blas<real>::dscal(n-ki, scale, &work[ki+ivp2n], 1);
							}
							work[j  +ivpn]  = X[0];
							work[j  +ivp2n] = X[2];
							work[jp1+ivpn]  = X[1];
							work[jp1+ivp2n] = X[3];
							temp = std::max(std::max(std::fabs(X[0]), std::fabs(X[2])),
							                std::max(std::fabs(X[1]), std::fabs(X[3])));
							if (temp>vmax)
							{
								vmax = temp;
							}
							vcrit = bignum / vmax;
						}
					}
					// Copy the vector x or Q*x to Vl and normalize.
					if (!over)
					{
						// no back-transform: copy x to Vl and normalize.
						vlcol  = ldvl * is;
						vlcolp = vlcol + ldvl;
						Blas<real>::dcopy(n-ki, &work[ki+ivpn],  1, &Vl[ki+vlcol],  1);
						Blas<real>::dcopy(n-ki, &work[ki+ivp2n], 1, &Vl[ki+vlcolp], 1);
						emax = ZERO;
						for (k=kip1; k<n; k++)
						{
							emax = std::max(emax, std::fabs(Vl[k+vlcol])+std::fabs(Vl[k+vlcolp]));
						}
						remax = ONE / emax;
						Blas<real>::dscal(n-ki, remax, &Vl[ki+vlcol],  1);
						Blas<real>::dscal(n-ki, remax, &Vl[ki+vlcolp], 1);
						for (k=0; k<ki; k++)
						{
							Vl[k+vlcol]  = ZERO;
							Vl[k+vlcolp] = ZERO;
						}
					}
					else if (nb==1)
					{
						// version 1: back-transform each vector with GEMV, Q*x.
						vlcol  = ldvl * ki;
						vlcolp = vlcol + ldvl;
						if (ki<n-2)
						{
							Blas<real>::dgemv("N", n, n-kip2, ONE, &Vl[vlcolp+ldvl], ldvl,
							                  &work[kip2+ivpn],  1, work[ki+ivpn],  &Vl[vlcol], 1);
							Blas<real>::dgemv("N", n, n-kip2, ONE, &Vl[vlcolp+ldvl], ldvl,
							                  &work[kip2+ivp2n], 1, work[kip1+ivp2n], &Vl[vlcolp],
							                  1);
						}
						else
						{
							Blas<real>::dscal(n, work[ki  +ivpn],  &Vl[vlcol],  1);
							Blas<real>::dscal(n, work[kip1+ivp2n], &Vl[vlcolp], 1);
						}
						emax = ZERO;
						for (k=0; k<n; k++)
						{
							emax = std::max(emax, std::fabs(Vl[k+vlcol])+std::fabs(Vl[k+vlcolp]));
						}
						remax = ONE / emax;
						Blas<real>::dscal(n, remax, &Vl[vlcol],  1);
						Blas<real>::dscal(n, remax, &Vl[vlcolp], 1);
					}
					else
					{
						// version 2: back-transform block of vectors with GEMM zero out above
						// vector could go from ki-nv+2 to ki
						for (k=0; k<ki; k++)
						{
							work[k+ivpn]  = ZERO;
							work[k+ivp2n] = ZERO;
						}
						iscomplex[iv]   =  ip;
						iscomplex[iv+1] = -ip;
						iv++;
						ivpn = ivp2n;//(iv+1) * n;
						ivp2n = ivpn + n;
						// back-transform and normalization is done below
					}
				}
				if (nb>1)
				{
					// Blocked version of back-transform
					// For complex case, ki2 includes both vectors (ki and ki+1)
					if (ip==0)
					{
						ki2 = ki;
					}
					else
					{
						ki2 = kip1;
					}
					// Columns 0:iv of work are valid vectors.
					// When the number of vectors stored reaches nb-1 or nb,
					// or if this was last vector, do the GEMM
					if (iv>=nb-2 || ki2==n-1)
					{
						Blas<real>::dgemm("N", "N", n, iv+1, n-ki2+iv, ONE, &Vl[ldvl*(ki2-iv)],
						                  ldvl, &work[ki2-iv+n], n, ZERO, &work[(nb+1)*n], n);
						// normalize vectors
						for (k=0; k<=iv; k++)
						{
							workcol = (nb+k+1) * n;
							if (iscomplex[k]==0)
							{
								// real eigenvector
								ii = Blas<real>::idamax(n, &work[workcol], 1);
								remax = ONE / std::fabs(work[ii+workcol]);
							}
							else if (iscomplex[k]==1)
							{
								// first eigenvector of conjugate pair
								emax = ZERO;
								for (ii=0; ii<n; ii++)
								{
									emax = std::max(emax,
									    std::fabs(work[ii+workcol])+std::fabs(work[ii+workcol+n]));
								}
								remax = ONE / emax;
							//}else if (iscomplex[k]==-1){
							//    // second eigenvector of conjugate pair
							//    // reuse same remax as previous k
							}
							Blas<real>::dscal(n, remax, &work[workcol], 1);
						}
						dlacpy("F", n, iv+1, &work[(nb+1)*n], n, &Vl[ldvl*(ki2-iv)], ldvl);
						iv = 0;
					}
					else
					{
						iv++;
					}
					ivpn = (iv+1) * n;
					ivp2n = ivpn + n;
				}// blocked back-transform
				is++;
				if (ip!=0)
				{
					is++;
				}
			}
		}
	}

	/*! §dtrexc
	 *
	 * §dtrexc reorders the real Schur factorization of a real matrix $A = Q T Q^T$, so that the
	 * diagonal block of $T$ with row index §ifst is moved to row §ilst.\n
	 * The real Schur form $T$ is reordered by an orthogonal similarity transformation $Z^T T Z$,
	 * and optionally the matrix $Q$ of Schur vectors is updated by postmultiplying it with $Z$.\n
	 * $T$ must be in Schur canonical form (as returned by §dhseqr), that is, block upper
	 * triangular with 1 by 1 and 2 by 2 diagonal blocks; each 2 by 2 diagonal block has its
	 * diagonal elements equal and its off-diagonal elements of opposite sign.
	 * \param[in] compq
	 *     = 'V': update the matrix $Q$ of Schur vectors;\n
	 *     = 'N': do not update $Q$.
	 *
	 * \param[in] n
	 *     The order of the matrix $T$. $\{n}\ge 0$.\n
	 *     If $\{n}=0$ arguments §ilst and §ifst may be any value.
	 *
	 * \param[in,out] T
	 *     an array, dimension (§ldt,§n)\n
	 *     On entry, the upper quasi-triangular matrix $T$, in Schur Schur canonical form.\n
	 *     On exit, the reordered upper quasi-triangular matrix, again in Schur canonical form.
	 *
	 * \param[in] ldt The leading dimension of the array §T. $\{ldt}\ge\max(1,\{n})$.
	 *
	 * \param[in,out] Q
	 *     an array, dimension (§ldq,§n)\n
	 *     On entry, if §compq = 'V', the matrix $Q$ of Schur vectors.\n
	 *     On exit, if §compq = 'V', §Q has been postmultiplied by the orthogonal transformation
	 *              matrix $Z$ which reorders $T$.\n &emsp;&emsp;&emsp;&nbsp;
	 *              If §compq = 'N', §Q is not referenced.
	 *
	 * \param[in] ldq
	 *     The leading dimension of the array §Q. $\{ldq}\ge 1$, and if §compq = 'V',
	 *     $\{ldq}\ge\max(1,\{n})$.
	 *
	 * \param[in,out] ifst, ilst
	 *     Specify the reordering of the diagonal blocks of $T$. The block with row index §ifst is
	 *     moved to row §ilst, by a sequence of transpositions between adjacent blocks.\n
	 *     On exit, if §ifst pointed on entry to the second row of a 2 by 2 block, it is changed to
	 *     point to the first row; §ilst always points to the first row of the block in its final
	 *     position (which may differ from its input value by +1 or -1).\n
	 *     $0\le\{ifst}<\{n}$; $0\le\{ilst}<\{n}$.\n
	 *     NOTE: zero-based indices!
	 *
	 * \param[out] work an array, dimension (§n)
	 * \param[out] info
	 *     =0: successful exit\n
	 *     <0: if $\{info}=-i$, the $i$-th argument had an illegal value\n
	 *     =1: two adjacent blocks were too close to swap (the problem is very ill-conditioned);
	 *         $T$ may have been partially reordered, and §ilst points to the first row of the
	 *         current position of the block being moved.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void dtrexc(char const* const compq, int const n, real* const T, int const ldt,
	                   real* const Q, int const ldq, int& ifst, int& ilst, real* const work,
	                   int& info) /*const*/
	{
		// Decode and test the input arguments.
		info = 0;
		bool wantq = (std::toupper(compq[0])=='V');
		if (!wantq && std::toupper(compq[0])!='N')
		{
			info = -1;
		}
		else if (n<0)
		{
			info = -2;
		}
		else if (ldt<std::max(1, n))
		{
			info = -4;
		}
		else if (ldq<1 || (wantq && ldq<std::max(1, n)))
		{
			info = -6;
		}
		else if ((ifst<0 || ifst>=n) && (n>0))
		{
			info = -7;
		}
		else if ((ilst<0 || ilst>=n) && (n>0))
		{
			info = -8;
		}
		if (info!=0)
		{
			xerbla("DTREXC", -info);
			return;
		}
		// Quick return if possible
		if (n<=1)
		{
			return;
		}
		// Determine the first row of specified block and find out it is 1 by 1 or 2 by 2.
		if (ifst>0)
		{
			if (T[ifst+ldt*(ifst-1)]!=ZERO)
			{
				ifst--;
			}
		}
		int nbf = 1;
		if (ifst<n-1)
		{
			if (T[ifst+1+ldt*ifst]!=ZERO)
			{
				nbf = 2;
			}
		}
		// Determine the first row of the final block and find out it is 1 by 1 or 2 by 2.
		if (ilst>0)
		{
			if (T[ilst+ldt*(ilst-1)]!=ZERO)
			{
				ilst--;
			}
		}
		int nbl = 1;
		if (ilst<n-1)
		{
			if (T[ilst+1+ldt*ilst]!=ZERO)
			{
				nbl = 2;
			}
		}
		if (ifst==ilst)
		{
			return;
		}
		int here, nbnext;
		if (ifst<ilst)
		{
			// Update ilst
			if (nbf==2 && nbl==1)
			{
				ilst--;
			}
			if (nbf==1 && nbl==2)
			{
				ilst++;
			}
			here = ifst;
			do
			{
				// Swap block with next one below
				if (nbf==1 || nbf==2)
				{
					// Current block either 1 by 1 or 2 by 2
					nbnext = 1;
					if (here+nbf<n-1)
					{
						if (T[here+nbf+1+ldt*(here+nbf)]!=ZERO)
						{
							nbnext = 2;
						}
					}
					dlaexc(wantq, n, T, ldt, Q, ldq, here, nbf, nbnext, work, info);
					if (info!=0)
					{
						ilst = here;
						return;
					}
					here += nbnext;
					// Test if 2 by 2 block breaks into two 1 by 1 blocks
					if (nbf==2)
					{
						if (T[here+1+ldt*here]==ZERO)
						{
							nbf = 3;
						}
					}
				}
				else
				{
					// Current block consists of two 1 by 1 blocks each of which must be swapped
					// individually
					nbnext = 1;
					if (here+3<n)
					{
						if (T[here+3+ldt*(here+2)]!=ZERO)
						{
							nbnext = 2;
						}
					}
					dlaexc(wantq, n, T, ldt, Q, ldq, here+1, 1, nbnext, work, info);
					if (info!=0)
					{
						ilst = here;
						return;
					}
					if (nbnext==1)
					{
						// Swap two 1 by 1 blocks, no problems possible
						dlaexc(wantq, n, T, ldt, Q, ldq, here, 1, nbnext, work, info);
						here++;
					}
					else
					{
						// Recompute nbnext in case 2 by 2 split
						if (T[here+2+ldt*(here+1)]==ZERO)
						{
							nbnext = 1;
						}
						if (nbnext==2)
						{
							// 2 by 2 Block did not split
							dlaexc(wantq, n, T, ldt, Q, ldq, here, 1, nbnext, work, info);
							if (info!=0)
							{
								ilst = here;
								return;
							}
							here += 2;
						}
						else
						{
							// 2 by 2 Block did split
							dlaexc(wantq, n, T, ldt, Q, ldq, here, 1, 1, work, info);
							dlaexc(wantq, n, T, ldt, Q, ldq, here+1, 1, 1, work, info);
							here += 2;
						}
					}
				}
			} while (here<ilst);
		}
		else
		{
			here = ifst;
			do
			{
				// Swap block with next one above
				if (nbf==1 || nbf==2)
				{
					// Current block either 1 by 1 or 2 by 2
					nbnext = 1;
					if (here>=2)
					{
						if (T[here-1+ldt*(here-2)]!=ZERO)
						{
							nbnext = 2;
						}
					}
					dlaexc(wantq, n, T, ldt, Q, ldq, here-nbnext, nbnext, nbf, work, info);
					if (info!=0)
					{
						ilst = here;
						return;
					}
					here -= nbnext;
					// Test if 2 by 2 block breaks into two 1 by 1 blocks
					if (nbf==2)
					{
						if (T[here+1+ldt*here]==ZERO)
						{
							nbf = 3;
						}
					}
				}
				else
				{
					// Current block consists of two 1 by 1 blocks each of which must be swapped
					// individually
					nbnext = 1;
					if (here>=2)
					{
						if (T[here-1+ldt*(here-2)]!=ZERO)
						{
							nbnext = 2;
						}
					}
					dlaexc(wantq, n, T, ldt, Q, ldq, here-nbnext, nbnext, 1, work, info);
					if (info!=0)
					{
						ilst = here;
						return;
					}
					if (nbnext==1)
					{
						// Swap two 1 by 1 blocks, no problems possible
						dlaexc(wantq, n, T, ldt, Q, ldq, here, nbnext, 1, work, info);
						here--;
					}
					else
					{
						// Recompute nbnext in case 2 by 2 split
						if (T[here+ldt*(here-1)]==ZERO)
						{
							nbnext = 1;
						}
						if (nbnext==2)
						{
							// 2 by 2 Block did not split
							dlaexc(wantq, n, T, ldt, Q, ldq, here-1, 2, 1, work, info);
							if (info!=0)
							{
								ilst = here;
								return;
							}
							here -= 2;
						}
						else
						{
							// 2 by 2 Block did split
							dlaexc(wantq, n, T, ldt, Q, ldq, here, 1, 1, work, info);
							dlaexc(wantq, n, T, ldt, Q, ldq, here-1, 1, 1, work, info);
							here -= 2;
						}
					}
				}
			} while (here>ilst);
		}
		ilst = here;
	}

	/*! §ieeeck
	 *
	 * §ieeeck is called from the §ilaenv to verify that Infinity and possibly NaN arithmetic is
	 * safe (i.e.will not trap).
	 * \param[in] ispec
	 *     Specifies whether to test just for inifinity arithmetic or whether to test for infinity
	 *     and NaN arithmetic.\n
	 *         0: Verify infinity arithmetic only.\n
	 *         1: Verify infinity and NaN arithmetic.
	 *
	 * \param[in] Zero
	 *     Must contain the value 0.0\n
	 *     This is passed to prevent the compiler from optimizing away some code.
	 *
	 * \param[in] One
	 *     Must contain the value 1.0\n
	 *     This is passed to prevent the compiler from optimizing away some code.
	 *
	 * \return
	 *     0: Arithmetic failed to produce the correct answers\n
	 *     1: Arithmetic produced the correct answers
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static int ieeeck(int const ispec, real const Zero, real const One) /*const*/
	{
		real posinf = One / Zero;
		if (posinf<=One)
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

	/*! §iladlc scans a matrix for its last non-zero column.
	 *
	 * §iladlc scans $A$ for its last non-zero column.
	 * \param[in] m   The number of rows of the matrix $A$.
	 * \param[in] n   The number of columns of the matrix $A$.
	 * \param[in] A   an array, dimension (§lda,§n)\n The §m by §n matrix $A$.
	 * \param[in] lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \return The index of the last non-zero column.\n NOTE: zero-based index!
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static int iladlc(int const m, int const n, real const* const A, int const lda) /*const*/
	{
		int ila, lastcol=lda*(n-1);
		// Quick test for the common case where one corner is non-zero.
		if (n==0)
		{
			return -1;
		}
		else if (A[lastcol]!=ZERO || A[m-1+lastcol]!=ZERO)
		{
			return n-1;
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

	/*! §iladlr scans a matrix for its last non-zero row.
	 *
	 * §iladlr scans $A$ for its last non-zero row.
	 * \param[in] m   The number of rows of the matrix $A$.
	 * \param[in] n   The number of columns of the matrix $A$.
	 * \param[in] A   an array, dimension (§lda,§n)\n The §m by §n matrix $A$.
	 * \param[in] lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \return The index of the last non-zero row.\n NOTE: zero-based index!
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static int iladlr(int const m, int const n, real const* const A, int const lda) /*const*/
	{
		int lastrow = m - 1;
		// Quick test for the common case where one corner is non - zero.
		if (m==0)
		{
			return -1;
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
				while (i>=0 && A[i+colj]==ZERO)
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

	/*! §ilaenv
	 *
	 * §ilaenv is called from the LAPACK routines to choose problem-dependent parameters for the
	 * local environment. See §ispec for a description of the parameters.\n
	 * §ilaenv returns an integer:\n
	 *     if >=0: §ilaenv returns the value of the parameter specified by §ispec \n
	 *     if < 0: if $-k$, the $k$-th argument had an illegal value.\n
	 * This version provides a set of parameters which should give good, but not optimal,
	 * performance on many of the currently available computers. Users are encouraged to modify
	 * this subroutine to set the tuning parameters for their particular machine using the option
	 * and problem size information in the arguments.\n
	 * This routine will not function correctly if it is converted to all lower case. Converting it
	 * to all upper case is allowed.
	 * \param[in] ispec
	 *     Specifies the parameter to be returned.\n
	 *     \li 1: the optimal blocksize; if this value is 1, an unblocked algorithm will give the
	 *            best performance.
	 *     \li 2: the minimum block size for which the block routine should be used; if the usable
	 *            block size is less than this value, an unblocked routine should be used.
	 *     \li 3: the crossover point (in a block routine, for §n less than this value, an
	 *            unblocked routine should be used)
	 *     \li 4: the number of shifts, used in the nonsymmetric eigenvalue routines (DEPRECATED)
	 *     \li 5: the minimum column dimension for blocking to be used; rectangular blocks must
	 *            have dimension at least §k by §m, where §k is given by §ilaenv(2, ...) and §m by
	 *            §ilaenv(5, ...)
	 *     \li 6: the crossover point for the SVD (when reducing an §m by §n matrix to bidiagonal
	 *            form, if $\max(\{m},\{n})/\min(\{m},\{n})$ exceeds this value, a QR factorization
	 *            is used first to reduce the matrix to triangular form.)
	 *     \li 7: the number of processors
	 *     \li 8: the crossover point for the multishift QR method for nonsymmetric eigenvalue
	 *            problems (DEPRECATED)
	 *     \li 9: maximum size of the subproblems at the bottom of the computation tree in the
	 *            divide-and-conquer algorithm (used by §xgelsd and §xgesdd)
	 *     \li 10: ieee NaN arithmetic can be trusted not to trap
	 *     \li 11: infinity arithmetic can be trusted not to trap
	 *     \li $12\le\{ispec}\le 16$: §xhseqr or related subroutines, see §iparmq for detailed
	 *                                explanation
	 *
	 * \param[in] name The name of the calling subroutine, in either upper case or lower case.
	 * \param[in] opts
	 *     The character options to the subroutine name, concatenated into a single character
	 *     string. For example, §uplo = 'U', §trans = 'T', and §diag = 'N' for a triangular routine
	 *     would be specified as §opts = 'UTN'.
	 *
	 * \param[in] n1, n2, n3, n4
	 *     Problem dimensions for the subroutine name; these may not all be required.
	 *
	 * \return The requested parameter.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date November 2017
	 * \remark
	 *     The following conventions have been used when calling §ilaenv from the LAPACK routines:
	 *     \li 1) §opts is a concatenation of all of the character options to subroutine §name, in
	 *            the same order that they appear in the argument list for §name, even if they are
	 *            not used in determining the value of the parameter specified by §ispec.
	 *     \li 2) The problem dimensions §n1, §n2, §n3, §n4 are specified in the order that they
	 *            appear in the argument list for §name. §n1 is used first, §n2 second, and so on,
	 *            and unused problem dimensions are passed a value of -1.
	 *     \li 3) The parameter value returned by §ilaenv is checked for validity in the calling
	 *            subroutine. For example, §ilaenv is used to retrieve the optimal blocksize for
	 *            §strtri as follows:\n
	 *                §nb = §ilaenv(1, 'STRTRI', §uplo[0]+§diag[0], §n, -1, -1, -1);\n
	 *                if ($\{nb}\le 1$) $\{nb} = \max(1,\{n})$;                                  */
	static int ilaenv(int const ispec, char const* const name, char const* const opts,
	                  int const n1, int const n2, int const n3, int const n4) /*const*/
	{
		bool sname, cname;
		char c1;
		switch (ispec)
		{
			case 1:
			case 2:
			case 3:
				{
					// Convert name to upper case.
					char subnam[12];
					std::strncpy(subnam, name, 12);
					int nb, nbmin;
					for (nb=0; nb<11; nb++)
					{
						subnam[nb] = std::toupper(subnam[nb]);
					}
					subnam[11] = '\0';
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
					bool twostage = (std::strlen(subnam)>=11 && subnam[10]=='2');
					switch (ispec)
					{
						case 1:
							// ispec = 1: block size
							// In these examples, separate code is provided for setting nb for real
							// and complex. We assume that nb will take the same value in single or
							// double precision.
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
								else if (std::strncmp(c3, "QRF", 3)==0
								      || std::strncmp(c3, "RQF", 3)==0
								      || std::strncmp(c3, "LQF", 3)==0
								      || std::strncmp(c3, "QLF", 3)==0)
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
								else if (std::strncmp(c3, "QR ", 3))
								{
									if (n3==1)
									{
										if (sname)
										{
											//m*n
											if (n1*n2<=131072 || n1<=8192)
											{
												nb = n1;
											}
											else
											{
												nb = 32768 / n2;
											}
										}
										else
										{
											if (n1*n2<=131072 || n1<=8192)
											{
												nb=n1;
											}
											else
											{
												nb = 32768 / n2;
											}
										}
									}
									else
									{
										if (sname)
										{
											nb = 1;
										}
										else
										{
											nb = 1;
										}
									}
								}
								else if (std::strncmp(c3, "LQ ", 3))
								{
									if (n3==2)
									{
										if (sname)
										{
											//m*n
											if (n1*n2<=131072 || n1<=8192)
											{
												nb = n1;
											}
											else
											{
												nb = 32768 / n2;
											}
										}
										else
										{
											if (n1*n2<=131072 || n1<=8192)
											{
												nb = n1;
											}
											else
											{
												nb = 32768 / n2;
											}
										}
									}
									else
									{
										if (sname)
										{
											nb = 1;
										}
										else
										{
											nb = 1;
										}
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
										if (twostage)
										{
											nb = 192;
										}
										else
										{
											nb = 64;
										}
									}
									else
									{
										if (twostage)
										{
											nb = 192;
										}
										else
										{
											nb = 64;
										}
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
									if (twostage)
									{
										nb = 192;
									}
									else
									{
										nb = 64;
									}
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
									if (std::strncmp(c4, "QR", 2)==0
									 || std::strncmp(c4, "RQ", 2)==0
									 || std::strncmp(c4, "LQ", 2)==0
									 || std::strncmp(c4, "QL", 2)==0
									 || std::strncmp(c4, "HR", 2)==0
									 || std::strncmp(c4, "TR", 2)==0
									 || std::strncmp(c4, "BR", 2)==0)
									{
										nb = 32;
									}
								}
								else if (c3[0]=='M')
								{
									if (std::strncmp(c4, "QR", 2)==0
									 || std::strncmp(c4, "RQ", 2)==0
									 || std::strncmp(c4, "LQ", 2)==0
									 || std::strncmp(c4, "QL", 2)==0
									 || std::strncmp(c4, "HR", 2)==0
									 || std::strncmp(c4, "TR", 2)==0
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
									if (std::strncmp(c4, "QR", 2)==0
									 || std::strncmp(c4, "RQ", 2)==0
									 || std::strncmp(c4, "LQ", 2)==0
									 || std::strncmp(c4, "QL", 2)==0
									 || std::strncmp(c4, "HR", 2)==0
									 || std::strncmp(c4, "TR", 2)==0
									 || std::strncmp(c4, "BR", 2)==0)
									{
										nb = 32;
									}
								}
								else if (c3[0]=='M')
								{
									if (std::strncmp(c4, "QR", 2)==0
									 || std::strncmp(c4, "RQ", 2)==0
									 || std::strncmp(c4, "LQ", 2)==0
									 || std::strncmp(c4, "QL", 2)==0
									 || std::strncmp(c4, "HR", 2)==0
									 || std::strncmp(c4, "TR", 2)==0
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
								else if (std::strncmp(c3, "EVC", 3)==0)
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
							else if (std::strncmp(c2, "GG", 2)==0)
							{
								nb = 32;
								if (std::strncmp(c3, "HD3", 3)==0)
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
									if (std::strncmp(c4, "QR", 2)==0
									 || std::strncmp(c4, "RQ", 2)==0
									 || std::strncmp(c4, "LQ", 2)==0
									 || std::strncmp(c4, "QL", 2)==0
									 || std::strncmp(c4, "HR", 2)==0
									 || std::strncmp(c4, "TR", 2)==0
									 || std::strncmp(c4, "BR", 2)==0)
									{
										nbmin = 2;
									}
								}
								else if (c3[0]=='M')
								{
									if (std::strncmp(c4, "QR", 2)==0
									 || std::strncmp(c4, "RQ", 2)==0
									 || std::strncmp(c4, "LQ", 2)==0
									 || std::strncmp(c4, "QL", 2)==0
									 || std::strncmp(c4, "HR", 2)==0
									 || std::strncmp(c4, "TR", 2)==0
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
									if (std::strncmp(c4, "QR", 2)==0
									 || std::strncmp(c4, "RQ", 2)==0
									 || std::strncmp(c4, "LQ", 2)==0
									 || std::strncmp(c4, "QL", 2)==0
									 || std::strncmp(c4, "HR", 2)==0
									 || std::strncmp(c4, "TR", 2)==0
									 || std::strncmp(c4, "BR", 2)==0)
									{
										nbmin = 2;
									}
								}
								else if (c3[0]=='M')
								{
									if (std::strncmp(c4, "QR", 2)==0
									 || std::strncmp(c4, "RQ", 2)==0
									 || std::strncmp(c4, "LQ", 2)==0
									 || std::strncmp(c4, "QL", 2)==0
									 || std::strncmp(c4, "HR", 2)==0
									 || std::strncmp(c4, "TR", 2)==0
									 || std::strncmp(c4, "BR", 2)==0)
									{
										nbmin = 2;
									}
								}
							}
							else if (std::strncmp(c2, "GG", 2)==0)
							{
								nbmin = 2;
								if (std::strncmp(c3, "HD3", 3)==0)
								{
									nbmin = 2;
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
									if (std::strncmp(c4, "QR", 2)==0
									 || std::strncmp(c4, "RQ", 2)==0
									 || std::strncmp(c4, "LQ", 2)==0
									 || std::strncmp(c4, "QL", 2)==0
									 || std::strncmp(c4, "HR", 2)==0
									 || std::strncmp(c4, "TR", 2)==0
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
									if (std::strncmp(c4, "QR", 2)==0
									 || std::strncmp(c4, "RQ", 2)==0
									 || std::strncmp(c4, "LQ", 2)==0
									 || std::strncmp(c4, "QL", 2)==0
									 || std::strncmp(c4, "HR", 2)==0
									 || std::strncmp(c4, "TR", 2)==0
									 || std::strncmp(c4, "BR", 2)==0)
									{
										nx = 128;
									}
								}
							}
							return nx;
							break;
					}
				}
				break;
			case 4:
				// ispec = 4: number of shifts (used by xhseqr)
				return 6;
				break;
			case 5:
				// ispec = 5: minimum column dimension (not used)
				return 2;
				break;
			case 6:
				// ispec = 6: crossover point for SVD (used by xgelss and xgesvd)
				return int(real(std::min(n1, n2)) * real(1.6));
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
				// ispec = 9: maximum size of the subproblems at the bottom of the computation tree
				//            in the divide-and-conquer algorithm (used by xgelsd and xgesdd)
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
				// 12 <= ispec <= 16: xhseqr or related subroutines.
				return iparmq(ispec, name, opts, n1, n2, n3, n4);
				break;
			default:
				// Invalid value for ispec
				return -1;
		}
		return -1;
	}

	/*! §iparam2stage
	 *
	 * This program sets problem and machine dependent parameters useful for §xhetrd_2stage,
	 * §xhetrd_h@2hb, §xhetrd_hb2st, §xgebrd_2stage, §xgebrd_ge2gb, §xgebrd_gb2bd  and related
	 * subroutines for eigenvalue problems.\n
	 * It is called whenever §ilaenv is called with $17\le\{ispec}\le 21$.\n
	 * It is called whenever §ilaenv2stage is called with $1\le\{ispec}\le 5$ with a direct
	 * conversion $\{ispec}+16$.
	 * \param[in] ispec
	 *     ispec specifies which tunable parameter §iparam2stage should return.
	 *     17. the optimal blocksize §nb for the reduction to band
	 *     18. the optimal blocksize §ib for the eigenvectors singular vectors update routine
	 *     19. The length of the array that store the Housholder  representation for the second
	 *         stage Band to Tridiagonal or Bidiagonal
	 *     20. The workspace needed for the routine in input.
	 *     21. For future release.
	 *
	 * \param[in] name Name of the calling subroutine
	 * \param[in] opts
	 *     The character options to the subroutine §name, concatenated into a single character
	 *     string. For example, §UPLO ="U", §TRANS ="T", and §DIAG ="N" for a triangular routine
	 *     would be specified as §opts ="UTN".
	 *
	 * \param[in] ni  the size of the matrix
	 * \param[in] nbi
	 *     used in the reduciton, (e.g., the size of the band),
	 *     needed to compute workspace and §lhous2.
	 *
	 * \param[in] ibi
	 *     represents the §ib of the reduciton, needed to compute workspace and §lhous2.
	 *
	 * \param[in] nxi needed in the future release.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016
	 * \remark
	 *     Implemented by Azzam Haidar.\n\n
	 *     All details are available in technical report, SC11, SC13 papers.\n\n
	 *         Azzam Haidar, Hatem Ltaief, and Jack Dongarra.\n
	 *         Parallel reduction to condensed forms for symmetric eigenvalue problems using
	 *         aggregated fine-grained and memory-aware kernels. In Proceedings of 2011
	 *         International Conference for High Performance Computing, Networking, Storage and
	 *         Analysis (SC '11), New York, NY, USA, Article 8 , 11 pages.\n
	 *         http://doi.acm.org/10.1145/2063384.2063394 \n\n
	 *         A. Haidar, J. Kurzak, P. Luszczek, 2013.\n
	 *         An improved parallel singular value algorithm and its implementation  for multicore
	 *         hardware, In Proceedings of 2013 International Conference for High Performance
	 *         Computing, Networking, Storage and Analysis (SC '13). Denver, Colorado, USA, 2013.
	 *         Article 90, 12 pages.\n
	 *         http://doi.acm.org/10.1145/2503210.2503292 \n\n
	 *         A. Haidar, R. Solca, S. Tomov, T. Schulthess and J. Dongarra.\n
	 *         A novel hybrid CPU-GPU generalized eigensolver for electronic structure calculations
	 *         based on fine-grained memory aware tasks. International Journal of High Performance
	 *         Computing Applications. Volume 28 Issue 2, Pages 196-209, May 2014.\n
	 *         http://hpc.sagepub.com/content/28/2/196                                           */
	static int iparam2stage(int const ispec, char const* const name, char const* const opts,
	                        int const ni, int const nbi, int const ibi, int const nxi) /*const*/
	{
//#if defined(_OPENMP)
//		use omp_lib
//#endif
		// Invalid value for ispec
		if (ispec<17||ispec>21)
		{
			return -1;
		}
		// Get the number of threads
		int nthreads = 1;
//#if defined(_OPENMP)
//		nthreads = OMP_GET_NUM_THREADS()
//#endif
		bool cprec = false;
		char prec, algo[3], stag[5], subnam[13];
		if (ispec != 19)
		{
			// Convert name to upper case
			for (int i=0; i<12; i++)
			{
				subnam[i] = std::toupper(name[i]);
			}
			subnam[12] = '\0';
			prec  = subnam[0];
			std::strncpy(algo, &subnam[3], 3);
			std::strncpy(stag, &subnam[7], 5);
			cprec = (prec=='C' || prec=='Z');
			// Invalid value for PRECISION
			if (!(prec=='S' || prec=='D' || cprec))
			{
				return -1;
			}
		}
		if (ispec == 17 || ispec == 18)
		{
			// ispec = 17, 18: block size kd, ib
			// Could be also dependent from N but for now it depend only on sequential or parallel
			int kd, ib;
			if (nthreads>4)
			{
				if (cprec)
				{
					kd = 128;
					ib = 32;
				}
				else
				{
					kd = 160;
					ib = 40;
				}
			}
			else if (nthreads>1)
			{
				if (cprec)
				{
					kd = 64;
					ib = 32;
				}
				else
				{
					kd = 64;
					ib = 32;
				}
			}
			else
			{
				if (cprec)
				{
					kd = 16;
					ib = 16;
				}
				else
				{
					kd = 32;
					ib = 16;
				}
			}
			if (ispec==17)
			{
				return kd;
			}
			if (ispec==18)
			{
				return ib;
			}
		}
		else if (ispec == 19)
		{
			// ispec = 19:
			// lhous length of the Houselholder representation matrix (V,T) of the second stage.
			// should be >= 1. Will add the vect OPTION HERE next release
			int lhous;
			char vect  = opts[0];
			if (vect=='N')
			{
				lhous = std::max(1, 4*ni);
			}
			else
			{
				// This is not correct, it need to call the algo and the stage2
				lhous = std::max(1, 4*ni) + ibi;
			}
			if (lhous>=0)
			{
				return lhous;
			}
			else
			{
				return -1;
			}
		}
		else if (ispec == 20)
		{
			// ispec = 20: (21 for future use)
			// lwork length of the workspace for  either or both stages for TRD and BRD.
			// should be >= 1.
			// TRD:
			// TRD_stage 1: = LT + LW + LS1 + LS2 = LDT*kd + N*kd + N*max(kd,factoptnb) + LDS2*kd
			// where LDT=LDS2=kd = N*kd + N*max(kd,factoptnb) + 2*kd*kd
			// TRD_stage 2: = (2NB+1)*N + kd*nthreads
			// TRD_both   : = max(stage1,stage2) + AB (AB=(kd+1)*N)
			//              = N*kd + N*max(kd+1,factoptnb) + max(2*kd*kd, kd*nthreads) + (kd+1)*N
			int lwork = -1;
			subnam[0] = prec;
			std::strncpy(&subnam[1], "GEQRF", 6);
			int qroptnb = ilaenv(1, subnam, " ", ni, nbi, -1, -1);
			std::strncpy(&subnam[1], "GELQF", 6);
			int lqoptnb = ilaenv(1, subnam, " ", nbi, ni, -1, -1);
			// Could be QR or LQ for TRD and the max for BRD
			int factoptnb = std::max(qroptnb, lqoptnb);
			if (std::strncmp(algo, "TRD", 3)==0)
			{
				if (std::strncmp(stag, "2STAG", 5)==0)
				{
					lwork = ni*nbi + ni*std::max(nbi+1,factoptnb)
					        + std::max(2*nbi*nbi, nbi*nthreads) + (nbi+1)*ni;
				}
				else if (std::strncmp(stag, "HE2HB", 5)==0 || std::strncmp(stag, "SY2SB", 5)==0)
				{
					lwork = ni*nbi + ni*std::max(nbi,factoptnb) + 2*nbi*nbi;
				}
				else if (std::strncmp(stag, "HB2ST", 5)==0 || std::strncmp(stag, "SB2ST", 5)==0)
				{
					lwork = (2*nbi+1)*ni + nbi*nthreads;
				}
			}
			else if (std::strncmp(algo, "BRD", 3)==0)
			{
				if (std::strncmp(stag, "2STAG", 5)==0)
				{
					lwork = 2*ni*nbi + ni*std::max(nbi+1,factoptnb)
					        + std::max(2*nbi*nbi, nbi*nthreads) + (nbi+1)*ni;
				}
				else if (std::strncmp(stag, "GE2GB", 5)==0)
				{
					lwork = ni*nbi + ni*std::max(nbi,factoptnb) + 2*nbi*nbi;
				}
				else if (std::strncmp(stag, "GB2BD", 5)==0)
				{
					lwork = (3*nbi+1)*ni + nbi*nthreads;
				}
			}
			lwork = std::max(1, lwork);
			if (lwork>0)
			{
				return lwork;
			}
			else
			{
				return -1;
			}
		}
		else if (ispec == 21)
		{
			// ispec = 21 for future use
			return nxi;
		}
		return -1;
	}

	/*! §iparmq
	 *
	 * This program sets problem and machine dependent parameters useful for §xhseqr and related
	 * subroutines for eigenvalue problems.
	 * It is called whenever §ilaenv is called with $12\le\{ispec}\le 16$
	 * \param[in] ispec
	 *     specifies which tunable parameter §iparmq should return.
	 *     \li 12: (inmin) Matrices of order §nmin or less are sent directly to §xlahqr, the
	 *                     implicit double shift QR algorithm. §nmin must be at least 11.
	 *     \li 13: (inwin) Size of the deflation window. This is best set greater than or equal to
	 *                     the number of simultaneous shifts §ns. Larger matrices benefit from
	 *                     larger deflation windows.
	 *     \li 14: (inibl) Determines when to stop nibbling and invest in an (expensive)
	 *                     multi-shift QR sweep. If the aggressive early deflation subroutine finds
	 *                     §ld converged eigenvalues from an order §nw deflation window and
	 *                     $\{ld}>(\{nw}\ \{nibble})/100$, then the next QR sweep is skipped and
	 *                     early deflation is applied immediately to the remaining active diagonal
	 *                     block. Setting §iparmq(§ispec =14)=0 causes §ttqre to skip a multi-shift
	 *                     QR sweep whenever early deflation finds a converged eigenvalue. Setting
	 *                     §iparmq(§ispec =14) greater than or equal to 100 prevents §ttqre from
	 *                     skipping a multi-shift QR sweep.
	 *     \li 15: (nshfts) The number of simultaneous shifts in a multi-shift QR iteration.
	 *     \li 16: (iacc22) §iparmq is set to 0, 1 or 2 with the following meanings.\n
	 *                      0: During the multi-shift QR/QZ sweep, blocked eigenvalue reordering,
	 *                         blocked Hessenberg-triangular reduction, reflections and/or
	 *                         rotations are not accumulated when updating the far-from-diagonal
	 *                         matrix entries.\n
	 *                      1: During the multi-shift QR/QZ sweep, blocked eigenvalue reordering,
	 *                         blocked Hessenberg-triangular reduction, reflections and/or
	 *                         rotations are accumulated, and matrix-matrix multiplication is used
	 *                         to update the far-from-diagonal matrix entries.\n
	 *                      2: During the multi-shift QR/QZ sweep, blocked eigenvalue reordering,
	 *                         blocked Hessenberg-triangular reduction, reflections and/or
	 *                         rotations are accumulated, and 2 by 2 block structure is exploited
	 *                         during matrix-matrix multiplies.\n
	 *                      (If §xtrmm is slower than §xgemm, then §iparmq(§ispec =16)=1 may be
	 *                       more efficient than §iparmq(§ispec =16)=2 despite the greater level of
	 *                       arithmetic work implied by the latter choice.)
	 *
	 * \param[in] name Name of the calling subroutine
	 * \param[in] opts This is a concatenation of the string arguments to §ttqre.
	 * \param[in] n    §n is the order of the Hessenberg matrix $H$.
	 * \param[in] ilo, ihi
	 *     It is assumed that $H$ is already upper triangular in rows and columns $1:\{ilo}-1$ and
	 *     $\{ihi}+1:\{n}$.
	 *
	 * \param[in] lwork The amount of workspace available.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2017
	 * \remark
	 *     Little is known about how best to choose these parameters. It is possible to use
	 *     different values of the parameters for each of §chseqr, §dhseqr, §shseqr and §zhseqr.\n
	 *
	 *     It is probably best to choose different parameters for different matrices and different
	 *     parameters at different times during the iteration, but this has not been implemented
	 *     yet.\n
	 *
	 *     The best choices of most of the parameters depend in an ill-understood way on the
	 *     relative execution rate of §xlaqr3 and §xlaqr5 and on the nature of each particular
	 *     eigenvalue problem. Experiment may be the only practical way to determine which choices
	 *     are most effective.\n
	 *
	 *     Following is a list of default values supplied by §iparmq. These defaults may be
	 *     adjusted in order to attain better performance in any particular computational
	 *     environment.\n
	 *     \f[\begin{tabular}{rl}
	 *         \(\{iparmq}\)(\(\{ispec}\) = 12)
	 *             &The \(\{xlahqr}\) vs \(\{xlaqr0}\) crossover point.\\
	 *             &Default: 75. (Must be at least 11.)\\
	 *         \(\{iparmq}\)(\(\{ispec}\) = 13) &
	 *             Recommended deflation window size.\\
	 *             &This depends on \(\{ilo}\), \(\{ihi}\) and \(\{ns}\), the number of\\
	 *             &simultaneous shifts returned by\\
	 *             &\(\{iparmq}\)(\(\{ispec}\) = 15).\\
	 *             &The default for \((\{ihi}-\{ilo}+1)\le 500\) is \(\{ns}\).\\
	 *             & The default for \((\{ihi}-\{ilo}+1)>500\) is \(3\{ns}/2\).\\
	 *         \(\{iparmq}\)(\(\{ispec}\) = 14) & Nibble crossover point. Default: 14.\\
	 *         \(\{iparmq}\)(\(\{ispec}\) = 15)
	 *             &Number of simultaneous shifts, \(\{ns}\).\\
	 *             &a multi-shift QR iteration. If \(\{ihi}-\{ilo}+1\) is ...\\
	 *             &\begin{tabular}{rrr}
	 *                 \(\ge\)... & but less than... & the default is \\
	 *                          0 &               30 &        ns = 2+ \\
	 *                         30 &               60 &        ns = 4+ \\
	 *                         60 &              150 &        ns = 10 \\
	 *                        150 &              590 &        ns = ** \\
	 *                        590 &             3000 &        ns = 64 \\
	 *                       3000 &             6000 &       ns = 128 \\
	 *                       6000 &         infinity &       ns = 256
	 *             \end{tabular}\\
	 *             &(+) By default matrices of this order are passed\\
	 *                  &~~~~~~to the implicit double shift routine \(\{xlahqr}\).\\
	 *                  &~~~~~~See \(\{iparmq}\)(\(\{ispec}\) =12) above.\\
	 *                  &~~~~~~These values of \(\{ns}\) are used only in case of\\
	 *                  &~~~~~~a rare \(\{xlahqr}\) failure.\\
	 *             &(**) an ad-hoc function increasing from 10 to 64.\\
	 *         \(\{iparmq}\)(\(\{ispec}\) = 16)
	 *             &Select structured matrix multiply.\\
	 *             &(See \(\{ispec}\) = 16 above for details.) Default: 3.
	 *     \end{tabular}\f]                                                                      */
	static int iparmq(int const ispec, char const* const name, char const* const opts, int const n,
	                  int const ilo, int const ihi, int const lwork) /*const*/
	{
		const int INMIN=12, INWIN=13, INIBL=14, ISHFTS=15, IACC22=16, NMIN=75, K22MIN=14,
				  KACMIN=14, NIBBLE=14, KNWSWP=500;
		int nh = 0, ns = 0;
		if (ispec==ISHFTS || ispec==INWIN || ispec==IACC22)
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
				ns = std::max(10, nh / int(std::round(std::log(real(nh))/std::log(TWO))));
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
			// Matrices of order smaller than nmin get sent to xlahqr, the classic double shift
			// algorithm. This must be at least 11.
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
			// nshfts: The number of simultaneous shifts
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
			 * elements and whether to use 2 by 2 block structure while doing it. A small amount of
			 * work could be saved by making this choice dependent also upon the nh = ihi-ilo+1. */
			// Convert name to upper case
			char subnam[7];
			for (int i=0; i<6; i++)
			{
				subnam[i] = std::toupper(name[i]);
			}
			subnam[6] = '\0';
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

	/*! §xerbla
	 *
	 * §xerbla is an error handler for the LAPACK routines.
	 * It is called by a LAPACK routine if an input parameter has an invalid value.
	 * A message is printed and an exception is thrown.\n
	 * Installers may consider modifying the throw statement in order to call
	 * system-specific exception-handling facilities.
	 * \param[in] srname The name of the routine which called §xerbla.
	 * \param[in] info
	 *     The position of the invalid parameter in the parameter list of the calling routine.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	static void xerbla(char const* const srname, int const info) /*const*/
	{
		std::cerr << "On entry to " << srname << " parameter number " << info
				  << " had an illegal value.";
		throw info;
	}
};
#endif