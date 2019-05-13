#ifndef TESTING_EIG_HEADER
#define TESTING_EIG_HEADER

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

/*!\class Testing_eig
 * \brief A template class containing LAPACK eigenvalue testing routines.
 * Testing_eig contains the LAPACK routines for testing eigenvalue-related routines.
 * The template type is meant to be double, but can be any floating point type                   */
template<class real>
class Testing_eig : public Lapack_dyn<real>
{
private:
	// constants

	const real ZERO = real(0.0); //!< A constant zero (0.0) value
	const real ONE  = real(1.0); //!< A constant one  (1.0) value

public:
	virtual ~Testing_eig(){}

	// LAPACK TESTING EIG (alphabetically)

	/*! §alasum
	 *
	 * §alasum prints a summary of results from one of the §-chk- routines.
	 * \param[in] type  The LAPACK path name.
	 * \param[in] nout  The output stream to which results are to be printed.
	 * \param[in] nfail The number of tests which did not pass the threshold ratio.
	 * \param[in] nrun  The total number of tests.
	 * \param[in] nerrs The number of error messages recorded.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	void alasum(char const* const type, std::ostream& nout, int const nfail, int const nrun,
	            int const nerrs) const
	{
		char typecopy[4];
		std::strncpy(typecopy, type, 3);
		typecopy[3] = '\0';
		if (nfail>0)
		{
			nout << ' ' << typecopy << ": " << std::setw(6) << nfail << " out of " << std::setw(6)
			     << nrun << " tests failed to pass the threshold\n";
		}
		else
		{
			nout << "\n All tests for " << typecopy << " routines passed the threshold ("
			     << std::setw(6) << nrun << " tests run)\n";
		}
		if (nerrs>0)
		{
			nout << "      " << std::setw(6) << nerrs << " error messages recorded\n";
		}
		nout.flush();
	}

	/*! §dbdt01
	 *
	 * §dbdt01 reconstructs a general matrix $A$ from its bidiagonal form\n
	 *     $A = Q B P^T$\n
	 * where $Q$ (§m by $\min(\{m},\{n})$) and $P^T$ ($\min(\{m},\{n})$ by §n) are orthogonal
	 * matrices and $B$ is bidiagonal.\n
	 * The test ratio to test the reduction is\n
	 *     $\{resid} = \frac{\|A - Q B P^T\|}{n \|A\| \{eps}}$\n
	 * where §eps is the machine precision.
	 * \param[in] m  The number of rows of the matrices $A$ and $Q$.
	 * \param[in] n  The number of columns of the matrices $A$ and $P^T$.
	 * \param[in] kd
	 *     If $\{kd}=0$, $B$ is diagonal and the array §e is not referenced.\n
	 *     If $\{kd}=1$, the reduction was performed by §xgebrd; $B$ is upper bidiagonal if
	 *                   $\{m}\ge\{n}$, and lower bidiagonal if $\{m}<\{n}$.\n
	 *     If $\{kd}=-1$, the reduction was performed by §xgbbrd; $B$ is always upper bidiagonal.
	 *
	 * \param[in] A   an array, dimension (§lda,§n)\n The §m by §n matrix $A$.
	 * \param[in] lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{m})$.
	 * \param[in] Q
	 *     an array, dimension (§ldq,§n)\n
	 *     The §m by $\min(\{m},\{n})$ orthogonal matrix $Q$ in the reduction $A = Q B P^T$.
	 *
	 * \param[in] ldq The leading dimension of the array §Q. $\{ldq}\ge\max(1,\{m})$.
	 * \param[in] d
	 *     an array, dimension ($\min(\{m},\{n})$)\n
	 *     The diagonal elements of the bidiagonal matrix $B$.
	 *
	 * \param[in] e
	 *     an array, dimension ($\min(\{m},\{n})-1$)\n
	 *     The superdiagonal elements of the bidiagonal matrix $B$ if $\{m}\ge\{n}$, or the
	 *     subdiagonal elements of $B$ if $\{m}<\{n}$.
	 *
	 * \param[in] Pt
	 *     an array, dimension (§ldpt,§n)\n
	 *     The $\min(\{m},\{n})$ by §n orthogonal matrix $P^T$ in the reduction $A = Q B P^T$.
	 *
	 * \param[in] ldpt
	 *     The leading dimension of the array §Pt.\n $\{ldpt}\ge\max(1,\min(\{m},\{n}))$.
	 *
	 * \param[out] work an array, dimension ($\{m}+\{n}$)
	 * \param[out] resid The test ratio: $\frac{\|A - Q B P^T\|}{n \|A\| \{eps}}$
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	void dbdt01(int const m, int const n, int const kd, real const* const A, int const lda,
	            real const* const Q, int const ldq, real const* const d, real const* const e,
	            real const* const Pt, int const ldpt, real* const work, real& resid) const
	{
		// Quick return if possible
		if (m<=0 || n<=0)
		{
			resid = ZERO;
			return;
		}
		// Compute A - Q * B * P^T one column at a time.
		resid = ZERO;
		int i, j;
		if (kd!=0)
		{
			// B is bidiagonal.
			if (kd!=0 && m>=n)
			{
				// B is upper bidiagonal and m >= n.
				for (j=0; j<n; j++)
				{
					Blas<real>::dcopy(m, &A[lda*j], 1, work, 1);
					for (i=0; i<n-1; i++)
					{
						work[m+i] = d[i]*Pt[i+ldpt*j] + e[i]*Pt[i+1+ldpt*j];
					}
					work[m+n-1] = d[n-1] * Pt[n-1+ldpt*j];
					Blas<real>::dgemv("No transpose", m, n, -ONE, Q, ldq, &work[m], 1, ONE, work,
					                  1);
					resid = std::max(resid, Blas<real>::dasum(m, work, 1));
				}
			}
			else if (kd<0)
			{
				// B is upper bidiagonal and m < n.
				for (j=0; j<n; j++)
				{
					Blas<real>::dcopy(m, &A[lda*j], 1, work, 1);
					for (i=0; i<m-1; i++)
					{
						work[m+i] = d[i]*Pt[i+ldpt*j] + e[i]*Pt[i+1+ldpt*j];
					}
					work[m+m-1] = d[m-1] * Pt[m-1+ldpt*j];
					Blas<real>::dgemv("No transpose", m, m, -ONE, Q, ldq, &work[m], 1, ONE, work,
					                  1);
					resid = std::max(resid, Blas<real>::dasum(m, work, 1));
				}
			}
			else
			{
				// B is lower bidiagonal.
				for (j=0; j<n; j++)
				{
					Blas<real>::dcopy(m, &A[lda*j], 1, work, 1);
					work[m] = d[0] * Pt[ldpt*j];
					for (i=1; i<m; i++)
					{
						work[m+i] = e[i-1]*Pt[i-1+ldpt*j] + d[i]*Pt[i+ldpt*j];
					}
					Blas<real>::dgemv("No transpose", m, m, -ONE, Q, ldq, &work[m], 1, ONE, work,
					                  1);
					resid = std::max(resid, Blas<real>::dasum(m, work, 1));
				}
			}
		}
		else
		{
			// B is diagonal.
			if (m>=n)
			{
				for (j=0; j<n; j++)
				{
					Blas<real>::dcopy(m, &A[lda*j], 1, work, 1);
					for (i=0; i<n; i++)
					{
						work[m+i] = d[i] * Pt[i+ldpt*j];
					}
					Blas<real>::dgemv("No transpose", m, n, -ONE, Q, ldq, &work[m], 1, ONE, work,
					                  1);
					resid = std::max(resid, Blas<real>::dasum(m, work, 1));
				}
			}
			else
			{
				for (j=0; j<n; j++)
				{
					Blas<real>::dcopy(m, &A[lda*j], 1, work, 1);
					for (i=0; i<m; i++)
					{
						work[m+i] = d[i] * Pt[i+ldpt*j];
					}
					Blas<real>::dgemv("No transpose", m, m, -ONE, Q, ldq, &work[m], 1, ONE, work,
					                  1);
					resid = std::max(resid, Blas<real>::dasum(m, work, 1));
				}
			}
		}
		// Compute norm(A - Q * B * P^T) / (n * norm(A) * eps)
		real anorm = this->dlange("1", m, n, A, lda, work);
		real eps = this->dlamch("Precision");
		if (anorm<=ZERO)
		{
			if (resid!=ZERO)
			{
				resid = ONE / eps;
			}
		}
		else
		{
			if (anorm>=resid)
			{
				resid = (resid/anorm) / (real(n)*eps);
			}
			else
			{
				if (anorm<ONE)
				{
					resid = (std::min(resid, real(n)*anorm)/anorm) / (real(n)*eps);
				}
				else
				{
					resid = std::min(resid/anorm, real(n)) / (real(n)*eps);
				}
			}
		}
	}

	/*! §dbdt02
	 *
	 * §dbdt02 tests the change of basis $C = U^T B$ by computing the residual\n
	 *     $\{resid} = \frac{\|B - U C\|}{\max(\{m},\{n}) \|B\| \{eps}}$,\n
	 * where $B$ and $C$ are §m by §n matrices, $U$ is an §m by §m orthogonal matrix, and §eps is
	 * the machine precision.
	 * \param[in] m
	 *     The number of rows of the matrices $B$ and $C$ and the order of the matrix $Q$.
	 *
	 * \param[in] n   The number of columns of the matrices $B$ and $C$.
	 * \param[in] B   an array, dimension (§ldb,§n)\n The §m by §n matrix $B$.
	 * \param[in] ldb The leading dimension of the array §B. $\{ldb}\ge\max(1,\{m})$.
	 * \param[in] C
	 *     an array, dimension (§ldc,§n)\n The §m by §n matrix $C$, assumed to contain $U^T B$.
	 *
	 * \param[in]  ldc   The leading dimension of the array §C. $\{ldc}\ge\max(1,\{m})$.
	 * \param[in]  U     an array, dimension (§ldu,§m)\n The §m by §m orthogonal matrix $U$.
	 * \param[in]  ldu   The leading dimension of the array §U. $\{ldu}\ge\max(1,\{m})$.
	 * \param[out] work  an array, dimension (§m)
	 * \param[out] resid $\{resid}=\frac{\|B - U C\|}{\max(\{m},\{n}) \|B\| \{eps}}$.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	void dbdt02(int const m, int const n, real const* const B, int const ldb, real const* const C,
	            int const ldc, real const* const U, int const ldu, real* const work, real& resid)
	            const
	{
		// Quick return if possible
		resid = ZERO;
		if (m<=0 || n<=0)
		{
			return;
		}
		real realmn = real(std::max(m, n));
		real eps = this->dlamch("Precision");
		// Compute norm(B - U * C)
		for (int j=0; j<n; j++)
		{
			Blas<real>::dcopy(m, &B[ldb*j], 1, work, 1);
			Blas<real>::dgemv("No transpose", m, m, -ONE, U, ldu, &C[ldc*j], 1, ONE, work, 1);
			resid = std::max(resid, Blas<real>::dasum(m, work, 1));
		}
		// Compute norm of B.
		real bnorm = this->dlange("1", m, n, B, ldb, work);
		if (bnorm<=ZERO)
		{
			if (resid!=ZERO)
			{
				resid = ONE / eps;
			}
		}
		else
		{
			if (bnorm>=resid)
			{
				resid = (resid/bnorm) / (realmn*eps);
			}
			else
			{
				if (bnorm<ONE)
				{
					resid = (std::min(resid, realmn*bnorm)/bnorm) / (realmn*eps);
				}
				else
				{
					resid = std::min(resid/bnorm, realmn) / (realmn*eps);
				}
			}
		}
	}

	/*! §dbdt03
	 *
	 * §dbdt03 reconstructs a bidiagonal matrix $B$ from its SVD:\n
	 *     $S = U^T B V$\n
	 * where $U$ and $V$ are orthogonal matrices and $S$ is diagonal.\n
	 * The test ratio to test the singular value decomposition is\n
	 *     $\{resid}=\frac{\|B - U S V^T\|}{\{n} \|B\| \{eps}}$\n
	 * where §eps is the machine precision.
	 * \param[in] uplo
	 *     Specifies whether the matrix $B$ is upper or lower bidiagonal.\n
	 *     = 'U': Upper bidiagonal\n
	 *     = 'L': Lower bidiagonal
	 *
	 * \param[in] n The order of the matrix $B$.
	 * \param[in] kd
	 *     The bandwidth of the bidiagonal matrix $B$. If $\{kd}=1$, the matrix $B$ is bidiagonal,
	 *     and if $\{kd}=0$, $B$ is diagonal and §e is not referenced. If §kd is greater than 1, it
	 *     is assumed to be 1, and if §kd is less than 0, it is assumed to be 0.
	 *
	 * \param[in] d
	 *     an array, dimension (§n)\n The §n diagonal elements of the bidiagonal matrix $B$.
	 *
	 * \param[in] e
	 *     an array, dimension ($\{n}-1$)\n
	 *     The $\{n}-1$ superdiagonal elements of the bidiagonal matrix $B$ if §uplo = 'U', or the
	 *     $\{n}-1$ subdiagonal elements of $B$ if §uplo = 'L'.
	 *
	 * \param[in] U
	 *     an array, dimension (§ldu,§n)\n
	 *     The §n by §n orthogonal matrix $U$ in the reduction $B = U^T A P$.
	 *
	 * \param[in] ldu The leading dimension of the array §U. $\{ldu}\ge\max(1,\{n})$
	 * \param[in] s
	 *     an array, dimension (§n)\n
	 *     The singular values from the SVD of $B$, sorted in decreasing order.
	 *
	 * \param[in] Vt
	 *     an array, dimension (§ldvt,§n)\n
	 *     The §n by §n orthogonal matrix $V^T$ in the reduction $B = U S V^T$.
	 *
	 * \param[in]  ldvt  The leading dimension of the array §Vt.
	 * \param[out] work  an array, dimension ($2\{n}$)
	 * \param[out] resid The test ratio: $\frac{\|B - U S V^T\|}{\{n} \|A\| \{eps}}$
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	void dbdt03(char const* const uplo, int const n, int const kd, real const* const d,
	            real const* const e, real const* const U, int const ldu, real const* const s,
	            real const* const Vt, int const ldvt, real* const work, real& resid) const
	{
		// Quick return if possible
		resid = ZERO;
		if (n<=0)
		{
			return;
		}
		// Compute B - U * S * V^T one column at a time.
		int i, j;
		real bnorm = ZERO;
		if (kd>=1)
		{
			// B is bidiagonal.
			if (std::toupper(uplo[0])=='U')
			{
				// B is upper bidiagonal.
				for (j=0; j<n; j++)
				{
					for (i=0; i<n; i++)
					{
						work[n+i] = s[i] * Vt[i+ldvt*j];
					}
					Blas<real>::dgemv("No transpose", n, n, -ONE, U, ldu, &work[n], 1, ZERO, work,
					                  1);
					work[j] += d[j];
					if (j>0)
					{
						work[j-1] += e[j-1];
						bnorm = std::max(bnorm, std::fabs(d[j])+std::fabs(e[j-1]));
					}
					else
					{
						bnorm = std::max(bnorm, std::fabs(d[j]));
					}
					resid = std::max(resid, Blas<real>::dasum(n, work, 1));
				}
			}
			else
			{
				// B is lower bidiagonal.
				for (j=0; j<n; j++)
				{
					for (i=0; i<n; i++)
					{
						work[n+i] = s[i] * Vt[i+ldvt*j];
					}
					Blas<real>::dgemv("No transpose", n, n, -ONE, U, ldu, &work[n], 1, ZERO, work,
					                  1);
					work[j] += d[j];
					if (j<n-1)
					{
						work[j+1] += e[j];
						bnorm = std::max(bnorm, std::fabs(d[j])+std::fabs(e[j]));
					}
					else
					{
						bnorm = std::max(bnorm, std::fabs(d[j]));
					}
					resid = std::max(resid, Blas<real>::dasum(n, work, 1));
				}
			}
		}
		else
		{
			// B is diagonal.
			for (j=0; j<n; j++)
			{
				for (i=0; i<n; i++)
				{
					work[n+i] = s[i] * Vt[i+ldvt*j];
				}
				Blas<real>::dgemv("No transpose", n, n, -ONE, U, ldu, &work[n], 1, ZERO, work, 1);
				work[j] += d[j];
				resid = std::max(resid, Blas<real>::dasum(n, work, 1));
			}
			j = Blas<real>::idamax(n, d, 1);
			bnorm = std::fabs(d[j]);
		}
		// Compute norm(B - U * S * V^T) / (n * norm(B) * eps)
		real eps = this->dlamch("Precision");
		if (bnorm<=ZERO)
		{
			if (resid!=ZERO)
			{
				resid = ONE / eps;
			}
		}
		else
		{
			if (bnorm>=resid)
			{
				resid = (resid/bnorm) / (real(n)*eps);
			}
			else
			{
				if (bnorm<ONE)
				{
					resid = (std::min(resid, real(n)*bnorm)/bnorm) / (real(n)*eps);
				}
				else
				{
					resid = std::min(resid/bnorm, real(n)) / (real(n)*eps);
				}
			}
		}
	}

	/*! §dbdt04
	 *
	 * §dbdt04 reconstructs a bidiagonal matrix $B$ from its (partial) SVD:\n
	 *     $S = U^T B V$\n
	 * where $U$ and $V$ are orthogonal matrices and $S$ is diagonal.\n
	 * The test ratio to test the singular value decomposition is\n
	 *     \f{equation*}{\{resid} = \frac{\| S - U^T B V \|}{\{n} \|B\| \{eps}}\f}\n
	 * where §Vt = $V^T$ and §eps is the machine precision.
	 * \param[in] uplo
	 *     Specifies whether the matrix $B$ is upper or lower bidiagonal.\n
	 *         = 'U': Upper bidiagonal\n
	 *         = 'L': Lower bidiagonal
	 *
	 * \param[in] n The order of the matrix $B$.
	 * \param[in] d
	 *     an array, dimension (§n)\n
	 *     The §n diagonal elements of the bidiagonal matrix $B$.
	 *
	 * \param[in] e
	 *     an array, dimension (§n-1)\n
	 *     The (§n-1) superdiagonal elements of the bidiagonal matrix $B$ if §uplo = 'U',\n
	 *     or the (§n-1) subdiagonal elements of                      $B$ if §uplo = 'L'.
	 *
	 * \param[in] s
	 *     an array, dimension (§ns)\n
	 *     The singular values from the (partial) SVD of $B$, sorted in decreasing order.
	 *
	 * \param[in] ns The number of singular values/vectors from the (partial) SVD of $B$.
	 * \param[in] U
	 *     an array, dimension (§ldu,§ns)\n
	 *     The §n by §ns orthogonal matrix $U$ in $S = U^T B V$.
	 *
	 * \param[in] ldu The leading dimension of the array §U. \n $\{ldu}\ge\max(1,\{n})$
	 * \param[in] Vt
	 *     an array, dimension (§ldvt,§n)\n
	 *     The §n by §ns orthogonal matrix $V$ in $S = U^T B V$.
	 *
	 * \param[in]  ldvt The leading dimension of the array §Vt.
	 * \param[out] work an array, dimension ($2\{n}$)
	 * \param[out] resid The test ratio: $\| S - U^T B V\| / (\{n} \|B\| \{eps})$
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	void dbdt04(char const* const uplo, int const n, real const* const d, real const* const e,
	            real const* const s, int const ns, real const* const U, int const ldu,
	            real const* const Vt, int const ldvt, real* const work, real& resid) const
	{
		// Quick return if possible.
		resid = ZERO;
		if (n<=0 || ns<=0)
		{
			return;
		}
		real eps = this->dlamch("Precision");
		// Compute S - U^t B V.
		real bnorm = ZERO;
		int i, j, k;
		if (std::toupper(uplo[0])=='U')
		{
			// B is upper bidiagonal.
			k = -1;
			for (i=0; i<ns; i++)
			{
				for (j=0; j<n-1; j++)
				{
					k++;
					work[k] = d[j]*Vt[i+ldvt*j] + e[j]*Vt[i+ldvt*(j+1)];
				}
				k++;
				work[k] = d[n-1]*Vt[i+ldvt*(n-1)];
			}
			bnorm = std::fabs(d[0]);
			for (i=1; i<n; i++)
			{
				bnorm = std::max(bnorm, std::fabs(d[i])+std::fabs(e[i-1]));
			}
		}
		else
		{
			// B is lower bidiagonal.
			k = -1;
			for (i=0; i<ns; i++)
			{
				k++;
				work[k] = d[0] * Vt[i];
				for (j=0; j<n-1; j++)
				{
					k++;
					work[k] = e[j]*Vt[i+ldvt*j] + d[j+1]*Vt[i+ldvt*(j+1)];
				}
			}
			bnorm = std::fabs(d[n-1]);
			for (i=0; i<n-1; i++)
			{
				bnorm = std::max(bnorm, std::fabs(d[i])+std::fabs(e[i]));
			}
		}
		Blas<real>::dgemm("T", "N", ns, ns, n, -ONE, U, ldu, work, n, ZERO, &work[n*ns], ns);
		// ||S - U^T B V||
		k = n * ns;
		for (i=0; i<ns; i++)
		{
			work[k+i] += s[i];
			resid = std::max(resid, Blas<real>::dasum(ns, &work[k], 1));
			k += ns;
		}
		if (bnorm<=ZERO)
		{
			if (resid!=ZERO)
			{
				resid = ONE / eps;
			}
		}
		else
		{
			if (bnorm>=resid)
			{
				resid = (resid / bnorm) / (real(n)*eps);
			}
			else
			{
				if (bnorm<ONE)
				{
					resid = (std::min(resid, real(n)*bnorm)/bnorm) / (real(n)*eps);
				}
				else
				{
					resid = std::min(resid/bnorm, real(n)) / (real(n)*eps);
				}
			}
		}
	}

	/*! §dchkbl
	 *
	 * §dchkbl tests §dgebal, a routine for balancing a general real matrix and isolating some of
	 * its eigenvalues.
	 * \param[in] nin  input stream of test examples.
	 * \param[in] nout output stream.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	void dchkbl(std::istream& nin, std::ostream& nout) const
	{
		const int lda = 20;
		int i, ihi=0, ihiin, ilo=0, iloin, info, j, knt, n, ninfo;
		//real anorm, meps;
		real rmax, sfmin, temp, vmax;
		int* lmax = new int[3];
		real* A = new real[lda*lda];
		real* Ain = new real[lda*lda];
		real* dummy = new real[1];
		real* scale = new real[lda];
		real* scalin = new real[lda];
		lmax[0] = 0;
		lmax[1] = 0;
		lmax[2] = 0;
		ninfo = 0;
		knt = 0;
		rmax = ZERO;
		vmax = ZERO;
		sfmin = this->dlamch("S");
		std::string doubleStr;
		std::stringstream strStr;
		unsigned DInd;
		//meps = dlamch("E");
		nin >> n;
		while (!nin.eof() && n!=0)
		{
			for (i=0; i<n; i++)
			{
				for (j=0; j<n; j++)
				{
					nin >> doubleStr;
					DInd = doubleStr.find('D',0);
					if (DInd!=doubleStr.npos)
					{
						doubleStr[DInd] = 'e';
					}
					strStr.str(doubleStr);
					strStr >> A[i+lda*j];
					strStr.clear();
				}
			}
			nin >> iloin;
			nin >> ihiin;
			iloin--;// compensate for 0-based indexing
			ihiin--;
			for (i=0; i<n; i++)
			{
				for (j=0; j<n; j++)
				{
					nin >> doubleStr;
					DInd = doubleStr.find('D',0);
					if (DInd!=doubleStr.npos)
					{
						doubleStr[DInd] = 'e';
					}
					strStr.str(doubleStr);
					strStr >> Ain[i+lda*j];
					strStr.clear();
				}
			}
			for (i=0; i<n; i++)
			{
				nin >> doubleStr;
				DInd = doubleStr.find('D',0);
				if (DInd!=doubleStr.npos)
				{
					doubleStr[DInd] = 'e';
				}
				strStr.str(doubleStr);
				strStr >> scalin[i];
				strStr.clear();
			}
			//anorm = dlange("M", n, n, A, lda, DUMMY);
			knt++;
			this->dgebal("B", n, A, lda, ilo, ihi, scale, info);
			if (info!=0)
			{
				ninfo++;
				lmax[0] = knt;
			}
			if (ilo!=iloin || ihi!=ihiin)
			{
				ninfo++;
				lmax[1] = knt;
			}
			for (i=0; i<n; i++)
			{
				for (j=0; j<n; j++)
				{
					temp = std::max(A[i+lda*j], Ain[i+lda*j]);
					temp = std::max(temp, sfmin);
					vmax = std::max(vmax, std::fabs(A[i+lda*j]-Ain[i+lda*j])/temp);
				}
			}
			for (i=0; i<n; i++)
			{
				temp = std::max(scale[i], scalin[i]);
				temp = std::max(temp, sfmin);
				vmax = std::max(vmax, std::fabs(scale[i]-scalin[i])/temp);
			}
			if (vmax>rmax)
			{
				lmax[2] = knt;
				rmax = vmax;
			}
			nin >> n;
		}
		nout << " .. test output of DGEBAL .. \n";
		nout << " value of largest test error            = " << std::setprecision(4);
		nout << std::setw(12) << rmax << '\n';
		nout << " example number where info is not zero  = " << std::setw(4) << lmax[0] << '\n';
		nout << " example number where ILO or IHI wrong  = " << std::setw(4) << lmax[1] << '\n';
		nout << " example number having largest error    = " << std::setw(4) << lmax[2] << '\n';
		nout << " number of examples where info is not 0 = " << std::setw(4) << ninfo << '\n';
		nout << " total number of examples tested        = " << std::setw(4) << knt << std::endl;
		delete[] lmax;
		delete[] A;
		delete[] Ain;
		delete[] dummy;
		delete[] scale;
		delete[] scalin;
	}

	/*! §dget22
	 *
	 * §dget22 does an eigenvector check.
	 * The basic test is:\n
	 *     $\{result}[0]=\frac{\left|AE-EW\right|}{|A||E|\{ulp}}$\n
	 * using the 1-norm. It also tests the normalization of $E$:\n
	 *     $\{result}[1]=\frac{\max_j{\left|\operatorname{m-norm}(E[j])-1\right|}}{\{n}\,\{ulp}}$\n
	 * where $E[j]$ is the $j$-th eigenvector, and $\operatorname{m-norm}$ is the max-norm of a
	 * vector. If an eigenvector is complex, as determined from $\{wi}[j]$ nonzero, then the
	 * max-norm of the vector $(e_r+\mathrm{i}\cdot e_i)$ is the maximum of\n
	 *     $|e_r[0]|+|e_i[0]|, \ldots, |e_r[\{n}-1]|+|e_i[\{n}-1]|$\n
	 * §W is a block diagonal matrix, with a 1 by 1 block for each real eigenvalue and a 2 by 2
	 * block for each complex conjugate pair.
	 * If eigenvalues $j$ and $j+1$ are a complex conjugate pair, so that $\{wr}[j]=\{wr}[j+1]=w_r$
	 * and $\{wi}[j]=-\{wi}[j+1]=w_i$, then the 2 by 2 block corresponding to the pair will be:\n
	 *     $\b{bm} w_r & w_i \\
	 *            -w_i & w_r \e{bm}$\n
	 * Such a block multiplying an §n by 2 matrix $\b{bm}u_r&u_i\e{bm}$ on the right will be the
	 * same as multiplying $u_r+\mathrm{i}\cdot u_i$ by $w_r+\mathrm{i}\cdot w_i$.\n
	 * To handle various schemes for storage of left eigenvectors, there are options to use
	 * $A$-transpose instead of $A$, $E$-transpose instead of $E$, and/or $W$-transpose instead of
	 * $W$.
	 * \param[in] transa
	 *     Specifies whether or not §A is transposed.\n
	 *     ='N': No transpose\n
	 *     ='T': Transpose\n
	 *     ='C': Conjugate transpose (= Transpose)
	 *
	 * \param[in] transe
	 *     Specifies whether or not §E is transposed.\n
	 *     ='N': No transpose, eigenvectors are in columns of §E \n
	 *     ='T': Transpose, eigenvectors are in rows of §E \n
	 *     ='C': Conjugate transpose (= Transpose)
	 *
	 * \param[in] transw
	 *     Specifies whether or not W is transposed.\n
	 *     ='N': No transpose\n
	 *     ='T': Transpose, use $-\{wi}[j]$ instead of $\{wi}[j]$\n
	 *     ='C': Conjugate transpose, use $-\{wi}[j]$ instead of $\{wi}[j]$
	 *
	 * \param[in] n   The order of the matrix $A$. $\{n}\ge 0$
	 * \param[in] A   an array, dimension (§lda,§n)\n The matrix whose eigenvectors are in $E$.
	 * \param[in] lda The leading dimension of the array §A. $\{lda}\ge\max(1,\{n})$.
	 * \param[in] E
	 *     an array, dimension (§lde,§n)\n
	 *     The matrix of eigenvectors. If §transe ='N', the eigenvectors are stored in the columns
	 *     of §E, if §transe ='T' or 'C', the eigenvectors are stored in the rows of §E.
	 *
	 * \param[in] lde    The leading dimension of the array §E. $\{lde}\ge\max(1,\{n})$.
	 * \param[in] wr, wi
	 *     arrays, dimension (§n)\n
	 *     The real and imaginary parts of the eigenvalues of $A$. Purely real eigenvalues are
	 *     indicated by $\{wi}[j]=0$. Complex conjugate pairs are indicated by
	 *     $\{wr}[j]=\{wr}[j+1]$ and $\{wi}[j]=-\{wi}[j+1]$ non-zero; the real part is assumed to
	 *     be stored in the $j$-th row/column and the imaginary part in the $(j+1)$-th row/column.
	 *
	 * \param[out] work   an array, dimension ($\{n}(\{n}+1)$)
	 * \param[out] result
	 *     an array, dimension (2)\n
	 *     $\{result}[0]=\frac{|AE-EW|}{|A||E|\{ulp}}$\n
	 *     $\{result}[1]=\frac{\max_j|\operatorname{m-norm}(E[j])-1|}{\{n}\cdot\{ulp}}$
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	void dget22(char const* const transa, char const* const transe, char const* const transw,
	            int const n, real const* const A, int const lda, real const* const E,
	            int const lde, real const* const wr, real const* const wi, real* const work,
	            real* const result) const
	{
		// Initialize result (in case n=0)
		result[0] = ZERO;
		result[1] = ZERO;
		if (n<=0)
		{
			return;
		}
		real unfl = this->dlamch("Safe minimum");
		real ulp  = this->dlamch("Precision");
		int itrnse = 0;
		int ince   = 1;
		char norma[2], norme[2];
		norma[0] = 'O';
		norma[1] = '\0';
		norme[0] = 'O';
		norme[1] = '\0';
		if (std::toupper(transa[0])=='T' || std::toupper(transa[0])=='C')
		{
			norma[0] = 'I';
		}
		if (std::toupper(transe[0])=='T' || std::toupper(transe[0])=='C')
		{
			norme[0] = 'I';
			itrnse = 1;
			ince   = lde;
		}
		// Check normalization of E
		real enrmin = ONE / ulp;
		real enrmax = ZERO;
		int ecol, ipair, j, jvec;
		if (itrnse==0)
		{
			// Eigenvectors are column vectors.
			real temp1;
			ipair = 0;
			for (jvec=0; jvec<n; jvec++)
			{
				temp1 = ZERO;
				if (ipair==0 && jvec<n-1 && wi[jvec]!=ZERO)
				{
					ipair = 1;
				}
				ecol = lde * jvec;
				if (ipair==1)
				{
					// Complex eigenvector
					for (j=0; j<n; j++)
					{
						temp1 = std::max(temp1, std::fabs(E[j+ecol])+std::fabs(E[j+ecol+lde]));
					}
					enrmin = std::min(enrmin, temp1);
					enrmax = std::max(enrmax, temp1);
					ipair = 2;
				}
				else if (ipair==2)
				{
					ipair = 0;
				}
				else
				{
					// Real eigenvector
					for (j=0; j<n; j++)
					{
						temp1 = std::max(temp1, std::fabs(E[j+ecol]));
					}
					enrmin = std::min(enrmin, temp1);
					enrmax = std::max(enrmax, temp1);
					ipair = 0;
				}
			}
		}
		else
		{
			// Eigenvectors are row vectors.
			for (jvec=0; jvec<n; jvec++)
			{
				work[jvec] = ZERO;
			}
			for (j=0; j<n; j++)
			{
				ipair = 0;
				for (jvec=0; jvec<n; jvec++)
				{
					if (ipair==0 && jvec<n-1 && wi[jvec]!=ZERO)
					{
						ipair = 1;
					}
					ecol = lde * jvec;
					if (ipair==1)
					{
						work[jvec] = std::max(work[jvec],
						                      std::fabs(E[j+ecol])+std::fabs(E[j+ecol+lde]));
						work[jvec+1] = work[jvec];
					}
					else if (ipair==2)
					{
						ipair = 0;
					}
					else
					{
						work[jvec] = std::max(work[jvec], std::fabs(E[j+ecol]));
						ipair = 0;
					}
				}
			}
			for (jvec=0; jvec<n; jvec++)
			{
				enrmin = std::min(enrmin, work[jvec]);
				enrmax = std::max(enrmax, work[jvec]);
			}
		}
		// Norm of A:
		real anorm = std::max(this->dlange(norma, n, n, A, lda, work), unfl);
		// Norm of E:
		real enorm = std::max(this->dlange(norme, n, n, E, lde, work), ulp);
		// Norm of error:
		// Error =  AE - EW
		this->dlaset("Full", n, n, ZERO, ZERO, work, n);
		ipair = 0;
		int ierow = 0;
		int iecol = 0;
		real Wmat[2*2];
		for (int jcol=0; jcol<n; jcol++)
		{
			if (itrnse==1)
			{
				ierow = jcol;
			}
			else
			{
				iecol = jcol;
			}
			if (ipair==0 && wi[jcol]!=ZERO)
			{
				ipair = 1;
			}
			if (ipair==1)
			{
				Wmat[0,0] =  wr[jcol];
				Wmat[1,0] = -wi[jcol];
				Wmat[0,1] =  wi[jcol];
				Wmat[1,1] =  wr[jcol];
				Blas<real>::dgemm(transe, transw, n, 2, 2, ONE, &E[ierow+lde*iecol], lde, Wmat, 2,
				                  ZERO, &work[n*jcol], n);
				ipair = 2;
			}
			else if (ipair==2)
			{
				ipair = 0;
			}
			else
			{
				Blas<real>::daxpy(n, wr[jcol], &E[ierow+lde*iecol], ince, &work[n*jcol], 1);
				ipair = 0;
			}
		}
		Blas<real>::dgemm(transa, transe, n, n, n, ONE, A, lda, E, lde, -ONE, work, n);
		real errnrm = this->dlange("One", n, n, work, n, &work[n*n]) / enorm;
		// Compute result[0] (avoiding under/overflow)
		if (anorm>errnrm)
		{
			result[0] = (errnrm/anorm) / ulp;
		}
		else
		{
			if (anorm<ONE)
			{
				result[0] = (std::min(errnrm, anorm)/anorm) / ulp;
			}
			else
			{
				result[0] = std::min(errnrm/anorm, ONE) / ulp;
			}
		}
		// Compute result[1] : the normalization error in E.
		result[1] = std::max(std::fabs(enrmax-ONE), std::fabs(enrmin-ONE)) / (real(n)*ulp);
	}

	/*! §dlahd2
	 *
	 * §dlahd2 prints header information for the different test paths.
	 * \param[in] iounit
	 *     On entry, §iounit specifies the output stream to which the header information should be
	 *     printed.
	 *
	 * \param[in] path
	 *     On entry, §path contains the name of the path for which the header information is to be
	 *     printed. Current paths are\n
	 *     "DHS", "ZHS": Non-symmetric eigenproblem.\n
	 *     "DST", "ZST": Symmetric eigenproblem.\n
	 *     "DSG", "ZSG": Symmetric Generalized eigenproblem.\n
	 *     "DBD", "ZBD": Singular Value Decomposition (SVD)\n
	 *     "DBB", "ZBB": General Banded reduction to bidiagonal form\n
	 *     These paths also are supplied in single precision
	 *     (replace leading D by S and leading Z by C in path names).
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	void dlahd2(std::ostream& iounit, char const* const path) const
	{
		if (!iounit.good())
		{
			return;
		}
		char pathcopy[4];
		pathcopy[0] = std::toupper(path[0]);
		pathcopy[1] = std::toupper(path[1]);
		pathcopy[2] = std::toupper(path[2]);
		pathcopy[3] = '\0';
		bool sord = (pathcopy[0]=='S' || pathcopy[0]=='D');
		char const* str9999 = ":  no header available";
		if (!sord && !(pathcopy[0]=='C' || pathcopy[0]=='Z'))
		{
			iounit << " " << pathcopy << str9999 << std::endl;
		}
		char c2[2];
		std::strncpy(c2, &pathcopy[1], 2);
		if (std::strncmp(c2, "HS", 2)==0)
		{
			char const* str9988 = " Matrix types (see xCHKHS for details): ";
			char const* str9987 = "\n Special Matrices:\n"
			    "  1=Zero matrix.                          5=Diagonal: geometr. spaced entries.\n"
			    "  2=Identity matrix.                      6=Diagonal: clustered entries.\n"
			    "  3=Transposed Jordan block.              7=Diagonal: large, evenly spaced.\n"
			    "  4=Diagonal: evenly spaced entries.      8=Diagonal: small, evenly spaced.";
			char const* str9986a = " Dense, Non-Symmetric Matrices:\n"
			    "  9=Well-cond., evenly spaced eigenvals. 14=Ill-cond., geomet. spaced eigenals.\n"
			    " 10=Well-cond., geom. spaced eigenvals.  15=Ill-conditioned, clustered e.vals.\n"
			    " 11=Well-conditioned, clustered e.vals.  16=Ill-cond., random complex ";
			char const* str9986b = "\n 12=Well-cond., random complex ";
			char const* str9986c = "    17=Ill-cond., large rand. complx ";
			char const* str9986d = "\n 13=Ill-conditioned, evenly spaced.   "
			                       "   18=Ill-cond., small rand. complx ";
			char const* str9985 =
			    " 19=Matrix with random O(1) entries.     21=Matrix with small random entries."
			    "\n 20=Matrix with large random entries.   ";
			char const* str9984a =
			    "\n Tests performed:   (H is Hessenberg, T is Schur, U and Z are ";
			char const* str9984b = ",\n                    ";
			char const* str9984c = ", W is a diagonal matrix of eigenvalues,\n"
			    "                    L and R are the left and right eigenvector matrices)\n"
			    "  1 = | A - U H U";
			char const* str9984d = " | / (|A| n ulp)           2 = | I - U U";
			char const* str9984e = " | / (n ulp)\n  3 = | H - Z T Z";
			char const* str9984f = " | / (|H| n ulp )           4 = | I - Z Z";
			char const* str9984g = " | / (n ulp)\n  5 = | A - UZ T (UZ)";
			char const* str9984h = " | / (|A| n ulp)       6 = | I - UZ (UZ)";
			char const* str9984i =
			    " | / (n ulp)\n  7 = | T(e.vects.) - T(no e.vects.) | / (|T| ulp)\n"
			    "  8 = | W(e.vects.) - W(no e.vects.) | / (|W| ulp)\n"
			    "  9 = | TR - RW | / (|T| |R| ulp)      10 = | LT - WL | / ( |T| |L| ulp)\n"
			    " 11= |HX - XW| / (|H| |X| ulp)  (inv.it) 12= |YH - WY| / (|H| |Y| ulp)  (inv.it)";
			if (sord)
			{
				// Real Non-symmetric Eigenvalue Problem:
				iounit << "\n " << pathcopy << " -- Real Non-symmetric eigenvalue problem\n";
				// Matrix types
				iounit << str9988 << '\n';
				iounit << str9987 << '\n';
				iounit << str9986a << "pairs " << str9986b << "pairs " << str9986c << "prs."
				       << str9986d << "prs.\n";
				iounit << str9985 << '\n';
				// Tests performed
				iounit << str9984a << "orthogonal" << str9984b << "'=transpose" << str9984c
				       << "'" << str9984d << "'" << str9984e << "'" << str9984f << "'" << str9984g
				       << "'" << str9984h << "'" << str9984i << std::endl;
			}
			else
			{
				// Complex Non-symmetric Eigenvalue Problem:
				iounit << "\n " << pathcopy << " -- Complex Non-symmetric eigenvalue problem\n";
				// Matrix types
				iounit << str9988 << '\n';
				iounit << str9987 << '\n';
				iounit << str9986a << "e.vals" << str9986b << "e.vals" << str9986c << "e.vs"
				       << str9986d << "e.vs\n";
				iounit << str9985 << '\n';
				// Tests performed
				iounit << str9984a << "unitary" << str9984b << "*=conj.transp." << str9984c << '*'
				       << str9984d << '*' << str9984e << '*' << str9984f << '*' << std::endl;
			}
		}
		else if (std::strncmp(c2, "ST", 2)==0)
		{
			char const* str9983 = " Matrix types (see xDRVST for details): ";
			char const* str9982 = "\n Special Matrices:\n"
			    "  1=Zero matrix.                          5=Diagonal: clustered entries.\n"
			    "  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n"
			    "  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n"
			    "  4=Diagonal: geometr. spaced entries.";
			char const* str9981 = " Matrices:\n"
			    "  8=Evenly spaced eigenvals.             12=Small, evenly spaced eigenvals.\n"
			    "  9=Geometrically spaced eigenvals.      13=Matrix with random O(1) entries.\n"
			    " 10=Clustered eigenvalues.               14=Matrix with large random entries.\n"
			    " 11=Large, evenly spaced eigenvals.      15=Matrix with small random entries.";
			if (sord)
			{
				// Real Symmetric Eigenvalue Problem:
				iounit << "\n " << pathcopy << " -- Real Symmetric eigenvalue problem";
				// Matrix types
				iounit << str9983 << '\n';
				iounit << str9982 << '\n';
				iounit << " Dense Symmetric" << str9981 << '\n';
				// Tests performed
				iounit << "\n Tests performed:  See sdrvst.f" << std::endl;
			}
			else
			{
				// Complex Hermitian Eigenvalue Problem:
				iounit << "\n " << pathcopy << " -- Complex Hermitian eigenvalue problem\n";
				// Matrix types
				iounit << str9983 << '\n';
				iounit << str9982 << '\n';
				iounit << " Dense Hermitian" << str9981 << '\n';
				// Tests performed
				iounit << "\n Tests performed:  See cdrvst.f" << std::endl;
			}
		}
		else if (std::strncmp(c2, "SG", 2)==0)
		{
			char const* str9980 = " Matrix types (see xDRVSG for details): ";
			char const* str9979 ="\n Special Matrices:\n"
			    "  1=Zero matrix.                          5=Diagonal: clustered entries.\n"
			    "  2=Identity matrix.                      6=Diagonal: large, evenly spaced.\n"
			    "  3=Diagonal: evenly spaced entries.      7=Diagonal: small, evenly spaced.\n"
			    "  4=Diagonal: geometr. spaced entries.";
			char const* str9978a = " Dense or Banded ";
			char const* str9978b = " Matrices: \n"
			    "  8=Evenly spaced eigenvals.          15=Matrix with small random entries.\n"
			    "  9=Geometrically spaced eigenvals.   16=Evenly spaced eigenvals, KA=1, KB=1.\n"
			    " 10=Clustered eigenvalues.            17=Evenly spaced eigenvals, KA=2, KB=1.\n"
			    " 11=Large, evenly spaced eigenvals.   18=Evenly spaced eigenvals, KA=2, KB=2.\n"
			    " 12=Small, evenly spaced eigenvals.   19=Evenly spaced eigenvals, KA=3, KB=1.\n"
			    " 13=Matrix with random O(1) entries.  20=Evenly spaced eigenvals, KA=3, KB=2.\n"
			    " 14=Matrix with large random entries. 21=Evenly spaced eigenvals, KA=3, KB=3.";
			if (sord)
			{
				// Real Symmetric Generalized Eigenvalue Problem:
				iounit << "\n " << pathcopy
				       << " -- Real Symmetric Generalized eigenvalue problem\n";
				// Matrix types
				iounit << str9980 << '\n';
				iounit << str9979 << '\n';
				iounit << str9978a << "Symmetric" << str9978b << '\n';
				// Tests performed
				iounit << "\n Tests performed:   \n"
				          "(For each pair (A,B), where A is of the given type \n"
				          " and B is a random well-conditioned matrix. D is \n"
				          " diagonal, and Z is orthogonal.)\n"
				          " 1 = DSYGV, with ITYPE=1 and UPLO='U':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n"
				          " 2 = DSPGV, with ITYPE=1 and UPLO='U':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n"
				          " 3 = DSBGV, with ITYPE=1 and UPLO='U':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n"
				          " 4 = DSYGV, with ITYPE=1 and UPLO='L':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n"
				          " 5 = DSPGV, with ITYPE=1 and UPLO='L':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n"
				          " 6 = DSBGV, with ITYPE=1 and UPLO='L':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n";
				iounit << " 7 = DSYGV, with ITYPE=2 and UPLO='U':"
				          "  | A B Z - Z D | / (|A| |Z| n ulp)     \n"
				          " 8 = DSPGV, with ITYPE=2 and UPLO='U':"
				          "  | A B Z - Z D | / (|A| |Z| n ulp)     \n"
				          " 9 = DSPGV, with ITYPE=2 and UPLO='L':"
				          "  | A B Z - Z D | / (|A| |Z| n ulp)     \n"
				          "10 = DSPGV, with ITYPE=2 and UPLO='L':"
				          "  | A B Z - Z D | / (|A| |Z| n ulp)     \n"
				          "11 = DSYGV, with ITYPE=3 and UPLO='U':"
				          "  | B A Z - Z D | / (|A| |Z| n ulp)     \n"
				          "12 = DSPGV, with ITYPE=3 and UPLO='U':"
				          "  | B A Z - Z D | / (|A| |Z| n ulp)     \n"
				          "13 = DSYGV, with ITYPE=3 and UPLO='L':"
				          "  | B A Z - Z D | / (|A| |Z| n ulp)     \n"
				          "14 = DSPGV, with ITYPE=3 and UPLO='L':"
				          "  | B A Z - Z D | / (|A| |Z| n ulp)     " << std::endl;
			}
			else
			{
				// Complex Hermitian Generalized Eigenvalue Problem:
				iounit << "\n " << pathcopy
				       << " -- Complex Hermitian Generalized eigenvalue problem\n";
				// Matrix types
				iounit << str9980 << '\n';
				iounit << str9979 << '\n';
				iounit << str9978a << "Hermitian" << str9978b << '\n';
				// Tests performed
				iounit << "\n Tests performed:   \n"
				          "(For each pair (A,B), where A is of the given type \n"
				          " and B is a random well-conditioned matrix. D is \n"
				          " diagonal, and Z is unitary.)\n"
				          " 1 = ZHEGV, with ITYPE=1 and UPLO='U':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n"
				          " 2 = ZHPGV, with ITYPE=1 and UPLO='U':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n"
				          " 3 = ZHBGV, with ITYPE=1 and UPLO='U':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n"
				          " 4 = ZHEGV, with ITYPE=1 and UPLO='L':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n"
				          " 5 = ZHPGV, with ITYPE=1 and UPLO='L':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n"
				          " 6 = ZHBGV, with ITYPE=1 and UPLO='L':"
				          "  | A Z - B Z D | / (|A| |Z| n ulp)     \n";
				iounit << " 7 = ZHEGV, with ITYPE=2 and UPLO='U':"
				          "  | A B Z - Z D | / (|A| |Z| n ulp)     \n"
				          " 8 = ZHPGV, with ITYPE=2 and UPLO='U':"
				          "  | A B Z - Z D | / (|A| |Z| n ulp)     \n"
				          " 9 = ZHPGV, with ITYPE=2 and UPLO='L':"
				          "  | A B Z - Z D | / (|A| |Z| n ulp)     \n"
				          "10 = ZHPGV, with ITYPE=2 and UPLO='L':"
				          "  | A B Z - Z D | / (|A| |Z| n ulp)     \n"
				          "11 = ZHEGV, with ITYPE=3 and UPLO='U':"
				          "  | B A Z - Z D | / (|A| |Z| n ulp)     \n"
				          "12 = ZHPGV, with ITYPE=3 and UPLO='U':"
				          "  | B A Z - Z D | / (|A| |Z| n ulp)     \n"
				          "13 = ZHEGV, with ITYPE=3 and UPLO='L':"
				          "  | B A Z - Z D | / (|A| |Z| n ulp)     \n"
				          "14 = ZHPGV, with ITYPE=3 and UPLO='L':"
				          "  | B A Z - Z D | / (|A| |Z| n ulp)     " << std::endl;
			}
		}
		else if (std::strncmp(c2, "BD", 2)==0)
		{
			char const* str9973 = " Matrix types (see xCHKBD for details):\n Diagonal matrices:\n"
			    "   1: Zero                             5: Clustered entries\n"
			    "   2: Identity                         6: Large, evenly spaced entries\n"
			    "   3: Evenly spaced entries            7: Small, evenly spaced entries\n"
			    "   4: Geometrically spaced entries\n General matrices:\n"
			    "   8: Evenly spaced sing. vals.       12: Small, evenly spaced sing vals\n"
			    "   9: Geometrically spaced sing vals  13: Random, O(1) entries\n"
			    "  10: Clustered sing. vals.           14: Random, scaled near overflow\n"
			    "  11: Large, evenly spaced sing vals  15: Random, scaled near underflow";
			char const* str9972a = "\n Test ratios:  "
			    "(B: bidiagonal, S: diagonal, Q, P, U, and V: ";
			char const* str9972b = "\n                X: m x nrhs, Y = Q' X, and Z = U' Y)";
			char const* str9971 =
			    "   1: norm(A - Q B P') / (norm(A) max(m,n) ulp)\n"
			    "   2: norm(I - Q' Q)   / (m ulp)\n"
			    "   3: norm(I - P' P)   / (n ulp)\n"
			    "   4: norm(B - U S V') / (norm(B) min(m,n) ulp)\n"
			    "   5: norm(Y - U Z)    / (norm(Z) max(min(m,n),k) ulp)\n"
			    "   6: norm(I - U' U)   / (min(m,n) ulp)\n"
			    "   7: norm(I - V' V)   / (min(m,n) ulp)\n"
			    "   8: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n"
			    "   9: norm(S - S1)     / (norm(S) ulp), where S1 is computed\n"
			    "                                            without computing U and V'\n"
			    "  10: Sturm sequence test (0 if sing. vals of B within THRESH of S)\n"
			    "  11: norm(A - (QU) S (V' P')) / (norm(A) max(m,n) ulp)\n"
			    "  12: norm(X - (QU) Z)         / (|X| max(M,k) ulp)\n"
			    "  13: norm(I - (QU)'(QU))      / (M ulp)\n"
			    "  14: norm(I - (V' P') (P V))  / (N ulp)\n"
			    "  15: norm(B - U S V') / (norm(B) min(m,n) ulp)\n"
			    "  16: norm(I - U' U)   / (min(m,n) ulp)\n"
			    "  17: norm(I - V' V)   / (min(m,n) ulp)\n"
			    "  18: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n"
			    "  19: norm(S - S1)     / (norm(S) ulp), where S1 is computed\n"
			    "                                            without computing U and V'\n"
			    "  20: norm(B - U S V')  / (norm(B) min(m,n) ulp)  DBDSVX(V,A)\n"
			    "  21: norm(I - U' U)    / (min(m,n) ulp)\n"
			    "  22: norm(I - V' V)    / (min(m,n) ulp)\n"
			    "  23: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n"
			    "  24: norm(S - S1)      / (norm(S) ulp), where S1 is computed\n"
			    "                                             without computing U and V'\n"
			    "  25: norm(S - U' B V) / (norm(B) n ulp)  DBDSVX(V,I)\n"
			    "  26: norm(I - U' U)    / (min(m,n) ulp)\n"
			    "  27: norm(I - V' V)    / (min(m,n) ulp)\n"
			    "  28: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n"
			    "  29: norm(S - S1)      / (norm(S) ulp), where S1 is computed\n"
			    "                                             without computing U and V'\n"
			    "  30: norm(S - U' B V) / (norm(B) n ulp)  DBDSVX(V,V)\n"
			    "  31: norm(I - U' U)    / (min(m,n) ulp)\n"
			    "  32: norm(I - V' V)    / (min(m,n) ulp)\n"
			    "  33: Test ordering of S  (0 if nondecreasing, 1/ulp  otherwise)\n"
			    "  34: norm(S - S1)      / (norm(S) ulp), where S1 is computed\n"
			    "                                             without computing U and V'";
			if (sord)
			{
				// Real Singular Value Decomposition:
				iounit << "\n " << pathcopy << " -- Real Singular Value Decomposition\n";
				// Matrix types
				iounit << str9973 << '\n';
				// Tests performed
				iounit << str9972a << "orthogonal" << str9972b << '\n';
				iounit << str9971 << std::endl;
			}
			else
			{
				// Complex Singular Value Decomposition:
				iounit << "\n " << pathcopy << " -- Complex Singular Value Decomposition\n";
				// Matrix types
				iounit << str9973 << '\n';
				// Tests performed
				iounit << str9972a << "unitary   " << str9972b << '\n';
				iounit << str9971 << std::endl;
			}
		}
		else if (std::strncmp(c2, "BB", 2)==0)
		{
			char const* str9970 = " Matrix types (see xCHKBB for details):\n Diagonal matrices:\n"
			    "   1: Zero                             5: Clustered entries\n"
			    "   2: Identity                         6: Large, evenly spaced entries\n"
			    "   3: Evenly spaced entries            7: Small, evenly spaced entries\n"
			    "   4: Geometrically spaced entries\n General matrices:\n"
			    "   8: Evenly spaced sing. vals.       12: Small, evenly spaced sing vals\n"
			    "   9: Geometrically spaced sing vals  13: Random, O(1) entries\n"
			    "  10: Clustered sing. vals.           14: Random, scaled near overflow\n"
			    "  11: Large, evenly spaced sing vals  15: Random, scaled near underflow";
			char const* str9969a = "\n Test ratios:  (B: upper bidiagonal, Q and P: ";
			char const* str9969b = "\n                C: m x nrhs, PT = P', Y = Q' C)\n"
			    " 1: norm(A - Q B PT) / (norm(A) max(m,n) ulp)\n"
			    " 2: norm(I - Q' Q)   / (m ulp)\n"
			    " 3: norm(I - PT PT')   / (n ulp)\n"
			    " 4: norm(Y - Q' C)   / (norm(Y) max(m,nrhs) ulp)";
			if (sord)
			{
				// Real General Band reduction to bidiagonal form:
				iounit << "\n " << pathcopy << " -- Real Band reduc. to bidiagonal form\n";
				// Matrix types
				iounit << str9970 << '\n';
				// Tests performed
				iounit << str9969a << "orthogonal" << str9969b << std::endl;
			}
			else
			{
				// Complex Band reduction to bidiagonal form:
				iounit << "\n " << pathcopy << " -- Complex Band reduc. to bidiagonal form\n";
				// Matrix types
				iounit << str9970 << '\n';
				// Tests performed
				iounit << str9969a << "unitary   " << str9969b << std::endl;
			}
		}
		else
		{
			iounit << " " << pathcopy << str9999 << std::endl;
			return;
		}
		return;
	}

	/*! §dlasum
	 *
	 * §dlasum prints a summary of the results from one of the test routines.
	 * \param[in] type   The LAPACK path name.
	 * \param[in] iounit The output stream to which results are to be printed.
	 * \param[in] ie     The number of tests which produced an error.
	 * \param[in] nrun   The total number of tests.
	 * \authors Univ.of Tennessee
	 * \authors Univ.of California Berkeley
	 * \authors Univ.of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December December 2016                                                              */
	void dlasum(char const* const type, std::ostream& iounit, int const ie, int const nrun) const
	{
		char typecopy[4];
		std::strncpy(typecopy, type, 3);
		typecopy[3] = '\0';
		if (ie>0)
		{
			iounit << " " << typecopy << ": " << std::setw(4) << ie << " out of " << std::setw(5)
			       << nrun << " tests failed to pass the threshold\n";
		}
		else
		{
			iounit << "\n " << "All tests for " << typecopy << " passed the threshold ("
			       << std::setw(5) << nrun << " tests run)\n";
		}
		iounit.flush();
	}

	/*! §dort01
	 *
	 * §dort01 checks that the matrix $U$ is orthogonal by computing the ratio\n
	 *     $\{resid} = \frac{\|I - UU^T\|}{\{n} \, \{eps}}$, if §rowcol ='R',\n
	 * or\n
	 *     $\{resid} = \frac{\|I - U^TU\|}{\{m} \, \{eps}}$, if §rowcol ='C'.\n
	 * Alternatively, if there isn't sufficient workspace to form $I - U*U^T$ or $I - U^T*U$, the
	 * ratio is computed as\n
	 *     $\{resid} = \max\left(\frac{|I - UU^T|}{\{n} \, \{eps}}\right)$, if §rowcol ='R',\n
	 * or\n
	 *     $\{resid} = \max\left(\frac{|I - U^TU|}{\{m} \, \{eps}}\right)$, if §rowcol ='C'.\n
	 * where §eps is the machine precision. §rowcol is used only if $\{m}=\{n}$;
	 * if $\{m}>\{n}$, §rowcol is assumed to be 'C', and if $\{m}<\{n}$, §rowcol is assumed to be
	 * 'R'.
	 * \param[in] rowcol
	 *     Specifies whether the rows or columns of $U$ should be checked for orthogonality.
	 *     Used only if $\{m}=\{n}$.\n
	 *     ='R': Check for orthogonal rows of $U$\n
	 *     ='C': Check for orthogonal columns of $U$
	 *
	 * \param[in] m The number of rows of the matrix $U$.
	 * \param[in] n The number of columns of the matrix $U$.
	 * \param[in] U
	 *     an array, dimension (§ldu,§N)\n
	 *     The orthogonal matrix $U$.\n
	 *     §U is checked for orthogonal columns if $\{m}>\{n}$ or if $\{m}=\{n}$ and §rowcol ='C'.
	 *     \n §U is checked for orthogonal rows if $\{m}<\{n}$ or if $\{m}=\{n}$ and §rowcol ='R'.
	 *
	 * \param[in]  ldu  The leading dimension of the array §U. $\{ldu}\ge\max(1,\{m})$.
	 * \param[out] work an array, dimension (§lwork)
	 * \param[in] lwork
	 *     The length of the array §work. For best performance, §lwork should be at least
	 *     $\{n}(\{n}+1)$ if §rowcol ='C' or $\{m}(\{m}+1)$ if §rowcol ='R', but the test will be
	 *     done even if §lwork is 0.
	 *
	 * \param[out] resid
	 *     $\{resid} = \frac{\|I - UU^T\|}{\{n} \, \{eps}}$, if §rowcol ='R', or\n
	 *     $\{resid} = \frac{\|I - U^TU\|}{\{m} \, \{eps}}$, if §rowcol ='C'.\n
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	void dort01(char const* const rowcol, int const m, int const n, real const* const U,
	            int const ldu, real* const work, int const lwork, real& resid) const
	{
		resid = ZERO;
		// Quick return if possible
		if (m<=0 || n<=0)
		{
			return;
		}
		real eps = this->dlamch("Precision");
		char transu[2];
		transu[1] = '\0';
		int k;
		if (m<n || (m==n && std::toupper(rowcol[0])=='R'))
		{
			transu[0] = 'N';
			k = n;
		}
		else
		{
			transu[0] = 'T';
			k = m;
		}
		int mnmin = std::min(m, n);
		int ldwork;
		if ((mnmin+1)*mnmin<=lwork)
		{
			ldwork = mnmin;
		}
		else
		{
			ldwork = 0;
		}
		int i, j, uj;
		real tmp;
		if (ldwork>0)
		{
			// Compute I - U*U^T or I - U^T*U.
			this->dlaset("Upper", mnmin, mnmin, ZERO, ONE, work, ldwork);
			Blas<real>::dsyrk("Upper", transu, mnmin, k, -ONE, U, ldu, ONE, work, ldwork);
			// Compute norm(I - U*U^T) / (k * eps) .
			resid = dlansy("1", "Upper", mnmin, work, ldwork, &work[ldwork*mnmin]);
			resid = (resid/real(k)) / eps;
		}
		else if (transu[0]=='T')
		{
			// Find the maximum element in abs(I - U^T*U) / (m * eps)
			for (j=0; j<n; j++)
			{
				uj = ldu * j;
				for (i=0; i<=j; i++)
				{
					if (i!=j)
					{
						tmp = ZERO;
					}
					else
					{
						tmp = ONE;
					}
					tmp -= Blas<real>::ddot(m, &U[ldu*i], 1, &U[uj], 1);
					resid = std::max(resid, std::fabs(tmp));
				}
			}
			resid = (resid/real(m)) / eps;
		}
		else
		{
			// Find the maximum element in abs(I - U*U') / (n * eps)
			for (j=0; j<m; j++)
			{
				for (i=0; i<=j; i++)
				{
					if (i!=j)
					{
						tmp = ZERO;
					}
					else
					{
						tmp = ONE;
					}
					tmp -= Blas<real>::ddot(n, &U[j], ldu, &U[i], ldu);
					resid = std::max(resid, std::fabs(tmp));
				}
			}
			resid = (resid/real(n)) / eps;
		}
	}

	// TODO: xlaenv, ilaenv, xerbla
};

#endif