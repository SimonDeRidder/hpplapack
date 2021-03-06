#ifndef TESTING_EIG_HEADER
#define TESTING_EIG_HEADER

#include <cstring>
#include <cmath>
#include <ctime>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

#include "Blas.hpp"
#include "Lapack_dyn.hpp"
#include "Testing_matgen.hpp"

/*!\class Testing_eig
 * \brief A template class containing LAPACK eigenvalue testing routines.
 * Testing_eig contains the LAPACK routines for testing eigenvalue-related routines.
 * The template type is meant to be double, but can be any floating point type                   */
template<class real>
class Testing_eig : public Lapack_dyn<real>
{
private:
	// constants

	real const ZERO = real(0.0); //!< A constant zero (0.0) value
	real const ONE  = real(1.0); //!< A constant one  (1.0) value
	real const TWO  = real(2.0); //!< A constant two  (2.0) value

	// "Common" variables

	/*! A struct containing I/O and error info */
	struct infostruct
	{
		int info;                         //!< integer containing subroutine error info
		std::ostream& nout;                //!< input stream
		bool ok;                           //!< flag to state all is ok
		bool lerr;                         //!< flag to indicate an error has occured
	} infoc = {0, std::cout, true, false}; //!< A classwide infostruct instance

	/*! A struct containing routine names */
	struct srnamstruct
	{
		char srnam[32]; //!< routine name variable
	} srnamc = {'\0'};  //!< A classwide srnamstruct instance

	/*! A struct containing parameters to emulate different environments */
	struct laenvstruct
	{
		int iparms[100]; //!< array containing parameters. used in §ilaenv, §iparmq
	} claenv;            //!< A classwide leanvstruct instance

	/*! A struct containing parameters to emulate different environments */
	struct envirstruct
	{
		int nproc;
		int nshift;
		int maxb;
	} cenvir;       //!< A classwide envirstruct instance

	struct selstruct
	{
		int selopt;
		int seldim;
		bool selval[20];
		real selwr[20];
		real selwi[20];
	} sslct;       //!< A classwide selstruct instance

	Testing_matgen<real> const MatGen; //!< matgen instance

public:
	virtual ~Testing_eig(){}

	// LAPACK TESTING EIG (alphabetically)

	/*! §alareq
	 *
	 * §alareq handles input for the LAPACK test program. It is called to evaluate the input line
	 * which requested §nmats matrix types for §path. The flow of control is as follows:\n
	 * §If $\{nmats}=\{ntypes}$\n
	 * &emsp;$\{dotype}[0:\{ntypes}-1]=\{true}$;\n
	 * §else \n
	 * {\n
	 * &emsp;Read the next input line for §nmats matrix types
	 * &emsp;Set $\{dotype}[I-1]=\{true}$ for each valid type $I$\n
	 * }
	 * \param[in]  path   A LAPACK path name for testing.
	 * \param[in]  nmats  The number of matrix types to be used in testing this path.
	 * \param[out] dotype
	 *     a boolean array, dimension (§ntypes)\n
	 *     The vector of flags indicating if each type will be tested.
	 *
	 * \param[in] ntypes The maximum number of matrix types for this path.
	 * \param[in] nin    The input stream for input.
	 * \param[in] nout   The output stream for output.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	void alareq(char const* const path, int const nmats, bool* const dotype, int const ntypes,
	            std::istream& nin, std::ostream& nout) const
	{
		int i;
		if (nmats>=ntypes)
		{
			// Test everything if nmats >= ntypes.
			for (i=0; i<ntypes; i++)
			{
				dotype[i] = true;
			}
		}
		else
		{
			for (i=0; i<ntypes; i++)
			{
				dotype[i] = false;
			}
			// Read a line of matrix types if 0 < nmats < ntypes.
			int nreq[100];
			if (nmats>0)
			{
				char line[80];
				nin.getline(line, 80);
				if (!nin.good())
				{
					nout << "\n *** End of file reached when trying to read matrix types for "
					     << path << "\n *** Check that you are requesting the right number of "
					     << "types for each path\n";
					nout << std::endl;
					return;
				}
				int lenp = std::strlen(line);
				char const* str9994a = " ==> Specify ";
				char const* str9994b =
				    " matrix types on this line or adjust NTYPES on previous line";
				int i1, ic;
				i = -1;
				for (int j=0; j<nmats; j++)
				{
					nreq[j] = 0;
					i1 = -1;
					while (true)
					{
						i++;
						if (i>=lenp)
						{
							if (j==nmats-1 && i1>=0)
							{
								break;
							}
							else
							{
								nout << "\n\n *** Not enough matrix types on input line\n" << line
								     << '\n';
								nout << str9994a << std::setw(4) << nmats << str9994b << std::endl;
								return;
							}
						}
						if (line[i]!=' ' && line[i]!=',')
						{
							i1 = i;
							// Check that a valid integer was read
							ic = line[i1] - '0';
							if (ic<0 || ic>9)
							{
								nout << "\n\n *** Invalid integer value in column " << std::setw(2)
								     << i+1 << " of input line:\n"  << line << '\n';
								nout << str9994a << std::setw(4) << nmats << str9994b << std::endl;
								return;
							}
							nreq[j] = 10*nreq[j] + ic;
						}
						else if (i1>=0)
						{
							break;
						}
					}
				}
			}
			int nt;
			bool firstt = true;
			for (i=0; i<nmats; i++)
			{
				nt = nreq[i];
				if (nt>0 && nt<=ntypes)
				{
					if (dotype[nt-1])
					{
						if (firstt)
						{
							nout << '\n';
						}
						firstt = false;
						nout << " *** Warning:  duplicate request of matrix type " << std::setw(2)
						     << nt << " for " << path << std::endl;
					}
					dotype[nt-1] = true;
				}
				else
				{
					nout << " *** Invalid type request for " << path << ", type  " << std::setw(4)
					     << nt << ": must satisfy  1 <= type <= " << std::setw(2) << ntypes
					     << std::endl;
				}
			}
		}
	}

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
	 *     subdiagonal elements of $B$ if $\{m}<\{n}$.\n
	 *     not referenced if $\{kd}=0$.
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

	/*! §dchkbd
	 *
	 * §dchkbd checks the singular value decomposition (SVD) routines.\n
	 * §dgebrd reduces a real general $m$ by $n$ matrix $A$ to upper or lower bidiagonal form $B$
	 * by an orthogonal transformation: $Q^T A P = B$ (or $A = Q B P^T$). The matrix $B$ is upper
	 * bidiagonal if $m\ge n$ and lower bidiagonal if $m<n$.\n
	 * §dorgbr generates the orthogonal matrices $Q$ and $P^T$ from §dgebrd. Note that $Q$ and $P$
	 * are not necessarily square.\n
	 * §dbdsqr computes the singular value decomposition of the bidiagonal matrix $B$ as $B=USV^T$.
	 * It is called three times to compute
	 *  1. $B=US_1V^T$, where $S_1$ is the diagonal matrix of singular values and the columns of
	 *     the matrices $U$ and $V$ are the left and right singular vectors, respectively, of $B$.
	 *  2. Same as 1., but the singular values are stored in $S_2$ and the singular vectors are not
	 *     computed.
	 *  3. $A = (UQ)S(P^TV^T)$, the SVD of the original matrix $A$.
	 *  .
	 * In addition, §DBDSQR has an option to apply the left orthogonal matrix $U$ to a matrix $X$,
	 * useful in least squares applications.\n
	 * §dbdsdc computes the singular value decomposition of the bidiagonal matrix $B$ as $B=USV^T$
	 * using divide-and-conquer. It is called twice to compute
	 *  1. $B=US_1V^T$, where $S_1$ is the diagonal matrix of singular values and the columns of
	 *     the matrices $U$ and $V$ are the left and right singular vectors, respectively, of $B$.
	 *  2. Same as 1., but the singular values are stored in $S_2$ and the singular vectors are not
	 *     computed.
	 *  .\n
	 * §dbdsvdx computes the singular value decomposition of the bidiagonal matrix $B$ as $B=USV^T$
	 * using bisection and inverse iteration. It is called six times to compute
	 *  1. $B=US_1V^T$, §RANGE ='A', where $S_1$ is the diagonal matrix of singular values and the
	 *     columns of the matrices $U$ and $V$ are the left and right singular vectors,
	 *     respectively, of $B$.
	 *  2. Same as 1., but the singular values are stored in $S_2$ and the singular vectors are not
	 *     computed.
	 *  3. $B=US_1V^T$, §RANGE ='I', where $S_1$ is the diagonal matrix of singular values and the
	 *     columns of the matrices $U$ and $V$ are the left and right singular vectors,
	 *     respectively, of $B$.
	 *  4. Same as 3., but the singular values are stored in $S_2$ and the singular vectors are not
	 *     computed.
	 *  5. $B=US_1V^T$, §RANGE ='V', where $S_1$ is the diagonal matrix of singular values and the
	 *     columns of the matrices $U$ and $V$ are the left and right singular vectors,
	 *     respectively, of $B$.
	 *  6. Same as 5., but the singular values are stored in $S_2$ and the singular vectors are not
	 *     computed.
	 *  .
	 * For each pair of matrix dimensions ($m$,$n$) and each selected matrix type, an $m$ by $n$
	 * matrix $A$ and an $m$ by §nrhs matrix $X$ are generated.
	 * The problem dimensions are as follows\n
	 * $\begin{tabular}{ll}
	 *     \(A\):           & \(m\times n\)                                             \\
	 *     \(Q\):           & \(m\times\min(m,n)\) (but \(m\times m\) if \(\{nrhs}>0\)) \\
	 *     \(P\):           & \(\min(m,n)\times n\)                                     \\
	 *     \(B\):           & \(\min(m,n)\times\min(m,n)\)                              \\
	 *     \(U\),\(V\):     & \(\min(m,n)\times\min(m,n)\)                              \\
	 *     \(S_1\),\(S_2\): & diagonal, order \(\min(m,n)\)                             \\
	 *     \(X\):           & \(m\times\{nrhs}\)    \end{tabular}$\n
	 * For each generated matrix, 14 tests are performed:
	 * - Test §dgebrd and §dorgbr \n
	 *     1)  $\frac{|A - Q B \{Pt}   |}{|A|\max(m,n)\{ulp}}$, $\{Pt}=P^T$\n
	 *     2)  $\frac{|I - Q^T Q       |}{m\,\{ulp}}$\n
	 *     3)  $\frac{|I - \{Pt}\{Pt}^T|}{n\,\{ulp}}$
	 * - Test §dbdsqr on bidiagonal matrix $B$\n
	 *     4)  $\frac{|B - U S_1 \{Vt} |}{|B|\min(m,n)\{ulp}}$, $\{Vt}=V^T$\n
	 *     5)  $\frac{|Y - U Z         |}{|Y|\max(\min(m,n),k)\{ulp}}$,
	 *         where $Y=Q^TX$ and $Z=U^TY$.\n
	 *     6)  $\frac{|I - U^T U       |}{\min(m,n)\{ulp}}$\n
	 *     7)  $\frac{|I - \{Vt}\{Vt}^T|}{\min(m,n)\{ulp}}$\n
	 *     8)  §s1 contains $\min(m,n)$ nonnegative values in decreasing order.
	 *         (Return 0 if §true, $1/\{ulp}$ if §false.)\n
	 *     9)  $\frac{|S_1-S_2|}{|S_1|\{ulp}}$, where $S_2$ is computed without computing $U$ and
	 *         $V$.\n
	 *     10) 0 if the true singular values of $B$ are within §thresh of those in §s1.
	 *         $2\{thresh}$ if they are not. (Tested using §dsvdch)
	 * - Test §dbdsqr on matrix $A$\n
	 *     11) $\frac{|A - (QU) S (\{Vt}\{Pt}) |}{|A|\max(m,n)\{ulp}}$\n
	 *     12) $\frac{|X - (QU) Z              |}{|X|\max(m,k)\{ulp}}$\n
	 *     13) $\frac{|I - (QU)^T (QU)         |}{m\,\{ulp}}$\n
	 *     14) $\frac{|I - (\{Vt}\{Pt})(\{Pt}^T\{Vt}^T)|}{n\,\{ulp}}$
	 * - Test §dbdsdc on bidiagonal matrix $B$\n
	 *     15) $\frac{|B - U S_1 \{Vt} |}{|B|\min(m,n)\{ulp}}$, $\{Vt}=V^T$\n
	 *     16) $\frac{|I - U^T U       |}{\min(m,n)\{ulp}}$\n
	 *     17) $\frac{|I - \{Vt}\{Vt}^T|}{\min(m,n)\{ulp}}$\n
	 *     18) §s1 contains $\min(m,n)$ nonnegative values in decreasing order.
	 *         (Return 0 if §true, $1/\{ulp}$ if §false.)\n
	 *     19) $\frac{|S_1-S_2|}{|S_1|\{ulp}}$, where $S_2$ is computed without computing $U$ and
	 *         $V$.
	 * - Test §dbdsvdx on bidiagonal matrix $B$\n
	 *     20) $\frac{|B - U S_1 \{Vt}   |}{|B|\min(m,n)\{ulp}}$, $\{Vt}=V^T$\n
	 *     21) $\frac{|I - U^T U         |}{   \min(m,n)\{ulp}}$\n
	 *     22) $\frac{|I - \{Vt} \{Vt}^T |}{   \min(m,n)\{ulp}}$\n
	 *     23) §s1 contains $\min(m,n)$ nonnegative values in decreasing order.
	 *         (Return 0 if §true, $1/\{ulp}$ if §false.)\n
	 *     24) $\frac{|S_1 - s2 |}{|S_1|\{ulp}}$, where $S_2$ is computed without computing $U$ and
	 *         $V$.\n
	 *     25) $\frac{|S_1 - U^T B \{Vt}^T |}{|S|\, n\,\{ulp}}$ &emsp; §dbdsvdx('V', 'I')\n
	 *     26) $\frac{|I   - U^T U         |}{\min(m,n)\{ulp}}$\n
	 *     27) $\frac{|I   - \{Vt} \{Vt}^T |}{\min(m,n)\{ulp}}$\n
	 *     28) §s1 contains $\min(m,n)$ nonnegative values in decreasing order.
	 *         (Return 0 if §true, $1/\{ulp}$ if §false.)\n
	 *     29) $\frac{|S_1 - s2 |}{|S_1|\{ulp}}$, where $S_2$ is computed without computing $U$ and
	 *         $V$.\n
	 *     30) $\frac{|S_1 - U^T B \{Vt}^T |}{|S_1|\,n\,\{ulp}}$ &emsp; §dbdsvdx('V', 'V')\n
	 *     31) $\frac{|I   - U^T U         |}{\min(m,n) \{ulp}}$\n
	 *     32) $\frac{|I   - \{Vt} \{Vt}^T |}{\min(m,n) \{ulp}}$\n
	 *     33) §s1 contains $\min(m,n)$ nonnegative values in decreasing order.
	 *         (Return 0 if §true, $1/\{ulp}$ if §false.)\n
	 *     34) $\frac{|S_1 - s2 |}{|S_1|\{ulp}}$, where $S_2$ is computed without computing $U$ and
	 *         $V$.
	 * .
	 * The possible matrix types are
	 * 1. The zero matrix.
	 * 2. The identity matrix.
	 * 3. A diagonal matrix with evenly spaced entries $1, \ldots, \{ulp}$ and random signs.
	 *    ($\{ulp}=(\text{first number larger than }1) - 1$)
	 * 4. A diagonal matrix with geometrically spaced entries $1, \ldots, \{ulp}$ and random signs.
	 * 5. A diagonal matrix with "clustered" entries $1, \{ulp}, \ldots, \{ulp}$ and random signs.
	 * 6. Same as 3., but multiplied by $\sqrt{\text{overflow threshold}}$
	 * 7. Same as 3., but multiplied by $\sqrt{\text{underflow threshold}}$
	 * 8. A matrix of the form $UDV$, where $U$ and $V$ are orthogonal and $D$ has evenly spaced
	 *    entries $1, \ldots, \{ulp}$ with random signs on the diagonal.
	 * 9. A matrix of the form $UDV$, where $U$ and $V$ are orthogonal and $D$ has geometrically
	 *    spaced entries $1, \ldots, \{ulp}$ with random signs on the diagonal.
	 * 10. A matrix of the form $UDV$, where $U$ and $V$ are orthogonal and $D$ has "clustered"
	 *     entries $1, \{ulp}, \ldots, \{ulp}$ with random signs on the diagonal.
	 * 11. Same as 8., but multiplied by $\sqrt{\text{overflow threshold}}$
	 * 12. Same as 8., but multiplied by $\sqrt{\text{underflow threshold}}$
	 * 13. Rectangular matrix with random entries chosen from (-1,1).
	 * 14. Same as 13., but multiplied by $\sqrt{\text{overflow threshold}}$
	 * 15. Same as 13., but multiplied by $\sqrt{\text{underflow threshold}}$
	 * .\n
	 * Special case:
	 * 16. A bidiagonal matrix with random entries chosen from a logarithmic distribution on
	 *     $[\{ulp}^2,\{ulp}^{-2}]$ (i.e., each entry is $e^x$, where $x$ is chosen uniformly on
	 *     $[2\log(\{ulp}), -2\log(\{ulp})]$.)\n For *this* type:
	 *     - §dgebrd is not called to reduce it to bidiagonal form.
	 *     - the bidiagonal is $\min(m,n)\times\min(m,n)$; if $m<n$, the matrix will be lower
	 *       bidiagonal, otherwise upper.
	 *     - only tests 5-8 and 14 are performed.
	 *     .
	 * .
	 * A subset of the full set of matrix types may be selected through the logical array §dotype.
	 * \param[in] nsizes
	 *     The number of values of $m$ and $n$ contained in the vectors §mval and §nval.
	 *     The matrix sizes are used in pairs $(m,n)$.
	 *
	 * \param[in] mval
	 *     an integer array, dimension (§nsizes)\n The values of the matrix row dimension $m$.
	 *
	 * \param[in] nval
	 *     an integer array, dimension (§nsizes)\n The values of the matrix column dimension $n$.
	 *
	 * \param[in] ntypes
	 *     The number of elements in §dotype. If it is zero, §dchkbd does nothing. It must be at
	 *     least zero.\n If it is $\{MAXTYP}+1$ and §nsizes is 1, then an additional type,
	 *     $\{MAXTYP}+1$ is defined, which is to use whatever matrices are in $A$ and $B$.
	 *     This is only useful if $\{dotype}[0:\{MAXTYP}-1]$ is §false and $\{dotype}[\{MAXTYP}]$
	 *     is §true.
	 *
	 * \param[in] dotype
	 *     a boolean array, dimension (§ntypes)\n
	 *     If $\{dotype}[j]$ is §true, then for each size $(m,n)$, a matrix of type $j+1$ will be
	 *     generated. If §ntypes is smaller than the maximum number of types defined
	 *     (constant §MAXTYP), then types $\{ntypes}+1$ through §MAXTYP will not be generated. If
	 *     §ntypes is larger than §MAXTYP, $\{dotype}[\{MAXTYP}]$ through $\{dotype}[\{ntypes}-1]$
	 *     will be ignored.
	 *
	 * \param[in] nrhs
	 *     The number of columns in the "right-hand side" matrices $X$, $Y$, and $Z$, used in
	 *     testing §dbdsqr. If $\{nrhs}=0$, then the operations on the right-hand side will not be
	 *     tested. §nrhs must be at least 0.
	 *
	 * \param[in,out] iseed
	 *     an integer array, dimension (4)\n
	 *     On entry §iseed specifies the seed of the random number generator. The array elements
	 *     should be between 0 and 4095; if not they will be reduced modulo 4096. Also,
	 *     $\{iseed}[3]$ must be odd. The values of §iseed are changed on exit, and can be used in
	 *     the next call to §dchkbd to continue the same random number sequence.
	 *
	 * \param[in] thresh
	 *     The threshold value for the test ratios. A result is included in the output file if
	 *     $\{result}\ge\{thresh}$. To have every test ratio printed, use $\{thresh}=0$. Note that
	 *     the expected value of the test ratios is O(1), so §thresh should be a reasonably small
	 *     multiple of 1, e.g., 10 or 100.
	 *
	 * \param[out] A
	 *     an array, dimension (§lda,§nmax)\n where §nmax is the maximum value of $n$ in §nval.
	 *
	 * \param[in] lda
	 *     The leading dimension of the array §A.
	 *     $\{lda}\ge\max(1,\{mmax})$, where §mmax is the maximum value of $m$ in §mval.
	 *
	 * \param[out] bd, be, s1, s2 arrays,   dimension ($\max(\min(\{mval}[j],\{nval}[j]))$)
	 * \param[out] X              an array, dimension (§ldx,§nrhs)
	 * \param[in]  ldx
	 *     The leading dimension of the arrays §X, §Y, and §Z.  $\{ldx}\ge\max(1,\{mmax})$
	 *
	 * \param[out] Y, Z arrays,   dimension (§ldx,§nrhs)
	 * \param[out] Q    an array, dimension (§ldq,§mmax)
	 * \param[in]  ldq  The leading dimension of the array $Q$. $\{ldq}\ge\max(1,\{mmax})$.
	 * \param[out] Pt   an array, dimension (§ldpt,§nmax)
	 * \param[in]  ldpt
	 *     The leading dimension of the arrays §Pt, §U, and §Vt.
	 *     $\{ldpt}\ge\max(1,\max(\min(\{mval}[j],\{nval}[j])))$.
	 *
	 * \param[out] U, Vt arrays, dimension (§ldpt,$\max(\min(\{mval}[j],\{nval}[j]))$)
	 * \param[out] work  an array, dimension (§lwork)
	 * \param[in]  lwork
	 *     The number of entries in §work. This must be at least $3(m+n)$ and
	 *     $m(m+\max(m,n,k)+1) + n\,\min(m,n)$ for all pairs $(m,n)=(\{mval}[j],\{nval}[j])$
	 *
	 * \param[out] iwork an integer array, dimension at least $8\min(m,n)$
	 * \param[in]  nout
	 *     the output stream for printing out error messages
	 *     (e.g., if a routine returns §iinfo not equal to 0.)
	 *
	 * \param[out] info
	 *     If 0, then everything ran OK.\n
	 *     $\begin{tabular}{rl}
	 *          -1: & \(\{nsizes}<0\)                          \\
	 *          -2: & Some \(\{mval}[j]<0\)                    \\
	 *          -3: & Some \(\{nval}[j]<0\)                    \\
	 *          -4: & \(\{ntypes}<0\)                          \\
	 *          -6: & \(\{nrhs}  <0\)                          \\
	 *          -8: & \(\{thresh}<0\)                          \\
	 *         -11: & \(\{lda}<1\) or \(\{lda} < \{mmax}\), where §mmax is \(\max(\{mval}[j])\). \\
	 *         -17: & \(\{ldb}<1\) or \(\{ldb} < \{mmax}\).   \\
	 *         -21: & \(\{ldq}<1\) or \(\{ldq} < \{mmax}\).   \\
	 *         -23: & \(\{ldpt}<1\) or \(\{ldpt}< \{mnmax}\)   \\
	 *         -27: & \{lwork} too small.                      \end{tabular}$\n
	 *     If §dlatmr, §dlatms, §dgebrd, §dorgbr, or §dbdsqr returns an error code,
	 *     the absolute value of it is returned.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016
	 * \remark
	 *     Some Local Variables and Parameters:\n
	 *         $\begin{tabular}{ll}
	 *             \{MAXTYP}        &  The number of types defined.                             \\
	 *             \{ntest}         & The number of tests performed, or which can be performed so
	 *                                far, for the current matrix.                              \\
	 *             \{mmax}          & Largest value in \{mval}.                                 \\
	 *             \{nmax}          & Largest value in \{nval}.                                 \\
	 *             \{mnmin}         & \(\min(\{mval}[j], \{nval}[j])\)
	 *                                (the dimension of the bidiagonal matrix.)                 \\
	 *             \{mnmax}         & The maximum value of \{mnmin} for \(j=1,\ldots,\{nsizes}\)\\
	 *             \{nfail}         & The number of tests which have exceeded \{thresh}         \\
	 *             \{cond}, \{imode}& Values to be passed to the matrix generators.             \\
	 *             \{anorm}         & Norm of \{A}; passed to matrix generators.                \\
	 *             \{ovfl}, \{unfl} & Overflow and underflow thresholds.                        \\
	 *             \{rtovfl},\{rtunfl} & Square roots of the previous 2 values.                 \\
	 *             \{ulp}, ulpinv   & Finest relative precision and its inverse. \end{tabular}$\n
	 *     The following four arrays decode §jtype: \n
	 *         $\begin{tabular}{ll}
	 *             \(\{KTYPE}[j]\) & The general type (1-10).                                \\
	 *             \(\{KMODE}[j]\) & The \{mode} value to be passed to the matrix generator. \\
	 *             \(\{KMAGN}[j]\) & The order of magnitude (O(1), O(\(\sqrt{\{overflow}}\)),
	 *                               O(\(\sqrt{\{underflow}}\))) \end{tabular}$                  */
	void dchkbd(int const nsizes, int const* const mval, int const* const nval, int const ntypes,
	            bool const* const dotype, int const nrhs, int* const iseed, real const thresh,
	            real* const A, int const lda, real* const bd, real* const be, real* const s1,
	            real* const s2, real* const X, int const ldx, real* const Y, real* const Z,
	            real* const Q, int const ldq, real* const Pt, int const ldpt, real* const U,
	            real* const Vt, real* const work, int const lwork, int* const iwork,
	            std::ostream& nout, int& info)
	{
		real const HALF = real(0.5);
		int const MAXTYP = 16;
		int const KTYPE[MAXTYP] = {1, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 9, 9, 9, 10};
		int const KMAGN[MAXTYP] = {1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3,  0};
		int const KMODE[MAXTYP] = {0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0,  0};
		// Check for errors
		info = 0;
		bool badmm = false;
		bool badnn = false;
		int mmax   = 1;
		int nmax   = 1;
		int mnmax  = 1;
		int minwrk = 1;
		int j;
		for (j=0; j<nsizes; j++)
		{
			mmax = std::max(mmax, mval[j]);
			if (mval[j]<0)
			{
				badmm = true;
			}
			nmax = std::max(nmax, nval[j]);
			if (nval[j]<0)
			{
				badnn = true;
			}
			mnmax = std::max(mnmax, std::min(mval[j], nval[j]));
			minwrk = std::max(std::max(minwrk, 3*(mval[j]+nval[j])),
			                  mval[j]*(mval[j]+std::max(std::max(mval[j], nval[j]), nrhs)+1)
			                  +nval[j]*std::min(nval[j], mval[j]));
		}
		// Check for errors
		if (nsizes<0)
		{
			info = -1;
		}
		else if (badmm)
		{
			info = -2;
		}
		else if (badnn)
		{
			info = -3;
		}
		else if (ntypes<0)
		{
			info = -4;
		}
		else if (nrhs<0)
		{
			info = -6;
		}
		else if (lda<mmax)
		{
			info = -11;
		}
		else if (ldx<mmax)
		{
			info = -17;
		}
		else if (ldq<mmax)
		{
			info = -21;
		}
		else if (ldpt<mnmax)
		{
			info = -23;
		}
		else if (minwrk>lwork)
		{
			info = -27;
		}
		if (info!=0)
		{
			this->xerbla("DCHKBD", -info);
			return;
		}
		// Initialize constants
		char path[4];
		path[0] = 'D'; //Double precision
		std::strncpy(&path[1], "BD", 3);
		int nfail   = 0;
		int ntest   = 0;
		real unfl   = this->dlamch("Safe minimum");
		real ovfl   = this->dlamch("Overflow");
		this->dlabad(unfl, ovfl);
		real ulp    = this->dlamch("Precision");
		real ulpinv = ONE / ulp;
		int log2ui  = int(std::log(ulpinv)/std::log(TWO));
		real rtunfl = std::sqrt(unfl);
		real rtovfl = std::sqrt(ovfl);
		infoc.info  = 0;
		//real abstol = 2 * unfl;
		char const* str9998a = " DCHKBD: ";
		char const* str9998b = " returned INFO=";
		char const* str9998c = ".\n         M=";
		// Loop over sizes, types
		bool bidiag = false;
		char uplo[2];
		uplo[1] = '\0';
		int i, iinfo, il=0, imode, itemp, itype, iu=0, iwbd=0, iwbe=0, iwbs, iwbz=0, iwwork=0,
		    jsize, jtype, m, mnmin, mnminm, mnmin2=0, mq=0, mtypes, n, ns1, ns2;
		real amninv, anorm=ZERO, cond, temp1, temp2, vl, vu;
		int ioldsd[4], iseed2[4];
		real result[40];
		bool skiptoend = false;
		for (jsize=0; jsize<nsizes; jsize++)
		{
			m = mval[jsize];
			n = nval[jsize];
			mnmin = std::min(m, n);
			mnminm = mnmin - 1;
			amninv = ONE / std::max(std::max(m, n), 1);
			if (nsizes!=1)
			{
				mtypes = std::min(MAXTYP, ntypes);
			}
			else
			{
				mtypes = std::min(MAXTYP+1, ntypes);
			}
			for (jtype=0; jtype<mtypes; jtype++)
			{
				if (!dotype[jtype])
				{
					continue;
				}
				for (j=0; j<4; j++)
				{
					ioldsd[j] = iseed[j];
				}
				for (j=0; j<34; j++)
				{
					result[j] = -ONE;
				}
				uplo[0] = ' ';
				// Compute "A"
				// Control parameters:
				//     KMAGN  KMODE        KTYPE
				// =1  O(1)   clustered 1  zero
				// =2  large  clustered 2  identity
				// =3  small  exponential  (none)
				// =4         arithmetic   diagonal, (w/ eigenvalues)
				// =5         random       symmetric, w/ eigenvalues
				// =6                      nonsymmetric, w/ singular values
				// =7                      random diagonal
				// =8                      random symmetric
				// =9                      random nonsymmetric
				// =10                     random bidiagonal (log. distrib.)
				if (mtypes<=MAXTYP)
				{
					itype = KTYPE[jtype];
					imode = KMODE[jtype];
					// Compute norm
					switch(KMAGN[jtype])
					{
						default:
						case 1:
							anorm = ONE;
							break;
						case 2:
							anorm = (rtovfl*ulp) * amninv;
							break;
						case 3:
							anorm = rtunfl * std::max(m, n) * ulpinv;
							break;
					}
					this->dlaset("Full", lda, n, ZERO, ZERO, A, lda);
					iinfo = 0;
					cond = ulpinv;
					bidiag = false;
					if (itype==1)
					{
						// Zero matrix
						iinfo = 0;
					}
					else if (itype==2)
					{
						// Identity
						for (j=0; j<mnmin; j++)
						{
							A[j+lda*j] = anorm;
						}
					}
					else if (itype==4)
					{
						// Diagonal Matrix, [Eigen]values Specified
						MatGen.dlatms(mnmin, mnmin, "S", iseed, "N", work, imode, cond, anorm, 0,
						              0, "N", A, lda, &work[mnmin], iinfo);
					}
					else if (itype==5)
					{
						// Symmetric, eigenvalues specified
						MatGen.dlatms(mnmin, mnmin, "S", iseed, "S", work, imode, cond, anorm, m,
						              n, "N", A, lda, &work[mnmin], iinfo);
					}
					else if (itype==6)
					{
						// Nonsymmetric, singular values specified
						MatGen.dlatms(m, n, "S", iseed, "N", work, imode, cond, anorm, m, n, "N",
						              A, lda, &work[mnmin], iinfo);
					}
					else if (itype==7)
					{
						// Diagonal, random entries
						MatGen.dlatmr(mnmin, mnmin, "S", iseed, "N", work, 6, ONE, ONE, "T", "N",
						              nullptr, 1, ONE, nullptr, 1, ONE, "N", nullptr, 0, 0, ZERO,
						              anorm, "NO", A, lda, nullptr, iinfo);
					}
					else if (itype==8)
					{
						// Symmetric, random entries
						MatGen.dlatmr(mnmin, mnmin, "S", iseed, "S", work, 6, ONE, ONE, "T", "N",
						              nullptr, 1, ONE, nullptr, 1, ONE, "N", nullptr, m, n, ZERO,
						              anorm, "NO", A, lda, nullptr, iinfo);
					}
					else if (itype==9)
					{
						// Nonsymmetric, random entries
						MatGen.dlatmr(m, n, "S", iseed, "N", work, 6, ONE, ONE, "T", "N", nullptr,
						              1, ONE, nullptr, 1, ONE, "N", nullptr, m, n, ZERO, anorm,
						              "NO", A, lda, nullptr, iinfo);
					}
					else if (itype==10)
					{
						// Bidiagonal, random entries
						temp1 = -TWO * std::log(ulp);
						for (j=0; j<mnmin; j++)
						{
							bd[j] = std::exp(temp1*MatGen.dlarnd(2, iseed));
							if (j<mnminm)
							{
								be[j] = std::exp(temp1*MatGen.dlarnd(2, iseed));
							}
						}
						iinfo = 0;
						bidiag = true;
						if (m>=n)
						{
							uplo[0] = 'U';
						}
						else
						{
							uplo[0] = 'L';
						}
					}
					else
					{
						iinfo = 1;
					}
					if (iinfo==0)
					{
						// Generate Right-Hand Side
						if (bidiag)
						{
							MatGen.dlatmr(mnmin, nrhs, "S", iseed, "N", work, 6, ONE, ONE, "T",
							              "N", nullptr, 1, ONE, nullptr, 1, ONE, "N", nullptr,
							              mnmin, nrhs, ZERO, ONE, "NO", Y, ldx, nullptr, iinfo);
						}
						else
						{
							MatGen.dlatmr(m, nrhs, "S", iseed, "N", work, 6, ONE, ONE, "T", "N",
							              nullptr, 1, ONE, nullptr, 1, ONE, "N", nullptr, m, nrhs,
							              ZERO, ONE, "NO", X, ldx, nullptr, iinfo);
						}
					}
					// Error Exit
					if (iinfo!=0)
					{
						nout << str9998a << "Generator" << str9998b << std::setw(6) << iinfo
						     << str9998c << std::setw(6) << m << ", N=" << std::setw(6) << n
						     << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						return;
					}
				}
				// Call dgebrd and dorgbr to compute B, Q, and P, do tests.
				if (!bidiag)
				{
					// Compute transformations to reduce A to bidiagonal form: B = Q^T * A * P.
					this->dlacpy(" ", m, n, A, lda, Q, ldq);
					this->dgebrd(m, n, Q, ldq, bd, be, work, &work[mnmin], &work[2*mnmin],
					             lwork-2*mnmin, iinfo);
					// Check error code from dgebrd.
					if (iinfo!=0)
					{
						nout << str9998a << "DGEBRD" << str9998b << std::setw(6) << iinfo
						     << str9998c << std::setw(6) << m << ", N=" << std::setw(6) << n
						     << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						return;
					}
					this->dlacpy(" ", m, n, Q, ldq, Pt, ldpt);
					if (m>=n)
					{
						uplo[0] = 'U';
					}
					else
					{
						uplo[0] = 'L';
					}
					// Generate Q
					mq = m;
					if (nrhs<=0)
					{
						mq = mnmin;
					}
					this->dorgbr("Q", m, mq, n, Q, ldq, work, &work[2*mnmin], lwork-2*mnmin,
					             iinfo);
					// Check error code from DORGBR.
					if (iinfo!=0)
					{
						nout << str9998a << "DORGBR(Q)" << str9998b << std::setw(6) << iinfo
						     << str9998c << std::setw(6) << m << ", N=" << std::setw(6) << n
						     << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						return;
					}
					// Generate P'
					this->dorgbr("P", mnmin, n, m, Pt, ldpt, &work[mnmin], &work[2*mnmin],
					             lwork-2*mnmin, iinfo);
					// Check error code from dorgbr.
					if (iinfo!=0)
					{
						nout << str9998a << "DORGBR(P)" << str9998b << std::setw(6) << iinfo
						     << str9998c << std::setw(6) << m << ", N=" << std::setw(6) << n
						     << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						return;
					}
					// Apply Q^T to an m by nrhs matrix X:  Y = Q^T X.
					Blas<real>::dgemm("Transpose", "No transpose", m, nrhs, m, ONE, Q, ldq, X, ldx,
					                  ZERO, Y, ldx);
					// Test 1:  Check the decomposition A := Q * B * Pt
					//      2:  Check the orthogonality of Q
					//      3:  Check the orthogonality of Pt
					dbdt01(m, n, 1, A, lda,  Q,  ldq, bd, be, Pt, ldpt, work, result[0]);
					dort01("Columns", m, mq, Q,  ldq,  work, lwork,           result[1]);
					dort01("Rows", mnmin, n, Pt, ldpt, work, lwork,           result[2]);
				}
				// Use dbdsqr to form the SVD of the bidiagonal matrix B: B = U * s1 * Vt,
				// and compute Z = U' * Y.
				Blas<real>::dcopy(mnmin, bd, 1, s1, 1);
				if (mnmin>0)
				{
					Blas<real>::dcopy(mnminm, be, 1, work, 1);
				}
				this->dlacpy(" ", m, nrhs, Y, ldx, Z, ldx);
				this->dlaset("Full", mnmin, mnmin, ZERO, ONE, U,  ldpt);
				this->dlaset("Full", mnmin, mnmin, ZERO, ONE, Vt, ldpt);
				this->dbdsqr(uplo, mnmin, mnmin, mnmin, nrhs, s1, work, Vt, ldpt, U, ldpt, Z, ldx,
				             &work[mnmin], iinfo);
				// Check error code from dbdsqr.
				skiptoend = false;
				if (iinfo!=0)
				{
					nout << str9998a << "DBDSQR(vects)" << str9998b << std::setw(6) << iinfo
					     << str9998c << std::setw(6) << m << ", N=" << std::setw(6) << n
					     << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=(" << std::setw(5)
					     << ioldsd[0] << ',' << std::setw(5) << ioldsd[1] << ','
					     << std::setw(5) << ioldsd[2] << ',' << std::setw(5) << ioldsd[3]
					     << ')' << std::endl;
					info = std::abs(iinfo);
					if (iinfo<0)
					{
						return;
					}
					else
					{
						result[3] = ulpinv;
						skiptoend = true;
					}
				}
				if (!skiptoend)
				{
					// Use dbdsqr to compute only the singular values of the bidiagonal matrix B;
					// U, Vt, and Z should not be modified.
					Blas<real>::dcopy(mnmin, bd, 1, s2, 1);
					if (mnmin>0)
					{
						Blas<real>::dcopy(mnminm, be, 1, work, 1);
					}
					this->dbdsqr(uplo, mnmin, 0, 0, 0, s2, work, nullptr, ldpt, nullptr, ldpt,
					             nullptr, ldx, &work[mnmin], iinfo);
					// Check error code from dbdsqr.
					if (iinfo!=0)
					{
						nout << str9998a << "DBDSQR(values)" << str9998b << std::setw(6) << iinfo
						     << str9998c << std::setw(6) << m << ", N=" << std::setw(6) << n
						     << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						if (iinfo<0)
						{
							return;
						}
						else
						{
							result[8] = ulpinv;
							skiptoend = true;
						}
					}
				}
				if (!skiptoend)
				{
					// Test 4:  Check the decomposition B := U * s1 * Vt
					//      5:  Check the computation Z := U' * Y
					//      6:  Check the orthogonality of U
					//      7:  Check the orthogonality of Vt
					dbdt03(uplo, mnmin, 1, bd, be, U, ldpt, s1, Vt, ldpt, work,        result[3]);
					dbdt02(mnmin, nrhs, Y, ldx, Z, ldx,         U,  ldpt, work,        result[4]);
					dort01("Columns", mnmin, mnmin,             U,  ldpt, work, lwork, result[5]);
					dort01("Rows",    mnmin, mnmin,             Vt, ldpt, work, lwork, result[6]);
					// Test 8:  Check that the singular values are sorted in non-increasing order
					//          and are non-negative
					result[7] = ZERO;
					for (i=0; i<mnminm; i++)
					{
						if (s1[i]<s1[i+1])
						{
							result[7] = ulpinv;
						}
						if (s1[i]<ZERO)
						{
							result[7] = ulpinv;
						}
					}
					if (mnmin>=1)
					{
						if (s1[mnminm]<ZERO)
						{
							result[7] = ulpinv;
						}
					}
					// Test 9:  Compare DBDSQR with and without singular vectors
					temp2 = ZERO;
					for (j=0; j<mnmin; j++)
					{
						temp1 = std::fabs(s1[j]-s2[j])
						        / std::max(std::sqrt(unfl)*std::max(s1[0], ONE),
						                   ulp*std::max(std::fabs(s1[j]), std::fabs(s2[j])));
						temp2 = std::max(temp1, temp2);
					}
					result[8] = temp2;
					// Test 10:  Sturm sequence test of singular values
					//           Go up by factors of two until it succeeds
					temp1 = thresh * (HALF-ulp);
					for (j=0; j<=log2ui; j++)
					{
						// dsvdch(mnmin, bd, be, s1, temp1, iinfo);
						if (iinfo==0)
						{
							break;
						}
						temp1 *= TWO;
					}
					result[9] = temp1;
					// Use dbdsqr to form the decomposition A = (QU) S (Vt Pt)
					// from the bidiagonal form A = Q B Pt.
					if (!bidiag)
					{
						Blas<real>::dcopy(mnmin, bd, 1, s2, 1);
						if (mnmin>0)
						{
							Blas<real>::dcopy(mnminm, be, 1, work, 1);
						}
						this->dbdsqr(uplo, mnmin, n, m, nrhs, s2, work, Pt, ldpt, Q, ldq, Y, ldx,
						             &work[mnmin], iinfo);
						// Test 11:  Check the decomposition A := Q*U * s2 * Vt*Pt
						//      12:  Check the computation Z := U^T * Q^T * X
						//      13:  Check the orthogonality of Q*U
						//      14:  Check the orthogonality of Vt*Pt
						dbdt01(m, n, 0, A, lda, Q, ldq, s2, nullptr, Pt, ldpt, work, result[10]);
						dbdt02(m, nrhs, X, ldx, Y, ldx,              Q,  ldq,  work, result[11]);
						dort01("Columns", m, mq, Q,  ldq,  work, lwork,              result[12]);
						dort01("Rows", mnmin, n, Pt, ldpt, work, lwork,              result[13]);
					}
					// Use dbdsdc to form the SVD of the bidiagonal matrix B: B = U * s1 * Vt
					Blas<real>::dcopy(mnmin, bd, 1, s1, 1);
					if (mnmin>0)
					{
						Blas<real>::dcopy(mnminm, be, 1, work, 1);
					}
					this->dlaset("Full", mnmin, mnmin, ZERO, ONE, U,  ldpt);
					this->dlaset("Full", mnmin, mnmin, ZERO, ONE, Vt, ldpt);
					this->dbdsdc(uplo, "I", mnmin, s1, work, U, ldpt, Vt, ldpt, nullptr, nullptr,
					             &work[mnmin], iwork, iinfo);
					// Check error code from dbdsdc.
					if (iinfo!=0)
					{
						nout << str9998a << "DBDSDC(vects)" << str9998b << std::setw(6) << iinfo
						     << str9998c << std::setw(6) << m << ", N=" << std::setw(6) << n
						     << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						if (iinfo<0)
						{
							return;
						}
						else
						{
							result[14] = ulpinv;
							skiptoend = true;
						}
					}
				}
				if (!skiptoend)
				{
					// Use dbdsdc to compute only the singular values of the bidiagonal matrix B;
					// U and Vt should not be modified.
					Blas<real>::dcopy(mnmin, bd, 1, s2, 1);
					if (mnmin>0)
					{
						Blas<real>::dcopy(mnminm, be, 1, work, 1);
					}
					this->dbdsdc(uplo, "N", mnmin, s2, work, nullptr, 1, nullptr, 1, nullptr,
					             nullptr, &work[mnmin], iwork, iinfo);
					// Check error code from dbdsdc.
					if (iinfo!=0)
					{
						nout << str9998a << "DBDSDC(values)" << str9998b << std::setw(6) << iinfo
						     << str9998c << std::setw(6) << m << ", N=" << std::setw(6) << n
						     << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						if (iinfo<0)
						{
							return;
						}
						else
						{
							result[17] = ulpinv;
							skiptoend = true;
						}
					}
				}
				if (!skiptoend)
				{
					// Test 15: Check the decomposition B = U * s1 * Vt
					//      16: Check the orthogonality of U
					//      17: Check the orthogonality of Vt
					dbdt03(uplo, mnmin, 1, bd, be,  U,  ldpt, s1, Vt, ldpt, work, result[14]);
					dort01("Columns", mnmin, mnmin, U,  ldpt, work, lwork,        result[15]);
					dort01("Rows",    mnmin, mnmin, Vt, ldpt, work, lwork,        result[16]);
					// Test 18: Check that the singular values are sorted incnon-increasing order
					//          and are non-negative
					result[17] = ZERO;
					for (i=0; i<mnminm; i++)
					{
						if (s1[i]<s1[i+1])
						{
							result[17] = ulpinv;
						}
						if (s1[i]<ZERO)
						{
							result[17] = ulpinv;
						}
					}
					if (mnmin>=1)
					{
						if (s1[mnminm]<ZERO)
						{
							result[17] = ulpinv;
						}
					}
					// Test 19: Compare dbdsqr with and without singular vectors
					temp2 = ZERO;
					for (j=0; j<mnmin; j++)
					{
						temp1 = std::fabs(s1[j]-s2[j])
						        / std::max(std::sqrt(unfl)*std::max(s1[0], ONE),
						                   ulp*std::max(std::fabs(s1[0]), std::fabs(s2[0])));
						temp2 = std::max(temp1, temp2);
					}
					result[18] = temp2;
					// Use dbdsvdx to compute the SVD of the bidiagonal matrix B: B = U * s1 * Vt
					if (jtype==9 || jtype==15)
					{
						// =================================
						// Matrix types temporarily disabled
						// =================================
						for (j=19; j<34; j++)
						{
							result[j] = ZERO;
						}
						skiptoend = true;
					}
				}
				if (!skiptoend)
				{
					iwbs   = 0;
					iwbd   = iwbs + mnmin;
					iwbe   = iwbd + mnmin;
					iwbz   = iwbe + mnmin;
					iwwork = iwbz + 2*mnmin*(mnmin+1);
					mnmin2 = std::max(1, mnmin*2);
					Blas<real>::dcopy(mnmin, bd, 1, &work[iwbd], 1);
					if (mnmin>0)
					{
						Blas<real>::dcopy(mnminm, be, 1, &work[iwbe], 1);
					}
					this->dbdsvdx(uplo, "V", "A", mnmin, &work[iwbd], &work[iwbe], ZERO, ZERO, 0,
					              0, ns1, s1, &work[iwbz], mnmin2, &work[iwwork], iwork, iinfo);
					// Check error code from dbdsvdx.
					if (iinfo!=0)
					{
						nout << str9998a << "DBDSVDX(vects,A)" << str9998b << std::setw(6) << iinfo
						     << str9998c << std::setw(6) << m << ", N=" << std::setw(6) << n
						     << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						if (iinfo<0)
						{
							return;
						}
						else
						{
							result[19] = ulpinv;
							skiptoend = true;
						}
					}
				}
				if (!skiptoend)
				{
					j = iwbz;
					for (i=0; i<ns1; i++)
					{
						Blas<real>::dcopy(mnmin, &work[j], 1, &U[ldpt*i], 1);
						j += mnmin;
						Blas<real>::dcopy(mnmin, &work[j], 1, &Vt[i],  ldpt);
						j += mnmin;
					}
					// Use dbdsvdx to compute only the singular values of the bidiagonal matrix B;
					// U and Vt should not be modified.
					if (jtype==8)
					{
						// =================================
						// Matrix types temporarily disabled
						// =================================
						result[23] = ZERO;
						skiptoend = true;
					}
				}
				if (!skiptoend)
				{
					Blas<real>::dcopy(mnmin, bd, 1, &work[iwbd], 1);
					if (mnmin>0)
					{
						Blas<real>::dcopy(mnminm, be, 1, &work[iwbe], 1);
					}
					this->dbdsvdx(uplo, "N", "A", mnmin, &work[iwbd], &work[iwbe], ZERO, ZERO, 0,
					              0, ns2, s2, nullptr, mnmin2, &work[iwwork], iwork, iinfo);
					// Check error code from dbdsvdx.
					if (iinfo!=0)
					{
						nout << str9998a << "DBDSVDX(values,A)" << str9998b << std::setw(6)
						     << iinfo << str9998c << std::setw(6) << m << ", N=" << std::setw(6)
						     << n << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						if (iinfo<0)
						{
							return;
						}
						else
						{
							result[23] = ulpinv;
							skiptoend = true;
						}
					}
				}
				if (!skiptoend)
				{
					// Save s1 for tests 30-34.
					Blas<real>::dcopy(mnmin, s1, 1, &work[iwbs], 1);
					// Test 20: Check the decomposition B = U * s1 * Vt
					//      21: Check the orthogonality of U
					//      22: Check the orthogonality of Vt
					//      23: Check that the singular values are sorted in non-increasing order
					//          and are non-negative
					//      24: Compare dbdsvdx with and without singular vectors
					dbdt03(uplo, mnmin, 1, bd, be,  U,  ldpt, s1, Vt, ldpt, &work[iwbs+mnmin],
					       result[19]);
					dort01("Columns", mnmin, mnmin, U,  ldpt, &work[iwbs+mnmin], lwork-mnmin,
					       result[20]);
					dort01("Rows",    mnmin, mnmin, Vt, ldpt, &work[iwbs+mnmin], lwork-mnmin,
					       result[21]);
					result[22] = ZERO;
					for (i=0; i<mnminm; i++)
					{
						if (s1[i]<s1[i+1])
						{
							result[22] = ulpinv;
						}
						if (s1[i]<ZERO)
						{
							result[22] = ulpinv;
						}
					}
					if (mnmin>=1)
					{
						if (s1[mnminm]<ZERO)
						{
							result[22] = ulpinv;
						}
					}
					temp2 = ZERO;
					for (j=0; j<mnmin; j++)
					{
						temp1 = std::fabs(s1[j]-s2[j])
						        / std::max(std::sqrt(unfl)*std::max(s1[0], ONE),
						                   ulp*std::max(std::fabs(s1[0]), std::fabs(s2[0])));
						temp2 = std::max(temp1, temp2);
					}
					result[23] = temp2;
					anorm = s1[0];
					// Use dbdsvdx with RANGE='I': choose random values for il and iu, and ask for
					// the il-th through iu-th singular values and corresponding vectors.
					for (i=0; i<4; i++)
					{
						iseed2[i] = iseed[i];
					}
					if (mnmin<=1)
					{
						il = 0;
						iu = mnminm;
					}
					else
					{
						il = int(mnminm*MatGen.dlarnd(1, iseed2));
						iu = int(mnminm*MatGen.dlarnd(1, iseed2));
						if (iu<il)
						{
							itemp = iu;
							iu    = il;
							il    = itemp;
						}
					}
					Blas<real>::dcopy(mnmin, bd, 1, &work[iwbd], 1);
					if (mnmin>0)
					{
						Blas<real>::dcopy(mnminm, be, 1, &work[iwbe], 1);
					}
					this->dbdsvdx(uplo, "V", "I", mnmin, &work[iwbd], &work[iwbe], ZERO, ZERO, il,
					              iu, ns1, s1, &work[iwbz], mnmin2, &work[iwwork], iwork, iinfo);
					// Check error code from dbdsvdx.
					if (iinfo!=0)
					{
						nout << str9998a << "DBDSVDX(vects,I)" << str9998b << std::setw(6) << iinfo
						     << str9998c << std::setw(6) << m << ", N=" << std::setw(6) << n
						     << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						if (iinfo<0)
						{
							return;
						}
						else
						{
							result[24] = ulpinv;
							skiptoend = true;
						}
					}
				}
				if (!skiptoend)
				{
					j = iwbz;
					for (i=0; i<ns1; i++)
					{
						Blas<real>::dcopy(mnmin, &work[j], 1, &U[ldpt*i], 1);
						j += mnmin;
						Blas<real>::dcopy(mnmin, &work[j], 1, &Vt[i],  ldpt);
						j += mnmin;
					}
					// Use dbdsvdx to compute only the singular values of the
					// bidiagonal matrix B;  U and Vt should not be modified.
					Blas<real>::dcopy(mnmin, bd, 1, &work[iwbd], 1);
					if (mnmin>0)
					{
						Blas<real>::dcopy(mnminm, be, 1, &work[iwbe], 1);
					}
					this->dbdsvdx(uplo, "N", "I", mnmin, &work[iwbd], &work[iwbe], ZERO, ZERO, il,
					              iu, ns2, s2, nullptr, mnmin2, &work[iwwork], iwork, iinfo);
					// Check error code from DBDSVDX.
					if (iinfo!=0)
					{
						nout << str9998a << "DBDSVDX(values,I)" << str9998b << std::setw(6)
						     << iinfo << str9998c << std::setw(6) << m << ", N=" << std::setw(6)
						     << n << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						if (iinfo<0)
						{
							return;
						}
						else
						{
							result[28] = ulpinv;
							skiptoend = true;
						}
					}
				}
				if (!skiptoend)
				{
					// Test 25: Check s1 - U^T * B * Vt^T
					//      26: Check the orthogonality of U
					//      27: Check the orthogonality of Vt
					//      28: Check that the singular values are sorted in non-increasing order
					//          and are non-negative
					//      29: Compare dbdsvdx with and without singular vectors
					this->dbdt04(uplo, mnmin, bd, be, s1, ns1, U, ldpt, Vt, ldpt,
					             &work[iwbs+mnmin], result[24]);
					dort01("Columns", mnmin, ns1, U,  ldpt, &work[iwbs+mnmin], lwork-mnmin,
					       result[25]);
					dort01("Rows",    ns1, mnmin, Vt, ldpt, &work[iwbs+mnmin], lwork-mnmin,
					       result[26]);
					result[27] = ZERO;
					for (i=0; i<ns1-1; i++)
					{
						if (s1[i]<s1[i+1])
						{
							result[27] = ulpinv;
						}
						if (s1[i]<ZERO)
						{
							result[27] = ulpinv;
						}
					}
					if (ns1>=1)
					{
						if (s1[ns1-1]<ZERO)
						{
							result[27] = ulpinv;
						}
					}
					temp2 = ZERO;
					for (j=0; j<ns1; j++)
					{
						temp1 = std::fabs(s1[j]-s2[j])
						        / std::max(std::sqrt(unfl)*std::max(s1[0], ONE),
						                   ulp*std::max(std::fabs(s1[0]), std::fabs(s2[0])));
						temp2 = std::max(temp1, temp2);
					}
					result[28] = temp2;
					// Use dbdsvdx with RANGE='V': determine the values vl and vu of the il-th and
					// iu-th singular values and ask for all singular values in this range.
					Blas<real>::dcopy(mnmin, &work[iwbs], 1, s1, 1);
					if (mnmin>0)
					{
						if (il!=0)
						{
							vu = s1[il] + std::max(HALF*std::fabs(s1[il]-s1[il-1]),
							                       std::max(ulp*anorm, TWO*rtunfl));
						}
						else
						{
							vu = s1[0] + std::max(HALF*std::fabs(s1[mnminm]-s1[0]),
							                      std::max(ulp*anorm, TWO*rtunfl));
						}
						if (iu!=ns1-1)
						{
							vl = s1[iu] - std::max(std::max(ulp*anorm, TWO*rtunfl),
							                       HALF*std::fabs(s1[iu+1]-s1[iu]));
						}
						else
						{
							vl = s1[ns1-1] - std::max(std::max(ulp*anorm, TWO*rtunfl),
							                          HALF*std::fabs(s1[mnminm]-s1[0]));
						}
						vl = std::max(vl, ZERO);
						vu = std::max(vu, ZERO);
						if (vl>=vu)
						{
							vu = std::max(vu*2, vu+vl+HALF);
						}
					}
					else
					{
						vl = ZERO;
						vu = ONE;
					}
					Blas<real>::dcopy(mnmin, bd, 1, &work[iwbd], 1);
					if (mnmin>0)
					{
						Blas<real>::dcopy(mnminm, be, 1, &work[iwbe], 1);
					}
					this->dbdsvdx(uplo, "V", "V", mnmin, &work[iwbd], &work[iwbe], vl, vu, 0, 0,
					              ns1, s1, &work[iwbz], mnmin2, &work[iwwork], iwork, iinfo);
					// Check error code from dbdsvdx.
					if (iinfo!=0)
					{
						nout << str9998a << "DBDSVDX(vects,V)" << str9998b << std::setw(6) << iinfo
						     << str9998c << std::setw(6) << m << ", N=" << std::setw(6) << n
						     << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						if (iinfo<0)
						{
							return;
						}
						else
						{
							result[29] = ulpinv;
							skiptoend = true;
						}
					}
				}
				if (!skiptoend)
				{
					j = iwbz;
					for (i=0; i<ns1; i++)
					{
						Blas<real>::dcopy(mnmin, &work[j], 1, &U[ldpt*i], 1);
						j += mnmin;
						Blas<real>::dcopy(mnmin, &work[j], 1, &Vt[i],  ldpt);
						j += mnmin;
					}
					// Use dbdsvdx to compute only the singular values of the bidiagonal matrix B;
					// U and Vt should not be modified.
					Blas<real>::dcopy(mnmin, bd, 1, &work[iwbd], 1);
					if (mnmin>0)
					{
						Blas<real>::dcopy(mnminm, be, 1, &work[iwbe], 1);
					}
					this->dbdsvdx(uplo, "N", "V", mnmin, &work[iwbd], &work[iwbe], vl, vu, 0, 0,
					              ns2, s2, nullptr, mnmin2, &work[iwwork], iwork, iinfo);
					// Check error code from dbdsvdx.
					if (iinfo!=0)
					{
						nout << str9998a << "DBDSVDX(values,V)" << str9998b << std::setw(6)
						     << iinfo << str9998c << std::setw(6) << m << ", N=" << std::setw(6)
						     << n << ", JTYPE=" << std::setw(6) << jtype+1 << ", ISEED=("
						     << std::setw(5) << ioldsd[0] << ',' << std::setw(5) << ioldsd[1]
						     << ',' << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
						     << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						if (iinfo<0)
						{
							return;
						}
						else
						{
							result[33] = ulpinv;
							skiptoend = true;
						}
					}
				}
				if (!skiptoend)
				{
					// Test 30: Check s1 - U^T * B * Vt^T
					//      31: Check the orthogonality of U
					//      32: Check the orthogonality of Vt
					//      33: Check that the singular values are sorted in non-increasing order
					//          and are non-negative
					//      34: Compare dbdsvdx with and without singular vectors
					this->dbdt04(uplo, mnmin, bd, be, s1, ns1, U, ldpt, Vt, ldpt,
					             &work[iwbs+mnmin], result[29]);
					dort01("Columns", mnmin, ns1, U,  ldpt, &work[iwbs+mnmin], lwork-mnmin,
					       result[30]);
					dort01("Rows",    ns1, mnmin, Vt, ldpt, &work[iwbs+mnmin], lwork-mnmin,
					       result[31]);
					result[32] = ZERO;
					for (i=0; i<ns1-1; i++)
					{
						if (s1[i]<s1[i+1])
						{
							result[27] = ulpinv;
						}
						if (s1[i]<ZERO)
						{
							result[27] = ulpinv;
						}
					}
					if (ns1>=1)
					{
						if (s1[ns1-1]<ZERO)
						{
							result[27] = ulpinv;
						}
					}
					temp2 = ZERO;
					for (j=0; j<ns1; j++)
					{
						temp1 = std::fabs(s1[j]-s2[j])
						        / std::max(std::sqrt(unfl)*std::max(s1[0], ONE),
						                   ulp*std::max(std::fabs(s1[0]), std::fabs(s2[0])));
						temp2 = std::max(temp1, temp2);
					}
					result[33] = temp2;
				}
				// End of Loop -- Check for result[j] > thresh
				for (j=0; j<34; j++)
				{
					if (result[j]>=thresh)
					{
						if (nfail==0)
						{
							dlahd2(nout, path);
						}
						nout << " M=" << std::setw(5) << m << ", N=" << std::setw(5) << m
						     << ", type " << std::setw(2) << jtype+1 << ", seed=" << std::setw(4)
						     << ioldsd[0] << ',' << std::setw(4) << ioldsd[1] << ','
						     << std::setw(4) << ioldsd[2] << ',' << std::setw(4) << ioldsd[3]
						     << ", test(" << std::setw(2) << j+1 << ")=" << std::setw(11)
						     << std::setprecision(4) << result[j] << std::endl;
						nfail++;
					}
				}
				if (!bidiag)
				{
					ntest += 34;
				}
				else
				{
					ntest += 30;
				}
			}
		}
		// Summary
		alasum(path, nout, nfail, ntest, 0);
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
		int lmax[3];
		real* A      = new real[lda*lda];
		real* Ain    = new real[lda*lda];
		real* scale  = new real[lda];
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
		delete[] A;
		delete[] Ain;
		delete[] scale;
		delete[] scalin;
	}

	/*! §dchkee
	 *
	 * §dchkee tests the double precision LAPACK subroutines for the matrix eigenvalue problem.
	 * The test paths in this version are
	 * - NEP (Nonsymmetric Eigenvalue Problem):
	 *   Test §dgehrd, §dorghr, §dhseqr, §dtrevc, §dhsein, and §dormhr
	 * - SEP (Symmetric Eigenvalue Problem):
	 *   Test §dsytrd, §dorgtr, §dsteqr, §dsterf, §dstein, §dstedc, and drivers §dsyev(x),
	 *        §dsbev(x), §dspev(x), §dstev(x), §dsyevd, §dsbevd, §dspevd, §dstevd
	 * - SVD (Singular Value Decomposition):
	 *   Test §dgebrd, §dorgbr, §dbdsqr, §dbdsdc and the drivers §dgesvd, §dgesdd
	 * - DEV (Nonsymmetric Eigenvalue/eigenvector Driver):
	 *   Test §dgeev
	 * - DES (Nonsymmetric Schur form Driver):
	 *   Test §dgees
	 * - DVX (Nonsymmetric Eigenvalue/eigenvector Expert Driver):
	 *   Test §dgeevx
	 * - DSX (Nonsymmetric Schur form Expert Driver):
	 *   Test §dgeesx
	 * - DGG (Generalized Nonsymmetric Eigenvalue Problem):
	 *   Test §dgghd3, §dggbal, §dggbak, §dhgeqz, and §dtgevc
	 * - DGS (Generalized Nonsymmetric Schur form Driver):
	 *   Test §dgges
	 * - DGV (Generalized Nonsymmetric Eigenvalue/eigenvector Driver):
	 *   Test §dggev
	 * - DGX (Generalized Nonsymmetric Schur form Expert Driver):
	 *   Test §dggesx
	 * - DXV (Generalized Nonsymmetric Eigenvalue/eigenvector Expert Driver):
	 *   Test §dggevx
	 * - DSG (Symmetric Generalized Eigenvalue Problem):
	 *   Test §dsygst, §dsygv, §dsygvd, §dsygvx, §dspgst, §dspgv, §dspgvd, §dspgvx, §dsbgst,
	 *        §dsbgv, §dsbgvd, and §dsbgvx
	 * - DSB (Symmetric Band Eigenvalue Problem):
	 *   Test §dsbtrd
	 * - DBB (Band Singular Value Decomposition):
	 *   Test §dgbbrd
	 * - DEC (Eigencondition estimation):
	 *   Test §dlaln2, §dlasy2, §dlaequ, §dlaexc, §dtrsyl, §dtrexc, §dtrsna, §dtrsen, and §dlaqtr
	 * - DBL (Balancing a general matrix)
	 *   Test §dgebal
	 * - DBK (Back transformation on a balanced matrix)
	 *   Test §dgebak
	 * - DGL (Balancing a matrix pair)
	 *   Test §dggbal
	 * - DGK (Back transformation on a matrix pair)
	 *   Test §dggbak
	 * - GLM (Generalized Linear Regression Model):
	 *   Tests §dggglm
	 * - GQR (Generalized QR and RQ factorizations):
	 *   Tests §dggqrf and §dggrqf
	 * - GSV (Generalized Singular Value Decomposition):
	 *   Tests §dggsvd, §dggsvp, §dtgsja, §dlags2, §dlapll, and §dlapmt
	 * - CSD (CS decomposition):
	 *   Tests §dorcsd
	 * - LSE (Constrained Linear Least Squares):
	 *   Tests §dgglse
	 *
	 * Each test path has a different set of inputs, but the data sets for the driver routines
	 * §xev, §xes, §xvx, and §xsx can be concatenated in a single input file. The first line of
	 * input should contain one of the 3-character path names in columns 1-3. The number of
	 * remaining lines depends on what is found on the first line.\n
	 * The number of matrix types used in testing is often controllable from the input file. The
	 * number of matrix types for each path, and the test routine that describes them, is as
	 * follows:\n
	 * $\begin{tabular}{lrl}
	 *     Path name(s) & Types & Test routine         \\
	 *     \hline
	 *     DHS or NEP   & 21    & \{DCHKHS}            \\
	 *     DST or SEP   & 21    & \{DCHKST} (routines) \\
	 *                  & 18    & \{DDRVST} (drivers)  \\
	 *     DBD or SVD   & 16    & \{DCHKBD} (routines) \\
	 *                  &  5    & \{DDRVBD} (drivers)  \\
	 *     DEV          & 21    & \{DDRVEV}            \\
	 *     DES          & 21    & \{DDRVES}            \\
	 *     DVX          & 21    & \{DDRVVX}            \\
	 *     DSX          & 21    & \{DDRVSX}            \\
	 *     DGG          & 26    & \{DCHKGG} (routines) \\
	 *     DGS          & 26    & \{DDRGES}            \\
	 *     DGX          &  5    & \{DDRGSX}            \\
	 *     DGV          & 26    & \{DDRGEV}            \\
	 *     DXV          &  2    & \{DDRGVX}            \\
	 *     DSG          & 21    & \{DDRVSG}            \\
	 *     DSB          & 15    & \{DCHKSB}            \\
	 *     DBB          & 15    & \{DCHKBB}            \\
	 *     DEC          &  -    & \{DCHKEC}            \\
	 *     DBL          &  -    & \{DCHKBL}            \\
	 *     DBK          &  -    & \{DCHKBK}            \\
	 *     DGL          &  -    & \{DCHKGL}            \\
	 *     DGK          &  -    & \{DCHKGK}            \\
	 *     GLM          &  8    & \{DCKGLM}            \\
	 *     GQR          &  8    & \{DCKGQR}            \\
	 *     GSV          &  8    & \{DCKGSV}            \\
	 *     CSD          &  3    & \{DCKCSD}            \\
	 *     LSE          &  8    & \{DCKLSE}            \end{tabular}$
	 *
	 *---------------------------------------------------------------------------------------------
	 * NEP input file:
	 * - line 2:  §nn, integer\n
	 *            Number of values of §N.
	 * - line 3:  §nval, integer array, dimension (§nn)\n
	 *            The values for the matrix dimension §N.
	 * - line 4:  §nparms, integer\n
	 *            Number of values of the parameters §NB, §nbmin, §NX, §NS, and §MAXB.
	 * - line 5:  §nbval, integer array, dimension (§nparms)\n
	 *            The values for the blocksize §NB.
	 * - line 6:  §nbmin, integer array, dimension (§nparms)\n
	 *            The values for the minimum blocksize §nbmin.
	 * - line 7:  §nxval, integer array, dimension (§nparms)\n
	 *            The values for the crossover point §NX.
	 * - line 8:  §inmin, integer array, dimension (§nparms)\n
	 *            §LAHQR vs §TTQRE crossover point, $\ge 11$
	 * - line 9:  §inwin, integer array, dimension (§nparms)\n
	 *            recommended deflation window size
	 * - line 10: §inibl, integer array, dimension (§nparms)\n
	 *            nibble crossover point
	 * - line 11: §ishfts, integer array, dimension (§nparms)\n
	 *            number of simultaneous shifts)
	 * - line 12: §iacc22, integer array, dimension (§nparms)\n
	 *            select structured matrix multiply: 0, 1 or 2.
	 * - line 13: §thresh \n
	 *            Threshold value for the test ratios. Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold. To have all
	 *            of the test ratios printed, use $\{thresh}=0.0$.
	 * - line 14: §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 14 was 2:
	 * - line 15: §integer array, dimension (4 \n
	 *            Four integer values for the random number seed.
	 * - lines 15-EOF: The remaining lines occur in sets of 1 or 2 and allow the user to specify
	 *                 the matrix types. Each line contains a 3-character path name in columns 1-3,
	 *                 and the number of matrix types must be the first nonblank item in columns
	 *                 4-80. If the number of matrix types is at least 1 but is less than the
	 *                 maximum number of possible types, a second line will be read to get the
	 *                 numbers of the matrix types to be used. For example,\n
	 *                 NEP 21\n
	 *                 requests all of the matrix types for the nonsymmetric eigenvalue problem,
	 *                 while\n
	 *                 NEP  4\n
	 *                 9 10 11 12\n
	 *                 requests only matrices of type 9, 10, 11, and 12.\n
	 *                 The valid 3-character path names are "NEP" or "SHS" for the nonsymmetric
	 *                 eigenvalue routines.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * SEP or DSG input file:
	 * - line 2:  §nn, integer\n
	 *            Number of values of §N.
	 * - line 3:  §nval, integer array, dimension (§nn)\n
	 *            The values for the matrix dimension §N.
	 * - line 4:  §nparms, integer\n
	 *            Number of values of the parameters §NB, §nbmin, and §NX.
	 * - line 5:  §nbval, integer array, dimension (§nparms)\n
	 *            The values for the blocksize §NB.
	 * - line 6:  §nbmin, integer array, dimension (§nparms)\n
	 *            The values for the minimum blocksize §nbmin.
	 * - line 7:  §nxval, integer array, dimension (§nparms)\n
	 *            The values for the crossover point §NX.
	 * - line 8:  §thresh \n
	 *            Threshold value for the test ratios. Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold.
	 * - line 9:  §tstchk, logical\n
	 *            Flag indicating whether or not to test the LAPACK routines.
	 * - line 10: §tstdrv, logical\n
	 *            Flag indicating whether or not to test the driver routines.
	 * - line 11: §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits for the LAPACK routines
	 *            and driver routines.
	 * - line 12: §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 12 was 2:
	 * - line 13: §integer array, dimension (4)\n
	 *            Four integer values for the random number seed.
	 * - lines 13-EOF:  Lines specifying matrix types, as for NEP.\n
	 *                  The 3-character path names are "SEP" or "SST" for the symmetric eigenvalue
	 *                  routines and driver routines, and "DSG" for the routines for the symmetric
	 *                  generalized eigenvalue problem.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * SVD input file:
	 * - line 2:  §nn, integer\n
	 *            Number of values of §M and §N.
	 * - line 3:  §mval, integer array, dimension (§nn)\n
	 *            The values for the matrix row dimension §M.
	 * - line 4:  §nval, integer array, dimension (§nn)\n
	 *            The values for the matrix column dimension §N.
	 * - line 5:  §nparms, integer\n
	 *            Number of values of the parameter §NB, §nbmin, §NX, and §nrhs.
	 * - line 6:  §nbval, integer array, dimension (§nparms)\n
	 *            The values for the blocksize §NB.
	 * - line 7:  §nbmin, integer array, dimension (§nparms)\n
	 *            The values for the minimum blocksize §nbmin.
	 * - line 8:  §nxval, integer array, dimension (§nparms)\n
	 *            The values for the crossover point §NX.
	 * - line 9:  §nsval, integer array, dimension (§nparms)\n
	 *            The values for the number of right hand sides §nrhs.
	 * - line 10: §thresh \n
	 *            Threshold value for the test ratios.  Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold.
	 * - line 11: §tstchk, logical\n
	 *            Flag indicating whether or not to test the LAPACK routines.
	 * - line 12: §tstdrv, logical\n
	 *            Flag indicating whether or not to test the driver routines.
	 * - line 13: §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits for the LAPACK routines
	 *            and driver routines.
	 * - line 14: §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 14 was 2:
	 * - line 15: §integer array, dimension (4)\n
	 *            Four integer values for the random number seed.
	 * - lines 15-EOF: Lines specifying matrix types, as for NEP. The 3-character path names are
	 *                 "SVD" or "SBD" for both the SVD routines and the SVD driver routines.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * DEV and DES data files:
	 * - line 1:  "DEV" or "DES" in columns 1 to 3.
	 * - line 2:  §NSIZES, integer\n
	 *            Number of sizes of matrices to use. Should be at least 0 and at most 20. If
	 *            $\{NSIZES}=0$, no testing is done(although the remaining 3 lines are still read).
	 * - line 3:  §nn, integer array, dimension(§NSIZES)\n
	 *            Dimensions of matrices to be tested.
	 * - line 4:  §NB, §nbmin, §NX, §NS, §nbcol, integers\n
	 *            These integer parameters determine how blocking is done (see §ILAENV for details)
	 *            \n $\begin{tabular}{ll}
	 *                \{NB}     & block size                           \\
	 *                \{nbmin}  & minimum block size                   \\
	 *                \{NX}     & minimum dimension for blocking       \\
	 *                \{NS}     & number of shifts in \{xhseqr}        \\
	 *                \{nbcol}  & minimum column dimension for blocking\end{tabular}$
	 * - line 5:  §thresh, real\n
	 *            The test threshold against which computed residuals are compared. Should
	 *            generally be in the range from 10. to 20. If it is 0., all test case data will be
	 *            printed.
	 * - line 6:  §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits.
	 * - line 7:  §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 7 was 2:
	 * - line 8:  §integer array, dimension (4)\n
	 *            Four integer values for the random number seed.
	 * - lines 9 and following: Lines specifying matrix types, as for NEP. The 3-character path
	 *                          name is "DEV" to test §SGEEV, or "DES" to test §SGEES.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * The DVX data has two parts. The first part is identical to DEV, and the second part consists
	 * of test matrices with precomputed solutions.
	 * - line 1:  "DVX" in columns 1-3.
	 * - line 2:  §NSIZES, integer\n
	 *            If §NSIZES = 0, no testing of randomly generated examples is done, but any
	 *            precomputed examples are tested.
	 * - line 3:  §nn, integer array, dimension(§NSIZES)\n
	 * - line 4:  §NB, §nbmin, §NX, §NS, §nbcol, integers\n
	 * - line 5:  §thresh, real\n
	 * - line 6:  §tsterr, logical\n
	 * - line 7:  §newsd, integer\n
	 *
	 * If line 7 was 2:
	 * - line 8:  §integer array, dimension (4)\n
	 * - lines 9 and following: The first line contains "DVX" in columns 1-3 followed by the number
	 *                          of matrix types, possibly with a second line to specify certain
	 *                          matrix types. If the number of matrix types = 0, no testing of
	 *                          randomly generated examples is done, but any precomputed examples
	 *                          are tested.
	 * - remaining lines : Each matrix is stored on $1+2\{N}$ lines, where §N is its dimension. The
	 *                     first line contains the dimension (a single integer). The next §N lines
	 *                     contain the matrix, one row per line. The last §N lines correspond to
	 *                     each eigenvalue. Each of these last §N lines contains 4 real values: the
	 *                     real part of the eigenvalue, the imaginary part of the eigenvalue, the
	 *                     reciprocal condition number of the eigenvalues, and the reciprocal
	 *                     condition number of the eigenvector. The end of data is indicated by
	 *                     dimension §N=0. Even if no data is to be tested, there must be at least
	 *                     one line containing §N=0.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * The DSX data is like DVX. The first part is identical to DEV, and the
	 * second part consists of test matrices with precomputed solutions.
	 * - line 1:  "DSX" in columns 1-3.
	 * - line 2:  §NSIZES, integer\n
	 *            If §NSIZES = 0, no testing of randomly generated examples is done, but any
	 *            precomputed examples are tested.
	 * - line 3:  §nn, integer array, dimension(§NSIZES)\n
	 * - line 4:  §NB, §nbmin, §NX, §NS, §nbcol, integers\n
	 * - line 5:  §thresh, real\n
	 * - line 6:  §tsterr, logical\n
	 * - line 7:  §newsd, integer\n
	 *
	 * If line 7 was 2:
	 * - line 8:  §integer array, dimension (4)\n
	 * - lines 9 and following: The first line contains "DSX" in columns 1-3 followed by the number
	 *                          of matrix types, possibly with a second line to specify certain
	 *                          matrix types. If the number of matrix types = 0, no testing of
	 *                          randomly generated examples is done, but any precomputed examples
	 *                          are tested.
	 * - remaining lines : Each matrix is stored on $3+\{N}$ lines, where §N is its dimension. The
	 *                     first line contains the dimension §N and the dimension §M of an
	 *                     invariant subspace. The second line contains §M integers, identifying
	 *                     the eigenvalues in the invariant subspace (by their position in a list
	 *                     of eigenvalues ordered by increasing real part). The next §N lines
	 *                     contain the matrix. The last line contains the reciprocal condition
	 *                     number for the average of the selected eigenvalues, and the reciprocal
	 *                     condition number for the corresponding right invariant subspace. The end
	 *                     of data is indicated by a line containing §N=0 and §M=0. Even if no data
	 *                     is to be tested, there must be at least one line containing $\{N}=0$ and
	 *                     $\{M}=0$.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * DGG input file:
	 * - line 2:  §nn, integer\n
	 *            Number of values of §N.
	 * - line 3:  §nval, integer array, dimension (§nn)\n
	 *            The values for the matrix dimension §N.
	 * - line 4:  §nparms, integer\n
	 *            Number of values of the parameters §NB, §nbmin, §NS, §MAXB, and §nbcol.
	 * - line 5:  §nbval, integer array, dimension (§nparms)\n
	 *            The values for the blocksize §NB.
	 * - line 6:  §nbmin, integer array, dimension (§nparms)\n
	 *            The values for §nbmin, the minimum row dimension for blocks.
	 * - line 7:  §nsval, integer array, dimension (§nparms)\n
	 *            The values for the number of shifts.
	 * - line 8:  §mxbval, integer array, dimension (§nparms)\n
	 *            The values for §MAXB, used in determining minimum blocksize.
	 * - line 9:  §iacc22, integer array, dimension (§nparms)\n
	 *            select structured matrix multiply: 1 or 2)
	 * - line 10: §nbcol, integer array, dimension (§nparms)\n
	 *            The values for §nbcol, the minimum column dimension for blocks.
	 * - line 11: §thresh \n
	 *            Threshold value for the test ratios. Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold.
	 * - line 12: §tstchk, logical\n
	 *            Flag indicating whether or not to test the LAPACK routines.
	 * - line 13: §tstdrv, logical\n
	 *            Flag indicating whether or not to test the driver routines.
	 * - line 14: §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits for the LAPACK routines
	 *            and driver routines.
	 * - line 15: §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 15 was 2:
	 * - line 16: §integer array, dimension (4)\n
	 *            Four integer values for the random number seed.
	 * - lines 17-EOF: Lines specifying matrix types, as for NEP. The 3-character path name is
	 *                 "DGG" for the generalized eigenvalue problem routines and driver routines.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * DGS and DGV input files:
	 * - line 1:  "DGS" or "DGV" in columns 1 to 3.
	 * - line 2:  §nn, integer\n
	 *            Number of values of §N.
	 * - line 3:  §nval, integer array, dimension(§nn)\n
	 *            Dimensions of matrices to be tested.
	 * - line 4:  §NB, §nbmin, §NX, §NS, §nbcol, integers\n
	 *            These integer parameters determine how blocking is done (see §ILAENV for details)
	 *            \n $\begin{tabular}{ll}
	 *                \{NB}     & block size                            \\
	 *                \{nbmin}  & minimum block size                    \\
	 *                \{NX}     & minimum dimension for blocking        \\
	 *                \{NS}     & number of shifts in \{xhgeqr}         \\
	 *                \{nbcol}  & minimum column dimension for blocking \end{tabular}$
	 * - line 5:  §thresh, real\n
	 *            The test threshold against which computed residuals are compared. Should
	 *            generally be in the range from 10. to 20. If it is 0., all test case data will be
	 *            printed.
	 * - line 6:  §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits.
	 * - line 7:  §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 17 was 2:
	 * - line 7: §integer array, dimension (4)\n
	 *           Four integer values for the random number seed.
	 * - lines 7-EOF: Lines specifying matrix types, as for NEP. The 3-character path name is "DGS"
	 *                for the generalized eigenvalue problem routines and driver routines.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * DXV input files:
	 * - line 1:  "DXV" in columns 1 to 3.
	 * - line 2:  §N, integer\n
	 *            Value of §N.
	 * - line 3:  §NB, §nbmin, §NX, §NS, §nbcol, integers\n
	 *            These integer parameters determine how blocking is done (see §ILAENV for details)
	 *            \n $\begin{tabular}{ll}
	 *                \{NB}     & block size                            \\
	 *                \{nbmin}  & minimum block size                    \\
	 *                \{NX}     & minimum dimension for blocking        \\
	 *                \{NS}     & number of shifts in \{xhgeqr}         \\
	 *                \{nbcol}  & minimum column dimension for blocking \end{tabular}$
	 * - line 4:  §thresh, real\n
	 *            The test threshold against which computed residuals are compared. Should
	 *            generally be in the range from 10. to 20. Information will be printed about each
	 *            test for which the test ratio is greater than or equal to the threshold.
	 * - line 5:  §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits for the LAPACK routines
	 *            and driver routines.
	 * - line 6:  §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 6 was 2:
	 * - line 7: §integer array, dimension (4)\n
	 *           Four integer values for the random number seed.
	 *
	 * If line 2 was 0:
	 * - line 7-EOF: Precomputed examples are tested.
	 * - remaining lines : Each example is stored on $3+2\{N}$ lines, where §N is its dimension.
	 *                     The first line contains the dimension (a single integer). The next §N
	 *                     lines contain the matrix §A, one row per line. The next §N lines contain
	 *                     the matrix §B. The next line contains the reciprocals of the eigenvalue
	 *                     condition numbers. The last line contains the reciprocals of the
	 *                     eigenvector condition numbers. The end of data is indicated by dimension
	 *                     $\{N}=0$. Even if no data is to be tested, there must be at least one
	 *                     line containing $\{N}=0$.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * DGX input files:
	 * - line 1:  "DGX" in columns 1 to 3.
	 * - line 2:  §N, integer\n
	 *            Value of §N.
	 * - line 3:  §NB, §nbmin, §NX, §NS, §nbcol, integers\n
	 *            These integer parameters determine how blocking is done (see §ILAENV for details)
	 *            \n $\begin{tabular}{ll}
	 *                \{NB}     & block size                            \\
	 *                \{nbmin}  & minimum block size                    \\
	 *                \{NX}     & minimum dimension for blocking        \\
	 *                \{NS}     & number of shifts in \{xhgeqr}         \\
	 *                \{nbcol}  & minimum column dimension for blocking \end{tabular}$
	 * - line 4:  §thresh, real\n
	 *            The test threshold against which computed residuals are compared. Should
	 *            generally be in the range from 10. to 20. Information will be printed about each
	 *            test for which the test ratio is greater than or equal to the threshold.
	 * - line 5:  §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits for the LAPACK routines
	 *            and driver routines.
	 * - line 6:  §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 6 was 2:
	 * - line 7: §integer array, dimension (4)\n
	 *           Four integer values for the random number seed.
	 *
	 * If line 2 was 0:
	 *
	 * - line 7-EOF: Precomputed examples are tested.
	 * - remaining lines: Each example is stored on $3+2\{N}$ lines, where §N is its dimension. The
	 *                    first line contains the dimension (a single integer).  The next line
	 *                    contains an integer §k such that only the last k eigenvalues will be
	 *                    selected and appear in the leading diagonal blocks of $A$ and $B$. The
	 *                    next §N lines contain the matrix $A$, one row per line. The next §N lines
	 *                    contain the matrix $B$. The last line contains the reciprocal of the
	 *                    eigenvalue cluster condition number and the reciprocal of the deflating
	 *                    subspace (associated with the selected eigencluster) condition number.
	 *                    The end of data is indicated by dimension $\{N}=0$. Even if no data is to
	 *                    be tested, there must be at least one line containing $\{N}=0$.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * DSB input file:
	 * - line 2:  §nn, integer\n
	 *            Number of values of §N.
	 * - line 3:  §nval, integer array, dimension (§nn)\n
	 *            The values for the matrix dimension §N.
	 * - line 4:  §nk, integer\n
	 *            Number of values of §K.
	 * - line 5:  §kval, integer array, dimension (§nk)\n
	 *            The values for the matrix dimension §K.
	 * - line 6:  §thresh \n
	 *            Threshold value for the test ratios. Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold.
	 * - line 7:  §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 7 was 2:
	 * - line 8:  §integer array, dimension (4)\n
	 *            Four integer values for the random number seed.
	 * - lines 8-EOF: Lines specifying matrix types, as for NEP.
	 *                The 3-character path name is "DSB".
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * DBB input file:
	 * - line 2:  §nn, integer\n
	 *            Number of values of §M and §N.
	 * - line 3:  §mval, integer array, dimension (§nn)\n
	 *            The values for the matrix row dimension §M.
	 * - line 4:  §nval, integer array, dimension (§nn)\n
	 *            The values for the matrix column dimension §N.
	 * - line 4:  §nk, integer\n
	 *            Number of values of §K.
	 * - line 5:  §kval, integer array, dimension (§nk)\n
	 *            The values for the matrix bandwidth §K.
	 * - line 6:  §nparms, integer\n
	 *            Number of values of the parameter §nrhs
	 * - line 7:  §nsval, integer array, dimension (§nparms)\n
	 *            The values for the number of right hand sides §nrhs.
	 * - line 8:  §thresh \n
	 *            Threshold value for the test ratios. Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold.
	 * - line 9:  §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 9 was 2:
	 * - line 10: §integer array, dimension (4)\n
	 *            Four integer values for the random number seed.
	 * - lines 10-EOF: Lines specifying matrix types, as for SVD.
	 *                 The 3-character path name is "DBB".
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * DEC input file:
	 * - line 2:  §thresh, real
	 *            Threshold value for the test ratios. Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold.
	 * - lines 3-EOF: Input for testing the eigencondition routines consists of a set of specially
	 *                constructed test cases and their solutions. The data format is not intended
	 *                to be modified by the user.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * DBL and DBK input files:
	 * - line 1:  "DBL" in columns 1-3 to test §dgebal, or "DBK" in columns 1-3 to test §dgebak.
	 * .
	 * The remaining lines consist of specially constructed test cases.
	 *
	 *---------------------------------------------------------------------------------------------
	 * DGL and DGK input files:
	 * - line 1:  "DGL" in columns 1-3 to test §dggbal, or "DGK" in columns 1-3 to test §dggbak.
	 *
	 * The remaining lines consist of specially constructed test cases.
	 *
	 *---------------------------------------------------------------------------------------------
	 * GLM data file:
	 * - line 1:  "GLM" in columns 1 to 3.
	 * - line 2:  §nn, integer\n
	 *            Number of values of §M, §P, and §N.
	 * - line 3:  §mval, integer array, dimension(§nn)\n
	 *            Values of §M (row dimension).
	 * - line 4:  §pval, integer array, dimension(§nn)\n
	 *            Values of §P (row dimension).
	 * - line 5:  §nval, integer array, dimension(§nn)\n
	 *            Values of §N (column dimension), note $\{M}\le\{N}\le\{M}+\{P}$.
	 * - line 6:  §thresh, real\n
	 *            Threshold value for the test ratios. Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold.
	 * - line 7:  §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits for the LAPACK routines
	 *            and driver routines.
	 * - line 8:  §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 8 was 2:
	 * - line 9:  §integer array, dimension (4)\n
	 *            Four integer values for the random number seed.
	 * - lines 9-EOF: Lines specifying matrix types, as for NEP. The 3-character path name is "GLM"
	 *                for the generalized linear regression model routines.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * GQR data file:
	 * - line 1:  "GQR" in columns 1 to 3.
	 * - line 2:  §nn, integer\n
	 *            Number of values of §M, §P, and §N.
	 * - line 3:  §mval, integer array, dimension(§nn)\n
	 *            Values of §M.
	 * - line 4:  §pval, integer array, dimension(§nn)\n
	 *            Values of §P.
	 * - line 5:  §nval, integer array, dimension(§nn)\n
	 *            Values of §N.
	 * - line 6:  §thresh, real\n
	 *            Threshold value for the test ratios. Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold.
	 * - line 7:  §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits for the LAPACK routines
	 *            and driver routines.
	 * - line 8:  §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 8 was 2:
	 * - line 9:  §integer array, dimension (4)\n
	 *            Four integer values for the random number seed.
	 * - lines 9-EOF: Lines specifying matrix types, as for NEP. The 3-character path name is "GQR"
	 *                for the generalized QR and RQ routines.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * GSV data file:
	 * - line 1:  "GSV" in columns 1 to 3.
	 * - line 2:  §nn, integer\n
	 *            Number of values of §M, §P, and §N.
	 * - line 3:  §mval, integer array, dimension(§nn)\n
	 *            Values of §M (row dimension).
	 * - line 4:  §pval, integer array, dimension(§nn)\n
	 *            Values of §P (row dimension).
	 * - line 5:  §nval, integer array, dimension(§nn)\n
	 *            Values of §N (column dimension).
	 * - line 6:  §thresh, real\n
	 *            Threshold value for the test ratios. Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold.
	 * - line 7:  §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits for the LAPACK routines
	 *            and driver routines.
	 * - line 8:  §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 8 was 2:
	 * - line 9:  §integer array, dimension (4)\n
	 *            Four integer values for the random number seed.
	 * - lines 9-EOF: Lines specifying matrix types, as for NEP. The 3-character path name is "GSV"
	 *                for the generalized SVD routines.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * CSD data file:
	 * - line 1:  "CSD" in columns 1 to 3.
	 * - line 2:  §NM, integer\n
	 *            Number of values of §M, §P, and §N.
	 * - line 3:  §mval, integer array, dimension(§NM)\n
	 *            Values of §M (row and column dimension of orthogonal matrix).
	 * - line 4:  §pval, integer array, dimension(§NM)\n
	 *            Values of §P (row dimension of top-left block).
	 * - line 5:  §nval, integer array, dimension(§NM)\n
	 *            Values of §N (column dimension of top-left block).
	 * - line 6:  §thresh, real\n
	 *            Threshold value for the test ratios. Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold.
	 * - line 7:  §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits for the LAPACK routines
	 *            and driver routines.
	 * - line 8:  §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 8 was 2:
	 * - line 9:  §integer array, dimension (4)\n
	 *            Four integer values for the random number seed.
	 * - lines 9-EOF: Lines specifying matrix types, as for NEP. The 3-character path name is "CSD"
	 *                for the CSD routine.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * LSE data file:
	 * - line 1:  "LSE" in columns 1 to 3.
	 * - line 2:  §nn, integer\n
	 *            Number of values of §M, §P, and §N.
	 * - line 3:  §mval, integer array, dimension(§nn)\n
	 *            Values of §M.
	 * - line 4:  §pval, integer array, dimension(§nn)\n
	 *            Values of §P.
	 * - line 5:  §nval, integer array, dimension(§nn)\n
	 *            Values of §N, note P <= N <= P+M.
	 * - line 6:  §thresh, real\n
	 *            Threshold value for the test ratios. Information will be printed about each test
	 *            for which the test ratio is greater than or equal to the threshold.
	 * - line 7:  §tsterr, logical\n
	 *            Flag indicating whether or not to test the error exits for the LAPACK routines
	 *            and driver routines.
	 * - line 8:  §newsd, integer\n
	 *            A code indicating how to set the random number seed.\n
	 *            =0: Set the seed to a default value before each run\n
	 *            =1: Initialize the seed to a default value only before the first run\n
	 *            =2: Like 1, but use the seed values on the next line
	 *
	 * If line 8 was 2:
	 * - line 9:  §integer array, dimension (4)\n
	 *            Four integer values for the random number seed.
	 * - lines 9-EOF: Lines specifying matrix types, as for NEP. The 3-character path name is "GSV"
	 *                for the generalized SVD routines.
	 * .
	 *
	 *---------------------------------------------------------------------------------------------
	 * For SVD, we assume §nrhs may be as big as §N.
	 * \param[in] nmax
	 *     Maximum size of test matrices. (default: 132)
	 *     Must be at least 12 for some of the precomputed examples.
	 *
	 * \param[in] ncmax  maximum size of the C matrices for §ddrgsx. (default: 20)
	 * \param[in] need   Number of matrices to be allowed for. (default: 14 (for DGG))
	 * \param[in] lwork  Length of the §work array. (default: $\{nmax}(5\,\{nmax}+5)+1$)
	 * \param[in] liwork Length of the §iwork array. (default: $\{nmax}(5\,\{nmax}+20)$)
	 * \param[in] maxin  The maximum number of different parameter sets (default: 20)
	 * \param[in] maxt   The maximum number of types to be tested (default: 30)
	 * \param[in] nin    The input stream from which the test examples are read.
	 * \param[in] nout   The output stream for test result messages
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date June 2016                                                                           */
	void dchkee(int const nmax, int const ncmax, int const need, int const lwork, int const liwork,
	            int const maxin, int const maxt, std::istream& nin, std::ostream& nout)
	{
		/*
		int const nmax = 132;
		int const ncmax = 20;
		int const need = 14;
		int const lwork = nmax*(5*nmax+5)+1;
		int const liwork = nmax*(5*nmax+20);
		int const maxin = 20;
		int const maxt = 30;
		std::istream& nin  = std::cin;
		std::ostream& nout = std::cout;
		*/
		// allocate variables
		int ioldsd[4]  = {0, 0, 0, 1};
		bool csd, dbb, dgg, dsb, glm, gqr, gsv, lse, nep, dbk, dbl, sep, des, dev, dgk, dgl, dgs,
		     dgv, dgx, dsx, svd, dvx, dxv, tstchk=false, tstdif, tstdrv=false, tsterr;
		char c1, c3[4], path[4], vname[32], line[80];
		int i, i1, ic, info, itmp, k, lenp, maxtyp, newsd, nk, nn, nparms, nrhs, ntypes,
		    vers_major, vers_minor, vers_patch;
		real eps, thresh, thrshn;
		std::time_t s1, s2;
		bool* dotype = new bool[maxt];
		bool* logwrk = new bool[nmax];
		int iseed[4];
		int* iwork = new int[liwork];
		int* kval = new int[maxin];
		int* mval = new int[maxin];
		int* mxbval = new int[maxin];
		int* nbcol = new int[maxin];
		int* nbmin = new int[maxin];
		int* nbval = new int[maxin];
		int* nsval = new int[maxin];
		int* nval = new int[maxin];
		int* nxval = new int[maxin];
		int* pval = new int[maxin];
		int* inmin = new int[maxin];
		int* inwin = new int[maxin];
		int* inibl = new int[maxin];
		int* ishfts = new int[maxin];
		int* iacc22 = new int[maxin];
		int nmax2 = nmax * nmax;
		real* A = new real[nmax2 * need];
		real* B = new real[nmax2 * 5];
		real* C = new real[ncmax*ncmax * ncmax*ncmax];
		real* D = new real[nmax * 12];
		real* result = new real[500];
		real* taua = new real[nmax];
		real* taub = new real[nmax];
		real* work = new real[lwork];
		real* x = new real[5*nmax];
		// initialise
		std::time(&s1);
		bool fatal = false;
		infoc.nout.rdbuf(nout.rdbuf());
		char const* str9989a = " Invalid input value: ";
		char const* str9989b = "; must be >=";
		char const* str9988a = str9989a;
		char const* str9988b = "; must be <=";
		char const* str9990 = " routines were not tested";
		char const* str9980 = " *** Error code from ";
		char const* str9973 =
		    "\n -----------------------------------------------------------------------";
		bool errcont, endbreak;
		// Return to here to read multiple sets of data
		while (nin.good())
		{
			// Read the first line and set the 3-character test path
			nin.read(path, 3);
			path[3] = '\0';
			nin.ignore(100, '\n');
			if (!nin.good())
			{
				break;
			}
			nep = (std::strncmp(path, "NEP", 3)==0 || std::strncmp(path, "DHS", 3)==0);
			sep = (std::strncmp(path, "SEP", 3)==0 || std::strncmp(path, "DST", 3)==0
			    || std::strncmp(path, "DSG", 3)==0 || std::strncmp(path, "SE2", 3)==0);
			svd = (std::strncmp(path, "SVD", 3)==0 || std::strncmp(path, "DBD", 3)==0);
			dev = (std::strncmp(path, "DEV", 3)==0);
			des = (std::strncmp(path, "DES", 3)==0);
			dvx = (std::strncmp(path, "DVX", 3)==0);
			dsx = (std::strncmp(path, "DSX", 3)==0);
			dgg = (std::strncmp(path, "DGG", 3)==0);
			dgs = (std::strncmp(path, "DGS", 3)==0);
			dgx = (std::strncmp(path, "DGX", 3)==0);
			dgv = (std::strncmp(path, "DGV", 3)==0);
			dxv = (std::strncmp(path, "DXV", 3)==0);
			dsb = (std::strncmp(path, "DSB", 3)==0);
			dbb = (std::strncmp(path, "DBB", 3)==0);
			glm = (std::strncmp(path, "GLM", 3)==0);
			gqr = (std::strncmp(path, "GQR", 3)==0 || std::strncmp(path, "GRQ", 3)==0);
			gsv = (std::strncmp(path, "GSV", 3)==0);
			csd = (std::strncmp(path, "CSD", 3)==0);
			lse = (std::strncmp(path, "LSE", 3)==0);
			dbl = (std::strncmp(path, "DBL", 3)==0);
			dbk = (std::strncmp(path, "DBK", 3)==0);
			dgl = (std::strncmp(path, "DGL", 3)==0);
			dgk = (std::strncmp(path, "DGK", 3)==0);
			// Report values of parameters.
			if (std::strncmp(path, "   ", 3)==0)
			{
				continue;
			}
			else if (nep)
			{
				nout << " Tests of the Nonsymmetric Eigenvalue Problem routines\n";
			}
			else if (sep)
			{
				nout << " Tests of the Symmetric Eigenvalue Problem routines\n";
			}
			else if (svd)
			{
				nout << " Tests of the Singular Value Decomposition routines\n";
			}
			else if (dev)
			{
				nout << "\n Tests of the Nonsymmetric Eigenvalue Problem Driver\n"
				        "    DGEEV (eigenvalues and eigevectors)\n";
			}
			else if (des)
			{
				nout << "\n Tests of the Nonsymmetric Eigenvalue Problem Driver\n"
				        "    DGEES (Schur form)\n";
			}
			else if (dvx)
			{
				nout << "\n Tests of the Nonsymmetric Eigenvalue Problem Expert Driver\n"
				        "    DGEEVX (eigenvalues, eigenvectors and condition numbers)\n";
			}
			else if (dsx)
			{
				nout << "\n Tests of the Nonsymmetric Eigenvalue Problem Expert Driver\n"
				        "    DGEESX (Schur form and condition numbers)\n";
			}
			else if (dgg)
			{
				nout << "\n Tests of the Generalized Nonsymmetric Eigenvalue Problem routines\n";
			}
			else if (dgs)
			{
				nout << "\n Tests of the Generalized Nonsymmetric Eigenvalue Problem Driver "
				        "DGGES\n";
			}
			else if (dgx)
			{
				nout << "\n Tests of the Generalized Nonsymmetric Eigenvalue Problem Expert Driver"
				        " DGGESX\n";
			}
			else if (dgv)
			{
				nout << "\n Tests of the Generalized Nonsymmetric Eigenvalue Problem Driver "
				        "DGGEV\n";
			}
			else if (dxv)
			{
				nout << "\n Tests of the Generalized Nonsymmetric Eigenvalue Problem Expert Driver"
				        " DGGEVX\n";
			}
			else if (dsb)
			{
				nout << " Tests of DSBTRD\n"
				        " (reduction of a symmetric band matrix to tridiagonal form\n";
			}
			else if (dbb)
			{
				nout << " Tests of DGBBRD\n"
				        " (reduction of a general band matrix to real bidiagonal form)\n";
			}
			else if (glm)
			{
				nout << "\n Tests of the Generalized Linear Regression Model routines\n";
			}
			else if (gqr)
			{
				nout << "\n Tests of the Generalized QR and RQ routines\n";
			}
			else if (gsv)
			{
				nout << "\n Tests of the Generalized Singular Value Decomposition routines\n";
			}
			else if (csd)
			{
				nout << "\n Tests of the CS Decomposition routines\n";
			}
			else if (lse)
			{
				nout << "\n Tests of the Linear Least Squares routines\n";
			}
			else if (dbl)
			{
				// DGEBAL: Balancing
				dchkbl(nin, nout);
				continue;
			}
			else if (dbk)
			{
				// DGEBAK:  Back transformation
				//CALL DCHKBK(nin, nout)
				std::cerr << "DCHKBK not yet implemented" << std::endl;
				continue;
			}
			else if (dgl)
			{
				// DGGBAL:  Balancing
				//CALL DCHKGL(nin, nout)
				std::cerr << "DCHKGL not yet implemented" << std::endl;
				continue;
			}
			else if (dgk)
			{
				// DGGBAK:  Back transformation
				//CALL DCHKGK(nin, nout)
				std::cerr << "DCHKGK not yet implemented" << std::endl;
				continue;
			}
			else if ((std::strncmp(path, "DEC", 3)==0))
			{
				// DEC:  Eigencondition estimation
				nin >> thresh;
				nin.ignore(100, '\n');
				xlaenv(1, 1);
				xlaenv(12, 11);
				xlaenv(13, 2);
				xlaenv(14, 0);
				xlaenv(15, 2);
				xlaenv(16, 2);
				tsterr = true;
				//CALL DCHKEC(thresh, tsterr, nin, nout)
				std::cerr << "DCHKEC not yet implemented" << std::endl;
				continue;
			}
			else
			{
				nout << ' ' << path << ":  Unrecognized path name" << std::endl;
				continue;
			}
			this->ilaver(vers_major, vers_minor, vers_patch);
			nout << "\n LAPACK VERSION " << std::setw(1) << vers_major << '.' << std::setw(1)
			     << vers_minor << '.' << std::setw(1) << vers_patch << '\n';
			nout << "\n The following parameter values will be used:" << std::endl;
			// Read the number of values of M, P, and N.
			nin >> nn;
			nin.ignore(100, '\n');
			if (nn<0)
			{
				nout << str9989a << "   NN =" << std::setw(6) << nn << str9989b << std::setw(6)
				     << 1 << std::endl;
				nn = 0;
				fatal = true;
			}
			else if (nn>maxin)
			{
				nout << str9988a << "   NN =" << std::setw(6) << nn << str9988b << std::setw(6)
				     << maxin << std::endl;
				nn = 0;
				fatal = true;
			}
			// Read the values of M
			if (!(dgx || dxv))
			{
				if (svd)
				{
					std::strcpy(vname, "    M ");
				}
				else
				{
					std::strcpy(vname, "    N ");
				}
				for (i=0; i<nn; i++)
				{
					nin >> mval[i];
					if (mval[i]<0)
					{
						nout << str9989a << vname << '=' << std::setw(6) << mval[i] << str9989b
						     << std::setw(6) << 0 << std::endl;
						fatal = true;
					}
					else if (mval[i]>nmax)
					{
						nout << str9988a << vname << '='  << std::setw(6) << mval[i] << str9988b
						     << std::setw(6) << nmax << std::endl;
						fatal = true;
					}
				}
				nin.ignore(100, '\n');
				nout << "    M:    ";
				for (i=0; i<std::min(10, nn); i++)
				{
					nout << std::setw(6) << mval[i];
				}
				if (nn>10)
				{
					nout << "\n          ";
					for (i=10; i<std::min(20, nn); i++)
					{
						nout << std::setw(6) << mval[i];
					}
				}
				nout << std::endl;
			}
			// Read the values of P
			if (glm || gqr || gsv || csd || lse)
			{
				for (i=0; i<nn; i++)
				{
					nin >> pval[i];
					if (pval[i]<0)
					{
						nout << str9989a << " P  =" << std::setw(6) << pval[i] << str9989b
						     << std::setw(6) << 0 << std::endl;
						fatal = true;
					}
					else if (pval[i]>nmax)
					{
						nout << str9988a << " P  =" << std::setw(6) << pval[i] << str9988b
						     << std::setw(6) << nmax << std::endl;
						fatal = true;
					}
				}
				nin.ignore(100, '\n');
				nout << "    P:    ";
				for (i=0; i<std::min(10, nn); i++)
				{
					nout << std::setw(6) << pval[i];
				}
				if (nn>10)
				{
					nout << "\n          ";
					for (i=10; i<std::min(20, nn); i++)
					{
						nout << std::setw(6) << pval[i];
					}
				}
				nout << std::endl;
			}
			// Read the values of N
			if (svd || dbb || glm || gqr || gsv || csd || lse)
			{
				for (i=0; i<nn; i++)
				{
					nin >> nval[i];
					if (nval[i]<0)
					{
						nout << str9989a << "    N =" << std::setw(6) << nval[i] << str9989b
						     << std::setw(6) << 0 << std::endl;
						fatal = true;
					}
					else if (nval[i]>nmax)
					{
						nout << str9988a << "    N =" << std::setw(6) << nval[i] << str9988b
						     << std::setw(6) << nmax << std::endl;
						fatal = true;
					}
				}
				nin.ignore(100, '\n');
			}
			else
			{
				for (i=0; i<nn; i++)
				{
					nval[i] = mval[i];
				}
			}
			if (!(dgx || dxv))
			{
				nout << "    N:    ";
				for (i=0; i<std::min(10, nn); i++)
				{
					nout << std::setw(6) << nval[i];
				}
				if (nn>10)
				{
					nout << "\n          ";
					for (i=10; i<std::min(20, nn); i++)
					{
						nout << std::setw(6) << nval[i];
					}
				}
				nout << std::endl;
			}
			else
			{
				nout << "    N:    " << std::setw(6) << nn << std::endl;
			}
			// Read the number of values of K, followed by the values of K
			if (dsb || dbb)
			{
				nin >> nk;
				nin.ignore(100, '\n');
				for (i=0; i<nk; i++)
				{
					nin >> kval[i];
					if (kval[i]<0)
					{
						nout << str9989a << "    K =" << std::setw(6) << kval[i] << str9989b
						     << std::setw(6) << 0 << std::endl;
						fatal = true;
					}
					else if (kval[i]>nmax)
					{
						nout << str9988a << "    K =" << std::setw(6) << kval[i] << str9988b
						     << std::setw(6) << nmax << std::endl;
						fatal = true;
					}
				}
				nin.ignore(100, '\n');
				nout << "    K:    ";
				for (i=0; i<std::min(10, nk); i++)
				{
					nout << std::setw(6) << kval[i];
				}
				if (nk>10)
				{
					nout << "\n          ";
					for (i=10; i<std::min(20, nk); i++)
					{
						nout << std::setw(6) << kval[i];
					}
				}
				nout << std::endl;
			}
			if (dev || des || dvx || dsx)
			{
				// For the nonsymmetric QR driver routines, only one set of parameters is allowed.
				nin >> nbval[0] >> nbmin[0] >> nxval[0] >> inmin[0] >> inwin[0] >> inibl[0]
				    >> ishfts[0] >> iacc22[0];
				nin.ignore(100, '\n');
				if (nbval[0]<1)
				{
					nout << str9989a << "   NB =" << std::setw(6) << nbval[0] << str9989b
					     << std::setw(6) << 1 << std::endl;
					fatal = true;
				}
				else if (nbmin[0]<1)
				{
					nout << str9989a << "NBMIN =" << std::setw(6) << nbmin[0] << str9989b
					     << std::setw(6) << 1 << std::endl;
					fatal = true;
				}
				else if (nxval[0]<1)
				{
					nout << str9989a << "   NX =" << std::setw(6) << nxval[0] << str9989b
					     << std::setw(6) << 1 << std::endl;
					fatal = true;
				}
				else if (inmin[0]<1)
				{
					nout << str9989a << "   INMIN =" << std::setw(6) << inmin[0] << str9989b
					     << std::setw(6) << 1 << std::endl;
					fatal = true;
				}
				else if (inwin[0]<1)
				{
					nout << str9989a << "   INWIN =" << std::setw(6) << inwin[0] << str9989b
					     << std::setw(6) << 1 << std::endl;
					fatal = true;
				}
				else if (inibl[0]<1)
				{
					nout << str9989a << "   INIBL =" << std::setw(6) << inibl[0] << str9989b
					     << std::setw(6) << 1 << std::endl;
					fatal = true;
				}
				else if (ishfts[0]<1)
				{
					nout << str9989a << "   ISHFTS =" << std::setw(6) << ishfts[0] << str9989b
					     << std::setw(6) << 1 << std::endl;
					fatal = true;
				}
				else if (iacc22[0]<0)
				{
					nout << str9989a << "   IACC22 =" << std::setw(6) << iacc22[0] << str9989b
					     << std::setw(6) << 0 << std::endl;
					fatal = true;
				}
				xlaenv(1,  nbval[0]);
				xlaenv(2,  nbmin[0]);
				xlaenv(3,  nxval[0]);
				xlaenv(12, std::max(11, inmin[0]));
				xlaenv(13, inwin[0]);
				xlaenv(14, inibl[0]);
				xlaenv(15, ishfts[0]);
				xlaenv(16, iacc22[0]);
				nout << "    NB:   " << nbval[0] << '\n';
				nout << "    NBMIN:" << nbmin[0] << '\n';
				nout << "    NX:   " << nxval[0] << '\n';
				nout << "    INMIN:   " << inmin[0] << '\n';
				nout << "    INWIN: " << inwin[0] << '\n';
				nout << "    INIBL: " << inibl[0] << '\n';
				nout << "    ISHFTS: " << ishfts[0] << '\n';
				nout << "    IACC22: " << iacc22[0] << std::endl;
			}
			else if (dgs || dgx || dgv || dxv)
			{
				// For the nonsymmetric generalized driver routines,
				// only one set of parameters is allowed.
				nin >> nbval[0] >> nbmin[0] >> nxval[0] >> nsval[0] >> mxbval[0];
				nin.ignore(100, '\n');
				if (nbval[0]<1)
				{
					nout << str9989a << "   NB =" << std::setw(6) << nbval[0] << str9989b
					     << std::setw(6) << 1 << std::endl;
					fatal = true;
				}
				else if (nbmin[0]<1)
				{
					nout << str9989a << "NBMIN =" << std::setw(6) << nbmin[0] << str9989b
					     << std::setw(6) << 1 << std::endl;
					fatal = true;
				}
				else if (nxval[0]<1)
				{
					nout << str9989a << "   NX =" << std::setw(6) << nxval[0] << str9989b
					     << std::setw(6) << 1 << std::endl;
					fatal = true;
				}
				else if (nsval[0]<2)
				{
					nout << str9989a << "   NS =" << std::setw(6) << nsval[0] << str9989b
					     << std::setw(6) << 2 << std::endl;
					fatal = true;
				}
				else if (mxbval[0]<1)
				{
					nout << str9989a << " MAXB =" << std::setw(6) << mxbval[0] << str9989b
					     << std::setw(6) << 1 << std::endl;
					fatal = true;
				}
				xlaenv(1, nbval[0]);
				xlaenv(2, nbmin[0]);
				xlaenv(3, nxval[0]);
				xlaenv(4, nsval[0]);
				xlaenv(8, mxbval[0]);
				nout << "    NB:   " << nbval[0] << '\n';
				nout << "    NBMIN:" << nbmin[0] << '\n';
				nout << "    NX:   " << nxval[0] << '\n';
				nout << "    NS:   " << nsval[0] << '\n';
				nout << "    MAXB: " << mxbval[0] << std::endl;
			}
			else if (!dsb && !glm && !gqr && !gsv && !csd && !lse)
			{
				// For the other paths, the number of parameters can be varied from the input file.
				// Read the number of parameter values.
				nin >> nparms;
				nin.ignore(100, '\n');
				if (nparms<1)
				{
					nout << str9989a << "NPARMS=" << std::setw(6) << nparms << str9989b
					     << std::setw(6) << 1 << std::endl;
					nparms = 0;
					fatal = true;
				}
				else if (nparms>maxin)
				{
					nout << str9988a << "NPARMS=" << std::setw(6) << nparms << str9988b
					     << std::setw(6) << maxin << std::endl;
					nparms = 0;
					fatal = true;
				}
				// Read the values of NB
				if (!dbb)
				{
					for (i=0; i<nparms; i++)
					{
						nin >> nbval[i];
						if (nbval[i]<0)
						{
							nout << str9989a << "   NB =" << std::setw(6) << nbval[i] << str9989b
							     << std::setw(6) << 0 << std::endl;
							fatal = true;
						}
						else if (nbval[i]>nmax)
						{
							nout << str9988a << "   NB =" << std::setw(6) << nbval[i] << str9988b
							     << std::setw(6) << nmax << std::endl;
							fatal = true;
						}
					}
					nin.ignore(100, '\n');
					nout << "    NB:   ";
					for (i=0; i<std::min(10, nparms); i++)
					{
						nout << std::setw(6) << nbval[i];
					}
					if (nparms>10)
					{
						nout << "\n          ";
						for (i=10; i<std::min(20, nparms); i++)
						{
							nout << std::setw(6) << nbval[i];
						}
					}
					nout << std::endl;
				}
				// Read the values of nbmin
				if (nep || sep || svd || dgg)
				{
					for (i=0; i<nparms; i++)
					{
						nin >> nbmin[i];
						if (nbmin[i]<0)
						{
							nout << str9989a << "NBMIN =" << std::setw(6) << nbmin[i] << str9989b
							     << std::setw(6) << 0 << std::endl;
							fatal = true;
						}
						else if (nbmin[i]>nmax)
						{
							nout << str9988a << "NBMIN =" << std::setw(6) << nbmin[i] << str9988b
							     << std::setw(6) << nmax << std::endl;
							fatal = true;
						}
					}
					nin.ignore(100, '\n');
					nout << "    NBMIN:";
					for (i=0; i<std::min(10, nparms); i++)
					{
						nout << std::setw(6) << nbmin[i];
					}
					if (nparms>10)
					{
						nout << "\n          ";
						for (i=10; i<std::min(20, nparms); i++)
						{
							nout << std::setw(6) << nbmin[i];
						}
					}
					nout << std::endl;
				}
				else
				{
					for (i=0; i<nparms; i++)
					{
						nbmin[i] = 1;
					}
				}
				// Read the values of NX
				if (nep || sep || svd)
				{
					for (i=0; i<nparms; i++)
					{
						nin >> nxval[i];
						if (nxval[i]<0)
						{
							nout << str9989a << "   NX =" << std::setw(6) << nxval[i] << str9989b
							     << std::setw(6) << 0 << std::endl;
							fatal = true;
						}
						else if (nxval[i]>nmax)
						{
							nout << str9988a << "   NX =" << std::setw(6) << nxval[i] << str9988b
							     << std::setw(6) << nmax << std::endl;
							fatal = true;
						}
					}
					nin.ignore(100, '\n');
					nout << "    NX:   ";
					for (i=0; i<std::min(10, nparms); i++)
					{
						nout << std::setw(6) << nxval[i];
					}
					if (nparms>10)
					{
						nout << "\n          ";
						for (i=10; i<std::min(20, nparms); i++)
						{
							nout << std::setw(6) << nxval[i];
						}
					}
					nout << std::endl;
				}
				else
				{
					for (i=0; i<nparms; i++)
					{
						nxval[i] = 1;
					}
				}
				// Read the values of NSHIFT (if dgg) or nrhs (if svd or dbb).
				if (svd || dbb || dgg)
				{
					for (i=0; i<nparms; i++)
					{
						nin >> nsval[i];
						if (nsval[i]<0)
						{
							nout << str9989a << "   NS =" << std::setw(6) << nsval[i] << str9989b
							     << std::setw(6) << 0 << std::endl;
							fatal = true;
						}
						else if (nsval[i]>nmax)
						{
							nout << str9988a << "   NS =" << std::setw(6) << nsval[i] << str9988b
							     << std::setw(6) << nmax << std::endl;
							fatal = true;
						}
					}
					nin.ignore(100, '\n');
					nout << "    NS:   ";
					for (i=0; i<std::min(10, nparms); i++)
					{
						nout << std::setw(6) << nsval[i];
					}
					if (nparms>10)
					{
						nout << "\n          ";
						for (i=10; i<std::min(20, nparms); i++)
						{
							nout << std::setw(6) << nsval[i];
						}
					}
					nout << std::endl;
				}
				else
				{
					for (i=0; i<nparms; i++)
					{
						nsval[i] = 1;
					}
				}
				// Read the values for MAXB.
				if (dgg)
				{
					for (i=0; i<nparms; i++)
					{
						nin >> mxbval[i];
						if (mxbval[i]<0)
						{
							nout << str9989a << " MAXB =" << std::setw(6) << mxbval[i] << str9989b
							     << std::setw(6) << 0 << std::endl;
							fatal = true;
						}
						else if (mxbval[i]>nmax)
						{
							nout << str9988a << " MAXB =" << std::setw(6) << mxbval[i] << str9988b
							     << std::setw(6) << nmax << std::endl;
							fatal = true;
						}
					}
					nin.ignore(100, '\n');
					nout << "    MAXB: ";
					for (i=0; i<std::min(10, nparms); i++)
					{
						nout << std::setw(6) << mxbval[i];
					}
					if (nparms>10)
					{
						nout << "\n          ";
						for (i=10; i<std::min(20, nparms); i++)
						{
							nout << std::setw(6) << mxbval[i];
						}
					}
					nout << std::endl;
				}
				else
				{
					for (i=0; i<nparms; i++)
					{
						mxbval[i] = 1;
					}
				}
				// Read the values for inmin.
				if (nep)
				{
					for (i=0; i<nparms; i++)
					{
						nin >> inmin[i];
						if (inmin[i]<0)
						{
							nout << str9989a << " INMIN =" << std::setw(6) << inmin[i] << str9989b
							     << std::setw(6) << 0 << std::endl;
							fatal = true;
						}
					}
					nin.ignore(100, '\n');
					nout << "    INMIN: ";
					for (i=0; i<std::min(10, nparms); i++)
					{
						nout << std::setw(6) << inmin[i];
					}
					if (nparms>10)
					{
						nout << "\n          ";
						for (i=10; i<std::min(20, nparms); i++)
						{
							nout << std::setw(6) << inmin[i];
						}
					}
					nout << std::endl;
				}
				else
				{
					for (i=0; i<nparms; i++)
					{
						inmin[i] = 1;
					}
				}
				// Read the values for inwin.
				if (nep)
				{
					for (i=0; i<nparms; i++)
					{
						nin >> inwin[i];
						if (inwin[i]<0)
						{
							nout << str9989a << " INWIN =" << std::setw(6) << inwin[i] << str9989b
							     << std::setw(6) << 0 << std::endl;
							fatal = true;
						}
					}
					nin.ignore(100, '\n');
					nout << "    INWIN: ";
					for (i=0; i<std::min(10, nparms); i++)
					{
						nout << std::setw(6) << inwin[i];
					}
					if (nparms>10)
					{
						nout << "\n          ";
						for (i=10; i<std::min(20, nparms); i++)
						{
							nout << std::setw(6) << inwin[i];
						}
					}
					nout << std::endl;
				}
				else
				{
					for (i=0; i<nparms; i++)
					{
						inwin[i] = 1;
					}
				}
				// Read the values for inibl.
				if (nep)
				{
					for (i=0; i<nparms; i++)
					{
						nin >> inibl[i];
						if (inibl[i]<0)
						{
							nout << str9989a << " INIBL =" << std::setw(6) << inibl[i] << str9989b
							     << std::setw(6) << 0 << std::endl;
							fatal = true;
						}
					}
					nin.ignore(100, '\n');
					nout << "    INIBL: ";
					for (i=0; i<std::min(10, nparms); i++)
					{
						nout << std::setw(6) << inibl[i];
					}
					if (nparms>10)
					{
						nout << "\n          ";
						for (i=10; i<std::min(20, nparms); i++)
						{
							nout << std::setw(6) << inibl[i];
						}
					}
					nout << std::endl;
				}
				else
				{
					for (i=0; i<nparms; i++)
					{
						inibl[i] = 1;
					}
				}
				// Read the values for ishfts.
				if (nep)
				{
					for (i=0; i<nparms; i++)
					{
						nin >> ishfts[i];
						if (ishfts[i]<0)
						{
							nout << str9989a << " ISHFTS =" << std::setw(6) << ishfts[i]
							     << str9989b << std::setw(6) << 0 << std::endl;
							fatal = true;
						}
					}
					nin.ignore(100, '\n');
					nout << "    ISHFTS: ";
					for (i=0; i<std::min(10, nparms); i++)
					{
						nout << std::setw(6) << ishfts[i];
					}
					if (nparms>10)
					{
						nout << "\n          ";
						for (i=10; i<std::min(20, nparms); i++)
						{
							nout << std::setw(6) << ishfts[i];
						}
					}
					nout << std::endl;
				}
				else
				{
					for (i=0; i<nparms; i++)
					{
						ishfts[i] = 1;
					}
				}
				// Read the values for iacc22.
				if (nep || dgg)
				{
					for (i=0; i<nparms; i++)
					{
						nin >> iacc22[i];
						if (iacc22[i]<0)
						{
							nout << str9989a << " IACC22 =" << std::setw(6) << iacc22[i]
							     << str9989b << std::setw(6) << 0 << std::endl;
							fatal = true;
						}
					}
					nin.ignore(100, '\n');
					nout << "    IACC22: ";
					for (i=0; i<std::min(10, nparms); i++)
					{
						nout << std::setw(6) << iacc22[i];
					}
					if (nparms>10)
					{
						nout << "\n          ";
						for (i=10; i<std::min(20, nparms); i++)
						{
							nout << std::setw(6) << iacc22[i];
						}
					}
					nout << std::endl;
				}
				else
				{
					for (i=0; i<nparms; i++)
					{
						iacc22[i] = 1;
					}
				}
				// Read the values for nbcol.
				if (dgg)
				{
					for (i=0; i<nparms; i++)
					{
						nin >> nbcol[i];
						if (nbcol[i]<0)
						{
							nout << str9989a << "NBCOL =" << std::setw(6) << nbcol[i] << str9989b
							     << std::setw(6) << 0 << std::endl;
							fatal = true;
						}
						else if (nbcol[i]>nmax)
						{
							nout << str9988a << "NBCOL =" << std::setw(6) << nbcol[i] << str9988b
							     << std::setw(6) << nmax << std::endl;
							fatal = true;
						}
					}
					nin.ignore(100, '\n');
					nout << "    NBCOL:";
					for (i=0; i<std::min(10, nparms); i++)
					{
						nout << std::setw(6) << nbcol[i];
					}
					if (nparms>10)
					{
						nout << "\n          ";
						for (i=10; i<std::min(20, nparms); i++)
						{
							nout << std::setw(6) << nbcol[i];
						}
					}
					nout << std::endl;
				}
				else
				{
					for (i=0; i<nparms; i++)
					{
						nbcol[i] = 1;
					}
				}
			}
			// Calculate and print the machine dependent constants.
			{
				char const* str9981a = " Relative machine ";
				char const* str9981b = " is taken to be";
				nout << '\n';
				eps = this->dlamch("Underflow threshold");
				nout << str9981a << "underflow" << str9981b << std::setw(16)
				     << std::setprecision(6) << std::scientific << eps << '\n';
				eps = this->dlamch("Overflow threshold");
				nout << str9981a << "overflow " << str9981b << std::setw(16)
				     << std::setprecision(6) << std::scientific << eps << '\n';
				eps = this->dlamch("Epsilon");
				nout << str9981a << "precision" << str9981b << std::setw(16)
				     << std::setprecision(6) << std::scientific << eps << '\n'
				     << std::defaultfloat;
			}
			// Read the threshold value for the test ratios.
			nin >> thresh;
			nin.ignore(100, '\n');
			nout << "\n Routines pass computational tests if test ratio is less than"
			     << std::setw(8) << std::setprecision(2) << thresh << std::endl;
			if (sep || svd || dgg)
			{
				// Read the flag that indicates whether to test LAPACK routines.
				nin >> c1;
				nin.ignore(100, '\n');
				tstchk = (std::toupper(c1)=='T');
				// Read the flag that indicates whether to test driver routines.
				nin >> c1;
				nin.ignore(100, '\n');
				tstdrv = (std::toupper(c1)=='T');
			}
			// Read the flag that indicates whether to test the error exits.
			nin >> c1;
			nin.ignore(100, '\n');
			tsterr = (std::toupper(c1)=='T');
			// Read the code describing how to set the random number seed.
			nin >> newsd;
			nin.ignore(100, '\n');
			// If newsd = 2, read another line with 4 integers for the seed.
			if (newsd==2)
			{
				for (i=0; i<4; i++)
				{
					nin >> ioldsd[i];
				}
				nin.ignore(100, '\n');
			}
			for (i=0; i<4; i++)
			{
				iseed[i] = ioldsd[i];
			}
			if (fatal)
			{
				nout << "\n Execution not attempted due to input errors" << std::endl;
				delete[] dotype;
				delete[] logwrk;
				delete[] iwork;
				delete[] kval;
				delete[] mval;
				delete[] mxbval;
				delete[] nbcol;
				delete[] nbmin;
				delete[] nbval;
				delete[] nsval;
				delete[] nval;
				delete[] nxval;
				delete[] pval;
				delete[] inmin;
				delete[] inwin;
				delete[] inibl;
				delete[] ishfts;
				delete[] iacc22;
				delete[] A;
				delete[] B;
				delete[] C;
				delete[] D;
				delete[] result;
				delete[] taua;
				delete[] taub;
				delete[] work;
				delete[] x;
				return;
			}
			// Read the input lines indicating the test path and its parameters. The first three
			// characters indicate the test path, and the number of test matrix types must be the
			// first nonblank item in columns 4-80.
			endbreak = false;
			do
			{
				if (!(dgx || dxv))
				{
					while (true)
					{
						nin.getline(line, 80);
						if (!nin.good())
						{
							endbreak = true;
							break;
						}
						std::strncpy(c3, line, 3);
						c3[3] = '\0';
						lenp = std::strlen(line);
						i = 2;
						itmp = 0;
						i1 = -1;
						errcont = false;
						while (true)
						{
							i++;
							if (i>=lenp)
							{
								if (i1>=0)
								{
									break;
								}
								else
								{
									ntypes = maxt;
									break;
								}
							}
							if (line[i]!=' ' && line[i]!=',')
							{
								i1 = i;
								// Check that a valid integer was read
								ic = line[i1] - '0';
								if (ic<0 || ic>9)
								{
									nout << "\n\n *** Invalid integer value in column "
									     << std::setw(2) << i+1 << " of input line:\n" << line
									     << std::endl;
									errcont = true;
									break;
								}
								itmp = 10*itmp + ic;
							}
							else if (i1>=0)
							{
								break;
							}
						}
						if (errcont)
						{
							continue;
						}
						ntypes = itmp;
						// Skip the tests if ntypes is <= 0.
						if (!(dev || des || dvx || dsx || dgv || dgs) && ntypes<=0)
						{
							nout << "\n\n " << c3 << str9990 << std::endl;
							continue;
						}
						break;
					}
					if (endbreak)
					{
						break;
					}
				}
				else
				{
					if (dxv)
					{
						std::strncpy(c3, "DXV", 4);
					}
					if (dgx)
					{
						std::strncpy(c3, "DGX", 4);
					}
				}
				// Reset the random number seed.
				if (newsd==0)
				{
					for (k=0; k<4; k++)
					{
						iseed[k] = ioldsd[k];
					}
				}
				if (strncmp(c3, "DHS", 3)==0 || std::strncmp(c3, "NEP", 3)==0)
				{
					// -------------------------------------
					// NEP:  Nonsymmetric Eigenvalue Problem
					// -------------------------------------
					// Vary the parameters
					// NB    = block size
					// nbmin = minimum block size
					// NX    = crossover point
					// NS    = number of shifts
					// MAXB  = minimum submatrix size
					maxtyp = 21;
					ntypes = std::min(maxtyp, ntypes);
					alareq(c3, ntypes, dotype, maxtyp, nin, nout);
					xlaenv(1, 1);
					if (tsterr)
					{
						//CALL DERRHS("DHSEQR", nout)
						std::cerr << "DERRHS not yet implemented" << std::endl;
					}
					for (i=0; i<nparms; i++)
					{
						xlaenv(1,  nbval[i]);
						xlaenv(2,  nbmin[i]);
						xlaenv(3,  nxval[i]);
						xlaenv(12, std::max(11, inmin[i]));
						xlaenv(13, inwin[i]);
						xlaenv(14, inibl[i]);
						xlaenv(15, ishfts[i]);
						xlaenv(16, iacc22[i]);
						if (newsd==0)
						{
							for (k=0; k<4; k++)
							{
								iseed[k] = ioldsd[k];
							}
						}
						nout << "\n\n " << c3 << ":  NB =" << std::setw(4) << nbval[i]
						     << ", NBMIN =" << std::setw(4) << nbmin[i] << ", NX =" << std::setw(4)
						     << nxval[i] << ", INMIN=" << std::setw(4) << std::max(11, inmin[i])
						     << ", INWIN =" << std::setw(4) << inwin[i] << ", INIBL ="
						     << std::setw(4) << inibl[i] << ", ISHFTS =" << std::setw(4)
						     << ishfts[i] << ", IACC22 =" << std::setw(4) << iacc22[i]
						     << std::endl;
						//CALL DCHKHS(nn, nval, maxtyp, dotype, iseed, thresh, nout, A[0], nmax, A[nmax2], A[nmax2*2], A[nmax2*3], A[nmax2*4], nmax, A[nmax2*5], A[nmax2*6], D[0], D[nmax], D[nmax*2], D[nmax*3], D[nmax*4], D[nmax*5], A[nmax2*7], A[nmax2*8], A[nmax2*9], A[nmax2*10], A[nmax2*11], D[nmax*6], work, lwork, iwork, logwrk, result, info)
						std::cerr << "DCHKHS not yet implemented" << std::endl;
						if (info!=0)
						{
							nout << str9980 << "DCHKHS = " << std::setw(4) << info << std::endl;
						}
					}
				}
				else if (std::strncmp(c3, "DST", 3)==0 || std::strncmp(c3, "SEP", 3)==0
				      || std::strncmp(c3, "SE2", 3)==0)
				{
					// ----------------------------------
					// SEP:  Symmetric Eigenvalue Problem
					// ----------------------------------
					// Vary the parameters
					// NB    = block size
					// nbmin = minimum block size
					// NX    = crossover point
					maxtyp = 21;
					ntypes = std::min(maxtyp, ntypes);
					alareq(c3, ntypes, dotype, maxtyp, nin, nout);
					xlaenv(1, 1);
					xlaenv(9, 25);
					if (tsterr)
					{
						//CALL DERRST("DST", nout)
						std::cerr << "DERRST not yet implemented" << std::endl;
					}
					for (i=0; i<nparms; i++)
					{
						xlaenv(1, nbval[i]);
						xlaenv(2, nbmin[i]);
						xlaenv(3, nxval[i]);
						if (newsd==0)
						{
							for (k=0; k<4; k++)
							{
								iseed[k] = ioldsd[k];
							}
						}
						nout << "\n\n " << c3 << ":  NB =" << std::setw(4) << nbval[i]
						     << ", NBMIN =" << std::setw(4) << nbmin[i] << ", NX =" << std::setw(4)
						     << nxval[i] << std::endl;
						if (tstchk)
						{
							if (std::strncmp(c3, "SE2", 3)==0)
							{
								//CALL DCHKST2STG(nn, nval, maxtyp, dotype, iseed, thresh, nout, A[0], nmax, A[nmax2], D[0], D[nmax], D[nmax*2], D[nmax*3], D[nmax*4], D[nmax*5], D[nmax*6], D[nmax*7], D[nmax*8], D[nmax*9], D[nmax*10], A[nmax2*2], nmax, A[nmax2*3], A[nmax2*4], D[nmax*11], A[nmax2*5], work, lwork, iwork, liwork, result, info)
								std::cerr << "DCHKST2STG not yet implemented" << std::endl;
							}
							else
							{
								//CALL DCHKST(nn, nval, maxtyp, dotype, iseed, thresh, nout, A[0], nmax, A[nmax2], D[0], D[nmax], D[nmax*2], D[nmax*3], D[nmax*4], D[nmax*5], D[nmax*6], D[nmax*7], D[nmax*8], D[nmax*9], D[nmax*10], A[nmax2*2], nmax, A[nmax2*3], A[nmax2*4], D[nmax*11], A[nmax2*5], work, lwork, iwork, liwork, result, info)
								std::cerr << "DCHKST not yet implemented" << std::endl;
							}
							if (info!=0)
							{
								nout << str9980 << "DCHKST = " << std::setw(4) << info
								     << std::endl;
							}
						}
						if (tstdrv)
						{
							if ((std::strncmp(c3, "SE2", 3)==0))
							{
								//CALL DDRVST2STG(nn, nval, 18, dotype, iseed, thresh, nout, A[0], nmax, D[nmax*2], D[nmax*3], D[nmax*4], D[nmax*5], D[nmax*7], D[nmax*8], D[nmax*9], D[nmax*10], A[nmax2], nmax, A[nmax2*2], D[nmax*11], A[nmax2*3], work, lwork, iwork, liwork, result, info)
								std::cerr << "DDRVST2STG not yet implemented" << std::endl;
							}
							else
							{
								//CALL DDRVST(nn, nval, 18, dotype, iseed, thresh, nout, A[0], nmax, D[nmax*2], D[nmax*3], D[nmax*4], D[nmax*5], D[nmax*7], D[nmax*8], D[nmax*9], D[nmax*10], A[nmax2], nmax, A[nmax2*2], D[nmax*11], A[nmax2*3], work, lwork, iwork, liwork, result, info)
								std::cerr << "DDRVST not yet implemented" << std::endl;
							}
							if (info!=0)
							{
								nout << str9980 << "DDRVST = " << std::setw(4) << info
								     << std::endl;
							}
						}
					}
				}
				else if (std::strncmp(c3, "DSG", 3)==0)
				{
					// ----------------------------------------------
					// DSG:  Symmetric Generalized Eigenvalue Problem
					// ----------------------------------------------
					// Vary the parameters
					// NB    = block size
					// nbmin = minimum block size
					// NX    = crossover point
					maxtyp = 21;
					ntypes = std::min(maxtyp, ntypes);
					alareq(c3, ntypes, dotype, maxtyp, nin, nout);
					xlaenv(9, 25);
					for (i=0; i<nparms; i++)
					{
						xlaenv(1, nbval[i]);
						xlaenv(2, nbmin[i]);
						xlaenv(3, nxval[i]);
						if (newsd==0)
						{
							for (k=0; k<4; k++)
							{
								iseed[k] = ioldsd[k];
							}
						}
						nout << "\n\n " << c3 << ":  NB =" << std::setw(4) << nbval[i]
						     << ", NBMIN =" << std::setw(4) << nbmin[i] << ", NX =" << std::setw(4)
						     << nxval[i] << std::endl;
						if (tstchk)
						{
							//// CALL DDRVSG(nn, nval, maxtyp, dotype, iseed, thresh, nout, A[0], nmax, A[nmax2], nmax, D[nmax*2], A[nmax2*2], nmax, A[nmax2*3], A[nmax2*4], A[nmax2*5], A[nmax2*6], work, lwork, iwork, liwork, result, info)
							//CALL DDRVSG2STG(nn, nval, maxtyp, dotype, iseed, thresh, nout, A[0], nmax, A[nmax2], nmax, D[nmax*2], D[nmax*2], A[nmax2*2], nmax, A[nmax2*3], A[nmax2*4], A[nmax2*5], A[nmax2*6], work, lwork, iwork, liwork, result, info)
							std::cerr << "DDRVSG2STG not yet implemented" << std::endl;
							if (info!=0)
							{
								nout << str9980 << "DDRVSG = " << std::setw(4) << info
								     << std::endl;
							}
						}
					}
				}
				else if (std::strncmp(c3, "DBD", 3)==0 || std::strncmp(c3, "SVD", 3)==0)
				{
					// ----------------------------------
					// SVD:  Singular Value Decomposition
					// ----------------------------------
					// Vary the parameters
					// NB    = block size
					// nbmin = minimum block size
					// NX    = crossover point
					// nrhs  = number of right hand sides
					maxtyp = 16;
					ntypes = std::min(maxtyp, ntypes);
					alareq(c3, ntypes, dotype, maxtyp, nin, nout);
					xlaenv(1, 1);
					xlaenv(9, 25);
					// Test the error exits
					if (tsterr && tstchk)
					{
						//CALL DERRBD("DBD", nout)
						std::cerr << "DERRBD not yet implemented" << std::endl;
					}
					if (tsterr && tstdrv)
					{
						//CALL DERRED("DBD", nout)
						std::cerr << "DERRED not yet implemented" << std::endl;
					}
					for (i=0; i<nparms; i++)
					{
						nrhs = nsval[i];
						xlaenv(1, nbval[i]);
						xlaenv(2, nbmin[i]);
						xlaenv(3, nxval[i]);
						if (newsd==0)
						{
							for (k=0; k<4; k++)
							{
								iseed[k] = ioldsd[k];
							}
						}
						nout << "\n\n " << c3 << ":  NB =" << std::setw(4) << nbval[i]
						     << ", NBMIN =" << std::setw(4) << nbmin[i] << ", NX =" << std::setw(4)
						     << nxval[i] << ", NRHS =" << std::setw(4) << nrhs << std::endl;
						if (tstchk)
						{
							dchkbd(nn, mval, nval, maxtyp, dotype, nrhs, iseed, thresh, &A[0],
							       nmax, &D[0], &D[nmax], &D[nmax*2], &D[nmax*3], &A[nmax2], nmax,
							       &A[nmax2*2], &A[nmax2*3], &A[nmax2*4], nmax, &A[nmax2*5], nmax,
							       &A[nmax2*6], &A[nmax2*7], work, lwork, iwork, nout, info);
							if (info!=0)
							{
								nout << str9980 << "DCHKBD = " << std::setw(4) << info
								     << std::endl;
							}
						}
						if (tstdrv)
						{
							//CALL DDRVBD(nn, mval, nval, maxtyp, dotype, iseed, thresh, A[0], nmax, A[nmax2], nmax, A[nmax2*2], nmax, A[nmax2*3], A[nmax2*4], A[nmax2*5], D[0], D[nmax], D[nmax*2], work, lwork, iwork, nout, info)
							std::cerr << "DDRVBD not yet implemented" << std::endl;
						}
					}
				}
				else if (std::strncmp(c3, "DEV", 3)==0)
				{
					// --------------------------------------------
					// DEV:  Nonsymmetric Eigenvalue Problem Driver
					// DGEEV (eigenvalues and eigenvectors)
					// --------------------------------------------
					maxtyp = 21;
					ntypes = std::min(maxtyp, ntypes);
					if (ntypes<=0)
					{
						nout << "\n\n " << c3 << str9990 << std::endl;
					}
					else
					{
						if (tsterr)
						{
							//CALL DERRED(c3, nout)
							std::cerr << "DERRED not yet implemented" << std::endl;
						}
						alareq(c3, ntypes, dotype, maxtyp, nin, nout);
						ddrvev(nn, nval, ntypes, dotype, iseed, thresh, nout, &A[0], nmax,
						       &A[nmax2], &D[0], &D[nmax], &D[nmax*2], &D[nmax*3], &A[nmax2*2],
						       nmax, &A[nmax2*3], nmax, &A[nmax2*4], nmax, result, work, lwork,
						       iwork, info);
						if (info!=0)
						{
							nout << str9980 << "DGEEV = " << std::setw(4) << info << std::endl;
						}
					}
					nout << str9973 << std::endl;
					break;
				}
				else if (std::strncmp(c3, "DES", 3)==0)
				{
					// --------------------------------------------
					// DES:  Nonsymmetric Eigenvalue Problem Driver
					// DGEES (Schur form)
					// --------------------------------------------
					maxtyp = 21;
					ntypes = std::min(maxtyp, ntypes);
					if (ntypes<=0)
					{
						nout << "\n\n " << c3 << str9990 << std::endl;
					}
					else
					{
						if (tsterr)
						{
							//CALL DERRED(c3, nout)
							std::cerr << "DERRED not yet implemented" << std::endl;
						}
						alareq(c3, ntypes, dotype, maxtyp, nin, nout);
						//CALL DDRVES(nn, nval, ntypes, dotype, iseed, thresh, nout, A[0], nmax, A[nmax2], A[nmax2*2], D[0], D[nmax], D[nmax*2], D[nmax*3], A[nmax2*3], nmax, result, work, lwork, iwork, logwrk, info)
						std::cerr << "DDRVES not yet implemented" << std::endl;
						if (info!=0)
						{
							nout << str9980 << "DGEES = " << std::setw(4) << info << std::endl;
						}
					}
					nout << str9973 << std::endl;
					break;
				}
				else if (std::strncmp(c3, "DVX", 3)==0)
				{
					// --------------------------------------------------------------
					// DVX:  Nonsymmetric Eigenvalue Problem Expert Driver
					// DGEEVX (eigenvalues, eigenvectors and condition numbers)
					// --------------------------------------------------------------
					maxtyp = 21;
					ntypes = std::min(maxtyp, ntypes);
					if (ntypes<0)
					{
						nout << "\n\n " << c3 << str9990 << std::endl;
					}
					else
					{
						if (tsterr)
						{
							//CALL DERRED(c3, nout)
							std::cerr << "DERRED not yet implemented" << std::endl;
						}
						alareq(c3, ntypes, dotype, maxtyp, nin, nout);
						//CALL DDRVVX(nn, nval, ntypes, dotype, iseed, thresh, nin, nout, A[0], nmax, A[nmax2], D[0], D[nmax], D[nmax*2], D[nmax*3], A[nmax2*2], nmax, A[nmax2*3], nmax, A[nmax2*4], nmax, D[nmax*4], D[nmax*5], D[nmax*6], D[nmax*7], D[nmax*8], D[nmax*9], D[nmax*10], D[nmax*11], result, work, lwork, iwork, info)
						int _i, _n;// TODO remove substitute code from here------------------------
						while (nin.good())
						{
							nin >> _n;
							nin.ignore(100, '\n');
							if (!nin.good()) break;
							// Read input data until N=0
							if (_n==0) break;
							for (_i=0; _i<(_n*(((_n-1)/6)+1)); _i++)
							{
								nin.ignore(100, '\n');
							}
							for (_i=0; _i<_n; _i++)
							{
								nin.ignore(100, '\n');
							}
						}//           untill here -------------------------------------------------
						std::cerr << "DDRVVX not yet implemented" << std::endl;
						if (info!=0)
						{
							nout << str9980 << "DGEEVX = " << std::setw(4) << info << std::endl;
						}
					}
					nout << str9973 << std::endl;
					break;
				}
				else if (std::strncmp(c3, "DSX", 3)==0)
				{
					// ---------------------------------------------------
					// DSX:  Nonsymmetric Eigenvalue Problem Expert Driver
					// DGEESX (Schur form and condition numbers)
					// ---------------------------------------------------
					maxtyp = 21;
					ntypes = std::min(maxtyp, ntypes);
					if (ntypes<0)
					{
						nout << "\n\n " << c3 << str9990 << std::endl;
					}
					else
					{
						if (tsterr)
						{
							//CALL DERRED(c3, nout)
							std::cerr << "DERRED not yet implemented" << std::endl;
						}
						alareq(c3, ntypes, dotype, maxtyp, nin, nout);
						// CALL DDRVSX(nn, nval, ntypes, dotype, iseed, thresh, nin, nout, A[0], nmax, A[nmax2], A[nmax2*2], D[0], D[nmax], D[nmax*2], D[nmax*3], D[nmax*4], D[nmax*5], A[nmax2*3], nmax, A[nmax2*4], result, work, lwork, iwork, logwrk, info)
						int _i, _n, _ns;// TODO remove substitute code from here-------------------
						while (nin.good())
						{
							nin >> _n >> _ns;
							nin.ignore(100, '\n');
							if (!nin.good()) break;
							if (_n==0) break;
							if (_ns>0)
							{
								for (_i=0; _i<((_ns-1)/6)+1; _i++)
								{
									nin.ignore(100, '\n');
								}
							}
							for (_i=0; _i<(_n*(((_n-1)/6)+1)); _i++)
							{
								nin.ignore(100, '\n');
							}
							nin.ignore(100, '\n');
						}//           untill here -------------------------------------------------
						std::cerr << "DDRVVX not yet implemented" << std::endl;
						if (info!=0)
						{
							nout << str9980 << "DGEESX = " << std::setw(4) << info << std::endl;
						}
					}
					nout << str9973 << std::endl;
					break;
				}
				else if (std::strncmp(c3, "DGG", 3)==0)
				{
					// -------------------------------------------------
					// DGG:  Generalized Nonsymmetric Eigenvalue Problem
					// -------------------------------------------------
					// Vary the parameters
					// NB    = block size
					// nbmin = minimum block size
					// NS    = number of shifts
					// MAXB  = minimum submatrix size
					// iacc22: structured matrix multiply
					// NBCOL = minimum column dimension for blocks
					maxtyp = 26;
					ntypes = std::min(maxtyp, ntypes);
					alareq(c3, ntypes, dotype, maxtyp, nin, nout);
					xlaenv(1,1);
					if (tstchk && tsterr)
					{
						//CALL DERRGG(c3, nout)
						std::cerr << "DERRGG not yet implemented" << std::endl;
					}
					for (i=0; i<nparms; i++)
					{
						xlaenv(1, nbval[i]);
						xlaenv(2, nbmin[i]);
						xlaenv(4, nsval[i]);
						xlaenv(8, mxbval[i]);
						xlaenv(16, iacc22[i]);
						xlaenv(5, nbcol[i]);
						if (newsd==0)
						{
							for (k=0; k<4; k++)
							{
								iseed[k] = ioldsd[k];
							}
						}
						nout << "\n\n " << c3 << ":  NB =" << std::setw(4) << nbval[i]
						     << ", NBMIN =" << std::setw(4) << nbmin[i] << ", NS =" << std::setw(4)
						     << nsval[i] << ", MAXB =" << std::setw(4) << mxbval[i] << ", IACC22 ="
						     << std::setw(4) << iacc22[i] << ", NBCOL =" << std::setw(4)
						     << nbcol[i] << std::endl;
						tstdif = false;
						thrshn = real(10.0);
						if (tstchk)
						{
							//CALL DCHKGG(nn, nval, maxtyp, dotype, iseed, thresh, tstdif, thrshn, nout, A[0], nmax, A[nmax2], A[nmax2*2], A[nmax2*3], A[nmax2*4], A[nmax2*5], A[nmax2*6], A[nmax2*7], A[nmax2*8], nmax, A[nmax2*9], A[nmax2*10], A[nmax2*11], D[0], D[nmax], D[nmax*2], D[nmax*3], D[nmax*4], D[nmax*5], A[nmax2*12], A[nmax2*13], work, lwork, logwrk, result, info)
							std::cerr << "DCHKGG not yet implemented" << std::endl;
							if (info!=0)
							{
								nout << str9980 << "DCHKGG = " << std::setw(4) << info
								     << std::endl;
							}
						}
					}
				}
				else if (std::strncmp(c3, "DGS", 3)==0)
				{
					// -------------------------------------------------
					// DGS:  Generalized Nonsymmetric Eigenvalue Problem
					// DGGES (Schur form)
					// -------------------------------------------------
					maxtyp = 26;
					ntypes = std::min(maxtyp, ntypes);
					if (ntypes<=0)
					{
						nout << "\n\n " << c3 << str9990 << std::endl;
					}
					else
					{
						if (tsterr)
						{
							//CALL DERRGG(c3, nout)
							std::cerr << "DERRGG not yet implemented" << std::endl;
						}
						alareq(c3, ntypes, dotype, maxtyp, nin, nout);
						//CALL DDRGES(nn, nval, maxtyp, dotype, iseed, thresh, nout, A[0], nmax, A[nmax2], A[nmax2*2], A[nmax2*3], A[nmax2*6], nmax, A[nmax2*7], D[0], D[nmax], D[nmax*2], work, lwork, result, logwrk, info)
						std::cerr << "DERRGG not yet implemented" << std::endl;
						if (info!=0)
						{
							nout << str9980 << "DDRGES = " << std::setw(4) << info << std::endl;
						}
						// Blocked version
						xlaenv(16, 2);
						//CALL DDRGES3(nn, nval, maxtyp, dotype, iseed, thresh, nout, A[0], nmax, A[nmax2], A[nmax2*2], A[nmax2*3], A[nmax2*6], nmax, A[nmax2*7], D[0], D[nmax], D[nmax*2], work, lwork, result, logwrk, info)
						std::cerr << "DDRGES3 not yet implemented" << std::endl;
						if (info!=0)
						{
							nout << str9980 << "DDRGES3 = " << std::setw(4) << info << std::endl;
						}
					}
					nout << str9973 << std::endl;
					break;
				}
				else if (dgx)
				{
					// -------------------------------------------------
					// DGX:  Generalized Nonsymmetric Eigenvalue Problem
					// DGGESX (Schur form and condition numbers)
					// -------------------------------------------------
					maxtyp = 5;
					ntypes = maxtyp;
					if (nn<0)
					{
						nout << "\n\n " << c3 << str9990 << std::endl;
					}
					else
					{
						if (tsterr)
						{
							//CALL DERRGG(c3, nout)
							std::cerr << "DERRGG not yet implemented" << std::endl;
						}
						alareq(c3, ntypes, dotype, maxtyp, nin, nout);
						xlaenv(5, 2);
						// CALL DDRGSX(nn, ncmax, thresh, nin, nout, A[0], nmax, A[nmax2], A[nmax2*2], A[nmax2*3], A[nmax2*4], A[nmax2*5], D[0], D[nmax], D[nmax*2], C[0], ncmax*ncmax, A[nmax2*11], work, lwork, iwork, liwork, logwrk, info)
						std::cerr << "DDRGSX not yet implemented" << std::endl;
						if (info!=0)
						{
							nout << str9980 << "DDRGSX = " << std::setw(4) << info << std::endl;
						}
					}
					nout << str9973 << std::endl;
					break;
				}
				else if (std::strncmp(c3, "DGV", 3)==0)
				{
					// -------------------------------------------------
					// DGV:  Generalized Nonsymmetric Eigenvalue Problem
					// DGGEV (Eigenvalue/vector form)
					// -------------------------------------------------
					maxtyp = 26;
					ntypes = std::min(maxtyp, ntypes);
					if (ntypes<=0)
					{
						nout << "\n\n " << c3 << str9990 << std::endl;
					}
					else
					{
						if (tsterr)
						{
							//CALL DERRGG(c3, nout)
							std::cerr << "DERRGG not yet implemented" << std::endl;
						}
						alareq(c3, ntypes, dotype, maxtyp, nin, nout);
						//CALL DDRGEV(nn, nval, maxtyp, dotype, iseed, thresh, nout, A[0], nmax, A[nmax2], A[nmax2*2], A[nmax2*3], A[nmax2*6], nmax, A[nmax2*7], A[nmax2*8], nmax, D[0], D[nmax], D[nmax*2], D[nmax*3], D[nmax*4], D[nmax*5], work, lwork, result, info)
						std::cerr << "DDRGEV not yet implemented" << std::endl;
						if (info!=0)
						{
							nout << str9980 << "DDRGEV = " << std::setw(4) << info << std::endl;
						}
						// Blocked version
						//CALL DDRGEV3(nn, nval, maxtyp, dotype, iseed, thresh, nout, A[0], nmax, A[nmax2], A[nmax2*2], A[nmax2*3], A[nmax2*6], nmax, A[nmax2*7], A[nmax2*8], nmax, D[0], D[nmax], D[nmax*2], D[nmax*3], D[nmax*4], D[nmax*5], work, lwork, result, info)
						std::cerr << "DDRGEV3 not yet implemented" << std::endl;
						if (info!=0)
						{
							nout << str9980 << "DDRGEV3 = " << std::setw(4) << info << std::endl;
						}
					}
					nout << str9973 << std::endl;
					break;
				}
				else if (dxv)
				{
					// -------------------------------------------------
					// DXV:  Generalized Nonsymmetric Eigenvalue Problem
					// DGGEVX (eigenvalue/vector with condition numbers)
					// -------------------------------------------------
					maxtyp = 2;
					ntypes = maxtyp;
					if (nn<0)
					{
						nout << "\n\n " << c3 << str9990 << std::endl;
					}
					else
					{
						if (tsterr)
						{
							//CALL DERRGG(c3, nout)
							std::cerr << "DERRGG not yet implemented" << std::endl;
						}
						alareq(c3, ntypes, dotype, maxtyp, nin, nout);
						//CALL DDRGVX(nn, thresh, nin, nout, A[0], nmax, A[nmax2], A[nmax2*2], A[nmax2*3], D[0], D[nmax], D[nmax*2], A[nmax2*4], A[nmax2*5], iwork[0], iwork[1], D[nmax*3], D[nmax*4], D[nmax*5], D[nmax*6], D[nmax*7], D[nmax*8], work, lwork, iwork[2], liwork-2, result, logwrk, info)
						std::cerr << "DDRGVX not yet implemented" << std::endl;
						if (info!=0)
						{
							nout << str9980 << "DDRGVX = " << std::setw(4) << info << std::endl;
						}
					}
					nout << str9973 << std::endl;
					break;
				}
				else if (std::strncmp(c3, "DSB", 3)==0)
				{
					// ------------------------------
					// DSB:  Symmetric Band Reduction
					// ------------------------------
					maxtyp = 15;
					ntypes = std::min(maxtyp, ntypes);
					alareq(c3, ntypes, dotype, maxtyp, nin, nout);
					if (tsterr)
					{
						//CALL DERRST("DSB", nout)
						std::cerr << "DERRST not yet implemented" << std::endl;
					}
					//// CALL DCHKSB(nn, nval, nk, kval, maxtyp, dotype, iseed, thresh, nout, A[0], nmax, D[0], D[nmax], A[nmax2], nmax, work, lwork, result, info)
					//CALL DCHKSB2STG(nn, nval, nk, kval, maxtyp, dotype, iseed, thresh, nout, A[0], nmax, D[0],  D[nmax], D[nmax*2], D[nmax*3], D[nmax*4], A[nmax2], nmax, work, lwork, result, info)
					std::cerr << "DCHKSB2STG not yet implemented" << std::endl;
					if (info!=0)
					{
						nout << str9980 << "DCHKSB = " << std::setw(4) << info << std::endl;
					}
				}
				else if ((std::strncmp(c3, "DBB", 3)==0))
				{
					// ------------------------------
					// DBB:  General Band Reduction
					// ------------------------------
					maxtyp = 15;
					ntypes = std::min(maxtyp, ntypes);
					alareq(c3, ntypes, dotype, maxtyp, nin, nout);
					for (i=0; i<nparms; i++)
					{
						nrhs = nsval[i];
						if (newsd==0)
						{
							for (k=0; k<4; k++)
							{
								iseed[k] = ioldsd[k];
							}
						}
						nout << "\n\n " << c3 << ":  NRHS =" << std::setw(4) << nrhs << std::endl;
						//CALL DCHKBB(nn, mval, nval, nk, kval, maxtyp, dotype, nrhs, iseed, thresh, nout, A[0], nmax, A[nmax2], 2*nmax, D[0], D[nmax], A[nmax2*3], nmax, A[nmax2*4], nmax, A[nmax2*5], nmax, A[nmax2*6], work, lwork, result, info)
						std::cerr << "DCHKBB not yet implemented" << std::endl;
						if (info!=0)
						{
							nout << str9980 << "DCHKBB = " << std::setw(4) << info << std::endl;
						}
					}
				}
				else if (std::strncmp(c3, "GLM", 3)==0)
				{
					// -----------------------------------------
					// GLM:  Generalized Linear Regression Model
					// -----------------------------------------
					xlaenv(1, 1);
					if (tsterr)
					{
						//CALL DERRGG("GLM", nout)
						std::cerr << "DERRGG not yet implemented" << std::endl;
					}
					//CALL DCKGLM(nn, mval, pval, nval, ntypes, iseed, thresh, nmax, A[0], A[nmax2], B[0], B[nmax2], x, work, D[0], nin, nout, info)
					std::cerr << "DCKGLM not yet implemented" << std::endl;
					if (info!=0)
					{
						nout << str9980 << "DCKGLM = " << std::setw(4) << info << std::endl;
					}
				}
				else if (std::strncmp(c3, "GQR", 3)==0)
				{
					// ------------------------------------------
					// GQR:  Generalized QR and RQ factorizations
					// ------------------------------------------
					xlaenv(1, 1);
					if (tsterr)
					{
						//CALL DERRGG("GQR", nout)
						std::cerr << "DERRGG not yet implemented" << std::endl;
					}
					//CALL DCKGQR(nn, mval, nn, pval, nn, nval, ntypes, iseed, thresh, nmax, A[0], A[nmax2], A[nmax2*2], A[nmax2*3], taua, B[0], B[nmax2], B[nmax2*2], B[nmax2*3], B[nmax2*4], taub, work, D[0], nin, nout, info)
					std::cerr << "DCKGQR not yet implemented" << std::endl;
					if (info!=0)
					{
						nout << str9980 << "DCKGQR = " << std::setw(4) << info << std::endl;
					}
				}
				else if (std::strncmp(c3, "GSV", 3)==0)
				{
					// ----------------------------------------------
					// GSV:  Generalized Singular Value Decomposition
					// ----------------------------------------------
					xlaenv(1,1);
					if (tsterr)
					{
						//CALL DERRGG("GSV", nout)
						std::cerr << "DERRGG not yet implemented" << std::endl;
					}
					//CALL DCKGSV(nn, mval, pval, nval, ntypes, iseed, thresh, nmax, A[0], A[nmax2], B[0], B[nmax2], A[nmax2*2], B[nmax2*2], A[nmax2*3], taua, taub, B[nmax2*3], iwork, work, D[0], nin, nout, info)
					std::cerr << "DCKGSV not yet implemented" << std::endl;
					if (info!=0)
					{
						nout << str9980 << "DCKGSV = " << std::setw(4) << info << std::endl;
					}
				}
				else if (std::strncmp(c3, "CSD", 3)==0)
				{
					// ----------------------------------------------
					// CSD:  CS Decomposition
					// ----------------------------------------------
					xlaenv(1,1);
					if (tsterr)
					{
						//CALL DERRGG("CSD", nout)
						std::cerr << "DERRGG not yet implemented" << std::endl;
					}
					//CALL DCKCSD(nn, mval, pval, nval, ntypes, iseed, thresh, nmax, A[0], A[nmax2], A[nmax2*2], A[nmax2*3], A[nmax2*4], A[nmax2*5], A[nmax2*6], iwork, work, D[0], nin, nout, info)
					std::cerr << "DCKCSD not yet implemented" << std::endl;
					if (info!=0)
					{
						nout << str9980 << "DCKCSD = " << std::setw(4) << info << std::endl;
					}
				}
				else if (std::strncmp(c3, "LSE", 3)==0)
				{
					// --------------------------------------
					// LSE:  Constrained Linear Least Squares
					// --------------------------------------
					xlaenv(1, 1);
					if (tsterr)
					{
						//CALL DERRGG("LSE", nout)
						std::cerr << "DERRGG not yet implemented" << std::endl;
					}
					//CALL DCKLSE(nn, mval, pval, nval, ntypes, iseed, thresh, nmax, A[0], A[nmax2], B[0], B[nmax2], x, work, D[0], nin, nout, info)
					std::cerr << "DCKLSE not yet implemented" << std::endl;
					if (info!=0)
					{
						nout << str9980 << "DCKLSE = " << std::setw(4) << info << std::endl;
					}
				}
				else
				{
					nout << "\n\n " << c3 << ":  Unrecognized path name" << std::endl;
				}
			} while (!(dgx || dxv));
			if (endbreak)
			{
				break;
			}
		}
		delete[] dotype;
		delete[] logwrk;
		delete[] iwork;
		delete[] kval;
		delete[] mval;
		delete[] mxbval;
		delete[] nbcol;
		delete[] nbmin;
		delete[] nbval;
		delete[] nsval;
		delete[] nval;
		delete[] nxval;
		delete[] pval;
		delete[] inmin;
		delete[] inwin;
		delete[] inibl;
		delete[] ishfts;
		delete[] iacc22;
		delete[] A;
		delete[] B;
		delete[] C;
		delete[] D;
		delete[] result;
		delete[] taua;
		delete[] taub;
		delete[] work;
		delete[] x;
		nout << "\n\n End of tests\n";
		std::time(&s2);
		nout << " Total time used = " << std::setw(12) << std::setprecision(2) << s2-s1
		     << " seconds\n" << std::endl;
	}

	/*! §ddrvev
	 *
	 * §ddrvev checks the nonsymmetric eigenvalue problem driver §dgeev.\n
	 * When §ddrvev is called, a number of matrix "sizes" ("n"s) and a number of matrix "types"
	 * are specified. For each size ("n") and each type of matrix, one matrix will be generated and
	 * used to test the nonsymmetric eigenroutines. For each matrix, 7 tests will be performed:\n
	 * 1. $\frac{|\{A}\{Vr}-\{Vr}\{W}|}{\{n}|\{A}|\{ulp}}$\n
	 *    Here §Vr is the matrix of unit right eigenvectors. §W is a block diagonal matrix, with a
	 *    1 by 1 block for each real eigenvalue and a 2 by 2 block for each complex conjugate pair.
	 *    If eigenvalues $j$ and $j+1$ are a complex conjugate pair, so $\{wr}[j]=\{wr}[j+1]=w_r$
	 *    and $\{wi}[j]=-\{wi}[j+1]=w_i$, then the 2 by 2 block corresponding to the pair will be:
	 *    \n $\b{bm} w_r & w_i \\
	 *              -w_i & w_r \e{bm}$\n
	 *    Such a block multiplying an §n by 2 matrix $\b{bm}u_r & u_i\e{bm}$ on the right will be
	 *    the same as multiplying $u_r+i\,u_i$ by $w_r+i\,w_i$.
	 * 2. $\frac{|\{A}^H\{Vl}-\{Vl}\,\{W}^H|}{\{n}|\{A}|\,\{ulp}}$\n
	 *    Here §Vl is the matrix of unit left eigenvectors, $\{A}^H$ is the conjugate transpose of
	 *    §A, and §W is as above.\n
	 * 3. $\frac{\left||\{Vr}[i]|-1\right|}{\{ulp}}$ and whether the largest component is real\n
	 *    $\{Vr}[i]$ denotes the $i$-th column of §Vr.
	 * 4. $\frac{\left||\{Vl}[i]|-1\right|}{\{ulp}}$ and whether the largest component is real\n
	 *    $\{Vl}[i]$ denotes the $i$-th column of §Vl.
	 * 5. $\{W}_\text{full} = \{W}_\text{partial}$\n
	 *    $\{W}_\text{full}$ denotes the eigenvalues computed when both §Vr and §Vl are also
	 *    computed, and $\{W}_\text{partial}$ denotes the eigenvalues computed when only §W, only
	 *    §W and §Vr, or only §W and §Vl are computed.
	 * 6. $\{Vr}_\text{full} = \{Vr}_\text{partial}$\n
	 *    $\{Vr}_\text{full}$ denotes the right eigenvectors computed when both §Vr and §Vl are
	 *    computed, and $\{Vr}_\text{partial}$ denotes the result when only §Vr is computed.
	 * 7. $\{Vl}_\text{full} = \{Vl}_\text{partial}$\n
	 *    $\{Vl}_\text{full}$ denotes the left eigenvectors computed when both §Vr and §Vl are also
	 *    computed, and $\{Vl}_\text{partial}$ denotes the result when only §Vl is computed.
	 * .
	 * \n
	 * The "sizes" are specified by an array $\{nn}[0:\{nsizes}-1]$; the value of each element
	 * $\{nn}[j]$ specifies one size.\n The "types" are specified by a logical array
	 * $\{dotype}[0:\{ntypes}-1]$; if $\{dotype}[j]$ is true, then matrix type $"j"$ will be
	 * generated.\n Currently, the list of possible types is:
	 *     1.  The zero matrix.
	 *     2.  The identity matrix.
	 *     3.  A (transposed) Jordan block, with 1's on the diagonal.
	 *     4.  A diagonal matrix with evenly spaced entries $1,\ldots,\{ulp}$ and random signs.
	 *         ($\{ulp}=(\text{first number larger than }1)-1$)
	 *     5.  A diagonal matrix with geometrically spaced entries $1,\ldots,\{ulp}$ and random
	 *         signs.
	 *     6.  A diagonal matrix with "clustered" entries $1,\{ulp},\ldots,\{ulp}$ and random
	 *         signs.
	 *     7.  Same as 4, but multiplied by a constant near the overflow threshold
	 *     8.  Same as 4, but multiplied by a constant near the underflow threshold
	 *     9.  A matrix of the form $U^TTU$, where $U$ is orthogonal and $T$ has evenly spaced
	 *         entries $1,\ldots,\{ulp}$ with random signs on the diagonal and random O(1) entries
	 *         in the upper triangle.
	 *     10. A matrix of the form $U^TTU$, where $U$ is orthogonal and $T$ has geometrically
	 *         spaced entries $1,\ldots,\{ulp}$ with random signs on the diagonal and random O(1)
	 *         entries in the upper triangle.
	 *     11. A matrix of the form $U^TTU$, where $U$ is orthogonal and $T$ has "clustered"
	 *         entries $1,\{ulp},\ldots,\{ulp}$ with random signs on the diagonal and random O(1)
	 *         entries in the upper triangle.
	 *     12. A matrix of the form $U^TTU$, where $U$ is orthogonal and $T$ has real or complex
	 *         conjugate paired eigenvalues randomly chosen from $[\{ulp},1]$ and random O(1)
	 *         entries in the upper triangle.
	 *     13. A matrix of the form $X^TTX$, where $X$ has condition $\sqrt{\{ulp}}$ and $T$ has
	 *         evenly spaced entries $1,\ldots,\{ulp}$ with random signs on the diagonal and random
	 *         O(1) entries in the upper triangle.
	 *     14. A matrix of the form $X^TTX$, where $X$ has condition $\sqrt{\{ulp}}$ and $T$ has
	 *         geometrically spaced entries $1,\ldots,\{ulp}$ with random signs on the diagonal and
	 *         random O(1) entries in the upper triangle.
	 *     15. A matrix of the form $X^TTX$, where $X$ has condition $\sqrt{\{ulp}}$ and $T$ has
	 *         "clustered" entries $1,\{ulp},\ldots,\{ulp}$ with random signs on the diagonal and
	 *         random O(1) entries in the upper triangle.
	 *     16. A matrix of the form $X^TTX$, where $X$ has condition $\sqrt{\{ulp}}$ and $T$ has
	 *         real or complex conjugate paired eigenvalues randomly chosen from $[\{ulp},1]$ and
	 *         random O(1) entries in the upper triangle.
	 *     17. Same as 16, but multiplied by a constant near the overflow threshold
	 *     18. Same as 16, but multiplied by a constant near the underflow threshold
	 *     19. Nonsymmetric matrix with random entries chosen from $[-1,1]$. If §n is at least 4,
	 *         all entries in first two rows and last row, and first column and last two columns
	 *         are zero.
	 *     20. Same as 19, but multiplied by a constant near the overflow threshold
	 *     21. Same as 19, but multiplied by a constant near the underflow threshold
	 *
	 * \param[in] nsizes
	 *     The number of sizes of matrices to use. If it is zero, §ddrvev does nothing.
	 *     It must be at least zero.
	 *
	 * \param[in] nn
	 *     an integer array, dimension (§nsizes)\n
	 *     An array containing the sizes to be used for the matrices. Zero values will be skipped.
	 *     The values must be at least zero.
	 *
	 * \param[in] ntypes
	 *     The number of elements in §dotype. If it is zero, §ddrvev does nothing. It must be at
	 *     least zero. If it is $\{MAXTYP}+1$ and §nsizes is 1, then an additional type,
	 *     $\{MAXTYP}+1$ is defined, which is to use whatever matrix is in §A. This is only useful
	 *     if $\{dotype}[0:\{MAXTYP}-1]$ is §false and $\{dotype}[\{MAXTYP}]$ is §true.
	 *
	 * \param[in] dotype
	 *     a boolean array, dimension (§ntypes)\n
	 *     If $\{dotype}[j]$ is §true, then for each size in §nn a matrix of that size and of type
	 *     $j$ will be generated. If §ntypes is smaller than the maximum number of types defined
	 *     (parameter §MAXTYP), then types §ntypes through $\{MAXTYP}-1$ will not be generated. If
	 *     §ntypes is larger than §MAXTYP, $\{dotype}[\{MAXTYP}]$ through $\{dotype}[\{ntypes}-1]$
	 *     will be ignored.
	 *
	 * \param[in,out] iseed
	 *     an integer array, dimension (4)\n
	 *     On entry §iseed specifies the seed of the random number generator. The array elements
	 *     should be between 0 and 4095; if not they will be reduced modulo 4096. Also,
	 *     $\{iseed}[3]$ must be odd. The random number generator uses a linear congruential
	 *     sequence limited to small integers, and so should produce machine independent random
	 *     numbers. The values of §iseed are changed on exit, and can be used in the next call to
	 *     §ddrvev to continue the same random number sequence.
	 *
	 * \param[in] thresh
	 *     A test will count as "failed" if the "error", computed as described above, exceeds
	 *     §thresh. Note that the error is scaled to be O(1), so §thresh should be a reasonably
	 *     small multiple of 1, e.g., 10 or 100. In particular, it should not depend on the
	 *     precision (single vs. double) or the size of the matrix. It must be at least zero.
	 *
	 * \param[in] nounit
	 *     The output stream for printing out error messages
	 *     (e.g., if a routine returns §info not equal to 0.)
	 *
	 * \param[out] A
	 *     an array, dimension (§lda, $\max(\{nn})$)\n
	 *     Used to hold the matrix whose eigenvalues are to be computed.
	 *     On exit, §A contains the last matrix actually used.
	 *
	 * \param[in] lda
	 *     The leading dimension of §A, and §H. §lda must be at least 1 and at least $\max(\{nn})$.
	 *
	 * \param[out] H
	 *     an array, dimension (§lda, $\max(\{nn})$)\n
	 *     Another copy of the test matrix §A, modified by §dgeev.
	 *
	 * \param[out] wr, wi
	 *     arrays, dimension ($\max(\{nn})$)\n
	 *     The real and imaginary parts of the eigenvalues of §A.
	 *     On exit, $\{wr}+i\,\{wi}$ are the eigenvalues of the matrix in §A.
	 *
	 * \param[out] wr1, wi1
	 *     arrays, dimension ($\max(\{nn})$)\n
	 *     Like §wr, §wi, these arrays contain the eigenvalues of §A, but those computed when
	 *     §dgeev only computes a partial eigendecomposition, i.e. not the eigenvalues and left and
	 *     right eigenvectors.
	 *
	 * \param[out] Vl
	 *     an array, dimension (§ldvl, $\max(\{nn})$)\n §Vl holds the computed left eigenvectors.
	 *
	 * \param[in]  ldvl Leading dimension of §Vl. Must be at least $\max(1,\max(\{nn}))$.
	 * \param[out] Vr
	 *     an array, dimension (§ldvr, $\max(\{nn})$)\n §Vr holds the computed right eigenvectors.
	 *
	 * \param[in]  ldvr Leading dimension of §Vr. Must be at least $\max(1,\max(\{nn}))$.
	 * \param[out] Lre
	 *     an array, dimension (§ldlre,$\max(\{nn})$)\n
	 *     §Lre holds the computed right or left eigenvectors.
	 *
	 * \param[in]  ldlre  Leading dimension of §Lre. Must be at least $\max(1,\max(\{nn}))$.
	 * \param[out] result
	 *     an array, dimension (7)\n
	 *     The values computed by the seven tests described above.\n
	 *     The values are currently limited to $1/\{ulp}$, to avoid overflow.
	 *
	 * \param[out] work  an array, dimension (§nwork)
	 * \param[in]  nwork
	 *     The number of entries in §work.
	 *     This must be at least $5\{nn}[j]+2\{nn}[j]^2$ for all $j$.
	 *
	 * \param[out] iwork an integer array, dimension ($\max(\{nn})$)
	 * \param[out] info
	 *     If 0, then everything ran OK.\n
	 *      -1: $\{nsizes}<0$\n
	 *      -2: Some $\{nn}[j]<0$\n
	 *      -3: $\{ntypes}<0$\n
	 *      -6: $\{thresh}<0$\n
	 *      -9: $\{lda}  <1$ or $\{lda}  <\{nmax}$, where §nmax is $\max(\{nn}[j])$.\n
	 *     -16: $\{ldvl} <1$ or $\{ldvl} <\{nmax}$, where §nmax is $\max(\{nn}[j])$.\n
	 *     -18: $\{ldvr} <1$ or $\{ldvr} <\{nmax}$, where §nmax is $\max(\{nn}[j])$.\n
	 *     -20: $\{ldlre}<1$ or $\{ldlre}<\{nmax}$, where §nmax is $\max(\{nn}[j])$.\n
	 *     -23: §nwork too small.
	 *     If §dlatmr, §dlatms, §dlatme or §dgeev returns an error code, the absolute value of it
	 *     is returned.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     Some Local Variables and Parameters:\n
	 *     $\begin{tabular}{ll}
	 *         \{MAXTYP}                  & The number of types defined.                     \\
	 *         \{nmax}                    & Largest value in \{nn}.                          \\
	 *         \{nerrs}                   & The number of tests which have exceeded \{thresh}\\
	 *         \{cond}, \{conds},\{imode} & Values to be passed to the matrix generators.    \\
	 *         \{anorm}                   & Norm of \{A}; passed to matrix generators.       \\
	 *         \{ovfl}, \{unfl}           & Overflow and underflow thresholds.               \\
	 *         \{ulp}, \{ulpinv}          & Finest relative precision and its inverse.       \\
	 *         \{rtulpi}                  & Square roots of the previous 4 values. \end{tabular}$\n
	 *     The following four arrays decode §jtype: \n
	 *     $\begin{tabular}{ll}
	 *         \{KTYPE}[j]  & The general type (1-10) for type \(j+1\).                     \\
	 *         \{KMODE}[j]  & The \{MODE} value to be passed to the matrix generator
	 *                        for type \(j+1\).                                             \\
	 *         \{KMAGN}[j]  & The order of magnitude
	 *                        (O(1), O(\(\sqrt{\{overflow}}\)), O(\(\sqrt{\{underflow}}\))) \\
	 *         \{KCONDS}[j] & Select whether \{conds} is to be 1 or \(1/\sqrt(\{ulp})\).
	 *                        (0 means irrelevant.) \end{tabular}$                               */
	void ddrvev(int const nsizes, int const* const nn, int const ntypes, bool const* const dotype,
	            int* const iseed, real const thresh, std::ostream& nounit, real* const A,
	            int const lda, real* const H, real* const wr, real* const wi, real* const wr1,
	            real* const wi1, real* const Vl, int const ldvl, real* const Vr, int const ldvr,
	            real* const Lre, int const ldlre, real* const result, real* const work,
	            int const nwork, int* const iwork, int& info)
	{
		int const MAXTYP = 21;
		int const  KTYPE[MAXTYP] = {1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9};
		int const  KMAGN[MAXTYP] = {1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3};
		int const  KMODE[MAXTYP] = {0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1};
		int const KCONDS[MAXTYP] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0};
		char path[4];
		path[0] = 'D'; // Double precision'
		std::strncpy(&path[1], "EV", 2);
		path[3] = '\0';
		// Check for errors
		int ntestt = 0;
		int ntestf = 0;
		info = 0;
		// Important constants
		bool badnn = false;
		int nmax = 0;
		int j;
		for (j=0; j<nsizes; j++)
		{
			nmax = std::max(nmax, nn[j]);
			if (nn[j]<0)
			{
				badnn = true;
			}
		}
		// Check for errors
		if (nsizes<0)
		{
			info = -1;
		}
		else if (badnn)
		{
			info = -2;
		}
		else if (ntypes<0)
		{
			info = -3;
		}
		else if (thresh<ZERO)
		{
			info = -6;
		}
		else if (!nounit.good())
		{
			info = -7;
		}
		else if (lda<1 || lda<nmax)
		{
			info = -9;
		}
		else if (ldvl<1 || ldvl<nmax)
		{
			info = -16;
		}
		else if (ldvr<1 || ldvr<nmax)
		{
			info = -18;
		}
		else if (ldlre<1 || ldlre<nmax)
		{
			info = -20;
		}
		else if (5*nmax+2*std::pow(nmax, 2)>nwork)
		{
			info = -23;
		}
		if (info!=0)
		{
			this->xerbla("DDRVEV", -info);
			return;
		}
		// Quick return if nothing to do
		if (nsizes==0 || ntypes==0)
		{
			return;
		}
		// More Important constants
		real unfl = this->dlamch("Safe minimum");
		real ovfl = ONE / unfl;
		this->dlabad(unfl, ovfl);
		real ulp = this->dlamch("Precision");
		real ulpinv = ONE / ulp;
		real rtulpi = ONE / std::sqrt(ulp);
		// Set output string constants
		char const* str9993a = " DDRVEV: ";
		char const* str9993b = " returned INFO=";
		char const* str9993c = ".\n         N=";
		char const* str9993d = ", JTYPE=";
		char const* str9993e = ", ISEED=(";
		// Loop over sizes, types
		int nerrs = 0;
		int iinfo, imode, itype, iwk, jcol, jj, jsize, jtype, mtypes, n, nfail, nnwork, ntest;
		real anorm, cond, conds, tnrm, vmx, vrmx, vtst;
		int ioldsd[4];
		real res[2];
		for (jsize=0; jsize<nsizes; jsize++)
		{
			n = nn[jsize];
			if (nsizes!=1)
			{
				mtypes = std::min(MAXTYP, ntypes);
			}
			else
			{
				mtypes = std::min(MAXTYP+1, ntypes);
			}
			for (jtype=0; jtype<mtypes; jtype++)
			{
				if (!dotype[jtype])
				{
					continue;
				}
				// Save iseed in case of an error.
				for (j=0; j<4; j++)
				{
					ioldsd[j] = iseed[j];
				}
				// Compute "A"
				// Control parameters:
				//     KMAGN  KCONDS  KMODE        KTYPE
				// =1  O(1)   1       clustered 1  zero
				// =2  large  large   clustered 2  identity
				// =3  small          exponential  Jordan
				// =4                 arithmetic   diagonal, (w/ eigenvalues)
				// =5                 random log   symmetric, w/ eigenvalues
				// =6                 random       general, w/ eigenvalues
				// =7                              random diagonal
				// =8                              random symmetric
				// =9                              random general
				// =10                             random triangular
				if (mtypes<=MAXTYP)
				{
					itype = KTYPE[jtype];
					imode = KMODE[jtype];
					// Compute norm
					switch(KMAGN[jtype])
					{
						default:
						case 1:
							anorm = ONE;
							break;
						case 2:
							anorm = ovfl * ulp;
							break;
						case 3:
							anorm = unfl * ulpinv;
							break;
					}
					this->dlaset("Full", lda, n, ZERO, ZERO, A, lda);
					iinfo = 0;
					cond = ulpinv;
					// Special Matrices -- Identity & Jordan block
					if (itype==1)
					{
						// Zero
						iinfo = 0;
					}
					else if (itype==2)
					{
						// Identity
						for (jcol=0; jcol<n; jcol++)
						{
							A[jcol+lda*jcol] = anorm;
						}
					}
					else if (itype==3)
					{
						// Jordan Block
						for (jcol=0; jcol<n; jcol++)
						{
							A[jcol+lda*jcol] = anorm;
							if (jcol>0)
							{
								A[jcol+lda*(jcol-1)] = ONE;
							}
						}
					}
					else if (itype==4)
					{
						// Diagonal Matrix, [Eigen]values Specified
						MatGen.dlatms(n, n, "S", iseed, "S", work, imode, cond, anorm, 0, 0, "N",
						              A, lda, &work[n], iinfo);
					}
					else if (itype==5)
					{
						// Symmetric, eigenvalues specified
						MatGen.dlatms(n, n, "S", iseed, "S", work, imode, cond, anorm, n, n, "N",
						              A, lda, &work[n], iinfo);
					}
					else if (itype==6)
					{
						// General, eigenvalues specified
						if (KCONDS[jtype]==1)
						{
							conds = ONE;
						}
						else if (KCONDS[jtype]==2)
						{
							conds = rtulpi;
						}
						else
						{
							conds = ZERO;
						}
						MatGen.dlatme(n, "S", iseed, work, imode, cond, ONE, " ", "T", "T", "T",
						              &work[n], 4, conds, n, n, anorm, A, lda, &work[2*n],
						              iinfo);
					}
					else if (itype==7)
					{
						// Diagonal, random eigenvalues
						MatGen.dlatmr(n, n, "S", iseed, "S", work, 6, ONE, ONE, "T", "N", nullptr,
						              1, ONE, nullptr, 1, ONE, "N", nullptr, 0, 0, ZERO, anorm,
						              "NO", A, lda, nullptr, iinfo);
					}
					else if (itype==8)
					{
						// Symmetric, random eigenvalues
						MatGen.dlatmr(n, n, "S", iseed, "S", work, 6, ONE, ONE, "T", "N", nullptr,
						              1, ONE, nullptr, 1, ONE, "N", nullptr, n, n, ZERO, anorm,
						              "NO", A, lda, nullptr, iinfo);
					}
					else if (itype==9)
					{
						// General, random eigenvalues
						MatGen.dlatmr(n, n, "S", iseed, "N", work, 6, ONE, ONE, "T", "N", nullptr,
						              1, ONE, nullptr, 1, ONE, "N", nullptr, n, n, ZERO, anorm,
						              "NO", A, lda, nullptr, iinfo);
						if (n>=4)
						{
							this->dlaset("Full", 2,   n, ZERO, ZERO,  A,              lda);
							this->dlaset("Full", n-3, 1, ZERO, ZERO, &A[2],           lda);
							this->dlaset("Full", n-3, 2, ZERO, ZERO, &A[2+lda*(n-2)], lda);
							this->dlaset("Full", 1,   n, ZERO, ZERO, &A[n-1],         lda);
						}
					}
					else if (itype==10)
					{
						// Triangular, random eigenvalues
						MatGen.dlatmr(n, n, "S", iseed, "N", work, 6, ONE, ONE, "T", "N", nullptr,
						              1, ONE, nullptr, 1, ONE, "N", nullptr, n, 0, ZERO, anorm,
						              "NO", A, lda, nullptr, iinfo);
					}
					else
					{
						iinfo = 1;
					}
					if (iinfo!=0)
					{
						nounit << str9993a << "Generator" << str9993b << std::setw(6) << iinfo
						       << str9993c << std::setw(6) << n << str9993d << std::setw(6)
						       << jtype+1 << str9993e << std::setw(5) << ioldsd[0] << ','
						       << std::setw(5) << ioldsd[1] << ',' << std::setw(5) << ioldsd[2]
						       << ',' << std::setw(5) << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
						return;
					}
				}
				// Test for minimal and generous workspace
				for (iwk=1; iwk<=2; iwk++)
				{
					if (iwk==1)
					{
						nnwork = 4 * n;
					}
					else
					{
						nnwork = 5*n + 2*n*n;
					}
					nnwork = std::max(nnwork, 1);
					// Initialize result
					for (j=0; j<7; j++)
					{
						result[j] = -ONE;
					}
					// Compute eigenvalues and eigenvectors, and test them
					this->dlacpy("F", n, n, A, lda, H, lda);
					this->dgeev("V", "V", n, H, lda, wr, wi, Vl, ldvl, Vr, ldvr, work, nnwork,
					            iinfo);
					if (iinfo!=0)
					{
						result[0] = ulpinv;
						nounit << str9993a << "DGEEV1" << str9993b << std::setw(6) << iinfo
						       << str9993c << std::setw(6) << n << str9993d << std::setw(6)
						       << jtype+1 << str9993e << std::setw(5) << ioldsd[0] << ','
						       << std::setw(5) << ioldsd[1] << ',' << std::setw(5) << ioldsd[2]
						       << ',' << std::setw(5) << ioldsd[3] << ')' << std::endl;
						info = std::abs(iinfo);
					}
					else
					{
						// Do Test (1)
						dget22("N", "N", "N", n, A, lda, Vr, ldvr, wr, wi, work, res);
						result[0] = res[0];
						// Do Test (2)
						dget22("T", "N", "T", n, A, lda, Vl, ldvl, wr, wi, work, res);
						result[1] = res[0];
						// Do Test (3)
						int vj;
						for (j=0; j<n; j++)
						{
							vj = ldvr * j;
							tnrm = ONE;
							if (wi[j]==ZERO)
							{
								tnrm = Blas<real>::dnrm2(n, &Vr[vj], 1);
							}
							else if (wi[j]>ZERO)
							{
								tnrm = this->dlapy2(Blas<real>::dnrm2(n, &Vr[vj], 1),
								                    Blas<real>::dnrm2(n, &Vr[vj+ldvr], 1));
							}
							result[2] = std::max(result[2],
							                     std::min(ulpinv, std::fabs(tnrm-ONE)/ulp));
							if (wi[j]>ZERO)
							{
								vmx = ZERO;
								vrmx = ZERO;
								for (jj=0; jj<n; jj++)
								{
									vtst = this->dlapy2(Vr[jj+vj], Vr[jj+vj+ldvr]);
									if (vtst>vmx)
									{
										vmx = vtst;
									}
									if (Vr[jj+vj+ldvr]==ZERO && std::fabs(Vr[jj+vj])>vrmx)
									{
										vrmx = std::fabs(Vr[jj+vj]);
									}
								}
								if (vrmx / vmx<ONE-TWO*ulp)
								{
									result[2] = ulpinv;
								}
							}
						}
						// Do Test (4)
						for (j=0; j<n; j++)
						{
							vj = ldvl * j;
							tnrm = ONE;
							if (wi[j]==ZERO)
							{
								tnrm = Blas<real>::dnrm2(n, &Vl[vj], 1);
							}
							else if (wi[j]>ZERO)
							{
								tnrm = this->dlapy2(Blas<real>::dnrm2(n, &Vl[vj],      1),
								                    Blas<real>::dnrm2(n, &Vl[vj+ldvl], 1));
							}
							result[3] = std::max(result[3],
							                     std::min(ulpinv, std::fabs(tnrm-ONE)/ulp));
							if (wi[j]>ZERO)
							{
								vmx = ZERO;
								vrmx = ZERO;
								for (jj=0; jj<n; jj++)
								{
									vtst = this->dlapy2(Vl[jj+vj], Vl[jj+vj+ldvl]);
									if (vtst>vmx)
									{
										vmx = vtst;
									}
									if (Vl[jj+vj+ldvl]==ZERO && std::fabs(Vl[jj+vj])>vrmx)
									{
										vrmx = std::fabs(Vl[jj+vj]);
									}
								}
								if (vrmx/vmx < ONE-TWO*ulp)
								{
									result[3] = ulpinv;
								}
							}
						}
						// Compute eigenvalues only, and test them
						this->dlacpy("F", n, n, A, lda, H, lda);
						this->dgeev("N", "N", n, H, lda, wr1, wi1, nullptr, 1, nullptr, 1, work,
						            nnwork, iinfo);
						if (iinfo!=0)
						{
							result[0] = ulpinv;
							nounit << str9993a << "DGEEV2" << str9993b << std::setw(6) << iinfo
							       << str9993c << std::setw(6) << n << str9993d << std::setw(6)
							       << jtype+1 << str9993e << std::setw(5) << ioldsd[0] << ','
							       << std::setw(5) << ioldsd[1] << ',' << std::setw(5) << ioldsd[2]
							       << ',' << std::setw(5) << ioldsd[3] << ')' << std::endl;
							info = std::abs(iinfo);
						}
						else
						{
							// Do Test (5)
							for (j=0; j<n; j++)
							{
								if (wr[j]!=wr1[j] || wi[j]!=wi1[j])
								{
									result[4] = ulpinv;
								}
							}
							// Compute eigenvalues and right eigenvectors, and test them
							this->dlacpy("F", n, n, A, lda, H, lda);
							this->dgeev("N", "V", n, H, lda, wr1, wi1, nullptr, 1, Lre, ldlre,
							            work, nnwork, iinfo);
							if (iinfo!=0)
							{
								result[0] = ulpinv;
								nounit << str9993a << "DGEEV3" << str9993b << std::setw(6) << iinfo
								       << str9993c << std::setw(6) << n << str9993d << std::setw(6)
								       << jtype+1 << str9993e << std::setw(5) << ioldsd[0] << ','
								       << std::setw(5) << ioldsd[1] << ',' << std::setw(5)
								       << ioldsd[2] << ',' << std::setw(5) << ioldsd[3] << ')'
								       << std::endl;
								info = std::abs(iinfo);
							}
							else
							{
								// Do Test (5) again
								for (j=0; j<n; j++)
								{
									if (wr[j]!=wr1[j] || wi[j]!=wi1[j])
									{
										result[4] = ulpinv;
									}
								}
								// Do Test (6)
								for (j=0; j<n; j++)
								{
									for (jj=0; jj<n; jj++)
									{
										if (Vr[j+ldvr*jj]!=Lre[j+ldlre*jj])
										{
											result[5] = ulpinv;
										}
									}
								}
								// Compute eigenvalues and left eigenvectors, and test them
								this->dlacpy("F", n, n, A, lda, H, lda);
								this->dgeev("V", "N", n, H, lda, wr1, wi1, Lre, ldlre, nullptr, 1,
								            work, nnwork, iinfo);
								if (iinfo!=0)
								{
									result[0] = ulpinv;
									nounit << str9993a << "DGEEV4" << str9993b << std::setw(6)
									       << iinfo << str9993c << std::setw(6) << n << str9993d
									       << std::setw(6) << jtype+1 << str9993e << std::setw(5)
									       << ioldsd[0] << ',' << std::setw(5) << ioldsd[1] << ','
									       << std::setw(5) << ioldsd[2] << ',' << std::setw(5)
									       << ioldsd[3] << ')' << std::endl;
									info = std::abs(iinfo);
								}
								else
								{
									// Do Test (5) again
									for (j=0; j<n; j++)
									{
										if (wr[j]!=wr1[j] || wi[j]!=wi1[j])
										{
											result[4] = ulpinv;
										}
									}
									// Do Test (7)
									for (j=0; j<n; j++)
									{
										for (jj=0; jj<n; jj++)
										{
											if (Vl[j+ldvl*jj]!=Lre[j+ldlre*jj])
											{
												result[6] = ulpinv;
											}
										}
									}
								}
							}
						}
					}
					// End of Loop -- Check for result[j] > thresh
					ntest = 0;
					nfail = 0;
					for (j=0; j<7; j++)
					{
						if (result[j]>=ZERO)
						{
							ntest++;
						}
						if (result[j]>=thresh)
						{
							nfail++;
						}
					}
					if (nfail>0)
					{
						ntestf++;
					}
					if (ntestf==1)
					{
						nounit << "\n " << path << " -- Real Eigenvalue-Eigenvector Decomposition"
						       << " Driver\n Matrix types (see DDRVEV for details): \n";
						nounit << "\n Special Matrices:\n"
						          "  1=Zero matrix.                        "
						          "  5=Diagonal: geometr. spaced entries.\n"
						          "  2=Identity matrix.                    "
						          "  6=Diagonal: clustered entries.\n"
						          "  3=Transposed Jordan block.            "
						          "  7=Diagonal: large, evenly spaced.\n"
						          "  4=Diagonal: evenly spaced entries.    "
						          "  8=Diagonal: small, evenly spaced.\n";
						nounit << " Dense, Non-Symmetric Matrices:\n"
						          "  9=Well-cond., evenly spaced eigenvals."
						          " 14=Ill-cond., geomet. spaced eigenals.\n"
						          " 10=Well-cond., geom. spaced eigenvals. "
						          " 15=Ill-conditioned, clustered e.vals.\n"
						          " 11=Well-conditioned, clustered e.vals. "
						          " 16=Ill-cond., random complex \n"
						          " 12=Well-cond., random complex          "
						          " 17=Ill-cond., large rand. complx \n"
						          " 13=Ill-conditioned, evenly spaced.     "
						          " 18=Ill-cond., small rand. complx \n";
						nounit << " 19=Matrix with random O(1) entries.    "
						          " 21=Matrix with small random entries.\n"
						          " 20=Matrix with large random entries.   \n\n";
						nounit << " Tests performed with test threshold =" << std::setw(8)
						       << std::setprecision(2) << thresh << "\n\n"
						       << " 1 = | A VR - VR W | / (n |A| ulp) \n"
						          " 2 = | transpose(A) VL - VL W | / (n |A| ulp) \n"
						          " 3 = | |VR[i]| - 1 | / ulp \n"
						          " 4 = | |VL[i]| - 1 | / ulp \n"
						          " 5 = 0 if W same no matter if VR or VL computed,"
						                " 1/ulp otherwise\n"
						          " 6 = 0 if VR same no matter if VL computed,  1/ulp otherwise\n"
						          " 7 = 0 if VL same no matter if VR computed, "
						                  " 1/ulp otherwise\n\n";
						ntestf = 2;
					}
					for (j=0; j<7; j++)
					{
						if (result[j]>=thresh)
						{
							nounit << " N=" << std::setw(5) << n << ", IWK=" << std::setw(2) << iwk
							       << ", seed=" << std::setw(2) <<  ioldsd[0] << ','
							       << std::setw(2) <<  ioldsd[1] << ',' << std::setw(2)
							       << ioldsd[2] << ',' << std::setw(2) <<  ioldsd[3] << ','
							       << " type " << std::setw(2) << jtype+1 << ", test("
							       << std::setw(2) << j+1 << ")=" << std::setw(10)
							       << std::setprecision(3) << result[j] << std::endl;
						}
					}
					nerrs += nfail;
					ntestt += ntest;
				}
			}
		}
		// Summary
		dlasum(path, nounit, nerrs, ntestt);
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
				Wmat[0] =  wr[jcol];
				Wmat[1] = -wi[jcol];
				Wmat[2] =  wi[jcol];
				Wmat[3] =  wr[jcol];
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
			resid = this->dlansy("1", "Upper", mnmin, work, ldwork, &work[ldwork*mnmin]);
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

	/*! §dsxt1
	 *
	 * §dsxt1 computes the difference between a set of eigenvalues.\n
	 * §ijob =1: Computes $\max_i\left(\min_j\left|\{d1}[i]-\{d2}[j]\right|\right)$\n
	 * §ijob =2: Computes $\max_i\left(\frac{\min_j\left|\{d1}[i]-\{d2}[j]\right|}
	 *                                      {\{abstol}+|\{d1}[i]|\,\{ulp}}\right)$
	 * \param[in] ijob Specifies the type of tests to be performed. (See above.)
	 * \param[in] d1
	 *     an array, dimension (§n1)\n
	 *     The first array. §d1 should be in increasing order, i.e., $\{d1}[j]\le\{d1}[j+1]$.
	 *
	 * \param[in] n1 The length of §d1.
	 * \param[in] d2
	 *     an array, dimension (§n2)\n
	 *     The second array. §d2 should be in increasing order, i.e., $\{d2}[j]\le\{d2}[j+1]$.
	 *
	 * \param[in] n2     The length of §d2.
	 * \param[in] abstol The absolute tolerance, used as a measure of the error.
	 * \param[in] ulp    Machine precision.
	 * \param[in] unfl   The smallest positive number whose reciprocal does not overflow.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	real dsxt1(int const ijob, real const* const d1, int const n1, real const* const d2,
	           int const n2, real const abstol, real const ulp, real const unfl) const
	{
		real temp1=ZERO, temp2;
		int j = 0;
		for (int i=0; i<n1; i++)
		{
			while (d2[j]<d1[i] && j<n2-1)
			{
				j++;
			}
			if (j==0)
			{
				temp2 = std::fabs(d2[j]-d1[i]);
				if (ijob==2)
				{
					temp2 /= std::max(unfl, abstol+ulp*std::fabs(d1[i]));
				}
			}
			else
			{
				temp2 = std::min(std::fabs(d2[j]-d1[i]), std::fabs(d1[i]-d2[j-1]));
				if (ijob==2)
				{
					temp2 /= std::max(unfl, abstol+ulp*std::fabs(d1[i]));
				}
			}
			temp1 = std::max(temp1, temp2);
		}
		return temp1;
	}

	/*! §ilaenv
	 *
	 * §ilaenv returns problem-dependent parameters for the local environment.
	 * See §ispec for a description of the parameters.\n
	 * In this version, the problem-dependent parameters are contained in the integer array §iparms
	 * in the global struct §claenv and the value with index §ispec is copied to §ilaenv. This
	 * version of §ilaenv is to be used in conjunction with §xlaenv in TESTING and TIMING.
	 * \param[in] ispec
	 *     Specifies the parameter to be returned as the value of §ilaenv.\n
	 *     $\begin{tabular}{rl}
	 *         = 1: & the optimal blocksize; if this value is 1, an unblocked algorithm will give
	 *                the best performance.\\
	 *         = 2: & the minimum block size for which the block routine should be used; if the
	 *                usable block size is less than this value, an unblocked routine should be
	 *                used.\\
	 *         = 3: & the crossover point (in a block routine, for \{n} less than this value, an
	 *                unblocked routine should be used)\\
	 *         = 4: & the number of shifts, used in the nonsymmetric eigenvalue routines\\
	 *         = 5: & the minimum column dimension for blocking to be used; rectangular blocks must
	 *                have dimension at least \{k} by \{m}, where \{k} is given by \{ilaenv}(2,...)
	 *                and \{m} by \{ilaenv}(5,...)\\
	 *         = 6: & the crossover point for the SVD (when reducing an \{m} by \{n} matrix to
	 *                bidiagonal form, if \(\max(m,n)/\min(m,n)\) exceeds this value, a QR
	 *                factorization is used first to reduce the matrix to a triangular form.)\\
	 *         = 7: & the number of processors\\
	 *         = 8: & the crossover point for the multishift QR and QZ methods for nonsymmetric
	 *                eigenvalue problems.\\
	 *         = 9: & maximum size of the subproblems at the bottom of the computation tree in the
	 *                divide-and-conquer algorithm\\
	 *         =10: & IEEE NaN arithmetic can be trusted not to trap\\
	 *         =11: & infinity arithmetic can be trusted not to trap\\
	 *         12\(\le\)ispec\(\le\)16: & \{xhseqr} or one of its subroutines, see \{iparmq} for
	 *                                    detailed explanation \end{tabular}$\n
	 *     Other specifications (up to 100) can be added later.
	 *
	 * \param[in] name The name of the calling subroutine.
	 * \param[in] opts
	 *     The character options to the subroutine §name, concatenated into a single character
	 *     string. For example, §uplo ="U", §trans= "T", and §diag ="N" for a triangular routine
	 *     would be specified as §opts ="UTN".
	 *
	 * \param[in] n1, n2, n3, n4
	 *     Problem dimensions for the subroutine §name; these may not all be required.
	 *
	 * \return
	 *     $\ge 0$: the value of the parameter specified by §ispec \n
	 *     $<   0$: if $-k$, the $k$-th argument had an illegal value.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date November 2017
	 * \remark
	 *     The following conventions have been used when calling §ilaenv from the LAPACK routines:
	 *     1. §opts is a concatenation of all of the character options to subroutine §name, in the
	 *        same order that they appear in the argument list for §name, even if they are not used
	 *        in determining the value of the parameter specified by §ispec.
	 *     2. The problem dimensions §n1, §n2, §n3, §n4 are specified in the order that they appear
	 *        in the argument list for §name. §n1 is used first, §n2 second, and so on, and unused
	 *        problem dimensions are passed a value of -1.
	 *     3. The parameter value returned by §ilaenv is checked for validity in the calling
	 *        subroutine. For example, §ilaenv is used to retrieve the optimal blocksize for
	 *        §strtri as follows:\n
	 *            $\{nb}=\{ilaenv}(1, ``\text{strtri}", \{uplo+diag}, \{N}, -1, -1, -1);$\n
	 *            $\{if} (\{nb}\le 1) \{nb}=\max(1, \{N})$                                       */
	virtual int ilaenv(int const ispec, char const* const name, char const* const opts,
	                   int const n1, int const n2, int const n3, int const n4) const
	{
		if (ispec>=1 && ispec<=5)
		{
			// Return a value from the common block.
			return claenv.iparms[ispec-1];
		}
		else if (ispec==6)
		{
			// Compute SVD crossover point.
			return int(real(std::min(n1, n2))*real(1.6));
		}
		else if (ispec>=7 && ispec<=9)
		{
			// Return a value from the common block.
			return claenv.iparms[ispec-1];
		}
		else if (ispec==10)
		{
			// IEEE NaN arithmetic can be trusted not to trap
			return this->ieeeck(1, ZERO, ONE);
		}
		else if (ispec==11)
		{
			// Infinity arithmetic can be trusted not to trap
			return this->ieeeck(1, ZERO, ONE);
		}
		else if (ispec>=12 && ispec<=16)
		{
			// 12 <= ispec <= 16: xHSEQR or one of its subroutines.
			return claenv.iparms[ispec-1];
			// stdout << "ispec = " << ISPE << " ILAENV =" << claenv.iparms[ispec-1] << std::endl;
			// return iparmq(ispec, name, opts, n1, n2, n3, n4);
		}
		else if (ispec>=17 && ispec<=21)
		{
			// 17 <= ispec <= 21: 2stage eigenvalues SVD routines.
			if (ispec==17)
			{
				return claenv.iparms[0];
			}
			else
			{
				return this->iparam2stage(ispec, name, opts, n1, n2, n3, n4);
			}
		}
		else
		{
			// Invalid value for ispec
			return -1;
		}
		return -1;
	}

	/*! §xerbla
	 *
	 * This is a special version of §xerbla to be used only as part of the test program for testing
	 * error exits from the LAPACK routines. Error messages are printed if $\{info}\ne\{infot}$ or
	 * if $\{srname}\ne\{srnamt}$, where §infot and §srnamt are global variables.
	 * \param[in] srname
	 *     The name of the subroutine calling §xerbla. This name should match the global variable
	 *     §srnamt.
	 *
	 * \param[in] info
	 *     The error return code from the calling subroutine. §info should equal the global
	 *     variable §infot.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016
	 * \remark
	 *     The following variables are passed via the global variables §infoc and §srnamc: \n
	 *     $\begin{tabular}{lll}
	 *        \(\{info}\)   & integer      & Expected integer return code                     \\
	 *        \(\{nout}\)   & std::ostream & Output stream for printing error messages        \\
	 *        \(\{ok}\)     & boolean      & Set to true if \(\{info}=\{infot}\) and
	 *                                      \(\{srname}=\{srnamt}\), otherwise set to false   \\
	 *        \(\{lerr}\)   & boolean      & Set to true, indicating that xerbla was called   \\
	 *        \(\{srnamt}\) & char*        & Expected name of calling subroutine \end{tabular}$  */
	virtual void xerbla(char const* const srname, int const info)
	{
		infoc.lerr = true;
		if (info!=infoc.info)
		{
			if (infoc.info!=0)
			{
				infoc.nout << " *** XERBLA was called from " << srnamc.srnam << " with INFO = "
				           << std::setw(6) << info << " instead of " << std::setw(2) << infoc.info
				           << " ***" << std::endl;
			}
			else
			{
				infoc.nout << " *** On entry to " << srnamc.srnam << " parameter number "
				           << std::setw(6) << info << " had an illegal value ***" << std::endl;
			}
			infoc.ok = false;
		}
		if (srname!=srnamc.srnam)
		{
			infoc.nout << " *** XERBLA was called with SRNAME = " << srname << " instead of "
			           << std::setw(6) << srnamc.srnam << " ***" << std::endl;
			infoc.ok = false;
		}
	}

	/*! §xlaenv
	 *
	 * §xlaenv sets certain machine- and problem-dependent quantities which will later be retrieved
	 * by §ilaenv.
	 * \param[in] ispec
	 *     Specifies the parameter to be set in the global array §iparms. \n
	 *     $\begin{tabular}{rl}
	 *          1: & the optimal blocksize; if this value is 1, an unblocked algorithm will give
	 *               the best performance. \\
	 *          2: & the minimum block size for which the block routine should be used; if the
	 *               usable block size is less than this value, an unblocked routine should be
	 *               used. \\
	 *          3: & the crossover point (in a block routine, for \{N} less than this value, an
	 *               unblocked routine should be used) \\
	 *          4: & the number of shifts, used in the nonsymmetric eigenvalue routines \\
	 *          5: & the minimum column dimension for blocking to be used; rectangular blocks must
	 *               have dimension at least \{k} by \{m}, where \{k} is given by \{ilaenv}(2,...)
	 *               and \{m} by \{ilaenv}(5,...) \\
	 *          6: & the crossover point for the SVD (when reducing an \{m} by \{n} matrix to
	 *               bidiagonal form, if \(\max(\{m},\{n})/\min(\{m},\{n})\) exceeds this value, a
	 *               QR factorization is used first to reduce the matrix to a triangular form) \\
	 *          7: & the number of processors \\
	 *          8: & another crossover point, for the multishift QR and QZ methods for nonsymmetric
	 *               eigenvalue problems. \\
	 *          9: & maximum size of the subproblems at the bottom of the computation tree in the
	 *               divide-and-conquer algorithm (used by \{xgelsd} and \{xgesdd}) \\
	 *         10: & IEEE NaN arithmetic can be trusted not to trap \\
	 *         11: & infinity arithmetic can be trusted not to trap \\
	 *         \(12\le\{ispec}\le 16\): &
	 *               \{xhseqr} or one of its subroutines, see \{iparmq} for detailed explanation
	 *      \end{tabular}$
	 *
	 * \param[in] nvalue The value of the parameter specified by §ispec.
	 * \authors Univ. of Tennessee
	 * \authors Univ. of California Berkeley
	 * \authors Univ. of Colorado Denver
	 * \authors NAG Ltd.
	 * \date December 2016                                                                       */
	void xlaenv(int const ispec, int const nvalue)
	{
		if (ispec>=1 && ispec<=16)
		{
			claenv.iparms[ispec-1] = nvalue;
		}
	}
};

#endif