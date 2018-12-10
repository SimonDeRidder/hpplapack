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
    void alasum(char const* type, std::ostream& nout, int nfail, int nrun, int nerrs)
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
            nout << "\n All tests for " << typecopy << " routines passed the threshold ( "
                 << std::setw(6) << nrun << " tests run)\n";
        }
        if (nerrs>0)
        {
            nout << "      " << std::setw(6) << nerrs << " error messages recorded\n";
        }
        nout.flush();
    }

    /*! §dbdt03
     *
     * §dbdt03 reconstructs a bidiagonal matrix $B$ from its SVD:\n
     *     $S = U^T B V$\n
     * where $U$ and $V$ are orthogonal matrices and $S$ is diagonal.\n
     * The test ratio to test the singular value decomposition is\n
     *     $\{resid}=\frac{\|B - U S V^T\|}{\{n} \|B\| \epsilon}$\n
     * where $\epsilon$ is the machine precision.
     * \param[in] uplo
     *     Specifies whether the matrix $B$ is upper or lower bidiagonal.\n
     *     = 'U': Upper bidiagonal\n
     *     = 'L': Lower bidiagonal
     *
     * \param[in] n The order of the matrix $B$.
     * \param[in] kd
     *     The bandwidth of the bidiagonal matrix $B$. If $\{kd}=1$, the matrix $B$ is bidiagonal,
     *     and if $\{kd}=0$, $B$ is diagonal and §E is not referenced. If §kd is greater than 1, it
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
     * \param[out] resid The test ratio: $\frac{\|B - U S V^T\|}{\{n} \|A\| \epsilon}$
     * \authors Univ.of Tennessee
     * \authors Univ.of California Berkeley
     * \authors Univ.of Colorado Denver
     * \authors NAG Ltd.
     * \date December 2016                                                                       */
    void dbdt03(char const* uplo, int n, int kd, real const* d, real const* e, real const* U,
                int ldu, real const* s, real const* Vt, int ldvt, real* work, real& resid)
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
                    Blas<real>::dgemv("No transpose", n, n, -ONE, U, ldu, &work[n], 1, ZERO, work, 1);
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
                    Blas<real>::dgemv("No transpose", n, n, -ONE, U, ldu, &work[n], 1, ZERO, work, 1);
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
    void dbdt04(char const* uplo, int n, real const* d, real const* e, real const* s,
                       int ns, real const* U, int ldu, real const* Vt, int ldvt, real* work,
                       real& resid)
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
    void dchkbl(std::istream& nin, std::ostream& nout)
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

    // TODO: xlaenv, ilaenv, xerbla
};

#endif