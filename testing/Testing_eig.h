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

#include "Blas.h"
#include "Lapack_dyn.h"

template<class real>
class Testing_eig : public Lapack_dyn<real>
{
public:
    virtual ~Testing_eig(){}

    // constants

    const real ZERO = real(0.0);

    // LAPACK TESTING EIG (alphabetically)

    /* dchkbl tests dgebal, a routine for balancing a general real matrix and isolating some of
     * its eigenvalues.
     * Parameters: nin: inputs tream of test examples.
     *             nout: output stream.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016                                                                       */
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