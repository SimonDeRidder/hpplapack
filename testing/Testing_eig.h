#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cmath>

#include "Blas.h"

#ifndef TESTING_EIG_HEADER
#define TESTING_EIG_HEADER

template<class T>
class Testing_eig
{
public:
    // constants

    static constexpr T ZERO = T(0.0);

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
    static void dchkbl(std::istream& nin, std::ostream& nout)
    {
        const int lda = 20;
        int i, ihi, ihiin, ilo, iloin, info, j, knt, n, ninfo;
        //T anorm, meps;
        T rmax, sfmin, temp, temp2, vmax;
        int* lmax = new int[3];
        T* A = new T[lda*lda];
        T* AIN = new T[lda*lda];
        T* dummy = new T[1];
        T* scale = new T[lda];
        T* scalin = new T[lda];
        lmax[0] = 0;
        lmax[1] = 0;
        lmax[2] = 0;
        ninfo = 0;
        knt = 0;
        rmax = ZERO;
        vmax = ZERO;
        sfmin = dlamch("S");
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
                    strStr >> AIN[i+lda*j];
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
            dgebal("B", n, A, lda, ilo, ihi, scale, info);
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
                    temp = A[i+lda*j]>AIN[i+lda*j]?A[i+lda*j]:AIN[i+lda*j];
                    temp = temp>sfmin?temp:sfmin;
                    temp2 = fabs(A[i+lda*j]-AIN[i+lda*j])/temp;
                    vmax = vmax>temp2?vmax:temp2;
                }
            }
            for (i=0; i<n; i++)
            {
                temp = scale[i]>scalin[i]?scale[i]:scalin[i];
                temp = temp>sfmin?temp:sfmin;
                temp2 = fabs(scale[i]-scalin[i])/temp;
                vmax = vmax>temp2?vmax:temp2;
            }
            if (vmax>rmax)
            {
                lmax[2] = knt;
                rmax = vmax;
            }
            nin >> n;
        }
        nout << " .. test output of DGEBAL .. \n";
        nout << " value of largest test error            = " << std::setprecision(4) << std::setw(12);
        nout << rmax << '\n';
        nout << " example number where info is not zero  = " << std::setw(4) << lmax[0] << '\n';
        nout << " example number where ILO or IHI wrong  = " << std::setw(4) << lmax[1] << '\n';
        nout << " example number having largest error    = " << std::setw(4) << lmax[2] << '\n';
        nout << " number of examples where info is not 0 = " << std::setw(4) << ninfo << '\n';
        nout << " total number of examples tested        = " << std::setw(4) << knt << std::endl;
        delete[] lmax;
        delete[] A;
        delete[] AIN;
        delete[] dummy;
        delete[] scale;
        delete[] scalin;
    }

    // TODO: xlaenv, ilaenv
}

#endif