#ifndef TESTING_LIN_HEADER
#define TESTING_LIN_HEADER

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>
#include <algorithm>

#include "Blas.hpp"
#include "Lapack_dyn.hpp"
#include "Testing_matgen.hpp"

template<class real>
class Testing_lin : public Lapack_dyn<real>
{
public:
    // constants

    const real ZERO = real(0.0);
    const real ONE  = real(1.0);

    // "Common" variables

    struct infostruct
    {
        int info;
        std::ostream& iounit;
        bool ok;
        bool lerr;
    } infoc = {0, std::cout, true, false};

    struct srnamstruct
    {
        char srnam[32];
    } srnamc;

    struct laenvstruct
    {
        int iparms[100];
    } claenv;

    // matgen instance

    Testing_matgen<real> MatGen;

    // LAPACK TESTING LIN (alphabetically)

    /* alahd prints header information for the different test paths.
     * Parameters: iounit: The output stream to which the header information should be printed.
     *             path: char[3+]
     *                   The name of the path for which the header information is to be printed.
     *                   Current paths are:
     *                   _GE:  General matrices
     *                   _GB:  General band
     *                   _GT:  General Tridiagonal
     *                   _PO:  Symmetric or Hermitian positive definite
     *                   _PS:  Symmetric or Hermitian positive semi-definite
     *                   _PP:  Symmetric or Hermitian positive definite packed
     *                   _PB:  Symmetric or Hermitian positive definite band
     *                   _PT:  Symmetric or Hermitian positive definite tridiagonal
     *                   _SY:  Symmetric indefinite, with partial (Bunch-Kaufman) pivoting
     *                   _SR:  Symmetric indefinite, with rook (bounded Bunch-Kaufman) pivoting
     *                   _SK:  Symmetric indefinite, with rook (bounded Bunch-Kaufman) pivoting
     *                         (new storage format for factors: L and diagonal of D is stored in A,
     *                         subdiagonal of D is stored in E)
     *                   _SP:  Symmetric indefinite packed, with partial (Bunch-Kaufman) pivoting
     *                   _HA:  (complex) Hermitian , with Aasen Algorithm
     *                   _HE:  (complex) Hermitian indefinite, with partial (Bunch-Kaufman) pivoting
     *                   _HR:  (complex) Hermitian indefinite,
     *                         with rook (bounded Bunch-Kaufman) pivoting
     *                   _HK:  (complex) Hermitian indefinite,
     *                         with rook (bounded Bunch-Kaufman) pivoting
     *                         (new storage format for factors: L and diagonal of D is stored in A,
     *                         subdiagonal of D is stored in E)
     *                   _HP:  (complex) Hermitian indefinite packed,
     *                         with partial (Bunch-Kaufman) pivoting
     *                   _TR:  Triangular
     *                   _TP:  Triangular packed
     *                   _TB:  Triangular band
     *                   _QR:  QR (general matrices)
     *                   _LQ:  LQ (general matrices)
     *                   _QL:  QL (general matrices)
     *                   _RQ:  RQ (general matrices)
     *                   _QP:  QR with column pivoting
     *                   _TZ:  Trapezoidal
     *                   _LS:  Least Squares driver routines
     *                   _LU:  LU variants
     *                   _CH:  Cholesky variants
     *                   _QS:  QR variants
     *                   _QT:  QRT (general matrices)
     *                   _QX:  QRT (triangular-pentagonal matrices)
     *                   The first character must be one of S, D, C, or Z (C or Z only if complex).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void alahd(std::ostream& iounit, char const* path)
    {
        if (!iounit.good())
        {
            return;
        }
        char pathcopy[4], c1, c3, p2[2], sym[10];
        strncpy(pathcopy, path, 3);
        pathcopy[3] = '\0';
        c1 = toupper(path[0]);
        c3 = toupper(path[2]);
        p2[0] = toupper(path[1]);
        p2[1] = toupper(path[2]);
        bool sord = (c1=='S' || c1=='D');
        if (!(sord || c1=='C' || c1=='Z'))
        {
            return;
        }
        char const* str9892 = " indefinite matrices, \"rook\" (bounded Bunch-Kaufman) pivoting";
        char const* str9926 = ": Largest 2-Norm of 2-by-2 pivots\n"
                              "             - ( ( 1 + ALPHA ) / ( 1 - ALPHA ) ) + THRESH";
        char const* str9927 = ": ABS( Largest element in L )\n"
                              "             - ( 1 / ( 1 - ALPHA ) ) + THRESH";
        char const* str9928 = "       where ALPHA = ( 1 + SQRT( 17 ) ) / 8";
        char const* str9935 = ": norm( B - A * X )   / ( max(M,N) * norm(A) * norm(X) * EPS )";
        char const* str9938 = ": norm( I - Q''*Q )      / ( M * EPS )";
        char const* str9940 = ": norm(svd(A) - svd(R)) / ( M * norm(svd(R)) * EPS )";
        char const* str9941 = ": norm( C*Q' - C*Q' )/ ( ";
        char const* str9942 = ": norm( Q'*C - Q'*C )/ ( ";
        char const* str9943 = ": norm( C*Q - C*Q )  / ( ";
        char const* str9944 = ": norm( Q*C - Q*C )  / ( ";
        char const* str9945 = ": norm( I - Q*Q' )   / ( N * EPS )";
        char const* str9946 = ": norm( I - Q'*Q )   / ( M * EPS )";
        char const* str9951 = ": norm( s*b - A*x )  / ( norm(A) * norm(x) * EPS )";
        char const* str9953 = ": norm( U*D*U' - A ) / ( N * norm(A) * EPS ), or\n"
                         "       norm( L*D*L' - A ) / ( N * norm(A) * EPS )";
        char const* str9954 = ": norm( U' * U - A ) / ( N * norm(A) * EPS ), or\n"
                              "       norm( L * L' - A ) / ( N * norm(A) * EPS )";
        char const* str9955 = ": RCOND * CNDNUM - 1.0";
        char const* str9956 = ": (backward error)   / EPS";
        char const* str9957 = ": norm( X - XACT )   / ( norm(XACT) * (error bound) )";
        char const* str9958 = ": norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS ), refined";
        char const* str9959 = ": norm( X - XACT )   / ( norm(XACT) * CNDNUM * EPS )";
        char const* str9960 = ": norm( B - A * X )  / ( norm(A) * norm(X) * EPS )";
        char const* str9961 = ": norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )";
        char const* str9962 = ": norm( L * U - A )  / ( N * norm(A) * EPS )";
        char const* str9970 =
                    "    1. Diagonal                        5. Random, CNDNUM = sqrt(0.1/EPS)\n"
                    "    2. Upper triangular                6. Random, CNDNUM = 0.1/EPS\n"
                    "    3. Lower triangular                7. Scaled near underflow\n"
                    "    4. Random, CNDNUM = 2              8. Scaled near overflow";
        char const* str9971 =
                    "    1. Diagonal                        7. Random, CNDNUM = sqrt(0.1/EPS)\n"
                    "    2. Random, CNDNUM = 2              8. Random, CNDNUM = 0.1/EPS\n"
                    "    3. First row and column zero       9. Scaled near underflow\n"
                    "    4. Last row and column zero       10. Scaled near overflow\n"
                    "    5. Middle row and column zero     11. Block diagonal matrix\n"
                    "    6. Last n/2 rows and columns zero";
        char const* str9972 =
                    "    1. Diagonal                        6. Last n/2 rows and columns zero\n"
                    "    2. Random, CNDNUM = 2              7. Random, CNDNUM = sqrt(0.1/EPS)\n"
                    "    3. First row and column zero       8. Random, CNDNUM = 0.1/EPS\n"
                    "    4. Last row and column zero        9. Scaled near underflow\n"
                    "    5. Middle row and column zero     10. Scaled near overflow";
        char const* str9979 =
                    "    1. Diagonal                        7. Last n/2 columns zero\n"
                    "    2. Upper triangular                8. Random, CNDNUM = sqrt(0.1/EPS)\n"
                    "    3. Lower triangular                9. Random, CNDNUM = 0.1/EPS\n"
                    "    4. Random, CNDNUM = 2             10. Scaled near underflow\n"
                    "    5. First column zero              11. Scaled near overflow\n"
                    "    6. Last column zero";
        char const* str9987 = " factorization of general matrices";
        char const* str9991 = " indefinite packed matrices, partial (Bunch-Kaufman) pivoting";
        char const* str9992 = " indefinite matrices, partial (Bunch-Kaufman) pivoting";
        char const* str9995 = " positive definite packed matrices";
        if (std::strncmp(p2, "GE", 2)==0)
        {
            // GE: General dense
            iounit << "\n " << pathcopy << ":  General dense matrices\n";
            iounit << " Matrix types:\n";
            iounit << str9979 << '\n';
            iounit << " Test ratios:\n";
            iounit << "    1" << str9962 << '\n';
            iounit << "    2" << str9961 << '\n';
            iounit << "    3" << str9960 << '\n';
            iounit << "    4" << str9959 << '\n';
            iounit << "    5" << str9958 << '\n';
            iounit << "    6" << str9957 << '\n';
            iounit << "    7" << str9956 << '\n';
            iounit << "    8" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "GB", 2)==0)
        {
            iounit << "\n " << pathcopy << ":  General band matrices\n";
            iounit << " Matrix types:\n";
            iounit << "    1. Random, CNDNUM = 2              5. Random, CNDNUM = sqrt(0.1/EPS)\n"
                      "    2. First column zero               6. Random, CNDNUM = .01/EPS\n"
                      "    3. Last column zero                7. Scaled near underflow\n"
                      "    4. Last n/2 columns zero           8. Scaled near overflow\n";
            iounit << " Test ratios:\n";
            iounit << "    1" << str9962 << '\n';
            iounit << "    2" << str9960 << '\n';
            iounit << "    3" << str9959 << '\n';
            iounit << "    4" << str9958 << '\n';
            iounit << "    5" << str9957 << '\n';
            iounit << "    6" << str9956 << '\n';
            iounit << "    7" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "GT", 2)==0)
        {
            // GT: General tridiagonal
            iounit << "\n " << pathcopy << ":  General tridiagonal\n";
            iounit << " Matrix types (1-6 have specified condition numbers):\n"
                      "    1. Diagonal                        7. Random, unspecified CNDNUM\n"
                      "    2. Random, CNDNUM = 2              8. First column zero\n"
                      "    3. Random, CNDNUM = sqrt(0.1/EPS)  9. Last column zero\n"
                      "    4. Random, CNDNUM = 0.1/EPS       10. Last n/2 columns zero\n"
                      "    5. Scaled near underflow          11. Scaled near underflow\n"
                      "    6. Scaled near overflow           12. Scaled near overflow\n";
            iounit << " Test ratios:\n";
            iounit << "    1" << str9962 << '\n';
            iounit << "    2" << str9960 << '\n';
            iounit << "    3" << str9959 << '\n';
            iounit << "    4" << str9958 << '\n';
            iounit << "    5" << str9957 << '\n';
            iounit << "    6" << str9956 << '\n';
            iounit << "    7" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "PO", 2)==0 || std::strncmp(p2, "PP", 2)==0)
        {
            // PO: Positive definite full
            // PP: Positive definite packed
            if(sord)
            {
                std::strncpy(sym, "Symmetric", 10);
            }
            else
            {
                std::strncpy(sym, "Hermitian", 10);
            }
            if (c3=='O')
            {
                iounit << "\n " << pathcopy << ":  " << sym << " positive definite matrices\n";
            }
            else
            {
                iounit << "\n " << pathcopy << ":  " << sym << str9995 << '\n';
            }
            iounit << " Matrix types:\n";
            iounit << "    1. Diagonal                        6. Random, CNDNUM = sqrt(0.1/EPS)\n"
                      "    2. Random, CNDNUM = 2              7. Random, CNDNUM = 0.1/EPS\n"
                      "   *3. First row and column zero       8. Scaled near underflow\n"
                      "   *4. Last row and column zero        9. Scaled near overflow\n"
                      "   *5. Middle row and column zero\n"
                      "   (* - tests error exits from " << pathcopy;
                iounit << "TRF, no test ratios are computed)\n";
            iounit << " Test ratios:\n";
            iounit << "    1" << str9954 << '\n';
            iounit << "    2" << str9961 << '\n';
            iounit << "    3" << str9960 << '\n';
            iounit << "    4" << str9959 << '\n';
            iounit << "    5" << str9958 << '\n';
            iounit << "    6" << str9957 << '\n';
            iounit << "    7" << str9956 << '\n';
            iounit << "    8" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "PS", 2)==0)
        {
            // PS: Positive semi-definite full
            if (sord)
            {
                std::strncpy(sym, "Symmetric", 10);
            }
            else
            {
                std::strncpy(sym, "Hermitian", 10);
            }
            char eigcnm[5];
            if (c1=='S' || c1=='C')
            {
                std::strncpy(eigcnm, "1E04", 5);
            }
            else
            {
                std::strncpy(eigcnm, "1D12", 5);
            }
            iounit << "\n " << pathcopy << ":  " << sym << str9995 << '\n';
            iounit << " Matrix types:\n";
            iounit << "    1. Diagonal\n    2. Random, CNDNUM = 2\n"
                      "   *3. Nonzero eigenvalues of: D(1:RANK-1)=1 and D(RANK) = 1.0/";
                iounit << eigcnm << "\n"
                      "   *4. Nonzero eigenvalues of: D(1)=1 and  D(2:RANK) = 1.0/";
                iounit << eigcnm << "\n   *5. Nonzero eigenvalues of: D(I) = " << eigcnm;
                iounit << "**(-(I-1)/(RANK-1))  I=1:RANK\n    6. Random, CNDNUM = sqrt(0.1/EPS)\n"
                          "    7. Random, CNDNUM = 0.1/EPS\n    8. Scaled near underflow\n"
                          "    9. Scaled near overflow\n   (* - Semi-definite tests )\n";
            iounit << " Difference:\n";
            iounit << "   RANK minus computed rank, returned by " << c1 << "PSTRF\n";
            iounit << " Test ratio:\n";
            iounit << "   norm( P * U' * U * P' - A ) / ( N * norm(A) * EPS ) , or\n"
                      "   norm( P * L * L' * P' - A ) / ( N * norm(A) * EPS )\n";
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "PB", 2)==0)
        {
            // PB: Positive definite band
            if (sord)
            {
                iounit << "\n " << pathcopy << ":  Symmetric positive definite band matrices\n";
            }
            else
            {
                iounit << "\n " << pathcopy << ":  Hermitian positive definite band matrices\n";
            }
            iounit << "    1. Random, CNDNUM = 2              5. Random, CNDNUM = sqrt(0.1/EPS)\n"
                      "   *2. First row and column zero       6. Random, CNDNUM = 0.1/EPS\n"
                      "   *3. Last row and column zero        7. Scaled near underflow\n"
                      "   *4. Middle row and column zero      8. Scaled near overflow\n"
                      "   (* - tests error exits from " << pathcopy;
                iounit << "TRF, no test ratios are computed)\n";
            iounit << " Test ratios:\n";
            iounit << "    1" << str9954 << '\n';
            iounit << "    2" << str9960 << '\n';
            iounit << "    3" << str9959 << '\n';
            iounit << "    4" << str9958 << '\n';
            iounit << "    5" << str9957 << '\n';
            iounit << "    6" << str9956 << '\n';
            iounit << "    7" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "PT", 2)==0)
        {
            // PT: Positive definite tridiagonal
            if (sord)
            {
                iounit << "\n " << pathcopy << ":  Symmetric positive definite tridiagonal";
            }
            else
            {
                iounit << "\n " << pathcopy << ":  Hermitian positive definite tridiagonal";
            }
            iounit << " Matrix types (1-6 have specified condition numbers):\n"
                      "    1. Diagonal                        7. Random, unspecified CNDNUM\n"
                      "    2. Random, CNDNUM = 2              8. First row and column zero\n"
                      "    3. Random, CNDNUM = sqrt(0.1/EPS)  9. Last row and column zero\n"
                      "    4. Random, CNDNUM = 0.1/EPS       10. Middle row and column zero\n"
                      "    5. Scaled near underflow          11. Scaled near underflow\n"
                      "    6. Scaled near overflow           12. Scaled near overflow\n";
            iounit << " Test ratios:\n";
            iounit << "    1: norm( U'*D*U - A ) / ( N * norm(A) * EPS ), or\n"
                      "       norm( L*D*L' - A ) / ( N * norm(A) * EPS )\n";
            iounit << "    2" << str9960 << '\n';
            iounit << "    3" << str9959 << '\n';
            iounit << "    4" << str9958 << '\n';
            iounit << "    5" << str9957 << '\n';
            iounit << "    6" << str9956 << '\n';
            iounit << "    7" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "SY", 2)==0)
        {
            // SY: Symmetric indefinite full, with partial (Bunch-Kaufman) pivoting algorithm
            if (c3=='Y')
            {
                iounit << "\n " << pathcopy << ":  Symmetric" << str9992 << '\n';
            }
            else
            {
                iounit << "\n " << pathcopy << ":  Symmetric" << str9991 << '\n';
            }
            iounit << " Matrix types:\n";
            if (sord)
            {
                iounit << str9972 << '\n';
            }
            else
            {
                iounit << str9971 << '\n';
            }
            iounit << " Test ratios:\n";
            iounit << "    1" << str9953 << '\n';
            iounit << "    2" << str9961 << '\n';
            iounit << "    3" << str9960 << '\n';
            iounit << "    4" << str9960 << '\n';
            iounit << "    5" << str9959 << '\n';
            iounit << "    6" << str9958 << '\n';
            iounit << "    7" << str9956 << '\n';
            iounit << "    8" << str9957 << '\n';
            iounit << "    9" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "SR", 2)==0 || std::strncmp(p2, "SK", 2)==0)
        {
            // SR: Symmetric indefinite full, with rook (bounded Bunch-Kaufman) pivoting algorithm
            // SK: Symmetric indefinite full, with rook (bounded Bunch-Kaufman) pivoting algorithm,
            // (new storage format for factors:
            //                 L and diagonal of D is stored in A, subdiagonal of D is stored in E)
            iounit << "\n " << pathcopy << ":  Symmetric" << str9892 << '\n';
            iounit << " Matrix types:\n";
            if (sord)
            {
                iounit << str9972 << '\n';
            }
            else
            {
                iounit << str9971 << '\n';
            }
            iounit << " Test ratios:\n";
            iounit << "    1" << str9953 << '\n';
            iounit << "    2" << str9961 << '\n';
            iounit << "    3" << str9927 << '\n';
            iounit << str9928 << '\n';
            iounit << "    4" << str9926 << '\n';
            iounit << str9928 << '\n';
            iounit << "    5" << str9960 << '\n';
            iounit << "    6" << str9959 << '\n';
            iounit << "    7" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "SP", 2)==0)
        {
            // SP: Symmetric indefinite packed, with partial (Bunch-Kaufman) pivoting algorithm
            if (c3=='Y')
            {
                iounit << "\n " << pathcopy << ":  Symmetric" << str9992 << '\n';
            }
            else
            {
                iounit << "\n " << pathcopy << ":  Symmetric" << str9991 << '\n';
            }
            iounit << " Matrix types:\n";
            if (sord)
            {
                iounit << str9972 << '\n';
            }
            else
            {
                iounit << str9971 << '\n';
            }
            iounit << " Test ratios:\n";
            iounit << "    1" << str9953 << '\n';
            iounit << "    2" << str9961 << '\n';
            iounit << "    3" << str9960 << '\n';
            iounit << "    4" << str9959 << '\n';
            iounit << "    5" << str9958 << '\n';
            iounit << "    6" << str9956 << '\n';
            iounit << "    7" << str9957 << '\n';
            iounit << "    8" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "HA", 2)==0)
        {
            // HA: Hermitian, with Assen Algorithm
            iounit << "\n " << pathcopy << ":  Hermitian" << str9992 << '\n';
            iounit << " Matrix types:\n";
            iounit << str9972 << '\n';
            iounit << " Test ratios:\n";
            iounit << "    1" << str9953 << '\n';
            iounit << "    2" << str9961 << '\n';
            iounit << "    3" << str9960 << '\n';
            iounit << "    4" << str9960 << '\n';
            iounit << "    5" << str9959 << '\n';
            iounit << "    6" << str9958 << '\n';
            iounit << "    7" << str9956 << '\n';
            iounit << "    8" << str9957 << '\n';
            iounit << "    9" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "HE", 2)==0)
        {
            // HE: Hermitian indefinite full, with partial (Bunch-Kaufman) pivoting algorithm
            iounit << "\n " << pathcopy << ":  Hermitian" << str9992 << '\n';
            iounit << " Matrix types:\n";
            iounit << str9972 << '\n';
            iounit << " Test ratios:\n";
            iounit << "    1" << str9953 << '\n';
            iounit << "    2" << str9961 << '\n';
            iounit << "    3" << str9960 << '\n';
            iounit << "    4" << str9960 << '\n';
            iounit << "    5" << str9959 << '\n';
            iounit << "    6" << str9958 << '\n';
            iounit << "    7" << str9956 << '\n';
            iounit << "    8" << str9957 << '\n';
            iounit << "    9" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "HR", 2)==0)
        {
            // HR: Hermitian indefinite full, with rook (bounded Bunch-Kaufman) pivoting algorithm
            // (new storage format for factors:
            //  L and diagonal of D is stored in A, subdiagonal of D is stored in E)
            iounit << "\n " << pathcopy << ":  Hermitian" << str9892 << '\n';
            iounit << " Matrix types:\n";
            iounit << str9972 << '\n';
            iounit << " Test ratios:\n";
            iounit << "    1" << str9953 << '\n';
            iounit << "    2" << str9961 << '\n';
            iounit << "    3" << str9927 << '\n';
            iounit << str9928 << '\n';
            iounit << "    4" << str9926 << '\n';
            iounit << str9928 << '\n';
            iounit << "    5" << str9960 << '\n';
            iounit << "    6" << str9959 << '\n';
            iounit << "    7" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "HP", 2)==0)
        {
            // HP: Hermitian indefinite packed, with partial (Bunch-Kaufman) pivoting algorithm
            if (c3=='E')
            {
                iounit << "\n " << pathcopy << ":  Hermitian" << str9992 << '\n';
            }
            else
            {
                iounit << "\n " << pathcopy << ":  Hermitian" << str9991 << '\n';
            }
            iounit << " Matrix types:\n";
            iounit << str9972 << '\n';
            iounit << " Test ratios:\n";
            iounit << "    1" << str9953 << '\n';
            iounit << "    2" << str9961 << '\n';
            iounit << "    3" << str9960 << '\n';
            iounit << "    4" << str9959 << '\n';
            iounit << "    5" << str9958 << '\n';
            iounit << "    6" << str9956 << '\n';
            iounit << "    7" << str9957 << '\n';
            iounit << "    8" << str9955 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "TR", 2)==0 || std::strncmp(p2, "TP", 2)==0)
        {
            // TR: Triangular full
            // TP: Triangular packed
            if (c3=='R')
            {
                iounit << "\n " << pathcopy << ":  Triangular matrices\n";
            }
            else
            {
                iounit << "\n " << pathcopy << ":  Triangular packed matrices\n";
            }
            iounit << " Matrix types for " << pathcopy << " routines:\n"
                      "    1. Diagonal                        6. Scaled near overflow\n"
                      "    2. Random, CNDNUM = 2              7. Identity\n"
                      "    3. Random, CNDNUM = sqrt(0.1/EPS)  8. Unit triangular, CNDNUM = 2\n"
                      "    4. Random, CNDNUM = 0.1/EPS        9. Unit, CNDNUM = sqrt(0.1/EPS)\n"
                      "    5. Scaled near underflow          10. Unit, CNDNUM = 0.1/EPS\n";
            iounit << " Special types for testing " << pathcopy[0] << ":\n"
                      "   11. Matrix elements are O(1), large right hand side\n"
                      "   12. First diagonal causes overflow, offdiagonal column norms < 1\n"
                      "   13. First diagonal causes overflow, offdiagonal column norms > 1\n"
                      "   14. Growth factor underflows, solution does not overflow\n"
                      "   15. Small diagonal causes gradual overflow\n"
                      "   16. One zero diagonal element\n"
                      "   17. Large offdiagonals cause overflow when adding a column\n"
                      "   18. Unit triangular with large right hand side\n";
            iounit << " Test ratios:\n";
            iounit << "    1" << str9961 << '\n';
            iounit << "    2" << str9960 << '\n';
            iounit << "    3" << str9959 << '\n';
            iounit << "    4" << str9958 << '\n';
            iounit << "    5" << str9957 << '\n';
            iounit << "    6" << str9956 << '\n';
            iounit << "    7" << str9955 << '\n';
            iounit << " Test ratio for " << pathcopy[0] << ":\n    8" << str9951 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "TB", 2)==0)
        {
            // TB: Triangular band
            iounit << "\n " << pathcopy << ":  Triangular band matrices\n";
            iounit << " Matrix types for " << pathcopy << " routines:\n"
                      "    1. Random, CNDNUM = 2              6. Identity\n"
                      "    2. Random, CNDNUM = sqrt(0.1/EPS)  7. Unit triangular, CNDNUM = 2\n"
                      "    3. Random, CNDNUM = 0.1/EPS        8. Unit, CNDNUM = sqrt(0.1/EPS)\n"
                      "    4. Scaled near underflow           9. Unit, CNDNUM = 0.1/EPS\n"
                      "    5. Scaled near overflow\n";
            iounit << " Special types for testing " << pathcopy[0] << ":\n"
                      "   10. Matrix elements are O(1), large right hand side\n"
                      "   11. First diagonal causes overflow, offdiagonal column norms < 1\n"
                      "   12. First diagonal causes overflow, offdiagonal column norms > 1\n"
                      "   13. Growth factor underflows, solution does not overflow\n"
                      "   14. Small diagonal causes gradual overflow\n"
                      "   15. One zero diagonal element\n"
                      "   16. Large offdiagonals cause overflow when adding a column\n"
                      "   17. Unit triangular with large right hand side\n";
            iounit << " Test ratios:\n";
            iounit << "    1" << str9960 << '\n';
            iounit << "    2" << str9959 << '\n';
            iounit << "    3" << str9958 << '\n';
            iounit << "    4" << str9957 << '\n';
            iounit << "    5" << str9956 << '\n';
            iounit << "    6" << str9955 << '\n';
            iounit << " Test ratio for " << pathcopy[0] << ":\n    7" << str9951 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "QR", 2)==0)
        {
            // QR decomposition of rectangular matrices
            iounit << "\n " << pathcopy << ":  QR" << str9987 << '\n';
            iounit << " Matrix types:\n";
            iounit << str9970 << '\n';
            iounit << " Test ratios:\n";
            iounit << "    1: norm( R - Q' * A ) / ( M * norm(A) * EPS )\n";
            iounit << "    8: norm( R - Q' * A ) / ( M * norm(A) * EPS )       [RFPG]\n";
            iounit << "    2" << str9946 << '\n';
            iounit << "    3" << str9944 << "M * norm(C) * EPS )\n";
            iounit << "    4" << str9943 << "M * norm(C) * EPS )\n";
            iounit << "    5" << str9942 << "M * norm(C) * EPS )\n";
            iounit << "    6" << str9941 << "M * norm(C) * EPS )\n";
            iounit << "    7" << str9960 << '\n';
            iounit << "    9" << ": diagonal is not non-negative\n";
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "LQ", 2)==0)
        {
            // LQ decomposition of rectangular matrices
            iounit << "\n " << pathcopy << ":  LQ" << str9987 << '\n';
            iounit << " Matrix types:\n";
            iounit << str9970 << '\n';
            iounit << " Test ratios:\n";
            iounit << "    1: norm( L - A * Q' ) / ( N * norm(A) * EPS )\n";
            iounit << "    2" << str9945 << '\n';
            iounit << "    3" << str9944 << "M * norm(C) * EPS )\n";
            iounit << "    4" << str9943 << "M * norm(C) * EPS )\n";
            iounit << "    5" << str9942 << "M * norm(C) * EPS )\n";
            iounit << "    6" << str9941 << "M * norm(C) * EPS )\n";
            iounit << "    7" << str9960 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "QL", 2)==0)
        {
            // QL decomposition of rectangular matrices
            iounit << "\n " << pathcopy << ":  QL" << str9987 << '\n';
            iounit << " Matrix types:\n";
            iounit << str9970 << '\n';
            iounit << " Test ratios:\n";
            iounit << "    1: norm( L - Q' * A ) / ( M * norm(A) * EPS )\n";
            iounit << "    2" << str9946 << '\n';
            iounit << "    3" << str9944 << "M * norm(C) * EPS )\n";
            iounit << "    4" << str9943 << "M * norm(C) * EPS )\n";
            iounit << "    5" << str9942 << "M * norm(C) * EPS )\n";
            iounit << "    6" << str9941 << "M * norm(C) * EPS )\n";
            iounit << "    7" << str9960 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "RQ", 2)==0)
        {
            // RQ decomposition of rectangular matrices
            iounit << "\n " << pathcopy << ":  RQ" << str9987 << '\n';
            iounit << " Matrix types:\n";
            iounit << str9970 << '\n';
            iounit << " Test ratios:\n";
            iounit << "    1" << ": norm( R - A * Q' ) / ( N * norm(A) * EPS )\n";
            iounit << "    2" << str9945 << '\n';
            iounit << "    3" << str9944 << "N * norm(C) * EPS )" << '\n';
            iounit << "    4" << str9943 << "N * norm(C) * EPS )\n";
            iounit << "    5" << str9942 << "N * norm(C) * EPS )\n";
            iounit << "    6" << str9941 << "N * norm(C) * EPS )\n";
            iounit << "    7" << str9960 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "QP", 2)==0)
        {
            // QR decomposition with column pivoting
            iounit << "\n " << pathcopy << ":  QR factorization with column pivoting\n";
            iounit << " Matrix types (2-6 have condition 1/EPS):\n"
                      "    1. Zero matrix                     4. First n/2 columns fixed\n"
                      "    2. One small eigenvalue            5. Last n/2 columns fixed\n"
                      "    3. Geometric distribution          6. Every second column fixed\n";
            iounit << " Test ratios:\n";
            iounit << "    1" << str9940 << '\n';
            iounit << "    2" << ": norm( A*P - Q*R )     / ( M * norm(A) * EPS )\n";
            iounit << "    3" << str9938 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "TZ", 2)==0)
        {
            // TZ:  Trapezoidal
            iounit << "\n " << pathcopy << ":  RQ factorization of trapezoidal matrix\n";
            iounit << " Matrix types (2-3 have condition 1/EPS):\n"
                      "    1. Zero matrix\n"
                      "    2. One small eigenvalue\n"
                      "    3. Geometric distribution\n";
            iounit << " Test ratios (1-3: " << c1 << "TZRZF):\n";
            iounit << " Test ratios:\n";
            iounit << "    1" << str9940 << '\n';
            iounit << "    2" << ": norm( A - R*Q )       / ( M * norm(A) * EPS )\n";
            iounit << "    3" << str9938 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "LS", 2)==0)
        {
            // LS:  Least Squares driver routines for LS, LSD, LSS, LSX and LSY.
            iounit << "\n " << pathcopy << ":  Least squares driver routines\n";
            iounit << " Matrix types (1-3: full rank, 4-6: rank deficient):\n"
                      "    1 and 4. Normal scaling\n"
                      "    2 and 5. Scaled near overflow\n"
                      "    3 and 6. Scaled near underflow\n";
            iounit << " Test ratios:\n    (1-2: " << c1 << "GELS, 3-6: " << c1 << "GELSY, 7-10: ";
                iounit << c1 << "GELSS, 11-14: " << c1 << "GELSD, 15-16: " << c1 << "GETSLS)\n";
            iounit << "    1" << str9935 << '\n';
            iounit << "    2: norm( (A*X-B)' *A ) / ( max(M,N,NRHS) * norm(A) * norm(B) * EPS )\n"
                      "       if TRANS='N' and M>=N or TRANS='T' and M<N, otherwise check if X\n"
                      "       is in the row space of A or A' (overdetermined case)\n";
            iounit << "    3: norm(svd(A)-svd(R)) / ( min(M,N) * norm(svd(R)) * EPS )\n";
            iounit << "    4" << str9935 << '\n';
            iounit << "    5: norm( (A*X-B)' *A ) / ( max(M,N,NRHS) * norm(A) * norm(B) * EPS )\n";
            iounit << "    6: Check if X is in the row space of A or A'\n";
            iounit << "    7-10: same as 3-6    11-14: same as 3-6\n";
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "LU", 2)==0)
        {
            // LU factorization variants
            iounit << "\n " << pathcopy << ":  LU factorization variants\n";
            iounit << " Matrix types:\n";
            iounit << str9979 << '\n';
            iounit << " Test ratio:\n";
            iounit << "    1" << str9962 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "CH", 2)==0)
        {
            // Cholesky factorization variants
            iounit << "\n " << pathcopy << ":  Cholesky factorization variants\n";
            iounit << " Matrix types:\n";
            iounit << "    1. Diagonal                        6. Random, CNDNUM = sqrt(0.1/EPS)\n"
                      "    2. Random, CNDNUM = 2              7. Random, CNDNUM = 0.1/EPS\n"
                      "   *3. First row and column zero       8. Scaled near underflow\n"
                      "   *4. Last row and column zero        9. Scaled near overflow\n"
                      "   *5. Middle row and column zero\n"
                      "   (* - tests error exits, no test ratios are computed)\n";
            iounit << " Test ratio:\n";
            iounit << "    1" << str9954 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmp(p2, "QS", 2)==0)
        {
            // QR factorization variants
            iounit << "\n " << pathcopy << ":  QR factorization variants\n";
            iounit << " Matrix types:\n";
            iounit << str9970 << '\n';
            iounit << " Test ratios:" << std::endl;
        }
        else if (std::strncmp(p2, "QT", 2)==0)
        {
            // QRT (general matrices)
            iounit << "\n " << pathcopy << ":  QRT factorization for general matrices\n";
            iounit << " Test ratios:\n";
            iounit << "    1: norm( R - Q'*A ) / ( M * norm(A) * EPS )\n";
            iounit << "    2: norm( I - Q'*Q ) / ( M * EPS )\n";
            iounit << "    3: norm( Q*C - Q*C ) / ( M * norm(C) * EPS )\n";
            iounit << "    4: norm( Q'*C - Q'*C ) / ( M * norm(C) * EPS )\n";
            iounit << "    5: norm( C*Q - C*Q ) / ( M * norm(C) * EPS )\n";
            iounit << "    6: norm( C*Q' - C*Q' ) / ( M * norm(C) * EPS )" << std::endl;
        }
        else if (std::strncmp(p2, "QX", 2)==0)
        {
            // QRT (triangular-pentagonal)
            iounit << "\n " << pathcopy;
                iounit << ":  QRT factorization for triangular-pentagonal matrices\n";
            iounit << " Test ratios:\n";
            iounit << "    1: norm( R - Q'*A ) / ( (M+N) * norm(A) * EPS )\n";
            iounit << "    2: norm( I - Q'*Q ) / ( (M+N) * EPS )\n";
            iounit << "    3: norm( Q*C - Q*C ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    4: norm( Q'*C - Q'*C ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    5: norm( C*Q - C*Q ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    6: norm( C*Q' - C*Q' ) / ( (M+N) * norm(C) * EPS )" << std::endl;
        }
        else if (std::strncmp(p2, "TQ", 2)==0)
        {
            // QRT (triangular-pentagonal)
            iounit << "\n " << pathcopy << ":  LQT factorization for general matrices\n";
            iounit << " Test ratios:\n";
            iounit << "    1: norm( L - A*Q' ) / ( (M+N) * norm(A) * EPS )\n";
            iounit << "    2: norm( I - Q*Q' ) / ( (M+N) * EPS )\n";
            iounit << "    3: norm( Q*C - Q*C ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    4: norm( Q'*C - Q'*C ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    5: norm( C*Q - C*Q ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    6: norm( C*Q' - C*Q' ) / ( (M+N) * norm(C) * EPS )" << std::endl;
        }
        else if (std::strncmp(p2, "XQ", 2)==0)
        {
            // QRT (triangular-pentagonal)
            iounit << "\n " << pathcopy;
                iounit << ":  LQT factorization for triangular-pentagonal matrices\n";
            iounit << " Test ratios:\n";
            iounit << "    1: norm( L - A*Q' ) / ( (M+N) * norm(A) * EPS )\n";
            iounit << "    2: norm( I - Q*Q' ) / ( (M+N) * EPS )\n";
            iounit << "    3: norm( Q*C - Q*C ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    4: norm( Q'*C - Q'*C ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    5: norm( C*Q - C*Q ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    6: norm( C*Q' - C*Q' ) / ( (M+N) * norm(C) * EPS )" << std::endl;
        }
        else if (std::strncmp(p2, "TS", 2)==0)
        {
            // QRT (triangular-pentagonal)
            iounit << "\n " << pathcopy;
                iounit << ":  TS factorization for tall-skiny or short-wide matrices\n";
            iounit << " Test ratios:\n";
            iounit << "    1: norm( R - Q'*A ) / ( (M+N) * norm(A) * EPS )\n";
            iounit << "    2: norm( I - Q'*Q ) / ( (M+N) * EPS )\n";
            iounit << "    3: norm( Q*C - Q*C ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    4: norm( Q'*C - Q'*C ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    5: norm( C*Q - C*Q ) / ( (M+N) * norm(C) * EPS )\n";
            iounit << "    6: norm( C*Q' - C*Q' ) / ( (M+N) * norm(C) * EPS )" << std::endl;
        }
        else
        {
            // Print error message if no header is available.
            iounit << "\n " << pathcopy << ":  No header available" << std::endl;
        }
    }

    /* alareq handles input for the LAPACK test program. It is called to evaluate the input line
     * which requested nmats matrix types for nin. The flow of control is as follows:
     * If nmats = ntypes then
     *     dotype[0:ntypes-1] = TRUE
     * else
     *     Read the next input line for nmats matrix types
     *     Set dotype[I] = TRUE for each valid type I
     * endif
     * Parameters: nin: An LAPACK path name for testing.
     *             nmats: The number of matrix types to be used in testing this path.
     *             dotype: a boolean array, dimension (ntypes)
     *                     The vector of flags indicating if each type will be tested.
     *             ntypes: The maximum number of matrix types for this path.
     *             ntypes: The input stream.
     *             nout:The output stream.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void alareq(char const* path, int nmats, bool* dotype, int ntypes, std::istream& nin,
                std::ostream& nout)
    {
        int i;
        if (nmats>=ntypes)
        {
            // Test everything if nmats>=ntypes.
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
            bool firstt = true;
            int nreq[100];
            // Read a line of matrix types if 0 < nmats < ntypes.
            if (nmats>0)
            {
                char line[80];
                nin >> std::setw(80) >> line;
                if (nin.good())
                {
                    int lenp = std::strlen(line);
                    i = -1;
                    char c1;
                    int i1, ic=0, j, k;
                    char const INTSTR[10] = {'0','1','2','3','4','5','6','7','8','9'};
                    char const* str9994 = " matrix types on this line or adjust NTYPES on previous"
                                          " line";
                    for (j=0; j<nmats; j++)
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
                                    nout << "\n\n *** Not enough matrix types on input line\n"
                                         << std::setw(79) << line << std::endl;
                                    nout << " ==> Specify " << std::setw(4) << nmats << str9994
                                         << std::endl;
                                    return;
                                }
                            }
                            if (line[i]!=' ' && line[i]!=',')
                            {
                                i1 = i;
                                c1 = line[i1];
                                // Check that a valid integer was read
                                for (k=0; k<10; k++)
                                {
                                    if (c1==INTSTR[k])
                                    {
                                        ic = k;
                                        break;
                                    }
                                }
                                if (c1!=INTSTR[k])
                                {
                                    nout << "\n\n *** Invalid integer value in column "
                                         << std::setw(2) << i+1 << " of input line:\n"
                                         << std::setw(79) << line << std::endl;
                                    nout << " ==> Specify " << std::setw(4) << nmats << str9994
                                         << std::endl;
                                    return;
                                }
                                nreq[j] = 10*nreq[j] + ic;
                                continue;
                            }
                            else if (i1>=0)
                            {
                                break;
                            }
                            else
                            {
                                continue;
                            }
                        }
                    }
                }
            }
            if (nin.good())
            {
                int nt;
                for (i=0; i<nmats; i++)
                {
                    nt = nreq[i]-1;
                    if (nt>=0 && nt<ntypes)
                    {
                        if (dotype[nt])
                        {
                            if (firstt)
                            {
                                nout << std::endl;
                            }
                            firstt = false;
                            nout << " *** Warning:  duplicate request of matrix type "
                                 << std::setw(2) << nt+1 << " for " << std::setw(3) << path << std::endl;
                        }
                        dotype[nt] = true;
                    }
                    else
                    {
                        nout << " *** Invalid type request for " << std::setw(3) << path << ", type  "
                             << std::setw(4) << nt+1 << ": must satisfy  1 <= type <= " << std::setw(2)
                             << ntypes << std::endl;
                    }
                }
            }
        }
        if (!nin.good())
        {
            nout << "\n *** End of file reached when trying to read matrix types for " << path
                 << " *** Check that you are requesting the right number of types for each path\n"
                 << '\n' << std::endl;
        }
        return;
    }

    /* alasum prints a summary of results from one of the -CHK- routines.
     * Parameters: type: The LAPACK path name.
     *             nout: The output stream to which results are to be printed.
     *             nfail: The number of tests which did not pass the threshold ratio.
     *             nrun: The total number of tests.
     *             nerrs: The number of error messages recorded.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void alasum(char const* type, std::ostream& nout, int nfail, int nrun, int nerrs)
    {
        char typecopy[4];
        strncpy(typecopy, type, 3);
        typecopy[3] = '\0';
        if (nfail>0)
        {
            nout << ' ' << typecopy << ": " << std::setw(6) << nfail << " out of ";
            nout << std::setw(6) << nrun << " tests failed to pass the threshold\n";
        }
        else
        {
            nout << "\n All tests for " << typecopy << " routines passed the threshold ( ";
            nout << std::setw(6) << nrun << " tests run)\n";
        }
        if (nerrs>0)
        {
            nout << "      " << std::setw(6) << nerrs << " error messages recorded\n";
        }
        nout.flush();
    }

    /* dchkaa is the main test program for the DOUBLE PRECISION LAPACK linear equation routines
     * The program must be driven by a short data file. The first 15 records (not including the
     * first comment line) specify problem dimensions and program options using list-directed
     * input. The remaining lines specify the LAPACK test paths and the number of matrix types to
     * use in testing. An annotated example of a data file can be obtained by deleting the first 3
     * characters from the following 40 lines:
     * Data file for testing DOUBLE PRECISION LAPACK linear eqn. routines
     * 7                      Number of values of M
     * 0 1 2 3 5 10 16        Values of M (row dimension)
     * 7                      Number of values of N
     * 0 1 2 3 5 10 16        Values of N (column dimension)
     * 1                      Number of values of NRHS
     * 2                      Values of NRHS (number of right hand sides)
     * 5                      Number of values of NB
     * 1 3 3 3 20             Values of NB (the blocksize)
     * 1 0 5 9 1              Values of NX (crossover point)
     * 3                      Number of values of RANK
     * 30 50 90               Values of rank (as a % of N)
     * 20.0                   Threshold value of test ratio
     * T                      Put T to test the LAPACK routines
     * T                      Put T to test the driver routines
     * T                      Put T to test the error exits
     * DGE   11               List types on next line if 0 < NTYPES < 11
     * DGB    8               List types on next line if 0 < NTYPES <  8
     * DGT   12               List types on next line if 0 < NTYPES < 12
     * DPO    9               List types on next line if 0 < NTYPES <  9
     * DPS    9               List types on next line if 0 < NTYPES <  9
     * DPP    9               List types on next line if 0 < NTYPES <  9
     * DPB    8               List types on next line if 0 < NTYPES <  8
     * DPT   12               List types on next line if 0 < NTYPES < 12
     * DSY   10               List types on next line if 0 < NTYPES < 10
     * DSR   10               List types on next line if 0 < NTYPES < 10
     * DSK   10               List types on next line if 0 < NTYPES < 10
     * DSA   10               List types on next line if 0 < NTYPES < 10
     * DS2   10               List types on next line if 0 < NTYPES < 10
     * DSP   10               List types on next line if 0 < NTYPES < 10
     * DTR   18               List types on next line if 0 < NTYPES < 18
     * DTP   18               List types on next line if 0 < NTYPES < 18
     * DTB   17               List types on next line if 0 < NTYPES < 17
     * DQR    8               List types on next line if 0 < NTYPES <  8
     * DRQ    8               List types on next line if 0 < NTYPES <  8
     * DLQ    8               List types on next line if 0 < NTYPES <  8
     * DQL    8               List types on next line if 0 < NTYPES <  8
     * DQP    6               List types on next line if 0 < NTYPES <  6
     * DTZ    3               List types on next line if 0 < NTYPES <  3
     * DLS    6               List types on next line if 0 < NTYPES <  6
     * DEQ
     * DQT
     * DQX
     * Parameters: NMAX: The maximum allowable value for M and N.
     *             MAXIN: The number of different values that can be used for each of M, N, NRHS,
     *                    NB, NX and RANK
     *             MAXRHS: The maximum number of right hand sides
     *             MATMAX: The maximum number of matrix types to use for testing
     *             nin: The input stream
     *             nout: The output stream
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date April 2012                                                                           */
    void dchkaa(const int NMAX, const int MAXIN, const int MAXRHS, const int MATMAX,
                std::istream& nin, std::ostream& nout)
    {
        /*
        const int NMAX = 132;
        const int MAXIN = 12;
        const int MAXRHS = 16;
        const int MATMAX = 30;
        std::istream& nin = std::cin;
        std::ostream& nout = std::cout;
         */
        const int KDMAX = NMAX+(NMAX+1) / 4;
        const char INTSTR[10] = {'0','1','2','3','4','5','6','7','8','9'};
        const real THREQ=real(2.0);
        std::time_t s1, s2;
        std::time(&s1);
        // Read a dummy line.
        {
            char dummy[100];
            nin.getline(dummy, 100);
        }
        // Report values of parameters.
        int vers_major, vers_minor, vers_patch;
        this->ilaver(vers_major, vers_minor, vers_patch);
        nout << " Tests of the DOUBLE PRECISION LAPACK routines \n LAPACK VERSION " << std::setw(1)
             << vers_major << '.' << std::setw(1) <<  vers_minor << '.' << std::setw(1)
             << vers_patch << "\n\n The following parameter values will be used:" << std::endl;
        // Read the values of M
        char const* str99965 = " Invalid input value: ";
        char const* str9995b = "; must be <=";
        char const* str9996b = "; must be >=";
        bool fatal = false;
        int nm;
        std::string dummy;
        nin >> nm;
        std::getline(nin, dummy);
        if (nm<1)
        {
            nout << str99965 <<  " NM =" << std::setw(6) << nm << str9996b << std::setw(6) << 1
                 << std::endl;
            nm = 0;
            fatal = true;
        }
        else if (nm>MAXIN)
        {
            nout << str99965 << " NM =" << std::setw(6) << nm << str9995b << std::setw(6) << MAXIN
                 << std::endl;
            nm = 0;
            fatal = true;
        }
        int i;
        int* mval = new int[MAXIN];
        for (i=0; i<nm; i++)
        {
            nin >> mval[i];
        }
        std::getline(nin, dummy);
        for (i=0; i<nm; i++)
        {
            if (mval[i]<0)
            {
                nout << str99965 << " M  =" << std::setw(6) << mval[i] << str9996b << std::setw(6)
                     << 0 << std::endl;
                fatal = true;
            }
            else if (mval[i]>NMAX)
            {
                nout << str99965 <<  " M  =" << std::setw(6) << mval[i] << str9995b
                     << std::setw(6) << NMAX << std::endl;
                fatal = true;
            }
        }
        if (nm>0)
        {
            nout << "    M   :  ";
            for (i=0; i<nm; i++)
            {
                nout << std::setw(6) << mval[i];
                if (i%10==9 && i<nm-1)
                {
                    nout << "\n           ";
                }
            }
            nout << std::endl;
        }
        // Read the values of N
        int nn;
        nin >> nn;
        std::getline(nin, dummy);
        if (nn<1)
        {
            nout << str99965 << " NN =" << std::setw(6) << nn << str9996b << std::setw(6) << 1
                 << std::endl;
            nn = 0;
            fatal = true;
        }
        else if (nn>MAXIN)
        {
            nout << str99965 << " NN =" << std::setw(6) << nn << str9995b << std::setw(6) << MAXIN
                 << std::endl;
            nn = 0;
            fatal = true;
        }
        int* nval = new int[MAXIN];
        for (i=0; i<nn; i++)
        {
            nin >> nval[i];
        }
        std::getline(nin, dummy);
        for (i=0; i<nn; i++)
        {
            if (nval[i]<0)
            {
                nout << str99965 << " N  =" << std::setw(6) << nval[i] << str9996b
                     << std::setw(6) << 0 << std::endl;
                fatal = true;
            }
            else if (nval[i]>NMAX)
            {
                nout << str99965 << " N  =" << std::setw(6) << nval[i] << str9995b
                     << std::setw(6) << NMAX << std::endl;
                fatal = true;
            }
        }
        if (nn>0)
        {
            nout << "    N   :  ";
            for (i=0; i<nn; i++)
            {
                nout << std::setw(6) << nval[i];
                if ((i%10)==9 && i<nn-1)
                {
                    nout << "\n           ";
                }
            }
            nout << std::endl;
        }
        // Read the values of NRHS
        int nns;
        nin >> nns;
        std::getline(nin, dummy);
        if (nns<1)
        {
            nout << str99965 << " NNS=" << std::setw(6) << nns << str9996b << std::setw(6) << 1
                 << std::endl;
            nns = 0;
            fatal = true;
        }
        else if (nns>MAXIN)
        {
            nout << str99965 << " NNS=" << std::setw(6) << nns << str9995b << std::setw(6) << MAXIN
                 << std::endl;
            nns = 0;
            fatal = true;
        }
        int* nsval = new int[MAXIN];
        for (i=0; i<nns; i++)
        {
            nin >> nsval[i];
        }
        std::getline(nin, dummy);
        for (i=0; i<nns; i++)
        {
            if (nsval[i]<0)
            {
                nout << str99965 << "NRHS=" << std::setw(6) << nsval[i] << str9996b
                     << std::setw(6) << 0 << std::endl;
                fatal = true;
            }
            else if (nsval[i]>MAXRHS)
            {
                nout << str99965 << "NRHS=" << std::setw(6) << nsval[i] << str9995b
                     << std::setw(6) << MAXRHS << std::endl;
                fatal = true;
            }
        }
        if (nns>0)
        {
            nout << "    NRHS:  ";
            for (i=0; i<nns; i++)
            {
                nout << std::setw(6) << nsval[i];
                if (i%10==9 && i<nns-1)
                {
                    nout << "\n           ";
                }
            }
            nout << std::endl;
        }
        // Read the values of NB
        int nnb;
        nin >> nnb;
        std::getline(nin, dummy);
        if (nnb<1)
        {
            nout << str99965 << "NNB =" << std::setw(6) << nnb << str9996b << std::setw(6) << 1
                 << std::endl;
            nnb = 0;
            fatal = true;
        }
        else if (nnb>MAXIN)
        {
            nout << str99965 << "NNB =" << std::setw(6) << nnb << str9995b << std::setw(6) << MAXIN
                 << std::endl;
            nnb = 0;
            fatal = true;
        }
        int* nbval = new int[MAXIN];
        for (i=0; i<nnb; i++)
        {
            nin >> nbval[i];
        }
        std::getline(nin, dummy);
        for (i=0; i<nnb; i++)
        {
            if (nbval[i]<0)
            {
                nout << str99965 << " NB =" << std::setw(6) << nbval[i] << str9996b
                     << std::setw(6) << 0 << std::endl;
                fatal = true;
            }
        }
        if (nnb>0)
        {
            nout << "    NB  :  ";
            for (i=0; i<nnb; i++)
            {
                nout << std::setw(6) << nbval[i];
                if (i%10==9 && i<nnb-1)
                {
                    nout << "\n           ";
                }
            }
            nout << std::endl;
        }
        // Set nbval2 to be the set of unique values of NB
        int j, nb, nnb2 = 0;
        bool breakloop;
        int* nbval2 = new int[MAXIN];
        for (i=0; i<nnb; i++)
        {
            nb = nbval[i];
            breakloop = false;
            for (j=0; j<nnb2; j++)
            {
                if (nb==nbval2[j])
                {
                    breakloop = true;
                    break;
                }
            }
            if (!breakloop)
            {
                nnb2++;
                nbval2[nnb2-1] = nb;
            }
        }
        // Read the values of NX
        int* nxval = new int[MAXIN];
        for (i=0; i<nnb; i++)
        {
            nin >> nxval[i];
        }
        std::getline(nin, dummy);
        for (i=0; i<nnb; i++)
        {
            if (nxval[i]<0)
            {
                nout << str99965 << " NX =" << std::setw(6) << nxval[i] << str9996b
                     << std::setw(6) << 0 << std::endl;
               fatal = true;
            }
        }
        if (nnb>0)
        {
            nout << "    NX  :  ";
            for (i=0; i<nnb; i++)
            {
                nout << std::setw(6) << nxval[i];
                if (i%10==9 && i<nnb-1)
                {
                    nout << "\n           ";
                }
            }
            nout << std::endl;
        }
        // Read the values of rankval
        int nrank;
        nin >> nrank;
        std::getline(nin, dummy);
        if (nn<1)
        {
            nout << str99965 << " NRA=" << std::setw(6) << nrank << str9996b << std::setw(6) << 1
                 << std::endl;//nrank
            nrank = 0;
            fatal = true;
        }
        else if (nn>MAXIN)
        {
            nout << str99965 << " NRA=" << std::setw(6) << nrank << str9995b << std::setw(6)
                 << MAXIN << std::endl;//nrank
            nrank = 0;
            fatal = true;
        }
        int* rankval = new int[MAXIN];
        for (i=0; i<nrank; i++)
        {
            nin >> rankval[i];
        }
        std::getline(nin, dummy);
        for (i=0; i<nrank; i++)
        {
            if (rankval[i]<0)
            {
                nout << str99965 << " RAN=" << std::setw(6) << rankval[i] << str9996b
                     << std::setw(6) << 0 << std::endl;//rank
                fatal = true;
            }
            else if (rankval[i]>100)
            {
                nout << str99965 << " RAN=" << std::setw(6) << rankval[i] << str9995b
                     << std::setw(6) << 100 << std::endl;//rank
                fatal = true;
            }
        }
        if (nrank>0)
        {
            nout << "    RANK:  ";//'RANK % OF N'
            for (i=0; i<nrank; i++)
            {
                nout << std::setw(6) << rankval[i];
                if (i%10==9 && i<nrank-1)
                {
                    nout << "\n           ";
                }
            }
            nout << std::endl;
        }
        // Read the threshold value for the test ratios.
        bool tstchk, tstdrv, tsterr;
        real thresh;
        char c1;
        nin >> thresh;
        std::getline(nin, dummy);
        nout << "\n Routines pass computational tests if test ratio is less than" << std::setw(8)
             << std::setprecision(2) << thresh << '\n' << std::endl;
        // Read the flag that indicates whether to test the LAPACK routines.
        nin >> c1;
        tstchk = (std::toupper(c1)=='T');
        std::getline(nin, dummy);
        // Read the flag that indicates whether to test the driver routines.
        nin >> c1;
        tstdrv = (std::toupper(c1)=='T');
        std::getline(nin, dummy);
        // Read the flag that indicates whether to test the error exits.
        nin >> c1;
        tsterr = (std::toupper(c1)=='T');
        std::getline(nin, dummy);
        if (fatal)
        {
            nout << "\n Execution not attempted due to input errors" << std::endl;
            return;
        }
        // Calculate and print the machine dependent constants.
        char const* str9991a = " Relative machine ";
        char const* str9991b = " is taken to be";
        real eps = this->dlamch("Underflow threshold");
        nout << str9991a << "underflow" << str9991b << std::setw(16) << std::setprecision(6) << eps
             << std::endl;
        eps = this->dlamch("Overflow threshold");
        nout << str9991a << "overflow " << str9991b << std::setw(16) << std::setprecision(6) << eps
             << std::endl;
        eps = this->dlamch("Epsilon");
        nout << str9991a << "precision" << str9991b << std::setw(16) << std::setprecision(6) << eps
             << '\n' << std::endl;
        // start read loop
        char c2[2];
        char path[4];
        std::string aline;
        //char aline[72];
        bool* dotype = new bool[MATMAX];
        int* iwork = new int[25*NMAX];
        int* piv = new int[NMAX];
        int ldatot = (KDMAX+1)*NMAX;
        real* A = new real[ldatot * 7];
        int ldbtot = NMAX*MAXRHS;
        real* B = new real[ldbtot * 4];
        real* E = new real[NMAX];
        real* Rwork = new real[5*NMAX+2*MAXRHS];
        real* S = new real[2*NMAX];
        real* Work = new real[NMAX * 3*NMAX+MAXRHS+30];
        int lda = NMAX;
        int ic, k, la, lafac, nmats, nrhs, ntypes;
        char const* str9988 = " driver routines were not tested";
        char const* str9989 = " routines were not tested";
        char const* str9990 = ":  Unrecognized path name";
        while (!nin.eof())
        {
            // Read a test path and the number of matrix types to use.
            std::getline(nin, aline);
            //nin >> std::setw(72) >> aline;
            if (!nin.good())
            {
                break;
            }
            std::strncpy(path,aline.c_str(),3);
            path[3] = '\0';
            nmats = MATMAX;
            i = 2;
            breakloop = false;
            do
            {
                i++;
                if (i>=72)
                {
                    nmats = MATMAX;
                    breakloop = true;
                    break;
                }
            } while (aline[i]==' ');
            if (!breakloop)
            {
                nmats = 0;
                do
                {
                    c1 = aline[i];
                    breakloop = false;
                    for (k=0; k<10; k++)
                    {
                        if (c1==INTSTR[k])
                        {
                            ic = k;
                            breakloop = true;
                            break;
                        }
                    }
                    if (!breakloop)
                    {
                        break;
                    }
                    nmats = nmats*10 + ic;
                    i++;
                } while (i<72);
            }
            c1 = std::toupper(path[0]);
            c2[0] = std::toupper(path[1]);
            c2[1] = std::toupper(path[2]);
            nrhs = nsval[0];
            // Check first character for correct precision.
            if (c1!='D') // Double precision
            {
                nout << "\n " << path << str9990 << std::endl;
            }
            else if (nmats<=0)
            {
                // Check for a positive number of tests requested.
                nout << "\n " << path << str9989 << std::endl;
            }
            else if (std::strncmp(c2, "GE", 2)==0)
            {
                // GE:  general matrices
                ntypes = 11;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKGE(dotype, NM, mval, NN, nval, NNB2, nbval2, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKGE not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVGE(dotype, NN, nval, nrhs, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], &B[0+ldbtot*3], S, Work, Rwork, iwork, nout)
                    std::cerr << "DDRVGE not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "GB", 2)==0)
            {
                // GB:  general banded matrices
                la = (2*KDMAX+1)*NMAX;
                lafac = (3*KDMAX+1)*NMAX;
                ntypes = 8;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKGB(dotype, NM, mval, NN, nval, NNB2, nbval2, NNS, nsval, thresh, tsterr, &A[0+ldatot*0], la, &A[0+ldatot*2], lafac, &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKGB not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVGB(dotype, NN, nval, nrhs, thresh, tsterr, &A[0+ldatot*0], la, &A[0+ldatot*2], lafac, A(1, 6), &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], &B[0+ldbtot*3], S, Work, Rwork, iwork, nout)
                    std::cerr << "DDRVGB not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "GT", 2)==0)
            {
                // GT:  general tridiagonal matrices
                ntypes = 12;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKGT(dotype, NN, nval, NNS, nsval, thresh, tsterr, &A[0+ldatot*0], &A[0+ldatot*1], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKGT not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVGT(dotype, NN, nval, nrhs, thresh, tsterr, &A[0+ldatot*0], &A[0+ldatot*1], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DDRVGT not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "PO", 2)==0)
            {
                // PO:  positive definite matrices
                ntypes = 9;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKPO(dotype, NN, nval, NNB2, nbval2, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKPO not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVPO(dotype, NN, nval, nrhs, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], &B[0+ldbtot*3], S, Work, Rwork, iwork, nout)
                    std::cerr << "DDRVPO not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "PS", 2)==0)
            {
                // PS:  positive semi-definite matrices
                ntypes = 9;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKPS(dotype, NN, nval, NNB2, nbval2, NRANK, rankval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], piv, Work, Rwork, nout)
                    std::cerr << "DCHKPS not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "PP", 2)==0)
            {
                // PP:  positive definite packed matrices
                ntypes = 9;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKPP(dotype, NN, nval, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKPP not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVPP(dotype, NN, nval, nrhs, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], &B[0+ldbtot*3], S, Work, Rwork, iwork, nout)
                    std::cerr << "DDRVPP not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "PB", 2)==0)
            {
                // PB:  positive definite banded matrices
                ntypes = 8;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKPB(dotype, NN, nval, NNB2, nbval2, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKPB not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVPB(dotype, NN, nval, nrhs, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], &B[0+ldbtot*3], S, Work, Rwork, iwork, nout)
                    std::cerr << "DDRVPB not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "PT", 2)==0)
            {
                // PT:  positive definite tridiagonal matrices
                ntypes = 12;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKPT(dotype, NN, nval, NNS, nsval, thresh, tsterr, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, nout)
                    std::cerr << "DCHKPT not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVPT(dotype, NN, nval, nrhs, thresh, tsterr, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, nout)
                    std::cerr << "DDRVPT not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "SY", 2)==0)
            {
                // SY: symmetric indefinite matrices, with partial (Bunch-Kaufman) pivoting
                //     algorithm
                ntypes = 10;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKSY(dotype, NN, nval, NNB2, nbval2, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKSY not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVSY(dotype, NN, nval, nrhs, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DDRVSY not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "SR", 2)==0)
            {
                // SR:  symmetric indefinite matrices, with bounded Bunch-Kaufman (rook) pivoting
                //      algorithm
                ntypes = 10;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKSY_ROOK(dotype, NN, nval, NNB2, nbval2, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKSY_ROOK not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVSY_ROOK(dotype, NN, nval, nrhs, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DDRVSY_ROOK not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "SK", 2)==0)
            {
                // SK: symmetric indefinite matrices, with bounded Bunch-Kaufman (rook) pivoting
                //     algorithm, differnet matrix storage format than SR path version.
                ntypes = 10;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKSY_RK(dotype, NN, nval, NNB2, nbval2, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], E, &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKSY_RK not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVSY_RK(dotype, NN, nval, nrhs, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], E, &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DDRVSY_RK not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "SA", 2)==0)
            {
                // SA: symmetric indefinite matrices, with partial (Aasen's) pivoting algorithm
                ntypes = 10;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKSY_AA(dotype, NN, nval, NNB2, nbval2, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKSY_AA not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVSY_AA(dotype, NN, nval, nrhs, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DDRVSY_AA not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "S2", 2)==0)
            {
                // SA: symmetric indefinite matrices, with partial (Aasen's) pivoting algorithm
                ntypes = 10;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKSY_AA_2STAGE(dotype, NN, nval, NNB2, nbval2, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKSY_AA_2STAGE not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                   //CALL DDRVSY_AA_2STAGE(dotype, NN, nval, nrhs, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DDRVSY_AA_2STAGE not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "SP", 2)==0)
            {
                // SP: symmetric indefinite packed matrices, with partial (Bunch-Kaufman) pivoting
                //     algorithm
                ntypes = 10;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKSP(dotype, NN, nval, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKSP not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
                if (tstdrv)
                {
                    //CALL DDRVSP(dotype, NN, nval, nrhs, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DDRVSP not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "TR", 2)==0)
            {
                // TR: triangular matrices
                ntypes = 18;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKTR(dotype, NN, nval, NNB2, nbval2, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKTR not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "TP", 2)==0)
            {
                // TP: triangular packed matrices
                ntypes = 18;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKTP(dotype, NN, nval, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKTP not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "TB", 2)==0)
            {
                // TB: triangular banded matrices
                ntypes = 17;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKTB(dotype, NN, nval, NNS, nsval, thresh, tsterr, LDA, &A[0+ldatot*0], &A[0+ldatot*1], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKTB not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "QR", 2)==0)
            {
                // QR: QR factorization
                ntypes = 8;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKQR(dotype, NM, mval, NN, nval, NNB, nbval, nxval, nrhs, thresh, tsterr, NMAX, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &A[0+ldatot*3], &A[0+ldatot*4], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], &B[0+ldbtot*3], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKQR not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "LQ", 2)==0)
            {
                // LQ: LQ factorization
                ntypes = 8;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKLQ(dotype, NM, mval, NN, nval, NNB, nbval, nxval, nrhs, thresh, tsterr, NMAX, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &A[0+ldatot*3], &A[0+ldatot*4], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], &B[0+ldbtot*3], Work, Rwork, nout)
                    std::cerr << "DCHKLQ not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "QL", 2)==0)
            {
                // QL: QL factorization
                ntypes = 8;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKQL(dotype, NM, mval, NN, nval, NNB, nbval, nxval, nrhs, thresh, tsterr, NMAX, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &A[0+ldatot*3], &A[0+ldatot*4], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], &B[0+ldbtot*3], Work, Rwork, nout)
                    std::cerr << "DCHKQL not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "RQ", 2)==0)
            {
                // RQ: RQ factorization
                ntypes = 8;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKRQ(dotype, NM, mval, NN, nval, NNB, nbval, nxval, nrhs, thresh, tsterr, NMAX, &A[0+ldatot*0], &A[0+ldatot*1], &A[0+ldatot*2], &A[0+ldatot*3], &A[0+ldatot*4], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], &B[0+ldbtot*3], Work, Rwork, iwork, nout)
                    std::cerr << "DCHKRQ not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "QP", 2)==0)
            {
                // QP: QR factorization with pivoting
                ntypes = 6;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    this->dchkq3(dotype, nm, mval, nn, nval, nnb, nbval, nxval, thresh, A,
                                 &A[ldatot], B, &B[ldbtot*2], Work, iwork, nout);
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "TZ", 2)==0)
            {
                // TZ: Trapezoidal matrix
                ntypes = 3;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstchk)
                {
                    //CALL DCHKTZ(dotype, NM, mval, NN, nval, thresh, tsterr, &A[0+ldatot*0], &A[0+ldatot*1], &B[0+ldbtot*0], &B[0+ldbtot*2], Work, nout)
                    std::cerr << "DCHKTZ not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "LS", 2)==0)
            {
                // LS: Least squares drivers
                ntypes = 6;
                alareq(path, nmats, dotype, ntypes, nin, nout);
                if (tstdrv)
                {
                    //CALL DDRVLS(dotype, NM, mval, NN, nval, NNS, nsval, NNB, nbval, nxval, thresh, tsterr, &A[0+ldatot*0], &A[0+ldatot*1], &B[0+ldbtot*0], &B[0+ldbtot*1], &B[0+ldbtot*2], Rwork, &Rwork[NMAX], nout)
                    std::cerr << "DDRVLS not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9988 << std::endl;
                }
            }
            else if (std::strncmp(c2, "EQ", 2)==0)
            {
                // EQ: Equilibration routines for general and positive definite matrices
                //     (THREQ should be between 2 and 10)
                if (tstchk)
                {
                    //CALL DCHKEQ(THREQ, nout)
                    std::cerr << "DCHKEQ not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "QT", 2)==0)
            {
                // QT: QRT routines for general matrices
                if (tstchk)
                {
                    //CALL DCHKQRT(thresh, tsterr, NM, mval, NN, nval, NNB, nbval, nout)
                    std::cerr << "DCHKQRT not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "QX", 2)==0)
            {
                // QX: QRT routines for triangular-pentagonal matrices
                if (tstchk)
                {
                    //CALL DCHKQRTP(thresh, tsterr, NM, mval, NN, nval, NNB, nbval, nout)
                    std::cerr << "DCHKQRTP not yet implemented" << std::endl;
                }
                else
                {
                     nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "TQ", 2)==0)
            {
                // TQ: LQT routines for general matrices
                if (tstchk)
                {
                    //CALL DCHKLQT(thresh, tsterr, NM, mval, NN, nval, NNB, nbval, nout)
                    std::cerr << "DCHKLQT not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "XQ", 2)==0)
            {
                // XQ:  LQT routines for triangular-pentagonal matrices
                if (tstchk)
                {
                    //CALL DCHKLQTP(thresh, tsterr, NM, mval, NN, nval, NNB, nbval, nout)
                    std::cerr << "DCHKLQTP not yet implemented" << std::endl;
                }
                else
                {
                    nout << "\n " << path << str9989 << std::endl;
                }
            }
            else if (std::strncmp(c2, "TS", 2)==0)
            {
                // TS:  QR routines for tall-skinny matrices
                if (tstchk)
                {
                    //CALL DCHKTSQR(thresh, tsterr, NM, mval, NN, nval, NNB, nbval, nout)
                    std::cerr << "DCHKTSQR not yet implemented" << std::endl;
                }
                else
                {
                     nout << "\n " << path << str9989 << std::endl;
                }
            }
            else
            {
                nout << "\n " << path << str9990 << std::endl;
            }
            // Go back to get another input line.
        }
        // Branch to this line when the last record is read.
        std::time(&s2);
        nout << "\n End of tests" << std::endl;
        nout << " Total time used = " << std::setprecision(2) << std::setw(12) << s2-s1
             << " seconds\n" << std::endl;
        delete[] dotype;
        delete[] iwork;
        delete[] mval;
        delete[] nbval;
        delete[] nbval2;
        delete[] nsval;
        delete[] nval;
        delete[] nxval;
        delete[] rankval;
        delete[] piv;
        delete[] A;
        delete[] B;
        delete[] E;
        delete[] Rwork;
        delete[] S;
        delete[] Work;
    }

    /* dchkq3 tests dgeqp3.
     * Parameters: dotype: a boolean array, dimension (NTYPES)
     *                     The matrix types to be used for testing. Matrices of type j
     *                     (for 0<=j<NTYPES) are used for testing if dotype[j]==true; if
     *                     dotype[j]==false, then type j is not used.
     *             nm: The number of values of M contained in the vector mval.
     *             mval: an integer array, dimension (nm)
     *                   The values of the matrix row dimension M.
     *             nn: The number of values of N contained in the vector nval.
     *             nval: an integer array, dimension (nn)
     *                   The values of the matrix column dimension N.
     *             nnb: The number of values of NB and NX contained in the vectors nbval and nxval.
     *                  The blocking parameters are used in pairs (NB,NX).
     *             nbval: an integer array, dimension (nnb)
     *                    The values of the blocksize NB.
     *             nxval: an integer array, dimension (nnb)
     *                    The values of the crossover point NX.
     *             thresh: The threshold value for the test ratios. A result is included in the
     *                     output file if result>=thresh. To have every test ratio printed, use
     *                     thresh=0.
     *             A: an array, dimension (MMAX*NMAX) where MMAX is the maximum value of M in mval
     *                and NMAX is the maximum value of N in nval.
     *             CopyA: an array, dimension (MMAX*NMAX)
     *             S: an array, dimension min(MMAX,NMAX))
     *             tau: an array, dimension (MMAX)
     *             work: an array, dimension (MMAX*NMAX + 4*NMAX + MMAX)
     *             iwork: an integer array, dimension (2*NMAX)
     *             nout: The output stream.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void dchkq3(bool const* dotype, int nm, int const* mval, int nn, int const* nval,
                       int nnb, int const* nbval, int const* nxval, real thresh, real* A,
                       real* CopyA, real* S, real* tau, real* work, int* iwork, std::ostream& nout)
    {
        const int NTYPES = 6;
        const int NTESTS = 3;
        const char PATH[3] = {'D','Q','3'};
        // Initialize constants and the random number seed.
        int nrun = 0;
        int nfail = 0;
        int nerrs = 0;
        int iseed[4] = {1988, 1989, 1990, 1991};
        int i, ihigh, ilow, im, imode, in, inb, info, istep, k, lda, lw, lwork, m, mnmin, mode, n,
            nb, nx, temp;
        real result[NTESTS];
        real eps = this->dlamch("Epsilon");
        infoc.info = 0;
        for (im=0; im<nm; im++)
        {
            // Do for each value of M in mval.
            m = mval[im];
            lda = std::max(1, m);
            for (in=0; in<nn; in++)
            {
                // Do for each value of N in nval.
                n = nval[in];
                if (m>n)
                {
                    mnmin = n;
                    temp = m;
                }
                else
                {
                    mnmin = m;
                    temp = n;
                }
                lwork = std::max(std::max(m*temp+4*mnmin+temp, m*n+2*mnmin+4*n), 1);
                for (imode=0; imode<NTYPES; imode++)
                {
                    if (!dotype[imode])
                    {
                        continue;
                    }
                    // Do for each type of matrix
                    //   0:  zero matrix
                    //   1:  one small singular value
                    //   2:  geometric distribution of singular values
                    //   3:  first n/2 columns fixed
                    //   4:  last n/2 columns fixed
                    //   5:  every second column fixed
                    mode = imode+1;
                    if (imode>2)
                    {
                        mode = 1;
                    }
                    // Generate test matrix of size m by n using singular value distribution
                    // indicated by 'mode'.
                    for (i=0; i<n; i++)
                    {
                        iwork[i] = -1;
                    }
                    if (imode==0)
                    {
                        this->dlaset("Full", m, n, ZERO, ZERO, CopyA, lda);
                        for (i=0; i<mnmin; i++)
                        {
                            S[i] = ZERO;
                        }
                    }
                    else
                    {
                        MatGen.dlatms(m, n, "Uniform", iseed, "Nonsymm", S, mode, ONE/eps, ONE,
                                      m, n, "No packing", CopyA, lda, work, info);
                        if (imode>=3)
                        {
                            if (imode==3)
                            {
                                ilow = 0;
                                istep = 1;
                                ihigh = std::max(n/2, 1);
                            }
                            else if (imode==4)
                            {
                                ilow = std::max(n/2-1, 0);
                                istep = 1;
                                ihigh = n;
                            }
                            else if (imode==5)
                            {
                               ilow = 0;
                               istep = 2;
                               ihigh = n;
                            }
                            for (i=ilow; i<ihigh; i+=istep)
                            {
                                iwork[i] = 0;
                            }
                        }
                        dlaord("Decreasing", mnmin, S, 1);
                    }
                    for (inb=0; inb<nnb; inb++)
                    {
                        // Do for each pair of values (NB,NX) in nbval and nxval.
                        nb = nbval[inb];
                        xlaenv(1, nb);
                        nx = nxval[inb];
                        xlaenv(3, nx);
                        // Get a working copy of CopyA into A and a copy of vector iwork.
                        this->dlacpy("All", m, n, CopyA, lda, A, lda);
                        icopy(n, &iwork[0], 1, &iwork[n], 1);
                        // Compute the QR factorization with pivoting of A
                        lw = std::max(1, 2*n+nb*(n+1));
                        // Compute the QP3 factorization of A
                        std::strncpy(srnamc.srnam,"DGEQP3", 7);
                        this->dgeqp3(m, n, A, lda, &iwork[n], tau, work, lw, info);
                        // Compute norm(svd(A) - svd(R))
                        result[0] = this->dqrt12(m, n, A, lda, S, work, lwork);
                        // Compute norm(A*P - Q*R)
                        result[1] = this->dqpt01(m, n, mnmin, CopyA, A, lda, tau, &iwork[n], work,
                                           lwork);
                        // Compute Q'*Q
                        result[2] = this->dqrt11(m, mnmin, A, lda, tau, work, lwork);
                        // Print information about the tests that did not pass the threshold.
                        for (k=0; k<NTESTS; k++)
                        {
                            if (result[k]>=thresh)
                            {
                                if (nfail==0 && nerrs==0)
                                {
                                    alahd(nout, PATH);
                                }
                                nout << " DGEQP3 M =" << std::setw(5) << m << ", N ="
                                     << std::setw(5) << n << ", NB =" << std::setw(4) << nb
                                     << ", type " << std::setw(2) << imode+1 << ", test "
                                     << std::setw(2) << k+1 << ", ratio =" << std::setw(12)
                                     << std::setprecision(5) << result[k] << std::endl;
                                nfail++;
                            }
                        }
                        nrun += NTESTS;
                    }
                }
            }
        }
        // Print a summary of the results.
        alasum(PATH, nout, nfail, nrun, nerrs);
    }

    /* dlaord sorts the elements of a vector x in increasing or decreasing order.
     * Parameters: job: = 'I':  Sort in increasing order
     *                  = 'D':  Sort in decreasing order
     *             n: The length of the vector X.
     *             X: an array, dimension (1+(n-1)*incx)
     *                On entry, the vector of length n to be sorted.
     *                On exit, the vector x is sorted in the prescribed order.
     *             incx: The spacing between successive elements of X. incx>=0.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void dlaord(char const* job, int n, real* x, int incx)
    {
        int i, ix, ixnext;
        real temp;
        int inc = abs(incx);
        if (toupper(job[0])=='I')
        {
            // Sort in increasing order
            for (i=1; i<n; i++)
            {
                ix = i*inc;
                while (true)
                {
                    if (ix==0)
                    {
                        break;
                    }
                    ixnext = ix - inc;
                    if (x[ix]>x[ixnext])
                    {
                        break;
                    }
                    else
                    {
                        temp = x[ix];
                        x[ix] = x[ixnext];
                        x[ixnext] = temp;
                    }
                    ix = ixnext;
                }
            }
        }
        else if (toupper(job[0])=='D')
        {
            // Sort in decreasing order
            for (i=1; i<n; i++)
            {
                ix = i*inc;
                while (true)
                {
                    if (ix==0)
                    {
                        break;
                    }
                    ixnext = ix - inc;
                    if (x[ix]<x[ixnext])
                    {
                        break;
                    }
                    else
                    {
                        temp = x[ix];
                        x[ix] = x[ixnext];
                        x[ixnext] = temp;
                    }
                    ix = ixnext;
                }
            }
        }
    }

    /* dqpt01 tests the QR-factorization with pivoting of a matrix A. The array Af contains the
     * (possibly partial) QR-factorization of A, where the upper triangle of Af[0:k-1,0:k-1] is a
     * partial triangular factor, the entries below the diagonal in the first k columns are the
     * Householder vectors, and the rest of Af contains a partially updated matrix.
     * This function returns ||A*P - Q*R||/(||norm(A)||*eps*m)
     * Parameters: m: The number of rows of the matrices A and Af.
     *             n: The number of columns of the matrices A and Af.
     *             k: The number of columns of Af that have been reduced to upper triangular form.
     *             A: an array, dimension (lda, n)
     *                The original matrix A.
     *             Af: an array, dimension (lda,n)
     *                 The (possibly partial) output of dgeqpf. The upper triangle of Af(1:k,1:k)
     *                 is a partial triangular factor, the entries below the diagonal in the first
     *                 k columns are the Householder vectors, and the rest of Af contains a
     *                 partially updated matrix.
     *                 Af is modified but restored on exit
     *             lda: The leading dimension of the arrays A and Af.
     *             tau: an array, dimension (k)
     *                  Details of the Householder transformations as returned by dgeqpf.
     *             jpvt: an integer array, dimension (n)
     *                   Pivot information as returned by dgeqpf.
     *                   note: should contain zero-based indices
     *             work: an array, dimension (lwork)
     *             lwork: The length of the array work. lwork>=m*n+n.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    real dqpt01(int m, int n, int k, real const* A, real* Af, int lda, real
                       const* tau, int const* jpvt, real* work, int lwork)
    {
        // Test if there is enough workspace
        if (lwork<m*n+n)
        {
            xerbla("DQPT01", 10);
            return ZERO;
        }
        // Quick return if possible
        if (m<=0 || n<=0)
        {
            return ZERO;
        }
        int i, info, j, ldaj, jm;
        for (j=0; j<k; j++)
        {
            ldaj = lda*j;
            jm = j*m;
            for (i=0; i<=j && i<m; i++)
            {
                work[jm+i] = Af[i+ldaj];
            }
            for (i=j+1; i<m; i++)
            {
                work[jm+i] = ZERO;
            }
        }
        for (j=k; j<n; j++)
        {
            Blas<real>::dcopy(m, &Af[/*0+*/lda*j], 1, &work[j*m], 1);
        }
        this->dormqr("Left","No transpose", m, n, k, Af, lda, tau, work, m, &work[m*n], lwork-m*n,
                     info);
        for (j=0; j<n; j++)
        {
            // Compare i-th column of QR and jpvt[i]-th column of A
            Blas<real>::daxpy(m, -ONE, &A[/*0+*/lda*jpvt[j]], 1, &work[j*m], 1);
        }
        real rwork[1];
        real dqpt01 = this->dlange("One-norm", m, n, work, m, rwork) /
                      (real(std::max(m,n))*this->dlamch("Epsilon"));
        real norma = this->dlange("One-norm", m, n, A, lda, rwork);
        if (norma!=ZERO)
        {
            dqpt01 /= norma;
        }
        return dqpt01;
    }

    /* dqrt11 computes the test ratio
     *     || Q'*Q - I || / (eps * m)
     * where the orthogonal matrix Q is represented as a product of elementary transformations.
     * Each transformation has the form
     *     H(k) = I - tau[k] v(k) v(k)'
     * where v(k) is an m-vector of the form [0 ... 0 1 x(k)]', where x(k) is a vector of length
     * m-k stored in A[k+1:m-1,k].
     * Parameters: m: The number of rows of the matrix A.
     *             k: The number of columns of A whose subdiagonal entries contain information
     *                about orthogonal transformations.
     *             A: an array, dimension (lda,k)
     *                The (possibly partial) output of a QR reduction routine.
     *                A is modified but restored on exit
     *             lda: The leading dimension of the array A.
     *             tau: an array, dimension (k)
     *                  The scaling factors tau for the elementary transformations as computed by
     *                  the QR factorization routine.
     *             work: an array, dimension (lwork)
     *             lwork: The length of the array work. lwork >= m*m+m.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    real dqrt11(int m, int k, real* A, int lda, real const* tau, real* work,
                       int lwork)
    {
        // Test for sufficient workspace
        if (lwork<(m*m+m))
        {
            xerbla("DQRT11", 7);
            return ZERO;
        }
        // Quick return if possible
        if (m<=0)
        {
            return ZERO;
        }
        this->dlaset("Full", m, m, ZERO, ONE, work, m);
        int info;
        // Form Q
        this->dorm2r("Left", "No transpose", m, m, k, A, lda, tau, work, m, &work[m*m], info);
        // Form Q'*Q
        this->dorm2r("Left", "Transpose", m, m, k, A, lda, tau, work, m, &work[m*m], info);
        for (int j=0; j<m; j++)
        {
            work[j*m+j] -= ONE;
        }
        real rdummy[1];
        return this->dlange("One-norm", m, m, work, m, rdummy) / (real(m)*this->dlamch("Epsilon"));
    }

    /* dqrt12 computes the singular values 'svlues' of the upper trapezoid of A[0:m-1,0:n-1] and
     * returns the ratio
     *     || s - svlues||/(||svlues||*eps*max(m,n))
     * Parameters: m: The number of rows of the matrix A.
     *             n: The number of columns of the matrix A.
     *             A: an array, dimension (lda,n)
     *                The m-by-n matrix A. Only the upper trapezoid is referenced.
     *             lda: The leading dimension of the array A.
     *             s: an array, dimension (min(m,n))
     *                The singular values of the matrix A.
     *             work: an array, dimension (lwork)
     *             lwork: The length of the array work.
     *                    lwork >= max(m*n+4*min(m,n)+max(m,n), m*n+2*min(m,n)+4*n).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    real dqrt12(int m, int n, real const* A, int lda, real const* s, real* work, int lwork)
    {
        // Test that enough workspace is supplied
        int mn = std::min(m, n);
        int maxmn = std::max(m, n);
        int mtn = m*n;
        if (lwork<std::max(mtn+4*mn+maxmn, mtn+2*mn+4*n))
        {
            xerbla("DQRT12", 7);
            return ZERO;
        }
        // Quick return if possible
        if (mn<=ZERO)
        {
            return ZERO;
        }
        real nrmsvl = Blas<real>::dnrm2(mn, s, 1);
        // Copy upper triangle of A into work
        this->dlaset("Full", m, n, ZERO, ZERO, work, m);
        int i, mj, ldaj;
        for (int j=0; j<n; j++)
        {
            mj = m*j;
            ldaj = lda*j;
            for (i=0; i<=j && i<m; i++)
            {
                work[i+mj] = A[i+ldaj];
            }
        }
        // Get machine parameters
        real smlnum = this->dlamch("S") / this->dlamch("P");
        real bignum = ONE / smlnum;
        this->dlabad(smlnum, bignum);
        // Scale work if max entry outside range [SMLNUM,BIGNUM]
        real dummy[1];
        real anrm = this->dlange("M", m, n, work, m, dummy);
        int info;
        int iscl = 0;
        if (anrm>ZERO && anrm<smlnum)
        {
            // Scale matrix norm up to SMLNUM
            this->dlascl("G", 0, 0, anrm, smlnum, m, n, work, m, info);
            iscl = 1;
        }
        else if (anrm>bignum)
        {
            // Scale matrix norm down to BIGNUM
            this->dlascl("G", 0, 0, anrm, bignum, m, n, work, m, info);
            iscl = 1;
        }
        if (anrm!=ZERO)
        {
            // Compute SVD of work
            this->dgebd2(m, n, work, m, &work[mtn], &work[mtn+mn], &work[mtn+2*mn],
                         &work[mtn+3*mn], &work[mtn+4*mn], info);
            this->dbdsqr("Upper", mn, 0, 0, 0, &work[mtn], &work[mtn+mn], dummy, mn, dummy, 1,
                         dummy, mn, &work[mtn+2*mn], info);
            if (iscl==1)
            {
                if (anrm>bignum)
                {
                    this->dlascl("G", 0, 0, bignum, anrm, mn, 1, &work[mtn], mn, info);
                }
                if (anrm<smlnum)
                {
                    this->dlascl("G", 0, 0, smlnum, anrm, mn, 1, &work[mtn], mn, info);
                }
            }
        }
        else
        {
            for (i=0; i<mn; i++)
            {
                work[mtn+i] = ZERO;
            }
        }
        // Compare s and singular values of work
        Blas<real>::daxpy(mn, -ONE, s, 1, &work[mtn], 1);
        real dqrt12 = Blas<real>::dasum(mn, &work[mtn], 1) / (this->dlamch("Epsilon")*real(maxmn));
        if (nrmsvl!=ZERO)
        {
            dqrt12 /= nrmsvl;
        }
        return dqrt12;
    }

    /* icopy copies an integer vector x to an integer vector y.
     * Uses unrolled loops for increments equal to 1.
     * Parameters: n: The length of the vectors sx and sy.
     *             sx: an integer array, dimension (1+(n-1)*abs(incx))
     *                 The vector X.
     *             incx: The spacing between consecutive elements of sx.
     *             sy: an integer array, dimension (1+(n-1)*abs(incy))
     *                 The vector Y.
     *             incy: The spacing between consecutive elements of sy.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void icopy(int n, int const* sx, int incx, int* sy, int incy)
    {
        if (n<=0)
        {
            return;
        }
        int i;
        if (incx!=1 || incy!=1)
        {
            // Code for unequal increments or equal increments not equal to 1
            int ix = 0;
            int iy = 0;
            if (incx<0)
            {
                ix = (-n+1)*incx;
            }
            if (incy<0)
            {
                iy = (-n+1)*incy;
            }
            for (i=0; i<n; i++)
            {
                sy[iy] = sx[ix];
                ix += incx;
                iy += incy;
            }
            return;
        }
        // Code for both increments equal to 1
        // Clean-up loop
        int m = n % 7;
        if (m!=0)
        {
            for (i=0; i<m; i++)
            {
                sy[i] = sx[i];
            }
            if (n<7)
            {
                return;
            }
        }
        for (i=m; i<n; i+=7)
        {
            sy[i]   = sx[i];
            sy[i+1] = sx[i+1];
            sy[i+2] = sx[i+2];
            sy[i+3] = sx[i+3];
            sy[i+4] = sx[i+4];
            sy[i+5] = sx[i+5];
            sy[i+6] = sx[i+6];
        }
    }

    /* ilaenv returns problem-dependent parameters for the local environment. See ispec for a
     * description of the parameters.
     * In this version, the problem-dependent parameters are contained in the integer array iparms
     * in the struct claenv and the value with index ispec is copied to ilaenv. This version of
     * ilaenv is to be used in conjunction with xlaenv in testing and timing.
     * Paramters: ispec: Specifies the parameter to be returned as the value of ilaenv.
     *                   == 1: the optimal blocksize; if this value is 1, an unblocked algorithm
     *                         will give the best performance.
     *                   == 2: the minimum block size for which the block routine should be used;
     *                         if the usable block size is less than this value, an unblocked
     *                         routine should be used.
     *                   == 3: the crossover point (in a block routine, for N less than this value,
     *                         an unblocked routine should be used)
     *                   == 4: the number of shifts, used in the nonsymmetric eigenvalue routines
     *                   == 5: the minimum column dimension for blocking to be used; rectangular
     *                         blocks must have dimension at least k by m, where k is given by
     *                         ilaenv(2,...) and m by ilaenv(5,...)
     *                   == 6: the crossover point for the SVD (when reducing an m by n matrix to
     *                         bidiagonal form, if max(m,n)/min(m,n) exceeds this value, a QR
     *                         factorization is used first to reduce the matrix to a triangular
     *                         form.)
     *                   == 7: the number of processors
     *                   == 8: the crossover point for the multishift QR and QZ methods for
     *                         nonsymmetric eigenvalue problems.
     *                   == 9: maximum size of the subproblems at the bottom of the computation
     *                         tree in the divide-and-conquer algorithm
     *                   ==10: ieee NaN arithmetic can be trusted not to trap
     *                   ==11: infinity arithmetic can be trusted not to trap
     *                   Other specifications (up to 100) can be added later.
     *             name: The name of the calling subroutine.
     *             opts: The character options to the subroutine name, concatenated into a single
     *                   character string. For example, UPLO=='U', TRANS=='T', and DIAG=='N' for a
     *                   triangular routine would be specified as opts = 'UTN'.
     *             N1,
     *             N2,
     *             N3,
     *             N4: Problem dimensions for the subroutine name; these may not all be required.
     * Returns: >= 0: the value of the parameter specified by ispec
     *           < 0: if ==-k, the k-th argument had an illegal value.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date November 2017
     * Further Details:
     *     The following conventions have been used when calling ilaenv from the lapack routines:
     *     1) opts is a concatenation of all of the character options to subroutine name, in the
     *        same order that they appear in the argument list for name, even if they are not used
     *        in determining the value of the parameter specified by ispec.
     *     2) The problem dimensions N1, N2, N3, N4 are specified in the order that they appear in
     *        the argument list for name. N1 is used first, N2 second, and so on, and unused
     *        problem dimensions are passed a value of -1.
     *     3) The parameter value returned by ilaenv is checked for validity in the calling
     *        subroutine. For example, ilaenv is used to retrieve the optimal blocksize for STRTRI
     *        as follows:
     *            NB = ilaenv(1, 'STRTRI', UPLO // DIAG, N, -1, -1, -1 )
     *            if (NB<=1) NB = MAX(1, N)                                                      */
    virtual int ilaenv(int ispec, char const* name, char const* opts, int n1, int n2, int n3,
                       int n4)
    {
        if (ispec>=1 && ispec<=5)
        {
            // Return a value from the common block.
            char upname[4];
            upname[0] = std::toupper(name[1]);
            upname[1] = std::toupper(name[2]);
            upname[2] = std::toupper(name[3]);
            upname[3] = std::toupper(name[4]);
            if (std::strncmp(upname,"GEQR",4)==0)
            {
                if (n3==2)
                {
                    return claenv.iparms[1];
                }
                else
                {
                    return claenv.iparms[0];
                }
            }
            else if (std::strncmp(upname,"GELQ",4)==0)
            {
                if (n3==2)
                {
                    return claenv.iparms[1];
                }
                else
                {
                    return claenv.iparms[0];
                }
            }
            else
            {
                return claenv.iparms[ispec-1];
            }
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
            //return 0;
            return this->ieeeck(1, ZERO, ONE);
        }
        else if (ispec==11)
        {
            // Infinity arithmetic can be trusted not to trap
            // return 0;
            return this->ieeeck(0, ZERO, ONE);
        }
        else
        {
            // Invalid value for ISPEC
            return -1;
        }
    }

    /* This is a special version of xerbla to be used only as part of the test program for testing
     * error exits from the LAPACK routines. Error messages are printed if info!=infoc.info or if
     * srname!=snramc.snram.
     * Parameters: srname: The name of the subroutine calling xerbla. This name should match the
     *                     struct field snramc.snram.
     *             info: The error return code from the calling subroutine. info should equal the
     *                   struct field infoc.info.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Further Details:
     *     The following variables are passed via the struct fields infoc and srnamc:
     *     infoc.info   int           Expected integer return code
     *     infoc.iounit std::ostream& Unit number for printing error messages
     *     infoc.ok     bool          Set to true if info==infoc.info and srname==snramc.srnam,
     *                                otherwise set to false
     *     infoc.lerr   bool          Set to true, indicating that XERBLA was called
     *     snramc.srnam char*         Expected name of calling subroutine                              */
    virtual void xerbla(const char* srname, int info)
    {
        infoc.lerr = true;
        if (info!=infoc.info)
        {
            if (infoc.info!=0)
            {
                infoc.iounit << " *** XERBLA was called from " << srnamc.srnam << " with INFO = "
                             << info << " instead of " << infoc.info << " ***" << std::endl;
            }
            else
            {
                infoc.iounit << " *** On entry to " << srname << " parameter number " << info
                             << " had an illegal value ***" << std::endl;
            }
            infoc.ok = false;
        }
        if (std::strcmp(srname,srnamc.srnam)!=0)
        {
            infoc.iounit << " *** XERBLA was called with SRNAME = " << srname << " instead of "
                         << std::setw(9) << srnamc.srnam << " ***" << std::endl;
            infoc.ok = false;
        }
    }

    /* xlaenv sets certain machine- and problem-dependent quantities which will later be retrieved
     * by ilaenv.
     * Parameters: ispec: Specifies the parameter to be set in the field array iparms.
     *                    ==1: the optimal blocksize; if this value is 1, an unblocked algorithm
     *                         will give the best performance.
     *                    ==2: the minimum block size for which the block routine should be used;
     *                         if the usable block size is less than this value, an unblocked
     *                         routine should be used.
     *                    ==3: the crossover point (in a block routine, for N less than this value,
     *                         an unblocked routine should be used)
     *                    ==4: the number of shifts, used in the nonsymmetric eigenvalue routines
     *                    ==5: the minimum column dimension for blocking to be used; rectangular
     *                         blocks must have dimension at least k by m, where k is given by
     *                         ilaenv(2,...) and m by ilaenv(5,...)
     *                    ==6: the crossover point for the SVD (when reducing an m by n matrix to
     *                         bidiagonal form, if max(m,n)/min(m,n) exceeds this value, a QR
     *                         factorization is used first to reduce the matrix to a triangular
     *                         form)
     *                    ==7: the number of processors
     *                    ==8: another crossover point, for the multishift QR and QZ methods for
     *                         nonsymmetric eigenvalue problems.
     *                    ==9: maximum size of the subproblems at the bottom of the computation
     *                         tree in the divide-and-conquer algorithm (used by xGELSD and xGESDD)
     *                   ==10: ieee NaN arithmetic can be trusted not to trap
     *                   ==11: infinity arithmetic can be trusted not to trap
     *             nvalue: The value of the parameter specified by ispec.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    void xlaenv(int ispec, int nvalue)
    {
        if (ispec>=1 && ispec<=9)
        {
            claenv.iparms[ispec-1] = nvalue;
        }
    }
};
#endif