#include <limits>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cmath>

#include "Blas.h"

#ifndef TESTING_LIN_HEADER
#define TESTING_LIN_HEADER

template<class T>
class Testing_lin
{
public:
    // constants

    static constexpr T ZERO = T(0.0);
    static constexpr T ONE  = T(1.0);

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
    static void alahd(std::ostream& iounit, char const* path)
    {
        if (!iounit.good())
        {
            return;
        }
        char pathcopy[4], c1, c3, p2[2], *sym;
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
        if (std::strncmpi(p2, "GE", 2)==0)
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
        else if (std::strncmpi(p2, "GB", 2)==0)
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
        else if (std::strncmpi(p2, "GT", 2)==0)
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
        else if (std::strncmpi(p2, "PO", 2)==0 || std::strncmpi(p2, "PP", 2)==0)
        {
            // PO: Positive definite full
            // PP: Positive definite packed
            if(sord)
            {
                sym = "Symmetric";
            }
            else
            {
                sym = "Hermitian";
            }
            if (c3='O')
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
        else if (std::strncmpi(p2, "PS", 2)==0)
        {
            // PS: Positive semi-definite full
            if (sord)
            {
                sym = "Symmetric";
            }
            else
            {
                sym = "Hermitian";
            }
            char* eigcnm;
            if (c1=='S' || c1=='C')
            {
                eigcnm = "1E04";
            }
            else
            {
                eigcnm = "1D12";
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
        else if (std::strncmpi(p2, "PB", 2)==0)
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
        else if (std::strncmpi(p2, "PT", 2)==0)
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
        else if (std::strncmpi(p2, "SY", 2)==0)
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
        else if (std::strncmpi(p2, "SR", 2)==0 || std::strncmpi(p2, "SK", 2)==0)
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
        else if (std::strncmpi(p2, "SP", 2)==0)
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
        else if (std::strncmpi(p2, "HA", 2)==0)
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
        else if (std::strncmpi(p2, "HE", 2)==0)
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
        else if (std::strncmpi(p2, "HR", 2)==0)
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
        else if (std::strncmpi(p2, "HP", 2)==0)
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
        else if (std::strncmpi(p2, "TR", 2)==0 || std::strncmpi(p2, "TP", 2)==0)
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
        else if (std::strncmpi(p2, "TB", 2)==0)
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
        else if (std::strncmpi(p2, "QR", 2)==0)
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
        else if (std::strncmpi(p2, "LQ", 2)==0)
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
        else if (std::strncmpi(p2, "QL", 2)==0)
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
        else if (std::strncmpi(p2, "RQ", 2)==0)
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
        else if (std::strncmpi(p2, "QP", 2)==0)
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
        else if (std::strncmpi(p2, "TZ", 2)==0)
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
        else if (std::strncmpi(p2, "LS", 2)==0)
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
                      "       is in the row space of A or A' (overdetermined case)\n"
            iounit << "    3: norm(svd(A)-svd(R)) / ( min(M,N) * norm(svd(R)) * EPS )\n";
            iounit << "    4" << str9935 << '\n';
            iounit << "    5: norm( (A*X-B)' *A ) / ( max(M,N,NRHS) * norm(A) * norm(B) * EPS )\n";
            iounit << "    6: Check if X is in the row space of A or A'\n";
            iounit << "    7-10: same as 3-6    11-14: same as 3-6\n";
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmpi(p2, "LU", 2)==0)
        {
            // LU factorization variants
            iounit << "\n " << pathcopy << ":  LU factorization variants\n";
            iounit << " Matrix types:\n";
            iounit << str9979 << '\n';
            iounit << " Test ratio:\n";
            iounit << "    1" << str9962 << '\n';
            iounit << " Messages:" << std::endl;
        }
        else if (std::strncmpi(p2, "CH", 2)==0)
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
        else if (std::strncmpi(p2, "QS", 2)==0)
        {
            // QR factorization variants
            iounit << "\n " << pathcopy << ":  QR factorization variants\n";
            iounit << " Matrix types:\n";
            iounit << str9970 << '\n';
            iounit << " Test ratios:" << std::endl;
        }
        else if (std::strncmpi(p2, "QT", 2)==0)
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
        else if (std::strncmpi(p2, "QX", 2)==0)
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
        else if (std::strncmpi(p2, "TQ", 2)==0)
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
        else if (std::strncmpi(p2, "XQ", 2)==0)
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
        else if (std::strncmpi(p2, "TS", 2)==0)
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
    static void alasum(char const* type, std::ostream& nout, int nfail, int nrun, int nerrs)
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
     *                     output file if RESULT>=thresh. To have every test ratio printed, use
     *                     thresh=0.
     *             A: an array, dimension (MMAX*NMAX) where MMAX is the maximum value of M in mval
     *                and NMAX is the maximum value of N in nval.
     *             CopyA: an array, dimension (MMAX*NMAX)
     *             S: an array, dimension min(MMAX,NMAX))
     *             tau: an array, dimension (MMAX)
     *             WORK: an array, dimension (MMAX*NMAX + 4*NMAX + MMAX)
     *             IWORK: an integer array, dimension (2*NMAX)
     *             NOUT: The output stream.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dchkq3(bool const* dotype, int nm, int const* mval, int nn, int const* nval,
                       int nnb, int const* nbval, int const* nxval, T thresh, T* A, T* CopyA, T* S,
                       T* TAU, T* WORK, int* IWORK, std::ostream& NOUT)
    {
        const int NTYPES = 6;
        const int NTESTS = 3;
        char PATH[3] = {'D','Q','3'};
        int I, IHIGH, ILOW, IM, IMODE, IN, INB, INFO, ISTEP, K, LDA, LW, LWORK, M, MNMIN, MODE, N,
            NB, NERRS, NFAIL, NRUN, NX;
        T EPS;
        int ISEED[4];
        int ISEEDY[4] = {1988, 1989, 1990, 1991};
        T RESULT[NTESTS];
        // Scalars in Common (TODO: pass as argument?)
        bool LERR, OK;
        char SRNAMT[32];
        int INFOT, IOUNIT;
        // Common blocks (TODO: remove)
        //COMMON             / INFOC / INFOT, IOUNIT, OK, LERR
        //COMMON             / SRNAMC / SRNAMT
        // Initialize constants and the random number seed.
        NRUN = 0;
        NFAIL = 0;
        NERRS = 0;
        for (I=1; I<=4; I++)
        {
            ISEED(I) = ISEEDY(I);
        }
        EPS = dlamch("Epsilon");
        INFOT = 0;
        for (IM=1; IM<=nm; IM++)//90
        {
            // Do for each value of M in mval.
            M = mval[IM-1];
            LDA = ((1>M) ? 1 : M);
            for (IN=1; IN<=nn; IN++)//80
            {
                // Do for each value of N in nval.
                N = nval[IN-1];
                MNMIN = ((M<N) ? M : N);
                LWORK = ((M>N) ? M : N);
                LWORK = M*LWORK + 4*MNMIN + LWORK;
                if (LWORK<(M*N + 2*MNMIN + 4*N))
                {
                    LWORK = M*N + 2*MNMIN + 4*N;
                }
                if (LWORK<1)
                {
                    LWORK = 1;
                }
                for (IMODE=1; IMODE<=NTYPES; IMODE++)//70
                {
                    if (!dotype[IMODE-1])
                    {
                        continue;
                    }
                    // Do for each type of matrix
                    //   1:  zero matrix
                    //   2:  one small singular value
                    //   3:  geometric distribution of singular values
                    //   4:  first n/2 columns fixed
                    //   5:  last n/2 columns fixed
                    //   6:  every second column fixed
                    MODE = IMODE;
                    if (IMODE>3)
                    {
                        MODE = 1;
                    }
                    // Generate test matrix of size m by n using singular value distribution
                    // indicated by 'mode'.
                    for (I=1; I<=N; I++)
                    {
                        IWORK(I) = 0;
                    }
                    if (IMODE==1){
                        dlaset("Full", M, N, ZERO, ZERO, CopyA, LDA);
                        for (I=1; I<=MNMIN; I++)
                        {
                            S[I-1] = ZERO;
                        }
                    }
                    else
                    {
                        dlatms(M, N, "Uniform", ISEED, "Nonsymm", S, MODE, ONE / EPS, ONE, M, N,
                               "No packing", CopyA, LDA, WORK, INFO);
                        if (IMODE>=4)
                        {
                            if (IMODE==4)
                            {
                                ILOW = 1;
                                ISTEP = 1;
                                IHIGH = N / 2;
                                if (IHIGH<1)
                                {
                                    IHIGH = 1;
                                }
                            }
                            else if (IMODE==5)
                            {
                                ILOW = N / 2;
                                if (ILOW<1)
                                {
                                    ILOW = 1;
                                }
                                ISTEP = 1;
                                IHIGH = N;
                            }
                            else if (IMODE==6)
                            {
                               ILOW = 1;
                               ISTEP = 2;
                               IHIGH = N;
                            }
                            for (I=ILOW; I<=IHIGH; I+=ISTEP)
                            {
                                IWORK(I) = 1;
                            }
                        }
                        dlaord("Decreasing", MNMIN, S, 1);
                    }
                    for (INB=1; INB<=nnb; INB++)//60
                    {
                        // Do for each pair of values (NB,NX) in nbval and nxval.
                        NB = nbval[INB-1];
                        //xlaenv(1, NB);
                        NX = nxval[INB-1];
                        //xlaenv(3, NX);
                        // Get a working copy of CopyA into A and a copy of vector IWORK.
                        dlacpy("All", M, N, CopyA, LDA, A, LDA);
                        icopy(N, &IWORK[0], 1, &IWORK[N], 1);
                        // Compute the QR factorization with pivoting of A
                        LW = 2*N + NB*(N+1);
                        if (LW<1)
                        {
                            LW = 1;
                        }
                        // Compute the QP3 factorization of A
                        std::strncpy(SRNAMT,"DGEQP3", 7);
                        dgeqp3(M, N, A, LDA, &IWORK[N], TAU, WORK, LW, INFO);
                        // Compute norm(svd(A) - svd(R))
                        RESULT[0] = dqrt12(M, N, A, LDA, S, WORK, LWORK);
                        // Compute norm(A*P - Q*R)
                        RESULT[1] = dqpt01(M, N, MNMIN, CopyA, A, LDA, TAU, &IWORK[N], WORK,
                                           LWORK);
                        // Compute Q'*Q
                        RESULT[2] = dqrt11(M, MNMIN, A, LDA, TAU, WORK, LWORK);
                        // Print information about the tests that did not pass the threshold.
                        for (K=1; K<=NTESTS; K++)
                        {
                            if (RESULT(K)>=thresh)
                            {
                                if (NFAIL==0 && NERRS==0)
                                {
                                    alahd(NOUT, PATH);
                                }
                                NOUT << " DGEQP3 M =" << std::setw(5) << M << ", N ="
                                     << std::setw(5) << N << ", NB =" << std::setw(4) << NB
                                     << ", type " << std::setw(2) << IMODE << ", test "
                                     << std::setw(2) << K << ", ratio =" << std::setw(12)
                                     << std::setprecision(5) << RESULT(K) << std::endl;
                                NFAIL++;
                            }
                        }
                        NRUN += NTESTS;
                    }
                }
            }
        }
        // Print a summary of the results.
        alasum(PATH, NOUT, NFAIL, NRUN, NERRS);
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
    static void dlaord(char const* job, int n, T* x, int incx)
    {
        int i, ix, ixnext;
        T temp;
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
    static T dqpt01(int m, int n, int k, T const* A, T const* Af, int lda, T const* tau,
                    int const* jpvt, T* work, int lwork)
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
            dcopy(m, &Af[/*0+*/lda*j], 1, &work[j*m], 1);
        }
        dormqr("Left","No transpose", m, n, k, Af, lda, tau, work, m, &work[m*n], lwork-m*n, info);
        for (j=0; j<n; j++)
        {
            // Compare i-th column of QR and jpvt[i]-th column of A
            daxpy(m, -ONE, A[/*0+*/lda*jpvt[j]], 1, work[j*m], 1);
        }
        T rwork[1];
        T dqpt01 = dlange("One-norm", m, n, work, m, rwork) / (T(m>n?m:n)*dlamch("Epsilon"));
        T norma = dlange("One-norm", m, n, A, lda, rwork);
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
    static T dqrt11(int m, int k, T const* A, int lda, T const* tau, T* work, int lwork)
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
        dlaset("Full", m, m, ZERO, ONE, work, m);
        int info;
        // Form Q
        dorm2r("Left", "No transpose", m, m, k, A, lda, tau, work, m, &work[m*m], info);
        // Form Q'*Q
        dorm2r("Left", "Transpose", m, m, k, A, lda, tau, work, m, &work[m*m], info);
        for (int j=0; j<m; j++)
        {
            work[j*m+j] -= ONE;
        }
        T rdummy[1];
        return dlange("One-norm", m, m, work, m, rdummy) / (T(m)*DLAMCH("Epsilon"));
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
    static T dqrt12(int m, int n, T const* A, int lda, T const* s, T* work, int lwork)
    {
        // Test that enough workspace is supplied
        int mn = m<n?m:n;
        int maxmn = m>n?m:n;
        int mtn = m*n;
        if (lwork<mtn+4*mn+maxmn || lwork<mtn+2*mn+4*n)
        {
            xerbla("DQRT12", 7);
            return ZERO;
        }
        // Quick return if possible
        if (mn<=ZERO)
        {
            return ZERO;
        }
        T nrmsvl = dnrm2(mn, s, 1);
        // Copy upper triangle of A into work
        dlaset("Full", m, n, ZERO, ZERO, work, m);
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
        T smlnum = dlamch("S") / dlamch("P");
        T bignum = ONE / smlnum;
        dlabad(smlnum, bignum);
        // Scale work if max entry outside range [SMLNUM,BIGNUM]
        T dummy[1];
        T anrm = dlange("M", m, n, work, m, dummy);
        int info;
        int iscl = 0;
        if (anrm>ZERO && anrm<smlnum)
        {
            // Scale matrix norm up to SMLNUM
            dlascl("G", 0, 0, anrm, smlnum, m, n, work, m, info);
            iscl = 1;
        }
        else if (anrm>bignum)
        {
            // Scale matrix norm down to BIGNUM
            dlascl("G", 0, 0, anrm, bignum, m, n, work, m, info);
            iscl = 1;
        }
        if (anrm!=ZERO)
        {
            // Compute SVD of work
            dgebd2(m, n, work, m, &work[mtn], &work[mtn+mn], &work[mtn+2*mn], &work[mtn+3*mn],
                   &work[mtn+4*mn], info);
            dbdsqr("Upper", mn, 0, 0, 0, &work[mtn], &work[mtn+mn], dummy, mn, dummy, 1, dummy, mn,
                   &work[mtn+2*mn], info);
            if (iscl==1)
            {
                if (anrm>bignum)
                {
                    dlascl("G", 0, 0, bignum, anrm, mn, 1, &work[mtn], mn, info);
                }
                if (anrm<smlnum)
                {
                    dlascl("G", 0, 0, smlnum, anrm, mn, 1, &work[mtn], mn, info);
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
        daxpy(mn, -ONE, s, 1, &work[mtn], 1);
        T dqrt12 = dasum(mn, &work[mtn], 1) / (dlamch("Epsilon")*T(maxmn));
        if (nrmsvl!=ZERO)
        {
            dqrt12 = dqrt12 / NRMSVL;
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
    static void icopy(int n, int const* sx, int incx, int* sy, int incy)
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
}
#endif