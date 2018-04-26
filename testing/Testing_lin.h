#ifndef TESTING_LIN_HEADER
#define TESTING_LIN_HEADER

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cmath>

#include "Blas.h"
#include "Lapack_dyn.h"
#include "Testing_matgen.h"

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
        else if (std::strncmp(p2, "PS", 2)==0)
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
            lda = ((1>m) ? 1 : m);
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
                lwork = m*temp + 4*mnmin + temp;
                temp = m*n + 2*mnmin + 4*n;
                if (lwork<temp)
                {
                    lwork = temp;
                }
                if (lwork<1)
                {
                    lwork = 1;
                }
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
                        iwork[i] = 0;
                    }
                    if (imode==0)
                    {
                        dlaset("Full", m, n, ZERO, ZERO, CopyA, lda);
                        for (i=0; i<mnmin; i++)
                        {
                            S[i] = ZERO;
                        }
                    }
                    else
                    {
                        Testing_matgen<real>::dlatms(m, n, "Uniform", iseed, "Nonsymm", S, mode,
                                                     ONE/eps, ONE, m, n, "No packing", CopyA, lda,
                                                     work, info);
                        if (imode>=3)
                        {
                            if (imode==3)
                            {
                                ilow = 0;
                                istep = 1;
                                ihigh = n / 2;
                                if (ihigh<1)
                                {
                                    ihigh = 1;
                                }
                            }
                            else if (imode==4)
                            {
                                ilow = n / 2 - 1;
                                if (ilow<0)
                                {
                                    ilow = 0;
                                }
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
                                iwork[i] = 1;
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
                        dlacpy("All", m, n, CopyA, lda, A, lda);
                        icopy(n, &iwork[0], 1, &iwork[n], 1);
                        // Compute the QR factorization with pivoting of A
                        lw = 2*n + nb*(n+1);
                        if (lw<1)
                        {
                            lw = 1;
                        }
                        // Compute the QP3 factorization of A
                        std::strncpy(srnamc.srnam,"DGEQP3", 7);
                        dgeqp3(m, n, A, lda, &iwork[n], tau, work, lw, info);
                        // Compute norm(svd(A) - svd(R))
                        result[0] = dqrt12(m, n, A, lda, S, work, lwork);
                        // Compute norm(A*P - Q*R)
                        result[1] = dqpt01(m, n, mnmin, CopyA, A, lda, tau, &iwork[n], work,
                                           lwork);
                        // Compute Q'*Q
                        result[2] = dqrt11(m, mnmin, A, lda, tau, work, lwork);
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
    real dqpt01(int m, int n, int k, real const* A, real const* Af, int lda, real
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
        dormqr("Left","No transpose", m, n, k, Af, lda, tau, work, m, &work[m*n], lwork-m*n, info);
        for (j=0; j<n; j++)
        {
            // Compare i-th column of QR and jpvt[i]-th column of A
            Blas<real>::daxpy(m, -ONE, A[/*0+*/lda*jpvt[j]], 1, work[j*m], 1);
        }
        real rwork[1];
        real dqpt01 = dlange("One-norm", m, n, work, m, rwork) / (real(m>n?m:n)*this->dlamch("Epsilon"));
        real norma = dlange("One-norm", m, n, A, lda, rwork);
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
    real dqrt11(int m, int k, real const* A, int lda, real const* tau, real* work,
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
        real rdummy[1];
        return dlange("One-norm", m, m, work, m, rdummy) / (real(m)*this->dlamch("Epsilon"));
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
        real nrmsvl = Blas<real>::dnrm2(mn, s, 1);
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
        real smlnum = this->dlamch("S") / this->dlamch("P");
        real bignum = ONE / smlnum;
        dlabad(smlnum, bignum);
        // Scale work if max entry outside range [SMLNUM,BIGNUM]
        real dummy[1];
        real anrm = dlange("M", m, n, work, m, dummy);
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
        Blas<real>::daxpy(mn, -ONE, s, 1, &work[mtn], 1);
        real dqrt12 = Blas<real>::dasum(mn, &work[mtn], 1) / (this->dlamch("Epsilon")*real(maxmn));
        if (nrmsvl!=ZERO)
        {
            dqrt12 = dqrt12 / nrmsvl;
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
            return int(real((n1<n2?n1:n2))*real(1.6E0));
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
            return ieeeck(1, ZERO, ONE);
        }
        else if (ispec==11)
        {
            // Infinity arithmetic can be trusted not to trap
            // return 0;
            return ieeeck(0, ZERO, ONE);
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
     * Parameters: ispec: Specifies the parameter to be set in the field array IPARMS.
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