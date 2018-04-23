#include <limits>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <cmath>

#ifndef LAPACK_HEADER
#define LAPACK_HEADER

template<class T>
class Lapack
{
public:
    // constants

    static constexpr T ZERO = T(0.0);
    static constexpr T ONE  = T(1.0);
    static constexpr T TWO  = T(2.0);

    // BLAS SRC (alphabetically)

    /* dasum takes the sum of the absolute values.
     * Parameters: n: number of elements in input vector(s)
     *             dx: an array, dimension (1+(n-1)*abs(incx))
     *             incx: storage spacing between elements of dx
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date November 2017                                                                        */
    static T dasum(int n, T const* dx, int incx)
    {
        if (n<=0 || incx<=0)
        {
            return ZERO;
        }
        int i;
        T dtemp = ZERO;
        if (incx==1)
        {
            // code for increment equal to 1
            // clean-up loop
            int m = n%6;
            if (m!=0)
            {
                for (i=0; i<m; i++)
                {
                    dtemp += fabs(dx[i]);
                }
                if (n<6)
                {
                    return dtemp;
                }
            }
            for (i=m; i<n; i+=6)
            {
                dtemp += fabs(dx[i])+fabs(dx[i+1])+fabs(dx[i+2])
                        +fabs(dx[i+3]))+fabs(dx[i+4])+fabs(dx[i+5]);
            }
        }
        else
        {
            // code for increment not equal to 1
            int nincx = n*incx;
            for (i=0; i<nincx; i+=incx)
            {
                dtemp += fabs(dx[i]);
            }
        }
        return dtemp;
    }

    /* daxpy constant times a vector plus a vector.
     * Uses unrolled loops for increments equal to one.
     * Parameters: n: number of elements in input vector(s)
     *             da: On entry, da specifies the scalar alpha.
     *             dx: an array, dimension (1 + (n-1)*abs(incx))
     *             incx: storage spacing between elements of dx
     *             dy: an array, dimension (1 + (n-1)*abs(incy))
     *             incy: storage spacing between elements of dy
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: November 2017                                                                       */
    static void daxpy(int n, T da, T const* dx, int incx, T* dy, int incy)
    {
        if (n<=0)
        {
            return;
        }
        if (da==ZERO)
        {
            return;
        }
        int i;
        if (incx==1 && incy==1)
        {
            // code for both increments equal to 1
            // clean-up loop
            int m = n%4;
            if (m!=0)
            {
                for (i=0; i<m; i++)
                {
                   dy[i] = dy[i] + da*dx[i];
                }
            }
            if (n<4)
            {
                return;
            }
            for (i=m; i<n; i+=4)
            {
                dy[i]   += da*dx[i];
                dy[i+1] += da*dx[i+1];
                dy[i+2] += da*dx[i+2];
                dy[i+3] += da*dx[i+3];
            }
        }
        else
        {
            // code for unequal increments or equal increments not equal to 1
            int ix = 0;
            int iy = 0;
            if (incx<0)
            {
                ix = (1-n)*incx;
            }
            if (incy<0)
            {
                iy = (1-n)*incy;
            }
            for (i=0; i<n; i++)
            {
                dy[iy] += da*dx[ix];
                ix += incx;
                iy += incy;
            }
        }
    }

    /* dcopy copies a vector, x, to a vector, y.
     * uses unrolled loops for increments equal to one.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                   */
    static void dcopy(int n, T const* dx, int incx, T* dy, int incy)
    {
        if (n < 0)
        {
            return;
        }
        int i;
        if (incx == 1 && incy == 1)
        {
            // code for both increments equal to 1
            // clean-up loop
            int m = n % 7;
            if (m != 0)
            {
                for (i = 0; i < m; i++)
                {
                    dy[i] = dx[i];
                }
                if (n < 7)
                {
                    return;
                }
            }
            for (i = m; i < n; i += 7)
            {
                dy[i] = dx[i];
                dy[i + 1] = dx[i + 1];
                dy[i + 2] = dx[i + 2];
                dy[i + 3] = dx[i + 3];
                dy[i + 4] = dx[i + 4];
                dy[i + 5] = dx[i + 5];
                dy[i + 6] = dx[i + 6];
            }
        } else
        {
            // code for unequal increments or equal increments not equal to 1
            int ix = 0;
            int iy = 0;
            if (incx < 0)
            {
                ix = (-n + 1) * incx;
            }
            if (incy < 0)
            {
                iy = (-n + 1) * incy;
            }
            for (i = 0; i < n; i++)
            {
                dy[iy] = dx[ix];
                ix += incx;
                iy += incy;
            }
        }
    }

    /* ddot forms the dot product of two vectors.
     * uses unrolled loops for increments equal to one.
     * Parameters: n: number of elements in input vector(s)
     *             dx: an array, dimension (1 + (n-1)*abs(incx))
     *             incx: storage spacing between elements of dx
     *             dy: an array, dimension (1 + (n-1)*abs(incy))
     *             incy: storage spacing between elements of dy
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: November 2017                                                                       */
    static T ddot(int n, T const* dx, int incx, T const* dy, int incy)
    {
        if (n<=0)
        {
            return ZERO;
        }
        int i;
        T dtemp = ZERO;
        if (incx==1 && incy==1)
        {
            // code for both increments equal to 1
            // clean-up loop
            int m = n % 5;
            if (m!=0)
            {
                for (i=0; i<m; i++)
                {
                    dtemp += dx[i]*dy[i];
                }
                if (n<5)
                {
                    return dtemp;
                }
            }
            for (i=m; i<n; i+=5)
            {
                dtemp += dx[i]*dy[i] + dx[i+1]*dy[i+1] + dx[i+2]*dy[i+2]
                                     + dx[i+3]*dy[i+3] + dx[i+4]*dy[i+4];
            }
        }
        else
        {
            // code for unequal increments or equal increments not equal to 1
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
                dtemp += dx[ix]*dy[iy];
                ix += incx;
                iy += incy;
            }
        }
        return dtemp;
    }

    /* dgemm performs one of the matrix-matrix operations
     *     C : = alpha*op(A)*op(B) + beta*C,
     * where op(X) is one of op(X) = X or op(X) = X^T,
     * alpha and beta are scalars, and A, B and C are matrices,
     * with op(A) an m by k matrix, op(B) a k by n matrix and C an m by n matrix.
     * Parameters: TRANSA: On entry, TRANSA specifies the form of op(A) to be used
     *                     in the matrix multiplication as follows:
     *                         TRANSA = 'N' or 'n', op(A) = A.
     *                         TRANSA = 'T' or 't', op(A) = A^T.
     *                         TRANSA = 'C' or 'c', op(A) = A^T.
     *             TRANSB: On entry, TRANSB specifies the form of op(B) to be used
     *                     in the matrix multiplication as follows:
     *                         TRANSB = 'N' or 'n', op(B) = B.
     *                         TRANSB = 'T' or 't', op(B) = B^T.
     *                         TRANSB = 'C' or 'c', op(B) = B^T.
     *             m: On entry, m specifies the number of rows of the matrix op(A)
     *                and of the matrix C. m must be at least zero.
     *             n: On entry, n specifies the number of columns of the matrix op(B)
     *                and the number of columns of the matrix C. n must be at least zero.
     *             k: On entry, k specifies the number of columns of the matrix op(A)
     *                and the number of rows of the matrix op(B). k must be at least zero.
     *             alpha: On entry, alpha specifies the scalar alpha.
     *             A: an array of DIMENSION(lda, ka), where ka is k when TRANSA = 'N' or 'n',
     *                and is m otherwise. Before entry with TRANSA = 'N' or 'n',
     *                the leading m by k part of the array A must contain the matrix A,
     *                otherwise the leading k by m part of the array A must contain the matrix A.
     *             lda: On entry, lda specifies the first dimension of A as declared in the
     *                  calling (sub)program. When TRANSA = 'N' or 'n' then lda must be
     *                  at least max(1, m), otherwise lda must be at least max(1, k).
     *             B: an array of DIMENSION(ldb, kb), where kb is n when TRANSB = 'N' or 'n',
     *                and is k otherwise. Before entry with TRANSB = 'N' or 'n',
     *                the leading k by n part of the array B must contain the matrix B,
     *                otherwise the leading n by k part of the array B must contain the matrix B.
     *             ldb: On entry, ldb specifies the first dimension of B as declared in the
     *                  calling (sub)program. When TRANSB = 'N' or 'n' then ldb must be
     *                  at least max(1, k), otherwise ldb must be at least max(1, n).
     *             beta: On entry, beta specifies the scalar beta.
     *                   When beta is supplied as zero then C need not be set on input.
     *             C: an array of DIMENSION(ldc, n).
     *                Before entry, the leading m by n part of the array C must contain the
     *                matrix C, except when beta is zero, in which case C need not be set on entry.
     *                On exit, the array C is overwritten by the m by n matrix (alpha*op(A)*op(B) + beta*C).
     *             ldc: On entry, ldc specifies the first dimension of C as declared in the
     *                  calling (sub)program. ldc must be at least max(1, m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void dgemm(char const* transa, char const* transb, int m, int n, int k, T alpha, T const* A, int lda, T const* B, int ldb, T beta, T* C, int ldc)
    {
        // Set nota and notb as true if A and B respectively are not transposed and set nrowa
        // and nrowb as the number of rows and columns of A and the number of rows of B respectively.
        char uptransa = toupper(transa[0]);
        char uptransb = toupper(transb[0]);
        bool nota = (uptransa == 'N');
        bool notb = (uptransb == 'N');
        int nrowa, nrowb;
        if (nota)
        {
            nrowa = m;
        } else
        {
            nrowa = k;
        }
        if (notb)
        {
            nrowb = k;
        } else
        {
            nrowb = n;
        }
        // Test the input parameters.
        int info = 0;
        if (!nota && uptransa != 'C' && uptransa != 'T')
        {
            info = 1;
        } else if (!notb && uptransb != 'C' && uptransb != 'T')
        {
            info = 2;
        } else if (m < 0)
        {
            info = 3;
        } else if (n < 0)
        {
            info = 4;
        } else if (k < 0)
        {
            info = 5;
        } else if (lda < (1 > nrowa ? 1 : nrowa))
        {
            info = 8;
        } else if (ldb < (1 > nrowb ? 1 : nrowb))
        {
            info = 10;
        } else if (ldc < (1 > m ? 1 : m))
        {
            info = 13;
        }
        if (info != 0)
        {
            xerbla("DGEMM", info);
            return;
        }
        // Quick return if possible.
        if (m == 0 || n == 0 || ((alpha == ZERO || k == 0) && beta == ONE))
        {
            return;
        }
        // And if alpha==zero.
        int i, j, l;
        int acol, bcol, ccol;
        if (alpha == ZERO)
        {
            if (beta == ZERO)
            {
                for (j = 0; j < n; j++)
                {
                    ccol = ldc*j;
                    for (i = 0; i < m; i++)
                    {
                        C[i + ccol] = ZERO;
                    }
                }
            } else
            {
                for (j = 0; j < n; j++)
                {
                    ccol = ldc*j;
                    for (i = 0; i < m; i++)
                    {
                        C[i + ccol] *= beta;
                    }
                }
            }
            return;
        }
        // Start the operations.
        T TEMP;
        if (notb)
        {
            if (nota)
            {
                // Form  C = alpha * A * B + beta * C.
                for (j = 0; j < n; j++)
                {
                    ccol = ldc*j;
                    if (beta == ZERO)
                    {
                        for (i = 0; i < m; i++)
                        {
                            C[i + ccol] = ZERO;
                        }
                    } else if (beta != ONE)
                    {
                        for (i = 0; i < m; i++)
                        {
                            C[i + ccol] *= beta;
                        }
                    }
                    bcol = ldb*j;
                    for (l = 0; l < k; l++)
                    {
                        if (B[l + bcol] != ZERO)
                        {
                            acol = lda*l;
                            TEMP = alpha * B[l + bcol];
                            for (i = 0; i < m; i++)
                            {
                                C[i + ccol] += TEMP * A[i + acol];
                            }
                        }
                    }
                }
            } else
            {
                // Form  C = alpha * A^T * B + beta * C
                for (j = 0; j < n; j++)
                {
                    bcol = ldb*j;
                    ccol = ldc*j;
                    for (i = 0; i < m; i++)
                    {
                        acol = lda*i;
                        TEMP = ZERO;
                        for (l = 0; l < k; l++)
                        {
                            TEMP += A[l + acol] * B[l + bcol];
                        }
                        if (beta == ZERO)
                        {
                            C[i + ccol] = alpha*TEMP;
                        } else
                        {
                            C[i + ccol] = alpha * TEMP + beta * C[i + ccol];
                        }
                    }
                }
            }
        } else
        {
            if (nota)
            {
                // Form  C = alpha * A * B^T + beta * C
                for (j = 0; j < n; j++)
                {
                    ccol = ldc*j;
                    if (beta == ZERO)
                    {
                        for (i = 0; i < m; i++)
                        {
                            C[i + ccol] = ZERO;
                        }
                    } else if (beta != ONE)
                    {
                        for (i = 0; i < m; i++)
                        {
                            C[i + ccol] *= beta;
                        }
                    }
                    for (l = 0; l < k; l++)
                    {
                        bcol = ldb*l;
                        if (B[j + bcol] != ZERO)
                        {
                            acol = lda*l;
                            TEMP = alpha * B[j + bcol];
                            for (i = 0; i < m; i++)
                            {
                                C[i + ccol] += TEMP * A[i + acol];
                            }
                        }
                    }
                }
            } else
            {
                // Form  C = alpha * A^T * B^T + beta * C
                for (j = 0; j < n; j++)
                {
                    ccol = ldc*j;
                    for (i = 0; i < m; i++)
                    {
                        acol = lda*i;
                        TEMP = ZERO;
                        for (l = 0; l < k; l++)
                        {
                            TEMP += A[l + acol] * B[j + ldb * l];
                        }
                        if (beta == ZERO)
                        {
                            C[i + ccol] = alpha*TEMP;
                        } else
                        {
                            C[i + ccol] = alpha * TEMP + beta * C[i + ccol];
                        }
                    }
                }
            }
        }
    }

    /* dgemv  performs one of the matrix - vector operations
     *     y : = alpha*A*x + beta*y, or y : = alpha*A^T*x + beta*y,
     * where alpha and beta are scalars, x and y are vectors and A is an m by n matrix.
     * Parameters: trans: On entry, trans specifies the operation to be performed as follows:
     *                    trans = 'N' or 'n'   y : = alpha*A*x + beta*y.
     *                    trans = 'T' or 't'   y : = alpha*A^T*x + beta*y.
     *                    trans = 'C' or 'c'   y : = alpha*A^T*x + beta*y.
     *             m: specifies the number of rows of the matrix A.
     *                m must be at least zero.
     *             n: specifies the number of columns of the matrix A.
     *                n must be at least zero.
     *             alpha: specifies the scalar alpha.
     *             A: an array of DIMENSION(lda, n).
     *                Before entry, the leading m by n part of the array A must
     *                contain the matrix of coefficients.
     *             lda: specifies the first dimension of A as declared in the calling function.
     *                  lda must be at least max(1, m).
     *             x: an array of DIMENSION at least (1 + (n - 1)*abs(incx)) when trans = 'N' or 'n'
     *                                  and at least (1 + (m - 1)*abs(incx)) otherwise.
     *                Before entry, the incremented array x must contain the vector x.
     *             incx: specifies the increment for the elements of x. incx must not be zero.
     *             beta: specifies the scalar beta. When beta is supplied as zero then y need not be set on input.
     *             y: an array of DIMENSION at least (1 + (m - 1)*abs(incy)) when trans = 'N' or 'n'
     *                                  and at least (1 + (n - 1)*abs(incy)) otherwise.
     *                Before entry with beta non - zero, the incremented array y must contain the vector y.
     *                On exit, y is overwritten by the updated vector y.
     *             incy: specifies the increment for the elements of y. incy must not be zero.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                           */
    static void dgemv(char const* trans, int m, int n, T alpha, T const* A, int lda, T const* x, int incx, T beta, T* y, int incy)
    {
        // Test the input parameters.
        int info = 0;
        char uptrans = toupper(trans[0]);
        if (uptrans != 'N' && uptrans != 'T' && uptrans != 'C')
        {
            info = 1;
        } else if (m < 0)
        {
            info = 2;
        } else if (n < 0)
        {
            info = 3;
        } else if (lda < ((1 > m) ? 0 : (m - 1)))
        {
            info = 6;
        } else if (incx == 0)
        {
            info = 8;
        } else if (incy == 0)
        {
            info = 11;
        }
        if (info != 0)
        {
            xerbla("DGEMV", info);
            return;
        }
        // Quick return if possible.
        if ((m == 0) || (n == 0) || ((alpha == ZERO) && (beta == ONE)))
        {
            return;
        }
        // Set lenx and leny, the lengths of the vectors x and y, and set up the start points in x and y.
        int kx, ky, lenx, leny;
        if (uptrans == 'N')
        {
            lenx = n;
            leny = m;
        } else
        {
            lenx = m;
            leny = n;
        }
        if (incx > 0)
        {
            kx = 0;
        } else
        {
            kx = -(lenx - 1) * incx;
        }
        if (incy > 0)
        {
            ky = 0;
        } else
        {
            ky = -(leny - 1) * incy;
        }
        // Start the operations. In this version the elements of A are accessed sequentially with one pass through A.
        int i, iy;
        // First form  y : = beta*y.
        if (beta != ONE)
        {
            if (incy == 1)
            {
                if (beta == ZERO)
                {
                    for (i = 0; i < leny; i++)
                    {
                        y[i] = ZERO;
                    }
                } else
                {
                    for (i = 0; i < leny; i++)
                    {
                        y[i] *= beta;
                    }
                }
            } else
            {
                iy = ky;
                if (beta == ZERO)
                {
                    for (i = 0; i < leny; i++)
                    {
                        y[iy] = ZERO;
                        iy += incy;
                    }
                } else
                {
                    for (i = 0; i < leny; i++)
                    {
                        y[iy] *= beta;
                        iy += incy;
                    }
                }
            }
        }
        if (alpha == ZERO)
        {
            return;
        }
        int ix, j, jx, jy, colj;
        T temp;
        if (uptrans == 'N')
        {
            // Form  y : = alpha*A*x + y.
            jx = kx;
            if (incy == 1)
            {
                for (j = 0; j < n; j++)
                {
                    colj = lda*j;
                    temp = alpha * x[jx];
                    for (i = 0; i < m; i++)
                    {
                        y[i] += temp * A[i + colj];
                    }
                    jx += incx;
                }
            } else
            {
                for (j = 1; j < n; j++)
                {
                    colj = lda*j;
                    temp = alpha * x[jx];
                    iy = ky;
                    for (i = 0; i < m; i++)
                    {
                        y[iy] += temp * A[i + colj];
                        iy += incy;
                    }
                    jx += incx;
                }
            }
        } else
        {
            // Form  y : = alpha*A^T*x + y.
            jy = ky;
            if (incx == 1)
            {
                for (j = 0; j < n; j++)
                {
                    colj = lda*j;
                    temp = ZERO;
                    for (i = 0; i < m; i++)
                    {
                        temp += A[i + colj] * x[i];
                    }
                    y[jy] += alpha*temp;
                    jy += incy;
                }
            } else
            {
                for (j = 0; j < n; j++)
                {
                    colj = lda*j;
                    temp = ZERO;
                    ix = kx;
                    for (i = 0; i < m; i++)
                    {
                        temp += A[i + colj] * x[ix];
                        ix += incx;
                    }
                    y[jy] += alpha*temp;
                    jy += incy;
                }
            }
        }
    }

    /* dger performs the rank 1 operation
     *     A : = alpha*x*y^T + A,
     * where alpha is a scalar, x is an m element vector, y is an n element vector and A is an m by n matrix.
     * Parameters: m: On entry, m specifies the number of rows of the matrix A. m must be at least zero.
     *             n: On entry, n specifies the number of columns of the matrix A. n must be at least zero.
     *             alpha: On entry, alpha specifies the scalar alpha.
     *             x: an array of dimension at least (1 + (m - 1)*abs(incx)).
     *                Before entry, the incremented array x must contain the m element vector x.
     *             incx: On entry, incx specifies the increment for the elements of x. incx must not be zero.
     *             y: an array of dimension at least (1 + (n - 1)*abs(incy)).
     *                Before entry, the incremented array y must contain the n element vector y.
     *             incy: On entry, incy specifies the increment for the elements of y. incy must not be zero.
     *             A: an array of DIMENSION(lda, n).
     *                Before entry, the leading m by n part of the array A must contain the matrix of coefficients.
     *                On exit, A is overwritten by the updated matrix.
     *             lda: On entry, lda specifies the first dimension of A as declared in the calling(sub) program.
     *                  lda must be at least max(1, m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                           */
    static void dger(int m, int n, T alpha, T const* x, int incx, T const* y, int incy, T* A, int lda)
    {
        T temp;
        // Test the input parameters.
        int info = 0;
        if (m < 0)
        {
            info = 1;
        } else if (n < 0)
        {
            info = 2;
        } else if (incx == 0)
        {
            info = 5;
        } else if (incy == 0)
        {
            info = 7;
        } else if (lda < (1 > m ? 1 : m))
        {
            info = 9;
        }
        if (info != 0)
        {
            xerbla("DGER", info);
            return;
        }
        // Quick return if possible.
        if ((m == 0) || (n == 0) || (alpha == ZERO))
        {
            return;
        }
        // Start the operations. In this version the elements of A are accessed sequentially with one pass through A.
        int i, j, jy, colj;
        if (incy > 0)
        {
            jy = 0;
        } else
        {
            jy = -(n - 1) * incy;
        }
        if (incx == 1)
        {
            for (j = 0; j < n; j++)
            {
                if (y[jy] != ZERO)
                {
                    colj = lda*j;
                    temp = alpha * y[jy];
                    for (i = 0; i < m; i++)
                    {
                        A[i + colj] += x[i] * temp;
                    }
                }
                jy += incy;
            }
        } else
        {
            int ix, kx;
            if (incx > 0)
            {
                kx = 0;
            } else
            {
                kx = -(m - 1) * incx;
            }
            for (j = 0; j < n; j++)
            {
                if (y[jy] != ZERO)
                {
                    colj = lda*j;
                    temp = alpha * y[jy];
                    ix = kx;
                    for (i = 0; i < m; i++)
                    {
                        A[i + colj] += x[ix] * temp;
                        ix += incx;
                    }
                }
                jy += incy;
            }
        }
    }

    /* dnrm2 returns the euclidean norm of a vector:
     * sqrt(x^T*x)
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.										*/
    static T dnrm2(int n, T const* x, int incx)
    {
        T absxi, norm, scale, ssq;
        int ix;
        if (n < 1 || incx < 0)
        {
            norm = ZERO;
        } else if (n == 1)
        {
            norm = fabs(x[0]);
        } else
        {
            scale = ZERO;
            ssq = ONE;
            // The following loop is equivalent to this call to the LAPACK auxiliary routine:
            // dlassq(n, x, incx, scale, ssq);
            for (ix = 0; ix <= ((n - 1) * incx); ix += incx)
            {
                if (x[ix] != ZERO)
                {
                    absxi = fabs(x[ix]);
                    if (scale < absxi)
                    {
                        ssq = ONE + ssq * (scale / absxi)*(scale / absxi);
                        scale = absxi;
                    } else
                    {
                        ssq = ssq + (absxi / scale)*(absxi / scale);
                    }
                }
            }
            norm = scale * sqrt(ssq);
        }
        return norm;
    }

    /* drot applies a plane rotation.
     * Parameters: n: number of elements in input vector(s)
     *             dx: an array, dimension (1 + (n-1)*abs(incx))
     *             incx: storage spacing between elements of dx
     *             dy: an array, dimension (1 + (n-1)*abs(incy))
     *             incy: storage spacing between elements of dy
     *             c
     *             s
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: November 2017                                                                       */
    static void drot(int n, T* dx, int incx, T* dy, int incy, T c, T s)
    {
        if (n<=0)
        {
            return;
        }
        T dtemp;
        int i;
        if (incx==1 && incy==1)
        {
            // code for both increments equal to 1
            for (i=0; i<n; i++)
            {
                DTEMP = c*dx[i] + s*dy[i];
                dy[i] = c*dy[i] - s*dx[i];
                dx[i] = DTEMP;
            }
        }
        else
        {
            // code for unequal increments or equal increments not equal to 1
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
                DTEMP  = c*dx[ix] + s*dy[iy];
                dy[iy] = c*dy[iy] - s*dx[ix];
                dx[ix] = DTEMP;
                ix += incx;
                iy += incy;
            }
        }
    }

    /* dscal scales a vector by a constant. uses unrolled loops for increment equal to one.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.																*/
    static void dscal(int n, T da, T* dx, int incx)
    {
        int i;
        if (n < 0 || incx < 0)
        {
            return;
        }
        if (incx == 1)
        {
            // code for increment equal to 1
            // clean - up loop
            int m = n % 5;
            if (m != 0)
            {
                for (i = 0; i < m; i++)
                {
                    dx[i] = da * dx[i];
                }
                if (n < 5)
                {
                    return;
                }
            }
            for (i = m; i < n; i += 5)
            {
                dx[i] = da * dx[i];
                dx[i + 1] = da * dx[i + 1];
                dx[i + 2] = da * dx[i + 2];
                dx[i + 3] = da * dx[i + 3];
                dx[i + 4] = da * dx[i + 4];
            }
        } else
        {
            //code for increment not equal to 1
            int nincx = n*incx;
            for (i = 0; i < nincx; i += incx)
            {
                dx[i] = da * dx[i];
            }
        }
    }

    /* dswap interchanges two vectors. Uses unrolled loops for increments equal to one.
     * Parameters: n: The length of both vectors.
     *             dx: The first vector.
     *             incx: The increment of dx.
     *             dy: The second vector.
     *             incy: The increment of dy.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void dswap(int n, T* dx, int incx, T* dy, int incy)
    {
        if (n <= 0)
        {
            return;
        }
        T dtemp;
        int i;
        if (incx == 1 && incy == 1)
        {
            // code for both increments equal to 1
            // clean-up loop
            int m = n % 3;
            if (m != 0)
            {
                for (i = 0; i < m; i++)
                {
                    dtemp = dx[i];
                    dx[i] = dy[i];
                    dy[i] = dtemp;
                }
                if (n < 3)
                {
                    return;
                }
            }
            for (i = m; i < n; i += 3)
            {
                dtemp = dx[i];
                dx[i] = dy[i];
                dy[i] = dtemp;
                dtemp = dx[i + 1];
                dx[i + 1] = dy[i + 1];
                dy[i + 1] = dtemp;
                dtemp = dx[i + 2];
                dx[i + 2] = dy[i + 2];
                dy[i + 2] = dtemp;
            }
        } else
        {
            // code for unequal increments or equal increments not equal to 1
            int ix = 0;
            int iy = 0;
            if (incx < 0)
            {
                ix = (-n + 1) * incx;
            }
            if (incy < 0)
            {
                iy = (-n + 1) * incy;
            }
            for (i = 0; i < n; i++)
            {
                dtemp = dx[ix];
                dx[ix] = dy[iy];
                dy[iy] = dtemp;
                ix += incx;
                iy += incy;
            }
        }
    }

    /* dsymv performs the matrix-vector operation
     *     y := alpha*A*x + beta*y,
     * where alpha and beta are scalars, x and y are n element vectors and A is an n by n
     * symmetric matrix.
     * Parameters: uplo: On entry, uplo specifies whether the upper or lower triangular part of the
     *                   array A is to be referenced as follows:
     *                   uplo=='U' or 'u': Only the upper triangular part of A is to be referenced.
     *                   uplo=='L' or 'l': Only the lower triangular part of A is to be referenced.
     *             n: On entry, n specifies the order of the matrix A. n must be at least zero.
     *             alpha: On entry, alpha specifies the scalar alpha.
     *             A: an array, dimension (lda, n)
     *                Before entry with uplo=='U' or 'u', the leading n by n upper triangular part
     *                of the array A must contain the upper triangular part of the symmetric matrix
     *                and the strictly lower triangular part of A is not referenced.
     *                Before entry with uplo=='L' or 'l', the leading n by n lower triangular part
     *                of the array A must contain the lower triangular part of the symmetric matrix
     *                and the strictly upper triangular part of A is not referenced.
     *             lda: On entry, lda specifies the first dimension of A as declared in the calling
     *                  (sub)program. lda must be at least max(1, n).
     *             x: an array, dimension at least (1 + (n-1)*abs(incx)).
     *                Before entry, the incremented array x must contain the n element vector x.
     *             incx: On entry, incx specifies the increment for the elements of x.
     *                   incx must not be zero.
     *             beta: On entry, beta specifies the scalar beta. When beta is supplied as zero
     *                   then y need not be set on input.
     *             y: an array, dimension at least (1 + (n-1)*abs(incy)).
     *                Before entry, the incremented array y must contain the n element vector y.
     *                On exit, y is overwritten by the updated vector y.
     *             incy: On entry, incy specifies the increment for the elements of y.
     *                   incy must not be zero.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016
     * Further Details:
     *     Level 2 Blas routine.
     *     The vector and matrix arguments are not referenced when n==0, or m==0
     *     -- Written on 22-October-1986.
     *        Jack Dongarra, Argonne National Lab.
     *        Jeremy Du Croz, Nag Central Office.
     *        Sven Hammarling, Nag Central Office.
     *        Richard Hanson, Sandia National Labs.                                              */
    static void dsymv(char const* uplo, int n, T alpha, T const* A, int lda, T const* x, int incx,
                      T beta, T* y, int incy)
    {
        // Test the input parameters.
        int info = 0;
        char upuplo = std::toupper(uplo[0]);
        if (upuplo!='U' && upuplo!='L')
        {
            info = 1;
        }
        else if (n<0)
        {
            info = 2;
        }
        else if (lda<1 || lda<n)
        {
            info = 5;
        }
        else if (incx==0)
        {
            info = 7;
        }
        else if (incy==0)
        {
            info = 10;
        }
        if (info!=0)
        {
            xerbla("DSYMV ", info);
            return;
        }
        // Quick return if possible.
        if (n==0 || (alpha==ZERO && beta==ONE))
        {
            return;
        }
        // Set up the start points in x and y.
        int kx, ky;
        if (incx>0)
        {
            kx = 0;
        }
        else
        {
            kx = -(n-1)*incx;
        }
        if (incy>0)
        {
            ky = 0;
        }
        else
        {
            ky = -(n-1)*incy;
        }
        // Start the operations. In this version the elements of A are accessed sequentially with
        // one pass through the triangular part of A.
        // First form y := beta*y.
        int i, iy;
        if (beta!=ONE)
        {
            if (incy==1)
            {
                if (beta==ZERO)
                {
                    for (i=0; i<n; i++)
                    {
                        y[i] = ZERO;
                    }
                }
                else
                {
                    for (i=0; i<n; i++)
                    {
                        y[i] *= beta;
                    }
                }
            }
            else
            {
                iy = ky;
                if (beta==ZERO)
                {
                    for (i=0; i<n; i++)
                    {
                        y[iy] = ZERO;
                        iy += incy;
                    }
                }
                else
                {
                    for (i=0; i<n; i++)
                    {
                        y[iy] *= beta;
                        iy += incy;
                    }
                }
            }
        }
        if (alpha==ZERO)
        {
            return;
        }
        int ix, j, jx, jy, ldaj;
        T temp1, temp2;
        if (upuplo=='U')
        {
            // Form y when A is stored in upper triangle.
            if (incx==1 && incy==1)
            {
                for (j=0; j<n; j++)
                {
                    ldaj = lda*j;
                    temp1 = alpha*x[j];
                    temp2 = ZERO;
                    for (i=0; i<j; i++)
                    {
                        y[i] += temp1 * A[i+ldaj];
                        temp2 += A[i+ldaj] * x[i];
                    }
                    y[j] += temp1*A[j+ldaj] + alpha*temp2;
                }
            }
            else
            {
                jx = kx;
                jy = ky;
                for (j=0; j<n; j++)
                {
                    ldaj = lda*j;
                    temp1 = alpha*x[jx];
                    temp2 = ZERO;
                    ix = kx;
                    iy = ky;
                    for (i=0; i<j; i++)
                    {
                        y[iy] += temp1 * A[i+ldaj]:
                        temp2 += A[i+ldaj] * x[ix];
                        ix += incx;
                        iy += incy;
                    }
                    y[jy] += temp1*A[j+ldaj] + alpha*temp2;
                    jx += incx;
                    jy += incy;
                }
            }
        }
        else
        {
            // Form y when A is stored in lower triangle.
            if (incx==1 && incy==1)
            {
                for (j=0; j<n; j++)
                {
                    ldaj = lda*j;
                    temp1 = alpha*x[j];
                    temp2 = ZERO;
                    y[j] += temp1*A[j+ldaj];
                    for (i=j+1; i<n; i++)
                    {
                        y[i] += temp1 * A[i+ldaj];
                        temp2 += A[i+ldaj]*x[i];
                    }
                    y[j] += alpha * temp2;
                }
            }
            else
            {
                jx = kx;
                jy = ky;
                for (j=0; j<n; j++)
                {
                    ldaj = lda*j;
                    temp1 = alpha*x[jx];
                    temp2 = ZERO;
                    y[jy] += temp1*A[j+ldaj];
                    ix = jx;
                    iy = jy;
                    for (i=j+1; i<n; i++)
                    {
                        ix += incx;
                        iy += incy;
                        y[iy] += temp1 * A[i+ldaj];
                        temp2 += A[i+ldaj] * x[ix];
                    }
                    y[jy] += alpha * temp2;
                    jx += incx;
                    jy += incy;
                }
            }
        }
    }

    /* dsyr2 performs the symmetric rank 2 operation
     *      A := alpha*x*y^T + alpha*y*x^T + A,
     * where alpha is a scalar, x and y are n element vectors and A is an n by n symmetric matrix.
     * Parameters: uplo: On entry, uplo specifies whether the upper or lower triangular part of the
     *                   array A is to be referenced as follows:
     *                   uplo=='U' or 'u': Only the upper triangular part of A is to be referenced.
     *                   uplo=='L' or 'l': Only the lower triangular part of A is to be referenced.
     *             n: On entry, n specifies the order of the matrix A. n must be at least zero.
     *             alpha: On entry, alpha specifies the scalar alpha.
     *             x: an array, dimension at least (1 + (n-1)*abs(incx)).
     *                Before entry, the incremented array x must contain the n element vector x.
     *             incx: On entry, incx specifies the increment for the elements of x.
     *                   incx must not be zero.
     *             y: an array, dimension at least (1 + (n-1)*abs(incy)).
     *                Before entry, the incremented array y must contain the n element vector y.
     *             incy: On entry, incy specifies the increment for the elements of y.
     *                   incy must not be zero.
     *             A: an array, dimension (lda, n)
     *                Before entry with uplo=='U' or 'u', the leading n by n upper triangular part
     *                    of the array A must contain the upper triangular part of the symmetric
     *                    matrix and the strictly lower triangular part of A is not referenced.
     *                On exit, the upper triangular part of the array A is overwritten by the upper
     *                    triangular part of the updated matrix.
     *                Before entry with uplo=='L' or 'l', the leading n by n lower triangular part
     *                    of the array A must contain the lower triangular part of the symmetric
     *                    matrix and the strictly upper triangular part of A is not referenced.
     *                On exit, the lower triangular part of the array A is overwritten by the lower
     *                    triangular part of the updated matrix.
     *             lda: On entry, lda specifies the first dimension of A as declared in the calling
     *                  (sub)program. lda must be at least max(1, n).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016
     * Further Details:
     *     Level 2 Blas routine.
     *     -- Written on 22-October-1986.
     *        Jack Dongarra, Argonne National Lab.
     *        Jeremy Du Croz, Nag Central Office.
     *        Sven Hammarling, Nag Central Office.
     *        Richard Hanson, Sandia National Labs.                                              */
    static void dsyr2(char const* uplo, int n, T alpha, T const* x, int incx, T const* y, int incy,
                      T* A, int lda)
    {
        // Test the input parameters.
        int info = 0;
        char upuplo = std::toupper(uplo[0]);
        if (upuplo!='U' && upuplo!='L')
        {
            info = 1;
        }
        else if (n<0)
        {
            info = 2;
        }
        else if (incx==0)
        {
            info = 5;
        }
        else if (incy==0)
        {
            info = 7;
        }
        else if (lda<1 || lda<n)
        {
            info = 9;
        }
        if (info!=0)
        {
            xerbla("DSYR2 ", info);
            return;
        }
        // Quick return if possible.
        if (n==0 || alpha==ZERO)
        {
            return;
        }
        // Set up the start points in x and y if the increments are not both unity.
        int jx, jy, kx, ky;
        if (incx!=1 || incy!=1)
        {
            if (incx>0)
            {
                kx = 0;
            }
            else
            {
                kx = -(n-1)*incx;
            }
            if (incy>0)
            {
                ky = 0;
            }
            else
            {
                ky = -(n-1)*incy;
            }
            jx = kx;
            jy = ky;
        }
        // Start the operations. In this version the elements of A are accessed sequentially with
        // one pass through the triangular part of A.
        T temp1, temp2;
        int i, ix, iy, j, ldaj;
        if (upuplo=='U')
        {
            // Form A when A is stored in the upper triangle.
            if (incx==1 && incy==1)
            {
                for (j=0; j<n; j++)
                {
                    if (x[j]!=ZERO || y[j]!=ZERO)
                    {
                        ldaj = lda*j;
                        temp1 = alpha*y[j];
                        temp2 = alpha*x[j];
                        for (i=0; i<=j; i++)
                        {
                            A[i+ldaj] += x[i]*temp1 + y[i]*temp2;
                        }
                    }
                }
            }
            else
            {
                for (j=0; j<n; j++)
                {
                    if (x[jx]!=ZERO || y[jy]!=ZERO)
                    {
                        ldaj = lda*j;
                        temp1 = alpha*y[jy];
                        temp2 = alpha*x[jx];
                        ix = kx;
                        iy = ky;
                        for (i=0; i<=j; i++)
                        {
                            A[i+ldaj] += x[ix]*temp1 + y[iy]*temp2;
                            ix += incx;
                            iy += incy;
                        }
                    }
                    jx += incx;
                    jy += incy;
                }
            }
        }
        else
        {
            // Form A when A is stored in the lower triangle.
            if (incx==1 && incy==1)
            {
                for (j=0; j<n; j++)
                {
                    if (x[j]!=ZERO || y[j]!=ZERO)
                    {
                        ldaj = lda*j;
                        temp1 = alpha*y[j];
                        temp2 = alpha*x[j];
                        for (i=j; i<n; i++)
                        {
                            A[i+ldaj] += x[i]*temp1 + y[i]*temp2;
                        }
                    }
                }
            }
            else
            {
                for (j=0; j<n; j++)
                {
                    if (x[jx]!=ZERO || y[jy]!=ZERO)
                    {
                        ldaj = lda*j;
                        temp1 = alpha*y[jy];
                        temp2 = alpha*x[jx];
                        ix = jx;
                        iy = jy;
                        for (i=j; i<n; i++)
                        {
                            A[i+ldaj] += x[ix]*temp1 + y[iy]*temp2;
                            ix += incx;
                            iy += incy;
                        }
                    }
                    jx += incx;
                    jy += incy;
                }
            }
        }
    }

    /* dtrmm performs one of the matrix-matrix operations
     *     B = alpha*op(A)*B, or B = alpha*B*op(A),
     * where alpha is a scalar, B is an m by n matrix, A is a unit, or non-unit,
     * upper or lower triangular matrix and op(A) is one of
     *     op(A) = A or op(A) = A^T.
     * Parameters: side: On entry, side specifies whether op(A) multiplies B from
     *                   the left or right as follows:
     *                       side = 'L' or 'l'   B : = alpha*op(A)*B.
     *                       side = 'R' or 'r'   B : = alpha*B*op(A).
     *             uplo: On entry, uplo specifies whether the matrix A is an upper or
     *                   lower triangular matrix as follows :
     *                       uplo = 'U' or 'u'  A is an upper triangular matrix.
     *                       uplo = 'L' or 'l'  A is a lower triangular matrix.
     *             transa: On entry, transa specifies the form of op(A) to be used in
     *                     the matrix multiplication as follows:
     *                         transa = 'N' or 'n'  op(A) = A.
     *                         transa = 'T' or 't'  op(A) = A^T.
     *                         transa = 'C' or 'c'  op(A) = A^T.
     *             diag: On entry, diag specifies whether or not A is unit triangular as follows:
     *                       diag = 'U' or 'u'  A is assumed to be unit triangular.
     *                       diag = 'N' or 'n'  A is not assumed to be unit triangular.
     *             m: On entry, m specifies the number of rows of B. m must be at least zero.
     *             n: On entry, n specifies the number of columns of B. n must be at least zero.
     *             alpha: On entry, alpha specifies the scalar alpha. When alpha is zero then
     *                    A is not referenced and B need not be set before entry.
     *             A: an array of DIMENSION(lda, k), where k is m when side = 'L' or 'l' and is
     *                n when side = 'R' or 'r'. Before entry with uplo = 'U' or 'u', the leading
     *                k by k upper triangular part of the array A must contain the upper triangular
     *                matrix and the strictly lower triangular part of A is not referenced.
     *                Before entry with uplo = 'L' or 'l', the leading k by k lower triangular
     *                part of the array A must contain the lower triangular matrix and the strictly
     *                upper triangular part of A is not referenced.
     *                Note that when diag = 'U' or 'u', the diagonal elements of A are not
     *                referenced either, but are assumed to be unity.
     *             lda: On entry, lda specifies the first dimension of A as declared in the calling
     *                  (sub)program. When side = 'L' or 'l' then lda must be at least max(1, m),
     *                  when side = 'R' or 'r' then lda must be at least max(1, n).
     *             B: an array of DIMENSION(ldb, n).
     *                Before entry, the leading m by n part of the array B must contain the matrix B,
     *                and on exit is overwritten by the transformed matrix.
     *             ldb: On entry, ldb specifies the first dimension of B as declared in the calling
     *                  (sub)program. ldb must be at least max(1, m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void dtrmm(char const* side, char const* uplo, char const* transa, char const* diag, int m, int n, T alpha, T const* A, int lda, T* B, int ldb)
    {
        // Test the input parameters.
        bool lside = (toupper(side[0]) == 'L');
        int nrowa;
        if (lside)
        {
            nrowa = m;
        } else
        {
            nrowa = n;
        }
        bool nounit = (toupper(diag[0]) == 'N');
        bool upper = (toupper(uplo[0]) == 'U');
        char uptransa = toupper(transa[0]);
        char updiag = toupper(diag[0]);
        int info = 0;
        if (!lside && (toupper(side[0]) != 'R'))
        {
            info = 1;
        } else if (!upper && (toupper(uplo[0]) != 'L'))
        {
            info = 2;
        } else if (uptransa != 'N' && uptransa != 'T' && uptransa != 'C')
        {
            info = 3;
        } else if (updiag != 'U' && updiag != 'N')
        {
            info = 4;
        } else if (m < 0)
        {
            info = 5;
        } else if (n < 0)
        {
            info = 6;
        } else if (lda < (1 > nrowa ? 1 : nrowa))
        {
            info = 9;
        } else if (ldb < (1 > m ? 1 : m))
        {
            info = 11;
        }
        if (info != 0)
        {
            xerbla("DTRMM", info);
            return;
        }
        // Quick return if possible.
        if (m == 0 || n == 0)
        {
            return;
        }
        int i, j;
        // And when alpha==zero.
        if (alpha == ZERO)
        {
            for (j = 0; j < n; j++)
            {
                for (i = 0; i < m; i++)
                {
                    B[i + ldb * j] = ZERO;
                }
            }
            return;
        }
        T temp;
        int k, acolk, bcolj;
        // Start the operations.
        if (lside)
        {
            if (uptransa == 'N')
            {
                // Form B = alpha * A * B.
                if (upper)
                {
                    for (j = 0; j < n; j++)
                    {
                        bcolj = ldb*j;
                        for (k = 0; k < m; k++)
                        {
                            if (B[k + bcolj] != ZERO)
                            {
                                acolk = lda*k;
                                temp = alpha * B[k + bcolj];
                                for (i = 0; i < k; i++)
                                {
                                    B[i + bcolj] += temp * A[i + acolk];
                                }
                                if (nounit)
                                {
                                    temp *= A[k + acolk];
                                }
                                B[k + bcolj] = temp;
                            }
                        }
                    }
                } else
                {
                    for (j = 0; j < n; j++)
                    {
                        bcolj = ldb*j;
                        for (k = m - 1; k >= 0; k--)
                        {
                            if (B[k + bcolj] != ZERO)
                            {
                                acolk = lda*k;
                                temp = alpha * B[k + bcolj];
                                B[k + bcolj] = temp;
                                if (nounit)
                                {
                                    B[k + bcolj] *= A[k + acolk];
                                }
                                for (i = k + 1; i < m; i++)
                                {
                                    B[i + bcolj] += temp * A[i + acolk];
                                }
                            }
                        }
                    }
                }
            } else
            {
                // Form B = alpha * A^T * B.
                if (upper)
                {
                    for (j = 0; j < n; j++)
                    {
                        bcolj = ldb*j;
                        for (k = m - 1; k >= 0; k--)
                        {
                            acolk = lda*k;
                            temp = B[k + bcolj];
                            if (nounit)
                            {
                                temp *= A[k + acolk];
                            }
                            for (i = 0; i < k; i++)
                            {
                                temp += A[i + acolk] * B[i + bcolj];
                            }
                            B[k + bcolj] = alpha*temp;
                        }
                    }
                } else
                {
                    for (j = 0; j < n; j++)
                    {
                        bcolj = ldb*j;
                        for (k = 0; k < m; k++)
                        {
                            acolk = lda*k;
                            temp = B[k + bcolj];
                            if (nounit)
                            {
                                temp *= A[k + acolk];
                            }
                            for (i = k + 1; i < m; i++)
                            {
                                temp += A[i + acolk] * B[i + bcolj];
                            }
                            B[k + bcolj] = alpha*temp;
                        }
                    }
                }
            }
        } else
        {
            int acolj, bcolk;
            if (uptransa == 'N')
            {
                // Form B = alpha * B * A.
                if (upper)
                {
                    for (j = n - 1; j >= 0; j--)
                    {
                        acolj = lda*j;
                        bcolj = ldb*j;
                        temp = alpha;
                        if (nounit)
                        {
                            temp *= A[j + acolj];
                        }
                        for (i = 0; i < m; i++)
                        {
                            B[i + bcolj] *= temp;
                        }
                        for (k = 0; k < j; k++)
                        {
                            if (A[k + acolj] != ZERO)
                            {
                                bcolk = ldb*k;
                                temp = alpha * A[k + acolj];
                                for (i = 0; i < m; i++)
                                {
                                    B[i + bcolj] += temp * B[i + bcolk];
                                }
                            }
                        }
                    }
                } else
                {
                    for (j = 0; j < n; j++)
                    {
                        acolj = lda*j;
                        bcolj = ldb*j;
                        temp = alpha;
                        if (nounit)
                        {
                            temp *= A[j + acolj];
                        }
                        for (i = 0; i < m; i++)
                        {
                            B[i + bcolj] *= temp;
                        }
                        for (k = j + 1; k < n; k++)
                        {
                            if (A[k + acolj] != ZERO)
                            {
                                bcolk = ldb*k;
                                temp = alpha * A[k + acolj];
                                for (i = 0; i < m; i++)
                                {
                                    B[i + bcolj] += temp * B[i + bcolk];
                                }
                            }
                        }
                    }
                }
            } else
            {
                // Form B = alpha * B * A^T.
                if (upper)
                {
                    for (k = 0; k < n; k++)
                    {
                        acolk = lda*k;
                        bcolk = ldb*k;
                        for (j = 0; j < k; j++)
                        {
                            if (A[j + acolk] != ZERO)
                            {
                                bcolj = ldb*j;
                                temp = alpha * A[j + acolk];
                                for (i = 0; i < m; i++)
                                {
                                    B[i + bcolj] += temp * B[i + bcolk];
                                }
                            }
                        }
                        temp = alpha;
                        if (nounit)
                        {
                            temp *= A[k + acolk];
                        }
                        if (temp != ONE)
                        {
                            for (i = 0; i < m; i++)
                            {
                                B[i + bcolk] *= temp;
                            }
                        }
                    }
                } else
                {
                    for (k = n - 1; k >= 0; k--)
                    {
                        acolk = lda*k;
                        bcolk = ldb*k;
                        for (j = k + 1; j < n; j++)
                        {
                            if (A[j + acolk] != ZERO)
                            {
                                bcolj = ldb*j;
                                temp = alpha * A[j + acolk];
                                for (i = 0; i < m; i++)
                                {
                                    B[i + bcolj] += temp * B[i + bcolk];
                                }
                            }
                        }
                        temp = alpha;
                        if (nounit)
                        {
                            temp *= A[k + acolk];
                        }
                        if (temp != ONE)
                        {
                            for (i = 0; i < m; i++)
                            {
                                B[i + bcolk] *= temp;
                            }
                        }
                    }
                }
            }
        }
    }

    /* dtrmv performs one of the matrix-vector operations
     *     x : = A*x, or x : = A^T*x,
     * where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix.
     * Parameters: uplo: On entry, uplo specifies whether the matrix is an upper or lower triangular matrix as follows:
     *                   uplo = 'U' or 'u'   A is an upper triangular matrix.
     *                   uplo = 'L' or 'l'   A is a lower triangular matrix.
     *             trans: On entry, trans specifies the operation to be performed as follows:
     *                    trans = 'N' or 'n'   x : = A*x.
     *                    trans = 'T' or 't'   x : = A^T*x.
     *                    trans = 'C' or 'c'   x : = A^T*x.
     *             diag: On entry, diag specifies whether or not A is unit triangular as follows:
     *                   diag = 'U' or 'u'   A is assumed to be unit triangular.
     *                   diag = 'N' or 'n'   A is not assumed to be unit triangular.
     *             n: On entry, n specifies the order of the matrix A. n must be at least zero.
     *             A: an array of DIMENSION(lda, n).
     *                Before entry with  uplo = 'U' or 'u', the leading n by n upper triangular part of the array A must
     *                contain the upper triangular matrix and the strictly lower triangular part of A is not referenced.
     *                Before entry with uplo = 'L' or 'l', the leading n by n lower triangular part of the array A must
     *                contain the lower triangular matrix and the strictly upper triangular part of A is not referenced.
     *                Note that when  diag = 'U' or 'u', the diagonal elements of A are not referenced either,
     *                but are assumed to be unity.
     *             lda: On entry, lda specifies the first dimension of A as declared in the calling(sub) program.
     *                  lda must be at least max(1, n).
     *             x: an array of dimension at least (1 + (n - 1)*abs(incx)).
     *                Before entry, the incremented array x must contain the n element vector x.
     *                On exit, x is overwritten with the tranformed vector x.
     *             incx: On entry, incx specifies the increment for the elements of x. incx must not be zero.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                         */
    static void dtrmv(char const* uplo, char const* trans, char const* diag, int n, T const* A, int lda, T* x, int incx)
    {
        // Test the input parameters.
        int info = 0;
        char upuplo = toupper(uplo[0]);
        char uptrans = toupper(trans[0]);
        char updiag = toupper(diag[0]);
        if (upuplo != 'U' && upuplo != 'L')
        {
            info = 1;
        } else if (uptrans != 'N' && uptrans != 'T' && uptrans != 'C')
        {
            info = 2;
        } else if (updiag != 'U' && updiag != 'N')
        {
            info = 3;
        } else if (n < 0)
        {
            info = 4;
        } else if (lda < (1 > n ? 1 : n))
        {
            info = 6;
        } else if (incx == 0)
        {
            info = 8;
        }
        if (info != 0)
        {
            xerbla("DTRMV", info);
            return;
        }
        // Quick return if possible.
        if (n == 0)
        {
            return;
        }
        bool nounit = (updiag == 'N');
        // Set up the start point in x if the increment is not unity.
        // This will be(n - 1)*incx  too small for descending loops.
        int kx;
        if (incx < 0)
        {
            kx = -(n - 1) * incx;
        } else if (incx != 1)
        {
            kx = 0;
        }
        // Start the operations.In this version the elements of A are accessed sequentially with one pass through A.
        T temp;
        int i, ix, j, jx, colj;
        if (uptrans == 'N')
        {
            // Form  x : = A*x.
            if (upuplo == 'U')
            {
                if (incx == 1)
                {
                    for (j = 0; j < n; j++)
                    {
                        if (x[j] != ZERO)
                        {
                            colj = lda*j;
                            temp = x[j];
                            for (i = 0; i < j; i++)
                            {
                                x[i] += temp * A[i + colj];
                            }
                            if (nounit)
                            {
                                x[j] *= A[j + colj];
                            }
                        }
                    }
                } else
                {
                    jx = kx;
                    for (j = 0; j < n; j++)
                    {
                        if (x[jx] != ZERO)
                        {
                            colj = lda*j;
                            temp = x[jx];
                            ix = kx;
                            for (i = 0; i < j; i++)
                            {
                                x[ix] += temp * A[i + colj];
                                ix += incx;
                            }
                            if (nounit)
                            {
                                x[jx] *= A[j + colj];
                            }
                        }
                        jx += incx;
                    }
                }
            } else
            {
                if (incx == 1)
                {
                    for (j = n - 1; j >= 0; j--)
                    {
                        if (x[j] != ZERO)
                        {
                            colj = lda*j;
                            temp = x[j];
                            for (i = n - 1; i > j; i--)
                            {
                                x[i] += temp * A[i + colj];
                            }
                            if (nounit)
                            {
                                x[j] *= A[j + colj];
                            }
                        }
                    }
                } else
                {
                    kx += (n - 1) * incx;
                    jx = kx;
                    for (j = n - 1; j >= 0; j--)
                    {
                        if (x[jx] != ZERO)
                        {
                            colj = lda*j;
                            temp = x[jx];
                            ix = kx;
                            for (i = n - 1; i > j; i--)
                            {
                                x[ix] += temp * A[i + colj];
                                ix -= incx;
                            }
                            if (nounit)
                            {
                                x[jx] *= A[j + colj];
                            }
                        }
                        jx -= incx;
                    }
                }
            }
        } else
        {
            // Form  x : = A^T*x.
            if (upuplo == 'U')
            {
                if (incx == 1)
                {
                    for (j = n - 1; j >= 0; j--)
                    {
                        colj = lda*j;
                        temp = x[j];
                        if (nounit)
                        {
                            temp *= A[j + colj];
                        }
                        for (i = j - 1; i >= 0; i--)
                        {
                            temp += A[i + colj] * x[i];
                        }
                        x[j] = temp;
                    }
                } else
                {
                    jx = kx + (n - 1) * incx;
                    for (j = n - 1; j >= 0; j--)
                    {
                        colj = lda*j;
                        temp = x[jx];
                        ix = jx;
                        if (nounit)
                        {
                            temp *= A[j + colj];
                        }
                        for (i = j - 1; i >= 0; i--)
                        {
                            ix -= incx;
                            temp += A[i + colj] * x[ix];
                        }
                        x[jx] = temp;
                        jx -= incx;
                    }
                }
            } else
            {
                if (incx == 1)
                {
                    for (j = 0; j < n; j++)
                    {
                        colj = lda*j;
                        temp = x[j];
                        if (nounit)
                        {
                            temp *= A[j + colj];
                        }
                        for (i = j + 1; i < n; i++)
                        {
                            temp += A[i + colj] * x[i];
                        }
                        x[j] = temp;
                    }
                } else
                {
                    jx = kx;
                    for (j = 0; j < n; j++)
                    {
                        colj = lda*j;
                        temp = x[jx];
                        ix = jx;
                        if (nounit)
                        {
                            temp *= A[j + colj];
                        }
                        for (i = j + 1; i < n; i++)
                        {
                            ix += incx;
                            temp += A[i + colj] * x[ix];
                        }
                        x[jx] = temp;
                        jx += incx;
                    }
                }
            }
        }
    }

    /* idamax finds the index of element having max. absolute value.
     * Note: this is a zero-based index!
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static int idamax(int n, T const* dx, int incx)
    {
        if (n < 1 || incx <= 0)
        {
            return -1;
        }
        if (n == 1)
        {
            return 0;
        }
        T dmax, dnext;
        int i, ida = 0;
        if (incx == 1)
        {
            // code for increment equal to 1
            dmax = fabs(dx[0]);
            for (i = 1; i < n; i++)
            {
                dnext = fabs(dx[i]);
                if (dnext > dmax)
                {
                    ida = i;
                    dmax = dnext;
                }
            }
        } else
        {
            // code for increment not equal to 1
            dmax = fabs(dx[0]);
            int ix = incx;
            for (i = 1; i < n; i++)
            {
                dnext = fabs(dx[ix]);
                if (dnext > dmax)
                {
                    ida = i;
                    dmax = dnext;
                }
                ix += incx;
            }
        }
        return ida;
    }

    // LAPACK INSTALL (alphabetically)

    /* dlamch determines double precision machine parameters.
     * Parameters: cmach: cmach[0] Specifies the value to be returned by DLAMCH:
     *                    'E' or 'e', DLAMCH : = eps: relative machine precision
     *                    'S' or 's , DLAMCH := sfmin: safe minimum, such that 1 / sfmin does not overflow
     *                    'B' or 'b', DLAMCH : = base: base of the machine (radix)
     *                    'P' or 'p', DLAMCH : = eps*base
     *                    'N' or 'n', DLAMCH : = t: number of(base) digits in the mantissa
     *                    'R' or 'r', DLAMCH : = rnd: 1.0 when rounding occurs in addition, 0.0 otherwise
     *                    'M' or 'm', DLAMCH : = emin: minimum exponent before(gradual) underflow
     *                    'U' or 'u', DLAMCH : = rmin: underflow threshold - base^(emin - 1)
     *                    'L' or 'l', DLAMCH : = emax: largest exponent before overflow
     *                    'O' or 'o', DLAMCH : = rmax: overflow threshold - (base^emax)*(1 - eps)
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.																*/
    static T dlamch(char const* cmach)
    {
        T eps;
        // Assume rounding, not chopping.Always.
        T rnd = ONE;
        if (ONE == rnd)
        {
            eps = std::numeric_limits<T>::epsilon() * T(0.5);
        } else
        {
            eps = std::numeric_limits<T>::epsilon();
        }
        T sfmin, small;
        switch (toupper(cmach[0]))
        {
            case 'E':
                return eps;
            case 'S':
                sfmin = std::numeric_limits<T>::min();
                small = ONE / std::numeric_limits<T>::max();
                if (small > sfmin)
                {
                    //Use SMALL plus a bit, to avoid the possibility of rounding
                    // causing overflow when computing  1 / sfmin.
                    sfmin = small * (ONE + eps);
                }
                return sfmin;
            case 'B':
                return std::numeric_limits<T>::radix;
            case 'P':
                return eps * std::numeric_limits<T>::radix;
            case 'N':
                return std::numeric_limits<T>::digits;
            case 'R':
                return rnd;
            case 'M':
                return std::numeric_limits<T>::min_exponent;
            case 'U':
                return std::numeric_limits<T>::min();
            case 'L':
                return std::numeric_limits<T>::max_exponent;
            case 'O':
                return std::numeric_limits<T>::max();
            default:
                return T(0.0);
        }
    }

    // LAPACK SRC (alphabetically)

    /* dbdsqr computes the singular values and, optionally, the right and/or left singular vectors
     * from the singular value decomposition (SVD) of a real n-by-n (upper or lower) bidiagonal
     * matrix B using the implicit zero-shift QR algorithm. The SVD of B has the form
     *     B = Q * S * P^T
     * where S is the diagonal matrix of singular values, Q is an orthogonal matrix of left
     * singular vectors, and P is an orthogonal matrix of right singular vectors. If left singular
     * vectors are requested, this subroutine actually returns U*Q instead of Q, and, if right
     * singular vectors are requested, this subroutine returns P^T*Vt instead of P^T, for given
     * real input matrices U and Vt. When U and Vt are the orthogonal matrices that reduce a
     * general matrix A to bidiagonal form: A = U*B*Vt, as computed by DGEBRD, then
     *     A = (U*Q) * S * (P^T*Vt)
     * is the SVD of A. Optionally, the subroutine may also compute Q^T*C for a given real input
     * matrix C.
     * See "Computing Small Singular Values of Bidiagonal Matrices With Guaranteed High Relative
     *     Accuracy," by J. Demmel and W. Kahan, LAPACK Working Note #3 (or SIAM J. Sci. Statist.
     *     Comput. vol. 11, no. 5, pp. 873-912, Sept 1990)
     * and "Accurate singular values and differential qd algorithms," by B. Parlett and
     *     V. Fernando, Technical Report CPAM-554, Mathematics Department, University of California
     *     at Berkeley, July 1992
     * for a detailed description of the algorithm.
     * Parameters: uplo: ='U': B is upper bidiagonal.
     *                   ='L': B is lower bidiagonal.
     *             n: The order of the matrix B. n >= 0.
     *             ncvt: The number of columns of the matrix Vt. ncvt >= 0.
     *             nru: The number of rows of the matrix U. nru >= 0.
     *             ncc: The number of columns of the matrix C. ncc >= 0.
     *             d: an array, dimension (n)
     *                On entry, the n diagonal elements of the bidiagonal matrix B.
     *                On exit, if info=0, the singular values of B in decreasing order.
     *             e: an array, dimension (n-1)
     *                On entry, the n-1 offdiagonal elements of the bidiagonal matrix B.
     *                On exit, if info==0, e is destroyed; if info>0, d and e will contain the
     *                         diagonal and superdiagonal elements of a bidiagonal matrix
     *                         orthogonally equivalent to the one given as input.
     *             Vt: an array, dimension (ldvt, ncvt)
     *                 On entry, an n-by-ncvt matrix Vt.
     *                 On exit, Vt is overwritten by P^T * Vt.
     *                 Not referenced if ncvt==0.
     *             ldvt: The leading dimension of the array Vt.
     *                   ldvt>=max(1,n) if ncvt>0; ldvt>=1 if ncvt==0.
     *             U: an array, dimension (ldu, n)
     *                On entry, an nru-by-n matrix U.
     *                On exit, U is overwritten by U * Q.
     *                Not referenced if nru==0.
     *             ldu: The leading dimension of the array U. ldu >= max(1,nru).
     *             C: an array, dimension (ldc, ncc)
     *                On entry, an n-by-ncc matrix C.
     *                On exit, C is overwritten by Q^T * C.
     *                Not referenced if ncc = 0.
     *             ldc: The leading dimension of the array C.
     *                  ldc>=max(1,n) if ncc>0; ldc>=1 if ncc==0.
     *             work: an array, dimension (4*n)
     *             info: =0: successful exit
     *                   <0: If info = -i, the i-th argument had an illegal value
     *                   >0: if ncvt = nru = ncc = 0,
     *                       = 1, a split was marked by a positive value in e
     *                       = 2, current block of Z not diagonalized after 30*n iterations
     *                            (in inner while loop)
     *                       = 3, termination criterion of outer while loop not met
     *                            (program created more than n unreduced blocks)
     *                       else
     *                            the algorithm did not converge; d and e contain the elements of a
     *                            bidiagonal matrix which is orthogonally similar to the input
     *                            matrix B; if info==i, i elements of e have not converged to zero.
     * Internal Parameters: TOLMUL: default = max(10,min(100,EPS^(-1/8)))
     *                              TOLMUL controls the convergence criterion of the QR loop.
     *                              If it is positive, TOLMUL*EPS is the desired relative precision
     *                                  in the computed singular values.
     *                              If it is negative, abs(TOLMUL*EPS*sigma_max) is the desired
     *                                  absolute accuracy in the computed singular values
     *                                  (corresponds to relative accuracy abs(TOLMUL*EPS) in the
     *                                  largest singular value.)
     *                              abs(TOLMUL) should be between 1 and 1/EPS, and preferably
     *                                  between 10 (for fast convergence) and 0.1/EPS
     *                                  (for there to be some accuracy in the results).
     *                              Default is to lose at either one eighth or 2 of the available
     *                                  decimal digits in each computed singular value
     *                                  (whichever is smaller).
     *                      MAXITR: default = 6
     *                              MAXITR controls the maximum number of passes of the algorithm
     *                              through its inner loop. The algorithms stops (and so fails to
     *                              converge) if the number of passes through the inner loop
     *                              exceeds MAXITR*n^2.
     * Note:
     *     Bug report from Cezary Dendek.
     *         On March 23rd 2017, the integer variable MAXIT = MAXITR*n^2 is removed since it can
     *         overflow pretty easily (for n larger or equal than 18,919). We instead use
     *         maxitdivn = MAXITR*n.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2017                                                                            */
    static void dbdsqr(char const* uplo, int n, int ncvt, int nru, int ncc, T* d, T* e, T* Vt,
                       int ldvt, T* U, int ldu, T* C, int ldc, T* work, int& info)
    {
        const T NEGONE = T(-1.0);
        const T HNDRTH = T(0.01);
        const T TEN    = T(10.0);
        const T HNDRD  = T(100.0);
        const T MEIGTH = T(-0.125);
        const int MAXITR = 6;
        // Test the input parameters.
        info = 0;
        bool lower = (std::toupper(UPLO[0])=='L');
        if (!(std::toupper(UPLO[0])=='U') && !lower)
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
        else if ((ncvt==0 && ldvt<1) || (ncvt>0 && (ldvt<1 || ldvt<n)))
        {
            info = -9;
        }
        else if (ldu<1 || ldu<nru)
        {
            info = -11;
        }
        else if ((ncc==0 && ldc<1) || (ncc>0 && (ldc<1 || ldc<n)))
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
        T smin;
        if (n!=1)
        {
            // rotate is true if any singular vectors desired, false otherwise
            bool rotate = ((ncvt>0) || (nru>0) || (ncc>0));
            // If no singular vectors desired, use qd algorithm
            if (!rotate)
            {
                dlasq1(n, d, e, work, info);
                // If INFO equals 2, dqds didn't finish, try to finish
                if (info!=2)
                {
                    return;
                }
                info = 0;
            }
            int nm1 = n - 1;
            int nm12 = nm1 + nm1;
            int nm13 = nm12 + nm1;
            int IDIR = 0;
            // Get machine constants
            T eps = dlamch("Epsilon");
            T unfl = dlamch("Safe minimum");
            // If matrix lower bidiagonal, rotate to be upper bidiagonal by applying
            // Givens rotations on the left
            T cs, sn, r;
            if (lower)
            {
                for (i=0; i<n-1; i++)
                {
                    dlartg(d[i], e[i], cs, sn, r);
                    d[i] = r;
                    e[i] = sn*d[i+1];
                    d[i+1] = cs*d[i+1];
                    work[i] = cs;
                    work[nm1+i] = sn;
                }
                // Update singular vectors if desired
                if (nru>0)
                {
                    dlasr("R", "V", "F", nru, n, &work[0], &work[n-1], U, ldu);
                }
                if (ncc>0)
                {
                    dlasr("L", "V", "F", n, ncc, &work[0], &work[n-1], C, ldc);
                }
            }
            // Compute singular values to relative accuracy TOL (By setting TOL to be negative,
            // algorithm will compute singular values to absolute accuracy
            // abs(tol)*norm(input matrix))
            T tolmul = std::pow(eps,MEIGTH);
            if (HUNDRD<tolmul)
            {
                tolmul = HUNDRD;
            }
            if (TEN>tolmul)
            {
                tolmul = TEN;
            }
            T tol = tolmul * eps;
            // Compute approximate maximum, minimum singular values
            T smax = ZERO, temp;
            for (i=0; i<n; i++)
            {
                temp = std::fabs(d[i]);
                if (temp>smax)
                {
                    smax = temp;
                }
            }
            for (i=0; i<n-1; i++)
            {
                temp = std::fabs(e[i]);
                if (temp>smax)
                {
                    smax = temp;
                }
            }
            T sminl = ZERO, sminoa, mu, thresh;
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
                sminoa = sminoa / std::sqrt(T(n));
                temp = MAXITR * (n*(n*unfl));
                thresh = tol * sminoa;
                if (temp>thresh)
                {
                    thresh = temp;
                }
            }
            else
            {
                // Absolute accuracy desired
                temp = MAXITR * (n*(n*unfl));
                thresh = std::fabs(tol) * smax;
                if (temp>thresh)
                {
                    thresh = temp;
                }
            }
            // Prepare for main iteration loop for the singular values (MAXIT is the maximum number
            // of passes through the inner loop permitted before nonconvergence signalled.)
            int maxitdivn = MAXITR*n;
            int iterdivn = 0;
            int iter = -1;
            int oldll = -1;
            int oldm = -1;
            // m points to last element of unconverged part of matrix
            int m = n-1;
            // Begin main iteration loop
            int ll, lll;
            T abse, abss, cosl, cosr, f, g, h, oldcs, oldsn, shift, sigmn, sigmx, sinl, sinr, sll;
            bool breakloop1 = false, breakloop2;
            while (true)
            {
                // Check for convergence or exceeding iteration count
                if (m<=0)
                {
                    break
                }
                if (iter>=n)
                {
                    iter -= n;
                    iterdivn++;
                    if (iterdivn>=maxitdivn)
                    {
                        breakloop1 = true;
                        break;
                    }
                }
                // Find diagonal block of matrix to work on
                if (tol<ZERO && std::fabs(d[m])<=thresh)
                {
                    d[m] = ZERO;
                }
                smax = std::fabs(d[m]);
                smin = smax;
                loopbreak2 = false;
                for (lll=0; lll<m; lll++)
                {
                    ll = m - lll + 1;
                    abss = std::fabs(d[ll]);
                    abse = std::fabs(e[ll]);
                    if (tol<ZERO && abss<=thresh)
                    {
                        d[ll] = ZERO;
                    }
                    if (abse<=thresh)
                    {
                        loopbreak2 = true;
                        break;
                    }
                    if (abss<smin)
                    {
                        smin = abss;
                    }
                    temp = abss>abse?abss:abse;
                    if (temp>smax)
                    {
                        smax = temp;
                    }
                }
                if (loopbreak2)
                {
                    e[ll] = ZERO;
                    // Matrix splits since E[ll] = 0
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
                // E[ll] through E[m-1] are nonzero, E[ll-1] is zero
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
                        drot(ncvt, &Vt[m-1/*ldvt*0*/], ldvt, &Vt[m/*+ldvt*0*/], ldvt, cosr, sinr);
                    }
                    if (nru>0)
                    {
                        drot(nru, &U[/*0+*/ldu*(m-1)], 1, &U[/*0+*/ldu*m], 1, cosl, sinl);
                    }
                    if (ncc>0)
                    {
                        drot(ncc, &C[m-1/*+ldc*0*/], ldc, &C[m/*+ldc*0*/], ldc, cosl, sinl);
                    }
                    m -= 2;
                    continue;
                }
                // If working on new submatrix, choose shift direction
                // (from larger end diagonal element towards smaller)
                if (ll>oldm || m<oldll)
                {
                    if (std::fabs(d[ll])>=std::fabs(d[m])
                    {
                        // Chase bulge from top (big end) to bottom (small end)
                        IDIR = 1;
                    }
                    else
                    {
                        // Chase bulge from bottom (big end) to top (small end)
                        IDIR = 2;
                    }
                }
                // Apply convergence tests
                if (IDIR==1)
                {
                    // Run convergence test in forward direction
                    // First apply standard test to bottom of matrix
                    if (std::fabs(e[m-1])<=std::fabs(tol)*std::fabs(d[m]) || (tol<ZERO && std::fabs(e[m-1])<=thresh))
                    {
                        e[m-1] = ZERO;
                        continue;
                    }
                    if (tol>=ZERO)
                    {
                        // If relative accuracy desired, apply convergence criterion forward
                        mu = std::fabs(d[ll]);
                        sminl = mu;
                        loopbreak2 = false;
                        for (lll=ll; lll<m; lll++)
                        {
                            if (std::fabs(e[lll]<=tol*mu)
                            {
                                e[lll] = ZERO;
                                loopbreak2 = true;
                                break;
                            }
                            mu = std::fabs(d[lll+1]) * (mu/(mu+std::fabs(e[lll])));
                            if (mu<sminl)
                            {
                                sminl = mu;
                            }
                        }
                        if (loopbreak2)
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
                        loopbreak2 = false;
                        for (lll=m-1; lll>=ll; lll--)
                        {
                            if (std::fabs(e[lll])<=tol*mu)
                            {
                                e[lll] = ZERO;
                                loopbreak2 = true;
                                break
                            }
                            mu = std::fabs(d[lll]) * (mu / (mu+std::fabs(e[lll])));
                            if (mu<sminl)
                            {
                                sminl = mu;
                            }
                        }
                        if (loopbreak2)
                        {
                            continue;
                        }
                    }
                }
                oldll = ll;
                oldm = m;
                // Compute shift. First, test if shifting would ruin relative accuracy,
                // and if so set the shift to zero.
                temp = HNDRTH*tol;
                if (tol>=ZERO && n*tol*(sminl/smax)<=(eps>temp?eps:temp))
                {
                    // Use a zero shift to avoid loss of relative accuracy
                    shift = ZERO;
                }
                else
                {
                    // Compute the shift from 2-by-2 block at end of matrix
                    if (IDIR==1)
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
                    if (IDIR==1)
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
                                e[i-1] = oldsn*r;
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
                            dlasr("L", "V", "F", m-ll+1, ncvt, &work[0], &work[n-1],
                                  &Vt[ll/*+ldvt*0*/], ldvt);
                        }
                        if (nru>0)
                        {
                            dlasr("R", "V", "F", nru, m-ll+1, &work[nm12], &work[nm13],
                                  &U[/*0+*/ldu*ll], ldu);
                        }
                        if (ncc>0)
                        {
                            dlasr("L", "V", "F", m-ll+1, ncc, &work[nm12], &work[nm13+1],
                                  &C[ll/*+ldc*0*/], ldc);
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
                                e[i] = oldsn*r;
                            }
                            dlartg(oldcs*r, d[i-1]*sn, oldcs, oldsn, d[i]);
                            work[i-ll-1] = cs;
                            work[i-ll-1+nm1] = -sn;
                            work[i-ll-1+nm12] = oldcs;
                            work[i-ll-1+nm13] = -oldsn;
                        }
                        h = d[ll]*cs;
                        d[ll] = h*oldcs;
                        e[ll] = h*oldsn;
                        // Update singular vectors
                        if (ncvt>0)
                        {
                            dlasr("L", "V", "B", m-ll+1, ncvt, &work[nm12], &work[nm13],
                                  &Vt[ll/*+ldvt*0*/], ldvt)
                        }
                        if (nru>0)
                        {
                            dlasr("R", "V", "B", nru, m-ll+1, &work[0], &work[n-1],
                                  &U[/*0+*/ldu*ll], ldu);
                        }
                        if (ncc>0)
                        {
                            dlasr("L", "V", "B", m-ll+1, ncc, &work[0], &work[n-1],
                                  &C[ll/*+ldc*0*/], ldc);
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
                    if (IDIR==1)
                    {
                        // Chase bulge from top to bottom
                        // Save cosines and sines for later singular vector updates
                        f = (std::fabs(d[ll])-shift)
                            * ((T(ZERO<=d[ll])-T(ZERO>d[ll]))+shift/d[ll]);
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
                            g      = sinr*d[i+1];
                            d[i+1] = cosr*d[i+1];
                            dlartg(f, g, cosl, sinl, r);
                            d[i] = r;
                            f      = cosl*e[i]   + sinl*d[i+1];
                            d[i+1] = cosl*d[i+1] - sinl*e[i];
                            if (i+1<m)
                            {
                                g = sinl*e[i+1];
                                e[i+1] = cosl*e[i+1];
                            }
                            work[i-ll] = cosr;
                            work[i-ll+nm1] = sinr;
                            work[i-ll+nm12] = cosl;
                            work[i-ll+nm13] = sinl;
                        }
                        e[m-1] = f;
                        // Update singular vectors
                        if (ncvt>0)
                        {
                            dlasr("L", "V", "F", m-ll+1, ncvt, &work[0], &work[n-1],
                                  &Vt[ll/*+ldvt*0*/], ldvt);
                        }
                        if (nru>0)
                        {
                            dlasr("R", "V", "F", nru, m-ll+1, &work[nm12], &work[nm13],
                                  &U[/*0+*/ldu*ll], ldu);
                        }
                        if (ncc>0)
                        {
                            dlasr("L", "V", "F", m-ll+1, ncc, &work[nm12], &work[nm13],
                                  &C[ll/*+ldc*0*/], ldc);
                        }
                        // Test convergence
                        if (std::fabs(e[m-1])<=thresh)
                        {
                            e[m-2] = ZERO;
                        }
                    }
                    else
                    {
                        // Chase bulge from bottom to top
                        // Save cosines and sines for later singular vector updates
                        f = (std::fabs(d[m])-shift) * ((T(ZERO<=d[m])-T(ZERO>d[m]))+shift/d[m]);
                        g = e[m-1];
                        for (i=m; i>ll; i--)
                        {
                            dlartg(f, g, cosr, sinr, r);
                            if (i<m)
                            {
                                e[i] = r;
                            }
                            f      = cosr*d[i] + sinr*e[i-1];
                            e[i-1] = cosr*e[i-1] - sinr*d[i];
                            g      = sinr*d[i-1];
                            d[i-1] = cosr*d[i-1];
                            dlartg(f, g, cosl, sinl, r);
                            d[i]   = r;
                            f      = cosl*e[i-1] + sinl*d[i-1];
                            d[i-1] = cosl*d[i-1] - sinl*e[i-1];
                            if (i>ll+1)
                            {
                                g = sinl*e[i-2];
                                e[i-2] = cosl*e[i-2];
                            }
                            work[i-ll-1] = cosr;
                            work[i-ll-1+nm1] = -sinr;
                            work[i-ll-1+nm12] = cosl;
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
                            dlasr("L", "V", "B", m-ll+1, ncvt, &work[nm12], &work[nm13],
                                  &Vt[ll/*+ldvt*0*/], ldvt);
                        }
                        if (nru>0)
                        {
                            dlasr("R", "V", "B", nru, m-ll+1, &work[0], &work[n-1],
                                  &U[/*0+*/ldu*ll], ldu);
                        }
                        if (ncc>0)
                        {
                            dlasr("L", "V", "B", m-ll+1, ncc, &work[0], &work[n-1],
                                  &C[ll/*+ldc*0*/], ldc);
                        }
                    }
                }
                // QR iteration finished, go back and check convergence
            }
            if (loopbreak1)
            {
                // Maximum number of iterations exceeded, failure to converge
                info = 0;
                for (i=0; i<n-1; i++)
                {
                    if (e[i]!=ZERO)
                    {
                        info++;
                    }
                }
                return;
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
                    dscal(ncvt, NEGONE, &Vt[i/*+lda*0*/], ldvt);
                }
            }
        }
        // Sort the singular values into decreasing order (insertion sort on singular values,
        // but only one transposition per singular vector)
        int isub, j;
        for (i=0; i<n-1; i++)
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
            if (isub!=n-i-1)
            {
                // Swap singular values and vectors
                d[isub] = d[n-i-1];
                d[n-i-1] = smin;
                if (ncvt>0)
                {
                    dswap(ncvt, &Vt[isub/*+ldvt*0*/], ldvt, &Vt[n-i-1/*+ldvt*0*/], ldvt);
                }
                if (nru>0)
                {
                    dswap(nru, &U[/*0+*/ldu*isub], 1, &U[/*0+*/ldu*(n-i-1)], 1);
                }
                if (ncc>0)
                {
                    dswap(ncc, &C[isub/*+ldc*0*/], ldc, &C[n-i-1/*+ldc*0*/], ldc);
                }
            }
        }
    }

    /* dgebal balances a general real matrix A. This involves, first, permuting A by a similarity
     * transformation to isolate eigenvalues in the first 0 to ilo-1 and last ihi+1 to N elements on
     * the diagonal; and second, applying a diagonal similarity transformation to rows and columns ilo
     * to ihi to make the rows and columns as close in norm as possible. Both steps are optional.
     * Balancing may reduce the 1-norm of the matrix, and improve the accuracy of the computed
     * eigenvalues and/or eigenvectors.
     * Parameters: job: Specifies the operations to be performed on A:
     *                  'N': none: simply set ilo = 0, ihi = n-1, scale[i] = 1.0 for i = 0,...,n-1;
     *                  'P':  permute only;
     *                  'S':  scale only;
     *                  'B':  both permute and scale.
     *             n: The order of the matrix A. n >= 0.
     *             A: an array of dimension (lda,n)
     *                On entry, the input matrix A.
     *                On exit, A is overwritten by the balanced matrix.
     *                If job = 'N', A is not referenced. See Further Details.
     *             lda: The leading dimension of the array A. lda >= max(1,n).
     *             ilo,
     *             ihi: ilo and ihi are set to integers such that on exit A[i,j] = 0 if i > j and
     *                  j = 0,...,ilo-1 or i = ihi+1,...,n-1. If job = 'N' or 'S', ilo = 0 and ihi = n-1.
     *                  NOTE: zero-based indices!
     *             scale: an array of dimension (n)
     *                    Details of the permutations and scaling factors applied to A. If P[j] is the
     *                    index of the row and column interchanged with row and column j and D[j] is
     *                    the scaling factor applied to row and column j, then
     *                    scale[j] = P[j]    for j = 0,...,ilo-1
     *                             = D[j]    for j = ilo,...,ihi
     *                             = P[j]    for j = ihi+1,...,n-1.
     *                    The order in which the interchanges are made is n-1 to ihi+1, then 0 to ilo-1.
     *             info: = 0:  successful exit.
     *                   < 0:  if info = -i, the i-th argument had an illegal value.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: June 2017
     * Further Details:
     *     The permutations consist of row and column interchanges which put the matrix in the form
     *             ( T1   X   Y  )
     *     P A P = (  0   B   Z  )
     *             (  0   0   T2 )
     *     where T1 and T2 are upper triangular matrices whose eigenvalues lie along the diagonal.
     *     The column indices ilo and ihi mark the starting and ending columns of the submatrix B.
     *     Balancing consists of applying a diagonal similarity transformation inv(D) * B * D to
     *     make the 1-norms of each row of B and its corresponding column nearly equal.
     *     The output matrix is
     *     ( T1     X*D          Y    )
     *     (  0  inv(D)*B*D  inv(D)*Z ).
     *     (  0      0           T2   )
     *     Information about the permutations P and the diagonal matrix D is returned in the
     *     vector scale.
     *     This subroutine is based on the EISPACK routine BALANC.
     *     Modified by Tzu-Yi Chen, Computer Science Division, University of California at Berkeley, USA */
    static void dgebal(char const* job, int n, T* A, int lda, int& ilo, int& ihi, T* scale, int& info)
    {
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
        else if (lda<(1>n?1:n))
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
                        dswap(l+1, &A[/*0+*/lda*j], 1, &A[/*0+*/lda*m], 1);
                        dswap(n-k, &A[j+lda*k], lda, &A[m+lda*k], lda);
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
            T sclfac = TWO;
            T factor = 0.95;
            T c, ca, f, g, r, ra, s, sfmax1, sfmax2, sfmin1, sfmin2;
            int ica, ira;
            // Iterative loop for norm reduction
            sfmin1 = dlamch("Safemin") / dlamch("Precision");
            sfmax1 = ONE / sfmin1;
            sfmin2 = sfmin1*sclfac;
            sfmax2 = ONE / sfmin2;
            bool noconv = true;
            while (noconv)
            {
                noconv = false;
                for (i=k; i<=l; i++)
                {
                    c = dnrm2(l-k+1, &A[k+lda*i], 1);
                    r = dnrm2(l-k+1, &A[i+lda*k], lda);
                    ica = idamax(l+1, &A[/*0+*/lda*i], 1);
                    ca = fabs(A[ica+lda*i]);
                    ira = idamax(n-k, &A[i+lda*k], lda);
                    ra = fabs(A[i+lda*(ira+k)]);
                    // Guard against zero c or R due to underflow.
                    if (c==0.0 || r==0.0)
                    {
                        continue;
                    }
                    g = r / sclfac;
                    f = ONE;
                    s = c + r;
                    while (c<g && f<sfmax2 && c<sfmax2 && ca<sfmax2 && r>sfmin2 && g>sfmin2 && ra>sfmin2)
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
                    while (g>=r && r<sfmax2 && ra<sfmax2 && f>sfmin2 && c>sfmin2 && g>sfmin2 && ca>sfmin2)
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
                    dscal(n-k, g, &A[i+lda*k], lda);
                    dscal(l+1, f, &A[/*0+*/lda*i], 1);
                }
            }
        }
        ilo = k;
        ihi = l;
    }

    /* dgebd2 reduces a real general m by n matrix A to upper or lower bidiagonal form B by an
     * orthogonal transformation: Q^T*A*P = B.
     * If m>=n, B is upper bidiagonal; if m<n, B is lower bidiagonal.
     * Parameters: m: The number of rows in the matrix A. m>=0.
     *             n: The number of columns in the matrix A. n>=0.
     *             A: an array, dimension (lda,n)
     *                On entry, the m by n general matrix to be reduced.
     *                On exit,
     *                  if m>=n, the diagonal and the first superdiagonal are overwritten with the
     *                    upper bidiagonal matrix B; the elements below the diagonal, with the
     *                    array tauq, represent the orthogonal matrix Q as a product of elementary
     *                    reflectors, and the elements above the first superdiagonal, with the
     *                    array taup, represent the orthogonal matrix P as a product of elementary
     *                    reflectors;
     *                  if m<n, the diagonal and the first subdiagonal are overwritten with the
     *                    lower bidiagonal matrix B; the elements below the first subdiagonal, with
     *                    the array tauq, represent the orthogonal matrix Q as a product of
     *                    elementary reflectors, and the elements above the diagonal, with the
     *                    array taup, represent the orthogonal matrix P as a product of elementar
     *                    reflectors.
     *                See Further Details.
     *             lda: The leading dimension of the array A. lda >= max(1,m).
     *             d: an array, dimension (min(m,n))
     *                The diagonal elements of the bidiagonal matrix B: d[i] = A[i,i].
     *             e: an array, dimension (min(m,n)-1)
     *                The off-diagonal elements of the bidiagonal matrix B:
     *                  if m >= n, e[i] = A[i,i+1] for i = 0,1,...,n-2;
     *                  if m < n, e[i] = A[i+1,i] for i = 0,1,...,m-2.
     *             tauq: an array, dimension (min(m,n))
     *                   The scalar factors of the elementary reflectors which represent the
     *                   orthogonal matrix Q. See Further Details.
     *             taup: an array, dimension (min(m,n))
     *                   The scalar factors of the elementary reflectors which represent the
     *                   orthogonal matrix P. See Further Details.
     *             work: an array, dimension (max(m,n))
     *             info: ==0: successful exit.
     *                    <0: if info==-i, the i-th argument had an illegal value.
     * Authors: Univ. of Tennessee
     *          Univ. of California Berkeley
     *          Univ. of Colorado Denver
     *          NAG Ltd.
     * Date: June 2017
     * Further Details:
     *     The matrices Q and P are represented as products of elementary reflectors:
     *     If m >= n,
     *       Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)
     *     Each H(i) and G(i) has the form:
     *       H(i) = I - tauq * v * v^T  and G(i) = I - taup * u * u^T
     *     where tauq and taup are real scalars, and v and u are real vectors;
     *       v[0:i-1] = 0, v[i] = 1, and v[i+1:m-1] is stored on exit in A[i+1:m-1,i];
     *       u[0:i] = 0, u[i+1] = 1, and u[i+2:n-1] is stored on exit in A[i,i+2:n-1];
     *     tauq is stored in tauq[i] and taup in taup[i].
     *     If m < n,
     *       Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)
     *     Each H(i) and G(i) has the form:
     *       H(i) = I - tauq * v * v^T  and G(i) = I - taup * u * u^T
     *     where tauq and taup are real scalars, and v and u are real vectors;
     *       v[0:i] = 0, v[i+1] = 1, and v[i+2:m-1] is stored on exit in A[i+2:m-1,i];
     *       u[0:i-1] = 0, u[i] = 1, and u[i+1:n-1] is stored on exit in A[i,i+1:n-1];
     *     tauq is stored in tauq[i] and taup in taup[i].
     *     The contents of A on exit are illustrated by the following examples:
     *       m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):
     *       (  d   e   u1  u1  u1 )           (  d   u1  u1  u1  u1  u1 )
     *       (  v1  d   e   u2  u2 )           (  e   d   u2  u2  u2  u2 )
     *       (  v1  v2  d   e   u3 )           (  v1  e   d   u3  u3  u3 )
     *       (  v1  v2  v3  d   e  )           (  v1  v2  e   d   u4  u4 )
     *       (  v1  v2  v3  v4  d  )           (  v1  v2  v3  e   d   u5 )
     *       (  v1  v2  v3  v4  v5 )
     *     where d and e denote diagonal and off-diagonal elements of B, vi denotes an element of
     *     the vector defining H(i), and ui an element of the vector defining G(i).              */
    static void dgebd2(int m, int n, T* A, int lda, T* d, T* e, T* tauq, T* taup, T* work,
                       int& info)
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
                itemp = i+2<m?i+1:m-1;
                ldai = lda*i;
                ildai = i+ldai;
                dlarfg(m-i, A[ildai], &A[itemp+ldai], 1, tauq[i]);
                d[i] = A[ildai];
                A[ildai] = ONE;
                // Apply H(i) to A[i:m-1,i+1:n-1] from the left
                if (i<n-1)
                {
                    dlarf("Left", m-i, n-1-i, &A[ildai], 1, tauq[i], &A[ildai+lda], lda, &work);
                }
                A[ildai] = d[i];
                if (i<n-1)
                {
                    // Generate elementary reflector G(i) to annihilate A[i,i+2:n-1]
                    itemp = i+3<n?i+2:n-1;
                    dlarfg(n-1-i, A[ildai+lda], &A[i+lda*itemp], lda, taup[i]);
                    e[i] = A[ildai+lda];
                    A[ildai+lda] = ONE;
                    // Apply G(i) to A[i+1:m-1,i+1:n-1] from the right
                    dlarf("Right", m-1-i, n-1-i, &A[ildai+lda], lda, taup[i], &A[1+ildai+lda], lda,
                          &work);
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
                itemp = i+2<n?i+1:n-1;
                ldai = lda*i;
                ildai = i+ldai;
                dlarfg(n-i, A[ildai], &A[i+lda*itemp], lda, taup[i]);
                d[i] = A[ildai];
                A[ildai] = ONE;
                // Apply G(i) to A[i+1:m-1,i:n-1] from the right
                if (i<m-1)
                {
                    DLARF("Right", m-1-i, n-i, &A[ildai], lda, taup[i], &A[1+ildai], lda, &work);
                }
                A[ildai] = d[i];
                if (i<m-1)
                {
                    // Generate elementary reflector H(i) to annihilate A[i+2:m-1,i]
                    itemp = i+3<m?i+2:m-1;
                    dlarfg(m-1-i, A[1+ildai], &A[itemp+ldai], 1, tauq[i]);
                    e[i] = A[1+ildai];
                    A[1+ildai] = ONE;
                    // Apply H(i) to A[i+1:m-1,i+1:n-1] from the left
                    dlarf("Left", m-1-i, n-1-i, &A[1+ildai], 1, tauq[i], &A[1+ildai+lda], lda,
                          &work);
                    A[1+ildai] = e[i];
                }
                else
                {
                    tauq[i] = ZERO;
                }
            }
        }
    }

    /* dgeqp3 computes a QR factorization with column pivoting of a matrix A: A*P = Q*R using Level 3 BLAS.
     * Parameters: m: The number of rows of the matrix A. m >= 0.
     *             n: The number of columns of the matrix A. n >= 0.
     *             A: an array, dimension(lda, n)
     *                On entry, the m-by-n matrix A.
     *                On exit, the upper triangle of the array contains the min(m, n)-by-n upper trapezoidal matrix R;
     *                the elements below the diagonal, together with the array tau,
     *                represent the orthogonal matrix Q as a product of min(m, n) elementary reflectors.
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     *             jpvt: an integer array, dimension(n)
     *                   On entry, if jpvt[j]!= -1, the j-th column of A is permuted to the front of A*P(a leading column);
     *                             if jpvt[j] = -1, the j-th column of A is a free column.
     *                   On exit,  if jpvt[j] = k,  then the j-th column of A*P was the the k-th column of A.
     *                   Note: this array contains zero-based indices
     *             tau: an array, dimension(min(m, n))
     *                  The scalar factors of the elementary reflectors.
     *             work: an array, dimension(MAX(1, lwork))
     *                   On exit, if info = 0, work[0] returns the optimal lwork.
     *             lwork: The dimension of the array work. lwork >= 3 * n + 1.
     *                    For optimal performance lwork >= 2 * n + (n + 1)*nb, where nb is the optimal blocksize.
     *                    If lwork = -1, then a workspace query is assumed; the routine only calculates the
     *                    optimal size of the work array, returns this value as the first entry of the work array,
     *                    and no error message related to lwork is issued by xerbla.
     *             info: =0: successful exit.
     *                   <0: if info = -i, the i-th argument had an illegal value.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016
     * Further Details:
     *     The matrix Q is represented as a product of elementary reflectors
     *         Q = H(1) H(2) . ..H(k), where k = min(m, n).
     *     Each H(i) has the form
     *         H(i) = I - tau * v * v^T
     *     where tau is a real scalar, and v is a real / complex vector with v[0:i-1] = 0 and v[i] = 1;
     *     v[i+1:m-1] is stored on exit in A[i+1:m-1, i], and tau in tau[i].                                        */
    static void dgeqp3(int m, int n, T* A, int lda, int* jpvt, T* tau, T* work, int lwork, int& info)
    {
        const int INB = 1, INBMIN = 2, IXOVER = 3;
        // Test input arguments
        info = 0;
        bool lquery = (lwork == -1);
        if (m < 0)
        {
            info = -1;
        } else if (n < 0)
        {
            info = -2;
        } else if (lda < (1 > m ? 1 : m))
        {
            info = -4;
        }
        int iws = 0, minmn = 0, nb;
        if (info == 0)
        {
            minmn = m < n ? m : n;
            if (minmn == 0)
            {
                iws = 1;
                work[0] = 1;
            } else
            {
                iws = 3 * n + 1;
                nb = ilaenv(INB, "DGEQRF", " ", m, n, -1, -1);
                work[0] = 2 * n + (n + 1) * nb;
            }
            if ((lwork < iws) && !lquery)
            {
                info = -8;
            }
        }
        if (info != 0)
        {
            xerbla("DGEQP3", -info);
            return;
        } else if (lquery)
        {
            return;
        }
        // Move initial columns up front.
        int j;
        int nfxd = 1;
        for (j = 0; j < n; j++)
        {
            if (jpvt[j] != -1)
            {
                if (j != nfxd - 1)
                {
                    dswap(m, &A[/*0+*/lda * j], 1, &A[/*0+*/lda * (nfxd - 1)], 1);
                    jpvt[j] = jpvt[nfxd - 1];
                    jpvt[nfxd - 1] = j;
                } else
                {
                    jpvt[j] = j;
                }
                nfxd++;
            } else
            {
                jpvt[j] = j;
            }
        }
        nfxd--;
        // Factorize fixed columns
        // Compute the QR factorization of fixed columns and update remaining columns.
        if (nfxd > 0)
        {
            int na = m < nfxd ? m : nfxd;
            // dgeqr2(m, na, A, lda, tau, work, info);
            dgeqrf(m, na, A, lda, tau, work, lwork, info);
            iws = iws > int(work[0]) ? iws : int(work[0]);
            if (na < n)
            {
                // dorm2r("Left", "Transpose", m, n - na, na, A, lda, tau, &A[/*0+*/lda*na], lda, work, info);
                dormqr("Left", "Transpose", m, n - na, na, A, lda, tau, &A[/*0+*/lda * na], lda, work, lwork, info);
                iws = iws > int(work[0]) ? iws : int(work[0]);
            }
        }
        // Factorize free columns
        if (nfxd < minmn)
        {
            int sm = m - nfxd;
            int sn = n - nfxd;
            int sminmn = minmn - nfxd;
            // Determine the block size.
            nb = ilaenv(INB, "DGEQRF", " ", sm, sn, -1, -1);
            int nbmin = 2;
            int nx = 0;
            if ((nb > 1) && (nb < sminmn))
            {
                // Determine when to cross over from blocked to unblocked code.
                nx = ilaenv(IXOVER, "DGEQRF", " ", sm, sn, -1, -1);
                if (0 > nx)
                {
                    nx = 0;
                }
                if (nx < sminmn)
                {
                    // Determine if workspace is large enough for blocked code.
                    int minws = 2 * sn + (sn + 1) * nb;
                    iws = iws > minws ? iws : minws;
                    if (lwork < minws)
                    {
                        // Not enough workspace to use optimal nb: Reduce nb and determine the minimum value of nb.
                        nb = (lwork - 2 * sn) / (sn + 1);
                        nbmin = ilaenv(INBMIN, "DGEQRF", " ", sm, sn, -1, -1);
                        if (nbmin < 2)
                        {
                            nbmin = 2;
                        }
                    }
                }
            }
            // Initialize partial column norms. The first n elements of work store the exact column norms.
            for (j = nfxd; j < n; j++)
            {
                work[j] = dnrm2(sm, &A[nfxd + lda * j], 1);
                work[n + j] = work[j];
            }
            if ((nb >= nbmin) && (nb < sminmn) && (nx < sminmn))
            {
                // Use blocked code initially.
                j = nfxd;
                // Compute factorization: while loop.
                int topbmn = minmn - nx;
                int fjb, jb;
                while (j < topbmn)
                {
                    jb = topbmn - j;
                    if (nb < jb)
                    {
                        jb = nb;
                    }
                    // Factorize jb columns among columns j:n-1.
                    dlaqps(m, n - j, j, jb, fjb, &A[/*0+*/lda * j], lda, &jpvt[j], &tau[j], &work[j], &work[n + j], &work[2 * n], &work[2 * n + jb], n - j);
                    j += fjb;
                }
            } else
            {
                j = nfxd;
            }
            // Use unblocked code to factor the last or only block.
            if (j < minmn)
            {
                dlaqp2(m, n - j, j, &A[/*0+*/lda * j], lda, &jpvt[j], &tau[j], &work[j], &work[n + j], &work[2 * n]);
            }
        }
        work[0] = iws;
    }

    /* dgeqr2 computes a QR factorization of a real m by n matrix A : A = Q * R.
     * Parameters: m: The number of rows of the matrix A. m >= 0.
     *             n: The number of columns of the matrix A. n >= 0.
     *             A: an array, dimension(lda, n)
     *                On entry, the m by n matrix A.
     *                On exit, the elements on and above the diagonal of the array contain the
     *				  min(m, n) by n upper trapezoidal matrix R (R is upper triangular if m >= n);
     *				  the elements below the diagonal, with the array tau, represent the orthogonal
     *                matrix Q as a product of elementary reflectors(see Further Details).
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     *             tau: an array, dimension(min(m, n))
     *                  The scalar factors of the elementary reflectors(see Further Details).
     *             work: an array, dimension(n)
     *             info: 0:  successful exit
     *                  <0: if info = -i, the i - th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Further Details:
     *     The matrix Q is represented as a product of elementary reflectors
     *         Q = H(1) H(2) . ..H(k), where k = min(m, n).
     *     Each H(i) has the form
     *         H(i) = I - tau * v * v^T
     *     where tau is a real scalar, and v is a real vector with v[0:i-1] = 0 and v[i] = 1;
     *     v[i+1:m-1] is stored on exit in A[i+1:m-1,i], and tau in tau[i].						*/
    static void dgeqr2(int m, int n, T* A, int lda, T* tau, T* work, int& info)
    {
        int i, k, coli;
        T AII;
        // Test the input arguments
        info = 0;
        if (m < 0)
        {
            info = -1;
        } else if (n < 0)
        {
            info = -2;
        } else if (lda < (m > 1 ? m : 1))
        {
            info = -4;
        }
        if (info != 0)
        {
            xerbla("DGEQR2", -info);
            return;
        }
        k = (m < n ? m : n);
        for (i = 0; i < k; i++)
        {
            coli = lda*i;
            // Generate elementary reflector H[i] to annihilate A[i+1:m-1, i]
            dlarfg(m - i, A[i + coli], &A[(((i + 1) < (m - 1)) ? (i + 1) : (m - 1)) + coli], 1, tau[i]);
            if (i < (n - 1))
            {
                // Apply H[i] to A[i:m-1, i+1:n-1] from the left
                AII = A[i + coli];
                A[i + coli] = ONE;
                dlarf("Left", m - i, n - i - 1, &A[i + coli], 1, tau[i], &A[i + coli + lda], lda, work);
                A[i + coli] = AII;
            }
        }
    }

    /* dgeqrf computes a QR factorization of a real m-by-n matrix A:
     * A = Q * R.
     * Parameters: m: The number of rows of the matrix A. m >= 0.
     *             n: The number of columns of the matrix A. n >= 0.
     *             A: an array, dimension(lda, n)
     *                On entry, the m-by-n matrix A.
     *                On exit, the elements on and above the diagonal of the array
     *                contain the min(m, n)-by-n upper trapezoidal matrix R
     *                (R is upper triangular if m >= n); the elements below the diagonal,
     *                with the array tau, represent the orthogonal matrix Q as a
     *                product of min(m, n) elementary reflectors(see Further Details).
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     *             tau: an array, dimension(min(m, n))
     *                  The scalar factors of the elementary reflectors(see Further Details).
     *             work: an array, dimension(max(1, lwork))
     *                   On exit, if info = 0, work[0] returns the optimal lwork.
     *             lwork: The dimension of the array work. lwork >= max(1, n).
     *                    For optimum performance lwork >= n*nb, where nb is
     *                    the optimal blocksize.
     *                    If lwork = -1, then a workspace query is assumed; the routine
     *                    only calculates the optimal size of the work array, returns
     *                    this value as the first entry of the work array, and no error
     *                    message related to lwork is issued by xerbla.
     *             info: =0:  successful exit
     *                   <0:  if info = -i, the i-th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Further Details:
     *     The matrix Q is represented as a product of elementary reflectors
     *         Q = H(1) H(2) . ..H(k), where k = min(m, n).
     *     Each H(i) has the form
     *         H(i) = I - tau * v * v^T
     *     where tau is a real scalar, and v is a real vector with v[0:i-1] = 0 and
     *     v[i] = 1; v[i+1:m-1] is stored on exit in A[i+1:m-1, i], and tau in tau[i].        */
    static void dgeqrf(int m, int n, T* A, int lda, T* tau, T* work, int lwork, int& info)
    {
        // Test the input arguments
        info = 0;
        int nb = ilaenv(1, "DGEQRF", " ", m, n, -1, -1);
        work[0] = n*nb;
        bool lquery = (lwork == -1);
        if (m < 0)
        {
            info = -1;
        } else if (n < 0)
        {
            info = -2;
        } else if (lda < (1 > m ? 1 : m))
        {
            info = -4;
        } else if (lwork < (1 > n ? 1 : n) && !lquery)
        {
            info = -7;
        }
        if (info != 0)
        {
            xerbla("DGEQRF", -info);
            return;
        } else if (lquery)
        {
            return;
        }
        // Quick return if possible
        int k = (m < n ? m : n);
        if (k == 0)
        {
            work[0] = 1;
            return;
        }
        int nbmin = 2;
        int nx = 0;
        int iws = n;
        int ldwork = 0;
        if (nb > 1 && nb < k)
        {
            // Determine when to cross over from blocked to unblocked code.
            nx = ilaenv(3, "DGEQRF", " ", m, n, -1, -1);
            if (nx < 0)
            {
                nx = 0;
            }
            if (nx < k)
            {
                // Determine if workspace is large enough for blocked code.
                ldwork = n;
                iws = ldwork*nb;
                if (lwork < iws)
                {
                    //Not enough workspace to use optimal nb: reduce nb and determine the minimum value of nb.
                    nb = lwork / ldwork;
                    nbmin = ilaenv(2, "DGEQRF", " ", m, n, -1, -1);
                    if (nbmin < 2)
                    {
                        nbmin = 2;
                    }
                }
            }
        }
        int i, ib, iinfo, aind;
        if (nb >= nbmin && nb < k && nx < k)
        {
            // Use blocked code initially
            for (i = 0; i < (k - nx); i += nb)
            {
                ib = k - i;
                if (ib > nb)
                {
                    ib = nb;
                }
                aind = i + lda*i;
                // Compute the QR factorization of the current block
                //     A[i:m-1, i:i+ib-1]
                dgeqr2(m - i, ib, &A[aind], lda, &tau[i], work, iinfo);
                if ((i + ib) < n)
                {
                    // Form the triangular factor of the block reflector
                    //     H = H(i) H(i + 1) . ..H(i + ib - 1)
                    dlarft("Forward", "Columnwise", m - i, ib, &A[aind], lda, &tau[i], work, ldwork);
                    // Apply H^T to A[i:m-1, i+ib:n-1] from the left
                    dlarfb("Left", "Transpose", "Forward", "Columnwise", m - i, n - i - ib, ib,
                            &A[aind], lda, work, ldwork, &A[aind + lda * ib], lda, &work[ib], ldwork);
                }
            }
        } else
        {
            i = 0;
        }
        // Use unblocked code to factor the last or only block.
        if (i < k)
        {
            dgeqr2(m - i, n - i, &A[i + lda * i], lda, &tau[i], work, iinfo);
        }
        work[0] = iws;
    }

    /* disnan replaced by std::isnan */

    /* dlabad takes as input the values computed by dlamch for underflow and overflow, and returns
     * the square root of each of these values if the log of large is sufficiently large. This
     * subroutine is intended to identify machines with a large exponent range, such as the Crays,
     * and redefine the underflow and overflow limits to be the square roots of the values computed
     * by dlamch. This subroutine is needed because dlamch does not compensate for poor arithmetic
     * in the upper half of the exponent range, as is found on a Cray.
     * Parameters: small: On entry, the underflow threshold as computed by dlamch.
     *                    On exit, if log10(large) is sufficiently large, the square root of small,
     *                             otherwise unchanged.
     *             large: On entry, the overflow threshold as computed by dlamch.
     *                    On exit, if log10(large) is sufficiently large, the square root of large,
     *                             otherwise unchanged.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlabad(T& small, T& large)
    {
        // If it looks like we're on a Cray, take the square root of small and large to avoid
        // overflow and underflow problems.
        if (std::log10(large)>2000.0)
        {
            small = std::sqrt(small);
            large = std::sqrt(large);
        }
    }

    /* dlacpy copies all or part of a two-dimensional matrix A to another matrix B.
     * Parameters: uplo: Specifies the part of the matrix A to be copied to B.
     *                   'U': Upper triangular part
     *                   'L': Lower triangular part
     *                   Otherwise: All of the matrix A
     *             m: The number of rows of the matrix A. m>=0.
     *             n: The number of columns of the matrix A. n>=0.
     *             A: an array, dimension (lda,n)
     *                The m by n matrix A.
     *                If uplo=='U', only the upper triangle or trapezoid is accessed;
     *                if uplo=='L', only the lower triangle or trapezoid is accessed.
     *             lda: The leading dimension of the array A. lda>=max(1,m).
     *             B: an array, dimension (ldb,n)
     *                On exit, B = A in the locations specified by uplo.
     *             ldb: The leading dimension of the array B. ldb>=max(1,m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016                                                                       */
    static void dlacpy(char const* uplo, int m, int n, T const* A, int lda, T* B, int ldb)
    {
        int i, j, ldaj;
        if (toupper(uplo[0])=='U')
        {
            for (j=0; j<n; j++)
            {
                ldaj = lda*j;
                for (i=0; i<=j && i<m; i++)
                {
                    B[i+ldaj] = A[i+ldaj];
                }
            }
        }
        else if (toupper(uplo[0])=='L')
        {
            for (j=0; j<n; j++)
            {
                ldaj = lda*j;
                for (i=j; i<m; i++)
                {
                    B[i+ldaj] = A[i+ldaj];
                }
            }
        }
        else
        {
            for (j=0; j<n; j++)
            {
                ldaj = lda*j;
                for (i=0; i<m; i++)
                {
                    B[i+ldaj] = A[i+ldaj];
                }
            }
        }
    }

    /* dlange returns the value of the 1-norm, Frobenius norm, infinity-norm, or the largest
     * absolute value of any element of a general real rectangular matrix A.
     * Parameters: norm: Specifies the value to be returned in dlange as described above.
     *             m: The number of rows of the matrix A. m>=0. When m==0, dlange is set to zero.
     *             n: The number of columns of the matrix A. n>=0. When n==0, dlange is set to zero.
     *             A: an array, dimension (lda,n)
     *                The m by n matrix A.
     *             lda: The leading dimension of the array A. lda>=max(m,1).
     *             work: an array, dimension (MAX(1,lwork)), where lwork>=m when norm = 'I';
     *                   otherwise, work is not referenced.
     * return: dlange = max(abs(A(i,j))), norm = 'M' or 'm'
     *                  norm1(A),         norm = '1', 'O' or 'o'
     *                  normI(A),         norm = 'I' or 'i'
     *                  normF(A),         norm = 'F', 'f', 'E' or 'e'
     *         where norm1 denotes the  one norm of a matrix (maximum column sum), normI denotes
     *         the infinity norm of a matrix (maximum row sum) and normF denotes the Frobenius norm
     *         of a matrix (square root of sum of squares). Note that max(abs(A(i,j))) is not a
     *         consistent matrix norm.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016                                                                       */
    static T dlange(char const* norm, int m, int n, T const* A, int lda, T* work)
    {
        int i, j, ldacol;
        T scale, sum, dlange=ZERO, temp;
        char upNorm = toupper(norm[0]);
        if (m==0 || n==0)
        {
            dlange = ZERO;
        }
        else if (upNorm=='M')
        {
            //Find max(abs(A(i,j))).
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
            for (i=0; i<m; m++)
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

    /* dlapy2 returns sqrt(x^2 + y^2), taking care not to cause unnecessary overflow.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.																*/
    static T dlapy2(T x, T Y)
    {
        T w, xabs, yabs, z;
        xabs = fabs(x);
        yabs = fabs(Y);
        w = (xabs > yabs ? xabs : yabs);
        z = (xabs < yabs ? xabs : yabs);
        if (z == ZERO)
        {
            return w;
        } else
        {
            return w * sqrt(ONE + (z / w)*(z / w));
        }
    }

    /* dlaqp2 computes a QR factorization with column pivoting of the block A[offset:m-1, 0:n-1].
     * The block A[0:offset-1, 0:n-1] is accordingly pivoted, but not factorized.
     * Parameters: m: The number of rows of the matrix A. m >= 0.
     *             n: The number of columns of the matrix A. n >= 0.
     *             offset: The number of rows of the matrix A that must be pivoted but not factorized. offset >= 0.
     *             A: an array, dimension(lda, n)
     *                On entry, the m-by-n matrix A.
     *                On exit, the upper triangle of block A[offset:m-1, 0:n-1] is the triangular factor obtained;
     *                the elements in block A[offset:m-1, 0:n-1] below the diagonal, together with the array tau,
     *                represent the orthogonal matrix Q as a product of elementary reflectors.
     *                Block A[0:offset-1, 0:n-1] has been accordingly pivoted, but not factorized.
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     *             jpvt: an integer array, dimension(n)
     *                   On entry, if jpvt[i] != -1, the i-th column of A is permuted to the front of A*P (a leading column);
     *                             if jpvt[i] == -1, the i-th column of A is a free column.
     *                   On exit,  if jpvt[i] == k, then the i-th column of A*P was the k-th column of A.
     *                   Note: this array contains zero-based indices
     *             tau: an array, dimension(min(m, n))
     *                  The scalar factors of the elementary reflectors.
     *             vn1: an array, dimension(n)
     *                  The vector with the partial column norms.
     *             vn2: an array, dimension(n)
     *                  The vector with the exact column norms.
     *             work: an array, dimension(n)
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void dlaqp2(int m, int n, int offset, T* A, int lda, int* jpvt, T* tau, T* vn1, T* vn2, T* work)
    {
        int mn = ((m - offset) < n ? (m - offset) : n);
        T tol3z = sqrt(dlamch("Epsilon"));
        // Compute factorization.
        int i, itemp, j, offpi, pvt, acoli;
        T aii, temp, temp2;
        for (i = 0; i < mn; i++)
        {
            offpi = offset + i;
            // Determine ith pivot column and swap if necessary.
            pvt = i + idamax(n - i, &vn1[i], 1);
            acoli = lda*i;
            if (pvt != i)
            {
                dswap(m, &A[/*0+*/lda * pvt], 1, &A[/*0+*/acoli], 1);
                itemp = jpvt[pvt];
                jpvt[pvt] = jpvt[i];
                jpvt[i] = itemp;
                vn1[pvt] = vn1[i];
                vn2[pvt] = vn2[i];
            }
            // Generate elementary reflector H(i).
            if (offpi < m - 1)
            {
                dlarfg(m - offpi, A[offpi + acoli], &A[offpi + 1 + acoli], 1, tau[i]);
            } else
            {
                dlarfg(1, A[m - 1 + acoli], &A[m - 1 + acoli], 1, tau[i]);
            }
            if (i + 1 < n)
            {
                // Apply H(i)^T to A[offset+i:m-1, i+1:n-1] from the left.
                aii = A[offpi + acoli];
                A[offpi + acoli] = ONE;
                dlarf("Left", m - offpi, n - i - 1, &A[offpi + acoli], 1, tau[i], &A[offpi + acoli + lda], lda, work);
                A[offpi + acoli] = aii;
            }
            // Update partial column norms.
            for (j = i + 1; j < n; j++)
            {
                if (vn1[j] != ZERO)
                {
                    // NOTE: The following 6 lines follow from the analysis in Lapack Working Note 176.
                    temp = fabs(A[offpi + lda * j]) / vn1[j];
                    temp = ONE - temp*temp;
                    temp = (temp > ZERO ? temp : ZERO);
                    temp2 = vn1[j] / vn2[j];
                    temp2 = temp * temp2*temp2;
                    if (temp2 <= tol3z)
                    {
                        if (offpi < m - 1)
                        {
                            vn1[j] = dnrm2(m - offpi - 1, &A[offpi + 1 + lda * j], 1);
                            vn2[j] = vn1[j];
                        } else
                        {
                            vn1[j] = ZERO;
                            vn2[j] = ZERO;
                        }
                    } else
                    {
                        vn1[j] *= sqrt(temp);
                    }
                }
            }
        }
    }

    /* dlaqps computes a step of QR factorization with column pivotingof a real m-by-n matrix A by using Blas-3.
     * It tries to factorize nb columns from A starting from the row offset + 1,
     * and updates all of the matrix with Blas-3 xgemm.
     * In some cases, due to catastrophic cancellations, it cannot factorize nb columns.Hence,
     * the actual number of factorized columns is returned in kb.
     * Block A[0:offset-1, 0:n-1) is accordingly pivoted, but not factorized.
     * Parameters: m: The number of rows of the matrix A. m >= 0.
     *             n: The number of columns of the matrix A. n >= 0
     *             offset: The number of rows of A that have been factorized in previous steps.
     *             nb: The number of columns to factorize.
     *             kb: The number of columns actually factorized.
     *             A: an array, dimension(lda, n)
     *                On entry, the m-by-n matrix A.
     *                On exit, block A[offset:m-1, 0:kb-1] is the triangular factor obtained and
     *                block A[0:offset-1, 0:n-1] has been accordingly pivoted, but no factorized.
     *                The rest of the matrix, block A[offset:m-1, kb:n-1] has been updated.
     *             lda: The leading dimension of the array A.lda >= max(1, m).
     *             jpvt: an integer array, dimension(n)
     *                   jpvt[i] = k <==> Column k of the full matrix A has been permuted into position i in AP.
     *                   Note: this array contains zero-based indices
     *             tau: an array, dimension(kb)
     *                  The scalar factors of the elementary reflectors.
     *             vn1: an array, dimension(n)
     *                  The vector with the partial column norms.
     *             vn2: an array, dimension(n)
     *                  The vector with the exact column norms.
     *             auxv: an array, dimension(nb)
     *                   Auxiliar vector.
     *             F: an array, dimension(ldf, nb)
     *                Matrix F^T = L*Y^T*A.
     *             ldf: The leading dimension of the array F. ldf >= max(1, n).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void dlaqps(int m, int n, int offset, int nb, int& kb, T* A, int lda, int* jpvt, T* tau, T* vn1, T* vn2, T* auxv, T* F, int ldf)
    {
        int lastrk = m < (n + offset) ? m : (n + offset);
        int lsticc = -1;
        int k = -1;
        T tol3z = sqrt(dlamch("Epsilon"));
        // Beginning of while loop.
        int itemp, j, pvt, rk, acolk;
        T akk, temp, temp2;
        while ((k + 1 < nb) && (lsticc == -1))
        {
            k++;
            rk = offset + k;
            acolk = lda*k;
            // Determine i-th pivot column and swap if necessary
            pvt = k + idamax(n - k, &vn1[k], 1);
            if (pvt != k)
            {
                dswap(m, &A[/*0+*/lda * pvt], 1, &A[/*0+*/acolk], 1);
                dswap(k, &F[pvt/*+ldf*0*/], ldf, &F[k/*+ldf*0*/], ldf);
                itemp = jpvt[pvt];
                jpvt[pvt] = jpvt[k];
                jpvt[k] = itemp;
                vn1[pvt] = vn1[k];
                vn2[pvt] = vn2[k];
            }
            // Apply previous Householder reflectors to column k:
            //     A[rk:m-1, k] -= A[rk:m-1, 0:k-1] * F[k, 0:k-1]^T.
            if (k > 0)
            {
                dgemv("No transpose", m - rk, k, -ONE, &A[rk/*+lda*0*/], lda, &F[k/*+ldf*0*/], ldf, ONE, &A[rk + acolk], 1);
            }
            // Generate elementary reflector H(k).
            if (rk < m - 1)
            {
                dlarfg(m - rk, A[rk + acolk], &A[rk + 1 + acolk], 1, tau[k]);
            } else
            {
                dlarfg(1, A[rk + acolk], &A[rk + acolk], 1, tau[k]);
            }
            akk = A[rk + acolk];
            A[rk + acolk] = ONE;
            // Compute k-th column of F:
            // Compute  F[k+1:n-1, k] = tau[k] * A[rk:m-1, k+1:n-1]^T * A[rk:m-1, k].
            if (k < n - 1)
            {
                dgemv("Transpose", m - rk, n - k - 1, tau[k], &A[rk + acolk + lda], lda, &A[rk + acolk], 1, ZERO, &F[k + 1 + ldf * k], 1);
            }
            // Padding F[0:k, k] with zeros.
            for (j = 0; j <= k; j++)
            {
                F[j + ldf * k] = ZERO;
            }
            // Incremental updating of F:
            // F[0:n-1, k] -= tau[k] * F[0:n-1, 0:k-1] * A[rk:m-1, 0:k-1]^T * A[rk:m-1, k].
            if (k > 0)
            {
                dgemv("Transpose", m - rk, k, -tau[k], &A[rk/*lda*0*/], lda, &A[rk + acolk], 1, ZERO, auxv/*[0]*/, 1);
                dgemv("No transpose", n, k, ONE, F/*[0+ldf*0]*/, ldf, auxv, 1, ONE, &F[/*0+*/ldf * k], 1);
            }
            // Update the current row of A:
            // A[rk, k+1:n-1] -= A[rk, 0:k] * F[k+1:n-1, 0:k]^T.
            if (k < n - 1)
            {
                dgemv("No transpose", n - k - 1, k + 1, -ONE, &F[k + 1/*+ldf*0*/], ldf, &A[rk/*lda*0*/], lda, ONE, &A[rk + acolk + lda], lda);
            }
            // Update partial column norms.
            if (rk < lastrk - 1)
            {
                for (j = k + 1; j < n; j++)
                {
                    if (vn1[j] != ZERO)
                    {
                        // NOTE: The following 6 lines follow from the analysis in Lapack Working Note 176.
                        temp = fabs(A[rk + lda * j]) / vn1[j];
                        temp = (ONE + temp)*(ONE - temp);
                        temp = (ZERO > temp ? ZERO : temp);
                        temp2 = vn1[j] / vn2[j];
                        temp2 = temp * temp2 * temp2;
                        if (temp2 <= tol3z)
                        {
                            vn2[j] = T(lsticc + 1);
                            lsticc = j;
                        } else
                        {
                            vn1[j] *= sqrt(temp);
                        }
                    }
                }
            }
            A[rk + acolk] = akk;
        }
        kb = k + 1;
        rk = offset + kb - 1;
        // Apply the block reflector to the rest of the matrix:
        // A[offset+kb:m-1, kb:n-1] -= A[offset+kb:m-1, 0:kb-1] * F[kb:n-1, 0:kb-1]^T.
        if (kb < (n < (m - offset) ? n : (m - offset)))
        {
            dgemm("No transpose", "Transpose", m - rk - 1, n - kb, kb, -ONE, &A[rk + 1/*+lda*0*/], lda, &F[kb/*+ldf*0*/], ldf, ONE, &A[rk + 1 + lda * kb], lda);
        }
        // Recomputation of difficult columns.
        while (lsticc >= 0)
        {
            itemp = int(vn2[lsticc] - T(0.5)); // round vn2[lsticc]-1
            vn1[lsticc] = dnrm2(m - rk - 1, &A[rk + 1 + lda * lsticc], 1);
            // NOTE: The computation of vn1[lsticc] relies on the fact that dnrm2 does not fail on vectors with norm below the value of sqrt(dlamch("S"))
            vn2[lsticc] = vn1[lsticc];
            lsticc = itemp;
        }
    }

    /* dlarf applies a real elementary reflector H to a real m by n matrix
     * C, from either the left or the right.H is represented in the form
     *     H = I - tau * v * v^T
     * where tau is a real scalar and v is a real vector.
     * If tau = 0, then H is taken to be the unit matrix.
     * Parameters: side: 'L': form  H * C
     *                   'R': form  C * H
     *             m: The number of rows of the matrix C.
     *             n: The number of columns of the matrix C.
     *             v: an array, dimension (1 + (m - 1)*abs(incv)) if side = 'L'
     *                                 or (1 + (n - 1)*abs(incv)) if side = 'R'
     *                The vector v in the representation of H. v is not used if tau = 0.
     *             incv: The increment between elements of v. incv <> 0.
     *             tau: The value tau in the representation of H.
     *             C: an array, dimension(ldc, n)
     *                          On entry, the m by n matrix C.
     *                          On exit, C is overwritten by the matrix H * C if side = 'L',
     *                                                               or C * H if side = 'R'.
     *             ldc: The leading dimension of the array C. ldc >= max(1, m).
     *             work: an array, dimension (n) if side = 'L'
     *                                    or (m) if side = 'R'
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                      */
    static void dlarf(char const* side, int m, int n, T const* v, int incv, T tau, T* C, int ldc, T* work)
    {
        bool applyleft;
        int i, lastv = 0, lastc = 0;
        applyleft = (toupper(side[0]) == 'L');
        if (tau != ZERO)
        {
            //Set up variables for scanning v. LASTV begins pointing to the end of v.
            if (applyleft)
            {
                lastv = m;
            } else
            {
                lastv = n;
            }
            if (incv > 0)
            {
                i = (lastv - 1) * incv;
            } else
            {
                i = 0;
            }
            // Look for the last non - zero row in v.
            while (lastv > 0 && v[i] == ZERO)
            {
                lastv--;
                i -= incv;
            }
            if (applyleft)
            {
                // Scan for the last non - zero column in C[0:lastv-1,:].
                lastc = iladlc(lastv, n, C, ldc) + 1;
            } else
            {
                // Scan for the last non - zero row in C[:,0:lastv-1].
                lastc = iladlr(m, lastv, C, ldc) + 1;
            }
        }
        // Note that lastc==0 renders the BLAS operations null; no special case is needed at this level.
        if (applyleft)
        {
            //Form  H * C
            if (lastv > 0)
            {
                // work[0:lastc-1] = C[0:lastv-1,0:last-1]^T * v[0:lastv-1]
                dgemv("Transpose", lastv, lastc, ONE, C, ldc, v, incv, ZERO, work, 1);
                // C[0:lastv-1,0:lastc-1] -= v[0:lastv-1] * work[0:lastc-1]^T
                dger(lastv, lastc, -tau, v, incv, work, 1, C, ldc);
            }
        } else
        {
            // Form  C * H
            if (lastv > 0)
            {
                // work[0:lastc-1] = C[0:lastc-1,0:lastv-1] * v[0:lastv-1]
                dgemv("No transpose", lastc, lastv, ONE, C, ldc, v, incv, ZERO, work, 1);
                // C[0:lastc-1,0:lastv-1] -= work[0:lastc-1] * v[0:lastv-1]^T
                dger(lastc, lastv, -tau, work, 1, v, incv, C, ldc);
            }
        }
    }

    /* dlarfb applies a real block reflector H or its transpose H^T to a real m by n matrix C,
     * from either the left or the right.
     * Parameters: side: 'L': apply H or H^T from the Left
     *                   'R': apply H or H**T from the Right
     *             trans: 'N': apply H (No transpose)
     *                    'T': apply H^T (Transpose)
     *             direct: Indicates how H is formed from a product of elementary reflectors
     *                     'F': H = H(1) H(2) . ..H(k) (Forward)
     *                     'B': H = H(k) . ..H(2) H(1) (Backward)
     *             storev: Indicates how the vectors which define the elementary reflectors
     *                     are stored:
     *                     'C': Columnwise
     *                     'R': Rowwise
     *             m: The number of rows of the matrix C.
     *             n: The number of columns of the matrix C.
     *             k: The order of the matrix Tm(= the number of elementary reflectors
     *                whose product defines the block reflector).
     *             V: an array, dimension (ldv, k) if storev = 'C'
     *                                    (ldv, m) if storev = 'R' and side = 'L'
     *                                    (ldv, n) if storev = 'R' and side = 'R'
     *             ldv: he leading dimension of the array V.
     *                  If storev = 'C' and side = 'L', ldv >= max(1, m);
     *                  if storev = 'C' and side = 'R', ldv >= max(1, n);
     *                  if storev = 'R', ldv >= k.
     *             Tm: an array, dimension(ldt, k)
     *                The triangular k by k matrix Tm in the representation of the block reflector.
     *             ldt: The leading dimension of the array Tm. ldt >= k.
     *             C: an array, dimension(ldc, n)
     *                On entry, the m by n matrix C.
     *                On exit, C is overwritten by H * C or H^T * C or C * H or C * H^T.
     *             ldc: The leading dimension of the array C. ldc >= max(1, m).
     *             Work: an array, dimension(ldwork, k)
     *             ldwork: The leading dimension of the array Work.
     *                     If side = 'L', ldwork >= max(1, n);
     *                     if side = 'R', ldwork >= max(1, m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Further Details:
     *     The shape of the matrix V and the storage of the vectors which define the H(i) is best
     *     illustrated by the following example with n = 5 and k = 3. The elements equal to 1 are not stored;
     *     the corresponding array elements are modified but restored on exit.The rest of the array is not used.
     *     direct = 'F' and storev = 'C':        direct = 'F' and storev = 'R':
     *                  V = (1)                  V = (1 v1 v1 v1 v1)
     *                      (v1  1)                     (1 v2 v2 v2)
     *                      (v1 v2  1)                     (1 v3 v3)
     *                      (v1 v2 v3)
     *                      (v1 v2 v3)
     *     direct = 'B' and storev = 'C':         direct = 'B' and storev = 'R' :
     *                  V = (v1 v2 v3)                 V = (v1 v1  1)
     *                      (v1 v2 v3)                     (v2 v2 v2  1)
     *                      (1 v2 v3)                      (v3 v3 v3 v3  1)
     *                      (1 v3)
     *                      (1)                                                                                  */
    static void dlarfb(char const* side, char const* trans, char const* direct, char const* storev, int m, int n, int k,
            T* V, int ldv, T const* Tm, int ldt, T* C, int ldc, T* Work, int ldwork)
    {
        // Quick return if possible
        if (m < 0 || n < 0)
        {
            return;
        }
        char const* transt;
        if (toupper(trans[0]) == 'N')
        {
            transt = "Transpose";
        } else
        {
            transt = "No transpose";
        }
        int i, j, ccol, workcol;
        char upstorev = toupper(storev[0]);
        char updirect = toupper(direct[0]);
        char upside = toupper(side[0]);
        if (upstorev == 'C')
        {
            if (updirect == 'F')
            {
                // Let V = (V1) (first k rows)
                //         (V2)
                // where V1 is unit lower triangular.
                if (upside == 'L')
                {
                    // Form  H * C or H^T * C  where  C = (C1)
                    //                                    (C2)
                    // W = C^T * V = (C1^T * V1 + C2^T * V2)  (stored in Work)
                    // W = C1^T
                    for (j = 0; j < k; j++)
                    {
                        dcopy(n, &C[j/*+ldc*0*/], ldc, &Work[/*0+*/ldwork * j], 1);
                    }
                    // W : = W * V1
                    dtrmm("Right", "Lower", "No transpose", "Unit", n, k, ONE, V, ldv, Work, ldwork);
                    if (m > k)
                    {
                        // W = W + C2^T * V2
                        dgemm("Transpose", "No transpose", n, k, m - k, ONE, &C[k/*+ldc*0*/], ldc, &V[k/*+ldv*0*/], ldv, ONE, Work, ldwork);
                    }
                    // W = W * Tm^T or W * Tm
                    dtrmm("Right", "Upper", transt, "Non-unit", n, k, ONE, Tm, ldt, Work, ldwork);
                    // C = C - V * W^T
                    if (m > k)
                    {
                        // C2 = C2 - V2 * W^T
                        dgemm("No transpose", "Transpose", m - k, n, k, -ONE, &V[k/*+ldv*0*/], ldv, Work, ldwork, ONE, &C[k/*+ldc*0*/], ldc);
                    }
                    // W = W * V1^T
                    dtrmm("Right", "Lower", "Transpose", "Unit", n, k, ONE, V, ldv, Work, ldwork);
                    // C1 = C1 - W^T
                    for (j = 0; j < k; j++)
                    {
                        workcol = ldwork*j;
                        for (i = 0; i < n; i++)
                        {
                            C[j + ldc * i] -= Work[i + workcol];
                        }
                    }
                } else if (upside == 'R')
                {
                    // Form  C * H or C * H^T  where  C = (C1  C2)
                    // W = C * V = (C1*V1 + C2*V2)  (stored in Work)
                    // W = C1
                    for (j = 0; j < k; j++)
                    {
                        dcopy(m, &C[/*0+*/ldc * j], 1, &Work[/*0+*/ldwork * j], 1);
                    }
                    // W = W * V1
                    dtrmm("Right", "Lower", "No transpose", "Unit", m, k, ONE, V, ldv, Work, ldwork);
                    if (n > k)
                    {
                        // W = W + C2 * V2
                        dgemm("No transpose", "No transpose", m, k, n - k, ONE, &C[/*0+*/ldc * k], ldc, &V[k/*+ldv*0*/], ldv, ONE, Work, ldwork);
                    }
                    // W = W * Tm or W * Tm^T
                    dtrmm("Right", "Upper", trans, "Non-unit", m, k, ONE, Tm, ldt, Work, ldwork);
                    // C = C - W * V^T
                    if (n > k)
                    {
                        // C2 = C2 - W * V2^T
                        dgemm("No transpose", "Transpose", m, n - k, k, -ONE, Work, ldwork, &V[k/*+ldv*0*/], ldv, ONE, &C[/*0+*/ldc * k], ldc);
                    }
                    // W = W * V1^T
                    dtrmm("Right", "Lower", "Transpose", "Unit", m, k, ONE, V, ldv, Work, ldwork);
                    // C1 = C1 - W
                    for (j = 0; j < k; j++)
                    {
                        ccol = ldc*j;
                        workcol = ldwork*j;
                        for (i = 0; i < m; i++)
                        {
                            C[i + ccol] -= Work[i + workcol];
                        }
                    }
                }
            } else
            {
                // Let V = (V1)
                //         (V2) (last k rows)
                // where V2 is unit upper triangular.
                if (upside == 'L')
                {
                    // Form  H * C or H^T * C  where  C = (C1)
                    //                                    (C2)
                    // W = C^T * V = (C1^T * V1 + C2^T * V2)  (stored in Work)
                    // W = C2^T
                    for (j = 0; j < k; j++)
                    {
                        dcopy(n, &C[m - k + j/*+ldc*0*/], ldc, &Work[/*0+*/ldwork * j], 1);
                    }
                    // W = W * V2
                    dtrmm("Right", "Upper", "No transpose", "Unit", n, k, ONE, &V[m - k/*+ldv*0*/], ldv, Work, ldwork);
                    if (m > k)
                    {
                        // W = W + C1^T * V1
                        dgemm("Transpose", "No transpose", n, k, m - k, ONE, C, ldc, V, ldv, ONE, Work, ldwork);
                    }
                    // W = W * Tm^T or W * Tm
                    dtrmm("Right", "Lower", transt, "Non-unit", n, k, ONE, Tm, ldt, Work, ldwork);
                    // C = C - V * W^T
                    if (m > k)
                    {
                        // C1 = C1 - V1 * W^T
                        dgemm("No transpose", "Transpose", m - k, n, k, -ONE, V, ldv, Work, ldwork, ONE, C, ldc);
                    }
                    // W = W * V2^T
                    dtrmm("Right", "Upper", "Transpose", "Unit", n, k, ONE, &V[m - k/*+ldv*0*/], ldv, Work, ldwork);
                    // C2 = C2 - W^T
                    for (j = 0; j < k; j++)
                    {
                        ccol = m - k + j;
                        workcol = ldwork*j;
                        for (i = 0; i < n; i++)
                        {
                            C[ccol + ldc * i] -= Work[i + workcol];
                        }
                    }
                } else if (upside == 'R')
                {
                    // Form  C * H or C * H^T  where  C = (C1  C2)
                    // W = C * V = (C1*V1 + C2*V2)  (stored in Work)
                    // W = C2
                    ccol = ldc * (n - k);
                    for (j = 0; j < k; j++)
                    {
                        dcopy(m, &C[/*0+*/ccol + ldc * j], 1, &Work[/*0+*/ldwork * j], 1);
                    }
                    // W = W * V2
                    dtrmm("Right", "Upper", "No transpose", "Unit", m, k, ONE, &V[n - k/*+ldv*0*/], ldv, Work, ldwork);
                    if (n > k)
                    {
                        // W = W + C1 * V1
                        dgemm("No transpose", "No transpose", m, k, n - k, ONE, C, ldc, V, ldv, ONE, Work, ldwork);
                    }
                    // W : = W * Tm or W * Tm^T
                    dtrmm("Right", "Lower", trans, "Non-unit", m, k, ONE, Tm, ldt, Work, ldwork);
                    // C = C - W * V^T
                    if (n > k)
                    {
                        // C1 = C1 - W * V1^T
                        dgemm("No transpose", "Transpose", m, n - k, k, -ONE, Work, ldwork, V, ldv, ONE, C, ldc);
                    }
                    // W = W * V2^T
                    dtrmm("Right", "Upper", "Transpose", "Unit", m, k, ONE, &V[n - k/*+ldv*0*/], ldv, Work, ldwork);
                    // C2 = C2 - W
                    for (j = 0; j < k; j++)
                    {
                        ccol = ldc * (n - k + j);
                        workcol = ldwork*j;
                        for (i = 0; i < m; i++)
                        {
                            C[i + ccol] -= Work[i + workcol];
                        }
                    }
                }
            }
        } else if (upstorev == 'R')
        {
            if (updirect == 'F')
            {
                // Let V = (V1  V2)   (V1: first k columns)
                // where V1 is unit upper triangular.
                if (upside == 'L')
                {
                    // Form H * C or H^T * C where C = (C1)
                    //                                 (C2)
                    // W = C^T * V^T = (C1^T * V1^T + C2^T * V2^T) (stored in Work)
                    // W = C1^T
                    for (j = 0; j < k; j++)
                    {
                        dcopy(n, &C[j/*+ldc*0*/], ldc, &Work[/*0+*/ldwork * j], 1);
                    }
                    // W = W * V1^T
                    dtrmm("Right", "Upper", "Transpose", "Unit", n, k, ONE, V, ldv, Work, ldwork);
                    if (m > k)
                    {
                        // W = W + C2^T * V2^T
                        dgemm("Transpose", "Transpose", n, k, m - k, ONE, &C[k/*+ldc*0*/], ldc, &V[/*0+*/ldv * k], ldv, ONE, Work, ldwork);
                    }
                    // W = W * Tm^T or W * Tm
                    dtrmm("Right", "Upper", transt, "Non-unit", n, k, ONE, Tm, ldt, Work, ldwork);
                    // C = C - V^T * W^T
                    if (m > k)
                    {
                        // C2 = C2 - V2^T * W^T
                        dgemm("Transpose", "Transpose", m - k, n, k, -ONE, &V[/*0+*/ldv * k], ldv, Work, ldwork, ONE, &C[k/*+ldc*0*/], ldc);
                    }
                    // W = W * V1
                    dtrmm("Right", "Upper", "No transpose", "Unit", n, k, ONE, V, ldv, Work, ldwork);
                    // C1 = C1 - W^T
                    for (j = 0; j < k; j++)
                    {
                        workcol = ldwork*j;
                        for (i = 0; i < n; i++)
                        {
                            C[j + ldc * i] -= Work[i + workcol];
                        }
                    }
                } else if (upside == 'R')
                {
                    // Form C * H or C * H^T where  C = (C1  C2)
                    // W = C * V^T = (C1*V1^T + C2*V2^T)  (stored in Work)
                    // W = C1
                    for (j = 0; j < k; j++)
                    {
                        dcopy(m, &C[/*0+*/ldc * j], 1, &Work[/*0+*/ldwork * j], 1);
                    }
                    // W = W * V1^T
                    dtrmm("Right", "Upper", "Transpose", "Unit", m, k, ONE, V, ldv, Work, ldwork);
                    if (n > k)
                    {
                        // W = W + C2 * V2^T
                        dgemm("No transpose", "Transpose", m, k, n - k, ONE, &C[/*0+*/ldc * k], ldc, &V[/*0+*/ldv * k], ldv, ONE, Work, ldwork);
                    }
                    // W = W * Tm or W * Tm^T
                    dtrmm("Right", "Upper", trans, "Non-unit", m, k, ONE, Tm, ldt, Work, ldwork);
                    // C = C - W * V
                    if (n > k)
                    {
                        // C2 = C2 - W * V2
                        dgemm("No transpose", "No transpose", m, n - k, k, -ONE, Work, ldwork, &V[/*0+*/ldv * k], ldv, ONE, &C[/*0+*/ldc * k], ldc);
                    }
                    // W = W * V1
                    dtrmm("Right", "Upper", "No transpose", "Unit", m, k, ONE, V, ldv, Work, ldwork);
                    // C1 = C1 - W
                    for (j = 0; j < k; j++)
                    {
                        ccol = ldc*j;
                        workcol = ldwork*j;
                        for (i = 0; i < m; i++)
                        {
                            C[i + ccol] -= Work[i + workcol];
                        }
                    }
                }
            } else
            {
                // Let V = (V1  V2)   (V2: last k columns)
                // where V2 is unit lower triangular.
                if (upside == 'L')
                {
                    // Form H * C or H^T * C where C = (C1)
                    //                                 (C2)
                    // W = C^T * V^T = (C1^T * V1^T + C2^T * V2^T) (stored in Work)
                    // W = C2^T
                    for (j = 0; j < k; j++)
                    {
                        dcopy(n, &C[m - k + j/*+ldc*0*/], ldc, &Work[/*0+*/ldwork * j], 1);
                    }
                    // W = W * V2^T
                    dtrmm("Right", "Lower", "Transpose", "Unit", n, k, ONE, &V[/*0+*/ldv * (m - k)], ldv, Work, ldwork);
                    if (m > k)
                    {
                        // W = W + C1^T * V1^T
                        dgemm("Transpose", "Transpose", n, k, m - k, ONE, C, ldc, V, ldv, ONE, Work, ldwork);
                    }
                    // W = W * Tm^T or W * Tm
                    dtrmm("Right", "Lower", transt, "Non-unit", n, k, ONE, Tm, ldt, Work, ldwork);
                    // C = C - V^T * W^T
                    if (m > k)
                    {
                        // C1 = C1 - V1^T * W^T
                        dgemm("Transpose", "Transpose", m - k, n, k, -ONE, V, ldv, Work, ldwork, ONE, C, ldc);
                    }
                    // W = W * V2
                    dtrmm("Right", "Lower", "No transpose", "Unit", n, k, ONE, &V[/*0+*/ldv * (m - k)], ldv, Work, ldwork);
                    // C2 = C2 - W^T
                    for (j = 0; j < k; j++)
                    {
                        ccol = m - k + j;
                        workcol = ldwork*j;
                        for (i = 0; i < n; i++)
                        {
                            C[ccol + ldc * i] -= Work[i + workcol];
                        }
                    }
                } else if (upside == 'R')
                {
                    // Form  C * H or C * H^T where C = (C1  C2)
                    // W = C * V^T = (C1*V1^T + C2*V2^T) (stored in Work)
                    // W = C2
                    ccol = ldc * (n - k);
                    for (j = 0; j < k; j++)
                    {
                        dcopy(m, &C[/*0+*/ccol + ldc * j], 1, &Work[/*0+*/ldwork * j], 1);
                    }
                    // W = W * V2^T
                    dtrmm("Right", "Lower", "Transpose", "Unit", m, k, ONE, &V[/*0+*/ldv * (n - k)], ldv, Work, ldwork);
                    if (n > k)
                    {
                        // W = W + C1 * V1^T
                        dgemm("No transpose", "Transpose", m, k, n - k, ONE, C, ldc, V, ldv, ONE, Work, ldwork);
                    }
                    // W = W * Tm or W * Tm^T
                    dtrmm("Right", "Lower", trans, "Non-unit", m, k, ONE, Tm, ldt, Work, ldwork);
                    // C = C - W * V
                    if (n > k)
                    {
                        // C1 = C1 - W * V1
                        dgemm("No transpose", "No transpose", m, n - k, k, -ONE, Work, ldwork, V, ldv, ONE, C, ldc);
                    }
                    // W = W * V2
                    dtrmm("Right", "Lower", "No transpose", "Unit", m, k, ONE, &V[/*0+*/ldv * (n - k)], ldv, Work, ldwork);
                    // C1 = C1 - W
                    for (j = 0; j < k; j++)
                    {
                        ccol = ldc * (n - k + j);
                        workcol = ldwork*j;
                        for (i = 0; i < m; i++)
                        {
                            C[i + ccol] -= Work[i + workcol];
                        }
                    }
                }
            }
        }
    }

    /* dlarfg generates a real elementary reflector H of order n, such that
     *           H * (alpha) = (beta), H^T * H = I.
     *               (x    )   (0   )
     * where alpha and beta are scalars, and x is an (n - 1)-element real vector.
     * H is represented in the form
     *           H = I - tau * (1) * (1 v^T),
     *                         (v)
     * where tau is a real scalar and v is a real (n - 1)-element vector.
     * If the elements of x are all zero, then tau = 0 and H is taken to be the unit matrix.
     * Otherwise  1 <= tau <= 2.
     * Parameters: n: The order of the elementary reflector.
     *             alpha: On entry, the value alpha.
     *                    On exit, it is overwritten with the value beta.
     *             x: array, dimension (1 + (n - 2)*abs(incx))
     *                On entry, the vector x.
     *                On exit, it is overwritten with the vector v.
     *             incx: The increment between elements of x. incx > 0.
     *             tau:  The value tau.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.																	*/
    static void dlarfg(int n, T& alpha, T* x, int incx, T& tau)
    {
        int j, knt;
        T beta, rsafmin, safmin, xnorm;
        if (n <= 1)
        {
            tau = ZERO;
            return;
        }
        xnorm = dnrm2(n - 1, x, incx);
        if (xnorm == ZERO)
        {
            // H = I
            tau = ZERO;
        } else
        {
            // general case
            beta = -dlapy2(alpha, xnorm) * T((ZERO <= alpha) - (alpha < ZERO));
            safmin = dlamch("SafeMin") / dlamch("Epsilon");
            knt = 0;
            if (fabs(beta) < safmin)
            {
                // xnorm, beta may be inaccurate; scale x and recompute them
                rsafmin = ONE / safmin;
                do
                {
                    knt++;
                    dscal(n - 1, rsafmin, x, incx);
                    beta = beta*rsafmin;
                    alpha = alpha*rsafmin;
                } while (fabs(beta) < safmin);
                // New beta is at most 1, at least SAFMIN
                xnorm = dnrm2(n - 1, x, incx);
                beta = -dlapy2(alpha, xnorm) * T((ZERO <= alpha) - (alpha < ZERO));
            }
            tau = (beta - alpha) / beta;
            dscal(n - 1, ONE / (alpha - beta), x, incx);
            // If alpha is subnormal, it may lose relative accuracy
            for (j = 0; j < knt; j++)
            {
                beta = beta*safmin;
            }
            alpha = beta;
        }
    }

    /* dlarft forms the triangular factor A of a real block reflector H
     * of order n, which is defined as a product of k elementary reflectors.
     *     If direct = 'F', H = H(1) H(2) . ..H(k) and A is upper triangular;
     *     If direct = 'B', H = H(k) . ..H(2) H(1) and A is lower triangular.
     *     If storev = 'C', the vector which defines the elementary reflector
     *                      H(i) is stored in the i - th column of the array V, and
     *                      H = I - V * A * V^T
     *     If storev = 'R', the vector which defines the elementary reflector
     *                      H(i) is stored in the i - th row of the array V, and
     *                      H = I - V^T * A * V
     * Parameters: direct:  Specifies the order in which the elementary reflectors are
     *                      multiplied to form the block reflector:
     *                      'F': H = H(1) H(2) . ..H(k) (Forward)
     *                      'B': H = H(k) . ..H(2) H(1) (Backward)
     *             storev: Specifies how the vectors which define the elementary
     *                     reflectors are stored(see also Further Details) :
     *                     'C': columnwise
     *                     'R': rowwise
     *             n: The order of the block reflector H. n >= 0.
     *             k: The order of the triangular factor A(= the number of elementary reflectors).
     *                k >= 1.
     *             V: an array, dimension (ldv, k) if storev = 'C'
     *                                    (ldv, n) if storev = 'R'
     *             ldv: The leading dimension of the array V.
     *                  If storev = 'C', ldv >= max(1, n); if storev = 'R', ldv >= k.
     *             tau: an array, dimension(k)
     *                  tau(i) must contain the scalar factor of the elementary reflector H(i).
     *             A: an array, dimension(lda, k)
     *                The k by k triangular factor A of the block reflector. If direct = 'F', A is upper triangular;
     *                if direct = 'B', A is lower triangular. The rest of the array is not used.
     *             lda: The leading dimension of the array A. lda >= k.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Further Details:
     *     The shape of the matrix V and the storage of the vectors which define the H(i) is best illustrated by
     *     the following example with n = 5 and k = 3. The elements equal to 1 are not stored.
     * direct = 'F' and storev = 'C':         direct = 'F' and storev = 'R':
     *                V = (1)                 V = (1 v1 v1 v1 v1)
     *                    (v1  1)                    (1 v2 v2 v2)
     *                    (v1 v2  1)                    (1 v3 v3)
     *                    (v1 v2 v3)
     *                    (v1 v2 v3)
     * direct = 'B' and storev = 'C':         direct = 'B' and storev = 'R' :
     *                V = (v1 v2 v3)                 V = (v1 v1  1)
     *                    (v1 v2 v3)                     (v2 v2 v2  1)
     *                    (1 v2 v3)                      (v3 v3 v3 v3  1)
     *                    (1 v3)
     *                    (1)                                                                                            */
    static void dlarft(char const* direct, char const* storev, int n, int k, T const* V, int ldv, T const* tau, T* A, int lda)
    {
        // Quick return if possible
        if (n == 0)
        {
            return;
        }
        char updirect = toupper(direct[0]);
        char upstorev = toupper(storev[0]);
        int i, j, prevlastv, lastv, tcoli, vcol;
        if (updirect == 'F')
        {
            prevlastv = n - 1;
            for (i = 0; i < k; i++)
            {
                tcoli = lda*i;
                if (i > prevlastv)
                {
                    prevlastv = i;
                }
                if (tau[i] == ZERO)
                {
                    // H(i) = I
                    for (j = 0; j <= i; j++)
                    {
                        A[j + tcoli] = ZERO;
                    }
                } else
                {
                    // general case
                    if (upstorev == 'C')
                    {
                        vcol = ldv*i;
                        // Skip any trailing zeros.
                        for (lastv = n - 1; lastv > i; lastv--)
                        {
                            if (V[lastv + vcol] != ZERO)
                            {
                                break;
                            }
                        }
                        for (j = 0; j < i; j++)
                        {
                            A[j + tcoli] = -tau[i] * V[i + ldv * j];
                        }
                        j = lastv < prevlastv ? lastv : prevlastv;
                        // T[0:i-1, i] = -tau[i] * V[i:j, 0:i-1]^T * V[i:j, i]
                        dgemv("Transpose", j - i, i, -tau[i], &V[i + 1 /*+ldv*0*/], ldv, &V[i + 1 + vcol], 1, ONE, &A[/*0+*/tcoli], 1);
                    } else
                    {
                        // Skip any trailing zeros.
                        for (lastv = n - 1; lastv > i; lastv--)
                        {
                            if (V[i + ldv * lastv] != ZERO)
                            {
                                break;
                            }
                        }
                        vcol = ldv*i;
                        for (j = 0; j < i; j++)
                        {
                            A[j + tcoli] = -tau[i] * V[j + vcol];
                        }
                        j = lastv < prevlastv ? lastv : prevlastv;
                        // T[0:i-1, i] = -tau[i] * V[0:i-1, i:j] * V[i, i:j]^T
                        dgemv("No transpose", i, j - i, -tau[i], &V[/*0+*/vcol + ldv], ldv, &V[i + vcol + ldv], ldv, ONE, &A[/*0+*/tcoli], 1);
                    }
                    // T[0:i-1, i] = T[0:i-1, 0:i-1] * T[0:i-1, i]
                    dtrmv("Upper", "No transpose", "Non-unit", i, A, lda, &A[/*0+*/tcoli], 1);
                    A[i + tcoli] = tau[i];
                    if (i > 0)
                    {
                        if (lastv > prevlastv)
                        {
                            prevlastv = lastv;
                        }
                    } else
                    {
                        prevlastv = lastv;
                    }
                }
            }
        } else
        {
            prevlastv = 0;
            for (i = k - 1; i >= 0; i--)
            {
                tcoli = lda*i;
                if (tau[i] == ZERO)
                {
                    // H(i) = I
                    for (j = i; j < k; j++)
                    {
                        A[j + tcoli] = ZERO;
                    }
                } else
                {
                    // general case
                    if (i < k - 1)
                    {
                        if (upstorev == 'C')
                        {
                            vcol = ldv*i;
                            // Skip any leading zeros.
                            for (lastv = 0; lastv < i; lastv++)
                            {
                                if (V[lastv + vcol] != ZERO)
                                {
                                    break;
                                }
                            }
                            vcol = n - k - i; // misuse: not a col, but a row
                            for (j = (i + 1); j < k; j++)
                            {
                                A[j + tcoli] = -tau[i] * V[vcol + ldv * j];
                            }
                            j = lastv > prevlastv ? lastv : prevlastv;
                            // T[i+1:k-1, i] = -tau[i] * V[j:n-k+i, i+1:k-1]^T * V[j:n-k+i, i]
                            dgemv("Transpose", vcol - j, k - 1 - i, -tau[i], &V[j + ldv * (i + 1)], ldv, &V[j + ldv * i], 1, ONE, &A[i + 1 + tcoli], 1);
                        } else
                        {
                            // Skip any leading zeros.
                            for (lastv = 0; lastv < i; lastv++)
                            {
                                if (V[i + ldv * lastv] != ZERO)
                                {
                                    break;
                                }
                            }
                            vcol = ldv * (n - k + i);
                            for (j = (i + 1); j < k; j++)
                            {
                                A[j + tcoli] = -tau[i] * V[j + vcol];
                            }
                            j = lastv > prevlastv ? lastv : prevlastv;
                            // T[i+1:k-1, i] = -tau[i] * V[i+1:k-1, j:n-k+i] * V[i, j:n-k+i]^T
                            vcol = ldv*j;
                            dgemv("No transpose", k - 1 - i, n - k + i - j, -tau[i], &V[i + 1 + vcol], ldv, &V[i + vcol], ldv, ONE, &A[i + 1 + tcoli], 1);
                        }
                        // T[i+1:k-1, i] = T[i+1:k-1, i+1:k-1] * T[i+1:k-1, i]
                        dtrmv("Lower", "No transpose", "Non-unit", k - i - 1, &A[i + 1 + lda * (i + 1)], lda, &A[i + 1 + tcoli], 1);
                        if (i > 0)
                        {
                            if (lastv < prevlastv)
                            {
                                prevlastv = lastv;
                            }
                        } else
                        {
                            prevlastv = lastv;
                        }
                    }
                    A[i + tcoli] = tau[i];
                }
            }
        }
    }

    /* dlarnv returns a vector of n random real numbers from a uniform or normal distribution.
     * Parameters: idist: Specifies the distribution of the random numbers:
     *                    ==1: uniform (0,1)
     *                    ==2: uniform (-1,1)
     *                    ==3: normal (0,1)
     *             iseed: an integer array, dimension (4)
     *                    On entry, the seed of the random number generator; the array elements
     *                              must be between 0 and 4095, and iseed[3] must be odd.
     *                    On exit, the seed is updated.
     *             n: The number of random numbers to be generated.
     *             x: an array, dimension (n)
     *                The generated random numbers.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Further Details:
     *     This routine calls the auxiliary routine dlaruv to generate random real numbers from a
     *     uniform (0,1) distribution, in batches of up to 128 using vectorisable code. The Box-
     *     Muller method is used to transform numbers from a uniform to a normal distribution.   */
    static void dlarnv(int idist, int* iseed, int n, T* x)
    {
        const int LV = 128;
        const T TWOPI = 6.2831853071795864769252867663;
        int i, il, il2, iv;
        T* u[LV];
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

    /* dlartg generates a plane rotation so that
     *     [  cs  sn  ]  .  [ f ]  =  [ r ]   where cs^2 + sn^2 = 1.
     *     [ -sn  cs  ]     [ g ]     [ 0 ]
     * This is a slower, more accurate version of the BLAS1 routine DROTG,
     * with the following other differences:
     *     f and g are unchanged on return.
     *     If g==0, then cs==1 and sn==0.
     *     If f==0 and g!=0, then cs==0 and sn==1 without doing any floating point operations
     *         (saves work in DBDSQR when there are zeros on the diagonal).
     *     If f exceeds g in magnitude, cs will be positive.
     * Parameters: f: The first component of vector to be rotated.
     *             g: The second component of vector to be rotated.
     *             cs: The cosine of the rotation.
     *             sn: The sine of the rotation.
     *             r: The nonzero component of the rotated vector.
     * This version has a few statements commented out for thread safety (machine parameters are
     *   computed on each entry). 10 feb 03, SJH.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlartg(T f, T g, T& cs, T& sn, T& r)
    {
        //static bool first = true;
        /*static*/ T safmn2, safmx2;
        //if (first)
        //{
            T safmin = dlamch("S");
            T eps = dlamch("E");
            T base = dlamch("B");
            safmn2 = std::pow(base, std::log(safmin/eps) / std::log(base) / TWO);
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
            T f1 = f;
            T g1 = g;
            T scale = fabs(f1);
            scale = scale>fabs(g1)?scale:fabs(g1);
            int i, count = 0;
            if (scale>=safmx2)
            {
                do
                {
                    count++;
                    f1 *= safmn2;
                    g1 *= safmn2;
                    scale = fabs(f1);
                    scale = scale>fabs(g1)?scale:fabs(g1);
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
                    f1 *= safmx2
                    g1 *= safmx2
                    scale = fabs(f1);
                    scale = scale>fabs(g1)?scale:fabs(g1);
                } while (scale<=safmn2)
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

    /* dlaruv returns a vector of n random real numbers from a uniform (0,1) distribution (n<=128).
     * This is an auxiliary routine called by dlarnv and zlarnv.
     * Parameters: iseed: an integer array, dimension (4)
     *                    On entry, the seed of the random number generator; the array elements
     *                              must be between 0 and 4095, and iseed[3] must be odd.
     *                    On exit, the seed is updated.
     *             n: The number of random numbers to be generated. n<=128.
     *             x: an array, dimension (n)
     *                The generated random numbers.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Further Details:
     *    This routine uses a multiplicative congruential method with modulus 2^48 and multiplier
     *    33952834046453 (see G.S.Fishman, 'Multiplicative congruential random number generators
     *    with modulus 2^b: an exhaustive analysis for b = 32 and a partial analysis for b = 48',
     *    Math. Comp. 189, pp 331-344, 1990).
     *    48-bit integers are stored in 4 integer array elements with 12 bits per element. Hence
     *    the routine is portable across machines with integers of 32 bits or more.              */
    static void dlaruv(int* iseed, int n, T* x)
    {
        const int LV = 128;
        const int IPW2 = 4096;
        const T R = ONE / IPW2;
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
        int i, imul4, it1, it2, it3, it4;
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
                it1 = it1 + i1*MM[3+im4] + i2*MM[2+im4] + i3*MM[1+im4] + i4*MM[/*0+*/im4];
                it1 = it1 % IPW2;
                // Convert 48-bit integer to a real number in the interval (0,1)
                x[i] = R*(T(it1)+R*(T(it2)+R*(T(it3)+R*T(it4))));
                if (x[i]==ONE)
                {
                    // If a real number has n bits of precision, and the first n bits of the 48-bit
                    // integer above happen to be all 1 (which will occur about once every 2^n calls),
                    // then X[i] will be rounded to exactly 1.0.
                    // Since X[i] is not supposed to return exactly 0.0 or 1.0, the statistically
                    // correct thing to do in this situation is simply to iterate again.
                    // N.B. the case X[i] = 0.0 should not be possible.
                    i1 += 2;
                    i2 += 2;
                    i3 += 2;
                    i4 += 2;
                }
            } while (x[i]==ONE)
        }
        // Return final value of seed
        iseed[0] = it1;
        iseed[1] = it2;
        iseed[2] = it3;
        iseed[3] = it4;
    }

    /* dlas2 computes the singular values of the 2-by-2 matrix
     *     [ f  g ]
     *     [ 0  h ].
     * On return, ssmin is the smaller singular value and ssmax is the larger singular value.
     * Parameters: f: The [0,0] element of the 2-by-2 matrix.
     *             g: The [0,1] element of the 2-by-2 matrix.
     *             h: The [1,1] element of the 2-by-2 matrix.
     *             ssmin: The smaller singular value.
     *             ssmax: The larger singular value.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Further Details:
     *     Barring over/underflow, all output quantities are correct to within a few units in the
     *     last place (ulps), even in the absence of a guard digit in addition/subtraction.
     *     In IEEE arithmetic, the code works correctly if one matrix element is infinite.
     *     Overflow will not occur unless the largest singular value itself overflows, or is within
     *     a few ulps of overflow. (On machines with partial overflow, like the Cray, overflow may
     *     occur if the largest singular value is within a factor of 2 of overflow.)
     *     Underflow is harmless if underflow is gradual. Otherwise, results may correspond to a
     *     matrix modified by perturbations of size near the underflow threshold.                */
    static void dlas2(T f, T g, T h, T& ssmin, T& ssmax)
    {
        T fa = fabs(f);
        T ga = fabs(g);
        T ha = fabs(h);
        T fhmn, fhmx;
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
            T as, at, au, c;
            if (ga<fhmx)
            {
                as = ONE + fhmn/fhmx;
                at = (fhmx-fhmn) / fhmx;
                au = ga / fhmx;
                au = au*au;
                c = TWO / (srd::sqrt(as*as+au)+std::sqrt(at*at+au));
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
                    ssmin = (fhmn*c)*au;
                    ssmin = ssmin + ssmin;
                    ssmax = ga / (c+c);
                }
            }
        }
    }

    /* dlascl multiplies the m by n real matrix A by the real scalar cto/cfrom. This is done
     * without over/underflow as long as the final result cto*A(I,J)/cfrom does not over/underflow.
     * type specifies that A may be full, upper triangular, lower triangular, upper Hessenberg, or
     * banded.
     * Parameters: type: indices the storage type of the input matrix.
     *                   ='G': A is a full matrix.
     *                   ='L': A is a lower triangular matrix.
     *                   ='U': A is an upper triangular matrix.
     *                   ='H': A is an upper Hessenberg matrix.
     *                   ='B': A is a symmetric band matrix with lower bandwidth kl and upper
     *                         bandwidth ku and with the only the lower half stored.
     *                   ='Q': A is a symmetric band matrix with lower bandwidth kl and upper
     *                         bandwidth ku and with the only the upper half stored.
     *                   ='Z': A is a band matrix with lower bandwidth kl and upper bandwidth ku.
     *                         See DGBTRF for storage details.
     *             kl: The lower bandwidth of A. Referenced only if type=='B', 'Q' or 'Z'.
     *             ku: The upper bandwidth of A. Referenced only if type=='B', 'Q' or 'Z'.
     *             cfrom,
     *             cto: The matrix A is multiplied by cto/cfrom. A[i,j] is computed without
     *                  over/underflow if the final result cto*A[i,j]/cfrom can be represented
     *                  without over/underflow. cfrom must be nonzero.
     *             m: The number of rows of the matrix A. m>=0.
     *             n: The number of columns of the matrix A. n>=0.
     *             A: an array, dimension (lda,n)
     *                The matrix to be multiplied by cto/cfrom. See type for the storage type.
     *             lda: The leading dimension of the array A.
     *                  If type=='G', 'L', 'U', 'H': lda >= max(1,m);
     *                     type=='B'               : lda >= kl+1;
     *                     type=='Q'               : lda >= ku+1;
     *                     type=='Z'               : lda >= 2*kl+ku+1.
     *             info: ==0: successful exit
     *                    <0: if info = -i, the i-th argument had an illegal value.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2016                                                                            */
    static void dlascl(char const* type, int kl, int ku, T cfrom, T cto, int m, int n, T* A,
                       int lda, int& info)
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
            else if ((itype==4 && lda<(kl+1)) || (itype==5 && lda<(ku+1)) || (itype==6 && lda<(2*kl+ku+1)))
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
        T smlnum = dlamch("S");
        T bignum = ONE / smlnum;
        T cfromc = cfrom;
        T ctoc = cto;
        int i, j, ldaj, k1, k2, k3, k4;
        T cfrom1, cto1, mul;
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
                    for (i=(k1>0?k1:0); i<=ku; i++)
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
                    for (i=(k2>kl?k2:kl); i<k3 && i<(k4-j); i++)
                    {
                        A[i+ldaj] *= mul;
                    }
                }
            }
        } while (!done)
    }

    /* dlaset initializes an m-by-n matrix A to beta on the diagonal and alpha on the offdiagonals.
     * Parameters: uplo: Specifies the part of the matrix A to be set.
     *                   ='U': Upper triangular part is set; the strictly lower triangular part of
     *                         A is not changed.
     *                   ='L': Lower triangular part is set; the strictly upper triangular part of
     *                         A is not changed. Otherwise:  All of the matrix A is set.
     *             m: The number of rows of the matrix A. m>=0.
     *             n: The number of columns of the matrix A. n>=0.
     *             alpha: The constant to which the offdiagonal elements are to be set.
     *             beta: The constant to which the diagonal elements are to be set.
     *             A: an array, dimension (lda,n)
     *                On exit, the leading m-by-n submatrix of A is set as follows:
     *                if uplo = 'U', A(i,j) = alpha, 1<=i<=j-1, 1<=j<=n,
     *                if uplo = 'L', A(i,j) = alpha, j+1<=i<=m, 1<=j<=n,
     *                otherwise,     A(i,j) = alpha, 1<=i<=m, 1<=j<=n, i.ne.j,
     *                and, for all uplo, A(i,i) = beta, 1<=i<=min(m,n).
     *             lda: The leading dimension of the array A. lda>=max(1,m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlaset(char const* uplo, int m, int n, T alpha, T beta, T* A, int lda)
    {
        int i, j, ldaj;
        if (toupper(uplo[0])=='U')
        {
            // Set the strictly upper triangular or trapezoidal part of the array to ALPHA.
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
            // Set the strictly lower triangular or trapezoidal part of the array to ALPHA.
            for (j=0, j<m && j<n; j++)
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
            // Set the leading m-by-n submatrix to ALPHA.
            for (j=0; j<n; j++)
            {
                ldaj = lda*j;
                for (i=0; i<m; i++)
                {
                    A[i+ldaj] = alpha;
                }
            }
        }
        // Set the first min(M,N) diagonal elements to BETA.
        for (i=0, i<m && i<n; i++)
        {
            A[i+lda*i] = beta;
        }
    }

    /* dlasq1 computes the singular values of a real n-by-n bidiagonal matrix with diagonal d and
     * off-diagonal e. The singular values are computed to high relative accuracy, in the absence
     * of denormalization, underflow and overflow. The algorithm was first presented in
     *     "Accurate singular values and differential qd algorithms" by K. V. Fernando and
     *     B. N. Parlett, Numer. Math., Vol-67, No. 2, pp. 191-230, 1994,
     * and the present implementation is described in
     *     "An implementation of the dqds Algorithm (Positive Case)", LAPACK Working Note.
     * Parameters: n: The number of rows and columns in the matrix. n>=0.
     *             d: an array, dimension (n)
     *                On entry, d contains the diagonal elements of the bidiagonal matrix whose SVD
     *                is desired.
     *                On normal exit, d contains the singular values in decreasing order.
     *             e: an array, dimension (n)
     *                On entry, elements e[0:n-2] contain the off-diagonal elements of the
     *                bidiagonal matrix whose SVD is desired.
     *                On exit, e is overwritten.
     *             work: an array, dimension (4*n)
     *             info: =0: successful exit
     *                   <0: if info==-i, the i-th argument had an illegal value
     *                  > 0: the algorithm failed
     *                       =1, a split was marked by a positive value in e
     *                       =2, current block of Z not diagonalized after 100*n iterations
     *                           (in inner while loop). On exit d and e represent a matrix with the
     *                           same singular values which the calling subroutine could use to
     *                           finish the computation, or even feed back into dlasq1
     *                       =3, termination criterion of outer while loop not met
     *                           (program created more than n unreduced blocks)
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlasq1(int n, T* d, T* e, T* work, int& info)
    {
        int i, iinfo;
        T eps, scale, safmin, sigmn, sigmx, temp;
        info = 0
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
            temp = fabs(e[i]));
            sigmx = sigmx>temp?sigmx:temp;
        }
        d[n-1] = fabs(d[n-1]);
        // Early return if SIGMX is zero (matrix is already diagonal).
        if (sigmx==ZERO)
        {
            dlasrt("D", n, d, iinfo);
            return;
        }
        for (i=0; i<n; i++)
        {
           sigmx = sigmx>d[i]?sigmx:d[i];
        }
        // Copy d and e into work (in the Z format) and scale (squaring the input data makes
        // scaling by a power of the radix pointless).
        eps = dlamch("Precision");
        safmin = dlamch("Safe minimum");
        scale = std::sqrt(eps/safmin);
        dcopy(n, d, 1, work[0], 2);
        dcopy(n-1, e, 1, work[1], 2);
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

    /* dlasq2 computes all the eigenvalues of the symmetric positive definite tridiagonal matrix
     * associated with the qd array Z to high relative accuracy are computed to high relative
     * accuracy, in the absence of denormalization, underflow and overflow.
     * To see the relation of Z to the tridiagonal matrix, let L be a unit lower bidiagonal matrix
     * with subdiagonals Z[1,3,5,,..] and let U be an upper bidiagonal matrix with 1's above and
     * diagonal Z[0,2,4,,..]. The tridiagonal is L*U or, if you prefer, the symmetric tridiagonal
     * to which it is similar.
     * Note: dlasq2 defines a logical variable, IEEE, which is true on machines which follow
     * ieee-754 floating-point standard in their handling of infinities and NaNs, and false
     * otherwise. This variable is passed to dlasq3.
     * Parameters: n: The number of rows and columns in the matrix. n>=0.
     *             Z: an array, dimension (4*n)
     *                On entry Z holds the qd array. On exit, entries 1 to n hold the eigenvalues
     *                in decreasing order, Z[2*n] holds the trace, and Z[2*n+1] holds the sum of
     *                the eigenvalues. If n>2, then Z[2*n+2] holds the iteration count, Z[2*n+3]
     *                holds NDIVS/NIN^2, and Z[2*n+4] holds the percentage of shifts that failed.
     *             info: =0: successful exit
     *                   <0: if the i-th argument is a scalar and had an illegal value, then
     *                       info==-i, if the i-th argument is an array and the j-entry had an
     *                       illegal value, then info = -(i*100+j)
     *                   >0: the algorithm failed
     *                       =1, a split was marked by a positive value in Z
     *                       =2, current block of Z not diagonalized after 100*n iterations
     *                           (in inner while loop). On exit Z holds a qd array with the same
     *                           eigenvalues as the given Z.
     *                       =3, termination criterion of outer while loop not met
     *                           (program created more than n unreduced blocks)
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016
     * Further Details:
     *     Local Variables: i0:n0 defines a current unreduced segment of Z. The shifts are
     *     accumulated in SIGMA. Iteration count is in iter. Ping-pong is controlled by PP
     *     (alternates between 0 and 1).                                                         */
    static void dlasq2(int n, int Z, int& info)
    {
        const T CBIAS  = T(1.50);
        const T HALF   = T(0.5);
        const T FOUR   = T(4.0);
        const T HUNDRD = T(100.0);
        bool ieee, loopbreak;
        int i0, i1, i4, iinfo, ipn4, iter, iwhila, iwhilb, k, kmin, n0, n1, nbig, ndiv, nfail, pp,
            splt, ttype;
        T d, dee, deemin, desig, dmin, dmin1, dmin2, dn, dn1, dn2, e, emax, emin, eps, g, oldemn,
          qmax, qmin, s, safmin, sigma, tt, tau, temp, tol, tol2, trace, zmax, tempe, tempq;
        // Test the input arguments. (in case dlasq2 is not called by dlasq1)
        info = 0;
        eps = dlamch("Precision");
        safmin = dlamch("Safe minimum");
        tol = eps * HUNDRD;
        tol2 = tol * tol;
        if (n<0){
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
                }else{
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
            qmax = qmax>Z[k]?qmax:Z[k];
            emin = emin<Z[k+1]?emin:Z[k+1];
            zmax = qmax>zmax?qmax:zmax;
            zmax = zmax>Z[k+1]?zmax:Z[k+1];
        }
        if (Z[2*n-2]<ZERO)
        {
            info = -(200+2*n-1);
            xerbla("DLASQ2", 2);
            return;
        }
        d += Z[2*n-2];
        qmax = qmax>Z[2*n-2]?qmax:Z[2*n-2];
        zmax = qmax>zmax?qmax:zmax;
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
        for (k=2*n-1; k>=1, k-=2)
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
                    Z[i4-2*pp+3] = Z[i4+4] * (Z[i4+2]/Z(i4-2*pp+2));
                    d = Z[i4+4] * (d/Z[i4-2*pp+1]);
                }
                emin = emin<Z[i4-2*pp+3]?emin:Z[i4-2*pp+3];
            }
            Z[4*n0-pp+1] = d;
            // Now find qmax.
            qmax = Z[4*n0-pp+1];
            for (i4=4*i0-pp+2; i4<=4*n0-pp-2; i4+=4)
            {
                qmax = qmax>Z[i4+3]?qmax:Z[i4+3];
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
        NDIV = 2*(n0-i0);
        for (iwhila=0; iwhila<=n; iwhila++)
        {
            if (n0<0)
            {
                break;
            }
            // While array unfinished do
            // E[n0] holds the value of SIGMA when submatrix in i0:n0 splits from the rest of the
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
            // Find Gershgorin-type bound if Q's much greater than E's.
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
                    qmin = qmin<Z[i4]?qmin:Z[i4];
                    emax = emax>Z[i4-2]?emax:Z[i4-2];
                }
                temp = Z[i4-4] + Z[i4-2];
                qmax = qmax>temp?qmax:temp;
                emin = emin<Z[i4-2]?emin:Z[i4-2];
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
                    for(i4=4*i0; i4<=2*(i0+n0-1); i4+=4)
                    {
                        temp         = Z[i4];
                        Z[i4]        = Z[ipn4-i4-4];
                        Z[ipn4-i4-4] = temp;
                        temp         = Z[i4+1];
                        Z[i4+1]      = Z[ipn4-i4-3];
                        Z[ipn4-i4-3] = temp;
                        temp          = Z[i4+2];
                        Z[i4+2]       = Z[ipn4-i4-6];
                        Z[ipn4-i4-6] = temp;
                        temp         = Z[i4+3];
                        Z[i4+3]      = Z[ipn4-i4-5];
                        Z[ipn4-i4-5] = temp;
                    }
                }
            }
            // Put -(initial shift) into DMIN.
            dmin = TWO*std::sqrt(qmin)*std::sqrt(emax) - qmin;
            dmin = dmin<ZERO?dmin:ZERO;
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
                dlasq3(i0, n0, Z, pp, dmin, sigma, desig, qmax, nfail, iter, NDIV, ieee, ttype,
                       dmin1, dmin2, dn, dn1, dn2, g, tau);
                pp = 1 - pp;
                // When EMIN is very small check for splits.
                if (pp==0 && n0-i0>=3)
                {
                    if (Z[4*n0+3]<=tol2*qmax || Z[4*n0+2]<=tol2*sigma)
                    {
                        splt = i0 -1;
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
                                qmax = qmax>Z[i4+4]?qmax:Z[i4+4];
                                emin = emin<Z[i4+2]?emin:Z[i4+2];
                                oldemn = oldemn<Z[i4+3]?oldemn:Z[i4+3];
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
            // Maximum number of iterations exceeded, restore the shift SIGMA and place the new
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
            for(k=0; k<n; k++)
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
        Z[2*n+2] = T(iter);
        Z[2*n+3] = T(NDIV) / T(n*n);
        Z[2*n+4] = HUNDRD * nfail / T(iter);
        return;
    }

    /* dlasq3 checks for deflation, computes a shift (tau) and calls dqds. In case of failure it
     * changes shifts, and tries again until output is positive.
     * Parameters: i0: First index. (note: zero-based index!)
     *             n0: Last index. (note: zero-based index!)
     *             Z: an array, dimension (4*n0+4)
     *                Z holds the qd array.
     *             pp: pp=0 for ping, pp=1 for pong.
     *                 pp=2 indicates that flipping was applied to the Z array and that the initial
     *                 tests for deflation should not be performed.
     *             dmin: Minimum value of d.
     *             sigma: Sum of shifts used in current segment.
     *             desig: Lower order part of sigma
     *             qmax: Maximum value of q.
     *             nfail: Increment nfail by 1 each time the shift was too big.
     *             iter: Increment iter by 1 for each iteration.
     *             ndiv: Increment ndiv by 1 for each division.
     *             ieee: Flag for IEEE or non IEEE arithmetic (passed to dlasq5).
     *             ttype: Shift type.
     *             dmin1,
     *             dmin2,
     *             dn,
     *             dn1,
     *             dn2,
     *             g,
     *             tau: These are passed as arguments in order to save their values between calls
     *                  to dlasq3.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2016                                                                            */
    static void dlasq3(int i0, int& n0, T* Z, int& pp, T& dmin, T& sigma, T& desig, T qmax,
                       int& nfail, int& iter, int& ndiv, bool ieee, int& ttype, T& dmin1, T& dmin2,
                       T& dn, T& dn1, T& dn2, T& g, T& tau)
    {
        const T CBIAS  = T(1.50);
        const T QURTR  = T(0.25);
        const T HALF   = T(0.5);
        const T HUNDRD = T(100.0);
        int n0in = n0;
        T eps = dlamch("Precision");
        T tol = eps*HUNDRD;
        T tol2 = tol*tol;
        int ipn4, j4, nn;
        T s, tt, temp;
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
                // Check whether E[n0-1] is negligible, 1 eigenvalue.
                if (Z[nn-6]<=tol2*(sigma+Z[nn-4]) || Z[nn-2*pp-5]<=tol2*Z[nn-8])
                {
                    Z[4*n0] = Z[4*n0+pp] + sigma;
                    n0--;
                    continue;
                }
                // Check  whether E[n0-2] is negligible, 2 eigenvalues.
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
                 Z[nn-4] = Z[nn-4]*(Z[nn-8] / tt);
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
                temp = Z[4*n0+pp+2];
                dmin2 = dmin2<temp?dmin2:temp;
                temp = Z[4*n0+pp+2];
                temp = temp<Z[4*i0+pp+2]?temp:Z[4*i0+pp+2];
                Z[4*n0+pp+2] = temp<Z[4*i0+pp+6]?temp:Z[4*i0+pp+6];
                temp = Z[4*n0-pp+3];
                temp = temp<Z[4*i0-pp+3]?temp:Z[4*i0-pp+3];
                Z[4*n0-pp+3] = temp<Z[4*i0-pp+7]?temp:Z[4*i0-pp+7];
                temp = Z[4*i0+pp];
                temp = qmax>temp?qmax:temp;
                qmax = temp>Z[4*i0+pp+4]?temp:Z[4*i0+pp+4];
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
                // TAU too big. Select new TAU and try again.
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

    /* dlasq4 computes an approximation dn2 to the smallest eigenvalue using values of d from the
     * previous transform.
     * Parameters: i0: First index. (note: zero-based index!)
     *             n0: Last index. (note: zero-based index!)
     *             Z: an array, dimension (4*(n0+1))
     *                Z holds the qd array.
     *             pp: pp=0 for ping, pp=1 for pong.
     *             n0in: The value of n0 at start of EIGTEST. (note: zero-based index!)
     *             dmin: Minimum value of d.
     *             dmin1: Minimum value of d, excluding D[n0].
     *             dmin2: Minimum value of d, excluding D[n0] and D[n0-1].
     *             dn: d(N)
     *             dn1: d(N-1)
     *             dn2: d(N-2)
     *             dn2: This is the shift.
     *             dn2: Shift type.
     *             g: g is passed as an argument in order to save its value between calls to
     *                dlasq4.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2016
     * Further Details:
     *     CNST1 = 9/16                                                                          */
    static void dlasq4(int i0, int n0, T const* Z, int pp, int n0in, T dmin, T dmin1, T dmin2,
                       T dn, T dn1, T dn2, T& tau, int& ttype, T& g)
    {
        const T CNST1 = T(0.563);
        const T CNST2 = T(1.01);
        const T CNST3 = T(1.05);
        const T QURTR  = T(0.25);
        const T THIRD  = T(0.333);
        const T HALF   = T(0.5);
        const T HUNDRD = T(100.0);
        // A negative dmin forces the shift to take that absolute value
        // dn2 records the type of shift.
        if (dmin<=ZERO)
        {
            ttype = -1;
            return;
        }
        int nn = 4*(n0+1) + pp;
        int i4, np;
        T a2, b1, b2, gam, gap1, gap2, s, temp;
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
                        s = s>temp?s:temp;
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
                            s = s<temp?s:temp;
                        }
                        temp = THIRD*dmin;
                        s = s>temp?s:temp;
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
                    // Approximate contribution to norm squared from I<nn-1.
                    a2 += b2;
                    for (i4=np; i4>=(4*i0 + 2 + pp); i4-=4)//10
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
                        if (HUNDRD*(b2>b1?b2:b1)<a2 || CNST1<a2)
                        {
                            break;
                        }
                    }
                    a2 *= CNST3;
                    // Rayleigh quotient residual bound.
                    if (a2<CNST1)
                    {
                        s = gam*(ONE-std::sqrt(a2)) / (ONE+a2);
                    }
                }
            }
            else if (dmin==dn2)
            {
                // Case 5.
                ttype = -5;
                s = QURTR*dmin;
                // Compute contribution to norm squared from I>nn-2.
                np = nn - 2*pp - 1;
                b1 = Z[np-2];
                b2 = Z[np-6];
                gam = dn2;
                if (Z[np-8]>b2 || Z[np-4]>b1)
                {
                    return;
                }
                a2 = (Z[np-8]/b2) * (ONE+Z[np-4]/b1);
                // Approximate contribution to norm squared from I<nn-2.
                if (n0-i0>2)
                {
                    b2 = Z[nn-14] / Z[nn-16];
                    a2 = a2 + b2;
                    for (i4=nn-18; i4>=(4*i0 + 2 + pp); i4-=4)
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
                        if (HUNDRD*(b2>b1?b2:b1)<a2 || CNST1<a2)
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
                    for (i4=(4*n0 - 6 + pp); i4>=(4*i0 + 2 + pp); i4-=4)
                    {
                        a2 = b1;
                        if (Z[i4]>Z[i4-2])
                        {
                            return;
                        }
                        b1 *= Z[i4] / Z[i4-2];
                        b2 += b1;
                        if (HUNDRD*(b1>a2?b1:a2)<b2)
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
                    s = s>temp?s:temp;
                }
                else
                {
                    temp = a2 * (ONE-CNST2*b2);
                    s = s>temp?s:temp;
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
                    for (i4=(4*n0 - 6 + pp); i4>=(4*i0 + 2 + pp); i4-=4)//70
                    {
                        if (Z[i4]>Z[i4-2])
                        {
                            return;
                        }
                        b1 *= Z[i4] / Z[i4-2];
                        b2 += b1;
                        if (HUNDRD*b1<b2)
                        {
                            break;
                        }
                    }
                }
                b2 = std::sqrt(CNST3*b2);
                a2 = dmin2 / (ONE+b2*b2);
                gap2 = Z[nn-8] + Z[nn-10]
                       -std::sqrt(Z[nn-12])*std::sqrt(Z[nn-10]) - a2;
                if (gap2>ZERO && gap2>b2*a2)
                {
                    temp = a2 * (ONE-CNST2*a2*(b2/gap2)*b2);
                    s = s>temp?s:temp;
                }
                else
                {
                    temp = a2*(ONE-CNST2*b2);
                    s = s>temp?s:temp;
                }
            }
            else
            {
                s = QURTR*dmin2;
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

    /* dlasq5 computes one dqds transform in ping-pong form, one version for ieee machines another
     * for non ieee machines.
     * Parameters: i0: First index. (note: zero-based index!)
     *             n0: Last index. (note: zero-based index!)
     *             Z: an array, dimension (4*N)
     *                holds the qd array. EMIN is stored in Z(4*(n0+1)) to avoid an extra argument.
     *             pp: pp=0 for ping, pp=1 for pong.
     *             pp: This is the shift.
     *             sigma: This is the accumulated shift up to this step.
     *             dmin: Minimum value of d.
     *             dmin1: Minimum value of d, excluding d[n0].
     *             dmin2: Minimum value of d, excluding d[n0] and d[n0-1].
     *             dn: d[n0], the last value of d.
     *             dnm1: d[n0-1].
     *             dnm2: d[n0-2].
     *             ieee: Flag for ieee or non ieee arithmetic.
     *             eps: This is the value of epsilon used.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2017                                                                            */
    static void dlasq5(int i0, int n0, T* Z, int pp, T tau, T sigma, T& dmin, T& dmin1,
                       T& dmin2, T& dn, T& dnm1, T& dnm2, bool ieee, T eps)
    {
        const T HALF = T(0.5);
        if ((n0-i0-1)<=0)
        {
            return;
        }
        int j4, j4p2;
        T d, emin, temp, dthresh;
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
                        dmin = dmin<d?dmin:d;
                        Z[j4] = Z[j4-1]*temp;
                        emin = Z[j4]<emin?Z[j4]:emin;
                    }
                }
                else
                {
                    for (j4=4*i0+3; j4<4*(n0-2); j4+=4)
                    {
                        Z[j4-3] = d + Z[j4];
                        temp = Z[j4+2] / Z[j4-3];
                        d = d*temp - tau;
                        dmin = dmin<d?dmin:d;
                        Z[j4-1] = Z[j4]*temp;
                        emin = Z[j4-1]<emin?Z[j4-1]:emin;
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
                dmin = dmin<dnm1?dmin:dnm1;
                dmin1 = dmin;
                j4 += 4;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm1 + Z[j4p2];
                Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                dn = Z[j4p2+2]*(dnm1/Z[j4-2]) - tau;
                dmin = dmin<dn?dmin:dn;
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
                        dmin = dmin<d?dmin:d;
                        emin = emin<Z[j4]?emin:Z[j4];
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
                        dmin = dmin<d?dmin:d;
                        emin = emin<Z[j4-1]?emin:Z[j4-1];
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
                dmin = dmin<dnm1?dmin:dnm1;
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
                dmin = dmin<dn?dmin:dn;
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
                    for (j4=4*i0+3; j4<4*(n0-2), j4+=4)
                    {
                        Z[j4-2] = d + Z[j4-1];
                        temp = Z[j4+1] / Z[j4-2];
                        d = d*temp - tau;
                        if (d<dthresh)
                        {
                            d = ZERO;
                        }
                        dmin = dmin<d?dmin:d;
                        Z[j4] = Z[j4-1]*temp;
                        emin = Z[j4]<emin?Z[j4]:emin;
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
                        dmin = dmin<d?dmin:d;
                        Z[j4-1] = Z[j4]*temp;
                        emin = Z[j4-1]<emin?Z[j4-1]:emin;
                    }
                }
                // Unroll last two steps.
                dnm2 = d;
                dmin2 = dmin;
                j4 = 4*n0 - 5 - pp;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm2 + Z[j4p2];
                Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                dnm1 = Z[j4p2+2]*(dnm2 / Z[j4-2]) - tau;
                dmin = dmin<dnm1?dmin:dnm1;
                dmin1 = dmin;
                j4 += 4;
                j4p2 = j4 + 2*pp - 1;
                Z[j4-2] = dnm1 + Z[j4p2];
                Z[j4] = Z[j4p2+2] * (Z[j4p2]/Z[j4-2]);
                dn = Z[j4p2+2]*(dnm1/Z[j4-2]) - tau;
                dmin = dmin<dn?dmin:dn;
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
                        dmin = dmin<d?dmin:d;
                        emin = Z[j4]<emin?Z[j4]:emin;
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
                        dmin = dmin<d?dmin:d;
                        emin = Z[j4-1]<emin?Z[j4-1]:emin;
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
                dmin = dmin<dnm1?dmin:dnm1;
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
                dmin = dmin<dn?dmin:dn;
            }
        }
        Z[j4+2] = dn;
        Z(4*(n0+1)-pp) = emin;
    }

    /* dlasq6 computes one dqd (shift equal to zero) transform in ping-pong form, with protection
     * against underflow and overflow.
     * Parameters: i0: First index. (note: zero-based index!)
     *             n0: Last index. (note: zero-based index!)
     *             Z: an array, dimension (4*N)
     *                Z holds the qd array. EMIN is stored in Z[4*n0+3] to avoid an extra argument.
     *             pp: pp=0 for ping, pp=1 for pong.
     *             dmin: Minimum value of d.
     *             dmin1: Minimum value of d, excluding D[n0].
     *             dmin2: Minimum value of d, excluding D[n0] and D[n0-1].
     *             dn: d[n0], the last value of d.
     *             dnm1: d[n0-1].
     *             dnm2: d[n0-2].
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlasq6(int i0, int n0, T* Z, int pp, T& dmin, T& dmin1, T& dmin2, T& dn, T& dnm1,
                       T& dnm2)
    {
        if ((n0-i0-1)<=0)
        {
            return;
        }
        T safmin = dlamch("Safe minimum");
        int j4 = 4*i0 + pp + 1;
        T emin = Z[j4+4];
        T d = Z[j4];
        dmin = d;
        int j4p2;
        T temp;
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
                dmin = dmin<d?dmin:d;
                emin = emin<Z[j4]?emin:Z[j4];
            }
        }
        else
        {
            for (j4 = 4*i0+3; j4<4*(n0-2); j4+=4)
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
                dmin = dmin<d?dmin:d;
                emin = emin<Z[j4-1]?emin:Z[j4-1];
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
        dmin = dmin<dnm1?dmin:dnm1;
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
        dmin = dmin<dn?dmin:dn;
        Z[j4+2] = dn;
        Z(4*n0+4-pp) = emin;
    }

    /* dlasr applies a sequence of plane rotations to a real matrix A, from either the left or the
     * right.
     * When side=='L', the transformation takes the form
     *     A := P*A
     * and when side=='R', the transformation takes the form
     *     A := A*P^T
     * where P is an orthogonal matrix consisting of a sequence of z plane rotations, with z = m
     * when side=='L' and z = n when side=='R', and P^T is the transpose of P.
     * When direct=='F' (Forward sequence), then
     *     P = P(z-1) * ... * P(2) * P(1)
     * and when direct=='B' (Backward sequence), then
     *     P = P(1) * P(2) * ... * P(z-1)
     * where P(k) is a plane rotation matrix defined by the 2-by-2 rotation
     *     R(k) = (  c(k)  s(k) )
     *          = ( -s(k)  c(k) ).
     * When pivot=='V' (Variable pivot), the rotation is performed for the plane (k,k+1),
     * i.e., P(k) has the form
     *     P(k) = (  1                                            )
     *            (       ...                                     )
     *            (              1                                )
     *            (                   c(k)  s(k)                  )
     *            (                  -s(k)  c(k)                  )
     *            (                                1              )
     *            (                                     ...       )
     *            (                                            1  )
     * where R(k) appears as a rank-2 modification to the identity matrix in rows and columns
     * k and k+1.
     * When pivot=='T' (Top pivot), the rotation is performed for the plane (1,k+1),
     * so P(k) has the form
     *     P(k) = (  c(k)                    s(k)                 )
     *            (         1                                     )
     *            (              ...                              )
     *            (                     1                         )
     *            ( -s(k)                    c(k)                 )
     *            (                                 1             )
     *            (                                      ...      )
     *            (                                             1 )
     * where R(k) appears in rows and columns 1 and k+1.
     * Similarly, when pivot=='B' (Bottom pivot), the rotation is performed for the plane (k,z),
     * giving P(k) the form
     *     P(k) = ( 1                                             )
     *            (      ...                                      )
     *            (             1                                 )
     *            (                  c(k)                    s(k) )
     *            (                         1                     )
     *            (                              ...              )
     *            (                                     1         )
     *            (                 -s(k)                    c(k) )
     * where R(k) appears in rows and columns k and z. The rotations are performed without ever
     * forming P(k) explicitly.
     * Parameters: side: Specifies whether the plane rotation matrix P is applied to A on the left
     *                   or the right.
     *                   =='L': Left, compute A := P*A
     *                   =='R': Right, compute A:= A*P^T
     *             pivot: Specifies the plane for which P(k) is a plane rotation matrix.
     *                    =='V': Variable pivot, the plane (k,k+1)
     *                    =='T': Top pivot, the plane (1,k+1)
     *                    =='B': Bottom pivot, the plane (k,z)
     *             direct: Specifies whether P is a forward or backward sequence of plane
     *                     rotations.
     *                     =='F': Forward, P = P(z-1)*...*P(2)*P(1)
     *                     =='B': Backward, P = P(1)*P(2)*...*P(z-1)
     *             m: The number of rows of the matrix A. If m<=1, an immediate return is effected.
     *             n: The number of columns of the matrix A. If n<=1, an immediate return is
     *                effected.
     *             c: an array, dimension (m-1) if side=='L'
     *                                    (n-1) if side=='R'
     *                The cosines c(k) of the plane rotations.
     *             s: an array, dimension (m-1) if side=='L'
     *                                    (n-1) if side=='R'
     *                The sines s(k) of the plane rotations. The 2-by-2 plane rotation part of the
     *                matrix P(k), R(k), has the form
     *                    R(k) = (  c(k)  s(k) )
     *                           ( -s(k)  c(k) ).
     *             A: an array, dimension (lda,n)
     *                The m-by-n matrix A. On exit, A is overwritten by P*A if side=='R'
     *                                                            or by A*P^T if side=='L'.
     *             lda: The leading dimension of the array A. lda >= max(1,m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date December 2016                                                                        */
    static void dlasr(char const* side, char const* pivot, char const* direct, int m, int n,
                      T const* c, T const* s, T* A, int lda)
    {
        int i, info, j, aind1, aind2, aind3, aind4;
        T ctemp, stemp, temp;
        char upside = std::toupper(side[0]);
        char uppivot = std::toupper(pivot[0]);
        char updirect = std::toupper(direct[0]);
        // Test the input parameters
        info = 0
        if (!(upside=='L' || upside=='R'))
        {
            info = 1;
        }
        else if (!(uppivot=='V' || uppivot=='T' || uppivot=='B'))
        {
            info = 2;
        }
        else if (!(updirect=='F'||updirect=='B'))
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
        else if (lda<MAX(1, m))
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
                        if (ctemp!=ONE||stemp!=ZERO)
                        {
                            for (i=0; i<n; i++)
                            {
                                aind1 = j+lda*i;
                                aind2 = aind1+1;
                                // A[j+1,i] and A[j,i]
                                temp     = A[aind2];
                                A[aind2] = ctemp*temp-stemp*A[aind1];
                                A[aind1] = stemp*temp+ctemp*A[aind1];
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
                    for(j=1; j<m; j++)
                    {
                        ctemp = c[j-1];
                        stemp = s[j-1];
                        if (ctemp!=ONE || stemp!=ZERO)
                        {
                            for(i=0; i<n; i++)
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

    /* dlasrt sorts the numbers in d in increasing order (if id = 'I') or in decreasing order
     * (if id = 'D').
     * Use Quick Sort, reverting to Insertion sort on arrays of size<=20. Dimension of 'stack'
     * limits n to about 2^32.
     * Parameters: id: =='I': sort d in increasing order;
     *                 =='D': sort d in decreasing order.
     *             n: The length of the array d.
     *             d: an array, dimension (n)
     *                On entry, the array to be sorted.
     *                On exit, d has been sorted into increasing order (d[0] <= ... <= d[n-1]) or
     *                         into decreasing order (d[0] >= ... >= d[n-1]), depending on id.
     *             info: ==0: successful exit
     *                    <0: if info = -i, the i-th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date June 2016                                                                            */
    static void dlasrt(char const* id, int n, T* d, int& info)
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
        int stack[2*32];// stack[2, 32]
        int endd, i, j, start, stkpnt;
        T d1, d2, d3, dmnmx, tmp;
        stkpnt = 0;
        stack[0] = 0;//stack[0,0]
        stack[1] = n-1;//stack[1,0]
        do
        {
            start = stack[2*stkpnt];//stack[0,stkpnt]
            endd = stack[1+2*stkpnt];//stack[1,stkpnt]
            stkpnt--;
            if (endd-start<=SELECT && endd-start>0)
            {
                // Do Insertion sort on D[start:endd]
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
                // Partition D[start:endd] and stack parts, largest one first
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
                        stack[2*stkpnt] = start;//stack[0,stkpnt]
                        stack[1+2*stkpnt] = j;//stack[1,stkpnt]
                        stkpnt++;
                        stack[2*stkpnt] = j + 1;//stack[0,stkpnt]
                        stack[1+2*stkpnt] = endd;//stack[1,stkpnt]
                    }
                    else
                    {
                        stkpnt++;
                        stack[2*stkpnt] = j + 1;//stack[0,stkpnt]
                        stack[1+2*stkpnt] = endd;//stack[1,stkpnt]
                        stkpnt++;
                        stack[2*stkpnt] = start;//stack[0,stkpnt]
                        stack[1+2*stkpnt] = j;//stack[1,stkpnt]
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
                        stack[2*stkpnt] = start;//stack[0,stkpnt]
                        stack[1+2*stkpnt] = j;//stack[1,stkpnt]
                        stkpnt++;
                        stack[2*stkpnt] = j + 1;//stack[0,stkpnt]
                        stack[1+2*stkpnt] = endd;//stack[1,stkpnt]
                    }
                    else
                    {
                        stkpnt++;
                        stack[2*stkpnt] = j + 1;//stack[0,stkpnt]
                        stack[1+2*stkpnt] = endd;//stack[1,stkpnt]
                        stkpnt++;
                        stack[2*stkpnt] = start;//stack[0,stkpnt]
                        stack[1+2*stkpnt] = j;//stack[1,stkpnt]
                    }
                }
            }
        } while (stkpnt>=0);
    }

    /* dlassq updates a sum of squares represented in scaled form.
     * returns the values scl and smsq such that
     * (scl^2)*smsq = x(0)^2 + ... + x(n-1)^2 + (scale^2)*sumsq,
     * where x(i) = x[i*incx]. The value of sumsq is assumed to be non-negative and scl
     * returns the value
     * scl = max(scale, abs(x(i))).
     * scl and smsq are overwritten on scale and sumsq respectively.
     * The routine makes only one pass through the vector x.
     * Parameters: n: The number of elements to be used from the vector x.
     *             x: an array, dimension (n)
     *                The vector for which a scaled sum of squares is computed.
     *                x(i) = x[i*incx], 0 <= i < n.
     *             incx: The increment between successive values of the vector x.
     *                   incx > 0.
     *             scale: On entry, the value scale in the equation above.
     *                    On exit, scale is overwritten with scl, the scaling factor for
     *                    the sum of squares.
     *             sumsq: On entry, the value sumsq in the equation above.
     *                    On exit, sumsq is overwritten with smsq, the basic sum of squares from
     *                    which scl has been factored out.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016                                                                       */
    static void dlassq(int n, T const* x, int incx, int& scale, int& sumsq)
    {
        if (n>0)
        {
            T absxi, temp;
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

    /* dlasv2 computes the singular value decomposition of a 2-by-2 triangular matrix
     *     [ f  g ]
     *     [ 0  h ].
     * On return, abs(ssmax) is the larger singular value, abs(ssmin) is the smaller singular
     * value, and (csl,snl) and (csr,snr) are the left and right singular vectors for abs(ssmax),
     * giving the decomposition
     *     [ csl snl] [ f  g ] [csr -snr] = [ssmax  0  ]
     *     [-snl csl] [ 0  h ] [snr  csr]   [ 0   ssmin].
     * Parameters: f: The [0,0] element of the 2-by-2 matrix.
     *             g: The [0,1] element of the 2-by-2 matrix.
     *             h: The [1,1] element of the 2-by-2 matrix.
     *             ssmin: abs(ssmin) is the smaller singular value.
     *             ssmax: abs(ssmax) is the larger singular value.
     *             snl,
     *             csl: The vector (csl, snl) is a unit left singular vector for the singular value
     *                  abs(ssmax).
     *             snr,
     *             csr: The vector (csr, snr) is a unit right singular vector for the singular
     *                  value abs(ssmax).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016
     * Further Details:
     *   Any input parameter may be aliased with any output parameter.
     *   Barring over/underflow and assuming a guard digit in subtraction, all output quantities
     *     are correct to within a few units in the last place (ulps).
     *   In IEEE arithmetic, the code works correctly if one matrix element is infinite.
     *   Overflow will not occur unless the largest singular value itself overflows or is within
     *     a few ulps of overflow. (On machines with partial overflow, like the Cray, overflow may
     *     occur if the largest singular value is within a factor of 2 of overflow.)
     *   Underflow is harmless if underflow is gradual. Otherwise, results may correspond to a
     *     matrix modified by perturbations of size near the underflow threshold.                */
    static void dlasv2(T f, T g, T h, T& ssmin, T& ssmax, T& snr, T& csr, T& snl, T& csl)
    {
        bool gasmal, swap;
        int pmax;
        T a, clt, crt, d, fa, ft, ga, gt, ha, ht, l, m, mm, r, s, slt, srt, t, temp, tsign, tt;
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
                    }else{
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
                a = T(0.5) * (s+r);
                // Note that 1 <= a <= 1 + abs(M)
                ssmin = ha / a;
                ssmax = fa * a;
                if (mm==ZERO)
                {
                    // Note that m is very tiny
                   if (l==ZERO)
                   {
                      t = T((ZERO<=ft)-(ft<ZERO))*TWO*T((ZERO<=gt)-(gt<ZERO));
                   }
                   else
                   {
                      t = gt/(T((ZERO<=ft)-(ft<ZERO))*std::fabs(d)) + m/t;
                   }
                }
                else
                {
                   t = (m/(s+t) + m/(r+l)) * (ONE + a);
                }
                l = std::sqrt(t*t+T(4.0));
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
        if (pmax==0)
        {
            tsign = T((ZERO<=csr)-(csr<ZERO))*T((ZERO<=csl)-(csl<ZERO))*T((ZERO<=f)-(f<ZERO));
        }
        if (pmax==1)
        {
            tsign = T((ZERO<=snr)-(snr<ZERO))*T((ZERO<=csl)-(csl<ZERO))*T((ZERO<=g)-(g<ZERO));
        }
        if (pmax==2)
        {
            tsign = T((ZERO<=snr)-(snr<ZERO))*T((ZERO<=snl)-(snl<ZERO))*T((ZERO<=h)-(h<ZERO));
        }
        ssmax = tsign*std::fabs(ssmax);
        ssmin = tsign*T((ZERO<=f)-(f<ZERO))*T((ZERO<=h)-(h<ZERO))*std::fabs(ssmin);
    }

    /* dorg2r generates an m by n real matrix Q with orthonormal columns,
     * which is defined as the first n columns of a product of k elementary reflectors of order m
     *     Q = H(1) H(2) . ..H(k)
     * as returned by dgeqrf.
     * Parameters: m: The number of rows of the matrix Q. m >= 0.
     *             n: The number of columns of the matrix Q. m >= n >= 0.
     *             k: The number of elementary reflectors whose product defines the matrix Q.
     *                n >= k >= 0.
     *             A: an array, dimension(lda, n)
     *                On entry, the i-th column must contain the vector which defines the
     *                elementary reflector H(i), for i = 1, 2, ..., k, as returned by dgeqrf
     *                in the first k columns of its array argument A.
     *                On exit, the m-by-n matrix Q.
     *             lda: The first dimension of the array A. lda >= max(1, m).
     *             tau: an array, dimension(k)
     *                  tau(i) must contain the scalar factor of the elementary reflector H(i),
     *                  as returned by dgeqrf.
     *             work: an array, dimension(n)
     *             info: ==0: successful exit
     *                   < 0: if info = -i, the i-th argument has an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void dorg2r(int m, int n, int k, T* A, int lda, T const* tau, T* work, int& info)
    {
        // Test the input arguments
        info = 0;
        if (m < 0)
        {
            info = -1;
        } else if (n < 0 || n > m)
        {
            info = -2;
        } else if (k < 0 || k > n)
        {
            info = -3;
        } else if (lda < (1 > m ? 1 : m))
        {
            info = -5;
        }
        if (info != 0)
        {
            xerbla("DORG2R", -info);
            return;
        }
        // Quick return if possible
        if (n < 0)
        {
            return;
        }
        int i, j, acoli;
        // Initialise columns k:n-1 to columns of the unit matrix
        for (i = k; i < n; i++)
        {
            acoli = lda*i;
            for (j = 0; j < m; j++)
            {
                A[j + acoli] = ZERO;
            }
            A[i + acoli] = ONE;
        }
        for (i = k - 1; i >= 0; i--)
        {
            acoli = lda*i;
            // Apply H(i) to A[i:m-1, i:n-1] from the left
            if (i < n - 1)
            {
                A[i + acoli] = ONE;
                dlarf("Left", m - i, n - i - 1, &A[i + acoli], 1, tau[i], &A[i + acoli + lda], lda, work);
            }
            if (i < m - 1)
            {
                dscal(m - i - 1, -tau[i], &A[i + 1 + acoli], 1);
            }
            A[i + acoli] = ONE - tau[i];
            // Set A(1:i - 1, i) to zero
            for (j = 0; j < i; j++)
            {
                A[j + acoli] = ZERO;
            }
        }
    }

    /* dorgqr generates an m-by-n real matrix Q with orthonormal columns, which is defined
     * as the first n columns of a product of k elementary reflectors of order m
     *     Q = H(1) H(2) . ..H(k)
     * as returned by dgeqrf.
     * Parameters: m: The number of rows of the matrix Q. m >= 0.
     *             n: The number of columns of the matrix Q. m >= n >= 0.
     *             k: The number of elementary reflectors whose product defines the matrix Q. n >= k >= 0.
     *             A: an array, dimension(lda, n)
     *                On entry, the i-th column must contain the vector which defines the
     *                elementary reflector H(i), for i = 1, 2, ..., k, as returned by dgeqrf
     *                in the first k columns of its array argument A.
     *                On exit, the m-by-n matrix Q.
     *             lda: The first dimension of the array A. lda >= max(1, m).
     *             tau:  array, dimension(k). tau[i] must contain the scalar factor of the
     *                   elementary reflector H(i), as returned by dgeqrf.
     *             work: an array, dimension(max(1, lwork))
     *                   On exit, if info = 0, work[0] returns the optimal lwork.
     *             lwork: The dimension of the array work. lwork >= max(1, n).
     *                    For optimum performance lwork >= n*nb, where nb is the optimal blocksize.
     *                    If lwork = -1, then a workspace query is assumed; the routine only calculates
     *                    the optimal size of the work array, returns this value as the first entry of
     *                    the work array, and no error message related to lwork is issued by xerbla.
     *             info: =0:  successful exit
     *                   <0:  if info = -i, the i-th argument has an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void dorgqr(int m, int n, int k, T* A, int lda, T const* tau, T* work, int lwork, int& info)
    {
        // Test the input arguments
        info = 0;
        int nb = ilaenv(1, "DORGQR", " ", m, n, k, -1);
        int lwkopt = (1 > n ? 1 : n) * nb;
        work[0] = lwkopt;
        bool lquery = (lwork == -1);
        if (m < 0)
        {
            info = -1;
        } else if (n < 0 || n > m)
        {
            info = -2;
        } else if (k < 0 || k > n)
        {
            info = -3;
        } else if (lda < (1 > m ? 1 : m))
        {
            info = -5;
        } else if (lwork < (1 > n ? 1 : n) && !lquery)
        {
            info = -8;
        }
        if (info != 0)
        {
            xerbla("DORGQR", -info);
            return;
        } else if (lquery)
        {
            return;
        }
        // Quick return if possible
        if (n <= 0)
        {
            work[0] = 1;
            return;
        }
        int nbmin = 2;
        int nx = 0;
        int iws = n;
        int ldwork = 0;
        if (nb > 1 && nb < k)
        {
            // Determine when to cross over from blocked to unblocked code.
            nx = ilaenv(3, "DORGQR", " ", m, n, k, -1);
            if (nx < 0)
            {
                nx = 0;
            }
            if (nx < k)
            {
                // Determine if workspace is large enough for blocked code.
                ldwork = n;
                iws = ldwork*nb;
                if (lwork < iws)
                {
                    // Not enough workspace to use optimal nb:
                    // reduce nb and determine the minimum value of nb.
                    nb = lwork / ldwork;
                    nbmin = ilaenv(2, "DORGQR", " ", m, n, k, -1);
                    if (nbmin < 2)
                    {
                        nbmin = 2;
                    }
                }
            }
        }
        int i, j, kk, ki = 0, acol;
        if (nb >= nbmin && nb < k && nx < k)
        {
            // Use blocked code after the last block. The first kk columns are handled by the block method.
            ki = ((k - nx - 1) / nb) * nb;
            kk = ki + nb;
            if (kk > k)
            {
                kk = k;
            }
            // Set A[0:kk-1, kk:n-1] to zero.
            for (j = kk; j < n; j++)
            {
                acol = lda*j;
                for (i = 0; i < kk; i++)
                {
                    A[i + acol] = ZERO;
                }
            }
        } else
        {
            kk = 0;
        }
        int iinfo;
        // Use unblocked code for the last or only block.
        if (kk < n)
        {
            dorg2r(m - kk, n - kk, k - kk, &A[kk + lda * kk], lda, &tau[kk], work, iinfo);
        }
        if (kk > 0)
        {
            int ib, l;
            // Use blocked code
            for (i = ki; i >= 0; i -= nb)
            {
                ib = k - i;
                if (ib > nb)
                {
                    ib = nb;
                }
                acol = i + lda*i;
                if (i + ib < n)
                {
                    // Form the triangular factor of the block reflector H = H(i) H(i + 1) ... H(i + ib - 1)
                    dlarft("Forward", "Columnwise", m - i, ib, &A[acol], lda, &tau[i], work, ldwork);
                    // Apply H to A[i:m-1, i+ib:n-1] from the left
                    dlarfb("Left", "No transpose", "Forward", "Columnwise", m - i, n - i - ib, ib, &A[acol], lda, work, ldwork, &A[acol + lda * ib], lda, &work[ib], ldwork);
                }
                // Apply H to rows i:m-1 of current block
                dorg2r(m - i, ib, ib, &A[acol], lda, &tau[i], work, iinfo);
                // Set rows 0:i-1 of current block to zero
                for (j = i; j < (i + ib); j++)
                {
                    acol = lda*j;
                    for (l = 0; l < i; l++)
                    {
                        A[l + acol] = ZERO;
                    }
                }
            }
        }
        work[0] = iws;
    }

    /* dorm2r overwrites the general real m-by-n matrix C with
     *     Q * C   if side = 'L' and trans = 'N', or
     *     Q^T* C  if side = 'L' and trans = 'T', or
     *     C * Q   if side = 'R' and trans = 'N', or
     *     C * Q^T if side = 'R' and trans = 'T',
     * where Q is a real orthogonal matrix defined as the product of k elementary reflectors
     *     Q = H(1) H(2) . ..H(k)
     * as returned by dgeqrf. Q is of order m if side = 'L' and of order n if side = 'R'.
     * Parameters: side: 'L': apply Q or Q^T from the Left
     *                   'R': apply Q or Q^T from the Right
     *             trans: 'N': apply Q (No transpose)
     *                    'T': apply Q^T (Transpose)
     *             m: The number of rows of the matrix C. m >= 0.
     *             n: The number of columns of the matrix C. n >= 0.
     *             k: The number of elementary reflectors whose product defines the matrix Q.
     *                If side = 'L', m >= k >= 0;
     *                if side = 'R', n >= k >= 0.
     *             A: an array, dimension(lda, k)
     *                The i-th column must contain the vector which defines the elementary reflector H(i),
     *                for i = 1, 2, ..., k, as returned by dgeqrf in the first k columns of its array argument A.
     *                A is modified by the routine but restored on exit.
     *             lda: The leading dimension of the array A.
     *                  If side = 'L', lda >= max(1, m);
     *                  if side = 'R', lda >= max(1, n).
     *             tau: an array, dimension(k)
     *                  tau[i] must contain the scalar factor of the elementary reflector H(i), as returned by dgeqrf.
     *             C: an array, dimension(ldc, n)
     *                On entry, the m-by-n matrix C.
     *                On exit, C is overwritten by Q*C or Q^T*C or C*Q^T or C*Q.
     *             ldc: The leading dimension of the array C. ldc >= max(1, m).
     *             work: an array, dimension (n) if side = 'L',
     *                                       (m) if side = 'R'
     *             info: =0: successful exit
     *                   <0: if info = -i, the i-th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void dorm2r(char const* side, char const* trans, int m, int n, int k, T* A, int lda, T const* tau, T* C, int ldc, T* work, int& info)
    {
        // Test the input arguments
        bool left = (toupper(side[0]) == 'L');
        bool notran = (toupper(trans[0]) == 'N');
        // nq is the order of Q
        int nq;
        if (left)
        {
            nq = m;
        } else
        {
            nq = n;
        }
        info = 0;
        if (!left && (toupper(side[0]) != 'R'))
        {
            info = -1;
        } else if (!notran && (toupper(trans[0]) != 'T'))
        {
            info = -2;
        } else if (m < 0)
        {
            info = -3;
        } else if (n < 0)
        {
            info = -4;
        } else if (k < 0 || k > nq)
        {
            info = -5;
        } else if (lda < (1 > nq ? 1 : nq))
        {
            info = -7;
        } else if (ldc < (1 > m ? 1 : m))
        {
            info = -10;
        }
        if (info != 0)
        {
            xerbla("DORM2R", -info);
            return;
        }
        // Quick return if possible
        if (m == 0 || n == 0 || k == 0)
        {
            return;
        }
        int i1, i2, i3;
        if ((left && !notran) || (!left && notran))
        {
            i1 = 0;
            i2 = k;
            i3 = 1;
        } else
        {
            i1 = k - 1;
            i2 = -1;
            i3 = -1;
        }
        int ic, jc, mi = 0, ni = 0;
        if (left)
        {
            ni = n;
            jc = 0;
        } else
        {
            mi = m;
            ic = 0;
        }
        int i, aind;
        T aii;
        for (i = i1; i != i2; i += i3)
        {
            if (left)
            {
                // H(i) is applied to C[i:m-1, 0:n-1]
                mi = m - i;
                ic = i;
            } else
            {
                // H(i) is applied to C[0:m-1, i:n-1]
                ni = n - i;
                jc = i;
            }
            // Apply H(i)
            aind = i + lda*i;
            aii = A[aind];
            A[aind] = ONE;
            dlarf(side, mi, ni, &A[aind], 1, tau[i], &C[ic + ldc * jc], ldc, work);
            A[aind] = aii;
        }
    }

    /* dormqr overwrites the general real m-by-n matrix C with
     *                     side = 'L'   side = 'R'
     *     trans = 'N':    Q * C        C * Q
     *     trans = 'T':    Q^T * C      C * Q^T
     * where Q is a real orthogonal matrix defined as the product of k elementary reflectors
     *     Q = H(1) H(2) . ..H(k)
     * as returned by dgeqrf. Q is of order m if side = 'L' and of order n if side = 'R'.
     * Parameters: side: 'L': apply Q or Q^T from the Left;
     *                   'R': apply Q or Q**T from the Right.
     *             trans: 'N':  No transpose, apply Q;
     *                    'T':  Transpose, apply Q^T.
     *             m: The number of rows of the matrix C. m >= 0.
     *             n: The number of columns of the matrix C. n >= 0.
     *             k: The number of elementary reflectors whose product defines the matrix Q.
     *                If side = 'L', m >= k >= 0;
     *                if side = 'R', n >= k >= 0.
     *             A: an array, dimension(lda, k)
     *               The i-th column must contain the vector which defines the elementary reflector H(i),
     *               for i = 1, 2, ..., k, as returned by dgeqrf in the first k columns of its array argument A.
     *               A may be modified by the routine but is restored on exit.
     *             lda: The leading dimension of the array A.
     *                  If side = 'L', lda >= max(1, m);
     *                  if side = 'R', lda >= max(1, n).
     *             tau:  array, dimension(k)
     *                   tau(i) must contain the scalar factor of the elementary reflector H(i), as returned by dgeqrf.
     *             C: an array, dimension(ldc, n)
     *                On entry, the m-by-n matrix C.
     *                On exit, C is overwritten by Q*C or Q^T*C or C*Q^T or C*Q.
     *             ldc: The leading dimension of the array C. ldc >= max(1, m).
     *             work: an array, dimension(MAX(1, lwork))
     *                   On exit, if info = 0, work[0] returns the optimal lwork.
     *             lwork: The dimension of the array work.
     *                    If side = 'L', lwork >= max(1, n);
     *                    if side = 'R', lwork >= max(1, m).
     *                    For optimum performance lwork >= n*nb if side = 'L',
     *                    and lwork >= m*nb if side = 'R', where nb is the optimal blocksize.
     *                    If lwork = -1, then a workspace query is assumed;
     *                    the routine only calculates the optimal size of the work array,
     *                    returns this value as the first entry of the work array,
     *                    and no error message related to lwork is issued by xerbla.
     *             info: =0:  successful exit
     *                   <0:  if info = -i, the i-th argument had an illegal value
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void dormqr(char const* side, char const* trans, int m, int n, int k, T* A, int lda, T const* tau, T* C, int ldc, T* work, int lwork, int& info)
    {
        int const NBMAX = 64, LDT = NBMAX + 1;
        T* Tm = new T[LDT * NBMAX];
        // Test the input arguments
        info = 0;
        char upside = toupper(side[0]);
        char uptrans = toupper(trans[0]);
        bool left = (upside == 'L');
        bool notran = (uptrans == 'N');
        bool lquery = (lwork == -1);
        // nq is the order of Q and nw is the minimum dimension of work
        int nq, nw;
        if (left)
        {
            nq = m;
            nw = n;
        } else
        {
            nq = n;
            nw = m;
        }
        if (!left && (upside != 'R'))
        {
            info = -1;
        } else if (!notran && (uptrans != 'T'))
        {
            info = -2;
        } else if (m < 0)
        {
            info = -3;
        } else if (n < 0)
        {
            info = -4;
        } else if (k < 0 || k > nq)
        {
            info = -5;
        } else if (lda < (1 > nq ? 1 : nq))
        {
            info = -7;
        } else if (ldc < (1 > m ? 1 : m))
        {
            info = -10;
        } else if (lwork < (1 > nw ? 1 : nw) && !lquery)
        {
            info = -12;
        }
        char opts[2];
        opts[0] = upside;
        opts[1] = uptrans;
        int lwkopt, nb;
        if (info == 0)
        {
            // Determine the block size. nb may be at most NBMAX, where NBMAX is used to define the local array T.
            {
                nb = ilaenv(1, "DORMQR", opts, m, n, k, -1);
            }
            if (nb > NBMAX)
            {
                nb = NBMAX;
            }
            lwkopt = (1 > nw ? 1 : nw) * nb;
            work[0] = lwkopt;
        }
        if (info != 0)
        {
            xerbla("DORMQR", -info);
            return;
        } else if (lquery)
        {
            return;
        }
        // Quick return if possible
        if (m == 0 || n == 0 || k == 0)
        {
            work[0] = 1;
            return;
        }
        int nbmin = 2;
        int ldwork = nw;
        int iws;
        if (nb > 1 && nb < k)
        {
            iws = nw*nb;
            if (lwork < iws)
            {
                nb = lwork / ldwork;
                nbmin = ilaenv(2, "DORMQR", opts, m, n, k, -1);
                if (nbmin < 2)
                {
                    nbmin = 2;
                }
            }
        } else
        {
            iws = nw;
        }
        if (nb < nbmin || nb >= k)
        {
            // Use unblocked code
            int iinfo;
            dorm2r(side, trans, m, n, k, A, lda, tau, C, ldc, work, iinfo);
        } else
        {
            // Use blocked code
            int i1, i2, i3;
            if ((left && !notran) || (!left && notran))
            {
                i1 = 0;
                i2 = k - 1;
                i3 = nb;
            } else
            {
                i1 = ((k - 1) / nb) * nb;
                i2 = 0;
                i3 = -nb;
            }
            int ic, jc, mi = 0, ni = 0;
            if (left)
            {
                ni = n;
                jc = 0;
            } else
            {
                mi = m;
                ic = 0;
            }
            int i, ib, aind;
            for (i = i1; i != i2 + i3; i += i3)
            {
                ib = k - i;
                if (ib > nb)
                {
                    ib = nb;
                }
                // Form the triangular factor of the block reflector
                aind = i + lda*i;
                //     H = H(i) H(i + 1) . ..H(i + ib - 1)
                dlarft("Forward", "Columnwise", nq - i, ib, &A[aind], lda, &tau[i], Tm, LDT);
                if (left)
                {
                    // H or H^T is applied to C[i:m-1, 0:n-1]
                    mi = m - i;
                    ic = i;
                } else
                {
                    // H or H^T is applied to C[0:m-1, i:n-1]
                    ni = n - i;
                    jc = i;
                }
                // Apply H or H^T
                dlarfb(side, trans, "Forward", "Columnwise", mi, ni, ib, &A[aind], lda, Tm, LDT, &C[ic + ldc * jc], ldc, work, ldwork);
            }
        }
        work[0] = lwkopt;
        delete[] Tm;
    }

    /* ieeeck is called from the ilaenv to verify that Infinity and possibly NaN arithmetic is safe (i.e.will not trap).
     * Parameters: ispec: Specifies whether to test just for inifinity arithmetic or whether to test for infinity and NaN arithmetic.
     *                    0: Verify infinity arithmetic only.
     *                    1: Verify infinity and NaN arithmetic.
     *             Zero: Must contain the value 0.0
     *                   This is passed to prevent the compiler from optimizing away some code.
     *             One: Must contain the value 1.0
     *                  This is passed to prevent the compiler from optimizing away some code.
     * Returns: 0: Arithmetic failed to produce the correct answers
     *          1: Arithmetic produced the correct answers
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * TODO: Check whether these checks are still valid in c++                                                            */
    static int ieeeck(int ispec, T Zero, T One)
    {
        T posinf = One / Zero;
        if (posinf < One)
        {
            return 0;
        }
        T neginf = -One / Zero;
        if (neginf >= Zero)
        {
            return 0;
        }
        T negzro = One / (neginf + One);
        if (negzro != Zero)
        {
            return 0;
        }
        neginf = One / negzro;
        if (neginf >= Zero)
        {
            return 0;
        }
        T newzro = negzro + Zero;
        if (newzro != Zero)
        {
            return 0;
        }
        posinf = One / newzro;
        if (posinf <= One)
        {
            return 0;
        }
        neginf = neginf*posinf;
        if (neginf >= Zero)
        {
            return 0;
        }
        posinf = posinf*posinf;
        if (posinf <= One)
        {
            return 0;
        }
        // Return if we were only asked to check infinity arithmetic
        if (ispec == 0)
        {
            return 1;
        }
        T NaN1 = posinf + neginf;
        T NaN2 = posinf / neginf;
        T NaN3 = posinf / posinf;
        T NaN4 = posinf*Zero;
        T NaN5 = neginf*negzro;
        T NaN6 = NaN5*Zero;
        if (NaN1 == NaN1)
        {
            return 0;
        }
        if (NaN2 == NaN2)
        {
            return 0;
        }
        if (NaN3 == NaN3)
        {
            return 0;
        }
        if (NaN4 == NaN4)
        {
            return 0;
        }
        if (NaN5 == NaN5)
        {
            return 0;
        }
        if (NaN6 == NaN6)
        {
            return 0;
        }
        return 1;
    }

    /* iladlc scans A for its last non - zero column.
     * Parameters: m: The number of rows of the matrix A.
     *             n: The number of columns of the matrix A.
     *             A: an array, dimension(lda, n)
     *                The m by n matrix A.
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                           */
    static int iladlc(int m, int n, T const* A, int lda)
    {
        int ila, lastcol = lda * (n - 1);
        // Quick test for the common case where one corner is non - zero.
        if (n == 0)
        {
            return 0;
        } else if (A[lastcol] != ZERO || A[m - 1 + lastcol] != ZERO)
        {
            return n - 1;
        } else
        {
            int i;
            // Now scan each column from the end, returning with the first non-zero.
            for (ila = lastcol; ila >= 0; ila -= lda)
            {
                for (i = 0; i < m; i++)
                {
                    if (A[i + ila] != ZERO)
                    {
                        return ila / lda;
                    }
                }
            }
            return -1;
        }
    }

    /* iladlr scans A for its last non - zero row.
     * Parameters: m: The number of rows of the matrix A.
     *             n: The number of columns of the matrix A.
     *             A: an array, dimension(lda, n)
     *                The m by n matrix A.
     *             lda: The leading dimension of the array A. lda >= max(1, m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                           */
    static int iladlr(int m, int n, T const* A, int lda)
    {
        int lastrow = m - 1;
        // Quick test for the common case where one corner is non - zero.
        if (m == 0)
        {
            return 0;
        } else if (A[lastrow] != ZERO || A[lastrow + lda * (n - 1)] != ZERO)
        {
            return lastrow;
        } else
        {
            int i, j;
            // Scan up each column tracking the last zero row seen.
            int colj, ila = 0;
            for (j = 0; j < n; j++)
            {
                colj = lda*j;
                i = lastrow;
                //while ((A[(i>0 ? i : 0) + colj] == zero) && (i > 0))
                while ((A[i + colj] == ZERO) && (i > 0))// TODO: check if correct
                {
                    i--;
                }
                ila = (ila > i ? ila : i);
            }
            return ila;
        }
    }

    /* ilaenv is called from the LAPACK routines to choose problem-dependent parameters for the local environment.
     * See ispec for a description of the parameters.
     * ilaenv returns an integer: if >= 0: ilaenv returns the value of the parameter specified by ispec
     *                            if  < 0: if -k, the k-th argument had an illegal value.
     * This version provides a set of parameters which should give good, but not optimal, performance
     *     on many of the currently available computers.Users are encouraged to modify this subroutine to set
     *     the tuning parameters for their particular machine using the option and problem size information in the arguments.
     * This routine will not function correctly if it is converted to all lower case.
     *     Converting it to all upper case is allowed.
     * Parameters: ispec: Specifies the parameter to be returned.
     *                    1: the optimal blocksize; if this value is 1, an unblocked algorithm will give the best performance.
     *                    2: the minimum block size for which the block routine should be used;
     *                       if the usable block size is less than this value, an unblocked routine should be used.
     *                    3: the crossover point(in a block routine, for N less than this value, an unblocked routine should be used)
     *                    4: the number of shifts, used in the nonsymmetric eigenvalue routines(DEPRECATED)
     *                    5: the minimum column dimension for blocking to be used; rectangular blocks must have dimension
     *                       at least k by m, where k is given by ILAENV(2, ...) and m by ILAENV(5, ...)
     *                    6: the crossover point for the SVD (when reducing an m by n matrix to bidiagonal form,
     *                       if max(m, n) / min(m, n) exceeds this value, a QR factorization is used first to
     *                       reduce the matrix to a triangular form.)
     *                    7: the number of processors
     *                    8: the crossover point for the multishift QR method for nonsymmetric eigenvalue problems(DEPRECATED)
     *                    9: maximum size of the subproblems at the bottom of the computation tree in the
     *                       divide-and-conquer algorithm (used by xGELSD and xGESDD)
     *                    10: ieee NaN arithmetic can be trusted not to trap
     *                    11: infinity arithmetic can be trusted not to trap
     *                    12 <= ispec <= 16: xHSEQR or one of its subroutines, see IPARMQ for detailed explanation
     *             name: The name of the calling subroutine, in either upper case or lower case.
     *             opts: The character options to the subroutine name, concatenated into a single character string.
     *                   For example, UPLO = 'U', trans = 'T', and DIAG = 'N' for a triangular routine would be specified as opts = 'UTN'.
     *             n1: integer
     *             n2: integer
     *             n3: integer
     *             n4: integer; Problem dimensions for the subroutine name; these may not all be required.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Further Details:
     *     The following conventions have been used when calling ilaenv from the LAPACK routines:
     *      1)  opts is a concatenation of all of the character options to subroutine name, in the same order that they appear
     *          in the argument list for name, even if they are not used in determining the value of the parameter specified by ispec.
     *      2)  The problem dimensions n1, n2, n3, n4 are specified in the order that they appear in the argument list for name.
     *          n1 is used first, n2 second, and so on, and unused problem dimensions are passed a value of -1.
     *      3)  The parameter value returned by ILAENV is checked for validity in the calling subroutine.For example,
     *          ilaenv is used to retrieve the optimal blocksize for STRTRI as follows :
     *          nb = ilaenv(1, 'STRTRI', UPLO // DIAG, N, -1, -1, -1)
     *          if(nb<=1) nb = MAX(1, N)
     * TODO: optimize                                                                                                                  */
    static int ilaenv(int ispec, char const* name, char const* opts, int n1, int n2, int n3, int n4)
    {
        bool sname, cname;
        char c1;
        switch (ispec)
        {
            case 1:
            case 2:
            case 3:
                // Convert name to upper case.
                char subnam[6];
                std::strncpy(subnam, name, 6);
                int nb, nbmin;
                for (nb = 0; nb < 6; nb++)
                {
                    subnam[nb] = toupper(subnam[nb]);
                }
                c1 = subnam[0];
                sname = (c1 == 'S' || c1 == 'D');
                cname = (c1 == 'C' || c1 == 'Z');
                if (!(cname || sname))
                {
                    return 1;
                }
                char c2[2], c3[3], c4[2];
                std::strncpy(c2, subnam + 1, 2);
                std::strncpy(c3, subnam + 3, 3);
                std::strncpy(c4, c3 + 1, 2);
                switch (ispec)
                {
                    case 1:
                        // ispec = 1: block size
                        // In these examples, separate code is provided for setting nb for real and complex.
                        // We assume that nb will take the same value in single or double precision.
                        nb = 1;
                        if (std::strncmp(c2, "GE", 2) == 0)
                        {
                            if (std::strncmp(c3, "TRF", 3) == 0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                } else
                                {
                                    nb = 64;
                                }
                            } else if (std::strncmp(c3, "QRF", 3) == 0 || std::strncmp(c3, "RQF", 3) == 0 || std::strncmp(c3, "LQF", 3) == 0 || std::strncmp(c3, "QLF", 3) == 0)
                            {
                                if (sname)
                                {
                                    nb = 32;
                                } else
                                {
                                    nb = 32;
                                }
                            } else if (std::strncmp(c3, "HRD", 3) == 0)
                            {
                                if (sname)
                                {
                                    nb = 32;
                                } else
                                {
                                    nb = 32;
                                }
                            } else if (std::strncmp(c3, "BRD", 3) == 0)
                            {
                                if (sname)
                                {
                                    nb = 32;
                                } else
                                {
                                    nb = 32;
                                }
                            } else if (std::strncmp(c3, "TRI", 3) == 0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                } else
                                {
                                    nb = 64;
                                }
                            }
                        } else if (std::strncmp(c2, "PO", 2) == 0)
                        {
                            if (std::strncmp(c3, "TRF", 3) == 0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                } else
                                {
                                    nb = 64;
                                }
                            }
                        } else if (std::strncmp(c2, "SY", 2) == 0)
                        {
                            if (std::strncmp(c3, "TRF", 3) == 0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                } else
                                {
                                    nb = 64;
                                }
                            } else if (sname && std::strncmp(c3, "TRD", 3) == 0)
                            {
                                nb = 32;
                            } else if (sname && std::strncmp(c3, "GST", 3) == 0)
                            {
                                nb = 64;
                            }
                        } else if (cname && std::strncmp(c2, "HE", 2) == 0)
                        {
                            if (std::strncmp(c3, "TRF", 3) == 0)
                            {
                                nb = 64;
                            } else if (std::strncmp(c3, "TRD", 3) == 0)
                            {
                                nb = 32;
                            } else if (std::strncmp(c3, "GST", 3) == 0)
                            {
                                nb = 64;
                            }
                        } else if (sname && std::strncmp(c2, "OR", 2) == 0)
                        {
                            if (c3[0] == 'G')
                            {
                                if (std::strncmp(c4, "QR", 2) == 0 || std::strncmp(c4, "RQ", 2) == 0 || std::strncmp(c4, "LQ", 2) == 0 || std::strncmp(c4, "QL", 2) == 0 ||
                                        std::strncmp(c4, "HR", 2) == 0 || std::strncmp(c4, "TR", 2) == 0 || std::strncmp(c4, "BR", 2) == 0)
                                {
                                    nb = 32;
                                }
                            } else if (c3[0] == 'M')
                            {
                                if (std::strncmp(c4, "QR", 2) == 0 || std::strncmp(c4, "RQ", 2) == 0 || std::strncmp(c4, "LQ", 2) == 0 || std::strncmp(c4, "QL", 2) == 0 ||
                                        std::strncmp(c4, "HR", 2) == 0 || std::strncmp(c4, "TR", 2) == 0 || std::strncmp(c4, "BR", 2) == 0)
                                {
                                    nb = 32;
                                }
                            }
                        } else if (cname && std::strncmp(c2, "UN", 2) == 0)
                        {
                            if (c3[0] == 'G')
                            {
                                if (std::strncmp(c4, "QR", 2) == 0 || std::strncmp(c4, "RQ", 2) == 0 || std::strncmp(c4, "LQ", 2) == 0 || std::strncmp(c4, "QL", 2) == 0 ||
                                        std::strncmp(c4, "HR", 2) == 0 || std::strncmp(c4, "TR", 2) == 0 || std::strncmp(c4, "BR", 2) == 0)
                                {
                                    nb = 32;
                                }
                            } else if (c3[0] == 'M')
                            {
                                if (std::strncmp(c4, "QR", 2) == 0 || std::strncmp(c4, "RQ", 2) == 0 || std::strncmp(c4, "LQ", 2) == 0 || std::strncmp(c4, "QL", 2) == 0 ||
                                        std::strncmp(c4, "HR", 2) == 0 || std::strncmp(c4, "TR", 2) == 0 || std::strncmp(c4, "BR", 2) == 0)
                                {
                                    nb = 32;
                                }
                            }
                        } else if (std::strncmp(c2, "GB", 2) == 0)
                        {
                            if (std::strncmp(c3, "TRF", 3) == 0)
                            {
                                if (sname)
                                {
                                    if (n4 <= 64)
                                    {
                                        nb = 1;
                                    } else
                                    {
                                        nb = 32;
                                    }
                                } else
                                {
                                    if (n4 <= 64)
                                    {
                                        nb = 1;
                                    } else
                                    {
                                        nb = 32;
                                    }
                                }
                            }
                        } else if (std::strncmp(c2, "PB", 2) == 0)
                        {
                            if (std::strncmp(c3, "TRF", 3) == 0)
                            {
                                if (sname)
                                {
                                    if (n2 <= 64)
                                    {
                                        nb = 1;
                                    } else
                                    {
                                        nb = 32;
                                    }
                                } else
                                {
                                    if (n2 <= 64)
                                    {
                                        nb = 1;
                                    } else
                                    {
                                        nb = 32;
                                    }
                                }
                            }
                        } else if (std::strncmp(c2, "TR", 2) == 0)
                        {
                            if (std::strncmp(c3, "TRI", 3) == 0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                } else
                                {
                                    nb = 64;
                                }
                            }
                        } else if (std::strncmp(c2, "LA", 2) == 0)
                        {
                            if (std::strncmp(c3, "UUM", 3) == 0)
                            {
                                if (sname)
                                {
                                    nb = 64;
                                } else
                                {
                                    nb = 64;
                                }
                            }
                        } else if (sname && std::strncmp(c2, "ST", 2) == 0)
                        {
                            if (std::strncmp(c3, "EBZ", 3) == 0)
                            {
                                nb = 1;
                            }
                        }
                        return nb;
                        break;
                    case 2:
                        // ispec = 2: minimum block size
                        nbmin = 2;
                        if (std::strncmp(c2, "GE", 2) == 0)
                        {
                            if (std::strncmp(c3, "QRF", 3) == 0 || std::strncmp(c3, "RQF", 3) == 0 || std::strncmp(c3, "LQF", 3) == 0 || std::strncmp(c3, "QLF", 3) == 0)
                            {
                                if (sname)
                                {
                                    nbmin = 2;
                                } else
                                {
                                    nbmin = 2;
                                }
                            } else if (std::strncmp(c3, "HRD", 3) == 0)
                            {
                                if (sname)
                                {
                                    nbmin = 2;
                                } else
                                {
                                    nbmin = 2;
                                }
                            } else if (std::strncmp(c3, "BRD", 3) == 0)
                            {
                                if (sname)
                                {
                                    nbmin = 2;
                                } else
                                {
                                    nbmin = 2;
                                }
                            } else if (std::strncmp(c3, "TRI", 3) == 0)
                            {
                                if (sname)
                                {
                                    nbmin = 2;
                                } else
                                {
                                    nbmin = 2;
                                }
                            }
                        } else if (std::strncmp(c2, "SY", 2) == 0)
                        {
                            if (std::strncmp(c3, "TRF", 3) == 0)
                            {
                                if (sname)
                                {
                                    nbmin = 8;
                                } else
                                {
                                    nbmin = 8;
                                }
                            } else if (sname && std::strncmp(c3, "TRD", 3) == 0)
                            {
                                nbmin = 2;
                            }
                        } else if (cname && std::strncmp(c2, "HE", 2) == 0)
                        {
                            if (std::strncmp(c3, "TRD", 3) == 0)
                            {
                                nbmin = 2;
                            }
                        } else if (sname && std::strncmp(c2, "OR", 2) == 0)
                        {
                            if (c3[0] == 'G')
                            {
                                if (std::strncmp(c4, "QR", 2) == 0 || std::strncmp(c4, "RQ", 2) == 0 || std::strncmp(c4, "LQ", 2) == 0 || std::strncmp(c4, "QL", 2) == 0 ||
                                        std::strncmp(c4, "HR", 2) == 0 || std::strncmp(c4, "TR", 2) == 0 || std::strncmp(c4, "BR", 2) == 0)
                                {
                                    nbmin = 2;
                                }
                            } else if (c3[0] == 'M')
                            {
                                if (std::strncmp(c4, "QR", 2) == 0 || std::strncmp(c4, "RQ", 2) == 0 || std::strncmp(c4, "LQ", 2) == 0 || std::strncmp(c4, "QL", 2) == 0 ||
                                        std::strncmp(c4, "HR", 2) == 0 || std::strncmp(c4, "TR", 2) == 0 || std::strncmp(c4, "BR", 2) == 0)
                                {
                                    nbmin = 2;
                                }
                            }
                        } else if (cname && std::strncmp(c2, "UN", 2) == 0)
                        {
                            if (c3[0] == 'G')
                            {
                                if (std::strncmp(c4, "QR", 2) == 0 || std::strncmp(c4, "RQ", 2) == 0 || std::strncmp(c4, "LQ", 2) == 0 || std::strncmp(c4, "QL", 2) == 0 ||
                                        std::strncmp(c4, "HR", 2) == 0 || std::strncmp(c4, "TR", 2) == 0 || std::strncmp(c4, "BR", 2) == 0)
                                {
                                    nbmin = 2;
                                }
                            } else if (c3[0] == 'M')
                            {
                                if (std::strncmp(c4, "QR", 2) == 0 || std::strncmp(c4, "RQ", 2) == 0 || std::strncmp(c4, "LQ", 2) == 0 || std::strncmp(c4, "QL", 2) == 0 ||
                                        std::strncmp(c4, "HR", 2) == 0 || std::strncmp(c4, "TR", 2) == 0 || std::strncmp(c4, "BR", 2) == 0)
                                {
                                    nbmin = 2;
                                }
                            }
                        }
                        return nbmin;
                        break;
                    case 3:
                        // ispec = 3: crossover point
                        int nx = 0;
                        if (std::strncmp(c2, "GE", 2) == 0)
                        {
                            if (std::strncmp(c3, "QRF", 3) == 0 || std::strncmp(c3, "RQF", 3) == 0 || std::strncmp(c3, "LQF", 3) == 0 || std::strncmp(c3, "QLF", 3) == 0)
                            {
                                if (sname)
                                {
                                    nx = 128;
                                } else
                                {
                                    nx = 128;
                                }
                            } else if (std::strncmp(c3, "HRD", 3) == 0)
                            {
                                if (sname)
                                {
                                    nx = 128;
                                } else
                                {
                                    nx = 128;
                                }
                            } else if (std::strncmp(c3, "BRD", 3) == 0)
                            {
                                if (sname)
                                {
                                    nx = 128;
                                } else
                                {
                                    nx = 128;
                                }
                            }
                        } else if (std::strncmp(c2, "SY", 2) == 0)
                        {
                            if (sname && std::strncmp(c3, "TRD", 3) == 0)
                            {
                                nx = 32;
                            }
                        } else if (cname && std::strncmp(c2, "HE", 2) == 0)
                        {
                            if (std::strncmp(c3, "TRD", 3) == 0)
                            {
                                nx = 32;
                            }
                        } else if (sname && std::strncmp(c2, "OR", 2) == 0)
                        {
                            if (c3[0] == 'G')
                            {
                                if (std::strncmp(c4, "QR", 2) == 0 || std::strncmp(c4, "RQ", 2) == 0 || std::strncmp(c4, "LQ", 2) == 0 || std::strncmp(c4, "QL", 2) == 0 ||
                                        std::strncmp(c4, "HR", 2) == 0 || std::strncmp(c4, "TR", 2) == 0 || std::strncmp(c4, "BR", 2) == 0)
                                {
                                    nx = 128;
                                }
                            }
                        } else if (cname && std::strncmp(c2, "UN", 2) == 0)
                        {
                            if (c3[0] == 'G')
                            {
                                if (std::strncmp(c4, "QR", 2) == 0 || std::strncmp(c4, "RQ", 2) == 0 || std::strncmp(c4, "LQ", 2) == 0 || std::strncmp(c4, "QL", 2) == 0 ||
                                        std::strncmp(c4, "HR", 2) == 0 || std::strncmp(c4, "TR", 2) == 0 || std::strncmp(c4, "BR", 2) == 0)
                                {
                                    nx = 128;
                                }
                            }
                        }
                        return nx;
                        break;
                }
                break;
            case 4:
                // ispec = 4: number of shifts(used by xhseqr)
                return 6;
                break;
            case 5:
                // ispec = 5: minimum column dimension(not used)
                return 2;
                break;
            case 6:
                // ispec = 6: crossover point for SVD(used by xgelss and xgesvd)
                return int(T(n1 < n2 ? n1 : n2)*1.6);
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
                // ispec = 9: maximum size of the subproblems at the bottom of the computation
                // tree in the divide-and-conquer algorithm (used by xgelsd and xgesdd)
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
                // 12 <= ispec <= 16: xhseqr or one of its subroutines.
                return iparmq(ispec, name, opts, n1, n2, n3, n4);
                break;
            default:
                // Invalid value for ispec
                return -1;
        }
        return -1;
    }

    /*This program sets problem and machine dependent parameters useful for xHSEQR and its subroutines.
     * It is called whenever ilaenv is called with 12 <= ispec <= 16
     * Parameters: ispec: specifies which tunable parameter IPARMQ should return.
     *                    12: (inmin) Matrices of order nmin or less are sent directly to xLAHQR,
     *                        the implicit double shift QR algorithm. NMIN must be at least 11.
     *                    13: (inwin) Size of the deflation window. This is best set greater than
     *                        or equal to the number of simultaneous shifts ns. Larger matrices
     *                        benefit from larger deflation windows.
     *                    14: (inibl) Determines when to stop nibbling and invest in an(expensive)
     *                        multi-shift QR sweep. If the aggressive early deflation subroutine
     *                        finds LD converged eigenvalues from an order nw deflation window and
     *                        LD>(nw*NIBBLE) / 100, then the next QR sweep is skipped and early
     *                        deflation is applied immediately to the remaining active diagonal block.
     *                        Setting iparmq(ispec = 14) = 0 causes ttqre to skip a multi-shift QR
     *                        sweep whenever early deflation finds a converged eigenvalue. Setting
     *                        iparmq(ispec = 14) greater than or equal to 100 prevents ttqre from
     *                        skipping a multi-shift QR sweep.
     *                    15: (nshfts) The number of simultaneous shifts in a multi-shift QR iteration.
     *                    16: (iacc22) iparmq is set to 0, 1 or 2 with the following meanings.
     *                        0: During the multi-shift QR sweep, xlaqr5 does not accumulate
     *                           reflections and does not use matrix-matrix multiply to update
     *                           the far-from-diagonal matrix entries.
     *                        1: During the multi-shift QR sweep, xlaqr5 and/or xlaqr accumulates
     *                           reflections and uses matrix-matrix multiply to update the
     *                           far-from-diagonal matrix entries.
     *                        2: During the multi-shift QR sweep. xlaqr5 accumulates reflections and
     *                           takes advantage of 2-by-2 block structure during matrix-matrix multiplies.
     *                        (If xTRMM is slower than xgemm, then iparmq(ispec = 16) = 1 may be more
     *                         efficient than iparmq(ispec = 16) = 2 despite the greater level of
     *                         arithmetic work implied by the latter choice.)
     *             name: Name of the calling subroutine
     *             opts: This is a concatenation of the string arguments to ttqre.
     *             n: n is the order of the Hessenberg matrix H.
     *             ilo: integer
     *             ihi: integer
     *                  It is assumed that H is already upper triangular in rows and columns 1:ilo-1 and ihi+1:n.
     *             lwork: The amount of workspace available.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Further Details:
     *     Little is known about how best to choose these parameters. It is possible to use
     *         different values of the parameters for each of chseqr, dhseqr, shseqr and zhseqr.
     *     It is probably best to choose different parameters for different matrices and different parameters
     *         at different times during the iteration, but this has not been implemented yet.
     *     The best choices of most of the parameters depend in an ill-understood way on the relative execution
     *         rate of xlaqr3 and xlaqr5 and on the nature of each particular eigenvalue problem.
     *         Experiment may be the only practical way to determine which choices are most effective.
     *     Following is a list of default values supplied by iparmq. These defaults may be adjusted
     *         in order to attain better performance in any particular computational environment.
     *     iparmq(ispec = 12) The xlahqr vs xlaqr0 crossover point.
     *         Default: 75. (Must be at least 11.)
     *     iparmq(ispec = 13) Recommended deflation window size.
     *         This depends on ilo, ihi and ns, the number of simultaneous shifts returned by iparmq(ispec = 15).
     *         The default for (ihi-ilo+1)<=500 is ns. The default for (ihi-ilo+1)>500 is 3*ns/2.
     *     iparmq(ispec = 14) Nibble crossover point. Default: 14.
     *     iparmq(ispec = 15) Number of simultaneous shifts, ns.
     *         a multi-shift QR iteration. If ihi-ilo+1 is ...
     *         greater than or equal to...      but less than...      the default is
     *                                   0                    30                     ns = 2+
     *                                  30                    60                     ns = 4+
     *                                  60                   150                     ns = 10
     *                                 150                   590                     ns = **
     *                                 590                  3000                     ns = 64
     *                                3000                  6000                     ns = 128
     *                                6000                  infinity                 ns = 256
     *         (+) By default matrices of this order are passed to the implicit double shift routine xlahqr.
     *             See iparmq(ispec = 12) above. These values of ns are used only in case of a rare xlahqr failure.
     *         (**) an ad-hoc function increasing from 10 to 64.
     *     iparmq(ispec = 16) Select structured matrix multiply. (See ispec = 16 above for details.) Default: 3.         */
    static int iparmq(int ispec, char const* name, char const* opts, int n, int ilo, int ihi, int lwork)
    {
        const int INMIN = 12, INWIN = 13, INIBL = 14, ISHFTS = 15, IACC22 = 16, NMIN = 75, K22MIN = 14, KACMIN = 14, NIBBLE = 14, KNWSWP = 500;
        int nh, ns = 0, nstemp;
        if ((ispec == ISHFTS) || (ispec == INWIN) || (ispec == IACC22))
        {
            // Set the number simultaneous shifts
            nh = ihi - ilo + 1;
            ns = 2;
            if (nh >= 30)
            {
                ns = 4;
            }
            if (nh >= 60)
            {
                ns = 10;
            }
            if (nh >= 150)
            {
                nstemp = nh / int(std::log(T(nh)) / std::log(TWO));
                ns = 10 > nstemp ? 10 : nstemp;
            }
            if (nh >= 590)
            {
                ns = 64;
            }
            if (nh >= 3000)
            {
                ns = 128;
            }
            if (nh >= 6000)
            {
                ns = 256;
            }
            nstemp = ns - (ns % 2);
            ns = 2 > nstemp ? 2 : nstemp;
        }
        if (ispec == INMIN)
        {
            // Matrices of order smaller than NMIN get sent to xLAHQR, the classic double shift algorithm. This must be at least 11.
            return NMIN;
        } else if (ispec == INIBL)
        {
            // INIBL: skip a multi-shift qr iteration and whenever aggressive early
            // deflation finds at least(NIBBLE*(window size) / 100) deflations.
            return NIBBLE;
        } else if (ispec == ISHFTS)
        {
            // NSHFTS: The number of simultaneous shifts
            return ns;
        } else if (ispec == INWIN)
        {
            // nw: deflation window size.
            if (nh <= KNWSWP)
            {
                return ns;
            } else
            {
                return 3 * ns / 2;
            }
        } else if (ispec == IACC22)
        {
            // IACC22: Whether to accumulate reflections before updating the far-from-diagonal elements
            // and whether to use 2-by-2 block structure while doing it. A small amount of work could be
            // saved by making this choice dependent also upon the nh = ihi-ilo+1.
            if (ns >= KACMIN)
            {
                return 1;
            }
            if (ns >= K22MIN)
            {
                return 2;
            }
            return 0;
        } else
        {
            // invalid value of ispec
            return -1;
        }
    }

    /* xerbla is an error handler for the LAPACK routines.
     * It is called by an LAPACK routine if an input parameter has an invalid value.
     * A message is printed and execution stops.
     * Installers may consider modifying the STOP statement in order to call
     * system-specific exception-handling facilities.
     * Parameters: srname: The name of the routine which called xerbla.
     *             info: The position of the invalid parameter in the parameter list of the calling routine.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void xerbla(char const* srname, int info)
    {
        std::cerr << "On entry to " << srname << " parameter number " << info << " had an illegal value.";
        throw info;
    }

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

    // xlaenv seems to set values in common block iparms, claiming it is later used by ilaenv,
    // but since this does not happen, ignore use of xlaenv.

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
    static void dlagge(int m, int n, int kl, int ku, T const* d, T* A, int lda, int* iseed,
                       T* work, int& info)
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
            xerbla("DLAGGE", -info);
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
        T tau, wa, wb, wn;
        // pre- and post-multiply A by random orthogonal matrices
        for (i=(m<n?m:n)-1; i>=0; i--)
        {
            aind = i+lda*i;
            if (i<m-1)
            {
                // generate random reflection
                dlarnv(3, iseed, m-i, work);
                wn = dnrm2(m-i, work, 1);
                wa = std::fabs(wn)*T((ZERO<=work[0])-(work[0]<ZERO));
                if (wn==ZERO)
                {
                    tau = ZERO;
                }
                else
                {
                    wb = work[0] + wa;
                    dscal(m-i-1, ONE/wb, &work[1], 1);
                    work[0] = ONE;
                    tau = wb / wa;
                }
                // multiply A[i:m-1,i:n-1] by random reflection from the left
                dgemv("Transpose", m-i, n-i, ONE, &A[aind], lda, work, 1, ZERO, &work[m], 1);
                dger(m-i, n-i, -tau, work, 1, &work[m], 1, &A[aind], lda);
            }
            if (i<n-1)
            {
                // generate random reflection
                dlarnv(3, iseed, n-i, work);
                wn = dnrm2(n-i, work, 1);
                wa = std::fabs(wn)*T((ZERO<=work[0])-(work[0]<ZERO));
                if (wn==ZERO)
                {
                    tau = ZERO;
                }
                else
                {
                    wb = work[0] + wa;
                    dscal(n-i-1, ONE / wb, &work[1], 1);
                    work[0] = ONE;
                    tau = wb / wa;
                }
                // multiply A[i:m-1,i:n-1] by random reflection from the right
                dgemv("No transpose", m-i, n-i, ONE, &A[aind], lda, work, 1, ZERO, &work[n], 1);
                dger(m-i, n-i, -tau, &work[n], 1, work, 1, &A[aind], lda);
            }
        }
        // Reduce number of subdiagonals to KL and number of superdiagonals to KU
        for (i=0; i<m-1-kl || i<n-1-ku; i++)
        {
            if (kl<=ku)
            {
                // annihilate subdiagonal elements first (necessary if KL==0)
                if (i<m-1-kl && i<n)
                {
                    aind = kl+i+lda*i;
                    // generate reflection to annihilate A[kl+i+1:m-1,i]
                    wn = dnrm2(m-kl-i, &A[aind], 1);
                    wa = std::fabs(wn)*T((ZERO<=A[aind])-(A[aind]<ZERO));
                    if (wn==ZERO)
                    {
                        tau = ZERO;
                    }
                    else
                    {
                        wb = A[aind] + wa;
                        dscal(m-1-kl-i, ONE/wb, &A[1+aind], 1);
                        A[aind] = ONE;
                        tau = wb / wa;
                    }
                    // apply reflection to A[kl+i:m-1,i+1:n-1] from the left
                    dgemv("Transpose", m-kl-i, n-i-1, ONE, &A[aind+lda], lda, &A[aind], 1, ZERO,
                          work, 1);
                    dger(m-kl-i, n-i-1, -tau, &A[aind], 1, work, 1, &A[aind+lda], lda);
                    A[kl+i+lda*i] = -wa;
                }
                if (i<n-1-ku && i<m)
                {
                    aind = i+lda*(ku+i);
                    // generate reflection to annihilate A[i,ku+i+1:n-1]
                    wn = dnrm2(n-ku-i, &A[aind], lda);
                    wa = std::fabs(wn)*T((ZERO<=A[aind])-(A[aind]<ZERO));
                    if (wn==ZERO)
                    {
                        tau = ZERO
                    }
                    else
                    {
                        wb = A[aind] + wa;
                        dscal(n-ku-i-1, ONE/wb, &A[aind+lda], lda);
                        A[aind] = ONE;
                        tau = wb / wa;
                    }
                    // apply reflection to A[i+1:m-1,ku+i:n-1] from the right
                    dgemv("No transpose", m-i-1, n-ku-i, ONE, &A[1+aind], lda, &A[aind], lda, ZERO,
                          work, 1);
                    dger(m-i-1, n-ku-i, -tau, work, 1, &A[aind], lda, &A[1+aind], lda);
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
                    wn = dnrm2(n-ku-i, &A[aind], lda);
                    wa = std::fabs(wn)*T((ZERO<=A[aind])-(A[aind]<ZERO));
                    if (wn==ZERO)
                    {
                        tau = ZERO
                    }
                    else
                    {
                        wb = A[aind] + wa;
                        dscal(n-ku-i-1, ONE/wb, &A[aind+lda], lda);
                        A[aind] = ONE;
                        tau = wb / wa;
                    }
                    // apply reflection to A[i+1:m-1,ku+i:n-1] from the right
                    dgemv("No transpose", m-i-1, n-ku-i, ONE, &A[1+aind], lda, &A[aind], lda, ZERO,
                          work, 1);
                    dger(m-i-1, n-ku-i, -tau, work, 1, &A[aind], lda, &A[1+aind], lda);
                    A[aind] = -wa;
                }
                if (i<m-1-kl && i<n)
                {
                    aind = kl+i+lda*i;
                    // generate reflection to annihilate A[kl+i+1:m-1,i]
                    wn = dnrm2(m-kl-i, &A[aind], 1);
                    wa = std::fabs(wn)*T((ZERO<=A[aind])-(A[aind]<ZERO));
                    if (wn==ZERO)
                    {
                        tau = ZERO;
                    }
                    else
                    {
                        wb = A[aind] + wa;
                        dscal(m-kl-i-1, ONE/wb, &A[1+aind], 1);
                        A[aind] = ONE;
                        tau = wb / wa;
                    }
                    // apply reflection to A[kl+i:m-1,i+1:n-1] from the left
                    dgemv("Transpose", m-kl-i, n-i-1, ONE, &A[aind+lda], lda, &A[aind], 1, ZERO,
                          work, 1);
                    dger(m-kl-i, n-i-1, -tau, &A[aind], 1, work, 1, &A[aind+lda], lda);
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
    static void dlagsy(int n, int k, T const* d, T* A, int lda, int* iseed, T* work, int& info)
    {
        const T HALF = 0.5;
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
            xerbla("DLAGSY", -info);
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
        T alpha, tau, wa, wb, wn;
        for (i=n-2; i>=0; i--)
        {
            // generate random reflection
            dlarnv(3, iseed, n-i, work);
            wn = dnrm2(n-i, work, 1);
            wa = std::fabs(wn)*T((ZERO<=work[0])-(work[0]<ZERO));
            if (wn==ZERO)
            {
                tau = ZERO;
            }
            else
            {
                wb = work[0]+wa;
                dscal(n-i-1, ONE/wb, &work[1], 1);
                work[0] = ONE;
                tau = wb/wa;
            }
            // apply random reflection to A[i:n-1,i:n-1] from the left and the right
            aind = i+lda*i;
            // compute  y := tau * A * u
            dsymv("Lower", n-i, tau, &A[aind], lda, work, 1, ZERO, &work[n], 1);
            // compute  v := y - 1/2 * tau * (y, u) * u
            alpha = -HALF * tau * ddot(n-i, &work[n], 1, work, 1);
            daxpy(n-i, alpha, work, 1, &work[n], 1);
            // apply the transformation as a rank-2 update to A[i:n-1,i:n-1]
            dsyr2("Lower", n-i, -ONE, work, 1, &work[n], 1, &A[aind], lda);
        }
        // Reduce number of subdiagonals to K
        for (i=0; i<n-1-k; i++)
        {
            aind = k+i+lda*i;
            // generate reflection to annihilate A[k+i+1:n-1,i]
            wn = dnrm2(n-k-i, &A[aind], 1);
            wa = std::fabs(wn)*T((ZERO<=A[aind])-(A[aind]<ZERO));
            if (wn==ZERO)
            {
                tau = ZERO;
            }
            else
            {
                wb = A[aind]+wa;
                dscal(n-k-i-1, ONE/wb, &A[1+aind], 1);
                A[aind] = ONE;
                tau = wb/wa;
            }
            // apply reflection to A[k+i:n-1,i+1:k+i-1] from the left
            dgemv("Transpose", n-k-i, k-1, ONE, &A[aind+lda], lda, &A[aind], 1, ZERO, work, 1);
            dger(n-k-i, k-1, -tau, &A[aind], 1, work, 1, &A[aind+lda], lda);
            // apply reflection to A[k+i:n-1,k+i:n-1] from the left and the right
            // compute  y := tau * A * u
            dsymv('Lower', n-k-i, tau, &A[aind+lda*k], lda, &A[aind], 1, ZERO, work, 1);
            // compute  v := y - 1/2 * tau * (y, u) * u
            alpha = -HALF * tau * ddot(n-k-i, work, 1, &A[aind], 1);
            daxpy(n-k-i, alpha, &A[aind], 1, work, 1);
            // apply symmetric rank-2 update to A[k+i:n-1,k+i:n-1]
            dsyr2("Lower", n-k-i, -ONE, &A[aind], 1, work, 1, &A[aind+lda*k], lda);
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
            for (i=j+1; i<n; u++)
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
    static T dlaran(int* iseed)
    {
        const int M1 = 494;
        const int M2 = 322;
        const int M3 = 2508;
        const int M4 = 2549;
        const int IPW2 = 4096;
        const T R = ONE / IPW2;
        int it1, it2, it3, it4;
        T rndout;
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
            rndout = R*(T(it1)+R*(T(it2)+R*(T(it3)+R*(T(it4)))));
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
    static T dlarnd(int idist, int* iseed)
    {
        const T TWOPI = 6.2831853071795864769252867663;
        // Generate a real random number from a uniform (0,1) distribution
        T t1 = dlaran(iseed);
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
            T t2 = dlaran(iseed);
            return std::sqrt(-TWO*std::log(t1))*std::cos(TWOPI*t2);
        }
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
    static void dlarot(bool lrows, bool lleft, bool lright, int nl, T c, T s, T* A, int lda,
                       T& xleft, T& xright)
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
        T xt[2];
        T yt[2];
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
            xerbla("DLAROT", 4);
            return;
        }
        if (lda<=0 || (!lrows && lda<nl-nt))
        {
            xerbla("DLAROT", 8);
            return;
        }
        // Rotate
        drot(nl-nt, &A[ix], iinc, &A[iy], iinc, c, s);
        drot(nt, xt, 1, yt, 1, c, s);
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
    static void dlatm1(int mode, T cond, int irsign, int idist, int* iseed, T* d, int n, int& info)
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
            xerbla("DLATM1", -info);
            return;
        }
        // Compute d according to cond and mode
        if (mode!=0)
        {
            int i;
            T alpha, temp;
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
                        alpha = std::pow(cond, (-ONE/T(n-1)));
                        for (i=1; i<n; i++)
                        {
                            d[i] = std::pow(alpha, T(i))
                        }
                    }
                    break;
                case 4:
                    // Arithmetically distributed d values:
                    d[0] = ONE;
                    if (n>1)
                    {
                        temp = ONE / cond;
                        alpha = (ONE-temp) / T(n-1);
                        for (i=1; i<n; i++)
                        {
                            d[i] = T(n-1-i)*alpha + temp;
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
                    dlarnv(idist, iseed, n, d);
                    break;
            }
            // If mode neither -6 nor 0 nor 6, and IRSIGN = 1, assign random signs to d
            if ((mode!=-6 && mode!=0 && mode!=6) && irsign==1)
            {
                for (i=0; i<n; i++)
                {
                    temp = dlaran(iseed);
                    if (temp>T(0.5))
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
    static void dlatms(int m, int n, const char* dist, int* iseed, const char* sym, T* d, int mode,
                       T cond, T dmax, int kl, int ku, char const* pack, T* A, int lda, T* work,
                       int& info)
    {
        const T TWOPI = 6.2831853071795864769252867663;
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
        int irsign, isym;
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
                ipack = 3
                isympk = 2
                break;
            case 'R':
                ipack = 4
                isympk = 3
                break;
            case 'B':
                ipack = 5
                isympk = 3
                break;
            case 'Q':
                ipack = 6
                isympk = 2
                break;
            case 'Z':
                ipack = 7
                break;
            default:
                ipack = -1;
                break;
        }
        // Set certain internal parameters
        int mnmin = ((m<n) ? m : n);
        int llb = ((kl<m-1) ? kl : m-1);
        int uub = ((ku<n-1) ? ku : n-1);
        int mr = ((m<n+llb) ? m : n+llb);
        int nc = ((n<m+uub) ? n : m+uub);
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
            if (T(llb+uub)<T(0.3)*T(1>mr+nc?1:mr+nc))
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
            xerbla("DLATMS", -info);
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
        dlatm1(mode, cond, irsign, idist, iseed, d, mnmin, iinfo);
        if (iinfo!=0)
        {
            info = 1;
            return;
        }
        // Choose Top-Down if d is (apparently) increasing, Bottom-Up if d is (apparently)
        // decreasing.
        bool topdwn;
        if (std::fabs(d[0])<=std::fabs(d[mnmin-1]))
        {
            topdwn = true;
        }
        else
        {
            topdwn = false;
        }
        T alpha, temp, temp2;
        if (mode!=0 && std::abs(mode)!=6)
        {
            // Scale by dmax
            temp = std::fabs(d[0]);
            for (i=1; i<mnmin; i++)
            {
                temp2 = std::fabs(d[i]);
                if (temp2>temp)
                {
                    temp = temp2;
                }
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
            dscal(mnmin, alpha, d, 1);
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
        dlaset("Full", lda, n, ZERO, ZERO, A, lda);
        bool ilextr, iltemp;
        int icol, il, irow, itemp1, itemp2, jc, jch, jr;
        T angle, c, dummy, extra, s;
        if (llb==0 && uub==0)
        {
            // Diagonal Matrix: We are done, unless it is to be stored SP/PP/TP (pack=='R' or 'C')
            dcopy(mnmin, d, 1, &A[ioffst/*+lda*0*/], ilda+1);
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
                dcopy(mnmin, d, 1, &A[ioffst/*+lda*0*/], ilda+1);
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
                            icol = ((1>=jr-jkl) ? 0 : jr-jkl-1);
                            if (jr<m-1)
                            {
                                il = (n-2<jr+jku?n-1:jr+jku+1) - icol;
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
                                    dlartg(A[ir+ioffst+1+ldamiskm1*(ic+1)], extra, c, s, dummy);
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
                                    dlartg(A[irow+ioffst+1+ldamiskm1*(ic+1)], temp, c, s, dummy);
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
                            irow = ((1>=jc-jku) ? 0 : jc-jku-1);
                            if (jc<n-1)
                            {
                                il = (m-2<=jc+jkl?m-1:jc+jkl+1) - irow;
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
                                    dlartg(A[ir+1+ioffst+ldamiskm1*(ic+1)], extra, c, s,
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
                                    dlartg(A[ir+1+ioffst+ldamiskm1*(icol+1)], temp, c, s, dummy);
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
                        iendch = ((m-1<n+jkl) ? m-2 : n-1+jkl);
                        for (jc=(m+jku<=n-1?m+jku-1:n-2); jc>=-(jkl+1); jc--)
                        {
                            extra = ZERO;
                            angle = TWOPI * dlarnd(1, iseed);
                            c = std::cos(angle);
                            s = std::sin(angle);
                            irow = ((0>=jc-jku) ? 0 : jc-jku);
                            if (jc>=0)
                            {
                                il = (m-3<=jc+jkl?m-1:jc+jkl+2) - irow;
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
                                    dlartg(A[jch+ioffst+ldamiskm1*ic], extra, c, s, dummy);
                                }
                                if (0>ic)
                                {
                                    ic = 0;
                                }
                                icol = ((n-3<=jch+jku) ? n-2 : jch+jku+1);
                                iltemp = (jch+jku < n-2);
                                temp = ZERO;
                                dlarot(true, ilextr, iltemp, icol+2-ic, c, s,
                                       &A[jch+ioffst+ldamiskm1*ic], ilda, extra, temp);
                                if (iltemp)
                                {
                                    dlartg(A[jch+ioffst+ldamiskm1*icol], temp, c, s, dummy);
                                    il = (iendch-jch<=2+jkl+jku ? iendch-jch+1 : 3+jkl+jku);
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
                        iendch = ((n-1<=m+jku) ? n-2 : m-1+jku);
                        itemp1 = ((n+jkl<m-1) ? n+jkl-1 : m-2);
                        for (jr=itemp1; jr>=-jku-1; jr--)
                        {
                            extra = ZERO;
                            angle = TWOPI * dlarnd(1, iseed);
                            c = std::cos(angle);
                            s = std::sin(angle);
                            icol = ((0>=jr-jkl) ? 0 : jr-jkl);
                            if (jr>=0)
                            {
                                il = ((n-3<=jr+jku) ? n-1 : jr+jku+2) - icol;
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
                                    dlartg(&A[ir+ioffst+ldamiskm1*jch], extra, c, s, dummy);
                                }
                                if (0>ir)
                                {
                                    ir = 0;
                                }
                                irow = ((m-3<=jch+jkl) ? m-2 : jch+jkl+1);
                                iltemp = ((jch+jkl+2) < m);
                                temp = ZERO;
                                dlarot(false, ilextr, iltemp, irow+2-ir, c, s,
                                       &A[ir+ioffst+ldamiskm1*jch], ilda, extra, Stemp);
                                if (iltemp)
                                {
                                    dlartg(A[irow+ioffst+ldamiskm1*jch], temp, c, s, dummy);
                                    il = ((iendch-jch<=2+jkl+jku) ? iendch+1-jch : 3+jkl+jku);
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
                    dcopy(mnmin, d, 1, &A[ioffg/*+lda*0*/], ilda+1);
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
                            dlarot(true, true, false, (itemp1<=n-jc?itemp1:n-jc), c, s,
                                   &A[ioffg+(lda-iskew)*jc], ilda, temp, dummy);
                            // Chase EXTRA back up the matrix
                            icol = jc;
                            for (jch=jc-k-1; jch>=0; jch-=(k+1))
                            {
                                dlartg(A[jch+1+ioffg+ldamiskm1*(icol+1)], extra, c, s, dummy);
                                temp = A[jch+ioffg+ldamiskm1*(jch+1)];
                                dlarot(true, true, true, k+3, c, -s, &A[ioffg+(lda-iskew)*jch],
                                       ilda, temp, extra);
                                irow = ((1>=jch-k) ? 0 : jch-k-1);
                                il = ((jch+1<itemp1) ? jch+1 : itemp1);
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
                    dcopy(mnmin, d, 1, &A[ioffg/*+lda*0*/], ilda+1);
                    for (k=0; k<uub; k++)
                    {
                        for (jc=n-2; jc>=0; jc--)
                        {
                            il = ((n-jc-3<k) ? n-jc-1 : k+2);
                            extra = ZERO;
                            temp = A[ioffg+1+(lda-iskew)*jc];
                            angle = TWOPI * dlarnd(1, iseed);
                            c = std::cos(angle);
                            s = -std::sin(angle);
                            dlarot(false, true, n-2-jc>k, il+1, c, s, &A[ioffg+(lda-iskew)*jc],
                                   ilda, temp, extra);
                            icol = jc-k;
                            if (icol<0)
                            {
                                icol = 0;
                            }
                            dlarot(true, false, true, jc+2-icol, c, s,
                                   &A[jc+ioffg+ldamiskm1*icol], ilda, dummy, temp);
                            // Chase EXTRA back down the matrix
                            icol = jc;
                            for (jch=jc+k+1; jch<n-1; jch+=(k+1))
                            {
                                dlartg(A[jch+ioffg+ldamiskm1*icol], extra, c, s, dummy);
                                temp = A[ioffg+1+(lda-iskew)*jch];
                                dlarot(true, true, true, k+3, c, s, &A[jch+ioffg+ldamiskm1*icol],
                                       ilda, extra, temp);
                                il = ((n-3-jch<k) ? n-1-jch : k+2);
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
                    itemp1 = ((j+llb<m-1) ? j+llb : m-1);
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
}

#endif