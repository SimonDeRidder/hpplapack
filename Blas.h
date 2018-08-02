#ifndef BLAS_HEADER
#define BLAS_HEADER

#include <cstring>
#include <cmath>
#include <algorithm>

template<class real> class Blas
{
public:
    // constants

    static constexpr real ZERO = real(0.0);
    static constexpr real ONE  = real(1.0);

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
    static real dasum(int n, real const* dx, int incx)
    {
        if (n<=0 || incx<=0)
        {
            return ZERO;
        }
        int i;
        real dtemp = ZERO;
        if (incx==1)
        {
            // code for increment equal to 1
            // clean-up loop
            int m = n%6;
            if (m!=0)
            {
                for (i=0; i<m; i++)
                {
                    dtemp += std::fabs(dx[i]);
                }
                if (n<6)
                {
                    return dtemp;
                }
            }
            for (i=m; i<n; i+=6)
            {
                dtemp += std::fabs(dx[i])+std::fabs(dx[i+1])+std::fabs(dx[i+2])
                        +std::fabs(dx[i+3])+std::fabs(dx[i+4])+std::fabs(dx[i+5]);
            }
        }
        else
        {
            // code for increment not equal to 1
            int nincx = n*incx;
            for (i=0; i<nincx; i+=incx)
            {
                dtemp += std::fabs(dx[i]);
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
    static void daxpy(int n, real da, real const* dx, int incx, real* dy, int incy)
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
    static void dcopy(int n, real const* dx, int incx, real* dy, int incy)
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
    static real ddot(int n, real const* dx, int incx, real const* dy, int incy)
    {
        if (n<=0)
        {
            return ZERO;
        }
        int i;
        real dtemp = ZERO;
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
     *                On exit, the array C is overwritten by the m by n matrix
     *                         (alpha*op(A)*op(B) + beta*C).
     *             ldc: On entry, ldc specifies the first dimension of C as declared in the
     *                  calling (sub)program. ldc must be at least max(1, m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                      */
    static void dgemm(char const* transa, char const* transb, int m, int n, int k, real alpha,
                      real const* A, int lda, real const* B, int ldb, real beta, real* C, int ldc)
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
        } else if (lda < std::max(1, nrowa))
        {
            info = 8;
        } else if (ldb < std::max(1, nrowb))
        {
            info = 10;
        } else if (ldc < std::max(1, m))
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
        real TEMP;
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
     *             x: an array of DIMENSION at least (1 + (n - 1)*abs(incx)) when trans=='N' or 'n'
     *                                  and at least (1 + (m - 1)*abs(incx)) otherwise.
     *                Before entry, the incremented array x must contain the vector x.
     *             incx: specifies the increment for the elements of x. incx must not be zero.
     *             beta: specifies the scalar beta. When beta is supplied as zero then y need not
     *                   be set on input.
     *             y: an array of DIMENSION at least (1 + (m - 1)*abs(incy)) when trans=='N' or 'n'
     *                                  and at least (1 + (n - 1)*abs(incy)) otherwise.
     *                Before entry with beta non - zero, the incremented array y must contain the
     *                vector y.
     *                On exit, y is overwritten by the updated vector y.
     *             incy: specifies the increment for the elements of y. incy must not be zero.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                           */
    static void dgemv(char const* trans, int m, int n, real alpha, real const* A, int lda,
                      real const* x, int incx, real beta, real* y, int incy)
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
        } else if (lda < std::max(0, m-1))
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
        // Set lenx and leny, the lengths of the vectors x and y, and set up the start points
        // in x and y.
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
        // Start the operations. In this version the elements of A are accessed sequentially with
        // one pass through A.
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
        real temp;
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
                for (j = 0; j < n; j++)
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
     * where alpha is a scalar, x is an m element vector, y is an n element vector and A is an m by
     * n matrix.
     * Parameters: m: On entry, m specifies the number of rows of the matrix A. m must be at least
     *                zero.
     *             n: On entry, n specifies the number of columns of the matrix A. n must be at
     *                least zero.
     *             alpha: On entry, alpha specifies the scalar alpha.
     *             x: an array of dimension at least (1 + (m - 1)*abs(incx)).
     *                Before entry, the incremented array x must contain the m element vector x.
     *             incx: On entry, incx specifies the increment for the elements of x. incx must
     *                   not be zero.
     *             y: an array of dimension at least (1 + (n - 1)*abs(incy)).
     *                Before entry, the incremented array y must contain the n element vector y.
     *             incy: On entry, incy specifies the increment for the elements of y. incy must
     *                   not be zero.
     *             A: an array of DIMENSION(lda, n).
     *                Before entry, the leading m by n part of the array A must contain the matrix
     *                of coefficients.
     *                On exit, A is overwritten by the updated matrix.
     *             lda: On entry, lda specifies the first dimension of A as declared in the calling
     *                  (sub)program. lda must be at least max(1, m).
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                                           */
    static void dger(int m, int n, real alpha, real const* x, int incx, real const* y, int incy,
                     real* A, int lda)
    {
        real temp;
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
        } else if (lda < std::max(1, m))
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
    static real dnrm2(int n, real const* x, int incx)
    {
        real absxi, norm, scale, ssq, temp;
        int ix;
        if (n < 1 || incx < 0)
        {
            norm = ZERO;
        } else if (n == 1)
        {
            norm = std::fabs(x[0]);
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
                    absxi = std::fabs(x[ix]);
                    if (scale < absxi)
                    {
                        temp = scale / absxi;
                        ssq = ONE + ssq * temp * temp;
                        scale = absxi;
                    } else
                    {
                        temp = absxi / scale;
                        ssq += temp * temp;
                    }
                }
            }
            norm = scale * std::sqrt(ssq);
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
    static void drot(int n, real* dx, int incx, real* dy, int incy, real c, real s)
    {
        if (n<=0)
        {
            return;
        }
        real dtemp;
        int i;
        if (incx==1 && incy==1)
        {
            // code for both increments equal to 1
            for (i=0; i<n; i++)
            {
                dtemp = c*dx[i] + s*dy[i];
                dy[i] = c*dy[i] - s*dx[i];
                dx[i] = dtemp;
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
                dtemp  = c*dx[ix] + s*dy[iy];
                dy[iy] = c*dy[iy] - s*dx[ix];
                dx[ix] = dtemp;
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
    static void dscal(int n, real da, real* dx, int incx)
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
    static void dswap(int n, real* dx, int incx, real* dy, int incy)
    {
        if (n <= 0)
        {
            return;
        }
        real dtemp;
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
    static void dsymv(char const* uplo, int n, real alpha, real const* A, int lda, real const* x,
                      int incx, real beta, real* y, int incy)
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
        real temp1, temp2;
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
                        y[iy] += temp1 * A[i+ldaj];
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
    static void dsyr2(char const* uplo, int n, real alpha, real const* x, int incx, real const* y,
                      int incy, real* A, int lda)
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
        real temp1, temp2;
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
    static void dtrmm(char const* side, char const* uplo, char const* transa, char const* diag,
                      int m, int n, real alpha, real const* A, int lda, real* B, int ldb)
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
        } else if (lda < std::max(1, nrowa))
        {
            info = 9;
        } else if (ldb < std::max(1, m))
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
        real temp;
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
     * where x is an n element vector and A is an n by n unit, or non-unit, upper or lower
     * triangular matrix.
     * Parameters: uplo: On entry, uplo specifies whether the matrix is an upper or lower
     *                             triangular matrix as follows:
     *                   uplo=='U' or 'u'   A is an upper triangular matrix.
     *                   uplo=='L' or 'l'   A is a lower triangular matrix.
     *             trans: On entry, trans specifies the operation to be performed as follows:
     *                    trans=='N' or 'n' x := A*x.
     *                    trans=='T' or 't' x := A^T*x.
     *                    trans=='C' or 'c' x := A^T*x.
     *             diag: On entry, diag specifies whether or not A is unit triangular as follows:
     *                   diag=='U' or 'u': A is assumed to be unit triangular.
     *                   diag=='N' or 'n': A is not assumed to be unit triangular.
     *             n: On entry, n specifies the order of the matrix A. n must be at least zero.
     *             A: an array of DIMENSION(lda, n).
     *                Before entry with uplo=='U' or 'u', the leading n by n upper triangular part
     *                of the array A must contain the upper triangular matrix and the strictly
     *                lower triangular part of A is not referenced.
     *                Before entry with uplo=='L' or 'l', the leading n by n lower triangular part
     *                of the array A must contain the lower triangular matrix and the strictly
     *                upper triangular part of A is not referenced.
     *                Note that when  diag=='U' or 'u', the diagonal elements of A are not
     *                referenced either, but are assumed to be unity.
     *             lda: On entry, lda specifies the first dimension of A as declared in the calling
     *                  (sub)program. lda must be at least max(1, n).
     *             x: an array of dimension at least (1 + (n - 1)*abs(incx)).
     *                Before entry, the incremented array x must contain the n element vector x.
     *                On exit, x is overwritten with the tranformed vector x.
     *             incx: On entry, incx specifies the increment for the elements of x.incx must
     *                   not be zero.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.                                                                         */
    static void dtrmv(char const* uplo, char const* trans, char const* diag, int n, real const* A,
                      int lda, real* x, int incx)
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
        } else if (lda < std::max(1, n))
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
        real temp;
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
     *          NAG Ltd.                                                                         */
    static int idamax(int n, real const* dx, int incx)
    {
        if (n<1 || incx<=0)
        {
            return -1;
        }
        if (n==1)
        {
            return 0;
        }
        real dmax, dnext;
        int i, ida=0;
        if (incx==1)
        {
            // code for increment equal to 1
            dmax = std::fabs(dx[0]);
            for (i=1; i<n; i++)
            {
                dnext = std::fabs(dx[i]);
                if (dnext > dmax)
                {
                    ida = i;
                    dmax = dnext;
                }
            }
        } else
        {
            // code for increment not equal to 1
            dmax = std::fabs(dx[0]);
            int ix = incx;
            for (i=1; i<n; i++)
            {
                dnext = std::fabs(dx[ix]);
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

    /* xerbla is an error handler for the lapack routines. It is called by an lapack routine if an
     * input parameter has an invalid value. A message is printed and execution stops.
     * Installers may consider modifying the STOP statement in order to call system-specific
     * exception-handling facilities.
     * Parameters: srname: The name of the routine which called xerbla.
     *             info: The position of the invalid parameter in the parameter list of the calling
     *                   routine.
     * Authors: Univ.of Tennessee
     *          Univ.of California Berkeley
     *          Univ.of Colorado Denver
     *          NAG Ltd.
     * Date: December 2016                                                                       */
    static void xerbla(char const* srname, int info)
    {
        std::cout << " ** On entry to " << srname << " parameter number " <<  info
                  << " had an illegal value";
        throw info;
    }
};

#endif