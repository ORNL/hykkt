#pragma once

#include <cublas_v2.h>
#include "constants.hpp"

/*
 * @brief adds two vectors and overrides the value of one vector with the sum
 *
 * @param handle_cublas - handle to the cuBLAS library context
 * n - size of vectors x,y
 * x - vector being added to y
 * y - vector being overriden
 * alpha - scalar for vector x
 *
 * @pre vectors x and y are the same size
 * @post the new value of y is equal to x*alpha + y
*/
void sumVectors(cublasHandle_t& handle_cublas,
                int n,
                const double* x,
                double* y,
                const double* alpha = &ONE);

/*
 * @brief computes the dot product of two vectors
 *
 * @param handle_cublas - handle to the cuBLAS library context
 * n - size of vectors x,y
 * x, y - vectors used in dot product
 *
 * @pre vectors x and y are the same size
 * @post r is now equal to the dot product of x and y
*/
void dotProduct(cublasHandle_t& handle_cublas,
                int n,
                const double* x,
                const double* y,
                double* r);

/*
 * @brief scales a vector
 *
 * @param handle_cublas - handle to the cuBLAS library context
 * n - size of vector x
 * x - vector being scaled
 * alpha - scalar for vector x
 *
 * @pre
 * @post the new value of x is x scaled by alpha
*/
void scaleVector(cublasHandle_t& handle_cublas,
                 int n,
                 const double* alpha,
                 double* x);
