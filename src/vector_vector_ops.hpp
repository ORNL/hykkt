#pragma once

#include <cublas.h>
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
 * r - a pointer on the HOST
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
 * @brief computes the dot product of two vectors -> CURRENTLY ONLY OPTIMAL FOR VECTORS OF SIZE <= 1,000,000
 * CRUCIAL: before using the result r in your prograsm, CALL cudaDeviceSynchronize()
 *
 * @param handle_cublas - handle to the cuBLAS library context
 * n - size of vectors x,y
 * x, y - vectors used in dot product
 * r - a pointer on the DEVICE
 *
 * @pre vectors x and y are the same size
 * @post r is now equal to the dot product of x and y
*/
void deviceDotProduct(int n,
                const double* x,
                const double* y,
                double* r);

void deviceDotProduct(int n,
                const double* x,
                const double* y,
                double* r,
                cudaStream_t stream);

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

void fun_divide(const double* x, const double* y, double* z);
void fun_mem_copy(const double* x, double* y);