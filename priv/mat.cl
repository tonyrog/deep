// -*- c -*-
// 
//     Matrix operations
//

// M = number of columns in A
__kernel void mul_f(const int M,
		    __global float *A, __global float *B, 
		    __global float *C)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k;
    int kp = 0;
    int im = i*M;
    float tmp = 0.0f;

    for (k = 0; k < M; k++) {
	tmp += A[im+k]*B[kp+j];
	kp += M;
    }
    C[im+j] = tmp;
}

// M = number of columns in A
__kernel void add_f(const int M,
		    __global float *A, __global float *B, 
		    __global float *C)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int ix = i*M + j;
    C[ix] = A[ix] + B[ix];
}

// M = number of columns in A
__kernel void sub_f(const int M,
		    __global float *A, __global float *B, 
		    __global float *C)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int ix = i*M + j;
    C[ix] = A[ix] - B[ix];
}

// M = number of columns in A
__kernel void negate_f(const int M,
		       __global float *A, __global float *B)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int ix = i*M + j;
    B[ix] = -A[ix];
}

// M = number of columns in A
__kernel void sigmoid_f(const int M,
			__global float *A, __global float *B)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int ix = i*M + j;

    B[ix] = 1.0f/(1.0f+exp(-A[ix]));
}

// M = number of columns in A
__kernel void sigmoid_prime_f(const int M,
			      __global float *A, __global float *B)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int ix = i*M + j;
    float z = 1.0f/(1.0f+exp(-A[ix]));
    B[ix] = z*(1-z);
}
