
#if defined(CONFIG_USE_DOUBLE)

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double  real_t;
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
typedef double  real_t;
#else
#warning "double is not supported"
typedef float real_t;
#endif

#else
typedef float real_t;
#endif
