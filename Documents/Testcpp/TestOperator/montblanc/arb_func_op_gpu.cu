#if GOOGLE_CUDA

#include "arb_func_op_gpu.cuh"

ARBFUNC_NAMESPACE_BEGIN
ARBFUNC_ARB_FUNC_NAMESPACE_BEGIN


// Register a GPU kernel for ArbFunc
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("ArbFunc")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_GPU),
    ArbFunc<GPUDevice, float>);

// Register a GPU kernel for ArbFunc
// handling permutation ['double']
REGISTER_KERNEL_BUILDER(
    Name("ArbFunc")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_GPU),
    ArbFunc<GPUDevice, double>);



ARBFUNC_ARB_FUNC_NAMESPACE_STOP
ARBFUNC_NAMESPACE_STOP

#endif // #if GOOGLE_CUDA