#if GOOGLE_CUDA

#ifndef MONTBLANC_ARB_FUNC_OP_GPU_CUH
#define MONTBLANC_ARB_FUNC_OP_GPU_CUH

#include "arb_func_op.h"

// Required in order for Eigen::GpuDevice to be an actual type
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

ARBFUNC_NAMESPACE_BEGIN
ARBFUNC_ARB_FUNC_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::GpuDevice GPUDevice;

// LaunchTraits struct defining
// kernel block sizes for type permutations
template <typename FT> struct LaunchTraits {};

// Specialise for float
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<float>
{
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;
};
// Specialise for double
// Should really be .cu file as this is a concrete type
// but this works because this header is included only once
template <> struct LaunchTraits<double>
{
    static constexpr int BLOCKDIMX = 1024;
    static constexpr int BLOCKDIMY = 1;
    static constexpr int BLOCKDIMZ = 1;
};


// CUDA kernel outline
template <typename FT> 
__global__ void montblanc_arb_func(
    const FT * in_uvw,
    const tensorflow::int32 * in_antenna1,
    const tensorflow::int32 * in_antenna2,
    const FT * in_frequency,
    const FT * in_func_params,
    FT * out_out_shape)
    
{
    // Shared memory usage unnecesssary, but demonstrates use of
    // constant Trait members to create kernel shared memory.
    using LTr = LaunchTraits<FT>;
    __shared__ int buffer[LTr::BLOCKDIMX];

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= LTr::BLOCKDIMX)
        { return; }

    // Set shared buffer to thread index
    buffer[i] = i;
}

// Specialise the ArbFunc op for GPUs
template <typename FT>
class ArbFunc<GPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit ArbFunc(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create variables for input tensors
        const auto & in_uvw = context->input(0);
        const auto & in_antenna1 = context->input(1);
        const auto & in_antenna2 = context->input(2);
        const auto & in_frequency = context->input(3);
        const auto & in_func_params = context->input(4);
        

        // Allocate output tensors
        // Allocate space for output tensor 'out_shape'
        tf::Tensor * out_shape_ptr = nullptr;
        tf::TensorShape out_shape_shape = tf::TensorShape({ 1 });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, out_shape_shape, &out_shape_ptr));
        

        using LTr = LaunchTraits<FT>;

        // Set up our CUDA thread block and grid
        dim3 block(LTr::BLOCKDIMX);
        dim3 grid(1);

        // Get pointers to flattened tensor data buffers
        const auto fin_uvw = in_uvw.flat<FT>().data();
        const auto fin_antenna1 = in_antenna1.flat<tensorflow::int32>().data();
        const auto fin_antenna2 = in_antenna2.flat<tensorflow::int32>().data();
        const auto fin_frequency = in_frequency.flat<FT>().data();
        const auto fin_func_params = in_func_params.flat<FT>().data();
        auto fout_out_shape = out_shape_ptr->flat<FT>().data();
        

        // Get the GPU device
        const auto & device = context->eigen_device<GPUDevice>();

        // Call the montblanc_arb_func CUDA kernel
        montblanc_arb_func<FT>
            <<<grid, block, 0, device.stream()>>>(
                fin_uvw,
                fin_antenna1,
                fin_antenna2,
                fin_frequency,
                fin_func_params,
                fout_out_shape);
                
    }
};

ARBFUNC_ARB_FUNC_NAMESPACE_STOP
ARBFUNC_NAMESPACE_STOP

#endif // #ifndef MONTBLANC_ARB_FUNC_OP_GPU_CUH

#endif // #if GOOGLE_CUDA