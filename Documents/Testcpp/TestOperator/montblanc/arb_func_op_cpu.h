#ifndef MONTBLANC_ARB_FUNC_OP_CPU_H
#define MONTBLANC_ARB_FUNC_OP_CPU_H

#include "arb_func_op.h"

// Required in order for Eigen::ThreadPoolDevice to be an actual type
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

ARBFUNC_NAMESPACE_BEGIN
ARBFUNC_ARB_FUNC_NAMESPACE_BEGIN

// For simpler partial specialisation
typedef Eigen::ThreadPoolDevice CPUDevice;

// Specialise the ArbFunc op for CPUs
template <typename FT>
class ArbFunc<CPUDevice, FT> : public tensorflow::OpKernel
{
public:
    explicit ArbFunc(tensorflow::OpKernelConstruction * context) :
        tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext * context) override
    {
        namespace tf = tensorflow;

        // Create reference to input Tensorflow tensors
        const auto & in_uvw = context->input(0);
        const auto & in_antenna1 = context->input(1);
        const auto & in_antenna2 = context->input(2);
        const auto & in_frequency = context->input(3);
        const auto & in_func_params = context->input(4);
        

        // Extract Eigen tensors
        auto uvw = in_uvw.tensor<FT, 3>();
        auto antenna1 = in_antenna1.tensor<tensorflow::int32, 1>();
        auto antenna2 = in_antenna2.tensor<tensorflow::int32, 1>();
        auto frequency = in_frequency.tensor<FT, 1>();
        auto func_params = in_func_params.tensor<FT, 1>();
        

        // Allocate output tensors
        // Allocate space for output tensor 'out_shape'
        tf::Tensor * out_shape_ptr = nullptr;
        tf::TensorShape out_shape_shape = tf::TensorShape({ 1 });
        OP_REQUIRES_OK(context, context->allocate_output(
            0, out_shape_shape, &out_shape_ptr));
        
    }
};

ARBFUNC_ARB_FUNC_NAMESPACE_STOP
ARBFUNC_NAMESPACE_STOP

#endif // #ifndef MONTBLANC_ARB_FUNC_OP_CPU_H