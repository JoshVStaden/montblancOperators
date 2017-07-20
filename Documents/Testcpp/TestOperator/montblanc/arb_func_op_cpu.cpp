#include "arb_func_op_cpu.h"

#include "tensorflow/core/framework/shape_inference.h"

ARBFUNC_NAMESPACE_BEGIN
ARBFUNC_ARB_FUNC_NAMESPACE_BEGIN

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::Status;

auto shape_function = [](InferenceContext* c) {
    // Dummies for tests
    ShapeHandle input;
    DimensionHandle d;

    // Get input shapes
    ShapeHandle in_uvw = c->input(0);
    ShapeHandle in_antenna1 = c->input(1);
    ShapeHandle in_antenna2 = c->input(2);
    ShapeHandle in_frequency = c->input(3);
    ShapeHandle in_func_params = c->input(4);
    // Assert 'uvw' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_uvw, 3, &input),
        "uvw must have shape [None, None, 3] but is " +
        c->DebugString(in_uvw));
    // Assert 'uvw' dimension '2' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_uvw, 2), 3, &d),
        "uvw must have shape [None, None, 3] but is " +
        c->DebugString(in_uvw));
    

    // Assert 'antenna1' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_antenna1, 1, &input),
        "antenna1 must have shape [0] but is " +
        c->DebugString(in_antenna1));
    // Assert 'antenna1' dimension '0' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_antenna1, 0), 0, &d),
        "antenna1 must have shape [0] but is " +
        c->DebugString(in_antenna1));
    

    // Assert 'antenna2' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_antenna2, 1, &input),
        "antenna2 must have shape [0] but is " +
        c->DebugString(in_antenna2));
    // Assert 'antenna2' dimension '0' size
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithValue(c->Dim(in_antenna2, 0), 0, &d),
        "antenna2 must have shape [0] but is " +
        c->DebugString(in_antenna2));
    
 
    // Assert 'frequency' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_frequency, 1, &input),
        "frequency must have shape [None] but is " +
        c->DebugString(in_frequency));
    
    // Assert 'func_params' number of dimensions
    TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(in_func_params, 1, &input),
        "func_params must have shape [None] but is " +
        c->DebugString(in_func_params));
    
    

    // TODO: Supply a proper shapes for output variables here,
    // usually derived from input shapes
    // ShapeHandle output_1 = c->MakeShape({
    //      c->Dim(input_1, 0),  // input_1 dimension 0
    //      c->Dim(input_2, 1)}); // input_2 dimension 1""")

    ShapeHandle out_out_shape = c->MakeShape({ 1 });
    
    c->set_output(0, out_out_shape);
    

    // printf("output shape %s\\n", c->DebugString(out).c_str());;

    return Status::OK();
};

// Register the ArbFunc operator.
REGISTER_OP("ArbFunc")
    .Input("uvw: FT")
    .Input("antenna1: int32")
    .Input("antenna2: int32")
    .Input("frequency: FT")
    .Input("func_params: FT")
    .Output("out_shape: FT")
    .Attr("FT: {float, double} = DT_FLOAT")
    .Doc(R"doc(Predicts from arbitrary functions by computing the Fourier transform of a Gausian process analytically.
)doc")
    .SetShapeFn(shape_function);


// Register a CPU kernel for ArbFunc
// handling permutation ['float']
REGISTER_KERNEL_BUILDER(
    Name("ArbFunc")
    .TypeConstraint<float>("FT")
    .Device(tensorflow::DEVICE_CPU),
    ArbFunc<CPUDevice, float>);

// Register a CPU kernel for ArbFunc
// handling permutation ['double']
REGISTER_KERNEL_BUILDER(
    Name("ArbFunc")
    .TypeConstraint<double>("FT")
    .Device(tensorflow::DEVICE_CPU),
    ArbFunc<CPUDevice, double>);



ARBFUNC_ARB_FUNC_NAMESPACE_STOP
ARBFUNC_NAMESPACE_STOP