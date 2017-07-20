#ifndef MONTBLANC_ARB_FUNC_OP_H
#define MONTBLANC_ARB_FUNC_OP_H

// arbfunc namespace start and stop defines
#define ARBFUNC_NAMESPACE_BEGIN namespace arbfunc {
#define ARBFUNC_NAMESPACE_STOP }

//  namespace start and stop defines
#define ARBFUNC_ARB_FUNC_NAMESPACE_BEGIN namespace  {
#define ARBFUNC_ARB_FUNC_NAMESPACE_STOP }

ARBFUNC_NAMESPACE_BEGIN
ARBFUNC_ARB_FUNC_NAMESPACE_BEGIN

// General definition of the ArbFunc op, which will be specialised in:
//   - arb_func_op_cpu.h for CPUs
//   - arb_func_op_gpu.cuh for CUDA devices
// Concrete template instantions of this class are provided in:
//   - arb_func_op_cpu.cpp for CPUs
//   - arb_func_op_gpu.cu for CUDA devices
template <typename Device, typename FT>
class ArbFunc {};

ARBFUNC_ARB_FUNC_NAMESPACE_STOP
ARBFUNC_NAMESPACE_STOP

#endif // #ifndef MONTBLANC_ARB_FUNC_OP_H