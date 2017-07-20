import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

class TestArbFunc(unittest.TestCase):
    """ Tests the ArbFunc operator """

    def setUp(self):
        # Load the custom operation library
        self.montblanc = tf.load_op_library('montblanc.so')
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_arb_func(self):
        """ Test the ArbFunc operator """
        # List of type constraint for testing this operator
        type_permutations = [
            [np.float32],
            [np.float64]]

        # Run test with the type combinations above
        for FT in type_permutations:
            self._impl_test_arb_func(FT)

    def _impl_test_arb_func(self, FT):
        """ Implementation of the ArbFunc operator test """

        # Create input variables
        uvw = np.random.random(size=[1, 1, 3]).astype(FT)
        antenna1 = np.random.random(size=[0]).astype(np.int32)
        antenna2 = np.random.random(size=[0]).astype(np.int32)
        frequency = np.random.random(size=[1]).astype(FT)
        func_params = np.random.random(size=[1]).astype(FT)
        

        # Argument list
        np_args = [uvw, antenna1, antenna2, frequency, func_params]
        # Argument string name list
        arg_names = ['uvw', 'antenna1', 'antenna2', 'frequency',
            'func_params']
        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.montblanc.arb_func(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)
            S.run(cpu_op)
            S.run(gpu_ops)

if __name__ == "__main__":
    unittest.main()