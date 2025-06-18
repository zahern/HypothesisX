''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''

import numpy as np

''' ---------------------------------------------------------- '''
''' LOCAL INITIALISATION                                       '''
''' ---------------------------------------------------------- '''

_gpu_available = False
try:
    import cupy
    _gpu_available = False
except ImportError:
    pass

''' ---------------------------------------------------------- '''
''' CLASS                                                      '''
''' ---------------------------------------------------------- '''
class Device():
# {
    ''' ------------------------------------------------------ '''
    ''' Function                                               '''
    ''' ------------------------------------------------------ '''
    # QUERY. WHAT DOES np STAND FOR?
    def __init__(self):
    # {
        if _gpu_available:
            self.np = cupy
            self._using_gpu = True
        else:
            self.np = np
            self._using_gpu = False
    # }

    ''' ------------------------------------------------------ '''
    ''' Function                                               '''
    ''' ------------------------------------------------------ '''
    def enable_gpu_acceleration(self, device_id=0):
    # {
        if _gpu_available == False: raise Exception("CuPy not found. Verify installation")
        self.np = cupy
        self._using_gpu = True
        cupy.cuda.Device(device_id).use()
    # }

    ''' ------------------------------------------------------ '''
    ''' Function                                               '''
    ''' ------------------------------------------------------ '''
    def disable_gpu_acceleration(self):
    # {
        self.np = np
        self._using_gpu = False
    # }

    ''' ------------------------------------------------------ '''
    ''' Function.                                              '''
    ''' ------------------------------------------------------ '''
    def multiply_1(self, a, b):
    # {
        n, p, j, k = a.shape                # Extract the cardinality of each of the 4 dimensions of a
        r = b.shape[-1]                     # Cardinality of last dimension of b

        # return self.np.matmul(a.reshape(n, p*j, k), b).reshape(n, p, j, r)
        # Equivalent to:
        a = a.reshape(n, p * j, k)          # Collapse a into 3 dimensions
        a_b = np.matmul(a, b)               # Compute a * b
        a_b = a_b.reshape(n, p, j, r)       # Expand to 4 dimensions
        return a_b

    # }

    def multiply_2(self, a, b):
        return np.matmul(a, b)

    def multiply_3(self, a, b):
    # {
        n, p, j, r = a.shape                        # Extract the cardinality of each of the 4 dimensions
        k = b.shape[-1]                             # Cardinality of last dimension of b

        return self.np.matmul(b.reshape(n, p * j, k).transpose([0, 2, 1]), a.reshape(n, p * j, r))
        b_ = b.reshape(n, p * j, k)                     # Collapse into 3 dimensions
        a_ = a.reshape(n, p * j, r)                     # Collapse into 3 dimensions
        bT_a = np.matmul(b_.transpose([0, 2, 1]), a_)   # Compute b^T * a
        return bT_a
    # }

    ''' ------------------------------------------------------ '''
    ''' Function. Efficient einsum for common expressions      '''
    ''' ------------------------------------------------------ '''
    def cust_einsum(self, expr, a, b):
    # {
        #if True:    # QUERY: IS THIS NEEDED?

        if expr == 'npjk,nkr -> npjr':
            return self.multiply_1(a, b)
        elif expr == 'npjk,k -> npj':
            return self.multiply_2(a, b)
        elif expr == 'npjr,npjk -> nkr':
            return self.multiply_3(a,b)
        else: raise Exception(f"The expression {expr} is not supported by custeinsum")

        #QUERY. WHEN IS THIS CALLED? CODE IS UNREACHABLE
        #return self.np.einsum(expr, a, b)
    # }

    ''' ------------------------------------------------------ '''
    ''' Function                                               '''
    ''' ------------------------------------------------------ '''
    @property
    def using_gpu(self): return self._using_gpu


    def to_cpu(self, arr):
        return cupy.asnumpy(arr)


    def to_gpu(self, arr):
        return cupy.asarray(arr)

    ''' ------------------------------------------------------ '''
    ''' Function                                               '''
    ''' ------------------------------------------------------ '''
    def convert_array_cpu(self, arr): return cupy.asnumpy(arr)

    ''' ------------------------------------------------------ '''
    ''' Function                                               '''
    ''' ------------------------------------------------------ '''
    def convert_array_gpu(self, arr): return cupy.asarray(arr)

    ''' ------------------------------------------------------ '''
    ''' Function                                               '''
    ''' ------------------------------------------------------ '''
    def get_device_count(self):
    # {
        if _gpu_available: return cupy.cuda.runtime.getDeviceCount()
        else: return 0
    # }
# }

''' ---------------------------------------------------------- '''
''' GLOBAL OBJECT                                              '''
''' ---------------------------------------------------------- '''
device = Device()  # Create an object to use
