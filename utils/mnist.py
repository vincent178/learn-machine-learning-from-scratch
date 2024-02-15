import numpy as np

# The MNIST dataset is publicly available at https://yann.lecun.com/exdb/mnist/ 
# and consists of the following four parts: 
# - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, and 60,000 samples) 
# - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, and 60,000 labels) # - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, unzipped and 10,000 samples) 
# - Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, and 10,000 labels)
def download():
    print("TODO")

def load_mnist():
    x_train = load_idx_file("data/MNIST/raw/train-images-idx3-ubyte")
    x_test = load_idx_file("data/MNIST/raw/t10k-images-idx3-ubyte")
    t_train = load_idx_file("data/MNIST/raw/train-labels-idx1-ubyte")
    t_test = load_idx_file("data/MNIST/raw/t10k-labels-idx1-ubyte")
    return (x_train, t_train), (x_test, t_test)

# The IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
# The basic format according to http://yann.lecun.com/exdb/mnist/ is:

#     magic number
#     size in dimension 1
#     size in dimension 2
#     size in dimension 3
#     ....
#     size in dimension N
#     data

# The magic number is four bytes long. The first 2 bytes are always 0.

# The third byte codes the type of the data:
#     0x08: unsigned byte
#     0x09: signed byte
#     0x0B: short (2 bytes)
#     0x0C: int (4 bytes)
#     0x0D: float (4 bytes)
#     0x0E: double (8 bytes)

# The fouth byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
# The sizes in each dimension are 4-byte integers (big endian, like in most non-Intel processors).
def load_idx_file(filename):
    with open(filename, 'rb') as f:
        f.read(2)

        typ = f.read(1)
        match typ:
            case b'\x08':
                dtype = np.uint8
            case b'\x09':
                dtype = np.int8
            case b'\x0B':
                dtype = np.int16
            case b'\x0C':
                dtype = np.int32
            case b'\x0D':
                dtype = np.float32
            case b'\x0E':
                dtype = np.float64
            case _:
                raise ValueError('unknown type of the data')

        dims_num = ord(f.read(1))

        shape = []
        for _ in range(dims_num):
            dim_size = int.from_bytes(f.read(4), 'big')
            shape.append(dim_size)

        return np.frombuffer(f.read(), dtype).reshape(shape)
