import numpy as np
import binascii

# The MNIST dataset is publicly available at https://yann.lecun.com/exdb/mnist/ 
# and consists of the following four parts: 
# - Training set images: train-images-idx3-ubyte.gz (9.9 MB, 47 MB unzipped, and 60,000 samples) 
# - Training set labels: train-labels-idx1-ubyte.gz (29 KB, 60 KB unzipped, and 60,000 labels) # - Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 7.8 MB, unzipped and 10,000 samples) 
# - Test set labels: t10k-labels-idx1-ubyte.gz (5 KB, 10 KB unzipped, and 10,000 labels)
def download():
    print("TODO")

def load_images(train=True):
    if train:
        filename = "data/MNIST/raw/train-images-idx3-ubyte"
    else:
        filename = "data/MNIST/raw/t10k-images-idx3-ubyte"

    with open(filename, "rb") as f: 
        # IDX magic number is 4 bytes
        # 3 dimensions size is 3 * 4 bytes = 12 bytes
        # so offset is 16 bytes
        data = np.frombuffer(f.read(), np.uint8, offset=16).reshape((60000, 28, 28))
        return data

def load_labels(train=True):
    if train:
        filename = "data/MNIST/raw/train-labels-idx1-ubyte"
    else:
        filename = "data/MNIST/raw/t10k-labels-idx1-ubyte"

    with open(filename, "rb") as f: 
        # IDX magic number is 4 bytes
        # 1 dimension size is 4 bytes
        # so offset is 8 bytes
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        print(data[0:10])
        return data

def print_binary(binary_data):
    # original print
    print(binary_data)

    # print ascii string
    print(binascii.b2a_hex(binary_data))

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
def parse_idx_file(filename):
    with open(filename, 'rb') as f:
        print(f.read(2))

        typ = f.read(1)
        match typ:
            case b'\x08':
                print('unsigned byte')
            case b'\x09':
                print('signed byte')
            case b'\x0B':
                print('short')
            case b'\x0C':
                print('int')
            case b'\x0D':
                print('float')
            case b'\x0E':
                print('double')
            case _:
                print('unknown')

        dims_num = f.read(1)
        print(len(dims_num))
        print(ord(dims_num))

        for _ in range(ord(dims_num)):
            dim_n = f.read(4)
            size_n = int.from_bytes(dim_n, 'big')
            print(size_n)
            print_binary(dim_n)

       
