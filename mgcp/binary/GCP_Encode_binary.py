import numpy as np
from reedsolo import RSCodec

def binary_to_decimal_blocks(binary_message, block_length):
    # Ensure the binary message is divisible by block_length
    if len(binary_message) % block_length != 0:
        raise ValueError("The length of the binary message must be divisible by the block length.")
    
    # Group the binary message into blocks of block_length
    blocks = [binary_message[i:i + block_length] for i in range(0, len(binary_message), block_length)]
    
    # Convert each binary block to decimal
    decimal_blocks = [int("".join(map(str, block)), 2) for block in blocks]
    
    return decimal_blocks

def decimal_to_binary_blocks(decimal_list, block_length):
    # Convert each decimal number to binary with fixed block length
    return [f"{x:0{block_length}b}" for x in decimal_list]

def GCP_Encode_binary_brute(u, l, c1, codebook_binary):
    # Original calculations for block sizes and Reed-Solomon encoding
    k = len(u)
    c2 = 1
    K = int(np.ceil(k / l))
    N = K + c1 + c2
    q = 2**l

    U = binary_to_decimal_blocks(u, l)

    rsEncoder = RSCodec((c1 + c2), c_exp=l)

    X = list(rsEncoder.encode(U))

    x_blocks = decimal_to_binary_blocks(X, l)

    # Flatten the binary blocks into a single binary array
    x = [int(bit) for block in x_blocks for bit in block]

    if k % l != 0:
        x[(K-1)*l : K*l - (k % l)] = []

    # Extract the last l bits as check parity (drives the codebook index)
    check_par = x[-l*c2:]
    msg_idx = int("".join(str(b) for b in check_par), 2)
    
    # Make sure codebook is 0-indexed in Python
    if msg_idx >= len(codebook_binary):
        raise ValueError(f"Decimal index {msg_idx} exceeds codebook size {len(codebook_binary)}")
    
    # Remove those l bits from the tail and encode
    x = x[:-c2*l]
    x.extend(codebook_binary[msg_idx])
    n = len(x)

    return x, n, N, K, q, U, X

    
def GCP_Encode_binary_rep(u, l, c1, c2, t):
    # Original calculations for block sizes and Reed-Solomon encoding
    k = len(u)
    K = int(np.ceil(k / l))
    N = K + c1 + c2
    q = 2**l

    U = binary_to_decimal_blocks(u, l)

    rsEncoder = RSCodec((c1 + c2), c_exp=l)

    X = list(rsEncoder.encode(U))

    x_blocks = decimal_to_binary_blocks(X, l)

    # Flatten the binary blocks into a single binary array
    x = [int(bit) for block in x_blocks for bit in block]

    if k % l != 0:
        x[(K-1)*l : K*l - (k % l)] = []

    # Extract check parity and modify x
    check_par = x[-c2*l:]
    x = x[:-c2*l]
    x = np.append(x, np.repeat(check_par, t))
    n = len(x)

    return x, n, N, K, q, U, X, check_par

def GCP_Encode_binary_localized(u, l, c, w):
    # Original calculations for block sizes and Reed-Solomon encoding
    k = len(u)
    K = int(np.ceil(k / l))
    N = K + c
    q = 2**l

    U = binary_to_decimal_blocks(u, l)

    rsEncoder = RSCodec(c, c_exp=l)

    X = list(rsEncoder.encode(U))

    x_blocks = decimal_to_binary_blocks(X, l)

    # Flatten the binary blocks into a single binary array
    x = [int(bit) for block in x_blocks for bit in block]

    if k % l != 0:
        x[(K-1)*l : K*l - (k % l)] = []

    buffer = [1] * (w + 1) + [0] * (w + 1) + [1] * (w + 1)
    x = x[:k] + buffer + x[k:]
    n = len(x)
    x = np.array(x, dtype=int)

    return x, n, N, K, q
