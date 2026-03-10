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

def MGCP_Encode_Binary_p2(u, l, c1, codebook):
    # Original calculations for block sizes and Reed-Solomon encoding
    k = len(u)
    c2 = 1
    K = int(np.ceil(k / l))
    N = K + c1 + c2
    q = 2**l

    U = binary_to_decimal_blocks(u, l)
    rsEncoder = RSCodec((c1 + c2), c_exp=l)
    X = list(rsEncoder.encode(U))

    # Convert RS-encoded blocks back to binary
    x_blocks = decimal_to_binary_blocks(X, l)

    # Flatten the binary blocks into a single binary array
    x = [int(bit) for block in x_blocks for bit in block]

    # Extract check parity and modify x
    check_par = x[-c2*l:]
    x = x[:-c2*l]

    # Add marker after each two blocks
    msg_withMarker = []
    marker = [0, 0, 1]
    block_size = l

    for i in range(K + c1):
        block_start = i * block_size
        block_end = block_start + block_size
        block = x[block_start:block_end]
        msg_withMarker += list(block)

        # Add a marker after every two blocks
        if (i + 1) % 2 == 0:
            msg_withMarker += marker

    # Append check parity
    decimal_index = int("".join(str(b) for b in check_par), 2)

    # Make sure codebook is 0-indexed in Python
    if decimal_index >= len(codebook):
        raise ValueError(f"Decimal index {decimal_index} exceeds codebook size {len(codebook)}")
    
    parity = codebook[decimal_index]
    msg_withMarker += parity

    # Final DNA sequence output
    x = msg_withMarker
    n = len(x)

    return x, n, N, K, q, U, X
