import numpy as np
from reedsolo import RSCodec

def binary_to_dna(binary_message):
    # Convert binary message to a DNA sequence
    dna_map = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}
    return "".join(dna_map["".join(map(str, binary_message[i:i+2]))] for i in range(0, len(binary_message), 2))

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

def MGCP_Encode_DNA_p2(u, l, c1, codebook):
    # Original calculations for block sizes and Reed-Solomon encoding
    k = len(u)
    c2 = 1
    K = int(np.ceil(k / l))
    N = K + c1 + c2
    q = 2**l

    U = binary_to_decimal_blocks(u, l)
    rsEncoder = RSCodec((c1 + c2), c_exp=l)
    X = list(rsEncoder.encode(U))

    # Step 4: Convert RS-encoded blocks back to binary
    x_blocks = decimal_to_binary_blocks(X, l)

    # Flatten the binary blocks into a single binary array (optional)
    x = [int(bit) for block in x_blocks for bit in block]

    # Step 6: Extract check parity and modify x
    check_par = x[-c2*l:]
    x = x[:-c2*l]

    # Step 7: Convert to DNA sequence
    dna_msg = binary_to_dna(x)

    # Step 8: Add DNA-based markers after each block
    msg_withMarker = ''
    marker = 'AC'  # DNA marker
    block_size = l // 2  # Each block in DNA corresponds to l/2 bases

    for i in range(K + c1):
        block_start = i * block_size
        block_end = block_start + block_size
        block = dna_msg[block_start:block_end]
        msg_withMarker += block

        # Add marker after every two blocks
        if (i + 1) % 2 == 0:
            msg_withMarker += marker

    # Append check parity as DNA
    decimal_index = int("".join(str(b) for b in check_par), 2)

    # Make sure codebook is 0-indexed in Python
    if decimal_index >= len(codebook):
        raise ValueError(f"Decimal index {decimal_index} exceeds codebook size {len(codebook)}")

    parity_dna = codebook[decimal_index]
    msg_withMarker += parity_dna

    # Final DNA sequence output
    x = msg_withMarker
    n = len(x)

    return x, n, N, K, q, U, X