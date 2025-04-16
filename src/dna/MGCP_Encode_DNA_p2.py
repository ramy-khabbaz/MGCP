import numpy as np
import argparse
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

def MGCP_Encode_DNA_p2(u, l, c1, c2, t):
    # Original calculations for block sizes and Reed-Solomon encoding
    k = len(u)
    K = int(np.ceil(k / l))
    N = K + c1 + c2
    q = 2**l

    U = binary_to_decimal_blocks(u, l)

    rsEncoder = RSCodec((c1 + c2), c_exp=l)
    X = list(rsEncoder.encode(U))

    # Convert RS-encoded blocks back to binary
    x_blocks = decimal_to_binary_blocks(X, l)

    # Flatten the binary blocks into a single binary array (optional)
    x = [int(bit) for block in x_blocks for bit in block]

    #  Extract check parity and modify x
    check_par = x[-c2*l:]
    x = x[:-c2*l]
    x = np.append(x, np.repeat(check_par, t))

    # Convert to DNA sequence
    dna_msg = binary_to_dna(x)

    # Add DNA-based markers after each block
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

    # Append repeated check parity as DNA
    parity_dna = binary_to_dna(np.repeat(check_par, t))
    msg_withMarker += parity_dna

    # Final DNA sequence output
    x = msg_withMarker
    n = len(x)

    return x, n, N, K, q, U, X

def binary_message(value):
    """Convert a string to a binary message; raise an error if non-binary characters are present."""
    if any(char not in "01" for char in value):
        raise argparse.ArgumentTypeError("The message must contain only binary digits (0 and 1).")
    return value

def main():
    parser = argparse.ArgumentParser(description="MGCP Encode for marker period 2")
    parser.add_argument("-u", "--message", type=binary_message, required=True,
                        help="Binary message as a string (e.g., '10101010')")
    parser.add_argument("-l", "--block_length", type=int, required=True,
                        help="Block length (must divide the length of the message)")
    parser.add_argument("-c1", type=int, required=True, help="Value for c1 (Guess parity)")
    parser.add_argument("-c2", type=int, required=True, help="Value for c2 (Check parity)")
    parser.add_argument("-t", "--repeat", type=int, required=True,
                        help="Repeat count for check parity")
    args = parser.parse_args()

    if len(args.message) % 2 != 0:
        parser.error("The length of the message should be even.")

    if args.block_length % 2 != 0:
        parser.error("The length of the block should be even.")

    # Additional validations:
    if args.c1 % 2 != 0:
        parser.error("For marker period 2, c1 value must be even.")
    if len(args.message) % args.block_length != 0:
        parser.error("The length of the binary message must be divisible by the block length.")
    if (len(args.message) // args.block_length) % 2 != 0:
        parser.error("The number of blocks (length(message)/block_length) must be an even integer.")

    # Convert the binary message string to a list of integers.
    u = [int(bit) for bit in args.message]

    try:
        encoded, n, N, K, q, U, X = MGCP_Encode_DNA_p2(u, args.block_length, args.c1, args.c2, args.repeat)
        print("Encoded sequence:", encoded)
        print("n =", n, "N =", N, "K =", K, "q =", q)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()