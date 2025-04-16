import argparse
import numpy as np

def prob_value(value):
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(f"{value} is not a valid probability. It must be between 0 and 1.")
    return fvalue

def binary_message(value):
    """Validate that the input string contains only binary digits."""
    if any(ch not in '01' for ch in value):
        raise argparse.ArgumentTypeError("Input message must contain only binary digits (0 and 1).")
    # Convert string into a list of integers (each bit)
    return [int(ch) for ch in value]

def binary_channel(x, Pd, Pi, Ps):
    errs = []
    y = []

    for i in range(len(x)):
        r = np.random.choice([0, 1, 2, 3], p=[1 - Pd - Pi - Ps, Pd, Pi, Ps])
        if r == 0:
            y.append(x[i])
        elif r == 1:  # Deletion
            errs.append([-1, i])
        elif r == 2:  # Insertion
            y.append(np.random.randint(0, 2))  # Random binary value
            y.append(x[i])
            errs.append([1, i])
        elif r == 3:  # Substitution
            y.append(1 - x[i])  # Flip the bit
            errs.append([0, i])

    return y

def main():
    parser = argparse.ArgumentParser(description="Simulate a binary channel with deletion, insertion, and substitution errors.")
    parser.add_argument("-x", "--input", type=binary_message, required=True,
                        help="Binary sequence as a string (e.g., '10101010')")
    parser.add_argument("--Pd", type=prob_value, required=True,
                        help="Probability of deletion (between 0 and 1)")
    parser.add_argument("--Pi", type=prob_value, required=True,
                        help="Probability of insertion (between 0 and 1)")
    parser.add_argument("--Ps", type=prob_value, required=True,
                        help="Probability of substitution (between 0 and 1)")
    args = parser.parse_args()

    # Ensure the sum of probabilities does not exceed 1
    if args.Pd + args.Pi + args.Ps > 1:
        parser.error("The sum of Pd, Pi, and Ps must not exceed 1.")

    x = args.input

    # Call the binary_channel function
    y = binary_channel(x, args.Pd, args.Pi, args.Ps)
    print("Output sequence:", y)

if __name__ == "__main__":
    main()
