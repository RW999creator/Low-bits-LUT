import numpy as np
from collections import defaultdict

# ================================================================
# Generate activation vector given bit precision and vector length
# ================================================================
def generate_activation_vector(n_elements=8, precision_bits=4):
    """
    Generate an integer activation vector based on bit precision.
    
    Example:
    precision_bits = 4  --> range = 0-15
    precision_bits = 2  --> range = 0-3
    """
    act_range = 2**precision_bits
    A = np.random.randint(0, act_range, size=n_elements)
    return A


# ================================================================
# Generate all possible binary (-1, +1) weight vectors of length N
# ================================================================
def generate_all_binary_weights(n_elements=8):
    """
    Generates all 2^N binary weight combinations.
    Each weight is -1 or +1.
    """
    for i in range(2**n_elements):
        bits = [(1 if (i >> b) & 1 else -1) for b in range(n_elements)]
        yield tuple(bits)


# ================================================================
# Evaluate all weight vectors and analyze reuse statistics
# ================================================================
def evaluate_mapping(A):
    """
    Given activation vector A, compute MAC results for all
    binary weight vectors and analyze reuse.
    """
    n = len(A)
    weight_patterns = list(generate_all_binary_weights(n))
    
    print(f"\n--- Running experiment: N={n} ---")
    print(f"Total weight patterns = {len(weight_patterns)}")
    
    mac_results = {}
    for w in weight_patterns:
        mac = np.dot(np.array(w), A)
        mac_results[w] = float(mac)
    
    # Group keys by identical result value
    value_to_keys = defaultdict(list)
    for k, v in mac_results.items():
        value_to_keys[v].append(k)
    
    num_unique_values = len(value_to_keys)
    reuse_ratio = len(weight_patterns) / num_unique_values
    
    print(f"Unique output values = {num_unique_values}")
    print(f"Reuse ratio = {reuse_ratio:.2f}x  (Avg. #keys per value)")
    
    # Sort values by reuse count (descending)
    reuse_stats = sorted(
        [(val, len(keys)) for val, keys in value_to_keys.items()],
        key=lambda x: x[1], reverse=True
    )
    
    print("\nTop 10 most reused output values:")
    for val, cnt in reuse_stats[:10]:
        print(f"Value = {val:7.3f}    reused by {cnt} weight patterns")

    return {
        "activation_vector": A,
        "total_patterns": len(weight_patterns),
        "unique_values": num_unique_values,
        "reuse_ratio": reuse_ratio,
        "reuse_stats": reuse_stats,
    }


# ================================================================
# Example Run
# ================================================================
if __name__ == "__main__":
    
    # Configurable parameters
    n_elements = 16
    precision_bits = 8
    
    A = generate_activation_vector(
        n_elements=n_elements,
        precision_bits=precision_bits
    )

    print(f"Activation Vector (N={n_elements}, {precision_bits}bit):")
    print(A)
    
    results = evaluate_mapping(A)