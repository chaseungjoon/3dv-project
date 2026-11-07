"""Backend-independent progress arithmetic (NumPy tests / Torch runtime)."""
def normalized_progress(before_distance, after_distance, length, limit):
    if length < 0.05 or limit <= 0:
        raise ValueError('Invalid normalization length or clipping limit')
    return ((before_distance - after_distance) / length).clip(-limit, limit)
