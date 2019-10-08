def slice_data(total_size, batch_size, batch_threshold):
    """Generator to create slices containing batch_size elements, from 0 to n.

    The last slice may contain less than batch_size elements, when batch_size does not divide n.

    Arguments
    total_size -- Number of element in total, int
    batch_size -- Number of element in each batch, int
    batch_threshold -- Threshold to trigger slice

    Yields
    slice of batch_size elements
    """
    if total_size > batch_threshold:  # trigger slice
        start = 0
        for _ in range(int(total_size // batch_size)):
            end = start + batch_size
            yield slice(start, end)
            start = end
        if start < total_size:
            yield slice(start, total_size)
    else:  # full content
        yield slice(0, total_size)
