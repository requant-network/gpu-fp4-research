"""Declared persistent Faiss scratch sizing for the bounded search geometries."""


def faiss_reservation_bytes(population,batch):
    """Keep two FP32 distance tiles plus 16 MiB small-buffer/alignment margin.

    The previous fixed 64 MiB reservation overflowed into temporary device
    allocations at N=262144/B=128 and a captured search accessed freed scratch.
    Reserve the worst full-query tiles rather than relying on allocator reuse.
    The largest declared shape stays below Faiss's 1.5 GiB device reservation cap.
    """
    if not isinstance(population,int) or not isinstance(batch,int) or population<1 or batch<1:
        raise ValueError('Positive integer population and batch are required')
    return max(64<<20,2*population*batch*4+(16<<20))
