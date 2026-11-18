"""Single-owner FIFO batcher on one caller-supplied monotonic host clock."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    id: str
    arrival_ns: int


@dataclass(frozen=True)
class Batch:
    sequence: int
    requests: tuple[Request, ...]
    dispatch_ns: int
    reason: str

    @property
    def ids(self):
        return tuple(request.id for request in self.requests)


class Batcher:
    """At most one in flight; call complete only after host output is ready."""

    def __init__(self, arm, *, wait_ns):
        if arm not in ('singleton', 'batch8') or type(wait_ns) is not int or wait_ns <= 0:
            raise ValueError('unknown arm or invalid positive wait_ns')
        self._capacity = 1 if arm == 'singleton' else 8
        self._wait_ns = wait_ns
        self._pending = deque()
        self._seen = set()
        self._inflight = None
        self._last_ns = 0
        self._sequence = 0

    def _tick(self, now_ns):
        if type(now_ns) is not int or now_ns < self._last_ns:
            raise ValueError('expected nondecreasing integer host timestamps')
        self._last_ns = now_ns

    def arrive(self, request_id, arrival_ns):
        if not isinstance(request_id, str) or not request_id or request_id in self._seen:
            raise ValueError('invalid or duplicate request ID')
        self._tick(arrival_ns)
        self._seen.add(request_id)
        self._pending.append(Request(request_id, arrival_ns))

    def ready_at_ns(self):
        """Earliest FIFO size/deadline trigger; None if idle or blocked by work."""
        if not self._pending or self._inflight is not None:
            return None
        deadline = self._pending[0].arrival_ns + self._wait_ns
        if len(self._pending) >= self._capacity:
            return min(self._pending[self._capacity - 1].arrival_ns, deadline)
        return deadline

    def dispatch(self, now_ns):
        self._tick(now_ns)
        ready = self.ready_at_ns()
        if ready is None or ready > now_ns:
            return None
        size = min(len(self._pending), self._capacity)
        reason = 'singleton' if self._capacity == 1 else 'full' if size == self._capacity else 'deadline'
        batch = Batch(self._sequence, tuple(self._pending.popleft() for _ in range(size)), now_ns, reason)
        self._sequence += 1
        self._inflight = batch
        return batch

    def _require_inflight(self, batch):
        if batch is None or batch is not self._inflight:
            raise ValueError('batch is not the current in-flight object')

    def complete(self, batch, completion_ns, outputs):
        """One immutable 8-byte result per request, in dispatch order."""
        self._require_inflight(batch)
        outputs = tuple(outputs)
        if len(outputs) != len(batch.requests) or any(type(data) is not bytes or len(data) != 8 for data in outputs):
            raise ValueError('completion must supply one exact 8-byte witness per request')
        self._tick(completion_ns)
        self._inflight = None
        return [{'id': request.id, 'arrival_ns': request.arrival_ns, 'delivery_ns': completion_ns,
                 'status': 'delivered', 'output_hex': data.hex()}
                for request, data in zip(batch.requests, outputs)]

    def fail(self, batch, observed_ns, error):
        """Record a failed batch without claiming delivery. The caller decides whether to abort."""
        self._require_inflight(batch)
        if not isinstance(error, str) or not error:
            raise ValueError('expected a backend error description')
        self._tick(observed_ns)
        self._inflight = None
        return [{'id': request.id, 'arrival_ns': request.arrival_ns, 'delivery_ns': None,
                 'status': 'backend-error', 'output_hex': None, 'observed_ns': observed_ns, 'error': error}
                for request in batch.requests]
