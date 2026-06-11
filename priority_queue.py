"""Binary-heap priority queues for task scheduling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    """A schedulable unit keyed by ``task_id`` with integer priority."""

    task_id: str
    priority: int
    arrival_time: int = 0
    burst_time: int = 0


class _HeapPriorityQueueBase:
    """Shared array-heap mechanics for max- and min-heap variants."""

    def __init__(self) -> None:
        self._heap: list[Task] = []

    def is_empty(self) -> bool:
        """Return True when the queue contains no tasks. O(1)."""
        return len(self._heap) == 0

    def insert(self, task: Task) -> None:
        """Add ``task`` and restore the heap property. O(log n)."""
        self._heap.append(task)
        self._sift_up(len(self._heap) - 1)

    def _parent(self, index: int) -> int:
        return (index - 1) // 2

    def _left(self, index: int) -> int:
        return 2 * index + 1

    def _right(self, index: int) -> int:
        return 2 * index + 2

    def _swap(self, i: int, j: int) -> None:
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]

    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = self._parent(index)
            if not self._priority_above(self._heap[index], self._heap[parent]):
                break
            self._swap(index, parent)
            index = parent

    def _sift_down(self, index: int) -> None:
        size = len(self._heap)
        while True:
            target = index
            left = self._left(index)
            right = self._right(index)

            if left < size and self._priority_above(self._heap[left], self._heap[target]):
                target = left
            if right < size and self._priority_above(self._heap[right], self._heap[target]):
                target = right
            if target == index:
                break
            self._swap(index, target)
            index = target

    def _priority_above(self, task_a: Task, task_b: Task) -> bool:
        """Return True when ``task_a`` should sit closer to the root than ``task_b``."""
        raise NotImplementedError


class MaxPriorityQueue(_HeapPriorityQueueBase):
    """Max-heap priority queue: higher ``priority`` values leave first."""

    def peek_max(self) -> Task:
        """Return the highest-priority task without removing it. O(1)."""
        if self.is_empty():
            raise IndexError("peek_max from empty priority queue")
        return self._heap[0]

    def extract_max(self) -> Task:
        """Remove and return the highest-priority task. O(log n)."""
        if self.is_empty():
            raise IndexError("extract_max from empty priority queue")

        root = self._heap[0]
        last = self._heap.pop()
        if self._heap:
            self._heap[0] = last
            self._sift_down(0)
        return root

    def _priority_above(self, task_a: Task, task_b: Task) -> bool:
        return task_a.priority > task_b.priority


class MinPriorityQueue(_HeapPriorityQueueBase):
    """Min-heap priority queue: lower ``priority`` values leave first."""

    def peek_min(self) -> Task:
        """Return the lowest-priority task without removing it. O(1)."""
        if self.is_empty():
            raise IndexError("peek_min from empty priority queue")
        return self._heap[0]

    def extract_min(self) -> Task:
        """Remove and return the lowest-priority task. O(log n)."""
        if self.is_empty():
            raise IndexError("extract_min from empty priority queue")

        root = self._heap[0]
        last = self._heap.pop()
        if self._heap:
            self._heap[0] = last
            self._sift_down(0)
        return root

    def _priority_above(self, task_a: Task, task_b: Task) -> bool:
        return task_a.priority < task_b.priority


def _run_case(label: str, fn) -> None:
    print(f"\n--- {label} ---")
    fn()
    print("PASS")


def main() -> None:
    def max_queue_extract_order() -> None:
        pq = MaxPriorityQueue()
        for task_id, priority in [("a", 3), ("b", 9), ("c", 5), ("d", 1)]:
            pq.insert(Task(task_id, priority))

        extracted = [pq.extract_max().priority for _ in range(4)]
        assert extracted == [9, 5, 3, 1]
        assert pq.is_empty()

    def min_queue_extract_order() -> None:
        pq = MinPriorityQueue()
        for task_id, priority in [("a", 3), ("b", 9), ("c", 5), ("d", 1)]:
            pq.insert(Task(task_id, priority))

        extracted = [pq.extract_min().priority for _ in range(4)]
        assert extracted == [1, 3, 5, 9]
        assert pq.is_empty()

    def empty_queue_raises() -> None:
        max_pq = MaxPriorityQueue()
        min_pq = MinPriorityQueue()
        assert max_pq.is_empty()
        assert min_pq.is_empty()

        for queue, op in (
            (max_pq, "extract_max"),
            (max_pq, "peek_max"),
            (min_pq, "extract_min"),
            (min_pq, "peek_min"),
        ):
            try:
                if op == "extract_max":
                    queue.extract_max()
                elif op == "peek_max":
                    queue.peek_max()
                elif op == "extract_min":
                    queue.extract_min()
                else:
                    queue.peek_min()
            except IndexError:
                continue
            raise AssertionError(f"expected IndexError for {op} on empty queue")

    def peek_does_not_remove() -> None:
        pq = MaxPriorityQueue()
        pq.insert(Task("job", 7))
        assert pq.peek_max().task_id == "job"
        assert pq.extract_max().task_id == "job"
        assert pq.is_empty()

    def duplicate_priorities() -> None:
        pq = MaxPriorityQueue()
        pq.insert(Task("x", 5))
        pq.insert(Task("y", 5))
        pq.insert(Task("z", 5))

        priorities = {pq.extract_max().priority for _ in range(3)}
        assert priorities == {5}
        assert pq.is_empty()

    def interleaved_insert_extract() -> None:
        pq = MaxPriorityQueue()
        pq.insert(Task("low", 1))
        pq.insert(Task("high", 10))
        assert pq.extract_max().task_id == "high"
        pq.insert(Task("mid", 6))
        assert pq.extract_max().task_id == "mid"
        assert pq.extract_max().task_id == "low"
        assert pq.is_empty()

    _run_case("max queue extract order", max_queue_extract_order)
    _run_case("min queue extract order", min_queue_extract_order)
    _run_case("empty queue raises IndexError", empty_queue_raises)
    _run_case("peek does not remove task", peek_does_not_remove)
    _run_case("duplicate priorities", duplicate_priorities)
    _run_case("interleaved insert and extract", interleaved_insert_extract)
    print("\nAll cases passed.")


if __name__ == "__main__":
    main()
