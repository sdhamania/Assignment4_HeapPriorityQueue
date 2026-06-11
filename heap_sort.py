"""Heap sort using divide-and-conquer heapify on a max-heap."""

from __future__ import annotations

from copy import deepcopy


def heapify(arr: list, n: int, i: int) -> None:
    """Restore max-heap property at index ``i`` for a heap of size ``n``.

    Divide: compare node with its children.
    Conquer: recursively heapify the affected subtree.
    Combine: swap if needed so the largest value sits at the root of the subtree.
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def build_max_heap(arr: list) -> None:
    """Build a max-heap in place. Overall build cost is O(n)."""
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)


def heap_sort(arr: list) -> None:
    """Sort ``arr`` in place using heap sort.

    Build a max-heap, then repeatedly extract the maximum to the end
    and restore the heap property. Each extract costs O(log n).
    """
    n = len(arr)
    build_max_heap(arr)
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        heapify(arr, end, 0)


def is_sorted(arr: list) -> bool:
    """Return True if ``arr`` is non-decreasing."""
    return all(arr[k] <= arr[k + 1] for k in range(len(arr) - 1))


def _run_case(label: str, original: list) -> None:
    data = deepcopy(original)
    print(f"\n--- {label} ---")
    print(f"before: {original}")
    heap_sort(data)
    print(f"after:  {data}")
    passed = is_sorted(data)
    print(f"sorted check: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise AssertionError(f"sort failed case: {label}")


def main() -> None:
    _run_case("empty", [])
    _run_case("single element", [42])
    _run_case("typical mixed", [3, 7, 1, 9, 4])
    _run_case("duplicates", [5, 5, 1, 5, 2, 1])
    _run_case("already sorted", [1, 2, 3, 4, 5])
    _run_case("reverse sorted", [5, 4, 3, 2, 1])
    print("\nAll cases passed.")


if __name__ == "__main__":
    main()
