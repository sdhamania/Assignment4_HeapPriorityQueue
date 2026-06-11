"""Non-preemptive priority scheduler simulation using a max-heap priority queue."""

from __future__ import annotations

from dataclasses import dataclass

from priority_queue import MaxPriorityQueue, Task


@dataclass(frozen=True)
class PriorityChange:
    """Raise or lower a waiting task's priority at a simulation time."""

    time: int
    task_id: str
    new_priority: int


@dataclass
class ScheduleRecord:
    """Timing metrics for one completed task."""

    task_id: str
    arrival_time: int
    burst_time: int
    priority: int
    start_time: int
    completion_time: int
    waiting_time: int
    turnaround_time: int


@dataclass
class SimulationResult:
    """Full output from one scheduling run."""

    algorithm: str
    records: list[ScheduleRecord]
    gantt: list[tuple[str, int, int]]

    @property
    def avg_waiting_time(self) -> float:
        if not self.records:
            return 0.0
        return sum(record.waiting_time for record in self.records) / len(self.records)

    @property
    def avg_turnaround_time(self) -> float:
        if not self.records:
            return 0.0
        return sum(record.turnaround_time for record in self.records) / len(self.records)


def _enqueue_arrivals(
    ready_queue: MaxPriorityQueue,
    tasks: list[Task],
    next_index: int,
    current_time: int,
) -> int:
    """Insert every task that has arrived by ``current_time``."""
    while next_index < len(tasks) and tasks[next_index].arrival_time <= current_time:
        ready_queue.insert(tasks[next_index])
        next_index += 1
    return next_index


def _apply_priority_changes(
    ready_queue: MaxPriorityQueue,
    changes: list[PriorityChange],
    next_change: int,
    current_time: int,
) -> int:
    """Apply all priority updates scheduled for ``current_time``."""
    while next_change < len(changes) and changes[next_change].time == current_time:
        change = changes[next_change]
        try:
            current_priority = ready_queue.task_priority(change.task_id)
            if change.new_priority > current_priority:
                ready_queue.increase_key(change.task_id, change.new_priority)
            elif change.new_priority < current_priority:
                ready_queue.decrease_key(change.task_id, change.new_priority)
        except KeyError:
            # Task is running or not yet arrived; skip this update.
            pass
        next_change += 1
    return next_change


def run_npps(
    tasks: list[Task],
    priority_changes: list[PriorityChange] | None = None,
) -> SimulationResult:
    """Simulate non-preemptive priority scheduling with ``MaxPriorityQueue``."""
    ordered_tasks = sorted(tasks, key=lambda task: (task.arrival_time, task.task_id))
    ordered_changes = sorted(priority_changes or [], key=lambda change: (change.time, change.task_id))

    ready_queue = MaxPriorityQueue()
    next_task = 0
    next_change = 0
    current_time = 0
    records: list[ScheduleRecord] = []
    gantt: list[tuple[str, int, int]] = []

    while next_task < len(ordered_tasks) or not ready_queue.is_empty():
        next_task = _enqueue_arrivals(ready_queue, ordered_tasks, next_task, current_time)
        next_change = _apply_priority_changes(
            ready_queue, ordered_changes, next_change, current_time
        )

        if ready_queue.is_empty():
            current_time = ordered_tasks[next_task].arrival_time
            continue

        task = ready_queue.extract_max()
        start_time = current_time
        completion_time = start_time + task.burst_time

        records.append(
            ScheduleRecord(
                task_id=task.task_id,
                arrival_time=task.arrival_time,
                burst_time=task.burst_time,
                priority=task.priority,
                start_time=start_time,
                completion_time=completion_time,
                waiting_time=start_time - task.arrival_time,
                turnaround_time=completion_time - task.arrival_time,
            )
        )
        gantt.append((task.task_id, start_time, completion_time))

        current_time = completion_time
        next_task = _enqueue_arrivals(ready_queue, ordered_tasks, next_task, current_time)
        next_change = _apply_priority_changes(
            ready_queue, ordered_changes, next_change, current_time
        )

    return SimulationResult("NPPS", records, gantt)


def run_fcfs(tasks: list[Task]) -> SimulationResult:
    """Baseline first-come-first-served scheduler for comparison."""
    ordered_tasks = sorted(tasks, key=lambda task: (task.arrival_time, task.task_id))

    waiting: list[Task] = []
    next_task = 0
    current_time = 0
    records: list[ScheduleRecord] = []
    gantt: list[tuple[str, int, int]] = []

    while next_task < len(ordered_tasks) or waiting:
        while next_task < len(ordered_tasks) and ordered_tasks[next_task].arrival_time <= current_time:
            waiting.append(ordered_tasks[next_task])
            next_task += 1

        if not waiting:
            current_time = ordered_tasks[next_task].arrival_time
            continue

        waiting.sort(key=lambda task: (task.arrival_time, task.task_id))
        task = waiting.pop(0)
        start_time = current_time
        completion_time = start_time + task.burst_time

        records.append(
            ScheduleRecord(
                task_id=task.task_id,
                arrival_time=task.arrival_time,
                burst_time=task.burst_time,
                priority=task.priority,
                start_time=start_time,
                completion_time=completion_time,
                waiting_time=start_time - task.arrival_time,
                turnaround_time=completion_time - task.arrival_time,
            )
        )
        gantt.append((task.task_id, start_time, completion_time))
        current_time = completion_time

    return SimulationResult("FCFS", records, gantt)


def print_gantt(result: SimulationResult) -> None:
    """Print a simple Gantt chart for a scheduling run."""
    print(f"Gantt ({result.algorithm}):")
    for task_id, start, end in result.gantt:
        print(f"  {task_id}: [{start}, {end})")


def print_metrics(result: SimulationResult) -> None:
    """Print per-task and average timing metrics."""
    print(f"Metrics ({result.algorithm}):")
    for record in result.records:
        print(
            f"  {record.task_id}: wait={record.waiting_time}, "
            f"turnaround={record.turnaround_time}, priority={record.priority}"
        )
    print(f"  avg waiting time: {result.avg_waiting_time:.2f}")
    print(f"  avg turnaround time: {result.avg_turnaround_time:.2f}")


def _run_case(label: str, fn) -> None:
    print(f"\n--- {label} ---")
    fn()
    print("PASS")


def main() -> None:
    def mixed_arrivals() -> None:
        tasks = [
            Task("P1", priority=3, arrival_time=0, burst_time=4),
            Task("P2", priority=1, arrival_time=1, burst_time=2),
            Task("P3", priority=4, arrival_time=2, burst_time=1),
            Task("P4", priority=2, arrival_time=3, burst_time=3),
        ]
        result = run_npps(tasks)
        order = [record.task_id for record in result.records]
        assert order == ["P1", "P3", "P4", "P2"]
        print_gantt(result)
        print_metrics(result)

    def tie_breaking() -> None:
        tasks = [
            Task("A", priority=5, arrival_time=0, burst_time=2),
            Task("B", priority=5, arrival_time=2, burst_time=2),
            Task("C", priority=5, arrival_time=1, burst_time=2),
        ]
        result = run_npps(tasks)
        order = [record.task_id for record in result.records]
        assert order == ["A", "C", "B"]

    def dynamic_priority_change() -> None:
        tasks = [
            Task("T1", priority=5, arrival_time=0, burst_time=3),
            Task("T2", priority=8, arrival_time=0, burst_time=2),
            Task("T3", priority=4, arrival_time=0, burst_time=2),
        ]
        changes = [PriorityChange(time=0, task_id="T3", new_priority=9)]
        result = run_npps(tasks, changes)
        order = [record.task_id for record in result.records]
        assert order == ["T3", "T2", "T1"]
        print_gantt(result)

    def npps_vs_fcfs() -> None:
        tasks = [
            Task("J1", priority=3, arrival_time=0, burst_time=5),
            Task("J2", priority=1, arrival_time=1, burst_time=3),
            Task("J3", priority=5, arrival_time=2, burst_time=2),
        ]
        npps = run_npps(tasks)
        fcfs = run_fcfs(tasks)

        # When several tasks are ready, NPPS dispatches higher priority first.
        assert npps.avg_waiting_time < fcfs.avg_waiting_time
        assert [record.task_id for record in npps.records] == ["J1", "J3", "J2"]
        assert [record.task_id for record in fcfs.records] == ["J1", "J2", "J3"]
        print_metrics(npps)
        print_metrics(fcfs)

    _run_case("mixed arrivals", mixed_arrivals)
    _run_case("tie-breaking by arrival time", tie_breaking)
    _run_case("dynamic priority change", dynamic_priority_change)
    _run_case("NPPS vs FCFS comparison", npps_vs_fcfs)
    print("\nAll cases passed.")


if __name__ == "__main__":
    main()
