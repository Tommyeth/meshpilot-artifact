"""Atomic resource accounting within a controller process."""
from __future__ import annotations

import threading
from dataclasses import dataclass

from .models import CandidatePlan, ClusterSnapshot


class PhysicalExecutionError(RuntimeError):
    pass


@dataclass(slots=True)
class _Reservation:
    node_id: str
    cpu: float
    ram: float
    gpu: float
    vram: float
    bandwidth: float


class ReservationLedger:
    """Atomic controller-side reservation accounting for one campaign process."""

    def __init__(self, snapshot: ClusterSnapshot) -> None:
        self._lock = threading.Lock()
        self._capacity = {
            node.node_id: {
                "cpu": node.cpu_available,
                "ram": node.ram_available_gib,
                "gpu": node.gpu_available,
                "vram": node.vram_available_gib,
                "bandwidth": snapshot.link("client-0", node.node_id).reservable_mbps
                if snapshot.link("client-0", node.node_id)
                else 0.0,
            }
            for node in snapshot.nodes
        }
        self._used = {node_id: {key: 0.0 for key in values} for node_id, values in self._capacity.items()}

    def reserve(self, plan: CandidatePlan) -> _Reservation:
        requested = {
            "cpu": plan.allocated_cpu_cores,
            "ram": plan.allocated_ram_gib,
            "gpu": plan.allocated_gpu_units,
            "vram": plan.allocated_vram_gib,
            "bandwidth": plan.reserved_bandwidth_mbps,
        }
        with self._lock:
            if plan.node_id not in self._capacity:
                raise PhysicalExecutionError("reservation node is unknown")
            for key, amount in requested.items():
                if self._used[plan.node_id][key] + amount > self._capacity[plan.node_id][key] + 1e-9:
                    raise PhysicalExecutionError(f"reservation exceeds available {key}")
            for key, amount in requested.items():
                self._used[plan.node_id][key] += amount
        return _Reservation(plan.node_id, *requested.values())

    def release(self, reservation: _Reservation) -> None:
        amounts = {
            "cpu": reservation.cpu,
            "ram": reservation.ram,
            "gpu": reservation.gpu,
            "vram": reservation.vram,
            "bandwidth": reservation.bandwidth,
        }
        with self._lock:
            for key, amount in amounts.items():
                self._used[reservation.node_id][key] = max(0.0, self._used[reservation.node_id][key] - amount)
