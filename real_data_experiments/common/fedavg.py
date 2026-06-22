"""Standard sample-size weighted FedAvg aggregation."""

from __future__ import annotations

import copy

import torch


def fedavg_aggregate(local_state_dicts: list[dict[str, torch.Tensor]], sample_counts: list[int]) -> dict[str, torch.Tensor]:
    """
    Standard sample-size weighted FedAvg.

    Args:
        local_state_dicts: List of client model state_dict objects.
        sample_counts: List of local training sample counts.

    Returns:
        A newly allocated aggregated state_dict.
    """
    if len(local_state_dicts) != len(sample_counts):
        raise ValueError("len(local_state_dicts) must equal len(sample_counts).")
    if not local_state_dicts:
        raise ValueError("local_state_dicts must not be empty.")
    if any(count < 0 for count in sample_counts):
        raise ValueError("sample_counts must be non-negative.")

    total_samples = int(sum(sample_counts))
    if total_samples <= 0:
        raise ValueError("sum(sample_counts) must be positive.")

    reference_keys = list(local_state_dicts[0].keys())
    for state_dict in local_state_dicts[1:]:
        if list(state_dict.keys()) != reference_keys:
            raise ValueError("All client state_dict keys must match exactly.")

    aggregated_state = copy.deepcopy(local_state_dicts[0])
    for key in reference_keys:
        tensor = local_state_dicts[0][key]
        if torch.is_floating_point(tensor):
            accumulator = torch.zeros_like(tensor)
            for state_dict, sample_count in zip(local_state_dicts, sample_counts):
                weight = float(sample_count) / float(total_samples)
                accumulator = accumulator + state_dict[key].detach().clone() * weight
            aggregated_state[key] = accumulator
        else:
            aggregated_state[key] = tensor.detach().clone()

    return aggregated_state
