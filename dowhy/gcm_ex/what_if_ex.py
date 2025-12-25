"""This module provides functionality to answer what-if questions."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm._noise import compute_noise_from_data
from dowhy.gcm.causal_mechanisms import ClassifierFCM
from dowhy.gcm.causal_models import (
    InvertibleStructuralCausalModel,
    ProbabilisticCausalModel,
    StructuralCausalModel,
    validate_causal_dag,
)
from dowhy.gcm.fitting_sampling import draw_samples
from dowhy.graph import (
    DirectedGraph,
    get_ordered_predecessors,
    is_root_node,
    node_connected_subgraph_view,
    validate_node_in_graph,
)


def interventional_samples(
    causal_model: ProbabilisticCausalModel,
    interventions: Dict[Any, Dict[str, Callable]],
    observed_data: Optional[pd.DataFrame] = None,
    num_samples_to_draw: Optional[int] = None,
) -> pd.DataFrame:
    """Performs intervention on nodes in the causal graph.

    :param causal_model: The probabilistic causal model we perform this intervention on .
    :param interventions: Dictionary containing the interventions we want to perform, keyed by node name. Each value is a dictionary with 'condition' and 'intervention' keys.
    Example: {'X': {'condition': lambda row: row['A'] > 0, 'intervention': lambda x: x * 2}}
    :param observed_data: Optionally, data on which to perform interventions. If None are given, data is generated based
                          on the generative models.
    :param num_samples_to_draw: Sample size to draw from the interventional distribution.
    :return: Samples from the interventional distribution.
    """
    validate_causal_dag(causal_model.graph)
    for node in interventions:
        validate_node_in_graph(causal_model.graph, node)

    if observed_data is None and num_samples_to_draw is None:
        raise ValueError("Either observed_samples or num_samples_to_draw need to be set!")
    if observed_data is not None and num_samples_to_draw is not None:
        raise ValueError("Either observed_samples or num_samples_to_draw need to be set, not both!")

    if num_samples_to_draw is not None:
        observed_data = draw_samples(causal_model, num_samples_to_draw)

    return _interventional_samples(causal_model, observed_data, interventions)


def _interventional_samples(
    pcm: ProbabilisticCausalModel,
    observed_data: pd.DataFrame,
    interventions: Dict[Any, Dict[str, Callable]],
) -> pd.DataFrame:
    samples = observed_data.copy()

    affected_nodes = _get_nodes_affected_by_intervention(pcm.graph, interventions.keys())
    print(affected_nodes)
    sorted_nodes = nx.topological_sort(pcm.graph)

    # Simulating interventions by propagating the effects through the graph. For this, we iterate over the nodes based
    # on their topological order.
    for node in sorted_nodes:
        if node not in affected_nodes:
            print(f"{node} is not in affected nodes.")
            continue

        if is_root_node(pcm.graph, node):
            node_data = samples[node].to_numpy()
            print(f"{node} is root node.")
        # elif node in interventions.keys():
        #     node_data = samples[node].to_numpy()
        #     print(f"{node} is in interventions.")
        else:
            print(f"{node} is in affected nodes and not root node.")
            node_data = pcm.causal_mechanism(node).draw_samples(_parent_samples_of(node, pcm, samples))

        # After drawing samples of the node based on the data generation process, we apply the corresponding
        # intervention. The inputs of downstream nodes are therefore based on the outcome of the intervention in this
        # node.
        # print(f"pre_intervention_data:{node_data.reshape(-1).shape}, row_condition:{samples.shape}")
        node_data = _evaluate_intervention(node, interventions, node_data.reshape(-1), samples)

        # If data is updated by intervention directly or indirectly,
        # data is stored as new attribute POST(Attr) representing the updated value.
        samples[f"POST_{node}"] = node_data

    return samples


def _get_nodes_affected_by_intervention(causal_graph: DirectedGraph, target_nodes: Iterable[Any]) -> List[Any]:
    result = []

    for node in nx.topological_sort(causal_graph):
        if node in target_nodes:
            result.append(node)
            continue

        for target_node in target_nodes:
            if target_node in nx.ancestors(causal_graph, source=node):
                result.append(node)
                break

    return result


def _evaluate_intervention(
    node: Any,
    interventions: Dict[Any, Dict[str, Callable]],
    pre_intervention_data: np.ndarray,
    row_conditions: pd.DataFrame,
) -> np.ndarray:
    """
    Apply intervention to a node with optional row-specific conditions.

    :param node: The node to which the intervention is applied.
    :param interventions: Dictionary containing interventions with conditions and operations.
    :param pre_intervention_data: Data for the node before intervention.
    :param row_conditions: DataFrame with the same index as observed_data to check conditions row-wise.
    :return: Data for the node after applying the intervention.
    """
    if node in interventions:
        condition_fn = interventions[node].get("condition", lambda row: True)  # Default: always True
        intervention_fn = interventions[node]["intervention"]

        # Apply intervention only to rows satisfying the condition
        post_intervention_data = pre_intervention_data.copy()
        for idx, row in row_conditions.iterrows():
            numpy_index = row_conditions.index.get_loc(idx)
            if condition_fn(row):
                post_intervention_data[numpy_index] = intervention_fn(pre_intervention_data[numpy_index])

        return post_intervention_data
    else:
        return pre_intervention_data


def _parent_samples_of(node: Any, scm: ProbabilisticCausalModel, samples: pd.DataFrame) -> np.ndarray:
    return samples[get_ordered_predecessors(scm.graph, node)].to_numpy()
