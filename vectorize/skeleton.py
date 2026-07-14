"""Trace an eight-connected skeleton into deterministic graph edges."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from cellvector.domain.models import Point


Pixel = tuple[int, int]
ORTHOGONAL = ((-1, 0), (0, -1), (0, 1), (1, 0))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def trace_skeleton(mask: NDArray[np.bool_]) -> list[list[Point]]:
    """Return maximal endpoint/junction-to-endpoint/junction paths and loops."""

    if mask.ndim != 2:
        raise ValueError("skeleton mask must be two-dimensional")
    graph = _build_graph(np.asarray(mask, dtype=bool))
    visited: set[frozenset[Pixel]] = set()
    pixel_paths: list[list[Pixel]] = []

    terminals = sorted(node for node in graph.nodes if graph.degree[node] != 2)
    for start in terminals:
        for neighbor in sorted(graph.neighbors(start)):
            edge = frozenset((start, neighbor))
            if edge in visited:
                continue
            pixel_paths.append(_walk_edge(graph, start, neighbor, visited))

    for start, neighbor in sorted(graph.edges):
        edge = frozenset((start, neighbor))
        if edge in visited:
            continue
        pixel_paths.append(_walk_loop(graph, start, neighbor, visited))

    paths = [[Point(x=x, y=y) for y, x in path] for path in pixel_paths if len(path) >= 2]
    return sorted(paths, key=_path_sort_key)


def _build_graph(mask: NDArray[np.bool_]) -> nx.Graph:
    graph = nx.Graph()
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        node = (int(y), int(x))
        graph.add_node(node)
        for dy, dx in (*ORTHOGONAL, *DIAGONAL):
            ny, nx_ = node[0] + dy, node[1] + dx
            if not (0 <= ny < height and 0 <= nx_ < width and mask[ny, nx_]):
                continue
            if dy != 0 and dx != 0 and (mask[node[0], nx_] or mask[ny, node[1]]):
                continue
            graph.add_edge(node, (ny, nx_))
    return graph


def _walk_edge(
    graph: nx.Graph,
    start: Pixel,
    neighbor: Pixel,
    visited: set[frozenset[Pixel]],
) -> list[Pixel]:
    path = [start]
    previous, current = start, neighbor
    visited.add(frozenset((previous, current)))
    path.append(current)
    while graph.degree[current] == 2:
        candidates = sorted(node for node in graph.neighbors(current) if node != previous)
        next_node = candidates[0]
        edge = frozenset((current, next_node))
        if edge in visited:
            break
        visited.add(edge)
        previous, current = current, next_node
        path.append(current)
    return path


def _walk_loop(
    graph: nx.Graph,
    start: Pixel,
    neighbor: Pixel,
    visited: set[frozenset[Pixel]],
) -> list[Pixel]:
    path = [start]
    previous, current = start, neighbor
    while True:
        visited.add(frozenset((previous, current)))
        path.append(current)
        candidates = sorted(node for node in graph.neighbors(current) if node != previous)
        if not candidates:
            break
        next_node = candidates[0]
        if next_node == start:
            visited.add(frozenset((current, start)))
            path.append(start)
            break
        if frozenset((current, next_node)) in visited:
            break
        previous, current = current, next_node
    return path


def _path_sort_key(path: list[Point]) -> tuple[float, float, float, float]:
    first, last = path[0], path[-1]
    return (first.y, first.x, last.y, last.x)

