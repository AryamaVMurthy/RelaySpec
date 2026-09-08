"""DDTree builder adapted from https://github.com/liranringel/ddtree
Commit c96427a185677bf4133ed865dd1626a5041aef9b (ddtree.py).
Ringel and Romano, arXiv:2604.12989. Instrumentation removed; optional
node-score recording added without changing fixed-budget tree selection.
Matched-runtime algorithm baseline, not the full upstream runtime.

MIT License

Copyright (c) 2026 Liran Ringel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import heapq
import numpy as np
import torch

def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
    node_log_weights=None,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor]:

    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
        )

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1
        if node_log_weights is not None:
            node_log_weights.append(logw)

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs_np[depth - 1, rank]) + float(top_log_probs_np[depth - 1, rank + 1])
            heapq.heappush(heap, (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw))

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw))


    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility


def choose_budget(node_log_weights, choices, node_cost):
    """Greedy-throughput heuristic using DDTree factorized mass and affine cost.

    Not an estimator of actual target acceptance. node_cost is relative to the
    per-round fixed cost; both are tuning assumptions, not measured latency.
    """
    if node_cost <= 0 or not choices or any(b <= 0 or b > len(node_log_weights) for b in choices):
        raise ValueError("Invalid adaptive budget or relative cost")
    mass=np.cumsum(np.exp(np.asarray(node_log_weights)))
    return max(sorted(set(choices)), key=lambda b: (1.+mass[b-1])/(1.+node_cost*b))


def pack_ddtree(root, draft_logits, budget, extra_path=None, budget_policy=None):
    if budget < 0:
        raise ValueError("Node budget must be nonnegative (excludes root)")
    weights = [] if budget_policy else None
    ids, depths, parents, children, visible = build_ddtree_tree(draft_logits, budget, weights)
    if budget_policy:
        selected = choose_budget(weights, budget_policy["choices"], budget_policy["node_cost"])
        ids, depths = ids[:selected], depths[:selected]
        parents = parents[:selected+1]
        children = [{token:child for token,child in row.items() if child<=selected} for row in children[:selected+1]]
        visible = visible[:selected+1, :selected+1]
    if extra_path:
        # Experimental augmentation, separate from the unchanged DDTree builder:
        # merge one already-observed in-context continuation with shared prefixes.
        token_list, depth_list = ids.tolist(), depths.tolist()
        current = 0
        for depth, token in enumerate(extra_path[:len(draft_logits)], 1):
            token = int(token)
            if token in children[current]:
                current = children[current][token]
                continue
            index = len(parents)
            parents.append(current)
            children[current][token] = index
            children.append({})
            token_list.append(token); depth_list.append(depth)
            current = index
        ids = torch.tensor(token_list, dtype=torch.long)
        depths = torch.tensor(depth_list, dtype=torch.long)
        visible = torch.zeros(len(parents), len(parents), dtype=torch.bool)
        for node in range(len(parents)):
            current = node
            while current >= 0:
                visible[node, current] = True
                current = parents[current]
    leaf_paths = []
    for node, child_map in enumerate(children):
        if not child_map:
            path = []
            while node >= 0:
                path.append(node)
                node = parents[node]
            leaf_paths.append(path[::-1])
    lengths = [len(path) for path in leaf_paths]
    maximum = max(lengths)
    paths = torch.tensor([path+[0]*(maximum-len(path)) for path in leaf_paths], device=root.device)
    packed = torch.cat([root.reshape(1), ids.to(root.device)]).unsqueeze(0)
    proposal = packed[0, paths]
    return (packed, torch.cat([torch.zeros(1, dtype=torch.long), depths]).to(root.device),
            paths, visible.to(root.device), torch.tensor(lengths, device=root.device), proposal)
