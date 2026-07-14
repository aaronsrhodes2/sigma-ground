"""PartNet semantic hierarchy — the part tree every on-disk model carries.

Every one of the 32,537 local PartNet models ships a ``result.json``: a nested
tree of NAMED parts (chair → chair_back → back_soft_surface → …) whose leaf
``objs[]`` arrays map to the OBJ files ``load_parts`` streams. Until now the
pipeline flattened that tree and voxelized everything as one lump — this module
recovers it, so parts survive into the voxel field (Lane 1 of the actuation
epic) and, later, mobility annotations can attach joints to the same node ids.

The walkers here are EXTRACTED from ``tools/distill_partnet.py`` (which now
imports them back — one implementation, parity-gated by tests): ``_clean``,
``_subtree_objs``, ``_node_instances``, ``load_hierarchy``.

``part_groups(anno_id)`` is the articulation-facing view: the DEPTH-1 children
of the category root, one group per INSTANCE (chair 44164 → back, arm, arm,
seat, base), each carrying its node id (the id space PartNet-Mobility
annotations reference — result.json is therefore preferred over
result_after_merging.json here, the opposite of distill's default) and the OBJ
stems that are its geometry. Stray objs sitting directly on the root become a
flagged "body" group, so the partition is always exact.
"""
from __future__ import annotations

import json
import pathlib

_DATA = pathlib.Path("D:/datasets/shapenet/PartNet/data_v0")

# grouping-noise labels that are wrappers, not parts
_NOISE = {"containing_things", "other", "others", "other_leaf", ""}


def _clean(name: str, root: str = "") -> str:
    """Normalize a PartNet node label: strip grouping suffixes + category prefix."""
    name = (name or "").strip().lower()
    for suf in ("_side", "_group", "_set", "_unit"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    if root and name.startswith(root + "_"):
        name = name[len(root) + 1:]
    return name.strip("_")


def _subtree_objs(node) -> list:
    """Every OBJ stem in a node's subtree, in tree order."""
    objs = list(node.get("objs") or [])
    for ch in node.get("children") or []:
        objs += _subtree_objs(ch)
    return objs


def _node_instances(root_node, root_label: str) -> list:
    """Every named node below the category root as
    (cleaned_label, [subtree objs], is_leaf). A node IS an instance of its
    label; its geometry is its whole subtree. ``is_leaf`` distinguishes a real
    PART that happens to span the object (a bottle's body) from a subtype
    WRAPPER (regular_table) — only wrappers get whole-object-filtered."""
    out = []

    def walk(n, depth):
        nm = _clean(n.get("name") or "", root_label)
        if depth >= 1 and nm not in _NOISE:
            out.append((nm, _subtree_objs(n), not (n.get("children"))))
        for ch in n.get("children") or []:
            walk(ch, depth + 1)

    walk(root_node, 0)
    return out


def load_hierarchy(model_dir, prefer=("result.json", "result_after_merging.json")):
    """The model's part-tree root node, or None.

    ``prefer`` orders the candidate files: articulation reads ``result.json``
    FIRST (its node ids are the space PartNet-Mobility references); the distill
    tool keeps its historical merged-first preference by passing the reverse.
    """
    model_dir = pathlib.Path(model_dir)
    for fn in prefer:
        p = model_dir / fn
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                if isinstance(data, list) and data:
                    return data[0]
            except Exception:
                return None
    return None


def part_groups(anno_id, *, data_root=None) -> list | None:
    """The model's articulation-facing part partition, or None when the model
    (or its hierarchy) isn't on disk.

    Returns ``[{"name", "instance", "node_id", "objs", "flags"}, ...]`` where
    each entry is one DEPTH-1 child of the category root — one per INSTANCE
    (two arms → two groups, ``instance`` 0 and 1) — with the node's whole
    subtree of OBJ stems as its geometry. Root-level stray objs become a
    trailing ``"body"`` group flagged ``("root_objs",)`` so every OBJ is
    accounted for exactly once.
    """
    root_dir = pathlib.Path(data_root) if data_root else _DATA
    model_dir = root_dir / str(anno_id)
    root = load_hierarchy(model_dir)
    if root is None:
        return None
    root_label = _clean(root.get("name") or "")
    groups = []
    counts: dict = {}
    for child in root.get("children") or []:
        name = _clean(child.get("name") or "", root_label)
        if name in _NOISE:
            # a wrapper: promote ITS children instead, so noise never hides parts
            for sub in child.get("children") or []:
                sname = _clean(sub.get("name") or "", root_label)
                if sname in _NOISE:
                    sname = "part"
                inst = counts.get(sname, 0)
                counts[sname] = inst + 1
                groups.append({"name": sname, "instance": inst,
                               "node_id": sub.get("id"),
                               "objs": _subtree_objs(sub),
                               "flags": ("noise_wrapper_promoted",)})
            stray = list(child.get("objs") or [])
            if stray:
                inst = counts.get("body", 0)
                counts["body"] = inst + 1
                groups.append({"name": "body", "instance": inst,
                               "node_id": child.get("id"), "objs": stray,
                               "flags": ("noise_wrapper_objs",)})
            continue
        inst = counts.get(name, 0)
        counts[name] = inst + 1
        groups.append({"name": name, "instance": inst,
                       "node_id": child.get("id"),
                       "objs": _subtree_objs(child), "flags": ()})
    stray = list(root.get("objs") or [])
    if stray:
        inst = counts.get("body", 0)
        counts["body"] = inst + 1
        groups.append({"name": "body", "instance": inst,
                       "node_id": root.get("id"), "objs": stray,
                       "flags": ("root_objs",)})
    return groups or None


__all__ = ["part_groups", "load_hierarchy", "_node_instances",
           "_subtree_objs", "_clean", "_NOISE"]
