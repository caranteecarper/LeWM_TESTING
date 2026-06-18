import csv

import torch

from tworoom_v51_common import *


def build_groups(vis, top_rules):
    q = vis["q_v4_t"][vis["test_idx"]].float()
    corr = torch.corrcoef(q.T).nan_to_num(0.0)
    used = set()
    groups = []
    # Hard-enriched seeds plus co-activated partners.
    for row in top_rules[:8]:
        seed = int(row["unit_id"])
        if seed in used:
            continue
        partners = torch.topk(corr[seed].abs(), k=min(4, corr.shape[0])).indices.tolist()
        rules = []
        for r in partners:
            if r not in rules:
                rules.append(int(r))
        used.update(rules)
        groups.append(rules)
        if len(groups) >= 6:
            break
    return groups


def main():
    ensure_dirs()
    vis = load_visual()
    rule_dir = V51_FIG / "qv4_rule_link"
    group_dir = V51_FIG / "qv4_group_link"
    rule_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)
    top_rules = top_rule_rows(vis, "q_v4_t", k=8)
    for row in top_rules:
        save_unit_visuals(rule_dir, vis, "q_v4_t", int(row["unit_id"]), f"qv4_rule_{row['unit_id']}")
    groups = build_groups(vis, top_rules)
    group_rows = []
    for i, rules in enumerate(groups):
        group_rows.append(save_group_visuals(group_dir, vis, i, rules))
    with (V51_OUT / "qv4_rule_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(top_rules[0].keys()))
        writer.writeheader()
        writer.writerows(top_rules)
    with (V51_OUT / "qv4_group_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)
    text = f"""# V5.1 q_v4 Rule And Group Visual Link

## Top q_v4 Single Rules

{table_md(top_rules, ["unit_id", "top_hard_subset", "hard_enrichment", "mean_residual_mag", "spatial_compactness", "coverage_top10"])}

## q_v4 Rule Groups

V4 is a soft rule representation, so grouped/co-activated rules are often more interpretable than individual rules.

{table_md(group_rows, ["group_id", "rules", "top_hard_subset", "hard_enrichment", "mean_residual_mag", "spatial_compactness", "coverage_top10"])}

## Figures

- single-rule figures: `{rule_dir}`
- group figures: `{group_dir}`

Each unit/group has top images, activation-over-position, and residual-direction figures.
"""
    write_report("01_qv4_rule_and_group_visual_link.md", text)


if __name__ == "__main__":
    main()

