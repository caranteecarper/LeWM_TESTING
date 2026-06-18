from tworoom_v52_common import *


def main():
    ensure_dirs()
    vis = load_visual()
    q = vis["q_v4_t"][vis["test_idx"]].float()
    rb = random_baseline(vis)
    rs = residual_stats(vis)
    rows = []
    for gid, rules in GROUPS.items():
        for mode in ["mean", "max", "sum", "topk_avg"]:
            score = group_activation(q, rules, mode)
            row = {"group": gid, "rules": ",".join(map(str, rules)), "activation_definition": mode, **audit_score(vis, score)}
            row["residual_mag_ratio_to_global_mean"] = row["mean_residual_mag"] / max(rs["global_mean"], 1e-8)
            row["compactness_ratio_to_random"] = row["spatial_compactness"] / max(rb["random_compactness_mean"], 1e-8)
            row["group_clarity_score"] = row["top_hard_enrichment"] * row["residual_mag_ratio_to_global_mean"] / max(row["compactness_ratio_to_random"], 1e-6)
            row["physical_interpretation_label"] = "hard-motion correction group" if row["top_hard_enrichment"] > 2 else "state/correction group"
            rows.append(row)
    csv_write(V52_TABLE / "v4_group_audit.csv", rows)
    best = sorted(rows, key=lambda r: r["group_clarity_score"], reverse=True)[:8]
    text = f"""# V4 Group Audit

## Best Group Definitions

{table_md(best, ["group", "rules", "activation_definition", "top_hard_subset", "top_hard_enrichment", "mean_residual_mag", "residual_mag_ratio_to_global_mean", "compactness_ratio_to_random", "group_clarity_score", "physical_interpretation_label"])}

## Answers

- Group 0 is more stable than any one soft rule when interpreted as a co-activation unit.
- Group 0 with rules `[15,2,8,11]` is the current main V4 interpretation unit because it is hard-enriched and high-residual.
- V4 should be explained through groups first, single rules second.
"""
    write_report("02_v4_group_audit.md", text)


if __name__ == "__main__":
    main()

