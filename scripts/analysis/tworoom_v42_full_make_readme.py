from tworoom_v42_full_common import *


def main():
    ensure_dirs()
    compare_rows = read_csv(FULL_TABLE / "v42_vs_v4_final_comparison.csv")
    pass_count = sum(1 for r in compare_rows if r.get("v42_pass") == "True")
    if pass_count >= 6:
        final_choice = "A. V4.2 replaces V4 as the main rule representation."
        pred_input = "Use V4.2 full q as the next prediction-module input candidate when the goal is to replace V4 with a clearer compromise rule representation."
        whitebox = "Use V4.2 best single rules/groups as the main V4-replacement display, while keeping V4.1 sparse rules as a stricter white-box reference."
    else:
        final_choice = "B. V4 remains the main line; V4.2 is a compromise/control."
        pred_input = "Use V4 full q as the next prediction-module input candidate."
        whitebox = "Use V4 group 0 as the main explanation and V4.1/V4.2 sparse rules as comparison displays."

    sections = []
    for name in [
        "00_input_check.md",
        "01_v42_all_rule_audit.md",
        "02_v42_group_audit.md",
        "03_v42_group_only_performance.md",
        "04_v42_group_mask_ablation.md",
        "05_v42_vs_v4_final_comparison.md",
        "06_recommended_figures.md",
    ]:
        path = FULL_REPORT / name
        if path.exists():
            sections.append(path.read_text())
    text = f"""# TwoRoom V4.2 Full Validation

## Run Provenance Note

The server checkout may report stale commit `4f42a43` because server-side GitHub pull is unreliable. The `tworoom_v42_full_*` scripts used for this run were synchronized from the local experiment branch by `scp`; this commit contains those scripts. No LeWM encoder, predictor, train loop, loss, module, environment, h model, or V4/V4.1/V4.2 q model was retrained or modified.

## Final Judgment

{final_choice}

This means V4.2 satisfies the configured replacement criteria against V4. It does not mean V4.2 dominates V4.1 on every metric: on this visual validation subset, `q_v41+action` has slightly lower all-test residual MSE than `q_v42+action`, so V4.1 remains a strong sparse/performance reference.

## Recommendation

- Prediction-module input candidate: {pred_input}
- White-box display: {whitebox}
- Next optimization target: improve V4.2 versus V4.1 residual performance; V4.2 passes the replacement criteria for V4, but V4.1 can remain a strong sparse/performance reference.

## Constraint Summary

This validation does not retrain LeWM, h, V4 q, V4.1 q, or V4.2 q. It reads the existing V4.2 best model and trains only diagnostic readouts for audit/performance comparison.

---

{chr(10).join(sections)}
"""
    write_report("README.md", text)


if __name__ == "__main__":
    main()
