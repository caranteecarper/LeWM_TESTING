from tworoom_v5_common import *


def read(name):
    p = V5_REPORT / name
    return p.read_text() if p.exists() else f"# {name}\n\nMissing.\n"


def main():
    ensure_dirs()
    parts = [
        "# TwoRoom V5 Representation Visualization\n",
        """## Executive Summary

V5 visualizes representation layers after training. It does not modify LeWM, h, q, KANFIS, or the predictor. Image decoders are post-hoc and receive fixed z/h/q/hq tensors; current-frame image reconstruction does not use action.

The intended interpretation is qualitative plus diagnostic: z should retain richer visual detail, h should retain continuous state/room geometry, q should expose coarse rule partitions, and [h, q_v41] should be the strongest next-stage input candidate when both state continuity and white-box rules are useful.
""",
    ]
    for name in ["00_scope.md", "01_visual_dataset.md", "02_image_reconstruction.md", "03_semantic_maps.md", "04_rule_visual_link.md"]:
        parts += ["\n---\n", read(name)]
    write_report("README.md", "\n".join(parts))


if __name__ == "__main__":
    main()
