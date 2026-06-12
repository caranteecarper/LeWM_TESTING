# Official Code Provenance

Created at: 2026-06-12 21:10:46 +08:00

## Initial Repository Check

The originally active local workspace was not LeWM:

- Path: `E:\MyProjects\armyarmy`
- Remote: `https://github.com/jiangdaolime/media_military.git`
- Branch: `main`
- HEAD: `98f9885 合并公众号采集与往日素材选题卡链路`
- Status: untracked local files were present.

The previously used server code directory was also not a usable Git clone:

- Path: `/data/lzt26/le-wm-official-kan`
- Result: `fatal: not a git repository`

Therefore this experiment was not started from either existing workspace.

## Clean Official Clone

Official upstream remote:

```text
origin https://github.com/lucas-maes/le-wm.git
```

Clean clone path:

```text
E:\MyProjects\lewm_official_tworoom_readout
```

Repository state immediately after clone:

```text
branch: main
commit: 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
message: fix data config
```

Commands used for provenance:

```bash
git remote -v
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
```

## Experiment Branch

New branch created from the clean official commit:

```text
exp/tworoom-official-encoder-readout
```

Base commit:

```text
8edfeb336732b5f3ce7b8b210d0ba370a09e2cac fix data config
```

## Official Clean Tag

Read-only provenance tag created locally:

```text
official-lewm-clean-8edfeb3
```

This tag points to:

```text
8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
```

## Scope Guard

This branch must only add TwoRoom encoder readout analysis files and reports. It must not import or copy code from the prior modified KAN/KANFIS, FFN replacement, delta-input, or probe branches.

