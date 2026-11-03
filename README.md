# MeshPilot

Code and data for **MeshPilot: Guardrailed LLM Intent Compilation for Path-Aware
Compute Placement and Bandwidth Admission**.

MeshPilot maps operator preferences to four objective profiles. Deterministic
code selects a feasible worker, verifies the action, and reserves resources
atomically within one controller process.

## Reproduce the results

Use Python 3.11 or later. The tested environment is Python 3.13.7.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-replay.txt
python verify.py
```

The command checks data hashes, recomputes the paired comparison, fixed-balanced
control, profile bounds, fault rejection and concurrent admission, and summarizes
the recorded physical and scaling measurements. Outputs are written to
`reproduced/`; `reproduced/results.json` should report `PASS`.
No model API or worker connection is required.

| Result | Expected value |
|---|---:|
| Known-ID decision pairs | 500 |
| Direct / fixed-balanced / profile-compiled mean regret | 7.2622% / 4.6805% / 0.0867% |
| Direct-to-compiled mean reduction | 7.1755 percentage points |
| Seed-cluster 95% interval for that reduction | 2.1084–13.2557 percentage points |
| Compiled versus fixed-balanced wins / ties / losses | 124 / 374 / 2 |
| Direct choices exceeding every profile choice | 36 / 500 |
| Invalid proposals rejected | 1,600 / 1,600 |
| Atomic-admission overcommits | 0 / 100 trials |
| Recorded execution / recovery successes | 96 / 96 and 12 / 12 |

## Regenerate the scaling experiment

```sh
python -m meshpilot.scale_replay \
  --primary data/decisions.csv.gz \
  --primary-config configs/primary.json \
  --output reproduced/scaling --nodes 12,50,100,200,500,1000
```

This regenerates tasks and candidate sets while reusing recorded profiles.
It measures new planning and verification times on the local machine; compare
decisions and acceptance with the recorded experiment, not exact wall time.

Regression checks:

```sh
python -m unittest discover -s tests
```

## Contents

- `meshpilot/`: offline planner, verifier, reservation ledger and analyses.
- `data/decisions.csv.gz`: 600 direct and 600 profile-compiled records, including
  task, candidate and snapshot fields.
- `data/physical/`: execution and recovery observations, workloads and worker inputs.
- `data/scaling.csv.gz`: recorded timings for six cluster sizes.
- `configs/primary.json`: primary experiment configuration.
- `METHODS.txt`: scoring, statistical definitions and measurement scope.

Regret uses author-declared preferences and the configured analytic objective.
Physical observations document past runs; live replication requires worker setup
and transfer payloads. The repository supports frozen-data analysis and offline
logic replay. The recorded provider configuration is retained as metadata;
the reproduction commands use the offline backend.
