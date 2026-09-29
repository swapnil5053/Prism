# Evaluation

Two questions, measured separately:

1. **Are the audit rules right when the boxes are right?** (`oracle`) Runs the
   contrast, target-size and label checks on ground-truth boxes. CPU only.
2. **How good is the whole pipeline?** (`detect`) Runs the VLM, matches its
   boxes to ground truth (same kind, IoU ≥ 0.5), then scores the findings.
   Needs a GPU.

## Data

`prism-eval synth` builds random pages (nav bars, forms, cards) with varied
colours, sizes, font weights, missing labels and 1x/2x pixel ratios, renders
them in headless Chromium and labels them from the DOM. Text colours are
drawn to hit contrast targets from 1.6:1 to 14:1, including values just
either side of 4.5:1, so the borderline cases are well represented.

```bash
uv sync --extra eval
uv run playwright install chromium
uv run prism-eval synth --out eval/data/synth-dev  --count 300 --seed 13
uv run prism-eval synth --out eval/data/synth-test --count 300 --seed 99
```

The dev set (seed 13) was used while working on the rules. Numbers below are
from the test set (seed 99), which wasn't looked at until the rules were fixed.
Datasets aren't committed; they regenerate identically from the seed.

Any other dataset can be scored by converting it to the same format: a folder
of images plus `labels.jsonl` (see `src/prism/evaluation/dataset.py`).

## Results

### Audit rules on true boxes (`oracle`, synth-test, 300 pages)

| Rule | Precision | Recall | F1 |
|---|---|---|---|
| target-size (2.5.8) | 1.000 | 1.000 | 1.000 |
| text-contrast (1.4.3) | 0.870 | 0.931 | 0.899 |
| visible-label (3.3.2) | 1.000 | 0.748 | 0.856 |

Contrast estimation over 3,033 text elements: median relative error 4.1%,
90th percentile 10.3%. Pass/fail at 4.5:1 agrees with the true ratio 90.8% of
the time overall and **99.5%** when the true ratio is more than 10% away from
the threshold. Nearly all disagreements are borderline cases.

Target size is exact here because the boxes are exact; with detected boxes
it inherits the detector's box error.

Known misses:

- *Contrast*: whether text is "large" is guessed from box height, since font
  weight isn't visible. 18.66px bold text is treated as normal text.
- *Labels*: a heading sitting directly above an unlabeled input looks like a
  label and suppresses the finding. That's 35 of the 139 unlabeled controls.

Raw numbers: [`results/oracle-test.json`](results/oracle-test.json),
[`results/oracle-dev.json`](results/oracle-dev.json).

### Full pipeline (`detect`)

```bash
uv sync --extra worker --extra eval
uv run prism-eval detect --data eval/data/synth-test --quant nf4 --max-side 1280 \
    --results eval/results/detect-qwen25-3b-nf4.json
```

Reports detection precision/recall per element kind, end-to-end finding
precision/recall, valid-JSON and truncation rates, p50/p95 latency and peak
VRAM. Not run yet.
