# Evaluation

Two questions, measured separately:

1. **Are the audit rules right when the boxes are right?** (`oracle`) Runs the
   contrast, target-size and label checks on ground-truth boxes. CPU only.
2. **How good is the whole pipeline?** (`detect`) Runs the VLM on the same
   pages, matches its boxes to ground truth at IoU ≥ 0.5, then scores the
   findings. Needs a GPU.

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
Datasets aren't committed; they regenerate from the seed. Fonts differ between
operating systems, so a page rendered on Windows has slightly different boxes
from the same page rendered on Linux. The `oracle` numbers come from a Linux
render and the `detect` runs from a Windows one.

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

The VLM finds the elements, then the same rules run on its boxes. Runs were
on an RTX 4060 Laptop GPU (8 GB), Windows, against synth-test pages rendered
on that machine.

Two detection scores, because they fail differently:

- **boxes**: a predicted box counts if it overlaps a true element at IoU ≥ 0.5,
  whatever label it has. This is what the contrast check needs.
- **boxes + labels**: the label must match too. Target size only applies to
  interactive elements, so it needs the label.

Rule scores match each finding to ground truth by location.

| Model (4-bit NF4) | Input side | Prompt | Pages | Boxes F1 | Boxes + labels F1 | Target size F1 | Contrast F1 | Label F1 | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | 1280 | v1 | 60 | 0.560 | 0.228 | 0.189 | 0.594 | 0.178 | 2.9 GB |
| Qwen2.5-VL-3B | 896 | v1 | 60 | 0.632 | 0.272 | 0.188 | 0.728 | 0.186 | 2.6 GB |
| Qwen2.5-VL-7B | 1280 | v1 | 40 | 0.682 | 0.338 | 0.273 | 0.683 | 0.400 | 6.7 GB |
| **Qwen2.5-VL-3B** | **896** | **v2** | 60 | 0.638 | **0.449** | **0.713** | 0.681 | 0.187 | 2.6 GB |

Mean IoU of matched boxes is about 0.80 in every run.

What the runs showed:

- **Boxes were fine from the start; labels weren't.** With prompt v1 the 3B
  model never used the label "link" (138 of 211 links came back as "text")
  and boxed the words on a button instead of the button. The confusion
  matrices in the result files show it directly.
- **Prompt v2** defines each label and where its box goes. Link F1 went from 0
  to 0.69, button F1 from 0.21 to 0.43, and end-to-end target-size F1 from
  0.19 to 0.71, with box F1 unchanged.
- **Downscaling to 896 px helped** the 3B model (box F1 0.56 → 0.63): fewer
  image tokens, and these pages don't have detail that needs more.
- **7B finds more** (box F1 0.68, and far better on inputs: 0.64 vs 0.36) but
  was only run with prompt v1, on 40 pages. It fits in 8 GB at 6.7 GB peak.
- **Prompt v2 made the label check worse.** It told the model that words on a
  button are part of the button, and the model applied that to inputs and
  checkboxes too, folding their labels into the control. With fewer separate
  label elements, the visible-label rule reports labels as missing (precision
  0.12). That's the next prompt change.

Default config: Qwen2.5-VL-3B, NF4, 896 px, prompt v2. It has the best
end-to-end target-size score and uses a third of the memory of 7B.

Speed: 3B generates about 370-410 tokens per page. Throughput varied between
runs on the same model (5.7 to 13.9 tokens/s) with similar output lengths, so the
speed-up in the v2 run (24 s per page at the median) shouldn't be credited to
the prompt. A controlled timing run is still to do.

Reproduce with:

```bash
uv sync --extra worker --extra eval
uv run prism-eval detect --data eval/data/synth-test --limit 60 --max-side 896 \
    --prompt v2 --results eval/results/detect-qwen25-3b-nf4-896-p2.json
uv run prism-eval rescore --data eval/data/synth-test \
    --pages eval/results/detect-qwen25-3b-nf4-896-p2.pages.jsonl
```

Each run's raw model output is kept in `results/*.pages.jsonl`, so parser or
scoring changes can be re-scored without the GPU.
