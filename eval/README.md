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
| Qwen2.5-VL-3B | 896 | v3 | 60 | 0.581 | 0.389 | 0.624 | 0.586 | 0.183 | 2.7 GB |
| Qwen2.5-VL-7B | 896 | v2 | 60 | 0.520 | 0.434 | 0.537 | 0.577 | 0.205 | 6.3 GB |
| Qwen2.5-VL-7B | 896 | v3 | 60 | 0.512 | 0.434 | 0.521 | 0.546 | 0.299 | 6.3 GB |
| Qwen2.5-VL-7B | 1280 | v2 | 60 | 0.638 | **0.541** | 0.689 | **0.712** | 0.280 | 6.8 GB |

These runs all take text from the model. Taking it from OCR instead, and
then prompt v4, is covered below.

Mean IoU of matched boxes is about 0.80 for 3B and 0.73-0.79 for 7B.

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
- **The label check is weak end to end** (F1 around 0.2 on 3B). It needs both
  the input and its label text to be detected, and the models miss a lot of
  small text: in the v2 run 144 of 248 text elements weren't found at all.
- **Prompt v3 didn't fix it.** v3 asked for form labels as separate text. On
  3B the longer prompt made the model list fewer elements (9 per page against
  16 real ones), so every score dropped and the label check stayed at 0.18.
  On 7B it helped inputs (F1 0.33 → 0.51) and the label check (0.21 → 0.30),
  but not enough to change the default.
- **7B needs the higher resolution.** With prompt v2, box F1 went from 0.52
  at 896 px to 0.64 at 1280 px, and the runs where it looped until the
  2048-token limit dropped from 5 pages in 60 to 1.
- **7B at 1280 px vs 3B at 896 px** (both prompt v2): the same box F1 (0.64),
  but 7B gets the labels right more often (0.54 vs 0.45; links 0.90 vs 0.69,
  text 0.66 vs 0.52), which lifts contrast (0.71 vs 0.68) and the label check
  (0.28 vs 0.19). Target size is about even (0.69 vs 0.71). It costs 6.8 GB of
  VRAM against 2.6 GB.

### Text from OCR

Every run above missed a lot of small text. An OCR model (PP-OCRv6 small via
rapidocr, CPU) run on the same 60 pages finds 247 of 248 text elements and
204 of 211 links, at about 0.5-1 s per page. With `--text ocr` the model's
text items are dropped and replaced by OCR lines; an OCR line mostly inside a
detected control is treated as that control's caption instead
(`src/prism/vision/hybrid.py`).

Re-scoring the saved outputs of all eight runs this way, without re-running
the model (before → after):

| Run | Boxes F1 | Text F1 | Contrast F1 | Target size F1 | Label F1 |
|---|---|---|---|---|---|
| 3B, 1280, v1 | 0.56 → 0.72 | 0.41 → 0.56 | 0.59 → 0.79 | 0.19 → 0.19 | 0.18 → 0.23 |
| 3B, 896, v1 | 0.63 → 0.72 | 0.44 → 0.56 | 0.73 → 0.81 | 0.19 → 0.18 | 0.19 → 0.22 |
| **3B, 896, v2** | 0.64 → **0.76** | 0.52 → **0.74** | 0.68 → **0.80** | 0.71 → 0.70 | 0.19 → 0.24 |
| 3B, 896, v3 | 0.58 → 0.74 | 0.51 → 0.69 | 0.59 → 0.80 | 0.62 → 0.61 | 0.18 → 0.23 |
| 7B, 1280, v1 (40 pages) | 0.68 → 0.81 | 0.40 → 0.60 | 0.68 → 0.83 | 0.27 → 0.29 | 0.40 → 0.41 |
| 7B, 1280, v2 | 0.64 → 0.73 | 0.66 → 0.76 | 0.71 → 0.78 | 0.69 → 0.66 | 0.28 → 0.28 |
| 7B, 896, v2 | 0.52 → 0.69 | 0.44 → 0.70 | 0.58 → 0.75 | 0.54 → 0.51 | 0.21 → 0.19 |
| 7B, 896, v3 | 0.51 → 0.74 | 0.42 → 0.68 | 0.55 → 0.78 | 0.52 → 0.50 | 0.30 → 0.28 |

- Boxes and contrast improve in every run. Contrast with OCR text (0.80) is
  close to the ceiling set by the rule itself on perfect boxes (0.90).
- Text F1 stops short of the OCR's recall because of precision: placeholder
  text inside an input the model missed is scored as extra text.
- Target size and the label check barely move, because they depend on
  controls, which still come from the model. In the 3B v2 run, 25 of the
  59 false label findings are image placeholders the model called "input",
  and 17 of the 29 unlabeled controls were never detected.
- The 7B model's advantage came mostly from reading more text. With OCR, 3B
  is ahead on boxes, contrast and target size, and 7B on labels.

Raw numbers: [`results/detect-qwen25-3b-nf4-896-p2-ocr.json`](results/detect-qwen25-3b-nf4-896-p2-ocr.json),
[`results/detect-qwen25-7b-nf4-1280-p2-ocr.json`](results/detect-qwen25-7b-nf4-1280-p2-ocr.json).

### Prompt v4: controls only

With OCR supplying the text, prompt v4 asks the model only for controls and
images, without a "text" field. Run on the same 60 pages:

| Model | Prompt | Boxes F1 | Boxes + labels F1 | Target size F1 | Contrast F1 | Label F1 | Output tokens |
|---|---|---|---|---|---|---|---|
| 3B, 896 px | v2 + OCR | 0.761 | 0.534 | **0.702** | 0.801 | 0.240 | 362 |
| **3B, 896 px** | **v4 + OCR** | 0.764 | 0.495 | 0.686 | 0.797 | **0.341** | 303 |
| 7B, 1280 px | v2 + OCR | 0.728 | 0.586 | 0.662 | 0.780 | 0.276 | 425 |
| **7B, 1280 px** | **v4 + OCR** | **0.789** | **0.587** | 0.618 | **0.817** | **0.364** | 243 |

(Output tokens: median per page from the timing runs below.)

- **The label check finally moved**: 0.24 → 0.34 on 3B, 0.28 → 0.36 on 7B.
  "Box only the field itself" plus no text to write made the input boxes fit:
  input F1 went from 0.36 to 0.56 (3B) and 0.36 to 0.65 (7B), and false label
  findings fell from 59 to 44 (3B) and 46 to 25 (7B).
- **Target size dropped a little.** 7B lists fewer elements per page (7 against
  12) and misses more links (F1 0.90 → 0.72); 3B lists as many items as before
  but calls too many things buttons (243 against 147).
- **Token savings depend on the model.** 7B dropped its text items and wrote
  43% fewer tokens. 3B kept listing about 9 items a page and only saved the
  text fields, 16% fewer tokens. Each item still costs about 32 tokens, mostly
  digits: Qwen's tokenizer writes every digit of a coordinate as its own
  token.

v4 + OCR is now the default: it trades a little target-size F1 for a much
better label check and less time per page.

### Speed

`prism-eval timing` runs the first 10 pages three times each after a warm-up
page, times prefill and decoding separately, and samples the GPU's clock,
power and temperature while each page runs. Windows, plugged in, power mode
"Best performance":

| Setup | Output tokens | Prefill | Decoding | Model | OCR | Per page |
|---|---|---|---|---|---|---|
| 3B, 896 px, v2 | 362 | 0.63 s | 15.5 tokens/s | 25.2 s | - | 25.2 s |
| 3B, 896 px, v4 + OCR | 303 | 0.75 s | 15.0 tokens/s | 20.8 s | 1.1 s | 21.9 s |
| 7B, 1280 px, v2 | 425 | 1.08 s | 19.6 tokens/s | 23.2 s | - | 23.2 s |
| 7B, 1280 px, v4 + OCR | 243 | 1.11 s | 19.7 tokens/s | 13.6 s | 1.1 s | **14.7 s** |

(Medians per page; repeats of the same page differed by 1.5-3.3%.)

- **Decoding is 89-97% of the model's time.** Reading the image and the
  prompt (prefill) takes about a second. The only real lever is how many
  tokens the model writes, which is why v4 cuts 7B's time by 37%.
- **7B decodes faster than 3B.** During the 3B runs the GPU drew 20-38 W
  and didn't hold its top clock; during 7B it ran at 2.7 GHz and 55-73 W. So
  the 3B runs leave the GPU waiting, which points at fixed per-token work
  (Python in the generation loop, kernel launches, the 4-bit weights being
  unpacked layer by layer) rather than the GPU itself. That work grows with
  the number of layers, and 3B has more of them (36 against 28). Scaling 7B's
  speed by 28/36 predicts 15.3 tokens/s for 3B; it measured 15.5. Capturing
  the decode step in a CUDA graph is the usual fix for this kind of overhead
  and the next thing to try.
- **The machine has a slow mode.** The detect runs report latency too, but
  the same setups decoded 2.7-5.6x slower there than in the timing runs a few
  minutes later (3B v4: 5.6 against 15.0 tokens/s), and older logs show runs
  switching between the two speeds midway with nothing changed. Since
  decoding here is limited by the CPU side, Windows moving a background
  process onto efficiency cores would explain it; that's a guess that hasn't
  been tested. It is also what was behind the 5.7-13.9 tokens/s spread between
  earlier runs. Speed claims in this repo come from the timing files only.

### Default config

Qwen2.5-VL-3B, NF4, 896 px, prompt v4, text from OCR. It fits GPUs with 4 GB.
On an 8 GB card, 7B at 1280 px is both more accurate on everything except
target size and faster:

```bash
PRISM_DETECTOR_MODEL=Qwen/Qwen2.5-VL-7B-Instruct PRISM_DETECTOR_MAX_SIDE=1280 make worker
```

All of the prompt and model comparisons above were made on the same 60 test
pages, so the chosen setups are slightly flattered. `detect --skip 60` runs on
pages none of these choices looked at.

Reproduce with:

```bash
uv sync --extra worker --extra eval
uv run prism-eval detect --data eval/data/synth-test --limit 60 --max-side 896 \
    --prompt v2 --results eval/results/detect-qwen25-3b-nf4-896-p2.json
uv run prism-eval rescore --data eval/data/synth-test \
    --pages eval/results/detect-qwen25-3b-nf4-896-p2.pages.jsonl --text ocr
uv run prism-eval detect --data eval/data/synth-test --limit 60 --max-side 896 \
    --prompt v4 --text ocr --results eval/results/detect-qwen25-3b-nf4-896-p4-ocr.json
uv run prism-eval timing --data eval/data/synth-test --pages 10 --repeats 3 \
    --prompt v4 --text ocr --results eval/results/timing-qwen25-3b-896-p4-ocr.json
```

Each run's raw model output is kept in `results/*.pages.jsonl`, so parser or
scoring changes can be re-scored without the GPU.
