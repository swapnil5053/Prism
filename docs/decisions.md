# Design decisions

## Detect with a model, audit with code

The first version asked the VLM to report WCAG violations directly. It can't
do that reliably: a contrast ratio is arithmetic on two colours, and a 3B model
guessing it gives answers you can't test. So the model only finds elements
(what it's good at), and each check is a small function that measures pixels
and is unit-tested against known contrast ratios.

## Qwen2.5-VL-3B in 4-bit

With NF4 weights the 3B model peaks at 2.6 GB of VRAM, and it was trained on
grounding, so it emits boxes directly. Qwen2.5-VL writes boxes in pixels of
the resized image it saw; Qwen2-VL uses a 0-1000 grid. `vision/qwen.py`
handles both and converts to [0, 1] of the original image before anything
else sees them. The vision encoder stays in bf16; only the language model is
quantised.

The 7B model at 1280 px labels elements more accurately (box+label F1 0.54
vs 0.45) and still fits an 8 GB card at 6.8 GB. Most of that lead was on
text, though, and with OCR finding the text (below) the two are close: 7B is
ahead on labels and the label check, 3B on boxes, target size and contrast.
3B stays the default; it needs 2.6 GB. Switching is two environment variables.

The output is treated as untrusted: parsed tolerantly (code fences, truncated
arrays), validated with Pydantic, and rendered with `textContent` in the
browser and escaped in reports. Text inside a screenshot can say anything,
including instructions to the model, but the output is only ever data.

## Prompt

The first prompt listed the labels and nothing else. Boxes came back in the
right places (mean IoU 0.8) but links and most buttons were labelled "text",
which quietly disabled the target-size check. The second prompt says what each
label means and where its box goes. On the same 60 pages, label F1 went from
0.27 to 0.45 and end-to-end target-size F1 from 0.19 to 0.71. Measuring box
placement and labels separately is what made the problem visible; a single
strict score just looked uniformly bad.

A third prompt tried to keep form labels as separate text for the label
check. It made the 3B model list fewer elements overall and scored lower on
everything, so v2 stays the default and v3 is kept for comparison.

## Text from OCR

Scoring boxes by kind showed where the models were weak: small text. The 3B
model missed 144 of 248 text elements; the 7B at 1280 px still missed 92.
Contrast is checked on text, so every missed line is a missed finding.

A dedicated OCR model (PP-OCRv6 small through rapidocr, ONNX on the CPU)
finds 247 of the 248 and 204 of 211 links, in about half a second a page. So
the pipeline splits the job: OCR finds text, the VLM finds and classifies
controls and images, and `vision/hybrid.py` merges them. The VLM's own text
items are dropped. An OCR line counts as a control's caption, not separate
text, when at least half of it lies inside a button, link, input, checkbox
or icon; images don't swallow text, so a headline over a hero image stays.

Re-scoring the saved model outputs with OCR text improved box F1 and
contrast F1 on every one of the eight runs (3B v2: 0.64 to 0.76, and 0.68 to
0.80). It didn't help the label check. That check fails on controls, not
text: models call empty image placeholders "input", and miss more than half
of the unlabeled inputs and checkboxes.

With OCR doing the text, the VLM no longer needs to write it out. Generation
is where the time goes, so prompt v4 asks only for controls and images and
drops the "text" field, which should cut output tokens substantially.

rapidocr depends on the desktop OpenCV build, which needs libGL at import
and would break the slim worker image. A uv override swaps in the headless
build, which provides the same `cv2` module.

## Contrast from pixels

Pixels inside a text box are split into two clusters. The bigger one is the
background. For the text colour, we take the 20% of text pixels furthest from
the background, because the anti-aliased edges are a blend of both and
averaging them understates contrast.

The clustering is seeded with the most common colour and the pixel furthest
from it. Seeding with the darkest and lightest pixels (the obvious choice)
failed on small buttons, where the rounded corners are page background and
got picked as "text". Found by the synthetic benchmark; recall went from 0.77
to 0.93.

## Redis Streams, not pub/sub

Pub/sub drops messages sent before a subscriber connects. The browser opens
its WebSocket after the upload returns, and a fast job could finish before
that. A stream per analysis keeps the events (capped, and expiring an hour
after the job ends), so a client can connect late or reconnect with
`?after=<last id>` and miss nothing.

Closing the WebSocket doesn't cancel the job. Cancelling is an explicit
`POST /cancel`, so refreshing the page doesn't throw work away.

## Cancelling a running generation

`model.generate()` runs in a worker thread so the event loop stays free. An
asyncio task polls a Redis key twice a second and sets a `threading.Event`; a
`StoppingCriteria` checks that event on every token. The model loop never
waits on network I/O.

## One job per GPU

`max_jobs = 1`. Two concurrent generations on an 8 GB card either run out of
memory or run at half speed each, so queueing is better. Scaling out means
more worker processes, each with its own GPU, reading the same queue.

## Anonymous workspaces

No accounts. The first upload creates a workspace and sets an HMAC-signed,
httpOnly cookie with its id. Every read checks ownership and returns 404 for
someone else's analysis (so ids can't be probed). Enough isolation for a
demo without a login system.

## Synthetic benchmark

Real screenshot datasets rarely have exact colours or label relationships, so
the audit rules are measured on generated pages rendered in Chromium, where
the DOM gives exact ground truth. Rules were tuned on one seed and reported on
another. The detector itself should also be measured on real screenshots; the
dataset format is simple enough to convert one into.
