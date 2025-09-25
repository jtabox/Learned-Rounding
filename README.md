> Don't you hate it when you ask Copilot a question because you don't understand a single thing, and it ends with you having to fork a whole repo in order to save all the scripts and information from the rabbit hole Copilot pulled you into?

> Me too.

# Copilot being helpful, informative and overwhelming

## _What do the three versions mean?_

### Short answer

- **"learned"** — the original learned-rounding / AdaRound style converter. It does per-weight optimization with gradient-based updates (learning rate, regularization, iterations) to choose whether each weight should be rounded up or down so the quantized model's error on calibration data is small. This is the most straightforward learned-rounding variant (slower; fine-grained control via lr and reg_lambda).

- **"learned svd"** — a version that uses an SVD / low-rank principal-component approximation of each weight matrix during optimization. Instead of trying to directly minimize the full-matrix error, it projects the quantization error onto the top principal component(s) (typically the top-1) and optimizes rounding with respect to that principal direction. The SVD step focuses the effort on the most important error direction(s), giving much better FP8 results than naive rounding while being more efficient than optimizing against the full error everywhere.

- **"learned svd fast"** — the same SVD-based idea but implemented and parameterized for speed. It uses a faster/low-rank PCA routine (torch.pca_lowrank / pca_lowrank with a configurable top_k), more aggressive/efficient bookkeeping, and small implementation optimizations so large models are converted much faster while still getting the principal-component error-correction benefits. This is the one advertised in the README as the recommended/fast TPEC-Quant path.

### A bit more detail and tradeoffs

- What “learned rounding” means here: rather than using simple round-to-nearest when going from FP32 (or BF16) to FP8, the script treats each rounding decision as something to optimize. It generates calibration inputs, measures the model-layer error after naive quantization, and then iteratively flips/adjusts rounding choices to reduce the error on the calibration data (gradient-like or candidate flips). That typically yields much lower accuracy loss than naive rounding.

- Why SVD helps: weight matrices often have a few principal directions that matter most for the layer’s output. By computing a low-rank decomposition (SVD/PCA) and projecting the error onto the top components, you get a small, accurate surrogate for the parts of the error that matter most. Optimizing with that surrogate is faster (you don’t need to treat every matrix element equally) and often more effective for FP8, where small errors can be critical.

- Differences in code / hyperparameters you can tune:

  - "learned" scripts expose lr and reg_lambda (typical AdaRound hyperparams) and num_iter.
  - "learned_svd" simplifies some hyperparams and uses an SVD/PCA step (the repo’s variant appears to pick a top-1 component in that file).
  - "learned_svd_fast" exposes num_iter and top_k (how many principal components to keep) and uses pca_lowrank with fewer iterations / tuned parameters for speed. It also uses per-tensor scaling into FP8 range (the scripts are all “scaled FP8” variants).

### Which to use

- If you want best speed + very good quality: use learned_svd_fast (recommended by the README).
- If you want to test the SVD idea or inspect behavior: learned_svd (slower/more conservative SVD settings).
- If you want the classic AdaRound-style behavior (maybe slightly different quality characteristics): learned.

## _I understood less than 10% of what you said, what do?_

Use `learned_svd_fast` — it gives the best practical quality for FP8 in this repo (best balance of fidelity and speed). `learned_svd` can match or slightly beat it if you crank SVD settings and iterations, and plain `learned` (AdaRound-style) can be useful if you want the original regularized per-weight optimization and are willing to tune lr/reg and spend more time.

### Why learned_svd_fast is usually best

- It targets the top principal error directions of each weight matrix (the parts that matter most for outputs), so the optimizer fixes the important mistakes first — this yields much lower degradation when converting to low-precision FP8.
- It implements the same SVD/PCA idea but with fast PCA (torch.pca_lowrank / pca_lowrank) and tuned defaults, so you get comparable (often indistinguishable) quality to a heavy SVD run but much faster.
- The README explicitly advertises the fast SVD script as the recommended / practical path for large models.

### Tradeoffs and when another variant might be better

- `learned_svd` (non-fast): uses more conservative/robust SVD settings (more iterations / different q), so it can sometimes slightly outperform the fast variant on a per-layer basis — at the cost of time and memory. Use it if you want to squeeze every last point of quality and can wait.
- `learned` (AdaRound-style): this is the classical learned-rounding approach with lr and regularization hyperparameters. It can be better on some layers if carefully tuned (reg_lambda, learning rate, num_iter), but it does not exploit the low-rank/PCA acceleration and is overall slower for large models and less targeted for FP8 principal-error correction.

### Practical tuning tips to improve quality

- top_k: learned_svd_fast exposes top_k — default 1. Increase to 2–4 for layers that have more than one important direction (higher quality, slower).
- num_iter: increasing iterations can improve convergence for hard layers (try 500→1000).
- calib_samples: more calibration samples for bias correction reduces bias drift.
- For learned (AdaRound): tune lr and reg_lambda if you pick that script.

### How to verify for your model

- Run learned_svd_fast first (fast + good default quality).
- If you need better quality, run the non-fast learned_svd with larger PCA iterations or increase top_k.
- If you want to be thorough, pick a few representative weight tensors and compare dequantized weight error (L2 norm on the dequantized weight printed by the script) or run a small downstream eval (e.g., sample images, compute FID or perceptual similarity, or compute model log-probs/perplexity) and compare.

### Example command

`python convert_fp8_scaled_learned_svd_fast.py --input model.safetensors --num_iter 500 --top_k 1 --calib_samples 3072`

## _Ok, but which arguments should I use for a Flux model?_

Here are practical recommendations tuned for a vision model like Flux (image-diffusion style). I’ll give a short recommended command for quick use, a higher-quality config if you want to push fidelity, a fast/cheap config for quick experiments, and clear tuning rules so you can improve only where needed.

### Recommended (balanced, default-quality — best starting point)

- Why: good balance of speed and fidelity for most Flux/Chroma-style models.
- Command example:
  `python convert_fp8_scaled_learned_svd_fast.py --input /path/to/model.safetensors --num_iter 500 --top_k 1 --calib_samples 3072`

### Higher-quality (slower, better fidelity)

- Why: keep more principal components and run more optimization to reduce error on hard layers (use if you see visual degradation / artifacts).
- Settings:
  `--num_iter 1000` (or 1500–2000 for very critical layers)
  `--top_k 2` (raise to 3–4 only if needed)
  `--calib_samples 8192`
- Command example:
  `python convert_fp8_scaled_learned_svd_fast.py --input /path/to/model.safetensors --num_iter 1000 --top_k 2 --calib_samples 8192`

### Fast/experimental (quick, lower cost)

- Why: for iterative testing or to preview results quickly.
- Settings:
  `--num_iter 200–300`
  `--top_k 1`
  `--calib_samples 1024`
- Command example:
  `python convert_fp8_scaled_learned_svd_fast.py --input /path/to/model.safetensors --num_iter 250 --top_k 1 --calib_samples 1024`

### Heuristics and tuning guidance (what to change and when)

- **top_k (principal components):**
  - Default 1 is often enough for attention / linear layers. Increase to 2–4 for large weight matrices or layers where singular values decay slowly (conv blocks or very wide linear layers). Increasing top_k raises cost roughly proportional to k.
- **num_iter (optimization iterations):**
  - 400–600 is a good default. Increase to 1000–2000 for stubborn layers that still show artifacts after default run. More iterations improve convergence but cost more time.
- **calib_samples (bias correction):**
  - 3072 is reasonable. Increase (e.g., 8192) if you notice bias shifts or subtle output drifts; larger values reduce variance in bias correction at the cost of more compute.
- **Selective tuning:**
  - Don’t blanket-increase everything. Run the balanced conversion and inspect outputs (visual samples). Only increase top_k/num_iter for layers causing problems.
  - Heuristic: for a given weight matrix, if its top singular value is not overwhelmingly dominant (i.e., energy spread across several singular values), increasing top_k helps. If you want, I can give a tiny script to compute singular value decay for layers so you can pick which tensors to treat more aggressively.
- **Time / memory expectations:**
  - README notes ~10 minutes for a large model with the fast script; higher-quality settings (top_k>1, num_iter 1000+) can increase runtime severalx. Ensure you have GPU memory and allow more time for high-quality runs.
- **Device:**
  - Run on a CUDA GPU if available — the converter defaults to CUDA if available and will be far faster.

### Practical tuning flow (recommended)

1. Run balanced defaults (num_iter 500, top_k 1, calib_samples 3072).
2. Generate a small set of validation samples (a few images) and compare visually to FP16/BF16 baseline.
3. If you see artifacts:
   a. Increase top_k to 2 for the whole conversion and re-run; retest.
   b. If still present, increase num_iter to 1000 (or target only problematic layers).
   c. If bias-like shifts appear, increase calib_samples.
4. If you want to be surgical: compute singular-value spectra for large tensors and increase top_k only for tensors with slow decay.

If you want, I can provide a small script that inspects the singular-value decay of each 2D weight tensor in your model and recommends a top_k based on how quickly the singular values drop off. This can help you decide if top_k=1 is sufficient or if you should increase it for certain layers.

## _An inspection script? Sounds smart, sure_

Below is a small Python snippet you can run to inspect singular-value decay for each 2-D weight tensor in a safetensors model. It prints per-tensor metrics (top singular values, explained-energy of top components, condition number) and a simple recommendation for top_k to use in learned_svd_fast based on how much energy the top components capture.

Copy/paste and run on a machine with PyTorch and safetensors installed.

[svd_inspector.py](svd_inspector.py)

### How to interpret the output and quick recommendations

- top1_frac (S[0]/sum(S_probed)):
  - If > ~0.7–0.8: the matrix is strongly dominated by the first component → top_k=1 is likely sufficient.
  - If ~0.4–0.7: energy spread across a few components → try top_k=2.
  - If < ~0.4: energy is spread out → try top_k=3–4 (or compute more singulars to be sure).
- recommended top_k: the script finds the smallest k (within probed k) capturing energy_threshold (default 90%). Use that as a guideline.
- condition number estimate: very large condition numbers imply slowly decaying spectrum; these layers often benefit from higher top_k.

### Notes and tips

- The snippet probes only up to probe-k singulars. If recommended top_k equals probe-k, consider increasing probe-k to get a better estimate.
- For very large matrices, pca_lowrank is used (fast) to get top singulars; for small matrices, full svd is used.
- Typical defaults for vision/Flux (from my earlier recommendations):
  - Most layers: top_k = 1
  - If this inspector shows many layers with top1_frac < 0.6, consider using top_k = 2 globally or selectively increase top_k for those layers.
- If you want, I can adapt the script to:
  - write a per-tensor JSON report,
  - auto-generate a list of tensors to re-run with higher top_k, or
  - run a selective learned_svd_fast pass only for flagged tensors.

Would you like the script to also output a JSON file listing tensors and recommended top_k so you can feed it into a selective conversion workflow?

## _Inspection script v2? Uh, I guess, why not_

**[svd_inspector_with_json.py](svd_inspector_with_json.py)**

This is a version of the inspector that writes a JSON report listing every 2D weight tensor it inspects, the probed top singular values, energy fractions (using sigma^2 so it's consistent with Frobenius energy), an estimated condition, and a recommended top_k (plus a probe_limited flag if the probe size may be too small). Save and run the script on a machine with PyTorch and safetensors; the default output is svd_report.json.

Run this on your Flux/Chroma model and then either (a) use the global recommended top_k heuristic from the JSON, or (b) pick a selective list of tensors flagged probe_limited=True (or with low top1_energy_frac) and re-run learned_svd_fast with higher top_k just for those tensors. If you want, I can modify the converter so it accepts a JSON "per-tensor top_k override" file and runs selective re-quantization automatically.

## _Inspection script v3? Didn't v2 do that?_

**[selective_requant_from_inspector.py](selective_requant_from_inspector.py)**

### What I provided and why

- I created a practical "glue" script that automates a selective re-quantization workflow:
  - You run the inspector to learn which tensors have slowly decaying spectra and thus may need top_k > 1.
  - You run the fast learned-SVD quantizer once (cheap baseline), then use this script to selectively re-quantize only the flagged tensors (so you spend extra time only where it matters).
- This avoids re-running the whole model with a higher global top_k and is a good speed/quality tradeoff for large vision models (Flux/Chroma).

### Notes, limitations and next steps

- The script expects to import LearnedRoundingConverter from convert_fp8_scaled_learned_svd_fast.py in the repo. If you prefer, I can instead paste a self-contained converter class into this script (but that duplicates repo code).
- If you want a version that rewrites the conversion script itself to accept a per-tensor override JSON directly (so you can run a single command instead of two-stage), I can prepare a patch or a modified converter file for you.
- If you want, I can also:
  - Produce a tiny driver that computes singular-value spectra and automatically chooses a small list of tensors to re-run (more aggressive filtering),
  - Or produce a one-shot converter that uses the JSON to pick per-tensor top_k during a single pass.
- If you'd like the single-command approach (converter accepts a per-tensor JSON), tell me and I will prepare a modified convert_fp8_scaled_learned_svd_fast.py patch you can drop into your repo (or I can give exact edits to apply).

## _Inspection script v4? You've lost me, do whatever you want_

**[one_shot_learned_svd_quant.py](one_shot_learned_svd_quant.py)**

### What I did

I packaged a single self-contained orchestration script above that:

- Runs a quick SVD/PCA inspection of each 2D weight to recommend a top_k per tensor.
- Performs a baseline learned-SVD quantization pass using the repository's LearnedRoundingConverter (so you must run this from the repo root where that file exists).
- Re-quantizes only tensors whose inspector recommendation is larger than the baseline (or all if you pass --force-all).
- Applies bias correction the same way the repo scripts do and writes a merged safetensors output.

### What's next

Save this file into the Learned-Rounding repo root, ensure you have torch and safetensors installed, and run it with your model path and chosen options.

### If you'd like, I can

- Produce a patch to modify `convert_fp8_scaled_learned_svd_fast.py` to accept a per-tensor JSON override and perform everything in a single pass (removing the need to import the converter externally).
- Or make the script self-contained (embed the converter implementation) so it doesn't need repository imports — tell me which you prefer and I'll prepare it.

###### _Oh god, please stop offering me alternatives_

---

**Disclaimer**:

- I have no idea what I am doing.
- Copilot might be hallucinating wildly and I'll still just follow its instructions blindly.
- Not only do I not guarantee anything, I haven't the slightest idea what it is I could guarantee to begin with.
- For working scripts backed by actual science, please use the ones in the original repository, whose dev actually seems to know stuff: [Clybius/Learned-Rounding](https://github.com/Clybius/Learned-Rounding)

---

**Original README below.**

## Learned-Rounding

A repository of Python &amp; PyTorch scripts which (currently) converts .safetensors models into scaled FP8 variants, utilizing gradient descent for optimal rounding.

### TPEC-Quant (Top-Principal Error Correction Quantization)

- A novel method (Designed by Clybius) which utilizes SVD to calculate error on the top principal component and descends upon more accurate representations, leading to far better results in FP8 precision. Results are very akin to FP16/BF16 precision when scaled with a single scalar value.
- Obtainable via `convert_fp8_scaled_learned_svd_fast.py` in this repository.
- Natively supported in ComfyUI and other UI utilities based off of ComfyUI, thanks to their scaled FP8 implementation!
- Takes ~10 minutes to quantize a large AI image diffusion model (Chroma). Performance may largely vary depending on hardware, and can likely be improved upon with multiprocessing (not yet complete), a faster drive for reading/writing (partially broke), and a faster GPU/processing unit.
- Supports Chroma, FLUX, T5XXL (includes removal of decoder and extra tensors from a full model), and maybe more!

Usage: `python convert_fp8_scaled_learned_svd_fast.py --input /path/to/model.safetensors`

Arguments:

- `--input "/path/to/model.safetensors"`: Input safetensors file path.
- `--output "/path/to/output_model.safetensors"`: Output safetensors file path. If not provided, it will be created based on the input location and name.
- `--keep_distillation`: Exclude distillation layers from quantization. (Likely not helpful because ComfyUI may use Round-to-Nearest in place of this without further modification.) (Default False)
- `--t5xxl`: Exclude certain layers for T5XXL model compatibility. (Default False)
- `--calib_samples INT`: Number of random samples for calibration. Currently only used for bias correction. (Default 3072)
- `--num_iter INT`: Number of optimization iterations per tensor. Increasing will result in higher accuracy, but longer offline quantization times. (Default 500)
- `--top_k INT`: Top K principal components to descend upon. We usually target the top component for ease of descent and accuracy, but you can experiment with more components if 1 isn't working well. (Default 1)
