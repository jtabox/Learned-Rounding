# Auto FP8 Scaled Converter with Automatic Parameter Optimization

Converts Flux, Chroma, or T5XXL models to FP8 format for **~50% size reduction** with minimal quality loss. Automatically analyzes the incoming model and picks the best settings, no complex parameters to set.

## 🎯 What

- **Quantizes** models from FP16/BF16 to FP8 (float8_e4m3fn)
- **Automatically optimizes** conversion parameters by analyzing your model's structure
- **Uses TPEC-Quant** (Top-Principal Error Correction) - a smart learned-rounding method that minimizes quality loss
- **Preserves quality** through SVD-based optimization and bias correction
- **ComfyUI ready** - outputs are directly compatible with ComfyUI and similar inference tools

## 👀 Why

- **Half file size**
- **Faster loading**
- **Lower VRAM usage**
- **Minimal quality loss**

## 🚀 How

### Requirements

```bash
pip install torch safetensors tqdm
```

Requires PyTorch 2.1+ with FP8 support.

### Basic Usage

**For Flux/Chroma models:**

```bash
python auto_fp8_convert.py --input flux_model.safetensors
```

**For T5XXL models:**

```bash
python auto_fp8_convert.py --input t5xxl_model.safetensors --t5xxl
```

Now chill. The script will:

1. 🔍 Analyze the model's structure
2. 🎯 Pick optimal parameters automatically
3. ⚙️ Convert using learned SVD quantization
4. 💾 Save a ComfyUI-ready FP8 model

## 📖 Usage Examples

### Simple (recommended)

```bash
# Auto-everything - just point it at your model
python auto_fp8_convert.py --input my_model.safetensors
```

### Quality Presets

```bash
# Fast conversion (good for testing)
python auto_fp8_convert.py --input model.safetensors --quality fast

# Balanced (default, recommended)
python auto_fp8_convert.py --input model.safetensors --quality balanced

# High quality (slower, best results)
python auto_fp8_convert.py --input model.safetensors --quality high
```

### Advanced Options

```bash
# Custom output path
python auto_fp8_convert.py --input model.safetensors --output custom_name.safetensors

# Manual parameter override
python auto_fp8_convert.py --input model.safetensors --top-k 2 --num-iter 1000

# Keep distillation layers unquantized
python auto_fp8_convert.py --input model.safetensors --keep-distillation
```

### Quality Presets Specifications

| Preset     | Iterations | Samples | Speed  | Quality   | Best For                 |
| ---------- | ---------- | ------- | ------ | --------- | ------------------------ |
| `fast`     | 250        | 1024    | ⚡⚡⚡ | Good      | Testing, iteration       |
| `balanced` | 500        | 3072    | ⚡⚡   | Great     | Most use cases (default) |
| `high`     | 1000       | 8192    | ⚡     | Excellent | Final conversions        |

## 🔧 How It Works

The script uses **TPEC-Quant** (Top-Principal Error Correction Quantization), a sophisticated method that:

1. **Analyzes** the singular value decomposition (SVD) of each weight matrix
2. **Identifies** the most important error directions (principal components)
3. **Optimizes** rounding decisions to minimize error along those directions
4. **Corrects** bias drift caused by quantization

This is much smarter than naive rounding and produces FP8 models that closely match FP16 quality.

### Automatic Parameter Detection

The script samples the incoming model's layers to determine:

- **top_k**: How many principal components to optimize (typically 1-4)
- **Complexity**: Whether layers have concentrated or spread-out singular values

If you want manual control, you can override with `--top-k` and `--num-iter`.

## 📊 Expected Results

- **File size**: ~50% of original
- **Quality**: Typically 95-99% of FP16 quality (perceptually very close)
- **Speed**: Conversion takes ~5-15 minutes for large models on GPU
- **ComfyUI**: Drop-in replacement, works immediately

## ✅ ComfyUI Compatibility

This script outputs models that are **100% compatible** with ComfyUI's scaled FP8 implementation. The converted models include:

- ✅ `scaled_fp8` marker tensor
- ✅ Per-layer `.scale_weight` tensors
- ✅ Proper `.scale_input` tensors for T5XXL
- ✅ Correct `float8_e4m3fn` dtype

Just load the converted model in ComfyUI like any other model!

## 🐛 Troubleshooting

**"PyTorch does not support torch.float8_e4m3fn"**

- Update to PyTorch 2.1 or newer: `pip install --upgrade torch`

**Out of memory during conversion**

- Use `--quality fast` to reduce memory usage
- Close other GPU applications
- For very large models, the script automatically uses CPU if needed

**Model doesn't load in ComfyUI**

- Ensure you used the correct `--t5xxl` flag if converting T5XXL
- Check that ComfyUI is updated to a version supporting scaled FP8

**Quality is lower than expected**

- Try `--quality high` for more optimization iterations
- For specific problematic layers, you can manually set `--top-k 2` or `--top-k 3`
