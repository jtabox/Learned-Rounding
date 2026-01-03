## FP8 Scaled Quantizer Script with Optional Automatic Parameter Optimization

A script that quantizes **Flux**/**Chroma** image diffusion models and **T5XXL** text encoder models to FP8 (e4m3fn) scaled variants. Uses the TPEC-Quant method (*) applying (theoretically) optimized parameters calculated by analyzing the original model, while allowing manual overrides for customized tuning.

🔍 **Features**:

- Quality, top_k strategy, caps, and calibration size parameters are calculated based on model hardness and available VRAM.
- Available quality presets, T5 auto-detection, keep-distillation option, and ComfyUI-compatible quantized models.
- OOM resilience: automatic retry with lighter settings if CUDA runs out of memory.
- Per-layer or global auto top_k via SVD energy probe.
- No-grad, clamped updates inside the converter for stability and speed.
- Bias correction with on-device calibration inputs.

(*) The TPEC-Quant method is a novel quantization method designed by **Clybius** ([@GitHub](https://github.com/Clybius), [@HuggingFace](https://huggingface.co/Clybius)) that results in a better quantization precision in the produced FP8 models. More details and a quantization script can be found here: **[Learned-Rounding](https://github.com/Clybius/Learned-Rounding)**.

⚠️ **Note**:

- The script in this repo is _highly experimental_, written by GPT-5 Pro, and with me lacking the necessary knowledge to effectively check for hallucinations or other defects (Gemini Pro 3 and Claude Opus 4.5 did that for me though).
- There are no guarantees that the original model analysis and the derived optimizations are correct (or even that they actually are optimizations). But the script actually does work and produces usable FP8 models.
- For guaranteed, scientifically backed results and correctness, please use the original quantizer script(s) from [Clybius's repo](https://github.com/Clybius/Learned-Rounding).

## ⚡ Quick start

1. **Install**

Clone this repo and pip install the requirements:

```bash
git clone https://github.com/jtabox/Learned-Rounding
cd learned-rounding
pip install -r requirements.txt
```

2. **Run**

The most basic usage requires only the path to the model to convert:

```bash
python fp8_tpec_conv.py --input /path/to/model.safetensors
```

It will automatically analyze the model and derive optimized parameters for the conversion.

---

If you want to choose between quantization speed and quality, there are 3 quality "presets" you can specify:

```bash
# Fast conversion (good for testing)
python fp8_tpec_conv.py --input model.safetensors --quality fast

# Balanced (default, recommended)
python fp8_tpec_conv.py --input model.safetensors --quality balanced

# High quality (slower, best results)
python fp8_tpec_conv.py --input model.safetensors --quality high
```

Those presets use the following specifications:

| Preset     | Iterations | Samples | Speed | Quality | Best For                 |
| ---------- | ---------- | ------- | ----- | ------- | ------------------------ |
| `fast`     | 250        | 1024    | ⚡⚡⚡   | ➕       | Testing, iteration       |
| `balanced` | 500        | 3072    | ⚡⚡    | ➕➕      | Most use cases (default) |
| `high`     | 1000       | 8192    | ⚡     | ➕➕➕     | Final conversions        |

---

For even more control, most parameters can be manually overriden and some extra specified. Check the script's help:

```bash
python fp8_tpec_conv.py --help
```

## 🎯 Tuning guide

For a detailed overview of the various parameters and how to tune them for different model types, as well as multiple detailed, ready to use examples, check the [TPEC Tuning Guide](tpec_tuning_guide.md).

## 🤓 "I'd like to know more about..."

- TPEC-Quant? Ask [Clybius](https://github.com/Clybius) and check their [Learned-Rounding repo](https://github.com/Clybius/Learned-Rounding)
- Literally anything else? Idk, ask an LLM I guess.

## 👀 What even is this script?

###### _This started with me asking Copilot what the differences are between the 3 scripts in [Clybius's repo](https://github.com/Clybius/Learned-Rounding). Copilot started happily explaining stuff, while also suggesting various possible optimized values and settings on the way._

###### _These suggestions meant very little for me because my knowledge in this field is pretty much non-existent, but they did manage to send me into a FOMO trip, thinking that maybe "I'm missing out on speed increases by not applying these optimizations"._

###### _So I asked GPT-5 Pro and Claude Sonnet 4.5 to write me a script that would analyze the original model and derive those optimizations automatically. After some iterations and a final check by Gemini Pro 3 and Claude Opus 4.5, this is the result._

## 📝 License

lol what

Though the TPEC-Quant method is designed and implemented by [Clybius](https://github.com/Clybius/Learned-Rounding), so please check with them or their repo for any licensing queries.

## 🙏 Credits

- **[Clybius/Learned-Rounding](https://github.com/Clybius/Learned-Rounding)** - Original TPEC-Quant design and implementation
