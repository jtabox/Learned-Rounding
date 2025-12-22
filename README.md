## FP8 Scaled Quantizer Script with Optional Automatic Parameter Optimization

A script that quantizes **Flux**/**Chroma** image diffusion models and **T5XXL** text encoder models to FP8 (e4m3fn) scaled variants using the TPEC-Quant method (*), with the option to automatically derive the (theoretically) optimized parameters by analyzing the original model.

(*) Based on the original work of **Clybius** ([@GitHub](https://github.com/Clybius), [@HuggingFace](https://huggingface.co/Clybius)): The TPEC-Quant method, a novel quantization method they designed that can be found here: **[Learned-Rounding](https://github.com/Clybius/Learned-Rounding)**.

⚠️ **Note**:
- The script in this repo is _highly experimental_, written by GPT-5 Pro, and with me lacking the necessary knowledge to effectively check for hallucinations or other defects (Gemini Pro 3 and Claude Opus 4.5 did the final checks).
- There are no guarantees that the original model analysis and the derived optimizations are correct (or even that they actually are optimizations). But it does work and produces usable FP8 models.
- For guaranteed results and correctness, please use the original quantizer script(s) from [Clybius's repo](https://github.com/Clybius/Learned-Rounding).

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
It'll detect and use CUDA if available, auto-detect if the model is Flux/Chroma or T5XXL, apply default parameters for good quality and speed (see *balanced* quality preset below) and save the result in the same folder as the input model with `_fp8_tpec_scaled` suffix.

---

If you want a bit more control, there are 3 quality "presets" you can specify:

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

###### *This started with me asking Copilot what the differences are between the 3 original scripts in [Clybius's repo](https://github.com/Clybius/Learned-Rounding). Copilot started happily explaining stuff, while also suggesting various possible optimized values and settings on the way.*

###### *This meant very little for me because my knowledge in this field is pretty much non-existent, but it did manage to send me into a FOMO trip, thinking that maybe I'm missing out on speed increases by not applying those optimizations.*

###### *So I asked GPT-5 Pro and Claude Sonnet 4.5 to write me a script that would analyze the original model and derive those optimizations automatically, because I wouldn't be able to do it manually. This is the result, checked by Gemini Pro 3 and Claude Opus 4.5*

## 📝 License

lol what

Though the TPEC-Quant method design and implementation are by [Clybius](https://github.com/Clybius/Learned-Rounding), so please check their repo for their license details.

## 🙏 Credits

- **[Clybius/Learned-Rounding](https://github.com/Clybius/Learned-Rounding)** - Original TPEC-Quant design and implementation
