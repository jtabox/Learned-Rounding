# FP8 Scaled Autoconverter Scripts with Automatic Parameter Optimization

Quantize **Flux**, **Chroma**, and **T5XXL** models to FP8 scaled format. Theoretically with optimized parameters, which are derived automatically by analyzing the incoming model and pick the best settings.

**All the original work the scripts are based on comes from: [Clybius/Learned-Rounding](https://github.com/Clybius/Learned-Rounding)**

⚠️ **Note**: _Highly experimental_ scripts (i.e. not sure how much optimized things are, if at all). Use at your own risk. But they do work and produce usable FP8 models!

## 👀 What even is this?

###### This started with me asking Copilot which of the 3 original scripts in [Clybius's repo](https://github.com/Clybius/Learned-Rounding) I should choose for specific models and why.

###### Copilot started explaining stuff but also suggested various optimizations on the way, which on one side meant nothing for me because my all my knowledge in this field is pretty much its name _(it's Machine Learning, yeah?)_, but on the other side stressed me enough to wonder _"maybe I'm missing out on earth-shattering speeds by not applying those optimizations?"_.

###### So I ran the discussion with GPT-5-Pro and with Sonnet 4.5 (hence two scripts) and asked them to do stuff for me while I watch YouTube, pretending I understand what they are doing and giving valuable feedback.

- There's currently two variants, one from each LLM.
- Both seem to be working and producing valid FP8 quants.
- So until I find time to evaluate them and see if there even is a substantial difference, you can try both and see which one you like more:
  - `autoconvert_fp8_sonnet45.py` - generated with Sonnet 4.5
  - `autoconvert_fp8_gpt5pro.py` - generated with GPT-5-Pro
- The LLMs were even kind enough to write each their own README, so check those out for usage instructions and details, depending on your choice:
  - `README.sonnet45.md`
  - `README.gpt5pro.md`

###### _Honestly, at this point it's a kinda "pick your poison" situation. The Sonnet variant is more fancy, faster and with more choices, but the GPT version seems smarter, takes longer and makes the GPU work more, so maybe it's better for quality? I haven't done a ~~proper~~ comparison yet and will probably never do, will instead ask the 2 models to fight it out with each other and then I'll just pick the winner._

## 📝 License

lol what license...

Though the TPEC-Quant method design and implementation are by [Clybius](https://github.com/Clybius/Learned-Rounding), so please check with them before you start copyrighting and monetizing stuff left and right, ok? It's their baby...

## 🤓 "I'd like to know more about..."

- TPEC-Quant? Ask [Clybius](https://github.com/Clybius) and check their [Learned-Rounding repo](https://github.com/Clybius/Learned-Rounding)
- Literally anything else? Idk, ask an LLM I guess. That's how I ended up with all of this.

## 🙏 Credits

- **[Clybius/Learned-Rounding](https://github.com/Clybius/Learned-Rounding)** - Original TPEC-Quant design and implementation
