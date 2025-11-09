# THRML MNIST Diffusion Demo

A complete working demo showing how to train a diffusion-like Energy-Based Model on MNIST digits using THRML, and generate specific digits from pure random noise.

## What This Demo Does

1. **Trains** an energy-based model on MNIST digits using THRML's block Gibbs sampling
2. **Generates** specific digits (0-9) from pure random noise
3. **Visualizes** the generation process step-by-step
4. **Demonstrates** how THRML uses thermodynamic sampling to generate images

## Quick Start (Easiest Way)

### Linux/Mac:
```bash
bash setup_demo.sh
source venv/bin/activate
python demo_mnist_diffusion.py --digit 1 --epochs 3
```

### Windows:
```cmd
setup_demo.bat
venv\Scripts\activate.bat
python demo_mnist_diffusion.py --digit 1 --epochs 3
```

## Manual Installation

If you prefer to install manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat

# Install THRML
pip install -e .

# Install demo requirements
pip install -r demo_requirements.txt

# Run the demo
python demo_mnist_diffusion.py --digit 1 --epochs 3
```

## Usage

### Basic Usage
Generate a specific digit (e.g., the digit "1"):
```bash
python demo_mnist_diffusion.py --digit 1 --epochs 3
```

### Quick Test (1 epoch, faster):
```bash
python demo_mnist_diffusion.py --digit 1 --epochs 1
```

### Generate Different Digits:
```bash
python demo_mnist_diffusion.py --digit 0 --epochs 3
python demo_mnist_diffusion.py --digit 5 --epochs 3
python demo_mnist_diffusion.py --digit 9 --epochs 3
```

### All Options:
```bash
python demo_mnist_diffusion.py --help
```

Options:
- `--digit N`: Target digit to generate (0-9), default: 1
- `--epochs N`: Number of training epochs, default: 3
- `--batch-size N`: Batch size for training, default: 50

## Output

The demo generates two images:

1. **`generation_steps.png`** - Shows 10 steps of the generation process from pure noise to the target digit
2. **`final_digit_N.png`** - The final generated digit

## What You'll See

```
==========================================
THRML MNIST Diffusion Demo - Training on Digit 1
==========================================

Downloading MNIST dataset...
Downloaded 70000 images
Found 7877 images of digit 1
Training data shape: (1000, 784)
Test data shape: (100, 784)

Model architecture:
  - Total nodes: 1800
  - Total edges: 18000
  - Visible nodes: 784

Training for 3 epochs...

Epoch 1/3
  Batch 5/20, Loss: 0.0234
  Batch 10/20, Loss: 0.0198
  ...

==========================================
Generating Digit 1 from Pure Noise
==========================================

Starting from pure random noise...
Running Gibbs sampling to generate image...
Saved generation visualization to generation_steps.png
Saved final image to final_digit_1.png

==========================================
Demo Complete!
==========================================
```

## How It Works

THRML uses **block Gibbs sampling** on energy-based models to generate images:

1. **Model**: A two-layer Ising-like spin model with long-range connections
2. **Training**: Uses contrastive divergence to learn the energy function from MNIST data
3. **Generation**: Starts from pure random noise and uses Gibbs sampling to evolve towards samples from the learned distribution
4. **Result**: The model generates realistic digits through thermodynamic sampling!

This demonstrates THRML's core capability: efficient probabilistic sampling on graphical models, which is exactly what Extropic's future hardware will accelerate.

## Requirements

- Python 3.10+
- JAX (CPU version included, GPU optional)
- See `demo_requirements.txt` for full list

## Tips

- **Fewer epochs** (1-2): Faster, less refined output
- **More epochs** (5-10): Slower, better quality output
- **Different digits**: Try all digits 0-9 to see different generation patterns
- **Batch size**: Larger = faster but more memory

## Troubleshooting

**No MNIST data found?**
- The script automatically downloads MNIST on first run
- If download fails, it will try to use pre-existing test data in `tests/mnist_test_data/`

**Out of memory?**
- Reduce `--batch-size` (try 25 or 10)
- Use fewer training epochs

**Slow training?**
- This is normal on CPU
- For faster training, install JAX with CUDA support (GPU)
- Or use `--epochs 1` for a quick demo

## Next Steps

After running the demo, explore:

1. **THRML Examples**: Check out `examples/` directory for more complex models
2. **Documentation**: Visit [docs.thrml.ai](https://docs.thrml.ai)
3. **Custom Models**: Modify `demo_mnist_diffusion.py` to experiment with:
   - Different network architectures
   - Other datasets
   - Custom energy functions

## Citation

If you use THRML in your research, please cite:

```bibtex
@misc{jelinčič2025efficientprobabilistichardwarearchitecture,
      title={An efficient probabilistic hardware architecture for diffusion-like models},
      author={Andraž Jelinčič and Owen Lockwood and Akhil Garlapati and Guillaume Verdon and Trevor McCourt},
      year={2025},
      eprint={2510.23972},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.23972},
}
```
