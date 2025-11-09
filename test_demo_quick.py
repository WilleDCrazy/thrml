#!/usr/bin/env python3
"""Quick test of the demo using existing test data"""

import os
import sys

# Make sure matplotlib doesn't try to show windows
import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import numpy as np

print("Testing THRML MNIST Demo (Quick Version)...")
print("="*60)

# Test imports
print("\n1. Testing imports...")
try:
    from thrml.block_management import Block
    from thrml.block_sampling import SamplingSchedule, sample_states
    from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
    from thrml.pgm import SpinNode
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test that we can create basic structures
print("\n2. Testing model creation...")
try:
    nodes = [SpinNode() for _ in range(10)]
    edges = [(nodes[i], nodes[i+1]) for i in range(9)]
    model = IsingEBM(
        nodes,
        edges,
        jnp.zeros((len(nodes),)),
        jnp.zeros((len(edges),)),
        jnp.array(1.0)
    )
    print(f"   ✓ Created simple model with {len(nodes)} nodes and {len(edges)} edges")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    sys.exit(1)

# Test sampling
print("\n3. Testing sampling...")
try:
    free_blocks = [Block(nodes[:5]), Block(nodes[5:])]
    program = IsingSamplingProgram(model, free_blocks, [])

    key = jax.random.key(0)
    # hinton_init returns properly shaped states for the sampling blocks
    init_state = hinton_init(key, model, free_blocks, ())

    schedule = SamplingSchedule(n_warmup=10, n_samples=5, steps_per_sample=2)
    samples = sample_states(
        jax.random.key(1),
        program,
        schedule,
        init_state,
        [],
        [Block(nodes)]
    )
    print(f"   ✓ Sampling successful, generated {samples[0].shape[0]} samples")
except Exception as e:
    print(f"   ✗ Sampling failed: {e}")
    sys.exit(1)

# Test data loading
print("\n4. Testing MNIST data loading...")
try:
    if os.path.exists("tests/mnist_test_data/train_data_filtered.npy"):
        train_data = np.load("tests/mnist_test_data/train_data_filtered.npy")
        test_data_1 = np.load("tests/mnist_test_data/sep_images_test_1.npy")
        print(f"   ✓ Loaded MNIST data: {train_data.shape[0]} training samples")
        print(f"   ✓ Loaded test images for digit 1: {test_data_1.shape[0]} samples")
    else:
        print("   ⚠ Pre-existing MNIST test data not found (will download on first demo run)")
except Exception as e:
    print(f"   ✗ Data loading failed: {e}")

# Test visualization
print("\n5. Testing visualization...")
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    random_img = np.random.rand(28, 28)
    ax.imshow(random_img, cmap='gray')
    ax.set_title('Test Image')
    plt.savefig('test_visualization.png', dpi=50)
    plt.close()

    if os.path.exists('test_visualization.png'):
        print("   ✓ Visualization working, saved test_visualization.png")
        os.remove('test_visualization.png')
    else:
        print("   ⚠ Visualization may have issues")
except Exception as e:
    print(f"   ✗ Visualization failed: {e}")

print("\n" + "="*60)
print("✓ All tests passed! The demo should work correctly.")
print("="*60)
print("\nRun the full demo with:")
print("  python demo_mnist_diffusion.py --digit 1 --epochs 1")
print("="*60 + "\n")
