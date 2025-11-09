#!/usr/bin/env python3
"""
THRML MNIST Diffusion Demo

This demo trains a diffusion-like Energy-Based Model on MNIST digits using THRML's
block Gibbs sampling, then generates specific digits from pure noise.

Usage:
    python demo_mnist_diffusion.py --digit 1 --epochs 5
"""

import argparse
import os
from typing import Sequence, Type

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Key

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import (
    Edge,
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
    estimate_kl_grad,
    hinton_init,
)
from thrml.pgm import AbstractNode, SpinNode


def download_mnist_data(data_dir="demo_data"):
    """Download and prepare MNIST data for training."""
    os.makedirs(data_dir, exist_ok=True)

    print("Downloading MNIST dataset...")
    try:
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.to_numpy() if hasattr(mnist.data, 'to_numpy') else np.array(mnist.data)
        y = mnist.target.to_numpy() if hasattr(mnist.target, 'to_numpy') else np.array(mnist.target)
        y = y.astype(int)

        # Normalize to [0, 1] and then binarize
        X = (X / 255.0) > 0.5

        print(f"Downloaded {len(X)} images")
        return X, y
    except Exception as e:
        print(f"Error downloading MNIST: {e}")
        print("Trying to use pre-existing test data...")
        return None, None


def get_double_grid(
    side_len: int,
    jumps: Sequence[int],
    n_visible: int,
    node: Type[AbstractNode],
    key: Key[Array, ""],
) -> tuple[Block, Block, Block, Block, list[AbstractNode], list[Edge]]:
    """Create a double-layer grid structure for the EBM."""
    size = side_len**2
    assert n_visible <= size

    def get_idx(i, j):
        i = (i + side_len) % side_len
        j = (j + side_len) % side_len
        return i * side_len + j

    def get_coords(idx):
        return idx // side_len, idx % side_len

    def _make_edge(idx, di, dj):
        i, j = get_coords(idx)
        return jnp.array([idx, get_idx(i + di, j + dj)])

    make_edge = jax.jit(jax.vmap(_make_edge, in_axes=(0, None, None), out_axes=0))

    indices = jnp.arange(size)
    edges_arr = jnp.stack([indices, indices], axis=1)
    for d in jumps:
        left_edges = make_edge(indices, -d, 0)
        right_edges = make_edge(indices, d, 0)
        upper_edges = make_edge(indices, 0, -d)
        lower_edges = make_edge(indices, 0, d)
        edges_arr = jnp.concatenate([edges_arr, left_edges, right_edges, upper_edges, lower_edges], axis=0)

    deg = 4 * len(jumps) + 1
    total_edges = size * deg

    nodes_upper = [node() for _ in range(size)]
    nodes_lower = [node() for _ in range(size)]
    all_nodes = nodes_upper + nodes_lower
    all_edges = [(nodes_upper[i], nodes_lower[j]) for i, j in edges_arr]

    visible_indices = jax.random.permutation(key, jnp.arange(size))[:n_visible]
    visible_nodes = [nodes_upper[i] for i in visible_indices]
    upper_without_visible = [node for node in nodes_upper if node not in visible_nodes]

    return (
        Block(nodes_upper),
        Block(nodes_lower),
        Block(visible_nodes),
        Block(upper_without_visible),
        all_nodes,
        all_edges,
    )


def prepare_training_data(target_digit=1, n_train=1000, n_test=100):
    """Prepare MNIST training and test data for a specific digit."""
    # First try to use pre-existing test data (faster)
    test_data_path = "tests/mnist_test_data"
    if os.path.exists(test_data_path):
        try:
            print("Using pre-existing test data...")
            train_data = np.load(f"{test_data_path}/train_data_filtered.npy")[:n_train]

            # The test data may include label information - we only want the image data (first 784 dimensions)
            if train_data.shape[1] > 784:
                print(f"  Extracting image data (first 784 dims) from {train_data.shape[1]}-dim data")
                train_data = train_data[:, :784]

            # Try to load test images for the specific digit
            test_file = f"{test_data_path}/sep_images_test_{target_digit}.npy"
            if os.path.exists(test_file):
                test_images = np.load(test_file)[:n_test]
                if test_images.shape[1] > 784:
                    test_images = test_images[:, :784]
            else:
                print(f"  No test data for digit {target_digit}, using training data for testing")
                test_images = train_data[:n_test]

            print(f"  Loaded {train_data.shape[0]} training samples with shape {train_data.shape}")
            return jnp.array(train_data), jnp.array(test_images)
        except Exception as e:
            print(f"  Could not load pre-existing data: {e}")

    # Fallback to downloading
    X, y = download_mnist_data()

    if X is None:
        raise FileNotFoundError("No MNIST data found. Please run with internet connection first.")

    # Filter for target digit
    digit_mask = y == target_digit
    digit_images = X[digit_mask]

    print(f"Found {len(digit_images)} images of digit {target_digit}")

    # Split into train and test
    n_total = min(len(digit_images), n_train + n_test)
    train_images = digit_images[:n_train]
    test_images = digit_images[n_train:n_train + n_test]

    # Reshape to 28x28 flat
    train_data = train_images.reshape(-1, 28*28)
    test_data = test_images.reshape(-1, 28*28)

    return jnp.array(train_data), jnp.array(test_data)


def visualize_generation_steps(model, init_state, program, generation_blocks, target_digit, save_path="generation_steps.png"):
    """Visualize the generation process from noise to digit."""
    print(f"\nGenerating digit {target_digit} from pure noise...")

    # Create a longer sampling schedule to capture intermediate steps
    gen_schedule = SamplingSchedule(n_warmup=0, n_samples=100, steps_per_sample=5)

    key = jax.random.key(42)

    # Sample and get all intermediate states
    # Get all nodes from the generation blocks to observe
    all_nodes_block = Block([node for block in generation_blocks for node in block.nodes])
    all_samples = sample_states(
        key, program, gen_schedule, init_state, [], [all_nodes_block]
    )

    # The samples contain all nodes (1800 total), but we only want the first 784 (the image pixels)
    # Shape of all_samples[0] is (n_samples, n_nodes) since we used shape=() in hinton_init
    image_samples = all_samples[0][:, :784]  # [n_samples, 784]

    # Select specific timesteps to visualize
    n_steps = 10
    indices = np.linspace(0, image_samples.shape[0] - 1, n_steps, dtype=int)

    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Generating Digit {target_digit} from Pure Noise (THRML Gibbs Sampling)', fontsize=16)

    for idx, ax in enumerate(axes.flat):
        if idx < len(indices):
            # Get the image at this timestep
            sample_idx = indices[idx]
            img_data = image_samples[sample_idx].reshape(28, 28)

            ax.imshow(img_data, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Step {sample_idx}')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved generation visualization to {save_path}")

    # Also show the final generated image
    final_img = image_samples[-1].reshape(28, 28)

    plt.figure(figsize=(6, 6))
    plt.imshow(final_img, cmap='gray')
    plt.title(f'Final Generated Digit {target_digit}')
    plt.axis('off')
    plt.savefig(f'final_digit_{target_digit}.png', dpi=150, bbox_inches='tight')
    print(f"Saved final image to final_digit_{target_digit}.png")

    return final_img


def train_mnist_ebm(target_digit=1, n_epochs=3, batch_size=50):
    """Train an energy-based model on MNIST using THRML."""
    print(f"\n{'='*60}")
    print(f"THRML MNIST Diffusion Demo - Training on Digit {target_digit}")
    print(f"{'='*60}\n")

    # Prepare data
    train_data, test_data = prepare_training_data(target_digit=target_digit, n_train=1000, n_test=100)
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Create model architecture
    data_dim = 28 * 28
    (upper_grid, lower_grid, visible_nodes, upper_without_visible, all_nodes, all_edges) = get_double_grid(
        30, [1, 3, 9], data_dim, SpinNode, jax.random.key(0)
    )

    print(f"\nModel architecture:")
    print(f"  - Total nodes: {len(all_nodes)}")
    print(f"  - Total edges: {len(all_edges)}")
    print(f"  - Visible nodes: {len(visible_nodes.nodes)}")

    # Initialize model
    model = IsingEBM(
        all_nodes,
        all_edges,
        jnp.zeros((len(all_nodes),), dtype=float),
        jnp.zeros((len(all_edges),), dtype=float),
        jnp.array(1.0),
    )

    positive_sampling_blocks = [upper_without_visible, lower_grid]
    negative_sampling_blocks = [upper_grid, lower_grid]
    training_data_blocks = [visible_nodes]

    schedule_negative = SamplingSchedule(100, 20, 3)
    schedule_positive = SamplingSchedule(100, 10, 5)

    # Setup optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init((model.weights, model.biases))

    print(f"\nTraining for {n_epochs} epochs...")

    # Training loop (using JAX scan for efficiency like in test)
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        # Prepare batched data
        key_epoch = jax.random.key(epoch)
        key_shuffle = jax.random.split(key_epoch)[0]
        idxs = jax.random.permutation(key_shuffle, jnp.arange(len(train_data)))
        shuffled_data = train_data[idxs]

        n_batches = len(train_data) // batch_size
        tot_len = n_batches * batch_size
        batched_data = jnp.reshape(shuffled_data[:tot_len], (n_batches, batch_size, data_dim)).astype(jnp.bool_)

        def body_fun(carry, key_and_data):
            _key, _data = key_and_data
            _opt_state, _params = carry
            _model = eqx.tree_at(lambda m: (m.weights, m.biases), model, _params)

            key_train, key_init_pos, key_init_neg = jax.random.split(_key, 3)
            vals_free_pos = hinton_init(key_init_pos, _model, positive_sampling_blocks, (1, batch_size))
            vals_free_neg = hinton_init(key_init_neg, _model, negative_sampling_blocks, (batch_size,))

            ebm_spec = IsingTrainingSpec(
                _model,
                training_data_blocks,
                [],
                positive_sampling_blocks,
                negative_sampling_blocks,
                schedule_positive,
                schedule_negative,
            )

            grad_w, grad_b, _, _ = estimate_kl_grad(
                key_train, ebm_spec, _model.nodes, _model.edges, [_data], [], vals_free_pos, vals_free_neg
            )

            grads = (grad_w, grad_b)
            with jax.numpy_dtype_promotion("standard"):
                updates, _opt_state = optimizer.update(grads, _opt_state, _params)

            _weights, _biases = _params
            _weights += updates[0]
            _biases += updates[1]

            new_carry = _opt_state, (_weights, _biases)
            return new_carry, jnp.mean(jnp.abs(grad_w))

        params = model.weights, model.biases
        init_carry = opt_state, params

        keys = jax.random.split(jax.random.key(epoch), n_batches)
        out_carry, losses = jax.lax.scan(body_fun, init_carry, (keys, batched_data))

        opt_state, params = out_carry
        model = eqx.tree_at(lambda m: (m.weights, m.biases), model, params)

        avg_loss = float(jnp.mean(jnp.array(losses)))
        print(f"  Average Loss: {avg_loss:.4f}")

    print("\nTraining complete!")
    return model, upper_grid, lower_grid


def generate_from_noise(model, upper_grid, lower_grid, target_digit=1):
    """Generate a digit from pure random noise using trained model."""
    print(f"\n{'='*60}")
    print(f"Generating Digit {target_digit} from Pure Noise")
    print(f"{'='*60}\n")

    # Create sampling program for generation
    generation_blocks = [upper_grid, lower_grid]
    program = IsingSamplingProgram(model, generation_blocks, [])

    # Initialize from random noise (single sample, no batch dimension needed for inference)
    key_init = jax.random.key(123)
    init_state = hinton_init(key_init, model, generation_blocks, ())

    print("Starting from pure random noise...")
    print("Running Gibbs sampling to generate image...")

    # Visualize the generation process
    final_img = visualize_generation_steps(model, init_state, program, generation_blocks, target_digit)

    return final_img


def main():
    parser = argparse.ArgumentParser(description='THRML MNIST Diffusion Demo')
    parser.add_argument('--digit', type=int, default=1, help='Target digit to generate (0-9)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for training')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("THRML: Thermodynamic HypergRaphical Model Library")
    print("Energy-Based Model Demo on MNIST")
    print("="*60 + "\n")

    # Train the model
    model, upper_grid, lower_grid = train_mnist_ebm(
        target_digit=args.digit,
        n_epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Generate from noise
    generated_img = generate_from_noise(model, upper_grid, lower_grid, target_digit=args.digit)

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print(f"\nGenerated images saved:")
    print(f"  - generation_steps.png (shows the generation process)")
    print(f"  - final_digit_{args.digit}.png (final generated digit)")
    print("\nTHRML uses block Gibbs sampling on energy-based models to")
    print("generate images by sampling from the learned distribution!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
