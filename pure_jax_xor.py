import math

import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")


data = [
    ([0., 0.], [0]),
    ([0., 1.], [1]),
    ([1., 0.], [1]),
    ([1., 1.], [0]),
]


def generate_layer_parameters(key, d_in, d_out):
    weight_key, bias_key = jax.random.split(key)
    root_k = math.sqrt(1. / d_in)
    weights = (jax.random.uniform(weight_key, shape=(d_out, d_in)) * 2 * root_k) - root_k
    biases = (jax.random.uniform(bias_key, shape=(d_out,)) * 2 * root_k) - root_k
    return {
        "weights": weights,
        "biases": biases,
    }


@jax.jit
def forward(layers, inputs):
    x = inputs
    for layer in layers:
        x = jax.nn.sigmoid(
            x @ layer["weights"].T + layer["biases"]
        )
    return x


def step(layers, grad, learning_rate):
    layers = jax.tree.map(
        lambda p, g: p - g * learning_rate,
        layers,
        grad
    )
    return layers


@jax.jit
def calculate_loss(layers, inputs, target):
    result = forward(layers, inputs)
    return ((result - target) ** 2).mean()


def main():
    key = jax.random.key(42)

    layer_1_key, layer_2_key = jax.random.split(key)
    layers = [
        generate_layer_parameters(layer_1_key, 2, 2),
        generate_layer_parameters(layer_2_key, 2, 1),
    ]

    learning_rate = 0.1

    for epoch in range(10000):
        losses = []

        for x, y in data:

            loss, grad = jax.value_and_grad(calculate_loss)(layers, jnp.array(x), jnp.array(y))
            losses.append(loss.item())

            layers = step(layers, grad, learning_rate)

        if epoch % 100 == 0:
            avg_loss = sum(losses) / len(losses)
            print(f"Loss at epoch {epoch}: {avg_loss:.6f}")

    for x, y in data:
        result = forward(layers, jnp.array(x))
        print(f"{x=}: {result=}, {y=}")


if __name__ == "__main__":
    main()
