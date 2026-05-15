import math

import torch


data = [
    ([0., 0.], [0]),
    ([0., 1.], [1]),
    ([1., 0.], [1]),
    ([1., 1.], [0]),
]


def generate_layer_parameters(d_in, d_out):
    root_k = math.sqrt(1. / d_in)
    weights = (torch.rand(d_out, d_in) * 2 * root_k) - root_k
    biases = (torch.rand(d_out) * 2 * root_k) - root_k
    return {
        "weights": weights.requires_grad_(),
        "biases": biases.requires_grad_(),
    }


def forward(layers, inputs):
    x = inputs
    for layer in layers:
        x = torch.sigmoid(
            x @ layer["weights"].T + layer["biases"]
        )
    return x


def zero_grad(layers):
    for layer in layers:
        for p in (layer["weights"], layer["biases"]):
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


def calculate_loss(result, target):
    return ((result - target) ** 2).mean()


def main():
    torch.manual_seed(42)

    layers = [
        generate_layer_parameters(2, 2),
        generate_layer_parameters(2, 1),
    ]

    learning_rate = 0.1

    for epoch in range(10000):
        losses = []

        for x, y in data:
            zero_grad(layers)

            result = forward(layers, torch.tensor(x))
            loss = calculate_loss(result, torch.tensor(y))
            loss.backward()
            losses.append(loss.item())

            with torch.no_grad():
                for layer in layers:
                    for p in (layer["weights"], layer["biases"]):
                        if p.grad is not None:
                            p -= p.grad * learning_rate

        if epoch % 100 == 0:
            avg_loss = sum(losses) / len(losses)
            print(f"Loss at epoch {epoch}: {avg_loss:.6f}")

    with torch.no_grad():
        for x, y in data:
            result = forward(layers, torch.tensor(x))
            print(f"{x=}: {result=}, {y=}")


if __name__ == "__main__":
    main()
