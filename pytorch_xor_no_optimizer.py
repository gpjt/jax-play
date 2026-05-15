import torch


data = [
    ([0., 0.], [0]),
    ([0., 1.], [1]),
    ([1., 0.], [1]),
    ([1., 1.], [0]),
]


class XORModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 2, bias=True)
        self.layer1_activation = torch.nn.Sigmoid()
        self.layer2 = torch.nn.Linear(2, 1, bias=True)
        self.layer2_activation = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.layer1_activation(self.layer1(x))
        output = self.layer2_activation(self.layer2(hidden))
        return output


def calculate_loss(result, target):
    return ((result - target) ** 2).mean()


def step(model, learning_rate):
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= p.grad * learning_rate


def main():
    torch.manual_seed(42)

    model = XORModel()
    learning_rate = 0.1

    for epoch in range(10000):
        losses = []

        for x, y in data:
            model.zero_grad()

            result = model(torch.tensor(x))
            loss = calculate_loss(result, torch.tensor(y))
            loss.backward()
            losses.append(loss.item())

            step(model, learning_rate)

        if epoch % 100 == 0:
            avg_loss = sum(losses) / len(losses)
            print(f"Loss at epoch {epoch}: {avg_loss:.6f}")

    model.eval()
    with torch.no_grad():
        for x, y in data:
            result = model(torch.tensor(x))
            print(f"{x=}: {result=}, {y=}")


if __name__ == "__main__":
    main()
