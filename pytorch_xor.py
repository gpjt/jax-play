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


def calculate_loss(model, inputs, target):
    result = model(inputs)
    return ((result - target) ** 2).mean()


def main():
    torch.manual_seed(42)

    model = XORModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(10000):
        losses = []

        for x, y in data:
            optimizer.zero_grad()

            loss = calculate_loss(model, torch.tensor(x), torch.tensor(y))
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

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
