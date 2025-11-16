import torch
import matplotlib.pyplot as plt
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 classes
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# load the model
model = NeuralNetwork()
model.load_state_dict(torch.load("models/model1.pth"))
model.eval()


def visualize_digit_importance(model, target_digit):
    # Start from a blank image
    img = torch.zeros((1, 784), requires_grad=True)

    # Forward pass
    output = model(img)

    # Score for chosen digit
    score = output[0, target_digit]

    # Backprop: compute ∂score/∂pixels
    score.backward()

    # Get saliency and reshape to image
    saliency = img.grad.detach().reshape(28, 28)

    plt.imshow(saliency, cmap="hot")
    plt.title(f"Importance Map for Digit {target_digit}")
    plt.axis("off")
    plt.show()

for i in range(10):
    visualize_digit_importance(model, i)