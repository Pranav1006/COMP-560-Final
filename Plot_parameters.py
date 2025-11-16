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


first_layer = model.linear_relu_stack[0]
weights = first_layer.weight.data

num_units = 64
cols = 8
rows = num_units // cols


plt.figure(figsize=(12, 6))

for i in range(num_units):
    w = weights[i].reshape(28, 28)

    plt.subplot(rows, cols, i+1)
    plt.imshow(w, cmap="seismic")
    plt.axis("off")
    plt.title(f"Unit {i}", fontsize=8)

plt.tight_layout()
plt.show()