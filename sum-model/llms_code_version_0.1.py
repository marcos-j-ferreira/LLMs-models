import torch
import torch.nn as nn
import random

import torch.onnx

# This is model of sum
data = []
labels = []

for _ in range(1000):
    a = random.randint(0,100)
    b = random.randint(0,100)

    data.append([a,b])
    labels.append([a+b])

X = torch.tensor(data).float()
Y = torch.tensor(labels).float()


class Model_Sum(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural = nn.Linear(2, 1)
        self.active = nn.Identity() 

    def forward(self, x):
        x = self.neural(x)
        x = self.active(x) 
        return x

model = Model_Sum()

# Loss
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):
    model.train()

    y_pred = model(X)
    loss = criterion(y_pred, Y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | Loss: {loss.item()}')

with torch.no_grad():
    model.eval()

    y_pred = model(X)
    loss = criterion(y_pred, Y)
    print(f'Epoch: {epoch} | Loss: {loss.item()}')

# Test
test = torch.tensor([[1, 1]]).float()  
result = model(test)
print(result.item())


# ==== web ====

# exemplo de entrada (2 números)
dummy_input = torch.randn(1, 2)

torch.onnx.export(
    model,
    dummy_input,
    "model_sum.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    dynamo=False  
)

print("Modelo exportado para model_sum.onnx")
