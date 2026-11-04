import torch

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

print("Input shape:", X.shape)
print("Target shape:", Y.shape)

print("Inputs:")
print(X)

print("Targets:")
print(Y)

torch.manual_seed(42)
model = torch.nn.Linear(1, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Initial weight:", model.weight.item())
print("Initial bias:", model.bias.item())

predictions = model(X)

print("Predictions:")
print(predictions)
print("Predictions shape:", predictions.shape)

loss_fn = torch.nn.MSELoss()
loss = loss_fn(predictions, Y)

print("Loss:", loss.item())

loss.backward()

print("Weight gradient:", model.weight.grad)
print("Bias gradient:", model.bias.grad)

optimizer.step()

print("Updated weight:", model.weight.item())
print("Updated bias:", model.bias.item())

new_predictions = model(X)
new_loss = loss_fn(new_predictions, Y)
print("New loss:", new_loss.item())

losses = []

for epoch in range(5000):
    optimizer.zero_grad()
    
    predictions = model(X)
    loss = loss_fn(predictions, Y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}: loss = {loss.item():.6f}")

    print("Learned weight:", model.weight.item())
    print("Learned bias:", model.bias.item())


test_input = torch.tensor([[5.0]])

with torch.no_grad():
    test_prediction = model(test_input)

print("Prediction for x = 5:", test_prediction.item())
print("Expected value:", 11.0)
print("Difference:", abs(test_prediction.item() - 11.0))