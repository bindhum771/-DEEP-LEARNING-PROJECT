import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64 * 6 * 6, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(x)
        out = out.view(-1, 64 * 6 * 6)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Create model instance
model = CNN()
# Normally you would train it; here we just save random initial weights
torch.save(model.state_dict(), 'model_weights.pth')
print("✅ Model weights saved to 'model_weights.pth'")