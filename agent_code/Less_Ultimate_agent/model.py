import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from .state_to_features import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class DQN(nn.Module):
    gamma = 0.999
    learning_rate = 0.0001

    def __init__(self):
        super(DQN, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
        )

        # Define fully connected output layer
        self.out = nn.Sequential(
            nn.Linear(1152, 6),
            nn.Softmax(dim=1),
        )

        # Set device and move the model to the device (GPU in this case)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = torch.device('cpu')
        self.to(self.device)

        # Define loss functions
        self.imit_loss = nn.CrossEntropyLoss()
        self.reinf_loss = nn.SmoothL1Loss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

    def imitation_train_step(self, old_state, action, rule_act):
        if action is not None and rule_act is not None:
            state_action_value = torch.FloatTensor(old_state[np.newaxis, :]).to(self.device)
            state_action_value = self.forward(state_action_value)

            target = torch.tensor(np.eye(6)[ACTIONS.index(rule_act)], device=self.device).unsqueeze(0)
            
            loss = self.imit_loss(state_action_value, target).to(self.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log the loss
            with open("loss_log.txt", "a") as loss_log:
                loss_log.write(str(loss.item()) + "\t")
            with open("loss_log_imit.txt", "a") as loss_log:
                loss_log.write(str(loss.item()) + "\t")
