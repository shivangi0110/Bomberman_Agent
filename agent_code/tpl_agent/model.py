import torch.nn as nn
import torch
import torch.optim as optim
# from .callbacks import state_to_features, ACTIONS
# from .train import rule_based_act

class DQN(nn.Module):
    def __init__(self, dim_in, dim_out):
        
        # super(DQN, self).__init__()
        # self.model_sequence = nn.Sequential(
        #     nn.Linear(dim_in, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, dim_out),
        #     nn.Softmax(dim=0),
        # )

        super(DQN, self).__init__()
        self.model_sequence = nn.Sequential(
            nn.Linear(dim_in, 512),  # def 2048, 512, 15*15 is image size after conv
            nn.ReLU(),
            nn.Linear(512, dim_out),
            nn.Softmax(dim=0),
            )

    # def forward(self, x):
    #     return self.model_sequence(x)
    def forward(self, x):
        x = torch.tensor(x).float()
        return self.model_sequence(x).to('cpu')

    # def train_step(self, old_state_f, old_state):
    #     if rule_based_act(self, old_state) is not None:

    #         state_action_value = self.forward(old_state_f)

    #         target = torch.tensor(ACTIONS.index(rule_based_act(self, old_state)), dtype=torch.long).unsqueeze(0)

    #         loss = self.imitation_loss(state_action_value, target).to(self.device)

    #         with open("loss_log.txt", "a") as loss_log:
    #             loss_log.write(str(loss.item()) + "\t")

    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()