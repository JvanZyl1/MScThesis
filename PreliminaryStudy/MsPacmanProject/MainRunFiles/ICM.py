import torch
import torch.nn as nn

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, feature_extractor, num_actions, input_shape, device):
        super(IntrinsicCuriosityModule, self).__init__()
        self.device = device

        # Output size of feature extractor
        self.feature_extractor = feature_extractor.to(device)
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape).to(device)
            feat_extrac_out = self.feature_extractor(dummy_input)
            fc1_out_features = feat_extrac_out.size(1)

        # Inverse model
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * fc1_out_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        ).to(device)

        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(fc1_out_features + num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, fc1_out_features)
        ).to(device)

    def forward(self, state, next_state, action):
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        
        # Features
        phi = self.feature_extractor(state)
        phi_next = self.feature_extractor(next_state)

        # Inverse model
        inverse_input = torch.cat((phi, phi_next), dim=1)
        action_pred = self.inverse_model(inverse_input)

        # Forward model
        action_one_hot_encoded = nn.functional.one_hot(action, num_classes=action_pred.size(1)).float().to(self.device)
        action_one_hot_encoded = action_one_hot_encoded.view(action_one_hot_encoded.size(0), -1)
        forward_model_input = torch.cat((phi_next, action_one_hot_encoded), dim=1)
        predicted_next_state = self.forward_model(forward_model_input)

        return action_pred, predicted_next_state, phi_next
