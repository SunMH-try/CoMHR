import torch
from torch import nn
import torch.nn.functional as F

class PoseEncoder(nn.Module):
    def __init__(self, num_joints=18, input_dim=3, hidden_dim=1024, out_dim=2048):
        super().__init__()
        self.num_joints = num_joints
        self.input_dim = input_dim

        # 卷积特征提取
        self.conv1 = nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # 输出映射
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)  # -> [B, 3, J]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # -> [B, hidden_dim, J]

        # 使用 mask 进行加权平均池化
        if mask is not None:
            # mask: [B, J, 1] -> [B, 1, J]
            mask = mask.permute(0, 2, 1).float()
            weighted_sum = (x * mask).sum(dim=2)
            denom = mask.sum(dim=2).clamp(min=1e-6)
            x = weighted_sum / denom  # [B, hidden_dim]
        else:
            x = x.mean(dim=2)

        x = self.fc(x)
        return x
