# -------------------- 完整代码：mg3_final.py --------------------
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import StandardScaler


# -------------------- 配置参数 --------------------
class Config:
    # 数据路径
    data_root = "D:/JJ/pd"
    video_dir = os.path.join(data_root, "sp")
    sensor_dir = os.path.join(data_root, "bg")
    save_dir = os.path.join(data_root, "models")

    # 模型参数
    seq_length = 10
    frame_feat_dim = 128
    sensor_feat_dim = 64
    hidden_dim = 256
    dropout = 0.3
    pretrained = True

    # 训练参数
    batch_size = 4
    lr = 1e-4
    epochs = 3
    train_ratio = 0.8
    random_seed = 42

    # 传感器列配置
    sensor_columns = {
        "feature_cols": ["temp","wifi","humi","noise", "pm25","yiwei" ],
        "target_col": "value"
    }


# -------------------- 数据集类（多进程安全版本）-------------------
class MultimodalDataset(Dataset):
    def __init__(self, mode='train', sensor_scaler=None, target_scaler=None):
        self.mode = mode
        self.seq_length = Config.seq_length
        self.video_folders = []
        self.sensor_files = []
        self.index_map = []

        # 标准化器处理
        self.sensor_scaler = sensor_scaler
        self.target_scaler = target_scaler

        self._prepare_data()
        self._build_index_map()
        self._fit_scalers()  # 关键修改点：集中处理scaler初始化
        self.transform = self._get_transform()

    def _prepare_data(self):
        """加载所有有效数据路径"""
        for vid_dir in os.listdir(Config.video_dir):
            frame_dir = os.path.join(Config.video_dir, vid_dir)
            sensor_path = os.path.join(Config.sensor_dir, f"{vid_dir}.csv")

            if os.path.exists(sensor_path) and len(os.listdir(frame_dir)) >= Config.seq_length:
                self.video_folders.append(frame_dir)
                self.sensor_files.append(sensor_path)

    def _build_index_map(self):
        """构建训练/验证索引映射"""
        np.random.seed(Config.random_seed)
        self.index_map = []

        for vid_idx in range(len(self.video_folders)):
            frame_count = len(os.listdir(self.video_folders[vid_idx]))
            sensor_df = pd.read_csv(self.sensor_files[vid_idx])
            valid_length = min(frame_count, len(sensor_df)) - self.seq_length

            if valid_length <= 0:
                continue

            indices = np.arange(valid_length)
            np.random.shuffle(indices)
            split = int(Config.train_ratio * valid_length)

            if self.mode == 'train':
                self.index_map += [(vid_idx, i) for i in indices[:split]]
            else:
                self.index_map += [(vid_idx, i) for i in indices[split:]]

    def _fit_scalers(self):
        """统一处理标准化器拟合"""
        if self.mode == 'train' and self.sensor_scaler is None:
            # 合并所有训练数据拟合scaler
            all_features = []
            all_targets = []
            for sensor_path in self.sensor_files:
                df = pd.read_csv(sensor_path)
                features = df[Config.sensor_columns["feature_cols"]].values
                targets = df[Config.sensor_columns["target_col"]].values.reshape(-1, 1)
                all_features.append(features)
                all_targets.append(targets)

            self.sensor_scaler = StandardScaler().fit(np.vstack(all_features))
            self.target_scaler = StandardScaler().fit(np.vstack(all_targets))

    def _get_transform(self):
        """图像预处理"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vid_idx, start_idx = self.index_map[idx]

        # 加载传感器数据
        sensor_df = pd.read_csv(self.sensor_files[vid_idx])
        features = sensor_df[Config.sensor_columns["feature_cols"]].values
        targets = sensor_df[Config.sensor_columns["target_col"]].values.reshape(-1, 1)

        # 标准化处理（不再包含fit逻辑）
        features = self.sensor_scaler.transform(features)
        targets = self.target_scaler.transform(targets)

        # 截取窗口
        sensor_window = features[start_idx:start_idx + self.seq_length]
        eda_targets = targets[start_idx:start_idx + self.seq_length].flatten()

        # 加载图像序列
        frame_dir = self.video_folders[vid_idx]
        frame_files = sorted(os.listdir(frame_dir),
                             key=lambda x: int(x.split('_')[-1].split('.')[0]))

        frames = []
        for i in range(start_idx, start_idx + self.seq_length):
            img_path = os.path.join(frame_dir, frame_files[i])
            img = Image.open(img_path).convert('RGB')
            frames.append(self.transform(img))

        return {
            'frames': torch.stack(frames),  # [T, C, H, W]
            'sensor': torch.FloatTensor(sensor_window),  # [T, 5]
            'target': torch.FloatTensor(eda_targets)  # [T]
        }


# -------------------- 模型架构 --------------------
class Seq2SeqRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # 图像特征提取
        self.frame_encoder = models.resnet18(pretrained=Config.pretrained)
        self.frame_encoder.fc = nn.Linear(512, Config.frame_feat_dim)

        # 时序建模
        self.temporal_encoder = nn.LSTM(
            input_size=Config.frame_feat_dim,
            hidden_size=Config.hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # 传感器特征提取
        self.sensor_net = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, Config.sensor_feat_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(Config.seq_length)
        )

        # 特征融合与预测
        self.fusion = nn.Sequential(
            nn.Linear(2 * Config.hidden_dim + Config.sensor_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, frames, sensor):
        # 图像特征提取
        B, T = frames.size(0), frames.size(1)
        frame_features = self.frame_encoder(frames.view(-1, *frames.shape[2:]))
        frame_features = frame_features.view(B, T, -1)

        # 时序建模
        temporal_out, _ = self.temporal_encoder(frame_features)

        # 传感器特征
        sensor = sensor.permute(0, 2, 1)
        sensor_features = self.sensor_net(sensor).permute(0, 2, 1)

        # 特征融合
        combined = torch.cat([temporal_out, sensor_features], dim=2)
        return self.fusion(combined).squeeze(-1)


# -------------------- 训练流程 --------------------
def train():
    # 初始化环境
    torch.manual_seed(Config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Config.save_dir, exist_ok=True)

    # 初始化数据集（关键修改点：scaler传递逻辑）
    train_set = MultimodalDataset(mode='train')
    val_set = MultimodalDataset(
        mode='val',
        sensor_scaler=train_set.sensor_scaler,
        target_scaler=train_set.target_scaler
    )

    # 数据加载器（解决Windows多进程问题）
    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(
        train_set,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_set,
        batch_size=Config.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    # 初始化模型
    model = Seq2SeqRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # 训练记录
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(Config.epochs):
        model.train()
        epoch_train_loss = 0.0

        # 训练阶段
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{Config.epochs}', unit='batch') as pbar:
            for batch in pbar:
                optimizer.zero_grad()

                # 数据加载
                frames = batch['frames'].to(device, non_blocking=True)
                sensor = batch['sensor'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)

                # 前向传播
                outputs = model(frames, sensor)
                loss = criterion(outputs, targets)

                # 反向传播
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # 记录损失
                epoch_train_loss += loss.item() * frames.size(0)
                pbar.set_postfix(loss=loss.item())

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(device, non_blocking=True)
                sensor = batch['sensor'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)

                outputs = model(frames, sensor)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * frames.size(0)

        # 计算平均损失
        train_loss = epoch_train_loss / len(train_set)
        val_loss = epoch_val_loss / len(val_set)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'sensor_scaler': train_set.sensor_scaler,
                'target_scaler': train_set.target_scaler,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
            }, os.path.join(Config.save_dir, "best_model1.pth"))

        # 打印epoch信息
        print(f"Epoch {epoch + 1:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # 损失可视化
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(os.path.join(Config.save_dir, 'loss_curve.png'))
    plt.close()


if __name__ == "__main__":
    train()