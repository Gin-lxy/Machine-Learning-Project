import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_audio_features(file_path, fixed_length=130):
    """
    加载音频并提取特征
    """
    try:
        # 加载音频文件
        y, sr = librosa.load(file_path)
        
        # 提取Mel频谱图
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # 统一长度处理
        def pad_or_truncate(array, target_length):
            current_length = array.shape[1]
            if current_length > target_length:
                return array[:, :target_length]
            else:
                pad_width = ((0, 0), (0, target_length - current_length))
                return np.pad(array, pad_width, mode='constant')
        
        # 处理特征长度
        mel_spec_db = pad_or_truncate(mel_spec_db, fixed_length)
        mfccs = pad_or_truncate(mfccs, fixed_length)
        
        return {
            'mel_spec': mel_spec_db.astype(np.float32),
            'mfcc': mfccs.astype(np.float32)
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

class DeceptionModel(nn.Module):
    def __init__(self, time_steps=130):
        super().__init__()
        
        # CNN for Mel Spectrogram
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        # LSTM for MFCC
        self.lstm = nn.LSTM(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Calculating CNN output size
        self._cnn_output_size = self._get_cnn_output_size(time_steps)
        
        # Fusion Network
        self.fusion_net = nn.Sequential(
            nn.Linear(self._cnn_output_size + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _get_cnn_output_size(self, time_steps):
        # Helper function to calculate CNN output size
        mel_channels = 128
        height = mel_channels // 8  # After 3 maxpool layers
        width = time_steps // 8     # After 3 maxpool layers
        return 128 * height * width
    
    def forward(self, mel_spec, mfcc):
        # Process Mel Spectrogram with CNN
        batch_size = mel_spec.size(0)
        x_cnn = self.cnn(mel_spec)
        x_cnn_flat = x_cnn.view(batch_size, -1)
        
        # Process MFCC with LSTM
        x_lstm, _ = self.lstm(mfcc.transpose(1, 2))
        x_lstm_last = x_lstm[:, -1]  # Take the last output
        
        # Combine features
        combined = torch.cat([x_cnn_flat, x_lstm_last], dim=1)
        output = self.fusion_net(combined)
        
        return output

class DeceptionDataset(torch.utils.data.Dataset):
    def __init__(self, features_list, labels):
        self.features = features_list
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        mel_spec = torch.FloatTensor(self.features[idx]['mel_spec']).unsqueeze(0)
        mfcc = torch.FloatTensor(self.features[idx]['mfcc'])
        label = torch.FloatTensor([self.labels[idx]])
        
        return {
            'mel_spec': mel_spec,
            'mfcc': mfcc,
            'label': label
        }

def train_model(model, train_loader, val_loader, epochs=100, patience=10):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            mel_spec = batch['mel_spec'].to(device)
            mfcc = batch['mfcc'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(mel_spec, mfcc)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                mel_spec = batch['mel_spec'].to(device)
                mfcc = batch['mfcc'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(mel_spec, mfcc)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}')
    
    return history

def main():
    # 配置参数
    csv_path = 'CBU0521DD_stories_attributes.csv'
    audio_dir = 'Deception/Deception-main/CBU0521DD_stories'
    batch_size = 16
    fixed_length = 130
    
    # 加载数据
    df = pd.read_csv(csv_path)
    features_list = []
    labels = []
    
    print("Loading and processing audio files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = f"{audio_dir}/{row['filename']}"
        features = load_audio_features(audio_path, fixed_length=fixed_length)
        
        if features is not None:
            features_list.append(features)
            labels.append(1 if 'True Story' in row['Story_type'] else 0)
    
    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        features_list, labels, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = DeceptionDataset(X_train, y_train)
    val_dataset = DeceptionDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    
    # 初始化并训练模型
    model = DeceptionModel(time_steps=fixed_length)
    print("Starting training...")
    history = train_model(model, train_loader, val_loader)
    
    # 绘制训练历史
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()