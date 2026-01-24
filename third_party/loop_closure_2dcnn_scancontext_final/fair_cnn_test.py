import pickle
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from models.cnn_2d_models import Simple2DCNN

def calculate_metrics(y_true, y_pred):
    """Calculates and returns a dictionary of performance metrics."""
    acc_0 = accuracy_score(y_true, y_pred)
    errors = np.abs(np.array(y_pred) - np.array(y_true))
    acc_1 = np.mean(errors <= 1)
    acc_2 = np.mean(errors <= 2)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    return {
        "绝对准确率 (Top-1)": f"{acc_0 * 100:.1f}%",
        "误差≤1 准确率": f"{acc_1 * 100:.1f}%",
        "误差≤2 准确率": f"{acc_2 * 100:.1f}%",
        "加权精确率 (Precision)": f"{precision:.3f}",
        "加权召回率 (Recall)": f"{recall:.3f}",
        "加权F1分数 (F1-Score)": f"{f1:.3f}",
    }

def run_fair_cnn_test(database_path, model_path):
    print("="*60)
    print("🎯 公平的 2D CNN 模型诊断测试 (含损失值计算)")
    print("="*60)

    # 1. Load and split data
    print(f"📂 正在加载数据库: {database_path}")
    with open(database_path, 'rb') as f: data = pickle.load(f)
    all_features, all_labels = data['location_features'], data['location_labels']
    num_locations = data['num_locations']
    print("✅ 数据库加载成功。")

    print("\n✂️  正在复现训练时的60/20/20数据划分...")
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    stratify_temp = all_labels if np.min(counts) >= 3 else None
    X_train, X_temp, y_train, y_temp = train_test_split(all_features, all_labels, test_size=0.4, random_state=42, stratify=stratify_temp)
    stratify_final = y_temp if stratify_temp is not None else None
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=stratify_final)
    print(f"✅ 数据划分完成。测试集样本数: {len(X_test)}")

    # 2. Load trained model
    print(f"\n📂 正在加载已训练模型: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Simple2DCNN(num_classes=num_locations).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ 模型加载成功。")

    # 3. Run inference and calculate loss
    print(f"\n⚙️  正在对 {len(X_test)} 个测试样本执行模型推理并计算损失...")
    cnn_preds = []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in range(len(X_test)):
            sc_matrix = X_test[i]
            true_label = y_test[i]

            sc_tensor = torch.FloatTensor(sc_matrix).unsqueeze(0).unsqueeze(0).to(device)
            label_tensor = torch.LongTensor([true_label]).to(device)

            output = model(sc_tensor)
            
            loss = loss_fn(output, label_tensor)
            total_loss += loss.item()

            cnn_preds.append(torch.argmax(output, dim=1).item())
            
    avg_loss = total_loss / len(X_test)
    print("✅ 推理和损失计算完成。")

    # 4. Calculate and report metrics
    metrics = calculate_metrics(y_test, cnn_preds)
    metrics["平均测试损失 (Avg Loss)"] = f"{avg_loss:.4f}"

    print("\n" + "="*60)
    print("📊 2D CNN 公平诊断测试最终结果")
    print("="*60)
    for name, value in metrics.items():
        print(f"   - {name:<25}: {value}")
    print("\n" + "="*60)

if __name__ == '__main__':
    db_path = 'location_database.pkl'
    model_path = 'models/saved/trajectory_localizer_simple2dcnn_acc97.5.pth'
    run_fair_cnn_test(db_path, model_path)