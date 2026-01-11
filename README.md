# 因果推論在定價策略上的應用
# Causal Inference for Pricing Strategy

[![CI/CD Pipeline](https://github.com/AnnetteChiu/pricing-causal-analysis/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/AnnetteChiu/pricing-causal-analysis/actions)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

這個項目提供了一套完整的因果推論分析工具，專門用於定價策略的數據分析。包含了從基礎到進階的多種因果推論方法，幫助企業做出更科學的定價決策。

🎯 **適用於**: 電商平台、SaaS服務、零售業、訂閱服務等需要優化定價策略的企業
📊 **包含方法**: 7種主流因果推論方法，從A/B測試到機器��習因果推論
🚀 **即用即得**: 完整的演示代碼和詳細文檔，快速上手

## 快速開始 Quick Start

### 安裝 Installation

```bash
# 克隆倉庫
git clone https://github.com/AnnetteChiu/pricing-causal-analysis.git
cd pricing-causal-analysis

# 安裝依賴
pip install -r requirements.txt

# 或者直接安裝包
pip install -e .
```

### 運行演示 Run Demo

```bash
# 快速演示（推薦新手）
python demo_analysis.py

# 真實數據分析演示 🆕
python real_data_analysis.py

# 完整分析（包含所有方法）
python run_analysis.py

# 基礎分析
python pricing_causal_analysis.py
```

### 快速上手示例

```python
# 方法1: 使用模擬數據
from pricing_causal_analysis import PricingCausalAnalysis
analyzer = PricingCausalAnalysis()
data = analyzer.generate_synthetic_data(n_samples=5000)
analyzer.randomized_experiment_analysis()
print(f"降價效果: {analyzer.results['experiment_analysis']['ate_sales']:.0f} 單位銷量增加")

# 方法2: 使用真實數據 🆕
from real_data_analysis import analyze_real_data
column_mapping = {
    '你的價格列名': 'price',
    '你的銷量列名': 'sales_volume',
    '你的日期列名': 'date'
}
analyzer, data = analyze_real_data('your_data.csv', column_mapping)
```

## 功能特點 Features

### 🆕 真實數據支持
- **智能數據加載**: 支持CSV、Excel文件，自動處理編碼問題
- **列名自動識別**: 智能建議列名映射，快速適配你的數據
- **完整預處理流程**: 缺失值處理、異常值檢測、數據驗證
- **自動變數創建**: 智能生成處理變數和控制變數
- **一鍵分析**: 從原始數據到分析報告的完整流程

### 基礎因果推論方法
- **隨機實驗分析 (A/B測試)**: 最可靠的因果識別方法
- **回歸調整**: 控制混淆變數的基礎方法
- **傾向得分匹配**: 處理選擇偏誤的經典方法
- **價格彈性分析**: 測量不同客戶群體的價格敏感度

### 進階因果推論方法
- **差分差分法 (DID)**: 利用時間和橫截面變異識別因果效應
- **工具變數法 (IV)**: 處理內生性問題
- **回歸不連續設計 (RD)**: 利用閾值規則的準實驗方法
- **合成控制法**: 構建反事實對照組
- **機器學習因果推論**: Double ML和異質性處理效應分析

## 安裝要求

```bash
pip install -r requirements.txt
```

## 使用方法

### 快速開始

運行完整分析：
```python
python run_analysis.py
```

### 分步驟運行

1. **基礎分析**：
```python
python pricing_causal_analysis.py
```

2. **進階分析**：
```python
# 需要先運行基礎分析
from pricing_causal_analysis import main as run_basic_analysis
from advanced_pricing_methods import run_advanced_analysis

basic_analyzer = run_basic_analysis()
advanced_analyzer = run_advanced_analysis(basic_analyzer)
```

## 文件結構

```
├── pricing_causal_analysis.py    # 基礎因果推論分析
├── advanced_pricing_methods.py   # 進階因果推論方法
├── run_analysis.py              # 完整分析運行器
├── requirements.txt             # 依賴包列表
└── README.md                   # 說明文檔
```

## 分析流程

### 1. 數據生成
- 模擬真實的定價實驗數據
- 包含客戶特徵、競爭對手價格、季節性等因素
- 設定已知的真實因果效應用於方法驗證

### 2. 基礎分析
- 簡單相關性分析（展示可能的偏誤）
- 隨機實驗分析（金標準）
- 回歸調整分析
- 傾向得分匹配
- 價格彈性估計

### 3. 進階分析
- 差分差分法（面板數據分析）
- 工具變數法（處理內生性）
- 回歸不連續設計（閾值分析）
- 合成控制法（時間序列因果推論）
- 機器學習方法（處理複雜性和異質性）

### 4. 結果比較
- 不同方法的處理效應估計比較
- 方法特點分析（內部效度、外部效度、實施難度、成本）
- 實際應用建議

## 主要輸出

### 統計結果
- 各種方法的處理效應估計
- 統計顯著性檢驗
- 模型擬合度指標
- 異質性分析結果

### 可視化圖表
- 價格和銷量分佈對比
- 處理效應比較圖
- 差分差分法趨勢圖
- 回歸不連續設計散點圖
- 合成控制法時間序列圖
- 方法特點雷達圖

### 分析報告
- 數據概況總結
- 主要發現和統計結果
- 不同客戶群體的價格彈性
- 方法學建議和實施指導

## 實際應用場景

### 電商平台
- A/B測試優化商品定價
- 分析競爭對手價格影響
- 不同用戶群體的差異化定價

### 訂閱服務
- 價格變動對用戶留存的影響
- 新用戶定價策略優化
- 升級定價的因果效應分析

### 零售業
- 促銷活動效果評估
- 季節性定價策略
- 區域差異化定價分析

## 方法選擇指南

| 方法 | 適用場景 | 優點 | 缺點 |
|------|----------|------|------|
| 隨機實驗 | 有實驗條件 | 因果識別最可靠 | 成本高，實施難度大 |
| 回歸調整 | 觀察性數據 | 簡單易實施 | 依賴強假設 |
| 差分差分 | 政策變化 | 控制時間不變因子 | 需要平行趨勢 |
| 工具變數 | 內生性問題 | 處理內生性 | 需要強工具變數 |
| 機器學習 | 大數據環境 | 處理複雜關係 | 解釋性較差 |

## 注意事項

1. **假設檢驗**: 每種方法都有其識別假設，使用前需要仔細檢驗
2. **數據質量**: 確保數據的完整性和準確性
3. **外部效度**: 考慮分析結果的推廣性
4. **持續監控**: 定價策略需要持續監控和調整

## 擴展功能

可以根據具體需求添加：
- 實時數據接口
- 自動化報告生成
- 交互式儀表板
- 更多機器學習方法
- 貝葉斯因果推論

## 項目結構 Project Structure

```
pricing-causal-analysis/
├── 📊 pricing_causal_analysis.py    # 基礎因果推論分析
├── 🔬 advanced_pricing_methods.py   # 進階分析方法  
├── 🎯 demo_analysis.py              # 快速演示腳本
├── 🆕 real_data_analysis.py         # 真實數據分析演示
├── 🆕 real_data_loader.py           # 真實數據加載器
├── 🚀 run_analysis.py              # 完整分析運行器
├── 📋 requirements.txt             # 依賴列表
├── 📚 使用指南.md                  # 詳細使用指南
├── 🆕 真實數據使用指南.md          # 真實數據使用說明
├── 🧪 tests/                      # 測試文件
├── 📄 README.md                   # 項目說明
└── ⚙️  setup.py                   # 安裝配置
```

## 貢獻 Contributing

我們歡迎各種形式的貢獻！請查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解詳情。

### 如何貢獻
- 🐛 報告 Bug
- ✨ 提出新功能
- 📚 改進文檔
- 🧪 添加測試用例
- 🔧 代碼優化

## 許可證 License

本項目採用 MIT 許可證 - 查看 [LICENSE](LICENSE) 文件了解詳情。

## 作者 Author

**Annette Chiu** - [GitHub](https://github.com/AnnetteChiu)

## 致謝 Acknowledgments

- 感謝因果推論學術社區的理論基礎
- 感謝 Python 數據科學生態系統
- 感謝所有貢獻者和用戶的支持

## 引用 Citation

如果您在研究中使用了本項目，請引用：

```bibtex
@software{chiu2026pricing,
  author = {Chiu, Annette},
  title = {Causal Inference for Pricing Strategy},
  url = {https://github.com/AnnetteChiu/pricing-causal-analysis},
  year = {2026}
}
```

---

⭐ 如果這個項目對您有幫助，請給我們一個星標！

📧 有問題？[創建 Issue](https://github.com/AnnetteChiu/pricing-causal-analysis/issues/new) 或查看 [使用指南](使用指南.md)