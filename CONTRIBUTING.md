# 貢獻指南 Contributing Guide

感謝您對定價策略因果推論分析項目的興趣！我們歡迎各種形式的貢獻。

## 如何貢獻

### 報告問題 Bug Reports
如果您發現了問題，請：
1. 檢查是否已有相關的 issue
2. 創建新的 issue，包含：
   - 問題的詳細描述
   - 重現步驟
   - 預期行為 vs 實際行為
   - 環境信息（Python版本、操作系統等）

### 功能請求 Feature Requests
如果您有新功能的想法：
1. 創建 issue 描述您的想法
2. 說明為什麼這個功能有用
3. 提供可能的實現方案

### 代碼貢獻 Code Contributions

#### 開發環境設置
```bash
# 1. Fork 並 clone 倉庫
git clone https://github.com/AnnetteChiu/pricing-causal-analysis.git
cd pricing-causal-analysis

# 2. 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安裝依賴
pip install -r requirements.txt
pip install -e .
```

#### 提交流程
1. 創建新分支：`git checkout -b feature/your-feature-name`
2. 進行更改並測試
3. 提交更改：`git commit -m "Add: your feature description"`
4. 推送分支：`git push origin feature/your-feature-name`
5. 創建 Pull Request

#### 代碼規範
- 使用 Python PEP 8 風格指南
- 添加適當的註釋和文檔字符串
- 為新功能編寫測試
- 確保代碼通過現有測試

#### 測試
```bash
# 運行基本測試
python demo_analysis.py

# 運行完整測試
python run_analysis.py
```

## 項目結構

```
pricing-causal-analysis/
├── pricing_causal_analysis.py    # 基礎分析模塊
├── advanced_pricing_methods.py   # 進階分析方法
├── demo_analysis.py              # 演示腳本
├── run_analysis.py              # 完整分析運行器
├── requirements.txt             # 依賴列表
├── README.md                   # 項目說明
├── 使用指南.md                  # 使用指南
├── CONTRIBUTING.md             # 貢獻指南
├── LICENSE                     # 許可證
└── setup.py                   # 安裝配置
```

## 貢獻類型

### 歡迎的貢獻
- 🐛 Bug 修復
- ✨ 新功能實現
- 📚 文檔改進
- 🧪 測試用例添加
- 🎨 代碼優化
- 🌐 國際化支持
- 📊 新的可視化方法
- 🔬 新的因果推論方法

### 特別需要的貢獻
- 實際數據集的測試案例
- 更多行業應用場景
- 性能優化
- 用戶界面改進
- 更多統計檢驗方法

## 行為準則

請遵循以下原則：
- 尊重所有貢獻者
- 建設性的反饋和討論
- 專注於技術問題
- 保持專業和友善的態度

## 許可證

通過貢獻代碼，您同意您的貢獻將在 MIT 許可證下發布。

## 聯繫方式

如有任何問題，請：
- 創建 GitHub issue
- 參與 discussions
- 查看現有文檔

感謝您的貢獻！🎉