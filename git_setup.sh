#!/bin/bash

# Git 倉庫設置腳本
# Git Repository Setup Script

echo "🚀 開始設置 Git 倉庫..."

# 初始化 Git 倉庫
git init

# 添加所有文件
git add .

# 創建初始提交
git commit -m "🎉 Initial commit: 因果推論在定價策略上的應用

- ✨ 添加基礎因果推論分析模塊
- 🔬 添加進階分析方法 (DID, IV, RD, 合成控制法, Double ML)
- 📊 添加豐富的數據可視化功能
- 📋 添加自動化分析報告生成
- 🎯 添加演示分析腳本
- 📚 添加完整的使用指南和文檔
- 🧪 添加測試框架
- ⚙️ 添加 CI/CD 配置
- 📄 添加項目文檔和許可證"

# 設置遠程倉庫（需要替換為實際的倉庫地址）
echo "📡 設置遠程倉庫..."
echo "請手動執行以下命令來添加遠程倉庫："
echo "git remote add origin https://github.com/AnnetteChiu/pricing-causal-analysis.git"

# 創建主分支
git branch -M main

echo "✅ Git 倉庫設置完成！"
echo ""
echo "📋 接下來的步驟："
echo "1. 在 GitHub 上創建新倉庫: https://github.com/new"
echo "2. 倉庫名稱: pricing-causal-analysis"
echo "3. 執行: git remote add origin https://github.com/AnnetteChiu/pricing-causal-analysis.git"
echo "4. 執行: git push -u origin main"
echo ""
echo "🎉 完成後您的項目就會在 GitHub 上了！"