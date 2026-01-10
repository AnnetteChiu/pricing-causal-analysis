#!/bin/bash

echo "🚀 準備推送到 GitHub..."

# 檢查是否已經添加了遠程倉庫
if git remote get-url origin 2>/dev/null; then
    echo "✅ 遠程倉庫已存在"
else
    echo "📡 添加遠程倉庫..."
    git remote add origin https://github.com/AnnetteChiu/pricing-causal-analysis.git
fi

# 推送到 GitHub
echo "📤 推送代碼到 GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "🎉 成功推送到 GitHub！"
    echo "📍 倉庫地址: https://github.com/AnnetteChiu/pricing-causal-analysis"
    echo ""
    echo "✅ 接下來可以："
    echo "1. 訪問倉庫查看代碼"
    echo "2. 設置倉庫描述和標籤"
    echo "3. 啟用 Issues 和 Discussions"
    echo "4. 創建第一個 Release"
else
    echo "❌ 推送失敗，請檢查："
    echo "1. GitHub 倉庫是否已創建"
    echo "2. 網絡連接是否正常"
    echo "3. GitHub 認證是否正確"
fi