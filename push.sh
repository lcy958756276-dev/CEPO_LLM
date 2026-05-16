#!/bin/bash
# 文件名: push.sh
# 用法:
#   bash push.sh
# 或:
#   bash push.sh "更新说明"

set -e

# ==============================
# 配置
# ==============================
REPO_URL="https://github.com/lcy958756276-dev/CEPO_LLM.git"

# 提交信息，默认使用当前时间
COMMIT_MSG=${1:-"auto update $(date '+%Y-%m-%d %H:%M:%S')"}

# ==============================
# 初始化仓库（如果尚未初始化）
# ==============================
if [ ! -d ".git" ]; then
    echo ">>> 初始化 Git 仓库..."
    git init
fi

# ==============================
# 设置远程仓库
# ==============================
if git remote get-url origin >/dev/null 2>&1; then
    echo ">>> 更新 origin..."
    git remote set-url origin "$REPO_URL"
else
    echo ">>> 添加 origin..."
    git remote add origin "$REPO_URL"
fi

# ==============================
# 设置主分支为 main
# ==============================
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || true)

if [ -z "$CURRENT_BRANCH" ]; then
    git checkout -b main
elif [ "$CURRENT_BRANCH" != "main" ]; then
    git branch -M main
fi

# ==============================
# 添加全部文件
# ==============================
echo ">>> 添加文件..."
git add .

# ==============================
# 检查是否有变更
# ==============================
if git diff --cached --quiet; then
    echo ">>> 没有需要提交的变更。"
else
    echo ">>> 提交代码..."
    git commit -m "$COMMIT_MSG"
fi

# ==============================
# 推送到 GitHub
# ==============================
echo ">>> 推送到 GitHub..."
git push -u origin main

echo ">>> 上传完成。"