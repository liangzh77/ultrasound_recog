#!/bin/bash
# 方案 B: nnU-Net v2 训练脚本
#
# 用法:
#   bash scripts/08_train_nnunet.sh [preprocess|train|find_best|predict]
#
# 默认执行全部步骤。

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# 设置 nnU-Net 环境变量
export nnUNet_raw="$ROOT_DIR/data/nnunet"
export nnUNet_preprocessed="$ROOT_DIR/data/nnunet_preprocessed"
export nnUNet_results="$ROOT_DIR/runs/nnunet"

DATASET_ID=1
STEP=${1:-all}

echo "============================================"
echo "nnU-Net v2 训练流水线"
echo "============================================"
echo "  nnUNet_raw:          $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results:      $nnUNet_results"
echo "  Dataset ID:          $DATASET_ID"
echo "  Step:                $STEP"
echo "============================================"

# 预处理
if [ "$STEP" = "all" ] || [ "$STEP" = "preprocess" ]; then
    echo ""
    echo "[Step 1/4] 预处理..."
    nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity
fi

# 训练 (5-fold cross validation)
if [ "$STEP" = "all" ] || [ "$STEP" = "train" ]; then
    echo ""
    echo "[Step 2/4] 训练 (5-fold cross validation)..."
    for FOLD in 0 1 2 3 4; do
        echo "  Training fold $FOLD..."
        nnUNetv2_train $DATASET_ID 2d $FOLD
    done
fi

# 找最优配置
if [ "$STEP" = "all" ] || [ "$STEP" = "find_best" ]; then
    echo ""
    echo "[Step 3/4] 寻找最优配置..."
    nnUNetv2_find_best_configuration $DATASET_ID -c 2d
fi

# 预测
if [ "$STEP" = "all" ] || [ "$STEP" = "predict" ]; then
    echo ""
    echo "[Step 4/4] 在测试集上预测..."
    nnUNetv2_predict \
        -i "$nnUNet_raw/Dataset001_Knee/imagesTs" \
        -o "$nnUNet_results/predictions" \
        -d $DATASET_ID \
        -c 2d \
        --save_probabilities
fi

echo ""
echo "nnU-Net 训练流水线完成!"
