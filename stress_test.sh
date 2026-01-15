#!/bin/bash
set -e

# Stress Test: High Learning Rate (1e-3)

echo "=========================================="
echo "STARTING STRESS TEST (LR = 1e-3)"
echo "=========================================="

# 1. ReluGPT Stress Test
echo "[1.1] Running ReluGPT Stress Test..."
pixi run train --experiment ReluGPT --learning_rate 1e-3 --comment Relu_Stress_1e3 --max_iters 5000
echo "[1.2] Running ReluGPT Stress Test..."
pixi run train --experiment ReluGPT --learning_rate 1e-4 --comment Relu_Stress_1e4 --max_iters 10000
echo "[1.3] Running ReluGPT Stress Test..."
pixi run train --experiment ReluGPT --learning_rate 3e-3 --comment Relu_Stress_3e3 --max_iters 5000


# 2. GeluGPT Stress Test
echo "[2.1] Running GeluGPT Stress Test..."
pixi run train --experiment GeluGPT --learning_rate 1e-3 --comment Gelu_Stress_1e3 --max_iters 5000
echo "[2.2] Running GeluGPT Stress Test..."
pixi run train --experiment GeluGPT --learning_rate 1e-4 --comment Gelu_Stress_1e4 --max_iters 10000
echo "[2.3] Running GeluGPT Stress Test..."
pixi run train --experiment GeluGPT --learning_rate 3e-3 --comment Gelu_Stress_3e3 --max_iters 5000


echo "=========================================="
echo "STRESS TEST COMPLETED SUCCESSFULLY"
echo "=========================================="
