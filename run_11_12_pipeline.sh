#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh

LOG_DIR=~/projects/hubness/logs
mkdir -p $LOG_DIR
LOG=$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log

echo "=== 11a 시작 $(date) ===" | tee -a $LOG
conda activate hubness
python ~/projects/hubness/scripts/11a_extract_hubness_env.py 2>&1 | tee -a $LOG
echo "=== 11a 완료 $(date) ===" | tee -a $LOG

echo "=== 11b 시작 $(date) ===" | tee -a $LOG
conda activate imagebind
cd ~/projects/hubness/external/ImageBind
python ~/projects/hubness/scripts/11b_extract_imagebind_env.py 2>&1 | tee -a $LOG
echo "=== 11b 완료 $(date) ===" | tee -a $LOG

echo "=== 12 시작 $(date) ===" | tee -a $LOG
conda activate hubness
python ~/projects/hubness/scripts/12_compare_models.py 2>&1 | tee -a $LOG
echo "=== 12 완료 $(date) ===" | tee -a $LOG

echo "=== 전체 파이프라인 완료 $(date) ===" | tee -a $LOG