#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh

LOG_DIR=~/projects/hubness/logs
mkdir -p $LOG_DIR
LOG=$LOG_DIR/laion_pipeline_$(date +%Y%m%d_%H%M%S).log

cd ~/projects/hubness

echo "=== 00. gallery/index (LAION) 시작 $(date) ===" | tee -a $LOG
conda activate hubness
python scripts/02_build_index_laion.py 2>&1 | tee -a $LOG

echo "=== 01. 05_generate_hub_laion.py 단발 검증 시작 $(date) ===" | tee -a $LOG
python scripts/05_generate_hub_laion.py --mode universal --gc_index 20 2>&1 | tee -a $LOG
echo "=== 05_generate_hub_laion.py 검증 통과, 배치 생성으로 진행 $(date) ===" | tee -a $LOG

echo "=== 02. 08_batch_generate_laion.py 시작 $(date) ===" | tee -a $LOG
python scripts/08_batch_generate_laion.py 2>&1 | tee -a $LOG

echo "=== 03. 06_evaluate_hub_asr_laion.py 시작 $(date) ===" | tee -a $LOG
python scripts/06_evaluate_hub_asr_laion.py 2>&1 | tee -a $LOG

echo "=== 04. 07_hubness_detector_laion.py 시작 $(date) ===" | tee -a $LOG
python scripts/07_hubness_detector_laion.py 2>&1 | tee -a $LOG

echo "=== 05. 09_prepare_for_detector_laion.py 시작 $(date) ===" | tee -a $LOG
python scripts/09_prepare_for_detector_laion.py 2>&1 | tee -a $LOG

echo "=== 06. 10_run_ahd_scan_laion.py 시작 $(date) ===" | tee -a $LOG
conda activate ahd
python scripts/10_run_ahd_scan_laion.py 2>&1 | tee -a $LOG

echo "=== 07. results -> results_laion 전환 (11a/11b/12가 LAION hub를 보도록) $(date) ===" | tee -a $LOG
mv ~/projects/hubness/results ~/projects/hubness/results_openai_backup
ln -s ~/projects/hubness/results_laion ~/projects/hubness/results

echo "=== 08. 11a_extract_hubness_env.py 시작 $(date) ===" | tee -a $LOG
conda activate hubness
cd ~/projects/hubness
python scripts/11a_extract_hubness_env.py 2>&1 | tee -a $LOG

echo "=== 09. 11b_extract_imagebind_env.py 시작 $(date) ===" | tee -a $LOG
conda activate imagebind
cd ~/projects/hubness/external/ImageBind
python ~/projects/hubness/scripts/11b_extract_imagebind_env.py 2>&1 | tee -a $LOG

echo "=== 10. 12_compare_models.py 시작 $(date) ===" | tee -a $LOG
conda activate hubness
cd ~/projects/hubness
python scripts/12_compare_models.py 2>&1 | tee -a $LOG

echo "=== 11. results 원상복구 $(date) ===" | tee -a $LOG
rm ~/projects/hubness/results
mv ~/projects/hubness/results_openai_backup ~/projects/hubness/results

echo "=== 전체 LAION 파이프라인 완료 $(date) ===" | tee -a $LOG