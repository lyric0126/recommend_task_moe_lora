# MoCLE Evaluation Accuracy Delivery

This delivery adds a MoCLE checkpoint-step-290000 pseudo-user validation evaluation and combines it with the existing HydraLoRA accuracy summary.

## Main Files

- `SUMMARY.txt`: combined MoCLE + HydraLoRA result summary.
- `HYDRALORA_EVAL_SUMMARY.txt`: original HydraLoRA summary copied from the requested path.
- `mocle_ckpt290000_valid_accuracy.json`: full MoCLE predictions and scores.
- `mocle_ckpt290000_valid_accuracy.log`: MoCLE eval log.
- `mocle_train_summary.json`: MoCLE training summary.
- `mocle_data_build_stats.json`: source-data conversion stats.
- `eval_mocle_choice_accuracy.py`: evaluator used for the MoCLE run.

## Reproduction

```bash
cd /vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main
CUDA_VISIBLE_DEVICES=1 /home/liushaokun/miniconda3/envs/lavispy310/bin/python recommendation/movielens1m/eval_mocle_choice_accuracy.py \
  --base-model /vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B \
  --checkpoint outputs/human_like_v2fix_all_mocle_full_20260501_230156/checkpoint-step-290000 \
  --data /vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/data/hydralora_pseudo_v2fix_all/valid.json \
  --output deliveries/mocle_eval_accuracy_20260502/mocle_ckpt290000_valid_accuracy.json \
  --group-field domain --route-field domain --batch-size 16
```
