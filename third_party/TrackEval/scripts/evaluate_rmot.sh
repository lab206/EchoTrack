#python3 run_mot_challenge.py \
#--METRICS HOTA \
#--SEQMAP_FILE /data/wudongming/MOTR/seqmap_rmot_clean.txt \
#--SKIP_SPLIT_FOL True \
#--GT_FOLDER /data/Dataset/MOT17/images/train \
#--TRACKERS_FOLDER /data/wudongming/MOTR/exps/rmot_v2a/results_epoch249_ \
#--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
#--TRACKERS_TO_EVAL /data/wudongming/MOTR/exps/rmot_v2a/results_epoch249_ \
#--USE_PARALLEL True \
#--NUM_PARALLEL_CORES 2 \
#--SKIP_SPLIT_FOL True \
#--PLOT_CURVES False

python3 TrackEval/scripts/run_mot_challenge.py \
--METRICS HOTA \
--SEQMAP_FILE datasets/data_path/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER refer-kitti/KITTI/training/image_02 \
--TRACKERS_FOLDER exps/bi_attn_fft_dwt/results_epoch99 \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL exps/bi_attn_fft_dwt/results_epoch99 \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

#python3 TrackEval/scripts/run_mot_challenge.py \
#--METRICS HOTA \
#--SEQMAP_FILE datasets/data_path/seqmap.txt \
#--SKIP_SPLIT_FOL True \
#--GT_FOLDER refer-kitti/KITTI/training/image_02 \
#--TRACKERS_FOLDER exps/bi_attention/results_epoch99 \
#--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
#--TRACKERS_TO_EVAL exps/bi_attention/results_epoch99 \
#--USE_PARALLEL True \
#--NUM_PARALLEL_CORES 2 \
#--SKIP_SPLIT_FOL True \
#--PLOT_CURVES False



#python3 run_mot_challenge.py \
#--METRICS HOTA \
#--SEQMAP_FILE /data/wudongming/MOTR/seqmap_kitti_clean.txt \
#--SKIP_SPLIT_FOL True \
#--GT_FOLDER /data/Dataset/KITTI/training/image_02 \
#--TRACKERS_FOLDER /data/wudongming/FairMOT/exp/fairmot_kitti_2/result_epoch100 \
#--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
#--TRACKERS_TO_EVAL /data/wudongming/FairMOT/exp/fairmot_kitti_2/result_epoch100 \
#--USE_PARALLEL True \
#--NUM_PARALLEL_CORES 2 \
#--SKIP_SPLIT_FOL True \
#--PLOT_CURVES False