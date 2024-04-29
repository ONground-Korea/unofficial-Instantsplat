CUDA_VISIBLE_DEVICES=3 python train.py \
 -s /home/cvlab02/project/sdc/mipnerf360_dataset/bicycle \
 --exp_name bicycle_jacobian_rela \
 --eval \
 --port 1236 \
 --images images_4 \
 --pose_noise \
 --pose_representation '9D' \
 --pretrained_scene /home/cvlab02/project/sdb/freegs/output/sfm_bicycle/ 


 
