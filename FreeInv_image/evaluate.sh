python evaluation/evaluate.py \
--metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
--result_path evaluation_result.csv \
--edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_methods 1_ddim+p2p 1_null-text-inversion+p2p 4_edict+p2p 1_ddpm_edit 1_infedit 1_directinversion+p2p 1_freeinv+p2p