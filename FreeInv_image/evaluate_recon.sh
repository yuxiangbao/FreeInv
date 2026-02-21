python evaluation/evaluate_recon.py \
--metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" \
--result_path evaluation_result_recon.csv \
--edit_category_list 0 1 2 3 4 5 6 7 8 9 \
--tgt_methods 1_ddim+pnp 1_null-text-inversion+p2p 4_edict+p2p 1_ddpm_edit 1_infedit 1_directinversion+p2p 1_freeinv+pnp 1_ablation_ensemble+pnp 1_ablation_multi_branch+pnp