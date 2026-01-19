hostname
date
CUDA_VISIBLE_DEVICES=0,1 accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=allenai/Olmo-3-1025-7B \
    --tasks global_piqa_prompted_yor_latn,global_piqa_completions_yor_latn,global_mmlu_full_yo,belebele_yor_Latn,global_piqa_prompted_eng_latn,global_piqa_completions_eng_latn,global_mmlu_full_en,belebele_eng_Latn \
    --batch_size auto:4
    
date