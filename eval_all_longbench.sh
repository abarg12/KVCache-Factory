export CUDA_VISIBLE_DEVICES=0

max_capacity_prompts=128 # 128,2048 in paper
attn_implementation='flash_attention_2' # Support "flash_attention_2", "sdpa", "eager".
source_path='./'
model_path='meta-llama/Llama-3.2-1B'
# method='PyramidKV' # Support PyramidKV, SnapKV, H2O, StreamingLLM, CAM, L2Norm, ThinK
# merge_method='pivot' # Support "pivot"(LOOK-M_PivotMerge).
# quant_method='None' # Support kivi and kvquant, default None.
# nbits=8 # Quantization bit-width support 8,4,2. Need to set quant_method first.
save_dir=${source_path}"results_long_bench" # path to result save_dir

for method in "PyramidKV" "SnapKV" "H2O" "StreamingLLM" "CAM" "L2Norm" "ThinK"
do
python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} #\
    #--merge ${merge_method} \
    # --nbits ${nbits} \
    # --quant_method ${quant_method}
done