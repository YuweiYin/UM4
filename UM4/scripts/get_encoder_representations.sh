echo "MNMT baseline"
MODEL="/path/to/checkpoint.pt"
for lang in "en" "fr" "cs" "de" "fi" "et" "ro" "hi" "tr"; do
    echo "${lang}"
    python /path/to/scripts/get_encoder_representation.py \
        --ckpt_path ${MODEL} --src_fn /path/to/t-sne/parallel.${lang} --suffix "baseline"
done

echo "Ours"
MODEL="/path/to/checkpoint.pt"
for lang in "en" "fr" "cs" "de" "fi" "et" "ro" "hi" "tr"; do
    echo "${lang}"
    python /path/to/scripts/get_encoder_representation.py \
        --ckpt_path ${MODEL} --src_fn /path/to/t-sne/parallel.${lang} --suffix "ours"
done
