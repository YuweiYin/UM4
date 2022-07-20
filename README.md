# UM4: Unified Multilingual Multiple Teacher-Student Model for Zero-Resource Neural Machine Translation

![picture](https://yuweiyin.github.io/files/publications/2022-07-23-IJCAI-MNMT-UM4.png)

## Abstract

Most translation tasks among languages belong to
the zero-resource translation problem where parallel
corpora are unavailable. Multilingual neural machine
translation (MNMT) enables one-pass translation
using shared semantic space for all languages
compared to the two-pass pivot translation but often
underperforms the pivot-based method. In this paper,
we propose a novel method, named as **U**nified
**M**ultilingual **M**ultiple teacher-student **M**odel for
N**M**T (**UM4**). Our method unifies source-teacher,
target-teacher, and pivot-teacher models to guide
the student model for the zero-resource translation.
The source teacher and target teacher force the student
to learn the direct source to target translation
by the distilled knowledge on both source and target
sides. The monolingual corpus is further leveraged
by the pivot-teacher model to enhance the
student model. Experimental results demonstrate
that our model of 72 directions significantly outperforms
previous methods on the WMT benchmark.

## Data

* **Download**
  * Download Here: [Google Drive](https://drive.google.com/drive/folders/1Cr2MZUX_SHKQdfpip6LtODlQFiOTqquX?usp=sharing)
  * Raw Training/Test Data & SPM ([SentencePiece](https://github.com/google/sentencepiece)) model & Multilingual Dictionary
* **Overview**
  * Multilingual dataset with 9 languages and 56 zero-resource translation directions.
  * English (En), French (Fr), Czech (Cs), German(De), Finnish (Fi), Estonian (Et), Romanian (Ro), Hindi(Hi), and Turkish (Tr).
  * English (En) is treated as the pivot language. The other 8 languages form (8 * (8 - 1)) = 56 zero-resource translation directions.
* **Bitext Data**
  * All the bitext training data (e.g., Fr-En, Ro-En, En-Hi) are from the WMT benchmark of recent years.
  * For 72 = (9 * (9 - 1)) translation directions of 9 languages, [TED Talk dataset](http://phontron.com/data/ted_talks.tar.gz) is used as the valid and test sets.
* **Monolingual Data**
  * The English monolingual data is collected from [NewsCrawl](http://data.statmt.org/news-crawl) and randomly sample 1 million English sentences.


## Environment

* Python: >= 3.6
* [PyTorch](http://pytorch.org/): >= 1.5.0
* NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* [Fairseq](https://github.com/pytorch/fairseq): 1.0.0

```bash
cd UM4/fairseq
pip install --editable ./
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```


## UM4 Training

* Training with the **Source** and **Target** Teacher:

```bash
TEXT=/path/to/data-bin/
MODEL=/path/to/model/

python -m torch.distributed.launch \
  --nproc_per_node=8 --nnodes=4 --node_rank=${OMPI_COMM_WORLD_RANK} \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} train.py ${TEXT} \
  --save-dir ${MODEL} --arch "transformer_vaswani_wmt_en_de_big" --task "translation_multi_simple_epoch" \
  --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 \
  --encoder-langtok "tgt" --langtoks '{"main":("tgt",None)}' --langs "fr,cs,de,fi,et,ro,hi,tr" \
  --lang-pairs "fr-cs,fr-de,fr-fi,fr-et,fr-ro,fr-hi,fr-tr,cs-fr,cs-de,cs-fi,cs-et,cs-ro,cs-hi,cs-tr,de-fr,de-cs,de-fi,de-et,de-ro,de-hi,de-tr,fi-fr,fi-cs,fi-de,fi-et,fi-ro,fi-hi,fi-tr,et-fr,et-cs,et-de,et-fi,et-ro,et-hi,et-tr,ro-fr,ro-cs,ro-de,ro-fi,ro-et,ro-hi,ro-tr,hi-fr,hi-cs,hi-de,hi-fi,hi-et,hi-ro,hi-tr,tr-fr,tr-cs,tr-de,tr-fi,tr-et,tr-ro,tr-hi" \
  --share-all-embeddings --max-source-positions 512 --max-target-positions 512 \
  --criterion "label_smoothed_cross_entropy" --label-smoothing 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 5e-4 \
  --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-epoch 5 --warmup-updates 4000 \
  --max-update 10000000 --max-epoch 24 --max-tokens 4096 --update-freq 4 --score-size 4 \
  --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test \
  --fp16 --truncate-source --virtual-epoch-size 100000000 2>&1 | tee -a ${MODEL}/train.log
```

---

* Training with the **Source**, **Target**, and **Pivot** Teacher:

```bash
TEXT=/path/to/data-bin/
PSEUDO_TEXT=/path/to/pseudo-data-bin/
MODEL=/path/to/model/

python -m torch.distributed.launch \
  --nproc_per_node=8 --nnodes=4 --node_rank=${OMPI_COMM_WORLD_RANK} \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} train.py ${TEXT} \
  --save-dir ${MODEL} --arch "transformer_vaswani_wmt_en_de_big" --task "translation_multi_simple_epoch" \
  --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 \
  --encoder-langtok "tgt" --langtoks '{"main":("tgt",None),"pseudo":("tgt",None)}' --langs "fr,cs,de,fi,et,ro,hi,tr" \
  --lang-pairs "fr-cs,fr-de,fr-fi,fr-et,fr-ro,fr-hi,fr-tr,cs-fr,cs-de,cs-fi,cs-et,cs-ro,cs-hi,cs-tr,de-fr,de-cs,de-fi,de-et,de-ro,de-hi,de-tr,fi-fr,fi-cs,fi-de,fi-et,fi-ro,fi-hi,fi-tr,et-fr,et-cs,et-de,et-fi,et-ro,et-hi,et-tr,ro-fr,ro-cs,ro-de,ro-fi,ro-et,ro-hi,ro-tr,hi-fr,hi-cs,hi-de,hi-fi,hi-et,hi-ro,hi-tr,tr-fr,tr-cs,tr-de,tr-fi,tr-et,tr-ro,tr-hi" \
  --share-all-embeddings --max-source-positions 512 --max-target-positions 512 \
  --criterion "label_smoothed_cross_entropy" --label-smoothing 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 5e-4 \
  --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-epoch 5 --warmup-updates 4000 \
  --max-update 10000000 --max-epoch 24 --max-tokens 4096 --update-freq 4 --score-size 4 \
  --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test \
  --fp16 --truncate-source --virtual-epoch-size 100000000 \
  --extra-data {'pseudo':'${PSEUDO_TEXT}'} --extra-lang-pairs {'pseudo':'fr-cs,fr-de,fr-fi,fr-et,fr-ro,fr-hi,fr-tr,cs-fr,cs-de,cs-fi,cs-et,cs-ro,cs-hi,cs-tr,de-fr,de-cs,de-fi,de-et,de-ro,de-hi,de-tr,fi-fr,fi-cs,fi-de,fi-et,fi-ro,fi-hi,fi-tr,et-fr,et-cs,et-de,et-fi,et-ro,et-hi,et-tr,ro-fr,ro-cs,ro-de,ro-fi,ro-et,ro-hi,ro-tr,hi-fr,hi-cs,hi-de,hi-fi,hi-et,hi-ro,hi-tr,tr-fr,tr-cs,tr-de,tr-fi,tr-et,tr-ro,tr-hi'} 2>&1 | tee -a ${MODEL}/train.log
```


## Inference & Evaluation

* **Beam Search**: (during the inference) beam size = 5; length penalty = 1.0.
* **Metrics**: the case-sensitive detokenized BLEU using sacreBLEU:
  * BLEU+case.mixed+lang.{src}-{tgt}+numrefs.1+smooth.exp+tok.13a+version.1.3.1


## Citation

* arXiv: https://arxiv.org/abs/2207.04900
* IJCAI Anthology: https://www.ijcai.org/proceedings/2022/618

```bibtex
@inproceedings{um4,
  title     = {UM4: Unified Multilingual Multiple Teacher-Student Model for Zero-Resource Neural Machine Translation},
  author    = {Yang, Jian and Yin, Yuwei and Ma, Shuming and Zhang, Dongdong and Wu, Shuangzhi and Guo, Hongcheng and Li, Zhoujun and Wei, Furu},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {4454--4460},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/618},
  url       = {https://doi.org/10.24963/ijcai.2022/618},
}
```


## License

Please refer to the [LICENSE](./LICENSE) file for more details.


## Contact

If there is any question, feel free to create a GitHub issue or contact us by [Email](mailto:seckexyin@gmail.com).
