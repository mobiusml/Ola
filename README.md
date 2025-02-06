<p align="center" width="100%">
<img src="https://ola-omni.github.io/static/images/ola-icon.png" alt="967023137dff29e65b21544e7620e0f7.webp" width=60%>
</p>
<div>

## Ola: Pushing the Frontiers of Omni-Modal Language Model with Progressive Modality Alignment

<p align="left">
    <a href='https://github.com/liuzuyan' target='_blank'>Zuyan Liu<sup>*,1,2</sup></a>&emsp;
    <a href='https://github.com/dongyh20/' target='_blank'>Yuhao Dong<sup>*,2,3</sup></a>&emsp;
    Jiahui Wang<sup>1</sup></a>&emsp;<br>
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>3</sup></a>&emsp;
    Winston Hu<sup>2</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=TN8uDQoAAAAJ' target='_blank'>Jiwen Lu<sup>1,&#x2709</sup></a>&emsp;
   <a href='https://raoyongming.github.io/' target='_blank'>Yongming Rao<sup>2,1,&#x2709</sup></a>&emsp;
</p>


<p align="left"><sup>1</sup>Tsinghua University &ensp; <sup>2</sup>Tencent Hunyuna Research&ensp; <sup>3</sup>S-Lab, NTU&ensp; </p>

<p align="left"><sup>*</sup> Equal Contribution<sup>&ensp; &#x2709</sup>  Corresponding Author</p>

[![Ola](https://img.shields.io/badge/Rank_1-OpenCampass(<30B)-blue)](https://rank.opencompass.org.cn/leaderboard-multimodal/?m=REALTIME)    [![Ola](https://img.shields.io/badge/Rank_8-VideoMME-red)](https://rank.opencompass.org.cn/leaderboard-multimodal/?m=REALTIME) 

---

**Project Page:** [![Ola](https://img.shields.io/badge/Ola-project_page-orange)](https://ola-omni.github.io) 

**Weights in Huggingface:** [![hf_checkpoint](https://img.shields.io/badge/🤗-Ola_7b-green)](https://huggingface.co/THUdyh/Ola-7b)

**arXiv Paper:** [![arxiv](https://img.shields.io/badge/Arxiv-xxx-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/xxxx)

**Demo by Gradio:** (Coming Soon) [![demo](https://img.shields.io/badge/Ola-Demo-yellow)](https://huggingface.co/spaces/THUdyh/xxxx) 

**Training Data:** (Coming Soon) [![data](https://img.shields.io/badge/Ola-Data-purple)](https://huggingface.co/datasets/THUdyh/xxxx) 


## 📢 News

- 🎉[07/2/2025] The Ola is released! Check our [project page](https://ola-omni.github.io), [model weights](https://huggingface.co/THUdyh/Ola-7b), [arXiv paper](https://arxiv.org/abs/xxxx) for the strong omni-modal understanding model!

- 🔥[06/2/2025] [Ola-7b](https://huggingface.co/THUdyh/Ola-7b) achieves **Rank #1** on the OpenCompass Multi-modal Leaderboard among all the models under 30B parameters with average score of **72.6**. Check the impressive results [here](https://rank.opencompass.org.cn/leaderboard-multimodal/?m=REALTIME)!

## 🚀Coming Soon

- [ ] Evaluation code on omni-modal benchmarks
- [ ] Gradio Demo
- [ ] Training Data (Video, Audio, Cross-Modality)

## 🌟 Introduction

**Ola** is an Omni-modal language model that achieves competitive performance across image, video, and audio understanding compared to specialized counterparts. Ola pushes the frontiers of the omni-modal language mode with the design of progressive modality alignment strategy, omni-modal architecture, and the well-designed cross-modality training data. 

<p align="center" width="100%">
<img src="https://ola-omni.github.io/static/images/teaser.png" alt="teaser.png" width=100%>
</p>
<div>

### Architecture

<p align="center" width="100%">
<img src="https://ola-omni.github.io/static/images/method.png" alt="method.png" width=100%>
</p>
<div>

Ola supports omni-modal inputs including text, image, video, and audio, capable of processing the inputs simultaneously with competitive performance on understanding tasks for all these modalities. Meanwhile, Ola supports user-friendly real-time streaming decoding for texts and speeches thanks to the text detokenizer and the speech decoder.

### Training Strategies

<p align="center" width="100%">
<img src="https://ola-omni.github.io/static/images/training.png" alt="training.png" width=100%>
</p>
<div>

We visualize the relationships among modalities in the left part. Speech acts as the connection between language and audio knowledge, while video constructs the bridge with highly relevant visual and audio information. Therefore, we design the progressive alignment training strategy from primary to periphery. Furthermore, we design the cross-modality video-audio data to better capture the relationships among modalities.

### Performance

<p align="center" width="100%">
<img src="https://ola-omni.github.io/static/images/results.png" alt="results.png" width=100%>
</p>
<div>

Ola Achieves competitive performance across major multi-modal benchmarks when compared to state-of-the-art specialist-modal LLMs.

## Installation


#### 1. Clone this repository:
```bash
git clone https://github.com/Ola-Omni/Ola
cd Ola
```

#### 2. Install the required package:
```bash
conda create -n ola python=3.10 -y
conda activate ola
pip install --upgrade pip
pip install -e .
```
#### 3.Install additional packages for training cases

```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Quick Start

1. Download `Ola-7b` from [Huggingface](https://huggingface.co/THUdyh/Ola-7b) or skip the step to using the online weights directly.

2. Download audio encoder and put the weights under repo directory `path/to/Ola/pretrained`

- Speech Encoder (`Whisper-large-v3`):

```
import whisper
model = whisper.load_model("large-v3", download_root="models/speech_encoder/")
```

- Music Encoder ([Fine-tuned BEATs_iter3+ (AS2M) (cpt2)](https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea))

3. Run `inference/infer.py`

- Text & Image Understanding

```
python3 inference/infer.py --image_path *.png,jpg --text user_instruction
```

- Text & Video Understanding

```
python3 inference/infer.py --video_path *.mp4 --text user_instruction
```

- Text & Audio Understanding

```
python3 inference/infer.py --audio_path *.wav,mp3 --text user_instruction
```

- Audio & Image Understanding

```
python3 inference/infer.py --audio_path *.png,jpg --audio_path *.wav,mp3
```

## Evaluation

Coming Soon, Stay tuned!

## Training

Coming Soon, Stay tuned!


## Citation

If you find it useful for your research and applications, please cite our paper using this BibTeX:
```bibtex
{}
```

## Acknowledgement

- Our codebase is conducted on [LLaVA](https://github.com/LLaVA-VL/LLaVA-NeXT)

- Thanks to [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) team for the evaluation system!