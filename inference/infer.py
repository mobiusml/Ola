import os

os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32'
os.environ['VIDEO_RESIZE'] = "0x64"
os.environ['VIDEO_MAXRES'] = "480"
os.environ['VIDEO_MINRES'] = "288"
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['PAD2STRIDE'] = '1'

import gradio as gr
import torch
import re
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import transformers
import moviepy as mp
from typing import Dict, Optional, Sequence, List
import librosa
import whisper
from ola.conversation import conv_templates, SeparatorStyle
from ola.model.builder import load_pretrained_model
from ola.datasets.preprocess import tokenizer_image_token, tokenizer_speech_image_token, tokenizer_speech_question_image_token, tokenizer_speech_token
from ola.mm_utils import KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from ola.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX
import argparse

def load_pretrained_model_quantized(model_path, model_base=None, is_lora=False, quantize=True, torch_dtype=torch.float16, device="cuda:0", use_flash_attn=False, **kwargs):

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, HqqConfig
    import torch
    from ola.model import OlaQwenForCausalLM
    from ola.model.speech_encoder.builder import build_speech_encoder
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    from hqq.backends.torchao import patch_hqq_to_aoint4

    if(torch_dtype == torch.float16):
        from gemlite.helper import A16Wn
        from gemlite.helper import A8W8_fp8_dynamic as A8W8_dynamic #A8W8_int8_dynamic
    
    kwargs['torch_dtype'] = torch_dtype
    #kwargs['device_map'] = 'cpu'

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    model_cls = OlaQwenForCausalLM

    # Load Ola model
    if is_lora:
        assert model_base is not None, "model_base is required for LoRA models."
        from ola.model.language_model.ola_qwen import OlaConfigQwen
        lora_cfg_pretrained = OlaConfigQwen.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading Ola from base model...')
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)
        print('Loading additional Ola weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        print('Loading Ola from base model...')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = model_cls.from_pretrained(model_base, config=cfg_pretrained, **kwargs)
        
        speech_projector_weights = torch.load(os.path.join(model_path, 'speech_projector.bin'), map_location=device)
        speech_projector_weights = {k: v.to(torch_dtype) for k, v in speech_projector_weights.items()}
        model.load_state_dict(speech_projector_weights, strict=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = model_cls.from_pretrained(model_path, **kwargs)


    #model.model.vision_tower -  OK
    #model.model.vision_resampler -  OK
    #model.model.speech_projector - OK
    #model.model.speech_encoder - OK
    #model.model.layers -> Quantize - OK
    #model.lm_head - OK

    #Quantizer
    def _quantize_fct(linear_layer, quant_config, name, skip):
        if(name in skip):
            return linear_layer.to(device=device, dtype=torch_dtype)
        else:
            if(quant_config == 'A8W8_dynamic'):
                if(torch_dtype == torch.float16):
                    linear_layer = A8W8_dynamic(device=device).from_linear(linear_layer)
                else:
                    pass
            else:
                linear_layer = HQQLinear(linear_layer, quant_config=quant_config, compute_dtype=torch_dtype, device=device)
                if(torch_dtype == torch.float16):
                    linear_layer = A16Wn(device=device).from_hqqlinear(linear_layer) #Gemlite backend
                else:
                    linear_layer = patch_hqq_to_aoint4(linear_layer, None)
            return linear_layer
        
    def _quantize(model, quant_config, skip=['']):
        for name, layer in model.named_children():
            if isinstance(layer, torch.nn.Linear):
                setattr(model, name, _quantize_fct(layer, quant_config, name, skip))
            else:
                _quantize(layer, quant_config, skip)

    #LLM quantization
    _quantize(model.model.layers, quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=1))
    model.lm_head = model.lm_head.to(device=device, dtype=torch_dtype)

    #Speech encoder
    model.get_model().speech_encoder = build_speech_encoder(model.config)
    model.get_model().speech_encoder.to(device=device, dtype=torch_dtype)
    model.model.speech_projector.to(device=device, dtype=torch_dtype)

    #Image tower
    image_processor = None
    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()
    print("Loading vision tower...")
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device)
    model.model.vision_tower.to(device=device, dtype=torch_dtype)
    model.model.vision_resampler.to(device=device, dtype=torch_dtype)
    image_processor = vision_tower.image_processor
    print("Loading vision tower succeeded.")
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 16384

    model = model.to(device=device, dtype=compute_dtype).eval()

    return tokenizer, model, image_processor, context_len

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="/root/Ola-7b")
parser.add_argument('--text', type=str, default=None)
parser.add_argument('--audio_path', type=str, default=None)
parser.add_argument('--image_path', type=str, default=None)
parser.add_argument('--video_path', type=str, default=None)
args = parser.parse_args()

model_path = args.model_path
# python3 inference/infer.py --video_path /root/zmore/Ola/test.mp4 --text "provide a detailed summary of the visual and audio content"

###########################################################################
device, compute_dtype = 'cuda:0', torch.bfloat16 #torch.bfloat16 #bfloat16: tinygemm | float16: gemlite

tokenizer, model, image_processor, _ = load_pretrained_model_quantized(model_path, quantize=True, torch_dtype=compute_dtype, device=device)

###########################################################################


USE_SPEECH=False
cur_dir = os.path.dirname(os.path.abspath(__file__))

def load_audio(audio_file_name):
    speech_wav, samplerate = librosa.load(audio_file_name, sr=16000)
    if len(speech_wav.shape) > 1:
        speech_wav = speech_wav[:, 0]
    speech_wav = speech_wav.astype(np.float32)
    CHUNK_LIM = 480000
    SAMPLE_RATE = 16000
    speechs = []
    speech_wavs = []

    if len(speech_wav) <= CHUNK_LIM:
        speech = whisper.pad_or_trim(speech_wav)
        speech_wav = whisper.pad_or_trim(speech_wav)
        speechs.append(speech)
        speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
    else:
        for i in range(0, len(speech_wav), CHUNK_LIM):
            chunk = speech_wav[i : i + CHUNK_LIM]
            if len(chunk) < CHUNK_LIM:
                chunk = whisper.pad_or_trim(chunk)
            speechs.append(chunk)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
    mels = []
    for chunk in speechs:
        chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
        mels.append(chunk)

    mels = torch.cat(mels, dim=0)
    speech_wavs = torch.cat(speech_wavs, dim=0)
    if mels.shape[0] > 25:
        mels = mels[:25]
        speech_wavs = speech_wavs[:25]

    speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
    speech_chunks = torch.LongTensor([mels.shape[0]])
    return mels, speech_length, speech_chunks, speech_wavs

def extract_audio(videos_file_path):
    my_clip = mp.VideoFileClip(videos_file_path)
    return my_clip.audio

image_path = args.image_path
audio_path = args.audio_path
video_path = args.video_path
text = args.text

if video_path is not None:
    modality = "video"
    visual = video_path
    assert image_path is None

elif image_path is not None:
    visual = image_path
    modality = "image"
    assert video_path is None

elif audio_path is not None:
    modality = "text"


# input audio and video, do not parse audio in the video, else parse audio in the video
if audio_path:
    USE_SPEECH = True
elif modality == "video":
    USE_SPEECH = True
else:
    USE_SPEECH = False

speechs = []
speech_lengths = []
speech_wavs = []
speech_chunks = []
if modality == "video":
    vr = VideoReader(visual, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 64, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(frame) for frame in spare_frames]
elif modality == "image":
    image = [Image.open(visual)]
    image_sizes = [image[0].size]
else:
    images = [torch.zeros(1, 3, 224, 224).to(dtype=compute_dtype, device=device, non_blocking=True)]
    images_highres = [torch.zeros(1, 3, 224, 224).to(dtype=compute_dtype, device=device, non_blocking=True)]
    image_sizes = [(224, 224)]


if USE_SPEECH and audio_path:
    audio_path = audio_path
    speech, speech_length, speech_chunk, speech_wav = load_audio(audio_path)
    speechs.append(speech.to(dtype=compute_dtype, device=device))
    speech_lengths.append(speech_length.to(device=device))
    speech_chunks.append(speech_chunk.to(device=device))
    speech_wavs.append(speech_wav.to(device=device))
    print('load audio')
elif USE_SPEECH and not audio_path:
    # parse audio in the video
    audio = extract_audio(visual)
    audio.write_audiofile("./video_audio.wav")
    video_audio_path = './video_audio.wav'
    speech, speech_length, speech_chunk, speech_wav = load_audio(video_audio_path)
    speechs.append(speech.to(dtype=compute_dtype, device=device))
    speech_lengths.append(speech_length.to(device=device))
    speech_chunks.append(speech_chunk.to(device=device))
    speech_wavs.append(speech_wav.to(device=device))
else:
    speechs = [torch.zeros(1, 3000, 128).to(dtype=compute_dtype, device=device)]
    speech_lengths = [torch.LongTensor([3000]).to(device=device)]
    speech_wavs = [torch.zeros([1, 480000]).to(device=device)]
    speech_chunks = [torch.LongTensor([1]).to(device=device)]

conv_mode = "qwen_1_5"
if text:
    qs = text
else:
    qs = ''

if USE_SPEECH and audio_path and image_path: # image + speech instruction
    qs = DEFAULT_IMAGE_TOKEN + "\n" + "User's question in speech: " + DEFAULT_SPEECH_TOKEN + '\n'
elif USE_SPEECH and video_path: # video + audio
    qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + qs
elif USE_SPEECH and audio_path: # audio + text
    qs = DEFAULT_SPEECH_TOKEN + "\n" + qs
elif image_path or video_path: # image / video
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
elif text: # text
    qs = qs

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
if USE_SPEECH and audio_path and image_path: # image + speech instruction
    input_ids = tokenizer_speech_question_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
elif USE_SPEECH and video_path: # video + audio
    input_ids = tokenizer_speech_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
elif USE_SPEECH and audio_path: # audio + text
    input_ids = tokenizer_speech_token(prompt, tokenizer, SPEECH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
else:
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

if modality == "video":
    video_processed = []
    for idx, frame in enumerate(video):
        image_processor.do_resize = False
        image_processor.do_center_crop = False
        frame = process_anyres_video(frame, image_processor)

        if frame_idx is not None and idx in frame_idx:
            video_processed.append(frame.unsqueeze(0))
        elif frame_idx is None:
            video_processed.append(frame.unsqueeze(0))
    
    if frame_idx is None:
        frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()
    
    video_processed = torch.cat(video_processed, dim=0).to(device=device, dtype=compute_dtype)
    video_processed = (video_processed, video_processed)

    video_data = (video_processed, (384, 384), "video")
elif modality == "image":
    image_processor.do_resize = False
    image_processor.do_center_crop = False
    image_tensor, image_highres_tensor = [], []
    for visual in image:
        image_tensor_, image_highres_tensor_ = process_anyres_highres_image(visual, image_processor)
        image_tensor.append(image_tensor_)
        image_highres_tensor.append(image_highres_tensor_)
    if all(x.shape == image_tensor[0].shape for x in image_tensor):
        image_tensor = torch.stack(image_tensor, dim=0)
    if all(x.shape == image_highres_tensor[0].shape for x in image_highres_tensor):
        image_highres_tensor = torch.stack(image_highres_tensor, dim=0)
    if type(image_tensor) is list:
        image_tensor = [_image.to(device=device, dtype=compute_dtype) for _image in image_tensor]
    else:
        image_tensor = image_tensor.to(device=device, dtype=compute_dtype)
    if type(image_highres_tensor) is list:
        image_highres_tensor = [_image.to(device=device, dtype=compute_dtype) for _image in image_highres_tensor]
    else:
        image_highres_tensor = image_highres_tensor.to(device=device, dtype=compute_dtype)

pad_token_ids = 151643

attention_masks = input_ids.ne(pad_token_ids).long().to(device)
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

gen_kwargs = {}

if "max_new_tokens" not in gen_kwargs:
    gen_kwargs["max_new_tokens"] = 1024
if "temperature" not in gen_kwargs:
    gen_kwargs["temperature"] = 0.2
if "top_p" not in gen_kwargs:
    gen_kwargs["top_p"] = None
if "num_beams" not in gen_kwargs:
    gen_kwargs["num_beams"] = 1

with torch.inference_mode():
    if modality == "video":
        output_ids = model.generate(
            inputs=input_ids,
            images=video_data[0][0],
            images_highres=video_data[0][1],
            modalities=video_data[2],
            speech=speechs,
            speech_lengths=speech_lengths,
            speech_chunks=speech_chunks,
            speech_wav=speech_wavs,
            attention_mask=attention_masks,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
        )
    elif modality == "image":
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            images_highres=image_highres_tensor,
            image_sizes=image_sizes,
            modalities=['image'],
            speech=speechs,
            speech_lengths=speech_lengths,
            speech_chunks=speech_chunks,
            speech_wav=speech_wavs,
            attention_mask=attention_masks,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
        )
    elif modality == "text":
        output_ids = model.generate(
            input_ids,
            images=images,
            images_highres=images_highres,
            image_sizes=image_sizes,
            modalities=['text'],
            speech=speechs,
            speech_lengths=speech_lengths,
            speech_chunks=speech_chunks,
            speech_wav=speech_wavs,
            attention_mask=attention_masks,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            )

outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
outputs = outputs.strip()
if outputs.endswith(stop_str):
    outputs = outputs[:-len(stop_str)]
outputs = outputs.strip()

print(outputs)