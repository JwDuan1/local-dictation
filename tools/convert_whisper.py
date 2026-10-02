"""Optional, one-time offline conversion: official OpenAI large-v3.pt -> HF transformers -> CTranslate2.

Only needed where huggingface.co is blocked (faster-whisper would otherwise download the model itself).
The generation config and tokenizer are rebuilt locally from the official openai-whisper package instead
of being downloaded from the Hub. Needs: torch, transformers<5, tiktoken, ctranslate2, openai-whisper.
"""
import base64
import gzip
import os
import sys

import numpy as np
import whisper
from ctranslate2.converters import TransformersConverter
from tiktoken.load import load_tiktoken_bpe
from transformers import GenerationConfig, WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizerFast
from whisper.tokenizer import get_tokenizer

sys.path.insert(0, os.path.dirname(__file__))
import convert_openai_to_hf as c  # transformers v4.46.0 script, fetched from GitHub

LLM_HOME = os.environ.get("DICTATE_LLM_HOME", r"D:\LLM")
PT = os.path.join(LLM_HOME, "whisper", "large-v3.pt")  # downloaded from OpenAI's CDN (see README)
HF_DIR = os.path.join(LLM_HOME, "whisper", "hf-large-v3")
CT2_DIR = os.path.join(LLM_HOME, "whisper", "faster-whisper-large-v3")


def offline_generation_config(is_multilingual, num_languages=100, openai_version=None):
    tok = get_tokenizer(multilingual=True, num_languages=num_languages)
    mask = np.frombuffer(gzip.decompress(base64.b85decode(whisper._ALIGNMENT_HEADS["large-v3"])), dtype=bool)
    heads = [[int(l), int(h)] for l, h in np.argwhere(mask.reshape(32, 20))]
    return GenerationConfig(
        alignment_heads=heads,
        begin_suppress_tokens=[tok.encode(" ")[0], tok.eot],
        bos_token_id=tok.eot,
        eos_token_id=tok.eot,
        pad_token_id=tok.eot,
        decoder_start_token_id=tok.sot,
        is_multilingual=True,
        lang_to_id={f"<|{code}|>": i for code, i in zip(tok.all_language_codes, tok.all_language_tokens)},
        max_initial_timestamp_index=50,
        max_length=448,
        no_timestamps_token_id=tok.no_timestamps,
        prev_sot_token_id=tok.sot_prev,
        suppress_tokens=sorted(tok.non_speech_tokens),
        task_to_id={"transcribe": tok.transcribe, "translate": tok.translate},
    )


c._get_generation_config = offline_generation_config
c.load_tiktoken_bpe = load_tiktoken_bpe
c._TOKENIZERS["multilingual"] = os.path.join(os.path.dirname(whisper.__file__), "assets", "multilingual.tiktoken")

model, is_multilingual, num_languages = c.convert_openai_whisper_to_tfms(PT, HF_DIR)
print("multilingual", is_multilingual, "languages", num_languages, "mels", model.config.num_mel_bins)

tokenizer = c.convert_tiktoken_to_hf(is_multilingual, num_languages)
processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=WhisperFeatureExtractor(feature_size=model.config.num_mel_bins))
processor.save_pretrained(HF_DIR)
WhisperTokenizerFast.from_pretrained(HF_DIR).save_pretrained(HF_DIR, legacy_format=False)
model.half().save_pretrained(HF_DIR)
print("HF model saved")

TransformersConverter(HF_DIR, copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
    CT2_DIR, quantization="float16", force=True)
print("CT2 model saved to", CT2_DIR)
