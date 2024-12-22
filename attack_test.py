import whisper
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Union, TYPE_CHECKING
import tqdm
import sys
import warnings
import pandas as pd
import re  # 导入正则表达式库，用于处理文件名
from whisper import load_audio  # 假设您使用的是 OpenAI 的 Whisper 库
import jiwer  # 计算WER
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from whisper.audio import CHUNK_LENGTH, SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram, get_mel_spectrogram
from whisper.tokenizer import Tokenizer, get_tokenizer
from whisper.utils import compression_ratio, exact_div, format_timestamp, make_safe, optional_int, optional_float, \
    str2bool, get_writer
from whisper.decoding import DecodingTask, DecodingOptions

if TYPE_CHECKING:
    from whisper.model import Whisper

from torch.autograd import Variable
import torchaudio
from torch.optim import Adam
import random
import csv
import json
import os
import torch.nn as nn
from torchaudio.transforms import MFCC, MelSpectrogram, Spectrogram, AmplitudeToDB

# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
import torch.nn.functional as F



def seed_everything(seed: int = None) -> int:
    if seed is None:
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    elif isinstance(seed, str):
        seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


seed_everything(42)


@dataclass(frozen=True)
class LossResult:
    audio_features: Tensor
    language: str
    loss: Tensor
    logits: Tensor
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class LossTask(DecodingTask):

    def __init__(self, model, confidence, correct_first_word, options, *args, **kwargs):
        super(LossTask, self).__init__(model, options, *args, **kwargs)
        self.correct_first_word = correct_first_word
        self.confidence = confidence

    def _loss_from_logits(self, logits, tokens, loss_fct):
        loss = loss_fct(logits.transpose(1, 2), tokens).mean(dim=1)
        if self.correct_first_word:
            corrective_first_word_loss = loss_fct(logits[:, 0], tokens[:, 0])
            loss = loss + corrective_first_word_loss / logits.size(1)
        return loss

    def _decoder_forward(self, audio_features: Tensor, tokens: Tensor, init_tokens_length: int):
        self.inference.initial_token_length = tokens.shape[-1]
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(
            n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        logits = self.inference.logits(tokens[:, :-1], audio_features)
        loss = self._loss_from_logits(
            logits[:, (init_tokens_length - 1):], tokens[:, init_tokens_length:], loss_fct)
        self.inference.cleanup_caching()
        if self.confidence > 0:
            mask = torch.nn.functional.one_hot(
                tokens, num_classes=logits.size(-1))
            mask[:, :init_tokens_length] = 0
            logits = logits - self.confidence * mask[:, 1:]
        return loss, logits[:, (init_tokens_length - 1):], no_speech_probs, sum_logprobs

    def run(self, mel: Tensor, label: Union[str, torch.Tensor]) -> List[LossResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]
        audio_features: Tensor = self._get_audio_features(
            mel)  # encoder forward pass
        init_tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(
            n_audio, 1).to(audio_features.device)
        init_tokens_length = init_tokens.size(-1)
        if isinstance(label, str):
            label = torch.tensor([tokenizer.encode(label)])

        input_tokens: Tensor = torch.tensor(label).repeat(
            n_audio, 1).to(audio_features.device)
        eos_tokens: Tensor = torch.tensor([[tokenizer.eot]]).repeat(
            n_audio, 1).to(audio_features.device)
        tokens = torch.cat([init_tokens, input_tokens, eos_tokens], dim=-1)
        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(
            audio_features, tokens)

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        audio_features = audio_features.repeat_interleave(self.n_group, dim=0)
        tokens = tokens.repeat_interleave(
            self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        loss, logits, no_speech_probs, sum_logprobs = self._decoder_forward(
            audio_features, tokens, init_tokens_length)
        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin: (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist()
                                   for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i]
                                     for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]
        fields = (texts, languages, tokens, audio_features, logits, loss)
        return [
            LossResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                logits=logits,
                loss=loss,
            )
            for text, language, tokens, features, logits, loss in zip(*fields)
        ]

    def _get_lang_loss(self, mel: Tensor):
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        lang_logits = get_lang_logits(model, mel, tokenizer)
        language_en = "en"
        language_en = "<|" + language_en.strip("<|>") + "|>"
        lang_token_en = torch.LongTensor(tokenizer.encode(language_en, allowed_special={'<|en|>'}))
    
        language_zh = "zh"
        language_zh = "<|" + language_zh.strip("<|>") + "|>"
        lang_token_zh = torch.LongTensor(tokenizer.encode(language_zh, allowed_special={'<|zh|>'}))

        lang_logits_softmax = torch.softmax(lang_logits, dim=-1)
        scale = 10

        loss_increase_tar = -torch.log(lang_logits_softmax[0][lang_token_zh.item()])
        penalty = torch.exp(scale * torch.relu(lang_logits_softmax[0][lang_token_zh.item()] + 0.1 - lang_logits_softmax[0][lang_token_en.item()])) - 1
        loss = loss_increase_tar + 1 * penalty
    
        return loss

    def _get_lang_loss(self, mel: Tensor):
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        lang_logits = get_lang_logits(model, mel, tokenizer)
        language_en = "en"
        language_en = "<|" + language_en.strip("<|>") + "|>"
        lang_token_en = torch.LongTensor(tokenizer.encode(language_en, allowed_special={'<|en|>'}))
        lang_loss_en = get_lang_loss(lang_logits, lang_token_en)

        return lang_loss_en


def get_loss_from_mel(model: "Whisper", mel: Tensor, label: Union[str, torch.Tensor], confidence, correct_first_word,
                      options: DecodingOptions = DecodingOptions()):  # -> Union[LossResult, List[LossResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).
    Parameters
    ----------
    model: Whisper
        the Whisper model instance
    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)
    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments
    Returns
    -------
    result: Union[LossResult, List[LossResult]]
        The result(s) of decoding contained in `LossResult` dataclass instance(s)
    """
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)
    result = LossTask(model, confidence, correct_first_word,
                      options).run(mel, label)
    lang_loss = LossTask(model, confidence, correct_first_word, options)._get_lang_loss(mel)
    if single:
        result = result[0]

    return result, lang_loss


def get_loss(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor],
        label: Union[str, torch.Tensor],
        *,
        verbose: Optional[bool] = None,
        temperature: Union[float, Tuple[float, ...]] = (
                0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        **decode_options,
):
    """
    Transcribe an audio file using Whisper
    Parameters
    ----------
    model: Whisper
        The Whisper model instance
    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform
    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything
    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.
    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed
    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed
    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent
    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances
    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    dtype = torch.float16 if decode_options.get(
        "fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Running model on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False
    mel = log_mel_spectrogram(audio)

    decode_options["language"] = "en"
    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual,
                              language=language, task=task)

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    def add_segment(
            *, start: float, end: float, text_tokens: torch.Tensor, result: LossResult
    ):
        text = tokenizer.decode(
            [token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                "start": start,
                "end": end,
                "text": text,
                "tokens": text_tokens.tolist(),
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        if verbose:
            print(
                make_safe(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"))

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = mel.shape[-1]
    previous_seek_value = seek

    with tqdm.tqdm(total=num_frames, unit='frames', disable=verbose is not False) as pbar:
        while seek < num_frames:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, seek:], N_FRAMES).to(
                model.device).to(dtype)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result, lang_loss = get_loss_from_mel(
                model, segment, label, False, DecodingOptions(**decode_options))
            # result, lang_loss = get_loss_from_mel(
            #     model, mel, label,
            #     confidence,
            #     correct_first_word,
            #     DecodingOptions(**options)
            # )

            tokens = torch.tensor(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    # fast-forward to the next segment boundary
                    seek += segment.shape[-1]
                    continue

            timestamp_tokens: torch.Tensor = tokens.ge(
                tokenizer.timestamp_begin)
            consecutive = torch.where(
                timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
            if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = (
                            sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                            sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    add_segment(
                        start=timestamp_offset + start_timestamp_position * time_precision,
                        end=timestamp_offset + end_timestamp_position * time_precision,
                        text_tokens=sliced_tokens[1:-1],
                        result=result,
                    )
                    last_slice = current_slice
                last_timestamp_position = (
                        tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * input_stride
                all_tokens.extend(tokens[: last_slice + 1].tolist())
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = timestamps[-1].item(
                    ) - tokenizer.timestamp_begin
                    duration = last_timestamp_position * time_precision

                add_segment(
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result,
                )

                seek += segment.shape[-1]
                all_tokens.extend(tokens.tolist())

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek
        loss = result.loss
    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens):]),
        segments=all_segments,
        language=language,
        loss=loss,
        lang_loss=lang_loss
    )


def get_loss_single_segment(
        model: "Whisper",
        audio: torch.Tensor,
        label: Union[str, torch.Tensor],
        *,
        verbose: Optional[bool] = None,
        temperature: Union[float, Tuple[float, ...]] = (
                0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        confidence=0,
        correct_first_word=True,
        **options,
):
    """
    Transcribe an audio file using Whisper
    Parameters
    ----------
    model: Whisper
        The Whisper model instance
    audio: torch.Tensor
        The audio waveform
    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything
    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.
    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed
    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed
    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent
    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
    options: dict
        Keyword arguments to construct `DecodingOptions` instances
    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `options["language"]` is None.
    """
    dtype = torch.float16 if options.get("fp16", True) else torch.float32
    audio = audio.to(model.device).to(dtype)
    mel = log_mel_spectrogram(audio).to(dtype)
    mel = pad_or_trim(mel, N_FRAMES)
    options["language"] = "en"
    options["task"] = "transcribe"
    # language = options["language"]en
    # task = options.get("task", "transcribe")
    # result: LossResult = get_loss_from_mel(
    #     model, mel, label,
    #     confidence,
    #     correct_first_word,
    #     DecodingOptions(**options)
    # )

    result, lang_loss = get_loss_from_mel(
        model, mel, label,
        confidence,
        correct_first_word,
        DecodingOptions(**options)
    )

    loss = result.loss
    if loss.nelement() > 1:
        loss = loss.mean(dim=1)
    if loss.ndim == 0:
        loss = loss.unsqueeze(0)
    logits = result.logits
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    return dict(loss=loss, logits=logits, lang_loss=lang_loss, result=result)


# device = torch.device("cuda:0")
# model = whisper.load_model("base")
#
# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio("D:\ZWJ\deepspeech.pytorch-master\deepspeech.pytorch-master\data\Lib_clean\84-121123-0001.wav")
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)
#
# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")
#
# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)
#
# # print the recognized text
# print(result.text)
def get_lang_logits(
        model: "Whisper", mel: Tensor, tokenizer: Tokenizer = None
):
    if tokenizer is None:
        tokenizer = get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages
        )
    if (
            tokenizer.language is None
            or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    return logits


def get_lang_loss(logits, lang_token, reduction="none"):
    tokens = lang_token.to(device)
    loss_fct = nn.CrossEntropyLoss(reduction=reduction)
    loss_lang = loss_fct(logits, tokens)
    return loss_lang


def get_feature_parameters(feature):
    assert feature in (0, 1, 2, 3)

    feature_parameters = list()

    n_mfcc_list = (40,)
    n_mels_list = (80,)
    n_fft_list = (400,)
    hop_length_ratio = (0.5,)

    if feature <= 1:
        for n_fft in n_fft_list:
            for hop_length in hop_length_ratio:
                feature_parameters.append({"n_fft": n_fft, "hop_length": int(n_fft * hop_length)})
    elif feature == 2:
        for n_mels in n_mels_list:
            for n_fft in n_fft_list:
                for hop_length in hop_length_ratio:
                    feature_parameters.append({"n_mels": n_mels, "n_fft": n_fft, "hop_length": int(n_fft * hop_length)})
    else:
        for n_mfcc in n_mfcc_list:
            for n_mels in n_mels_list:
                for n_fft in n_fft_list:
                    for hop_length in hop_length_ratio:
                        feature_parameters.append(
                            {"n_mfcc": n_mfcc, "n_mels": n_mels, "n_fft": n_fft, "hop_length": int(n_fft * hop_length)})

    return feature_parameters


def get_feature_extractor(feature, feature_parameters_dict):
    # 如果 feature 的值为 0，返回 DBSpectrogram 类的实例
    if feature == 0:
        return DBSpectrogram(n_fft=feature_parameters_dict["n_fft"],
                             hop_length=feature_parameters_dict.get("hop_length")).to(device)
    # 如果 feature 的值为 1，返回 Spectrogram 类的实例
    elif feature == 1:
        return Spectrogram(n_fft=feature_parameters_dict["n_fft"],
                           hop_length=feature_parameters_dict.get("hop_length")).to(device)
    # 如果 feature 的值为 2，返回 MelSpectrogram 类的实例
    elif feature == 2:
        return MelSpectrogram(n_mels=feature_parameters_dict["n_mels"],
                              n_fft=feature_parameters_dict["n_fft"],
                              hop_length=feature_parameters_dict.get("hop_length")).to(device)
    # 如果 feature 不是 0、1 或 2，返回 MFCC 类的实例
    else:
        return MFCC(n_mfcc=feature_parameters_dict["n_mfcc"],
                    melkwargs={"n_mels": feature_parameters_dict["n_mels"],
                               "n_fft": feature_parameters_dict["n_fft"],
                               "hop_length": feature_parameters_dict.get("hop_length")}).to(device)


def mel_opt(audio_input, target_feature, perturbation, epsilon):
    #feature_extractor = get_feature_extractor(2, feature_parameters[0])
    dtype = torch.float16
    perturbation.requires_grad_(True)
    optimizer = Adam(params=[perturbation], lr=0.0001)
    for i in range(1000):
        audio_adv = audio_input + perturbation
        #integrated_command_feature = feature_extractor(audio_adv)
        integrated_command_feature = get_mel_spectrogram(audio_adv).to(dtype)
        l_mel = torch.norm(integrated_command_feature - target_feature)
        l_mel.backward()
        optimizer.step()
        with torch.no_grad():
        #     # perturbation = torch.clamp(perturbation, -epsilon * 0.1, epsilon * 0.1)
        #     torch.clip_(perturbation, -0.005, 0.005)
            pert_l2 = torch.norm(perturbation, p=2)
            if pert_l2 > clamp_l2:
                perturbation /= pert_l2
                perturbation *= clamp_l2 * 0.5
            perturbation.requires_grad_(True)
        optimizer.zero_grad()
    return perturbation


def calculate_epsilon(audio, SNR_dB):
    # Calculate the RMS of the audio
    A_signal = torch.sqrt(torch.mean(audio ** 2))

    # Convert SNR from dB to linear
    SNR_linear = 10 ** (SNR_dB / 20)

    # Calculate the maximum allowable noise level
    epsilon = A_signal / SNR_linear

    return epsilon


def match_tensor_length(tensor_A, tensor_B):
    len_A = tensor_A.size(0)
    len_B = tensor_B.size(0)

    if len_A < len_B:
        # 如果 tensor A 比 tensor B 短，使用 0 补齐到 tensor B 的长度
        padding_size = len_B - len_A
        padding = torch.zeros(padding_size, *tensor_A.shape[1:], dtype=tensor_A.dtype, device=tensor_A.device)
        tensor_A = torch.cat([tensor_A, padding], dim=0)
    elif len_A > len_B:
        # 如果 tensor A 比 tensor B 长，截断到 tensor B 的长度
        tensor_A = tensor_A[:len_B]

    return tensor_A



def attack_with_momentum_batch(audio_input, model_fri, label, file_name, audio_target):
    """
    使用动量梯度的方法对音频输入进行对抗攻击，目标是使模型将对抗样本识别为指定的标签。

    参数：
    - audio_input：原始音频输入，类型为张量（Tensor）。
    - model_fri：目标模型（如 Whisper 模型），用于评估对抗样本的效果。
    - model_eny：另一个辅助模型，用于进一步验证对抗样本的效果。
    - label：目标文本标签，即希望模型将对抗样本识别为的文本。
    """

    global epsilon
    global clamp_l2
    # 初始化扰动，设置为一个小的随机值，范围在 [-0.00001, 0.00001] 之间
    perturbation = Variable((0 * (torch.rand(size=audio_input.size()) - 0.5)).to(device), requires_grad=True)

    dtype = torch.float16
    target_feature = get_mel_spectrogram(audio_target).to(dtype)
    perturbation = mel_opt(audio_input, target_feature, perturbation, epsilon)

    # 初始化动量为与音频输入形状相同的零张量
    momentum = torch.zeros_like(audio_input)
    mu = 0.9  # 动量因子，控制动量的更新程度
    lr = epsilon * 0.02 # 学习率，基于 epsilon 调整

    #optimizer = Adam(params=[perturbation], lr=epsilon * 0.05)
    result = []
    best_result = []
    bSucceed = False

    SNR_dB = 25
    epsilon = calculate_epsilon(audio_input, SNR_dB)
    clamp_l2 = torch.norm(torch.ones_like(audio_input) * epsilon, p=2)
    for i in range(15000):  # 迭代次数设为 10000
        perturbation.requires_grad_(True)  # 确保扰动的梯度被计算
        audio_adv = audio_input + perturbation  # 生成对抗样本

        # 计算对抗样本在模型上的损失和 logits
        loss_and_logits = get_loss_single_segment(model_fri, audio_adv, label)
        alpha = loss_and_logits['loss'].item()  # 提取损失值

        # 如果语言损失小于 1，调整 alpha 的值
        if loss_and_logits['lang_loss'] < 1:
            alpha *= 0.1

        # 计算总损失，结合模型损失和语言损失
        loss = loss_and_logits['loss'] + loss_and_logits['lang_loss'] * alpha

        # 反向传播，计算梯度
        loss.backward()
        grad = perturbation.grad.data.detach()  # 获取扰动的梯度，并与计算图分离
        #optimizer.step()
        with torch.no_grad():
            # 更新动量项，使用一阶动量（指数加权平均）
            momentum = mu * momentum + grad / grad.norm(p=1)

            # 对动量进行自适应调整，抑制过小或过大的动量值
            momentum_abs = torch.abs(momentum)
            momentum_mean = torch.mean(momentum_abs)
            pert_zero = torch.zeros_like(momentum)

            # 将小于平均值 10% 的动量设为零
            momentum = torch.where(momentum_abs < momentum_mean * 0.01, pert_zero, momentum)

            # 将大于平均值两倍的动量减半
            momentum = torch.clamp(momentum, -momentum_mean*5, momentum_mean*5)

            # 归一化动量，用于更新扰动
            momentum_pert = momentum.clone()

            # 更新扰动，按动量的符号方向前进
            # if i > 1000:
            #     lr = epsilon * 0.02
            perturbation -= lr * momentum_pert.sign()

            # 计算扰动的 L2 范数
            pert_l2 = torch.norm(perturbation, p=2)
            if pert_l2 > clamp_l2:
                perturbation = perturbation / pert_l2 * clamp_l2

            # 生成新的对抗样本
            audio_adv = audio_input + perturbation

            # 将对抗样本转换为指定的数据类型并移动到模型设备上
            audio_ = audio_adv.to(model.device).to(dtype)

            # 计算对抗样本的对数 Mel 频谱图
            mel = log_mel_spectrogram(audio_).to(dtype)
            mel = pad_or_trim(mel, N_FRAMES)  # 对频谱图进行填充或截断

            # 使用模型对 Mel 频谱图进行解码，获得识别结果
            result_small_en = whisper.decode(model_fri, mel, options_en)

            # 每隔 10 次迭代，输出当前的结果和一些指标
            if i % 1000 == 0:
                noise_power = torch.sum(perturbation ** 2)  # 计算扰动的功率
                original_power = torch.sum(audio_input ** 2)  # 计算原始音频的功率
                snr_db = 10 * torch.log10(original_power / noise_power)  # 计算信噪比（以 dB 为单位）

            # 检查模型的识别结果是否与目标标签匹配（考虑了不同的标点符号）
            if result_small_en.text.upper() in [label[2:].upper(), label[2:].upper() + '.', label[2:].upper() + '?',
                                                label[2:].upper() + '。', label[2:].upper() + '？']:

                # torchaudio.save(savename, audio_adv.detach().cpu().reshape(1, -1), 16000)
                noise_power = torch.sum(perturbation ** 2)  # 计算扰动的功率
                original_power = torch.sum(audio_input ** 2)  # 计算原始音频的功率
                snr_db = 10 * torch.log10(original_power / noise_power)  # 计算信噪比（以 dB 为单位）


                result_small_zh = whisper.decode(model_fri, mel, options_en)
                print(result_small_zh.text, loss.item(), "step_num: ", i, "snr: ", snr_db.item(),
                      loss_and_logits['lang_loss'].item(), result_small_zh.text)
                print("SUCCEED!!!")

                # 计算 WER

                wer_value = jiwer.wer(label[2:].upper(), result_small_zh.text.upper())
                result = []
                result.append(label)
                result.append(i)
                result.append(snr_db.item())
                result.append(wer_value)
                best_result = result
                bSucceed = True

                SNR_dB += 5
                epsilon = calculate_epsilon(audio_input, SNR_dB)
                clamp_l2 = torch.norm(torch.ones_like(audio_input) * epsilon, p=2)

            elif i > 2000 or snr_db.item() > 44:
                if bSucceed:
                    return best_result

                else:
                    result_small_zh = whisper.decode(model_fri, mel, options_en)
                    print("Fail!!!", result_small_zh.text)

                    # 计算 WER

                    wer_value = jiwer.wer(label[2:].upper(), result_small_zh.text.upper())
                    result.append(label)
                    result.append(i)
                    result.append(snr_db.item())
                    result.append(wer_value)

                    return result


def attack_and_count():
    global epsilon
    global clamp_l2
    # 定义要使用的标签，并将其转换为大写
    # labels = [
    #     'what is my schedule for tomorrow',
    #     'remove all photos in my phone'
    #     'show me my last messages',
    #     'call the police for help quickly',
    #     'send a message to derek'
    # ]

    labels = [
        '- OK Google, browse to evil.com',
        '- What is my schedule for tomorrow',
        '- Remove all photos from my phone',
        '- Show me my last messages',
        '- Please send a message to Annie'
    ]

    # labels = [
    #     '- Be on time for the meeting tomorrow',
    #     '- The stuff is in the usual place',
    #     '- Please meet me at the cafe',
    #     '- Please check the email received yesterday',
    #     '- Call the police for help quickly'
    # ]

    target_file_paths = [
        r'/home/ckc/zwj/whisper_selective/data/tts_target/OK Google.wav',
        r'/home/ckc/zwj/whisper_selective/data/tts_target/What.mp3',
        r'/home/ckc/zwj/whisper_selective/data/tts_target/Remove.mp3',
        r'/home/ckc/zwj/whisper_selective/data/tts_target/Show.mp3',
        r'/home/ckc/zwj/whisper_selective/data/tts_target/Please send.mp3'
    ]

    # target_file_paths = [
    #     r'/home/ckc/zwj/whisper_selective/data/tts_target/Be on.mp3',
    #     r'/home/ckc/zwj/whisper_selective/data/tts_target/The stuff.mp3',
    #     r'/home/ckc/zwj/whisper_selective/data/tts_target/Please meet.mp3',
    #     r'/home/ckc/zwj/whisper_selective/data/tts_target/Please check.mp3',
    #     r'/home/ckc/zwj/whisper_selective/data/tts_target/Call.mp3'
    # ]

    combined_list = list(zip(labels, target_file_paths))

    # 指定要遍历的目录
    directory = r"/home/ckc/zwj/whisper_selective/data/zh2"  # 请替换为您的实际目录路径
    #directory = r"D:\ZWJ\deepspeech.pytorch-master\deepspeech.pytorch-master\data\Lib_clean"

    # 指定保存 Excel 文件的目录
    output_directory = r"/home/ckc/zwj/whisper_selective/experiment_data/tiny"# 请替换为您想要保存 Excel 文件的目录

    # 如果输出目录不存在，创建该目录
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历每个标签
    for label, path in combined_list:
        results = []  # 用于存储当前标签的所有结果

        audio_target = load_audio(path)
        audio_target = torch.tensor(audio_target).to(device)

        # 遍历目录中的所有文件
        for i, filename in enumerate(os.listdir(directory)):
            if filename.endswith('.mp3') or filename.endswith('.wav'):
                filepath = os.path.join(directory, filename)
                print(f"处理 mp3 文件: {filename}，使用标签: {label}, 目标文件: {path}")

                # 加载音频文件
                audio = load_audio(filepath)
                audio = torch.tensor(audio).to(device)

                length_A = audio.size(0)
                length_B = audio_target.size(0)

                # 确定需要padding的长度
                if length_A < length_B:
                    padding_size = length_B - length_A
                    audio = F.pad(audio, (0, padding_size), value=0)
                else:
                    audio = audio[0:audio_target.size(-1)]

                # 计算 epsilon 和 clamp_l2，用于后续的对抗攻击
                SNR_dB = 35
                epsilon = calculate_epsilon(audio, SNR_dB)
                clamp_l2 = torch.norm(torch.ones_like(audio) * epsilon, p=2)

                # 对音频进行对抗攻击
                result = attack_with_momentum_batch(audio, model, label, filename, audio_target)

                if result is not None and len(result) == 4:
                    label_result = result[0]
                    iterations = result[1]
                    snr_db = result[2]
                    wer_value = result[3]

                    results.append({
                        'Label': label,
                        'Filename': filename,
                        'Iterations': iterations,
                        'SNR_dB': snr_db,
                        'WER': wer_value
                    })
                else:
                    print(f"处理文件 {filename} 时出现错误，标签：{label}")

        df = pd.DataFrame(results)
        safe_label = re.sub(r'[\/:*?"<>|]', '_', label)
        excel_filename = f'results_{safe_label}.xlsx'
        excel_path = os.path.join(output_directory, excel_filename)
        df.to_excel(excel_path, index=False)
        print(f"标签 '{label}' 的结果已保存到 {excel_path}")

        # 清空结果列表，以便处理下一个标签
        results.clear()


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = whisper.load_model("tiny")
    #model = whisper.load_model("large-v2.pt")
    for param in model.parameters():
        param.requires_grad = False

    options_zh = whisper.DecodingOptions(task="transcribe", language="zh")
    options_en = whisper.DecodingOptions(task="transcribe", language="en")  # task="translate", task="transcribe",

    attack_and_count()


#write in anaconda3/envs/whisper_selective/lib/python3.9/site-packages/whisper/audio.py
# def get_mel_spectrogram(
#     audio: Union[str, np.ndarray, torch.Tensor],
#     n_mels: int = 80,
#     padding: int = 0,
#     device: Optional[Union[str, torch.device]] = None,
# ):
#     """
#     Compute the log-Mel spectrogram of

#     Parameters
#     ----------
#     audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
#         The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

#     n_mels: int
#         The number of Mel-frequency filters, only 80 is supported

#     padding: int
#         Number of zero samples to pad to the right

#     device: Optional[Union[str, torch.device]]
#         If given, the audio tensor is moved to this device before STFT

#     Returns
#     -------
#     torch.Tensor, shape = (80, n_frames)
#         A Tensor that contains the Mel spectrogram
#     """
#     if not torch.is_tensor(audio):
#         if isinstance(audio, str):
#             audio = load_audio(audio)
#         audio = torch.from_numpy(audio)

#     if device is not None:
#         audio = audio.to(device)
#     if padding > 0:
#         audio = F.pad(audio, (0, padding))
#     window = torch.hann_window(N_FFT).to(audio.device)
#     stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
#     magnitudes = stft[..., :-1].abs() ** 2

#     filters = mel_filters(audio.device, n_mels)
#     mel_spec = filters @ magnitudes

#     return mel_spec
