"""
Adapted from 
- https://github.com/SesameAILabs/csm for original generator object
- https://github.com/davidbrowne17/csm-streaming for streaming / optimizations
"""

from dataclasses import dataclass
from typing import List, Tuple
from dataclasses import dataclass
import math
import os
from typing import List, Tuple, Generator as PyGenerator, Optional, Callable
import time
import queue
import threading
import platform
from typing_extensions import OrderedDict
import wave
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
import logging
from functools import lru_cache
from typing import Callable, List, Optional, Tuple
import numpy as np

import torch
import torchaudio

from . import MIMI_SAMPLE_RATE, AUDIO_NUM_CODEBOOKS
from .models import Model
from .watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark
from .utils import load_llama3_tokenizer, load_mimi, load_model, reset_caches

logger = logging.getLogger(__name__)

@dataclass
class Segment:
    speaker: int
    text: str
    sample_rate = MIMI_SAMPLE_RATE
    audio: torch.Tensor


def generate_audio(model, audio_tokenizer, text_tokenizer, watermarker, text, speaker_id, device, use_amp=True, max_audio_length_ms=10_000):
    """Generate audio from text."""
    model.eval()
    generator = Generator(model, audio_tokenizer, text_tokenizer, watermarker)
    
    with torch.no_grad(), torch.amp.autocast(device_type=str(device), enabled=use_amp):
        audio = generator.generate(
            text=text,
            speaker=speaker_id,
            context=[],
            max_audio_length_ms=max_audio_length_ms,
        )
        audio = audio.squeeze().cpu().numpy()
    
    reset_caches(model)
    return audio


def load_csm_1b(model_name_or_checkpoint_path: str = "sesame/csm-1b", device: str = "cuda"):
    """
    Load the CSM-1B model with extreme optimizations and warmup.
    """

    # Enable all CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print(f"Loading CSM-1B model '{model_name_or_checkpoint_path}' with extreme optimizations...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    model = load_model(model_name_or_checkpoint_path, None, 0.5)
    model.eval()

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model.backbone = torch.compile(model.backbone,mode='reduce-overhead', fullgraph=True, backend='inductor')
    model.decoder = torch.compile(model.decoder,mode='reduce-overhead', fullgraph=True, backend='inductor')

    model.to(device=device, dtype=dtype)

    print("Model compilation complete. Creating generator...")

    generator = Generator(model)
    generator._stream_buffer_size = 20

    # Setup tokenization caching
    generator._tokenization_cache = {}

    original_tokenize_text = generator._tokenize_text_segment

    @lru_cache(maxsize=2048)
    def cached_tokenize_text_segment(text_str, speaker_int):
        return original_tokenize_text(text_str, speaker_int)

    generator._tokenize_text_segment = lambda text, speaker: cached_tokenize_text_segment(text, speaker)

    # Perform warmup
    warmup_generator(generator)

    return generator


class Generator:
    def __init__(
        self,
        model: Model,
        audio_tokenizer: torch.nn.Module | None = None,
        text_tokenizer = None,
        watermarker = None,
    ):
        self._model = model
        self._model.setup_caches(1)
        device = next(model.parameters()).device
        self.device = device

        self._text_tokenizer = load_llama3_tokenizer() if text_tokenizer is None else text_tokenizer
        self._audio_tokenizer = load_mimi(device=device) if audio_tokenizer is None else audio_tokenizer
        self._watermarker = load_watermarker(device=device) if watermarker is None else watermarker

        self.sample_rate = self._audio_tokenizer.sample_rate

        self._num_codebooks = AUDIO_NUM_CODEBOOKS

        self._stream_buffer_size = 20
        self.max_seq_len = 2048
        self._cache = OrderedDict()
        self._text_token_cache = {}
        torch.set_num_threads(16)
        torch.cuda.set_per_process_memory_fraction(0.95)

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize text segment with caching optimization for reduced latency.
        """
        # Check cache first
        cache_key = f"{speaker}:{text}"
        if not hasattr(self, '_text_token_cache'):
            self._text_token_cache = {}
        
        if cache_key in self._text_token_cache:
            return self._text_token_cache[cache_key]

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), self._num_codebooks+1, dtype=torch.long, device=self.device)
        text_frame_mask = torch.zeros(len(text_tokens), self._num_codebooks+1, dtype=torch.bool, device=self.device)
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True

        frame_tokens = [text_frame]
        frame_masks = [text_frame_mask]
        
        result = (torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0))
        
        self._text_token_cache[cache_key] = result
        
        return result

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        
        # Limit to the number of codebooks set in MIMI
        audio_tokens = audio_tokens[:self._num_codebooks, :]
        
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), self._num_codebooks+1).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), self._num_codebooks+1).bool().to(self.device)
        audio_frame[:, :self._num_codebooks] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :self._num_codebooks] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)
    
    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        total_len = text_tokens.size(0) + audio_tokens.size(0)

        if total_len > self.max_seq_len:
            overflow = total_len - self.max_seq_len

            if text_tokens.size(0) > overflow:
                text_tokens = text_tokens[overflow:]
                text_masks = text_masks[overflow:]
            else:
                audio_overflow = overflow - text_tokens.size(0)
                text_tokens = text_tokens[0:0]
                text_masks = text_masks[0:0]
                audio_tokens = audio_tokens[audio_overflow:]
                audio_masks = audio_masks[audio_overflow:]

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)
    
    @torch.inference_mode()
    def _decode_frames(self, frames):
        if not frames:
            return torch.tensor([])
        
        # Only use first N codebooks for faster decoding
        frames_reduced = [frame[:, :self._num_codebooks//2] for frame in frames]
        audio = self._audio_tokenizer.decode(torch.stack(frames_reduced).permute(1, 2, 0)).squeeze(0).squeeze(0)
        return audio

    @torch.inference_mode()
    def generate_stream(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.7,
        topk: int = 30,
        on_chunk_generated: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        """
        Generate audio in a streaming fashion, optimized for lower latency to first chunk.
        """
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)

        tokens, tokens_mask = [], []

        initial_batch_size = 20
        normal_batch_size = 20  
        initial_buffer_size = 20
        normal_buffer_size = 20
        
        batch_size = initial_batch_size
        buffer_size = initial_buffer_size
        first_chunk_delivered = False

        context_start = time.time()
        if context:
            for segment in context:
                segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
                tokens.append(segment_tokens)
                tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        if prompt_tokens.size(0) > self.max_seq_len:
            prompt_tokens = prompt_tokens[-self.max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-self.max_seq_len:]

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        expected_frame_count = buffer_size 
        frame_buffer = []

        zeros_1_1 = torch.zeros(1, 1).long().to(self.device)
        zeros_mask_1_1 = torch.zeros(1, 1).bool().to(self.device)

        def update_tokens(sample):
            nonlocal curr_tokens, curr_tokens_mask, curr_pos
            ones = torch.ones_like(sample).bool()
            curr_tokens = torch.cat([sample, zeros_1_1], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([ones, zeros_mask_1_1], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        with self._audio_tokenizer.streaming(1):
            i = 0
            generation_start = time.time()

            while i < max_generation_len:
                batch_end = min(i + batch_size, max_generation_len)
                batch_size_actual = batch_end - i

                batch_samples = []

                for _ in range(batch_size_actual):
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                        if torch.cuda.is_available() and hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available"):
                            try:
                                torch.cuda.synchronize()  # Force sync before checking
                                if sample.numel() == 0 or torch.isnan(sample).any():
                                    print("Warning: Generated empty or NaN sample, stopping generation")
                                    break
                            except:
                                print("Error checking tensor, stopping generation")
                                break
                    if torch.all(sample == 0):
                        break

                    batch_samples.append(sample)
                    update_tokens(sample)

                if not batch_samples:
                    break

                frame_buffer.extend(batch_samples)
                i += len(batch_samples)

                if len(frame_buffer) >= buffer_size:
                    frames_to_process = frame_buffer[:expected_frame_count]
                    
                    # If we don't have enough frames, pad with zeros to match expected shape
                    if len(frames_to_process) < expected_frame_count:
                        # Create padding frames (zeros)
                        padding_frames = [
                            torch.zeros_like(frames_to_process[0]) 
                            for _ in range(expected_frame_count - len(frames_to_process))
                        ]
                        
                        # Combine actual frames with padding
                        frames_to_process = frames_to_process + padding_frames
                    
                    frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
                    audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
                    
                    # Keep remaining frames for next iteration
                    frame_buffer = frame_buffer[expected_frame_count:]
                    
                    # Process and yield the chunk
                    cpu_chunk = audio_chunk.cpu()
                    if on_chunk_generated:
                        on_chunk_generated(cpu_chunk)
                    
                    # After first chunk is delivered, switch to normal batch and buffer sizes
                    if not first_chunk_delivered:
                        batch_size = normal_batch_size
                        buffer_size = normal_buffer_size
                        expected_frame_count = buffer_size
                        first_chunk_delivered = True
                    
                    yield cpu_chunk

                    # Occasionally print progress and sync GPU
                    if i >= 100 and (i % 100 == 0):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        print(f"Generated {i} frames ({i * 0.08:.2f}s of audio)")

            # Process any remaining frames
            if frame_buffer:
                # Pad frame buffer if necessary
                if len(frame_buffer) < expected_frame_count:
                    padding_frames = [
                        torch.zeros_like(frame_buffer[0]) 
                        for _ in range(expected_frame_count - len(frame_buffer))
                    ]
                    frames_to_process = frame_buffer + padding_frames
                else:
                    # Otherwise take as many frames as possible that are a multiple of expected_frame_count
                    frames_multiple = (len(frame_buffer) // expected_frame_count) * expected_frame_count
                    frames_to_process = frame_buffer[:frames_multiple]
                    
                frames_stacked = torch.stack(frames_to_process).permute(1, 2, 0)
                audio_chunk = self._audio_tokenizer.decode(frames_stacked).squeeze(0).squeeze(0)
                
                # Determine actual audio length (before padding)
                actual_frames_percentage = min(len(frame_buffer), expected_frame_count) / expected_frame_count
                actual_samples = int(audio_chunk.shape[0] * actual_frames_percentage)
                
                # Return only the non-padded portion of audio if we added padding
                if len(frame_buffer) < expected_frame_count:
                    audio_chunk = audio_chunk[:actual_samples]
                    
                cpu_chunk = audio_chunk.cpu()
                if on_chunk_generated:
                    on_chunk_generated(cpu_chunk)
                yield cpu_chunk

            # Print final performance metrics
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time = time.time() - generation_start
            frames_generated = i
            audio_seconds = frames_generated * 0.08
            rtf = total_time / audio_seconds if audio_seconds > 0 else float('inf')
            print(f"Total time: {total_time:.2f}s")
            print(f"Generated {frames_generated} frames ({audio_seconds:.2f}s of audio)")
            print(f"Real-time factor: {rtf:.3f}x (target: <1.0)")

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.8,
        topk: int = 40,
        stream: bool = False,
        output_file: Optional[str] = None,
    ):
        """
        Generate audio with optional streaming and file output.
        
        Args:
            text: Text to generate audio for
            speaker: Speaker ID
            context: List of context segments
            max_audio_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            stream: Whether to use streaming generation
            output_file: If provided and stream=True, output will be saved to this file
        
        Returns:
            torch.Tensor: Generated audio tensor
        """
        if stream:
            if output_file:
                # Setup streaming to file
                write_chunk, close_wav = stream_audio_to_wav(output_file, self.sample_rate)
                
                # Collect chunks while streaming to file
                audio_chunks = []
                t1 = time.time()
                
                for i, chunk in enumerate(self.generate_stream(
                    text, speaker, context, max_audio_length_ms, temperature, topk
                )):
                    # Write to file
                    write_chunk(chunk)
                    # Store for return value
                    audio_chunks.append(chunk)
                    
                    # Occasionally print progress
                    if i % 5 == 0:
                        print(f"Part {i+1} available after {time.time() - t1:.4f}s")
                        t1 = time.time()
                
                # Close file
                close_wav()
                print(f"Streaming complete, WAV file saved to {output_file}")
            else:
                # Just collect chunks without file output
                audio_chunks = []
                for chunk in self.generate_stream(text, speaker, context, max_audio_length_ms, temperature, topk):
                    audio_chunks.append(chunk)
            
            if not audio_chunks:
                return torch.tensor([])
            return torch.cat(audio_chunks)

        # Non-streaming generation remains unchanged
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []

        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        max_seq_len = 2048
        if prompt_tokens.size(0) > max_seq_len:
            prompt_tokens = prompt_tokens[-max_seq_len:]
            prompt_tokens_mask = prompt_tokens_mask[-max_seq_len:]

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        samples = []
        with self._audio_tokenizer.streaming(1):
            for _ in range(max_generation_len):
                sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                if torch.all(sample == 0):
                    break
                samples.append(sample)

                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

        if not samples:
            return torch.tensor([])

        return self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

def warmup_generator(gen: Generator, warmup_text: str = "Hello, this is a comprehensive warmup text that will exercise the model's generation capabilities.", speaker_id: int = 0):
    """
    Perform an extremely aggressive warmup to drastically reduce first-generation latency.
    """
    print("Starting maximum-intensity warmup sequence...")
    
    # Directly access and optimize the model's internal state
    if hasattr(gen._model, 'backbone') and hasattr(gen._model.backbone, 'positional_embedding'):
        # Force calculation of position embeddings to ensure they're cached
        with torch.inference_mode():
            positions = torch.arange(0, 2048).to(gen.device)
            _ = gen._model.backbone.positional_embedding(positions)
    
    # Pre-allocate CUDA memory to prevent fragmentation during generation
    if torch.cuda.is_available():
        print("Optimizing GPU memory allocation...")
        # Try to reserve a large chunk of memory
        try:
            import math
            reserved_memory = []
            # Reserve multiple blocks of different sizes
            for size_mb in [128, 256, 512, 256, 128, 64]:
                size = int(size_mb * 1024 * 1024 / 4)  # Convert MB to float32 elements
                tensor_size = int(math.sqrt(size))
                tensor = torch.ones((tensor_size, tensor_size), device=gen.device, dtype=torch.float32)
                tensor = tensor * 1.0  # Force allocation
                reserved_memory.append(tensor)
            torch.cuda.synchronize()
            
            # Now free the memory
            for tensor in reserved_memory:
                del tensor
            reserved_memory = []
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Memory pre-allocation: {e}")
    
    # Create multiple dummy audio segments with varying characteristics
    print("Creating diverse audio contexts...")
    audio_segments = []
    
    # Create 3 different audio patterns
    for i in range(3):
        length = 24000 * (i + 1)  # 1s, 2s, 3s
        audio = torch.zeros(length).to(gen.device)
        
        # Add different patterns to each segment
        if i == 0:
            # Sine wave pattern
            import math
            t = torch.linspace(0, 8 * math.pi, length).to(gen.device)
            audio = torch.sin(t) * 0.1
        elif i == 1:
            # Random noise pattern
            audio = torch.randn(length).to(gen.device) * 0.05
        else:
            # Pulse pattern
            audio[::800] = 0.2
            audio[::801] = -0.2
        
        segment = Segment(
            speaker=speaker_id,
            text=f"Warmup segment {i+1} with {length/24000:.1f}s of audio.",
            audio=audio
        )
        audio_segments.append(segment)
    
    # Force compilation of critical model components
    print("Forcing compilation of critical components...")
    
    # Directly exercise the audio tokenizer with real data
    with torch.inference_mode():
        for segment in audio_segments:
            # Force tokenization of both text and audio
            gen._tokenize_segment(segment)
    
    # Exercise the model's generation capabilities directly
    with torch.inference_mode():
        
        # Generate some sample frames to ensure model is compiled
        dummy_tokens = torch.ones(1, 10, gen._num_codebooks+1).long().to(gen.device)
        dummy_mask = torch.ones(1, 10, gen._num_codebooks+1).bool().to(gen.device)
        dummy_pos = torch.arange(0, 10).unsqueeze(0).to(gen.device)
        
        # Generate multiple frames with different parameters
        for temp in [0.6, 0.7, 0.8]:
            for topk in [20, 30, 40]:
                _ = gen._model.generate_frame(dummy_tokens, dummy_mask, dummy_pos, temp, topk)
    
    gen._text_token_cache.clear()
    
    print("Running final generation with exact same setup as a real request...")
    
    final_text = "This is the final warmup that exactly matches a real generation request."
    
    # First tokenize the text - to fill the cache
    gen._tokenize_text_segment(final_text, speaker_id)
    
    try:
        # Now run a complete generation with a single context segment
        generate_streaming_audio(
            generator=gen,
            text=final_text, 
            speaker=speaker_id,
            context=[audio_segments[0]],  # Just one context segment
            output_file="warmup_final.wav",
            max_audio_length_ms=6000,
            temperature=0.7,
            topk=30,
            play_audio=False
        )
    except Exception as e:
        print(f"Final warmup run exception (ignorable): {e}")
    
    # Force final synchronization and memory optimization
    if torch.cuda.is_available():
        print("Final GPU optimization...")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        try:
            # Allocate a large tensor to force compaction
            large_tensor = torch.empty(int(1e9//4), dtype=torch.float, device=gen.device)
            # Immediately delete it
            del large_tensor
        except RuntimeError:
            # Expected if there's not enough memory
            pass
            
        # Final cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("Maximum-intensity warmup complete. First generation should now be MUCH faster.")

def stream_audio_to_wav(filename, sample_rate):
    """
    Initialize a WAV writer for streaming audio chunks.
    
    Args:
        filename: Output WAV file path
        sample_rate: Audio sample rate in Hz
    
    Returns:
        tuple: (write_chunk, close) functions for writing audio data and closing the file
    """
    # Create a WAV file with the proper header
    wav_file = wave.open(filename, 'wb')
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(sample_rate)
    
    def write_chunk(audio_chunk):
        # Convert tensor to numpy and then to int16 PCM format
        if isinstance(audio_chunk, torch.Tensor):
            # Ensure it's on CPU and detached before converting to numpy
            audio_np = audio_chunk.detach().cpu().numpy()
        else:
            audio_np = audio_chunk
            
        # Normalize if needed (assuming audio is in [-1, 1] range)
        if audio_np.max() <= 1.0 and audio_np.min() >= -1.0:
            audio_int = (audio_np * 32767).astype(np.int16)
        else:
            audio_int = audio_np.astype(np.int16)
            
        # Write to WAV file
        wav_file.writeframes(audio_int.tobytes())
    
    def close():
        wav_file.close()
        
    return write_chunk, close

def generate_streaming_audio(
    generator: Generator,
    text: str,
    speaker: int,
    context: List[Segment],
    output_file: str,
    max_audio_length_ms: float = 90_000,
    temperature: float = 1.0,
    topk: int = 50,
    play_audio: bool = False,
):
    """
    Generate audio with streaming output and comprehensive timing metrics.
    Optimized for reduced first-chunk latency.
    """
    # Initialize the streaming WAV writer
    write_chunk, close_wav = stream_audio_to_wav(output_file, generator.sample_rate)
    
    # Set up audio playback if requested
    audio_queue = queue.Queue(maxsize=100) if play_audio else None
    stop_event = threading.Event()
    
    if play_audio:
        try:
            import sounddevice as sd
            
            # Get available sample rates for default output device to check compatibility
            device_info = sd.query_devices(kind='output')
            supported_rate = device_info.get('default_samplerate', 44100)
            need_resampling = abs(supported_rate - generator.sample_rate) > 100
            
            if need_resampling:
                try:
                    # Use resampling if sample rate doesn't match
                    import librosa
                    print(f"Resampling from {generator.sample_rate}Hz to {int(supported_rate)}Hz for playback")
                    
                    def audio_playback_worker():
                        while not stop_event.is_set() or not audio_queue.empty():
                            try:
                                chunk = audio_queue.get(timeout=0.5)
                                if isinstance(chunk, torch.Tensor) and chunk.numel() == 0:
                                    audio_queue.task_done()
                                    continue
                                    
                                audio_np = chunk.numpy() if isinstance(chunk, torch.Tensor) else chunk
                                
                                # Skip very short chunks (likely noise)
                                if len(audio_np) < 100:
                                    audio_queue.task_done()
                                    continue
                                    
                                # Resample to device's supported rate
                                resampled = librosa.resample(
                                    audio_np, 
                                    orig_sr=generator.sample_rate, 
                                    target_sr=int(supported_rate)
                                )
                                sd.play(resampled, supported_rate, blocking=True)
                                # Add a small delay to ensure audio finishes playing
                                time.sleep(0.05)
                                audio_queue.task_done()
                            except queue.Empty:
                                # If queue empty but not stopping, keep trying
                                if not stop_event.is_set():
                                    continue
                                else:
                                    break
                            except Exception as e:
                                print(f"Playback error: {e}")
                                audio_queue.task_done()
                except ImportError:
                    print("Librosa not found. Using direct playback which may cause sample rate warnings.")
                    need_resampling = False
            
            if not need_resampling:
                def audio_playback_worker():
                    while not stop_event.is_set() or not audio_queue.empty():
                        try:
                            chunk = audio_queue.get(timeout=0.5)
                            if isinstance(chunk, torch.Tensor) and chunk.numel() == 0:
                                audio_queue.task_done()
                                continue
                                
                            audio_np = chunk.numpy() if isinstance(chunk, torch.Tensor) else chunk
                            
                            # Skip very short chunks (likely noise)
                            if len(audio_np) < 100:
                                audio_queue.task_done()
                                continue
                                
                            sd.play(audio_np, generator.sample_rate, blocking=True)
                            # Add a small delay to ensure audio finishes playing
                            time.sleep(0.05)
                            audio_queue.task_done()
                        except queue.Empty:
                            # If queue empty but not stopping, keep trying
                            if not stop_event.is_set():
                                continue
                            else:
                                break
                        except Exception as e:
                            print(f"Playback error: {e}")
                            audio_queue.task_done()
            
            # Start playback thread
            playback_thread = threading.Thread(target=audio_playback_worker, daemon=False)
            playback_thread.start()
            
        except ImportError:
            print("sounddevice library not found. Install with 'pip install sounddevice' for real-time playback.")
            play_audio = False
    
    # Timing metrics
    chunk_times = []
    latency_to_first_chunk = None
    total_audio_duration = 0
    chunk_count = 0
    
    # Function to handle each generated chunk
    def on_chunk_generated(chunk):
        nonlocal chunk_count, latency_to_first_chunk, total_audio_duration
        
        current_time = time.time()
        if chunk_count == 0:
            latency_to_first_chunk = current_time - start_time
            print(f"First chunk latency: {latency_to_first_chunk*1000:.1f}ms")
            
        # Save chunk to WAV file
        write_chunk(chunk)
        
        # Update metrics
        chunk_count += 1
        chunk_duration = len(chunk) / generator.sample_rate
        total_audio_duration += chunk_duration
        chunk_times.append(current_time)
        
        # Send to audio player if enabled
        if play_audio and audio_queue is not None:
            try:
                audio_queue.put(chunk, timeout=1.0)
            except queue.Full:
                pass  # Skip if queue is full to avoid blocking
    
    if torch.cuda.is_available():
        print("Preparing GPU for low-latency generation...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Pre-allocate some GPU memory to avoid allocation during generation
        dummy_tensors = []
        for i in range(5):
            dummy = torch.ones((100, 100), device=generator.device)
            dummy = dummy + 1.0  # Force computation
            dummy_tensors.append(dummy)  # Keep reference to prevent deallocation
            
        torch.cuda.synchronize()
    
    # Set process priority to improve performance - use higher priority
    try:
        import psutil
        process = psutil.Process()
        if platform.system() == 'Windows':
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            process.nice(-1)
    except (ImportError, PermissionError, psutil.AccessDenied):
        pass
    
    print(f"Starting audio generation for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    start_time = time.time()
    
    # Generate audio in chunks, catching possible errors
    frame_count = 0
    audio_chunks = []  # Store all chunks for possible use at the end
    
    try:
        for audio_chunk in generator.generate_stream(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
            on_chunk_generated=on_chunk_generated
        ):
            frame_count += 1
            audio_chunks.append(audio_chunk)  # Store the chunk
            
            # Print timing info less frequently to reduce overhead
            if frame_count % 10 == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                if total_audio_duration > 0:
                    rtf = elapsed / total_audio_duration
                    remaining_time = (max_audio_length_ms/1000 - total_audio_duration) * rtf
                    print(f"Chunk {chunk_count}: {total_audio_duration:.1f}s audio in {elapsed:.1f}s "
                          f"(RTF: {rtf:.2f}x, Est. remaining: {remaining_time:.1f}s)")
    except Exception as e:
        print(f"Error during audio generation: {e}")
        import traceback
        traceback.print_exc()
    
    # Release dummy tensors to free memory
    if 'dummy_tensors' in locals():
        del dummy_tensors
    
    # Ensure all chunks are properly processed
    if play_audio and audio_queue is not None:
        print("Waiting for playback queue to finish...")
        try:
            timeout_start = time.time()
            while not audio_queue.empty() and time.time() - timeout_start < 5.0:
                time.sleep(0.1)
        except:
            pass
    
    # Add a small delay to ensure everything is processed
    time.sleep(0.5)
    
    # Signal audio worker that generation is complete
    stop_event.set()
    
    # Close WAV file
    close_wav()
    
    # Wait for audio playback to complete if enabled
    if play_audio and 'playback_thread' in locals():
        print("Waiting for audio playback to complete...")
        
        # First, ensure the queue is empty
        try:
            timeout_start = time.time()
            while not audio_queue.empty() and time.time() - timeout_start < 5.0:
                time.sleep(0.1)
        except:
            pass
            
        # Set a flag to indicate complete audio playback is needed
        if hasattr(sd, 'wait'):
            try:
                sd.wait()
            except:
                pass
                
        # Join the playback thread with timeout
        playback_thread.join(timeout=5.0)
        
        # Force sounddevice to stop if it's still playing
        try:
            sd.stop()
        except:
            pass
    
    # Calculate and print detailed performance metrics
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    # Calculate inter-chunk latency
    if len(chunk_times) > 1:
        inter_chunk_latencies = [chunk_times[i] - chunk_times[i-1] for i in range(1, len(chunk_times))]
        avg_inter_chunk_latency = sum(inter_chunk_latencies) / len(inter_chunk_latencies)
        max_inter_chunk_latency = max(inter_chunk_latencies) if inter_chunk_latencies else 0
        min_inter_chunk_latency = min(inter_chunk_latencies) if inter_chunk_latencies else 0
    else:
        avg_inter_chunk_latency = max_inter_chunk_latency = min_inter_chunk_latency = 0
    
    rtf = total_elapsed / total_audio_duration if total_audio_duration > 0 else float('inf')
    
    print("\n" + "="*50)
    print("AUDIO GENERATION PERFORMANCE METRICS")
    print("="*50)
    print(f"First chunk latency: {latency_to_first_chunk*1000:.1f}ms")
    print(f"Total generation time: {total_elapsed:.2f}s")
    print(f"Audio duration: {total_audio_duration:.2f}s")
    print(f"Real-time factor (RTF): {rtf:.3f}x (target: <1.0)")
    print(f"Number of chunks: {chunk_count}")
    print(f"Average chunk size: {(total_audio_duration/chunk_count)*1000:.1f}ms") if chunk_count > 0 else None
    print(f"Average inter-chunk latency: {avg_inter_chunk_latency*1000:.1f}ms")
    print(f"Min/Max inter-chunk latency: {min_inter_chunk_latency*1000:.1f}ms / {max_inter_chunk_latency*1000:.1f}ms")
    print(f"Chunks per second: {chunk_count/total_elapsed:.2f}")
    print(f"Output file: {output_file}")
    print("="*50)