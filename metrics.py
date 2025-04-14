import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer
import string
from torchmetrics.text import WordErrorRate


def compute_wer(audio, reference_text, sample_rate=24000, language="en"):
    """
    Compute Word Error Rate (WER) for a given audio and reference text.
    
    Args:
        audio: Audio as a numpy array or torch tensor
        reference_text: Ground truth transcription
        sample_rate: Sample rate of the audio (default: 24000)
        language: Language code (default: "en" for English)
        
    Returns:
        float: Word Error Rate (WER)
    """
    # Convert torch tensor to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().cpu().numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load ASR model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-small", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    processor = AutoProcessor.from_pretrained("openai/whisper-small")
    
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device
    )
    
    result = asr_pipeline({"raw": audio, "sampling_rate": sample_rate, "generate_kwargs": {"language": language}})
    recognized_text = result["text"]
    
    # Choose appropriate normalizer based on language
    normalizer = EnglishTextNormalizer() if language == "en" else BasicTextNormalizer()
    ref_normalized = normalize_text(reference_text, normalizer)
    rec_normalized = normalize_text(recognized_text, normalizer)
    
    # Calculate WER using torchmetrics
    wer_metric = WordErrorRate()
    wer = wer_metric([rec_normalized], [ref_normalized]).item()
    
    return wer


def normalize_text(text, normalizer):
    """Remove punctuation and normalize text."""
    remove_punc = string.punctuation.replace("'", "")  # keep apostrophes
    for punctuation in remove_punc:
        text = text.replace(punctuation, "")
    text = normalizer(text)
    return text.lower().strip()


def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate between reference and hypothesis texts."""
    # Split into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Initialize the matrix
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.uint32)
    
    # Fill the first row and column
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    
    # Computation
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                substitution = d[i-1, j-1] + 1
                insertion = d[i, j-1] + 1
                deletion = d[i-1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    
    # The last element contains the Levenshtein distance
    distance = d[len(ref_words), len(hyp_words)]
    
    # Calculate WER
    if len(ref_words) > 0:
        wer = float(distance) / len(ref_words)
    else:
        wer = 0 if len(hyp_words) == 0 else float('inf')
    
    return wer

