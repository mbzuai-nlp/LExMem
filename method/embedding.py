"""Embeddings with gte-Qwen2-1.5B-instruct (local checkpoint) and an on-disk cache.

Requires transformers 4.x (tested with 4.44.2): the checkpoint ships its own modeling_qwen.py
with bidirectional attention.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading

QUERY_INSTRUCTION = (
    'Given a question about a conversation, retrieve the memory unit summary '
    'or relation description most relevant to answering it.'
)


def query_text(text):
    return f'Instruct: {QUERY_INSTRUCTION}\nQuery: {text}'


def _key(kind, text):
    return hashlib.sha256(f'{kind}\n{text}'.encode()).hexdigest()


def last_token_pool(hidden, attention_mask):
    """Hidden state of the last non-padding token (left or right padding)."""
    import torch
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(1).values
    if (last < 0).any():
        raise ValueError('Cannot pool an empty token sequence')
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class GTEEncoder:
    def __init__(self, model_path, device='auto', dtype='auto', batch_size=8, max_length=8192):
        import torch
        from transformers import AutoConfig, AutoTokenizer, __version__ as transformers_version

        if int(transformers_version.split('.')[0]) >= 5:
            raise RuntimeError('gte-Qwen2-1.5B-instruct needs transformers 4.x (e.g. 4.44.2)')
        self.torch = torch
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if dtype == 'auto':
            dtype = ('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') \
                if self.device.startswith('cuda') else 'float32'
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        # Load the checkpoint's own modeling_qwen.py (bidirectional attention) directly.
        source = Path(model_path) / 'modeling_qwen.py'
        source_text = source.read_text()
        module_name = 'gte_qwen_' + hashlib.sha256(source_text.encode()).hexdigest()[:16]
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            exec(compile(source_text, str(source), 'exec'), module.__dict__)
        model_class = sys.modules[module_name].Qwen2Model
        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        self.model = model_class.from_pretrained(
            str(model_path), config=config, local_files_only=True,
            torch_dtype=getattr(torch, dtype), attn_implementation='sdpa').to(self.device).eval()

    def encode(self, texts):
        import numpy as np
        torch = self.torch
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start + self.batch_size],
                                   padding=True, truncation=False, return_tensors='pt')
            if batch['input_ids'].shape[1] > self.max_length:
                raise ValueError(f'Embedding input exceeds --embedding-max-length={self.max_length}')
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                output = self.model(**batch, use_cache=False, is_causal=False)
                pooled = last_token_pool(output.last_hidden_state, batch['attention_mask'])
                normalized = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.append(normalized.cpu().numpy())
        return np.concatenate(vectors, axis=0)

    def close(self):
        del self.model
        if self.device.startswith('cuda'):
            self.torch.cuda.empty_cache()


class Embedder:
    """Cached embeddings; the model is loaded only on a cache miss."""

    def __init__(self, cache_path, model_path, **encoder_kwargs):
        import numpy as np
        self._np = np
        self.cache_path = Path(cache_path)
        self.model_path = model_path
        self.encoder_kwargs = encoder_kwargs
        self._encoder = None
        self._vectors = {}
        self._lock = threading.Lock()
        if self.cache_path.exists():
            with np.load(self.cache_path, allow_pickle=False) as cached:
                for key, vector in zip(cached['keys'], cached['vectors']):
                    self._vectors[str(key)] = vector

    def _save(self):
        np = self._np
        keys = list(self._vectors)
        vectors = np.stack([self._vectors[k] for k in keys]) if keys else np.zeros((0, 1), dtype=np.float32)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix('.tmp.npz')
        with tmp.open('wb') as handle:
            np.savez(handle, keys=np.array(keys), vectors=vectors)
        tmp.replace(self.cache_path)

    def _embed(self, texts, kind, wrap):
        np = self._np
        with self._lock:  # workers share one model and one cache file
            keys = [_key(kind, text) for text in texts]
            missing = [i for i, key in enumerate(keys) if key not in self._vectors]
            if missing:
                if self._encoder is None:
                    self._encoder = GTEEncoder(self.model_path, **self.encoder_kwargs)
                vectors = np.asarray(self._encoder.encode([wrap(texts[i]) for i in missing]), dtype=np.float32)
                if vectors.shape[0] != len(missing) or not np.isfinite(vectors).all():
                    raise ValueError('Invalid embeddings')
                for i, vector in zip(missing, vectors):
                    self._vectors[keys[i]] = vector
                self._save()
            return [self._vectors[key].tolist() for key in keys]

    def embed_documents(self, texts):
        """Unit Keys: embedded without an instruction."""
        return self._embed(texts, 'document', lambda t: t)

    def embed_queries(self, texts):
        """Questions: embedded with the retrieval instruction."""
        return self._embed(texts, 'query', query_text)

    def close(self):
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None
