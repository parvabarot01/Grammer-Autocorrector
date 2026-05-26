"""RNN sequence-to-sequence baseline for grammar correction."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, List, Mapping, MutableSequence, Tuple

LOGGER = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional at import time
    torch = None
    nn = None


def _require_torch() -> Any:
    """Return the torch module or raise an informative error."""

    if torch is None or nn is None:
        raise ImportError(
            "torch is required for the RNN baseline. "
            "Install it with `pip install torch`."
        )
    return torch


class Encoder(nn.Module if nn is not None else object):
    """Bidirectional LSTM encoder."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the encoder network."""

        _require_torch()
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, source_tokens: Any) -> Tuple[Any, Any, Any]:
        """Encode source tokens into contextual states."""

        embedded = self.dropout(self.embedding(source_tokens))
        outputs, (hidden, cell) = self.lstm(embedded)
        projected_outputs = torch.tanh(self.output_projection(outputs))

        hidden_layers = []
        cell_layers = []
        for layer_index in range(self.num_layers):
            forward_hidden = hidden[2 * layer_index]
            backward_hidden = hidden[2 * layer_index + 1]
            forward_cell = cell[2 * layer_index]
            backward_cell = cell[2 * layer_index + 1]

            hidden_layers.append(
                torch.tanh(
                    self.hidden_projection(
                        torch.cat([forward_hidden, backward_hidden], dim=1)
                    )
                )
            )
            cell_layers.append(
                torch.tanh(
                    self.hidden_projection(
                        torch.cat([forward_cell, backward_cell], dim=1)
                    )
                )
            )

        projected_hidden = torch.stack(hidden_layers, dim=0)
        projected_cell = torch.stack(cell_layers, dim=0)
        return projected_outputs, projected_hidden, projected_cell


class Attention(nn.Module if nn is not None else object):
    """Bahdanau attention mechanism."""

    def __init__(self, hidden_dim: int) -> None:
        """Initialize the attention layers."""

        _require_torch()
        super().__init__()
        self.encoder_projection = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_projection = nn.Linear(hidden_dim, hidden_dim)
        self.score_projection = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_state: Any, encoder_outputs: Any, mask: Any = None) -> Any:
        """Compute attention weights over encoder outputs."""

        hidden_expanded = self.decoder_projection(hidden_state).unsqueeze(1)
        encoder_projected = self.encoder_projection(encoder_outputs)
        scores = self.score_projection(torch.tanh(encoder_projected + hidden_expanded))
        scores = scores.squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        return torch.softmax(scores, dim=-1)


class Decoder(nn.Module if nn is not None else object):
    """LSTM decoder with attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the decoder network."""

        _require_torch()
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_dim * 2 + embed_dim, vocab_size)

    def forward(
        self,
        input_token: Any,
        hidden: Any,
        cell: Any,
        encoder_outputs: Any,
        mask: Any = None,
    ) -> Tuple[Any, Any, Any, Any]:
        """Decode one step and return logits, hidden state, and attention weights."""

        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        attention_weights = self.attention(hidden[-1], encoder_outputs, mask)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat([embedded, context], dim=-1)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        logits = self.output_projection(
            torch.cat(
                [output.squeeze(1), context.squeeze(1), embedded.squeeze(1)],
                dim=1,
            )
        )
        return logits, hidden, cell, attention_weights


class RNNSeq2Seq(nn.Module if nn is not None else object):
    """Full encoder-decoder architecture with teacher forcing."""

    def __init__(self, encoder: Encoder, decoder: Decoder, device: str) -> None:
        """Initialize the seq2seq container."""

        _require_torch()
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        source_tokens: Any,
        target_tokens: Any,
        teacher_forcing_ratio: float = 0.5,
        pad_idx: int = 0,
    ) -> Any:
        """Run the seq2seq model for a full target sequence."""

        batch_size, target_length = target_tokens.shape
        vocab_size = self.decoder.output_projection.out_features
        outputs = torch.zeros(batch_size, target_length, vocab_size, device=self.device)
        encoder_outputs, hidden, cell = self.encoder(source_tokens)
        decoder_input = target_tokens[:, 0]
        mask = (source_tokens != pad_idx).int()

        for step in range(1, target_length):
            logits, hidden, cell, _ = self.decoder(
                decoder_input,
                hidden,
                cell,
                encoder_outputs,
                mask,
            )
            outputs[:, step, :] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            next_tokens = logits.argmax(dim=1)
            decoder_input = target_tokens[:, step] if teacher_force else next_tokens

        return outputs


class RNNGrammarCorrector:
    """Word-level RNN baseline for grammar correction experiments."""

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        device: str = "auto",
    ) -> None:
        """Initialize the baseline model configuration."""

        self.vocab_size_limit = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = self._resolve_device(device)
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.token_to_index = {
            token: index for index, token in enumerate(self.special_tokens)
        }
        self.index_to_token = list(self.special_tokens)
        self.model: Any = None
        self._rebuild_model_if_possible()

    def build_vocabulary(self, texts: List[str]) -> None:
        """Build a word-level vocabulary from training texts.

        Args:
            texts: Training corpus used to build the baseline vocabulary.
        """

        token_counts = Counter()
        for text in texts:
            token_counts.update(self._tokenize(text))

        most_common = token_counts.most_common(
            max(self.vocab_size_limit - len(self.special_tokens), 0)
        )
        self.index_to_token = list(self.special_tokens) + [
            token for token, _ in most_common
        ]
        self.token_to_index = {
            token: index for index, token in enumerate(self.index_to_token)
        }
        self._rebuild_model_if_possible()

    def encode_text(self, text: str) -> Any:
        """Encode text into a tensor of token ids."""

        torch_module = _require_torch()
        tokens = ["<SOS>"] + self._tokenize(text) + ["<EOS>"]
        token_ids = [
            self.token_to_index.get(token, self.token_to_index["<UNK>"])
            for token in tokens
        ]
        return torch_module.tensor(
            token_ids,
            dtype=torch_module.long,
            device=self.device,
        )

    def decode_tensor(self, tensor: Any) -> str:
        """Decode a tensor of token ids back into text."""

        ids = tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)
        tokens: MutableSequence[str] = []
        for index in ids:
            token = self.index_to_token[int(index)]
            if token == "<EOS>":
                break
            if token in {"<PAD>", "<SOS>"}:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def correct(self, text: str, max_length: int = 128) -> str:
        """Generate a greedy baseline correction for a single sentence."""

        torch_module = _require_torch()
        if self.model is None:
            raise RuntimeError("The RNN baseline model has not been initialized.")

        self.model.eval()
        source = self.encode_text(text).unsqueeze(0)
        with torch_module.no_grad():
            encoder_outputs, hidden, cell = self.model.encoder(source)
            decoder_input = torch_module.tensor(
                [self.token_to_index["<SOS>"]],
                dtype=torch_module.long,
                device=self.device,
            )
            generated_ids: List[int] = []
            mask = (source != self.token_to_index["<PAD>"]).int()

            for _ in range(max_length):
                logits, hidden, cell, _ = self.model.decoder(
                    decoder_input,
                    hidden,
                    cell,
                    encoder_outputs,
                    mask,
                )
                next_token = int(logits.argmax(dim=1).item())
                if next_token == self.token_to_index["<EOS>"]:
                    break
                generated_ids.append(next_token)
                decoder_input = torch_module.tensor(
                    [next_token],
                    dtype=torch_module.long,
                    device=self.device,
                )

        return self.decode_tensor(generated_ids)

    def train_epoch(self, dataloader: Any, optimizer: Any, criterion: Any) -> float:
        """Run one teacher-forced training epoch."""

        _require_torch()
        if self.model is None:
            raise RuntimeError("The RNN baseline model has not been initialized.")

        self.model.train()
        total_loss = 0.0
        batch_count = 0
        for batch in dataloader:
            source_tokens, target_tokens = self._unpack_batch(batch)
            optimizer.zero_grad()
            outputs = self.model(
                source_tokens,
                target_tokens,
                teacher_forcing_ratio=0.5,
                pad_idx=self.token_to_index["<PAD>"],
            )
            logits = outputs[:, 1:, :].reshape(-1, outputs.size(-1))
            labels = target_tokens[:, 1:].reshape(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1

        return total_loss / max(batch_count, 1)

    def evaluate_epoch(self, dataloader: Any, criterion: Any) -> float:
        """Evaluate the baseline for one epoch without teacher forcing noise."""

        torch_module = _require_torch()
        if self.model is None:
            raise RuntimeError("The RNN baseline model has not been initialized.")

        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        with torch_module.no_grad():
            for batch in dataloader:
                source_tokens, target_tokens = self._unpack_batch(batch)
                outputs = self.model(
                    source_tokens,
                    target_tokens,
                    teacher_forcing_ratio=0.0,
                    pad_idx=self.token_to_index["<PAD>"],
                )
                logits = outputs[:, 1:, :].reshape(-1, outputs.size(-1))
                labels = target_tokens[:, 1:].reshape(-1)
                loss = criterion(logits, labels)
                total_loss += float(loss.item())
                batch_count += 1

        return total_loss / max(batch_count, 1)

    def save(self, output_dir: str) -> None:
        """Save model weights and vocabulary metadata."""

        torch_module = _require_torch()
        if self.model is None:
            raise RuntimeError("Cannot save before building the RNN model.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch_module.save(self.model.state_dict(), output_path / "rnn_baseline.pt")
        metadata = {
            "vocab_size_limit": self.vocab_size_limit,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "device": self.device,
            "index_to_token": self.index_to_token,
        }
        (output_path / "rnn_baseline.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def load(self, model_dir: str) -> None:
        """Load weights and vocabulary from a saved checkpoint."""

        torch_module = _require_torch()
        metadata_path = Path(model_dir) / "rnn_baseline.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing RNN metadata file: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.vocab_size_limit = int(metadata["vocab_size_limit"])
        self.embed_dim = int(metadata["embed_dim"])
        self.hidden_dim = int(metadata["hidden_dim"])
        self.num_layers = int(metadata["num_layers"])
        self.dropout = float(metadata["dropout"])
        self.index_to_token = list(metadata["index_to_token"])
        self.token_to_index = {
            token: index for index, token in enumerate(self.index_to_token)
        }
        self._rebuild_model_if_possible()
        if self.model is None:
            raise RuntimeError("The RNN baseline model could not be initialized.")

        state_dict = torch_module.load(
            Path(model_dir) / "rnn_baseline.pt",
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def _resolve_device(self, device: str) -> str:
        """Resolve the runtime device string."""

        if device != "auto":
            return device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _rebuild_model_if_possible(self) -> None:
        """Construct the seq2seq module when torch is available."""

        if torch is None or nn is None:
            self.model = None
            return
        encoder = Encoder(
            vocab_size=len(self.index_to_token),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        decoder = Decoder(
            vocab_size=len(self.index_to_token),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.model = RNNSeq2Seq(encoder, decoder, self.device).to(self.device)

    def _tokenize(self, text: str) -> List[str]:
        """Split text into simple whitespace-delimited tokens."""

        return text.strip().split()

    def _unpack_batch(self, batch: Any) -> Tuple[Any, Any]:
        """Normalize a dataloader batch into `(source, target)` tensors."""

        if isinstance(batch, Mapping):
            source_tokens = batch["source"].to(self.device)
            target_tokens = batch["target"].to(self.device)
            return source_tokens, target_tokens
        source_tokens, target_tokens = batch
        return source_tokens.to(self.device), target_tokens.to(self.device)
