"""Optional item-text ablation. The original SASRec implementation is unchanged."""
import numpy as np
import torch
from torch import nn
from src.models.sasrec import SASRec


class FusedItemEmbedding(nn.Module):
    """Tied input/output vectors: ID + projection(frozen text), with zero padding."""
    def __init__(self, ids, text):
        super().__init__()
        text = torch.as_tensor(np.asarray(text).copy(), dtype=torch.float32)
        if (text.ndim != 2 or text.shape[0] != ids.num_embeddings or text.shape[1] < 1
                or not torch.isfinite(text).all() or torch.count_nonzero(text[0])):
            raise ValueError('Text must be finite [n_items + 1, dimension], with row 0 zero')
        self.ids = ids
        self.register_buffer('text', text)
        self.projection = nn.Linear(text.shape[1], ids.embedding_dim, bias=False)
        # Starts exactly at Model 0 for the same random seed; text learns through
        # this projection from the first update. A documented ablation choice.
        nn.init.zeros_(self.projection.weight)

    @property
    def weight(self):
        fused = self.ids.weight + self.projection(self.text)
        # Tied output weights otherwise backpropagate into padding row 0 even
        # though nn.Embedding suppresses its input-lookup gradient.
        return torch.cat((torch.zeros_like(fused[:1]), fused[1:]), dim=0)

    def forward(self, tokens):
        return self.ids(tokens) + self.projection(self.text[tokens])


class TextSASRec(SASRec):
    def __init__(self, n_items, text_embeddings, max_len=50, hidden_size=64,
                 num_heads=4, num_layers=2, dropout=0.2):
        super().__init__(n_items, max_len, hidden_size, num_heads, num_layers, dropout)
        self.item_embedding = FusedItemEmbedding(self.item_embedding, text_embeddings)
        self.architecture = dict(n_items=n_items, text_dimension=self.item_embedding.text.shape[1],
                                 max_len=max_len, hidden_size=hidden_size, num_heads=num_heads,
                                 num_layers=num_layers, dropout=dropout)


def load_text_checkpoint(path, device='cpu'):
    """Load a trusted project checkpoint, including its frozen text buffer."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    if metadata.get('model_class') != 'TextSASRec':
        raise ValueError('Not a TextSASRec checkpoint')
    architecture = dict(metadata['text_architecture'])
    dimension = architecture.pop('text_dimension')
    text = checkpoint['model']['item_embedding.text']
    if tuple(text.shape) != (architecture['n_items'] + 1, dimension):
        raise ValueError('Checkpoint text dimensions differ from metadata')
    model = TextSASRec(text_embeddings=text.numpy(), **architecture)
    model.load_state_dict(checkpoint['model'], strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('Nonfinite checkpoint weights')
    if torch.count_nonzero(model.item_embedding.ids.weight[0]):
        raise ValueError('Checkpoint padding row must be zero')
    return model.to(device).eval(), checkpoint
