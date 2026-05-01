import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM

from coral.datasets.gc_dataset import SyntheticGCDataset


class GenomicGCModel(nn.Module):
    def __init__(self, backbone_name="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
                 device="cuda"):
        super().__init__()
        self.device = device
        self.backbone = AutoModelForMaskedLM.from_pretrained(
            backbone_name, trust_remote_code=True, output_hidden_states=True
        )
        self.backbone.to(device)
        self.backbone.eval()
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.head.to(device)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.hidden_states[-1].size()).float()
            sum_embeddings = torch.sum(outputs.hidden_states[-1] * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = outputs.hidden_states[-1].mean(dim=1)
        return self.head(pooled).squeeze(-1)


def train_head(num_samples=25000, batch_size=512, epochs=5,
               backbone_name="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"):
    """Train the GC prediction head on synthetic sequences."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = SyntheticGCDataset(num_samples=num_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)
    model = GenomicGCModel(device=device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    print("Training GC Head")
    for epoch in range(epochs):
        total_loss = 0
        for seqs, targets in loader:
            targets = targets.to(device).float()
            inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True).to(device)
            preds = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | MAE Loss: {total_loss/len(loader):.5f}")

    return model, tokenizer
