import torch
from torch import nn
from transformers import VisualBertModel, VisualBertConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VisualBert(nn.Module):
    """
    VisualBert Classification Model
    vocab_size    = tokenizer length
    encoder_layer = 6
    n_heads       = 8
    num_class     = number of class in dataset
    """
    def __init__(self, vocab_size, layers, n_heads, num_class=59):
        super(VisualBert, self).__init__()
        VBconfig = VisualBertConfig(vocab_size=vocab_size, visual_embedding_dim=512, num_hidden_layers=layers,
                                    num_attention_heads=n_heads, hidden_size=2048)
        self.VisualBertEncoder = VisualBertModel(VBconfig)
        self.classifier = nn.Linear(VBconfig.hidden_size, num_class)

        self.dropout = nn.Dropout(VBconfig.hidden_dropout_prob)
        self.num_labels = num_class

    def forward(self, inputs, visual_embeds):
        # prepare visual embedding
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(device)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)

        # append visual features to text
        inputs.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        "output_attentions": True
                        })
        
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['visual_token_type_ids'] = inputs['visual_token_type_ids'].to(device)
        inputs['visual_attention_mask'] = inputs['visual_attention_mask'].to(device)

        '----------------- VQA -----------------'
        index_to_gather = inputs['attention_mask'].sum(1) - 2
        
        outputs = self.VisualBertEncoder(**inputs)
        sequence_output = outputs[0]

        index_to_gather = (index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(index_to_gather.size(0), 1,
                                                                              sequence_output.size(-1)))

        pooled_output = torch.gather(sequence_output, 1, index_to_gather)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_labels)
        return reshaped_logits
