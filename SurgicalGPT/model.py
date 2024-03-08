import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from transformers import VisualBertConfig, GPT2Config, GPT2Tokenizer
from transformers import VisualBertModel, GPT2Model, ViTModel, SwinModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SurgicalGPT(nn.Module):
    def __init__(self, num_class=59, vis_pos_emb=None):
        super(SurgicalGPT, self).__init__()
        # use default setting
        self.vis_pos_emb = vis_pos_emb
        
        # image processing
        self.img_feature_extractor = models.resnet18(pretrained=True)
        new_fc = nn.Sequential(*list(self.img_feature_extractor.fc.children())[:-1])
        self.img_feature_extractor.fc = new_fc

        # visual embedding
        VB_config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        VB_config.visual_embedding_dim = 512
        visual_bert = VisualBertModel(config=VB_config)
        self.visual_embedder = visual_bert.embeddings.visual_projection

        # question embedding
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        question_embedder = GPT2Model.from_pretrained('gpt2')
        question_embedder.config.pad_token_id = tokenizer.eos_token
        self.question_embedder = question_embedder.wte

        # decoder
        self.VCAdecoder = GPT2Model.from_pretrained('gpt2')
        self.VCAdecoder.config.pad_token_id = tokenizer.eos_token
 
        # intermediate layers
        self.intermediate_layer = nn.Linear(768, 512)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)

        # classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input, img):
        # image features
        img_feature = self.img_feature_extractor(img)
        img_feature = torch.unsqueeze(img_feature, dim=1)

        # visual embedding
        visual_embeds = self.visual_embedder(img_feature)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        visual_attention_mask = visual_attention_mask.to(device)

        if self.vis_pos_emb == 'zeroes':
            visual_id_type = torch.ones(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
            visual_position_id = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
        elif self.vis_pos_emb == 'pos':
            visual_id_type = torch.ones(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
            visual_position_id = torch.arange(0, visual_embeds.size()[1])
            visual_position_id = torch.unsqueeze(visual_position_id, 0)
            visual_position_id = visual_position_id.repeat(visual_embeds.size()[0], 1)
            visual_position_id = visual_position_id.to(device)

        # question embedding
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)
        question_embeds = self.question_embedder(input['input_ids'])
        question_attention_mask = input['attention_mask']
        
        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            question_id_type = torch.zeros(*question_embeds.size()[:-1], dtype=torch.long, device=device)
            question_position_id = torch.arange(0, question_embeds.size()[1])
            question_position_id = torch.unsqueeze(question_position_id, 0)
            question_position_id = question_position_id.repeat(question_embeds.size()[0], 1)
            question_position_id = question_position_id.to(device)

        # question first
        inputs_embeds = torch.cat((question_embeds, visual_embeds), dim=1)
        attention_mask = torch.cat((question_attention_mask, visual_attention_mask), dim=1)

        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            token_type_ids = torch.cat((question_id_type, visual_id_type), dim=1)
            position_ids = torch.cat((question_position_id, visual_position_id), dim=1)

        # decode
        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            decoder_output = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                             position_ids=position_ids, token_type_ids=token_type_ids)
        else:
            decoder_output = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        decoder_output = decoder_output.last_hidden_state.swapaxes(1, 2)
        decoder_output = F.adaptive_avg_pool1d(decoder_output, 1)
        decoder_output = decoder_output.swapaxes(1, 2).squeeze(1)

        # intermediate layers
        out = self.intermediate_layer(decoder_output)
        out = self.LayerNorm(out)
        out = self.dropout(out)

        # classifier
        out = self.classifier(out)
        return out
