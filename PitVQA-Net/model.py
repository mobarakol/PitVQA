import torch
from torch import nn
import torch.nn.functional as F
from BLIP.models.med import BertConfig, BertModel, BertLMHeadModel
from BLIP.models.blip import create_vit, init_tokenizer, load_checkpoint
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PitVQANet(nn.Module):
    def __init__(self,
                 med_config='/abs/path/to/BLIP/configs/med_config.json',  # change to your abs path
                 num_class=59,
                 image_size=224,
                 vit='base',
                 ):
        super().__init__()

        # visual encoder
        self.visual_encoder, vision_width = create_vit(vit, image_size, use_grad_checkpointing=False,
                                                       ckpt_layer=0, drop_path_rate=0.1)
        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # text encoder
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.vocab_size = self.tokenizer.vocab_size
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        # decoder
        self.gpt_decoder = GPT2Model.from_pretrained('gpt2')

        # intermediate_layers
        self.intermediate_layer = nn.Linear(768, 512)
        self.se_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)

        # classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, image, question):
        # visual encoder
        image_embeds = self.visual_encoder(image).to(device)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # text encoder
        encoder_question = self.tokenizer(question, return_tensors="pt", truncation=True,
                                          padding='max_length', max_length=25).to(image.device)

        text_output = self.text_encoder(input_ids=encoder_question.input_ids,
                                        attention_mask=encoder_question.attention_mask,
                                        encoder_hidden_states=image_embeds,
                                        encoder_attention_mask=image_atts,
                                        return_dict=True)
        text_embeds = text_output.last_hidden_state

        # decoder
        gpt_output = self.gpt_decoder(inputs_embeds=text_embeds,
                                      encoder_attention_mask=encoder_question.attention_mask)
        decoder_output = gpt_output.last_hidden_state

        # average pool
        decoder_output = decoder_output.swapaxes(1, 2)
        decoder_output = F.adaptive_avg_pool1d(decoder_output, 1)
        decoder_output = decoder_output.swapaxes(1, 2).squeeze(1)

        # intermediate layers
        out = self.intermediate_layer(decoder_output)
        out = torch.mul(out, self.se_layer(out))
        out = self.LayerNorm(out)
        out = self.dropout(out)

        # classifier
        out = self.classifier(out)
        return out
