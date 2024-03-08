
import torch
from torch import nn
import torch.nn.functional as F
from BLIP.models.med import BertConfig, BertModel, BertLMHeadModel
from BLIP.models.blip import create_vit, init_tokenizer, load_checkpoint

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BlipBert(nn.Module):
    def __init__(self,
                 med_config='/abs/path/to/BLIP/configs/med_config.json',  # change to your abs path
                 num_class=18,
                 image_size=224,
                 vit='base',
                 ):
        super().__init__()

        # visual encoder + tokenizer
        self.visual_encoder, vision_width = create_vit(vit, image_size, use_grad_checkpointing=False,
                                                       ckpt_layer=0, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()

        # text encoder
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

        # decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.output_hidden_states = True
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        # classifier
        hidden_size = decoder_config.hidden_size  # 768
        self.classifier = nn.Linear(hidden_size, num_class)

    def forward(self, image, question):
        # image embeddings and attention mask
        image_embeds = self.visual_encoder(image).to(device)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # process text data
        question_data = self.tokenizer(question, return_tensors="pt", truncation=True,
                                       padding='longest', max_length=25).to(image.device)
        question_data.input_ids[:, 0] = self.tokenizer.enc_token_id

        # text embeddings
        text_output = self.text_encoder(input_ids=question_data.input_ids,
                                        attention_mask=question_data.attention_mask,
                                        encoder_hidden_states=image_embeds,
                                        encoder_attention_mask=image_atts,
                                        return_dict=True)
        text_embeds = text_output.last_hidden_state

        # decoder
        decoder_output = self.text_decoder(input_ids=question_data.input_ids,
                                           attention_mask=question_data.attention_mask,
                                           encoder_hidden_states=text_embeds,
                                           encoder_attention_mask=question_data.attention_mask)

        decoder_last_hidden_states = decoder_output.hidden_states[-1]
        pooled_output = torch.mean(decoder_last_hidden_states, dim=1)
        logits = self.classifier(pooled_output)
        return logits
