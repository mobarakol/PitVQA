
import torch
import torch.nn as nn

from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from transformers import GPT2Model
import torch.nn.functional as F
from transformers import GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClipGPT(nn.Module):
    def __init__(self, num_class=59):
        super(ClipGPT, self).__init__()
        # prepare CLIP encoders (visual and text)
        config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel(config).to(device)

        self.config = model.config
        self.text_model = model.text_model
        self.vision_model = model.vision_model
        self.visual_projection = nn.Linear(model.visual_projection.in_features, 768)
        self.text_projection = nn.Linear(model.text_projection.in_features, 768)

        # decoder
        self.VCA_decoder = GPT2Model.from_pretrained('gpt2')

        # intermediate_layers
        self.intermediate_layer = nn.Linear(768, 512)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.2)

        # classifier
        self.classifier = nn.Linear(in_features=512, out_features=num_class)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        return_loss=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None
                                else self.config.output_hidden_states)

        # get vision and text features using CLIP encoders
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # get visual and text embeddings
        image_embeds = vision_outputs[0].to(device)
        image_embeds = self.visual_projection(image_embeds)
        image_embeds = self.dropout(image_embeds)

        text_embeds = text_outputs[0].to(device)
        text_embeds = self.text_projection(text_embeds)
        text_embeds = self.dropout(text_embeds)

        batch_size = image_embeds.shape[0]
        visual_seq_len = image_embeds.shape[1]

        # get text and visual attention mask
        text_attention_mask = attention_mask.to(device)
        visual_attention_mask = torch.ones((batch_size, visual_seq_len), dtype=torch.float).to(device)

        # concatenate text and visual embeddings (text first)
        inputs_embeds = torch.cat((text_embeds, image_embeds), dim=1).to(device)
        # concatenate text and visual attention mask (text first)
        inputs_attention_mask = torch.cat((text_attention_mask, visual_attention_mask), dim=1).to(device)

        # decode
        decoder_output = self.VCA_decoder(inputs_embeds=inputs_embeds, attention_mask=inputs_attention_mask)

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
