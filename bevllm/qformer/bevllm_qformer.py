"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
    X-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from bevllm import (
    BEVLLMBase,
    compute_sim_matrix,
    disabled_train,
)
from bevllm_outputs import BEVLLMOutput, BEVLLMOutputFeatures
import math

class PositionalEncoding(nn.Module):
    def __init__(self,scale_factor, d_model, max_seq_length, feature_space_shape):
        super(PositionalEncoding, self).__init__()

        

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe * scale_factor

        self.register_buffer('pe', pe.unsqueeze(0))
        
        self.view_map = self.calculate_view_map(feature_space_shape)
        #torch.save(self.view_map, "view_map.pt")


    def calculate_view_map(self, shape):
        channels = shape[0]
        width = int(math.sqrt(shape[1]))
        height = int(math.sqrt(shape[1]))



        # Calculate the center of the feature map
        center_x = (height - 1) / 2  # center x-coordinate
        center_y = (width - 1) / 2  # center y-coordinate

        view_map = torch.zeros(width, height).long()

        # Loop through each element in the feature map
        for i in range(width):
            for j in range(height):
                # Calculate the vector from the current element to the center
                x = torch.tensor(j - center_x)
                y = torch.tensor(center_y - i)
                
                # Calculate the angle using atan2
                angle = torch.atan2(y, x)
                
                # Convert angle from radians to degrees if needed
                angle_degrees = torch.rad2deg(angle)
                


                if angle_degrees < 30 and angle_degrees > -30:
                    view_map[i,j] = 0

                elif angle_degrees < 90 and angle_degrees > 30:
                    view_map[i,j] = 1

                elif angle_degrees < 150 and angle_degrees > 90:
                    view_map[i,j] = 2
                
                elif angle_degrees < -150 and angle_degrees > -180 or angle_degrees < 180 and angle_degrees > 150:
                    view_map[i,j] = 3

                elif angle_degrees > -150 and angle_degrees < -90:
                    view_map[i,j] = 4

                elif angle_degrees > -90 and angle_degrees < -30:
                    view_map[i,j] = 5
        
        return view_map


    def classify_views(self, bev_map, view):

        bs = bev_map.shape[0]
        channels = bev_map.shape[1]
        width = int(math.sqrt(bev_map.shape[2]))
        height = int(math.sqrt(bev_map.shape[2]))

        pos_vectors = []

        for v in view:      
            indicies = self.view_map.view(-1)
            pos_vector = self.pe[0,indicies].to(bev_map.device)
            if v == 6:
                pos_vector = pos_vector.view(width,height,channels).permute(2,0,1).unsqueeze(0)
            else:
                pos_vector_view = self.pe[0,v].unsqueeze(0).to(bev_map.device)
                torch.save(pos_vector_view, "pos_enc_vec.pt")
                placeholder = torch.zeros(1,channels).to(bev_map.device)
                pos_vector = torch.where(pos_vector == pos_vector_view, pos_vector_view, placeholder)
                pos_vector = pos_vector.view(width,height,channels).permute(2,0,1).unsqueeze(0)
            
            pos_vectors.append(pos_vector)

    
        return torch.flatten(torch.cat(pos_vectors), start_dim=2).to(bev_map.device)

    def forward(self, x, view):
        feature_space_shape = x.shape
        x = torch.flatten(x, start_dim=2).permute(0,2,1)
        x = x + self.classify_views(x,view)
        x = x.reshape(feature_space_shape)
        return x



@registry.register_model("bevllm")
@registry.register_model("bevllm_feature_extractor")
class BEVLLMQformer(BEVLLMBase):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        cache_dir,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        feature_sapce_shape = [768,32400]
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer("right", cache_dir)

        #self.bev_proj = nn.Linear(32400, 1408)

        #bev_config ="felix_brandstaetter_thesis/main/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
        #bev_checkpoint = "felix_brandstaetter_thesis/main/mmdetection3d/projects/BEVFusion/configs/bevfusion_converted.pth"
        #self.bev_fusion = self.init_bev(bev_config, bev_checkpoint)
        #for param in self.bev_fusion.parameters():
        #    param.requires_grad = False
        #self.bev_fusion.bbox_head = None
        #       

        self.bev_conv1= nn.Conv2d(in_channels=512, out_channels=768, kernel_size=(1,1), stride = (1, 1) )
        self.bev_proj = nn.Linear(768, 768)
        self.bev_proj_norm = nn.LayerNorm(768)
        self.vision_proj = nn.Linear(768, 768)
        self.vision_proj_norm = nn.LayerNorm(768)

        print(f'query token: {num_query_token}')

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 768, cross_attention_freq, cache_dir
        )

        #self.query_proj = nn.Linear(32400,768)
        #self.bev_proj = nn.Linear(32400, 768)

        scale = 0.06
        print(scale)
        self.view_position_embeddings = PositionalEncoding(scale, 768,6, feature_sapce_shape)
        self.view_position_embeddings = self.view_position_embeddings

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        #self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        #self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len 
 

    #def get_bev_config(self):
    #    return self.bev_fusion.cfg

    def forward(self, samples):
        bev = samples["bev"]
        text = samples["text"]
        view = samples["view"] #6 für alle encodings


        #bev = bev.to(self.device)
        bev_embeds = torch.flatten(bev, start_dim=2).to(bev.device) #512,32400
        
        bev_embeds = self.view_position_embeddings(bev_embeds, view) #512,32400 mit pos encoding
        #bev_embeds = self.bev_proj(bev_embeds)

        bev_embeds = bev_embeds.transpose(1,2).contiguous()

        bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).to(
            bev_embeds.device
        )

        query_tokens = self.query_tokens.expand(bev_embeds.shape[0], -1, -1)
        #query_tokens = self.view_position_embeddings(query_tokens, view)
        #query_tokens = self.query_proj(query_tokens)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=bev_embeds,
            encoder_attention_mask=bev_atts,
            use_cache=True,
            return_dict=True,
        )

        
        bev_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(bev_embeds.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        bev_feats_all = concat_all_gather(
            bev_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            bev_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), bev_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = bev_embeds.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            bev_embeds.device
        )                   
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        bev_embeds_world = all_gather_with_grad(bev_embeds)


        with torch.no_grad():
   
            sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)            
                
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(bev_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            bev_embeds.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [bev_embeds, image_embeds_neg, bev_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            bev_embeds.device
        )


        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(bev_embeds.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            bev_embeds.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return BEVLLMOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        bev = samples["bev"]
        view = samples["view"]

        print(bev)

        bev_embeds = torch.flatten(bev, start_dim=2).to("cuda")    


        bev_embeds = self.view_position_embeddings(bev_embeds, view)
        #bev_embeds = self.bev_proj(bev_embeds)
        bev_embeds = bev_embeds.transpose(1,2).contiguous()
        
        if not use_nucleus_sampling:
            bev_embeds = bev_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1


        bev_atts = torch.ones(bev_embeds.size()[:-1], dtype=torch.long).to(
            bev_embeds.device
        )

        model_kwargs = {
            "encoder_hidden_states": bev_embeds,
            "encoder_attention_mask": bev_atts,
        }

        input_ids = (
            torch.LongTensor(bev_embeds.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(bev_embeds.device)
        )

        query_tokens = self.query_tokens.expand(bev_embeds.shape[0], -1, -1)
        #query_tokens = self.view_position_embeddings(query_tokens, view)
        #query_tokens = self.query_proj(query_tokens)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_bev(self, bev_feats, view):

       #self.bev_fusion.to(self.device)
        #bev_embeds = self.bev_fusion.test_step(bev)
        bev_feats = self.bev_conv1(bev_feats) 
        bev_feats = bev_feats.permute(0,2,3,1)
        bev_feats = self.bev_proj(bev_feats)
        bev_feats = self.bev_proj_norm(bev_feats)
        bev_feats = bev_feats.float().reshape(bev_feats.shape[0], -1, bev_feats.shape[-1])
        bev_feats = self.view_position_embeddings(bev_feats, view)

        bev_atts = torch.ones(bev_feats.size()[:-1], dtype=torch.long).to(
            bev_feats.device
        )

        query_tokens = self.query_tokens.expand(bev_feats.shape[0], -1, -1)
        #query_tokens = self.view_position_embeddings(query_tokens, view)
        #query_tokens = self.query_proj(query_tokens)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=bev_feats,
            encoder_attention_mask=bev_atts,
            return_dict=True,
        )

        bev_feat_former = torch.cat([torch.mean(query_output.last_hidden_state, dim=1, keepdim = True) , query_output.last_hidden_state], dim=1)
        bev_feat_former = F.normalize(
            self.vision_proj(bev_feat_former), dim=-1
        )


        return bev_feat_former

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BEVLLMOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
