from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from bevllm.model.bevllm_arch import BevLLMMetaModel, BevLLMMetaForCausalLM
from transformers import BitsAndBytesConfig
from bevllm.model.Qformer import BertConfig, BertLMHeadModel
from bevllm.model.positional_encoding import PositionalEncoding

class BevLLMLlamaConfig(LlamaConfig):
    model_type = "bev_llama"


class BevLLMLlamaModel(BevLLMMetaModel, LlamaModel):
    config_class = BevLLMLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(BevLLMLlamaModel, self).__init__(config)

class BevLLMLlamaForCausalLM(LlamaForCausalLM, BevLLMMetaForCausalLM):
    config_class = BevLLMLlamaConfig

    def __init__(self, config, freeze_qformer = False):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = BevLLMLlamaModel(config)

        self.model.embed_tokens.requires_grad = True

        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        #self.qformer = BEVLLMQformer(num_query_token = 512, max_txt_len=100, cache_dir=config.cache_dir).to(self.devices[0])
        
        self.bev_conv1= nn.Conv2d(in_channels=config.num_query_token, out_channels=768, kernel_size=(1,1), stride = (1, 1) )
        self.bev_proj = nn.Linear(768, 768)
        self.bev_proj_norm = nn.LayerNorm(768)
        self.vision_proj = nn.Linear(768, 768)
        self.vision_proj_norm = nn.LayerNorm(768)

        self.view_position_embeddings = PositionalEncoding(config.pos_encoding_scale, 768, 6, [768,32400])

        self.qformer, self.query_tokens = self.init_Qformer(
            config.num_query_token, 768, config.cross_attention_freq
        )


        if freeze_qformer:
            for param in self.qformer.parameters():
                param.requires_grad = False

        self.qformer.cls = None
        self.qformer.bert.embeddings.word_embeddings = None
        self.qformer.bert.embeddings.position_embeddings = None
        for layer in self.qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
            

        self.qformer_proj = nn.Linear(256,config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        #for name, param in self.lm_head.named_parameters():
        #    param.requires_grad = False

        # Initialize weights and apply final processing
        self.post_init()


    def get_model(self):
        return self.model
    
    def get_view_embeddings(self):
        return self.view_position_embeddings

    def get_qformer(self):
        return self.qformer

    def get_projection(self):
        return self.qformer_proj

    def get_devices(self):
        return self.devices

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        bevs: Optional[torch.FloatTensor] = None,
        view: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:





        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                bevs,
                view,
            )
            
            #if input_ids is not None:
            #    input_ids.to(self.device+2)
            #if position_ids is not None:
            #    position_ids.to(self.device+2)
            #if attention_mask is not None:
            #    attention_mask.to(self.device+2)
            #if past_key_values is not None:
            #    past_key_values.to(self.device+2)
            #if inputs_embeds is not None:
            #    inputs_embeds.to(self.device+2)
            #if labels is not None:
            #    labels.to(self.device+2)
       
#            print(inputs_embeds)
#            print(labels)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        bevs: Optional[torch.Tensor] = None,
        view: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if bevs is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                bevs,
                view
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

     

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("felix_brandstaetter_thesis/main/model/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
        #encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        #Qformer = BertLMHeadModel.from_pretrained(
        #    "bert-base-uncased", config=encoder_config
        #)
        Qformer = BertLMHeadModel.from_pretrained(
            "felix_brandstaetter_thesis/main/model/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594", config=encoder_config
        )

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, 768)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

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

AutoConfig.register("bev_llama", BevLLMLlamaConfig)
AutoModelForCausalLM.register(BevLLMLlamaConfig, BevLLMLlamaForCausalLM)