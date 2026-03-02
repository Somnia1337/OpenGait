import torch
import torch.nn as nn
from einops import rearrange
from ..base_model import BaseModel
from ..modules import (
    SetBlockWrapper,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateFCs,
    SeparateBNNecks,
)


class BinGait(BaseModel):
    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg["backbone_cfg"])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])

        out_channels = model_cfg["SeparateFCs"]["out_channels"]
        parts = model_cfg["SeparateFCs"]["parts_num"]
        feat_dim = out_channels * parts
        hash_dim = model_cfg["hash_dim"]

        self.fc_refine = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.hash_fc = nn.Linear(feat_dim, hash_dim)

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

        self._accum_counter = 0
        self.Backbone.eval()
        self.Backbone.requires_grad_(False)

        self.FCs.eval()
        self.FCs.requires_grad_(False)

        self.BNNecks.eval()
        self.BNNecks.requires_grad_(False)

        self.TP.eval()
        self.TP.requires_grad_(False)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, "n s c h w -> n c s h w")

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        flat_feat = embed_1.view(embed_1.size(0), -1)
        identity = flat_feat
        z = self.hash_fc(self.fc_refine(flat_feat) + identity)

        hash_code = torch.sign(z)
        hash_code = (hash_code + 1) / 2

        retval = {
            "training_feat": {"hash": {"logits": z, "labels": labs}},
            "visual_summary": {
                "image/sils": rearrange(sils, "n c s h w -> (n s) c h w")
            },
            "inference_feat": {"embeddings": hash_code},
        }
        return retval
