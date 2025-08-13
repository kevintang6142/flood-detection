# cd_dependencies/networks.py (CPU-only version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision.transforms.functional import rgb_to_grayscale
from Unet.unetPlusPlus import unetPlusPlus

import models
from models.help_funcs import RCSA, Transformer, TransformerDecoder, TwoLayerConv2d

from .aspp import RCSA_ASPP
from .ELGCA import ELGCA
from .FUSE import DFIM
from .laplace import make_laplace_pyramid
from .MFDS import MDPM


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler."""
    if args.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + args.epoch_count - args.niter) / float(
                args.niter_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_iters, gamma=0.1
        )
    elif args.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif args.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.niter, eta_min=0
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented" % args.lr_policy
        )
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights."""

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            # ... (other init types)
            else:
                raise NotImplementedError
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f"Initializing network with {init_type} initialization")
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    """
    Initialize a network for CPU-only execution.
    The gpu_ids parameter is ignored but kept for compatibility.
    """
    # Force the network to the CPU
    net.to(torch.device("cpu"))
    print("Network initialized on CPU.")

    # Initialize weights
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type="normal", init_gain=0.02, gpu_ids=[]):
    """Define generator. gpu_ids is ignored."""
    if args.net_G == "base_resnet18":
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)
    elif args.net_G == "base_transformer_pos_s4":
        net = BASE_Transformer(
            input_nc=3,
            output_nc=2,
            token_len=4,
            resnet_stages_num=4,
            with_pos="learned",
        )
    elif args.net_G == "unet++":
        net = unetPlusPlus(6, 2)
    else:
        raise NotImplementedError(
            f"Generator model name [{args.net_G}] is not recognized"
        )

    # Initialize the network on the CPU
    return init_net(net, init_type, init_gain, gpu_ids=[])


def define_D(args, gpu_ids=[]):
    """Define discriminator (if needed)."""

    class NLayerDiscriminator(nn.Module):
        def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
            super(NLayerDiscriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(input_nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)
            )

        def forward(self, x):
            return self.model(x)

    norm = get_norm_layer(norm_type=args.norm)
    if args.net_D == "basic":
        net = NLayerDiscriminator(
            input_nc=args.output_nc, ndf=64, n_layers=3, norm_layer=norm
        )
    else:
        raise NotImplementedError(
            "Discriminator model name [%s] is not recognized" % args.net_D
        )
    return init_net(net, init_type="normal", init_gain=0.02, gpu_ids=gpu_ids)


class ResNet(torch.nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        resnet_stages_num=5,
        backbone="resnet18",
        output_sigmoid=False,
        if_upsample_2x=True,
    ):
        super(ResNet, self).__init__()
        expand = 1
        if backbone == "resnet18":
            self.resnet = models.resnet18(
                nc=input_nc,
                pretrained=False,
                replace_stride_with_dilation=[False, True, True],
            )
        elif backbone == "resnet34":
            self.resnet = models.resnet34(
                pretrained=True, replace_stride_with_dilation=[False, True, True]
            )
        elif backbone == "resnet50":
            self.resnet = models.resnet50(
                pretrained=True, replace_stride_with_dilation=[False, True, True]
            )
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.classifier2 = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.classifier3 = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)
        self.conv_pred2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_1 = self.resnet.layer1(x)
        x_2 = self.resnet.layer2(x_1)

        if self.resnet_stages_num > 3:
            x_4 = self.resnet.layer3(x_2)

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_4)

        if self.if_upsample_2x:
            x = self.upsamplex2(x_4)
            x_1 = self.upsamplex2(x_1)
        else:
            x = x_4
        x = self.conv_pred(x)
        x_1 = self.conv_pred2(x_1)
        return x_1, x


def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )


class BASE_Transformer(ResNet):
    def __init__(
        self,
        input_nc,
        output_nc,
        with_pos,
        resnet_stages_num=5,
        token_len=4,
        token_trans=True,
        enc_depth=1,
        dec_depth=1,
        dim_head=64,
        decoder_dim_head=64,
        tokenizer=True,
        if_upsample_2x=True,
        pool_mode="max",
        pool_size=2,
        backbone="resnet18",
        decoder_softmax=True,
        with_decoder_pos=None,
        with_decoder=True,
        with_fuion=True,
        use_drop_California=False,
        use_drop_Gloucester2=False,
    ):
        super(BASE_Transformer, self).__init__(
            input_nc,
            output_nc,
            backbone=backbone,
            resnet_stages_num=resnet_stages_num,
            if_upsample_2x=if_upsample_2x,
        )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(
            32, self.token_len, kernel_size=1, padding=0, bias=False
        )
        self.tokenizer = tokenizer
        if not self.tokenizer:
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        self.with_fuion = with_fuion
        dim = 32
        mlp_dim = 2 * dim
        num_patches = 4

        self.with_pos = with_pos
        if with_pos == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, 32))
        decoder_pos_size = 256 // 4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == "learned":
            self.pos_embedding_decoder = nn.Parameter(
                torch.randn(1, 32, decoder_pos_size, decoder_pos_size)
            )
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head

        self.transformer = Transformer(
            dim=dim,
            num_patches=num_patches,
            depth=self.enc_depth,
            heads=8,
            dim_head=self.dim_head,
            mlp_dim=mlp_dim,
            dropout=0.0,
        )
        self.transformer_decoder = TransformerDecoder(
            dim=dim,
            depth=self.dec_depth,
            heads=8,
            dim_head=self.decoder_dim_head,
            mlp_dim=mlp_dim,
            dropout=0.0,
            softmax=decoder_softmax,
        )
        self.resnet1 = ResNet(
            input_nc=input_nc,
            output_nc=output_nc,
            backbone=backbone,
            resnet_stages_num=resnet_stages_num,
            if_upsample_2x=if_upsample_2x,
        )
        self.resnet2 = ResNet(
            input_nc=input_nc,
            output_nc=output_nc,
            backbone=backbone,
            resnet_stages_num=resnet_stages_num,
            if_upsample_2x=if_upsample_2x,
        )
        self.ELGCA = ELGCA(dim=dim * 2, heads=4)
        self.ASPP = RCSA_ASPP(dim, dim, 1, 0.1)
        self.RCSA_rcsa = RCSA(nFeat=32)
        self.mdpm_1 = MDPM(32, 32)
        self.fusion_conv = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)
        self.fusex_x128 = DFIM(32, 32, 32)
        self.use_drop_California = use_drop_California
        self.use_drop_Gloucester2 = use_drop_Gloucester2
        if self.use_drop_California:
            self.drop_out = nn.Dropout(0.2)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum("bln,bcn->blc", spatial_attention, x)
        return tokens

    def _forward_reshape_tokens(self, x):
        if self.pool_mode == "max":
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == "ave":
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        tokens = rearrange(x, "b c h w -> b (h w) c")
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == "fix" or self.with_decoder_pos == "learned":
            x = x + self.pos_embedding_decoder
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer_decoder(x, m)
        x = rearrange(x, "b (h w) c -> b c h w", h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h, w, b, l, c])
        m = rearrange(m, "h w b l c -> l b c h w")
        m = m.sum(0)
        x = x + m
        return x

    def _forward_fusion(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.ELGCA(x)
        out = self.fusion_conv(x)
        return out

    def forward(self, x1, x2):
        grayscale_img_x1 = rgb_to_grayscale(x1)
        edge_feature_x1 = make_laplace_pyramid(grayscale_img_x1, 5, 1)[1]
        grayscale_img_x2 = rgb_to_grayscale(x2)
        edge_feature_x2 = make_laplace_pyramid(grayscale_img_x2, 5, 1)[1]

        x1_128, x1 = self.forward_single(x1)
        x2_128, x2 = self.forward_single(x2)

        x1 = self.ASPP(x1)
        x2 = self.ASPP(x2)
        x1_1 = x1
        x2_1 = x2

        x1_128 = self.ASPP(x1_128)
        x2_128 = self.ASPP(x2_128)

        x1_128, _ = self.mdpm_1(x1_128, edge_feature_x1)
        x2_128, _ = self.mdpm_1(x2_128, edge_feature_x2)

        x1_128 = self.RCSA_rcsa(x1_128)
        x2_128 = self.RCSA_rcsa(x2_128)

        x_128 = x1_128 - x2_128
        out1 = x_128

        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)

        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)

        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)

        if self.use_drop_California:
            x1 = self.drop_out(x1)
            x2 = self.drop_out(x2)

        if self.with_fuion:
            x1 = self._forward_fusion(x1_1, x1)
            x2 = self._forward_fusion(x2_1, x2)

        x1 = self.RCSA_rcsa(x1)
        x2 = self.RCSA_rcsa(x2)
        x = torch.abs(x1 - x2)
        out2 = x

        x = F.interpolate(
            x,
            size=[x1_128.shape[2], x1_128.shape[3]],
            mode="bilinear",
            align_corners=True,
        )
        x = self.fusex_x128(x, x_128)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex2(x)
        x = self.classifier(x)

        if self.use_drop_Gloucester2:
            x = self.drop_out(x)
        out3 = x
        self.CD = out3
        out1 = self.classifier2(out1)
        out2 = self.classifier3(out2)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return out2, out1, out3  # 64, 128, 256
