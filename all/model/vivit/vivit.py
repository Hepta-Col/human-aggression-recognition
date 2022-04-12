import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.vivit.module import Transformer, FSATransformerStack


def create_vivit(args):
    if args.model_name == 'vivit_2':
        return ViViT_2(img_h=args.input_size,
                       img_w=args.input_size,
                       patch_h=args.patch_size,
                       patch_w=args.patch_size,
                       num_classes=args.num_classes,
                       num_frames=args.num_frames,
                       depth=args.depth,
                       device=args.device)
    elif args.model_name == 'vivit_3':
        return ViViT_3(img_t=16,
                       img_h=args.input_size,
                       img_w=args.input_size,
                       patch_t=2,
                       patch_h=args.patch_size,
                       patch_w=args.patch_size,
                       num_classes=args.num_classes,
                       depth=args.depth,
                       device=args.device)
    else:
        raise NotImplementedError()


class ViViT_2(nn.Module):
    """ Model-2 backbone of ViViT """

    def __init__(self,
                 img_h, img_w, patch_h, patch_w,
                 num_classes, num_frames, device,
                 dim=192, depth=4, heads=3,
                 dim_head=64, channels=3, dropout=0., emb_dropout=0., scale_dim=4,
                 pool='cls', ):
        super().__init__()

        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert img_h == img_w and patch_h == patch_w
        image_size = img_h
        patch_size = patch_h
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),        # (p1 p2 c) is a patch
            nn.Linear(patch_dim, dim),
        )

        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, num_patches + 1, dim)).to(device)

        self.space_token = nn.Parameter(torch.randn(1, 1, dim)).to(device)
        self.space_transformer = Transformer(
            dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim)).to(device)
        self.temporal_transformer = Transformer(
            dim, depth, heads, dim_head, dim*scale_dim, dropout)

        # randomly zeroes some of the elements of the input tensor with prob p
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        """ x is a video: (b, T, C, H, W) """

        #! get token embeddings
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape    # n patches

        #! spacial transformer encoder forward
        cls_space_tokens = repeat(
            self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        #! temporal transformer encoder forward
        cls_temporal_tokens = repeat(
            self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


class ViViT_3(nn.Module):
    """ Model-3 backbone of ViViT """

    def __init__(self,
                 img_t, img_h, img_w, patch_t, patch_h, patch_w,
                 num_classes, depth, device, 
                 dim=192, heads=3,
                 dim_head=3, channels=3, dropout=0., emb_dropout=0., scale_dim=4,
                 pool='mean', ):
        super().__init__()

        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert img_t % patch_t == 0 and img_h % patch_h == 0 and img_w % patch_w == 0, "Video dimensions should be divisible by " \
            "tubelet size "

        tubelet_dim = channels * patch_t * patch_h * patch_w
        self.to_tubelet_embedding = nn.Sequential(
            Rearrange('b (t pt) c (h ph) (w pw) -> b t (h w) (pt ph pw c)',
                      pt=patch_t, ph=patch_h, pw=patch_w),
            nn.Linear(tubelet_dim, dim)
        )

        self.nt = img_t // patch_t
        self.nh = img_h // patch_h
        self.nw = img_w // patch_w

        self.pos_embedding = nn.Parameter(torch.randn(
            1, 1, self.nh * self.nw, dim)).repeat(1, self.nt, 1, 1).to(device)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.transformer_blocks = FSATransformerStack(dim, depth, heads, dim_head, dim*scale_dim,
                                                        self.nt, self.nh, self.nw, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        """
            x is a video: (b, C, T, H, W)
                - in default setting: [16, 16, 3, 256, 256]
        """
        
        #! get token embeddings
        tokens = self.to_tubelet_embedding(x)

        tokens += self.pos_embedding
        tokens = self.dropout(tokens)

        x = self.transformer_blocks(tokens)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
