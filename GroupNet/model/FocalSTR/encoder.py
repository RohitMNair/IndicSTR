from transformers import FocalNetConfig, FocalNetModel
import lightning.pytorch as pl

class FocalNetEncoder(pl.LightningModule):
    def __init__(self, hidden_dropout_prob: float = 0.0, initializer_range: float = 0.02,
                 image_size: int = 224, patch_size: int = 16, num_channels: int = 3,
                 embed_dim:int = 96, hidden_sizes:list = [192, 384, 768, 768], 
                 depths:list = [2, 2, 6, 2], focal_levels:list = [2, 2, 2, 2],
                 focal_windows:list = [3, 3, 3, 3], mlp_ratio:float = 4.0,
                 drop_path_rate:float = 0.1, layer_norm_eps:float = 1e-5):
        super().__init__()
        self.config = FocalNetConfig(
                            image_size = image_size,
                            patch_size = patch_size,
                            num_channels = num_channels,
                            embed_dim = embed_dim,
                            use_conv_embed = True, # better learning
                            hidden_sizes = hidden_sizes,
                            depths = depths,
                            focal_levels = focal_levels,
                            focal_windows = focal_windows,
                            hidden_act = "gelu",
                            mlp_ratio = mlp_ratio,
                            hidden_dropout_prob = hidden_dropout_prob,
                            drop_path_rate = drop_path_rate,
                            use_layerscale = False,
                            use_post_layernorm = False,
                            use_post_layernorm_in_modulation = False,
                            normalize_modulator = False,
                            initializer_range =  initializer_range,
                            layer_norm_eps = layer_norm_eps,
                            encoder_stride = 32,
                        )
        self.model = FocalNetModel(self.config, add_pooling_layer= False)

    def forward(self, x):
        return self.model(x, return_dict = True).last_hidden_state


if __name__ == '__main__':
    model = FocalNetEncoder(
        hidden_dropout_prob = 0.1, 
        initializer_range = 0.02,
        image_size= 128,
        patch_size= 8,
        num_channels = 3,
        embed_dim= 128,
        depths= [1, 1, 1, 1],
        focal_levels = [3, 3, 3, 3],
        focal_windows = [3, 3, 3, 3],
        mlp_ratio = 4.0,
        drop_path_rate = 0.1,
        layer_norm_eps = 1e-5
    )
    print(model.config.hidden_sizes)