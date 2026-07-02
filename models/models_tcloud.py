import torch
import os
import pandas as pd
import hashlib
import json
import torch.nn as nn
import segmentation_models_pytorch as smp
import loaders
from bamcd.model import BAM_CD
#from cloudseg.models.components.hrcloudnet import HRCloudNet
from cloudseg.models.components.cdnetv2 import CDnetV2
import segmentation_models_pytorch as smp
from swincloud.swincloud import SwinCloud
from libraries.utils import safe_get
from ssl4eo_l import SSL4EOLResNetUNet
from prithvi_eo2 import PrithviEO2Segmentation
from prithvi_lwir_structured_aux_fusion import PrithviLWIRStructuredAuxFusion
from prithvi_lwir_aux_cross_attention_fusion import PrithviLWIRAuxCrossAttentionFusion

def sort_lists(obj):
    """
    Recursively sort all lists in the object.
    """
    if isinstance(obj, dict):
        return {k: sort_lists(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return sorted((sort_lists(item) for item in obj), key=lambda x: json.dumps(x, sort_keys=True))
    else:
        return obj

def dict_to_hash_key(d, length=15):
    """
    Serialize a dictionary deterministically and hash it to a short key.
    
    Args:
        d (dict): Input dictionary.
        length (int): Desired length of output hash string.
    
    Returns:
        str: A short deterministic hash string.
    """ 
    sorted_obj = sort_lists(d)
    json_str = json.dumps(sorted_obj, sort_keys=True, separators=(',', ':'))
    #print(json_str)
    hash_digest = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    return hash_digest[:length]

def save_model_and_log_params(model, save_dir, model_file):
    os.makedirs(save_dir, exist_ok=True)

    #model_filename = f"model_{run_id}.pth"

    model_path = os.path.join(save_dir, model_file)
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

    # Prepare CSV log path
    '''
    csv_log_path = os.path.join(save_dir, "model_params_log.csv")
    #params_dict_fname = model_file.copy()
    #params_dict_fname["model_filename"] = model_file

    # Check for duplicates and write log
    if os.path.exists(csv_log_path):
        df = pd.read_csv(csv_log_path)
        if model_file in df["model_filename"].values:
            return model_path
        df = pd.concat([df, pd.DataFrame([params_dict_fname])], ignore_index=True)
    else:
        df = pd.DataFrame([params_dict_fname])

    df.to_csv(csv_log_path, index=False)
    '''
    return model_path

def tolist(item):
    if item is None:
        return None
    elif isinstance(item, list):
       return item
    elif isinstance(item, str):
       return [item]
    
def get_features_two_enc(featset):
    if isinstance(featset[0],str):
        return [featset[0]], [featset[1]]
    elif isinstance(featset[0],list):
        return featset[0], featset[1]

def ssl4eo_init_info(params_dict, initmodel):
    print("freeze_encoder param:", params_dict.get("freeze_encoder"), flush=True)
    print("conv1 shape:", tuple(initmodel.encoder.conv1.weight.shape), flush=True)
    print(
        "encoder trainable params:",
        sum(p.numel() for p in initmodel.encoder.parameters() if p.requires_grad),
        flush=True,
    )
    print(
    "decoder trainable params:",
    sum(p.numel() for n, p in initmodel.named_parameters()
    if p.requires_grad and not n.startswith("encoder.")),
    flush=True)

def init_model_and_loaders(params_dict, onlyloaders=False):
    featset=params_dict["features"]
    featset=tolist(featset)
    num_classes=params_dict["num_classes"]
    model_type=params_dict.get("model_type",None)
    thincloudcl=params_dict.get("thin_cloud_class",1)
    transformkey=params_dict.get("transform",None)
    batch_size=params_dict["batch_size"]
    input_files = os.path.join(params_dict["dataset_folder"],params_dict["dataset"])
    device=params_dict["device"]
    test=(params_dict["traintest"]=="test")
    target_band=params_dict["target_band"]
    workers = safe_get(params_dict,"cpuworkers",4)
    yshift= safe_get(params_dict,"yshift",1)
    dataset_dir=params_dict.get("dataset_dir",None)
    data_source = params_dict.get("data_source", "tiff")
    
    initmodel=None

    one_enc_models=["Unet","SegFormer","DeepLabV3","Swin-Unet","HRCloudNet","CDnetV2","SwinCloud",
                    "SSL4EO", "Prithvi"] 
    one_enc_models=["Fine Tune "+bm for bm in one_enc_models]+one_enc_models
    two_enc_models=["Siamese", "bam-cd"]
    two_enc_models=["Fine Tune "+bm for bm in two_enc_models]+two_enc_models
    
    
    if model_type in one_enc_models or model_type.startswith(tuple(one_enc_models)) or model_type is None:
        clear_bands=None
    elif model_type in two_enc_models:
        featset, clear_bands = get_features_two_enc(featset)

    train_loader, val_loader = loaders.get_loaders(input_files, None, featset, target_band, yshift=yshift, 
                                                   clear_bands=clear_bands, batch_size=batch_size, thincloudcl=thincloudcl, 
                                                   transformkey=transformkey, model_type=model_type, testrun=test,
                                                   dataset_dir=dataset_dir, workers=workers, data_source=data_source)

    
    if onlyloaders:
        return None, train_loader, val_loader
    
    if model_type=="Unet" or model_type=="Fine tuned Unet":
        initmodel = smp.Unet(encoder_name='resnet34', 
                            #encoder_weights=None, 
                            in_channels=len(featset), 
                            classes=num_classes).to(device)    
    elif model_type=="SegFormer":
        initmodel = smp.create_model(
        arch="segformer",
        encoder_name="mit_b2",    
        encoder_weights="imagenet",  # pretrained on ImageNet
        #encoder_weights=None,
        in_channels=len(featset),
        classes=num_classes
        ).to(device)
    elif model_type=="DeepLabV3":
        initmodel = smp.create_model(
        arch="deeplabv3plus",
        encoder_name='resnet34', 
        encoder_weights="imagenet",
        in_channels=len(featset),
        classes=num_classes
        ).to(device)
    elif model_type=="Swin-Unet":
        # initmodel = smp.Unet(encoder_name='swin_t', 
        #                     encoder_weights="imagenet", 
        #                     in_channels=len(featset), classes=num_classes).to(device)
        initmodel = smp.create_model(
                arch="upernet",
                encoder_name="tu-swinv2_cr_tiny_224",
                encoder_weights=None,
                in_channels=len(featset),
                classes=num_classes
                ).to(device)
    #elif model_type=="HRCloudNet":
    #    initmodel=HRCloudNet(in_channels=len(featset),num_classes=num_classes).to(device)
    elif model_type=="CDnetV2":
        initmodel=CDnetV2(in_channels=len(featset),num_classes=num_classes).to(device)
        #    elif model_type=="Siamese" or model_type=="bam-cd":
        #if not onlyloaders:
    elif model_type=="Siamese":
        initmodel = SiameseUNet(encoder_name="resnet34",
                            in_channels=1,  # because each input (cloudy, clear) is single-channel mask/band
                            num_classes=num_classes,  # clear, thin cloud, cloud
                            encoder_weights=None,
                            activation=None  # logits output (no activation here, explained below)
                            ).to(device)
    elif model_type=="bam-cd":
        fusion_mode=params_dict["fusion_mode"] if "fusion_mode" in params_dict else 'conc'
        if isinstance(featset[0], list):
            inchannum=len(featset[0])
        else:
            inchannum=1
        initmodel = BAM_CD(encoder_weights=None,
                decoder_attention_type="scse", # For adding Attention squeeze and exitation "scse"
                in_channels=inchannum,
                classes=num_classes,
                fusion_mode=fusion_mode, #'conc' or 'diff' concatenation or difference
                activation=None,
                siamese= False, #False
                return_features= False).to(device)
    elif model_type=="SwinCloud":
        initmodel = SwinCloud(img_size=224, num_classes=num_classes, in_chans=len(featset)).to(device)
    elif model_type.startswith("SSL4EO-L") :
        #if len(featset) != 1:
        #    raise ValueError("SSL4EO-L has been integrated here for LWIR-only training; configure exactly one feature/band.")
        backbone="resnet18"
        if model_type[-2:]=="50": backbone="resnet50"
        if model_type in ["SSL4EO-L" , "SSL4EO-L-50"]:
            initmodel = SSL4EOLResNetUNet(
                num_classes=num_classes,
                backbone=backbone,
                weights_name=params_dict.get("ssl4eo_weights", "LANDSAT_OLI_TIRS_TOA_MOCO"),
                in_channels=1,
                freeze_encoder=params_dict.get("freeze_encoder", False),
            ).to(device)
            ssl4eo_init_info(params_dict, initmodel)
        else:
            initmodel = SSL4EOLResNetUNet(
                num_classes=num_classes,
                backbone=backbone,
                weights_name=params_dict.get("ssl4eo_weights","LANDSAT_OLI_TIRS_TOA_MOCO"),
                in_channels=1,
                freeze_encoder=params_dict.get("freeze_encoder", False),
                input_mode="zero_pad_b10_b11_mean",
            ).to(device)
            ssl4eo_init_info(params_dict, initmodel)
    elif model_type == "Prithvi-LWIR":
            initmodel = PrithviEO2Segmentation(
            num_classes=num_classes,
            backbone=params_dict.get("prithvi_backbone", "prithvi_eo_v2_300_tl"),
            input_mode="adapt_patch_embed_avg",
            in_channels=len(featset),
            freeze_encoder=params_dict.get("freeze_encoder", False),
            decoder=params_dict.get("prithvi_decoder", "UperNetDecoder"),
            decoder_channels=params_dict.get("prithvi_decoder_channels", 256),
            img_size=params_dict.get("prithvi_img_size", 224),
        ).to(device)
    elif model_type == "Prithvi-LWIR-Z":
            initmodel = PrithviEO2Segmentation(
            num_classes=num_classes,
            backbone=params_dict.get("prithvi_backbone", "prithvi_eo_v2_300_tl"),
            input_mode="swir_lwir_zeros",
            in_channels=1,
            freeze_encoder=params_dict.get("freeze_encoder", False),
            decoder=params_dict.get("prithvi_decoder", "UperNetDecoder"),
            decoder_channels=params_dict.get("prithvi_decoder_channels", 256),
            img_size=params_dict.get("prithvi_img_size", 224),
        ).to(device)
    elif model_type == "Prithvi":
        if len(featset) != 6:
            raise ValueError(
                "Prithvi-HLS6 expects exactly 6 features in this order: "
                "Landsat B2, B3, B4, B5, B6, B7."
            )
        initmodel = PrithviEO2Segmentation(
            num_classes=num_classes,
            backbone=params_dict.get("prithvi_backbone", "prithvi_eo_v2_300_tl"),
            input_mode="native_hls6",
            in_channels=6,
            freeze_encoder=params_dict.get("freeze_encoder", False),
            decoder=params_dict.get("prithvi_decoder", "UperNetDecoder"),
            decoder_channels=params_dict.get("prithvi_decoder_channels", 256),
            img_size=params_dict.get("prithvi_img_size", 224),
        ).to(device)
    elif model_type == "Prithvi-Fusion":

        if len(featset) < 2:
            raise ValueError(
                    "Prithvi-LWIR-StructuredAuxFusion expects at least cloudy LWIR "
                    "and one auxiliary feature."
                )

        initmodel = PrithviLWIRStructuredAuxFusion(
            num_classes=num_classes,
            total_in_channels=len(featset),

            thermal_idx=params_dict.get("thermal_idx", 0),
            clear_idx=params_dict.get("clear_idx", 1),
            dem_idx=params_dict.get("dem_idx", 2),
            weather_indices=params_dict.get(
                "weather_indices",
                list(range(3, len(featset)))
            ),

            prithvi_backbone=params_dict.get("prithvi_backbone", "prithvi_eo_v2_300_tl"),
            prithvi_feature_channels=params_dict.get("prithvi_feature_channels", 64),
            clear_feature_channels=params_dict.get("clear_feature_channels", 32),
            dem_feature_channels=params_dict.get("dem_feature_channels", 16),
            weather_feature_channels=params_dict.get("weather_feature_channels", 16),
            fusion_hidden_channels=params_dict.get("fusion_hidden_channels", None),

            freeze_encoder=params_dict.get("freeze_encoder", False),
            prithvi_decoder=params_dict.get("prithvi_decoder", "UperNetDecoder"),
            prithvi_decoder_channels=params_dict.get("prithvi_decoder_channels", 256),
            prithvi_img_size=params_dict.get("prithvi_img_size", 256),

            branch_dropout=params_dict.get("branch_dropout", 0.0),
            weather_hidden_channels=params_dict.get("weather_hidden_channels", 64),
        ).to(device)
    elif model_type == "Prithvi-Cross":

        initmodel = PrithviLWIRAuxCrossAttentionFusion(
            num_classes=num_classes,
            total_in_channels=len(featset),

            thermal_idx=params_dict.get("thermal_idx", 0),
            clear_idx=params_dict.get("clear_idx", 1),
            dem_idx=params_dict.get("dem_idx", 2),
            weather_indices=params_dict.get(
                "weather_indices",
                list(range(3, len(featset)))
            ),

            prithvi_backbone=params_dict.get("prithvi_backbone", "prithvi_eo_v2_300_tl"),
            prithvi_feature_channels=params_dict.get("prithvi_feature_channels", 64),

            aux_embed_dim=params_dict.get("aux_embed_dim", 64),
            aux_token_grid=params_dict.get("aux_token_grid", 4),
            num_heads=params_dict.get("num_heads", 4),

            freeze_encoder=params_dict.get("freeze_encoder", False),
            prithvi_decoder=params_dict.get("prithvi_decoder", "UperNetDecoder"),
            prithvi_decoder_channels=params_dict.get("prithvi_decoder_channels", 256),
            prithvi_img_size=params_dict.get("prithvi_img_size", 256),
        ).to(device)
    else:
            print(f"Unrecognized Model Type: {model_type}")
            return None, None, None

    return initmodel, train_loader, val_loader


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        # Note: in_channels here is the TOTAL after concatenation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x is already upsampled and concatenated before being passed here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, use_batchnorm=True):
        super().__init__()
        
        encoder_channels = encoder_channels[::-1]  # [1024, 512, 256, 128, 128, 4]
        
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # First block: bottleneck (no skip)
        self.blocks.append(DecoderBlock(encoder_channels[0], decoder_channels[0], use_batchnorm))
        
        # Remaining blocks with skip connections
        for i in range(1, n_blocks):
            # Upsampling layer
            self.ups.append(nn.ConvTranspose2d(decoder_channels[i-1], decoder_channels[i-1], kernel_size=2, stride=2))
            
            # Calculate total input channels after upsampling + skip concatenation
            upsampled_channels = decoder_channels[i-1]
            skip_channels = encoder_channels[i] if i < len(encoder_channels) else 0
            total_channels = upsampled_channels + skip_channels
            
            self.blocks.append(DecoderBlock(total_channels, decoder_channels[i], use_batchnorm))
    
    def forward(self, *features):
        features = features[::-1]  # Reverse: deepest first
        
        # First block (bottleneck)
        x = self.blocks[0](features[0])
        
        # Remaining blocks with skip connections
        for i in range(1, len(self.blocks)):
            # Upsample
            x = self.ups[i-1](x)
            
            # Add skip connection if available
            if i < len(features):
                x = torch.cat([x, features[i]], dim=1)
            
            # Process through decoder block
            x = self.blocks[i](x)
        
        return x

class SiameseUNet(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        in_channels=1,
        num_classes=3,
        #encoder_weights="imagenet",
        encoder_weights=None,
        decoder_channels=[256, 128, 64, 32, 16, 8],
        activation=None,
    ):
        super().__init__()

        # Create two separate encoders instead of one shared encoder
        self.encoder_cloudy = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=len(decoder_channels)-1,
            weights=encoder_weights,
        )
        
        self.encoder_clear = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=len(decoder_channels)-1,
            weights=encoder_weights,
        )

        # Get encoder channels from one encoder (both should have same architecture)
        encoder_channels = self.encoder_cloudy.out_channels
        merged_channels = [c * 2 for c in encoder_channels]

        self.decoder = UnetDecoder(
            encoder_channels=merged_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            use_batchnorm=True,
        )

        self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        self.activation = smp.base.SegmentationHead(activation=activation) if activation else None

    def forward(self, cloudy, clear):
        # Use separate encoders for each input
        feats_cloudy = self.encoder_cloudy(cloudy)
        feats_clear = self.encoder_clear(clear)

        #print(f"Encoder out_channels: {self.encoder_cloudy.out_channels}")
        #print(f"Actual encoder outputs:")
        #for i, (f1, f2) in enumerate(zip(feats_cloudy, feats_clear)):
        #    print(f"  Level {i}: {f1.shape}")

        merged_feats = [torch.cat([f1, f2], dim=1) for f1, f2 in zip(feats_cloudy, feats_clear)]

        #print(f"Merged features shapes: {[f.shape for f in merged_feats]}")

        x = self.decoder(*merged_feats)
        logits = self.segmentation_head(x)

        if self.activation is not None:
            return self.activation(logits)
        return logits