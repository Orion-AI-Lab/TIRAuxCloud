import torch
import os
import pandas as pd
import hashlib
import json
import torch.nn as nn
from data.loaders import get_loaders
from model_builder.base_model import BaseModel
from models.bamcd.model import BAM_CD
from models.cloudseg.models.components.hrcloudnet import HRCloudNet
from models.cloudseg.models.components.cdnetv2 import CDnetV2
import segmentation_models_pytorch as smp
from models.swincloud.swincloud import SwinCloud

def find_file_recursive(filename, search_dir):
    """
    Recursively search for a file by name under a directory.

    Args:
        filename (str): Name of the file to search for.
        search_dir (str): Directory path to start the search from.

    Returns:
        list of str: Full paths to files that match the filename.
    """
 
    for root, dirs, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)

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
    

def init_model_and_loaders(params_dict, onlyloaders=False):
    featset=params_dict["features"]
    featset=tolist(featset)
    num_classes=params_dict["num_classes"]
    model_type=params_dict["model_type"]
    thincloudcl=params_dict.get("thin_cloud_class",1)
    transformkey=params_dict.get("transform",None)
    batch_size=params_dict["batch_size"]
    input_files = os.path.join(params_dict["dataset_folder"],params_dict["dataset"])
    device=params_dict["device"]
    test=(params_dict["traintest"]=="test")
    target_band=params_dict["target_band"]
    workers=params_dict.get("cpuworkers",4)
    yshift=params_dict.get("yshift",1)
    dataset_dir=params_dict.get("dataset_dir",None)
    
    initmodel=None

    one_enc_models=["Unet","SegFormer","DeepLabV3","Swin-Unet","HRCloudNet","CDnetV2","SwinCloud"]
    one_enc_models=["Fine Tune "+bm for bm in one_enc_models]+one_enc_models
    two_enc_models=["Siamese", "bam-cd"]
    two_enc_models=["Fine Tune "+bm for bm in two_enc_models]+two_enc_models
    
    
    if model_type in one_enc_models:
        clear_bands=None
    elif model_type in two_enc_models:
        featset, clear_bands = get_features_two_enc(featset)

    train_loader, val_loader = get_loaders(input_files, None, featset, target_band, yshift=yshift, 
                                                   clear_bands=clear_bands, batch_size=batch_size, thincloudcl=thincloudcl, 
                                                   transformkey=transformkey, model_type=model_type, testrun=test,
                                                   dataset_dir=dataset_dir, workers=workers)

    
    if onlyloaders:
        return None, train_loader, val_loader
    if model_type in MODEL_REGISTRY : 
            initmodel = MODEL_REGISTRY[model_type].from_config(params_dict).to(device) # TODO 
    elif model_type=="Unet" or model_type=="Fine tuned Unet":
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
        #    elif model_type=="Siamese" or model_type=="bam-cd":
        #if not onlyloaders:
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

class SiameseUNet(BaseModel, nn.Module):
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
    
    @property
    def name(self) -> str:
        return "Siamese"
    
    @classmethod
    def from_config(cls, config):    
        return cls(
            encoder_name="resnet34",
            in_channels=1, 
            num_classes=config["num_classes"],
            encoder_weights=None, 
            activation=None
        )
    
MODEL_REGISTRY = {
    "Siamese": SiameseUNet,
    "HRCloudNet": HRCloudNet, 
    "CDnetV2" : CDnetV2,
    "SwinCloud" : SwinCloud,
    "bam-cd" : BAM_CD
}
