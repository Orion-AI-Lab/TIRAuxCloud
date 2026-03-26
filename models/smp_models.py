from model_builder.base_model import BaseModel 
import segmentation_models_pytorch as smp

class UnetModel(BaseModel) : 
    def __init__(self,in_channels, num_classes):
        super().__init__()
        self._model = smp.Unet(
            encoder_name="resnet34",
            in_channels=in_channels,
            classes=num_classes
        )

    def forward(self,inputs) : 
        return self._model(inputs) 
    
    @property
    def name(self) -> str : 
        return "Unet" 

    @classmethod
    def from_config(cls, config):
        return cls(
            in_channels = len(config["features"]) ,
            num_classes = config["num_classes"]
        )

class SegFormerModel(BaseModel) :
    def __init__(self,in_channels,num_classes) : 
        super.__init__()
        self._model = smp.create_model(
            arch="segformer" ,
            encoder_name="mit_b2",
            encoder_weights="imagenet", # pretrained on ImageNet
            in_channels=in_channels,
            classes=num_classes
        )
    
    def forward(self, inputs):
        return self._model(inputs)
    
    @property
    def name(self) : 
        return "SegFormer"
    
    @classmethod
    def from_config(cls, config) : 
        return cls(
            in_channels = len(config["features"]),
            num_classes = config["num_classes"]
        )
    
class DeepLabV3Model(BaseModel):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self._model = smp.create_model(
            arch="deeplabv3plus",
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes
        )

    def forward(self, x):
        return self._model(x)

    @property
    def name(self) -> str:
        return "DeepLabV3"

    @classmethod
    def from_config(cls, config):
        return cls(
            in_channels=len(config["features"]),
            num_classes=config["num_classes"]
        )
    
class SwinUnetModel(BaseModel):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # initmodel = smp.Unet(encoder_name='swin_t', 
        #                     encoder_weights="imagenet", 
        #                     in_channels=len(featset), classes=num_classes).to(device)
        self._model = smp.create_model(
            arch="upernet",
            encoder_name="tu-swinv2_cr_tiny_224",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes
        )

    def forward(self, x):
        return self._model(x)

    @property
    def name(self) -> str:
        return "Swin-Unet"

    @classmethod
    def from_config(cls, config):
        return cls(
            in_channels=len(config["features"]),
            num_classes=config["num_classes"]
        )
