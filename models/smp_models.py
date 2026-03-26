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
    
