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

    def forward(self,x) : 
        return self._model(x) 
    
    @property
    def name(self) -> str : 
        return "Unet" 

    @classmethod
    def from_config(cls, config):
        return cls(
            in_channels = len(config["features"]) ,
            num_classes = config["num_classes"]
        )
