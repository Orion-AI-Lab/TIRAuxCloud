from models.bamcd.model import BAM_CD
from models.cloudseg.models.components.hrcloudnet import HRCloudNet
from models.cloudseg.models.components.cdnetv2 import CDnetV2
from models.smp_models import UnetModel, SegFormerModel, DeepLabV3Model, SwinUnetModel
from models.swincloud.swincloud import SwinCloud
from models.siamese_unet import SiameseUNet

print("[INFO] Model registry initialized: all available models successfully loaded.")