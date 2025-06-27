import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from image_patcher import ImagePatcher

class Flatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        input_shape = x.shape
        output_shape = [input_shape[i] for i in range(self.dim)] + [-1]
        return x.view(*output_shape)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class GatedAttentionMIL(nn.Module):

    def __init__(
                self,
                num_classes=1,
                backbone='r18',
                pretrained=True,
                L=512,
                D=128,
                K=1,
                feature_dropout=0.1,
                attention_dropout=0.1):

        super().__init__()
        self.L = L  
        self.D = D
        self.K = K
        if pretrained:
            if backbone == 'r18':
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet18(weights=weights)
            elif backbone == 'r34':
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet34(weights=weights)
            elif backbone == 'r50':
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.feature_extractor = models.resnet50(weights=weights)
        else:
            self.feature_extractor = models.resnet18()
        self.num_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = Identity()

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Dropout(attention_dropout)
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
            nn.Dropout(attention_dropout)
        )
        self.attention_weights = nn.Linear(self.D, self.K)
        self.classifier = nn.Sequential(
                                        nn.Linear(self.L * self.K,
                                                  self.D))        ## 1 for baseline, self.D  for multimodal
        self.feature_dropout = nn.Dropout(feature_dropout)

    def forward(self, x):
        bs, num_instances, ch, w, h = x.shape
        x = x.view(bs*num_instances, ch, w, h)
        H = self.feature_extractor(x)
        H = self.feature_dropout(H)
        H = H.view(bs, num_instances, -1)
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(torch.mul(A_V, A_U))
        A = torch.transpose(A, 2, 1)
        A = F.softmax(A, dim=2)
        m = torch.matmul(A, H)
        Y = self.classifier(m)
        return Y, A



class EarlyFusionModel(nn.Module):
    def __init__(self, modality, device, hidden_dim=128, out_dim=1):
        super(EarlyFusionModel, self).__init__()

        self.modality = modality
        
        self.mil = GatedAttentionMIL()


        if self.modality == 'image-clinical':
            tab_dim = 7
            total_dim = self.mil.D + tab_dim
        elif self.modality == 'image-clinical-radiomics':
            tab_dim = 7
            rad_dim = 102
            total_dim = self.mil.D + tab_dim + rad_dim + 1
        elif self.modality == 'image-clinical-radiomics-metalesion':
            tab_dim = 7
            rad_dim = 102
            metalesion_dim = 36
            total_dim = self.mil.D + tab_dim + rad_dim + metalesion_dim + 1
        else:
            total_dim = self.mil.D  # Default for 'image' modality


        self.rad_norm = nn.LayerNorm(rad_dim) if 'radiomics' in self.modality else None

        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, out_dim)

        self.patcher = ImagePatcher(patch_size=128,
                            overlap=0.5,
                            empty_thresh=0.75,
                            bag_size=-1)
        self.patcher.get_tiles(2294, 1914)
        self.device = device

    def forward(self, image, clinical_feat=None, radiomics_feat=None, metalesion_feat=None, mask_feat=None):

        instances, _, _ = self.patcher.convert_img_to_bag(image.squeeze(0))
        instances = instances.unsqueeze(0)
        instances = instances.to(self.device)

        image_feat, _ = self.mil(instances)
        x = image_feat.squeeze(0)
        
        if self.modality == 'image-clinical':
            x = torch.cat([x, clinical_feat], dim=1)

        elif self.modality == 'image-clinical-radiomics':
            rad_normed = self.rad_norm(radiomics_feat)
            mask_feat = mask_feat.unsqueeze(1)
            x = torch.cat([x, clinical_feat, rad_normed, mask_feat], dim=1)

        elif self.modality == 'image-clinical-radiomics-metalesion':
            rad_normed = self.rad_norm(radiomics_feat)
            mask_feat = mask_feat.unsqueeze(1)
            x = torch.cat([x, clinical_feat, rad_normed, metalesion_feat, mask_feat], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # logits
        return x
