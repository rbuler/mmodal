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
        self.device = device

        self.mil = GatedAttentionMIL()

        tab_dim = 7 if 'clinical' in modality else 0
        rad_dim = 102 if 'radiomics' in modality else 0
        metalesion_dim = 36 if 'metalesion' in modality else 0
        mask_dim = 1 if 'radiomics' in modality else 0
        total_dim = self.mil.D + tab_dim + rad_dim + metalesion_dim + mask_dim

        self.rad_norm = nn.LayerNorm(rad_dim) if rad_dim > 0 else None

        self.fc_layers = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

        self.patcher = ImagePatcher(
            patch_size=128,
            overlap=0.5,
            empty_thresh=0.75,
            bag_size=-1
        )
        self.patcher.get_tiles(2294, 1914)

    def forward(self, image, clinical_feat=None, radiomics_feat=None, metalesion_feat=None, mask_feat=None):

        instances, _, _ = self.patcher.convert_img_to_bag(image.squeeze(0))
        instances = instances.unsqueeze(0).to(self.device)
        image_feat, _ = self.mil(instances)
        x = image_feat.squeeze(0)

        # cat features
        if 'clinical' in self.modality:
            x = torch.cat([x, clinical_feat], dim=1)
        if 'radiomics' in self.modality:
            rad_normed = self.rad_norm(radiomics_feat) if self.rad_norm else radiomics_feat
            mask_feat = mask_feat.unsqueeze(1)
            x = torch.cat([x, rad_normed, mask_feat], dim=1)
        if 'metalesion' in self.modality:
            x = torch.cat([x, metalesion_feat], dim=1)

        x = self.fc_layers(x)
        return x


class DecisionLevelLateFusionModel(nn.Module):
    def __init__(self, modality, device, hidden_dim=128, out_dim=1):
        super(DecisionLevelLateFusionModel, self).__init__()
        
        self.modality = modality
        self.device = device
        
        # image branch (MIL) with classifier
        self.mil = GatedAttentionMIL()
        self.image_classifier = nn.Sequential(
            nn.Linear(self.mil.D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        # clinical branch with classifier
        if 'clinical' in self.modality:
            self.clinical_classifier = nn.Sequential(
                nn.Linear(7, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim)
            )
        
        # radiomics branch with classifier
        if 'radiomics' in self.modality:
            self.rad_dim = 102
            self.rad_norm = nn.LayerNorm(self.rad_dim)
            self.radiomics_classifier = nn.Sequential(
                nn.Linear(self.rad_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim)
            )
            self.mask_classifier = nn.Sequential(
                nn.Linear(1, hidden_dim // 8),
                nn.ReLU(),
                nn.Linear(hidden_dim // 8, out_dim)
            )
        
        # metalesion branch with classifier
        if 'metalesion' in self.modality:
            self.metalesion_dim = 36
            self.metalesion_classifier = nn.Sequential(
                nn.Linear(self.metalesion_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim)
            )
        
        num_modalities = 1 + ('clinical' in modality) + ('radiomics' in modality) * 2 + ('metalesion' in modality)
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        
        self.patcher = ImagePatcher(patch_size=128,
                            overlap=0.5,
                            empty_thresh=0.75,
                            bag_size=-1)
        self.patcher.get_tiles(2294, 1914)
        self.device = device
    
    def forward(self, image, clinical_feat=None, radiomics_feat=None, metalesion_feat=None, mask_feat=None):

        predictions = []
        
        instances, _, _ = self.patcher.convert_img_to_bag(image.squeeze(0))
        instances = instances.unsqueeze(0).to(self.device)
        image_feat, _ = self.mil(instances)
        image_pred = self.image_classifier(image_feat.squeeze(0))
        predictions.append(image_pred)
        
        if 'clinical' in self.modality:
            clinical_pred = self.clinical_classifier(clinical_feat)
            predictions.append(clinical_pred)
        
        if 'radiomics' in self.modality:
            rad_normed = self.rad_norm(radiomics_feat)
            radiomics_pred = self.radiomics_classifier(rad_normed)
            predictions.append(radiomics_pred)
            
            mask_feat = mask_feat.unsqueeze(1)
            mask_pred = self.mask_classifier(mask_feat)
            predictions.append(mask_pred)
        
        if 'metalesion' in self.modality:
            metalesion_pred = self.metalesion_classifier(metalesion_feat)
            predictions.append(metalesion_pred)
        
        stacked_preds = torch.stack(predictions, dim=0)
        normalized_weights = F.softmax(self.modality_weights, dim=0)
        final_pred = (stacked_preds * normalized_weights.view(-1, 1, 1)).sum(dim=0)

        return final_pred