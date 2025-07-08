import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from image_patcher import ImagePatcher

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class GatedAttentionMILImage(nn.Module):
    def __init__(self,
                 num_classes=4,
                 shared_attention=False,
                 shared_classifier=False,
                 backbone='r18',
                 pretrained=True,
                 L=512,
                 D=128,
                 feature_dropout=0.1,
                 attention_dropout=0.1):

        super().__init__()
        self.num_classes = num_classes
        self.shared_attention = shared_attention
        self.shared_classifier = shared_classifier
        self.L = L
        self.D = D

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

        self.feature_extractor.fc = Identity()
        self.feature_dropout = nn.Dropout(feature_dropout)

        if self.shared_attention:
            self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
            self.attention_weight = nn.Linear(self.D, 1)
            self.attention_dropout = nn.Dropout(attention_dropout)
        else:
            self.attention_V = nn.ModuleList([
                nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh()) for _ in range(self.num_classes)
            ])
            self.attention_U = nn.ModuleList([
                nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid()) for _ in range(self.num_classes)
            ])
            self.attention_weight = nn.ModuleList([
                nn.Linear(self.D, 1) for _ in range(self.num_classes)
            ])
            self.attention_dropout = nn.ModuleList([
                nn.Dropout(attention_dropout) for _ in range(self.num_classes)
            ])

        if self.shared_classifier:
            self.classifier = nn.Linear(self.L, self.num_classes)
        else:
            self.classifiers = nn.ModuleList([
                nn.Linear(self.L, 1) for _ in range(self.num_classes)
            ])

    def forward(self, x):
        # x shape: (batch_size, num_instances, channels, width, height)
        bs, num_instances, ch, w, h = x.shape

        x = x.view(bs * num_instances, ch, w, h)
        H = self.feature_extractor(x)  # (bs*num_instances, L)
        H = self.feature_dropout(H)
        H = H.view(bs, num_instances, -1)  # (bs, num_instances, L)

        if self.shared_attention:
            A_V = self.attention_V(H)       # (bs, num_instances, D)
            A_U = self.attention_U(H)       # (bs, num_instances, D)
            A = self.attention_weight(A_V * A_U)  # (bs, num_instances, 1)
            A = A.transpose(2, 1)           # (bs, 1, num_instances)
            A = self.attention_dropout(A)
            A = F.softmax(A, dim=2)         # attention weights sum to 1 over instances
            M = torch.matmul(A, H)          # (bs, 1, L)
            M = M.squeeze(1)                # (bs, L)
        else:
            M_list = []
            for i in range(self.num_classes):
                A_V = self.attention_V[i](H)     # (bs, num_instances, D)
                A_U = self.attention_U[i](H)     # (bs, num_instances, D)
                A = self.attention_weight[i](A_V * A_U)  # (bs, num_instances, 1)
                A = A.transpose(2, 1)            # (bs, 1, num_instances)
                A = self.attention_dropout[i](A)
                A = F.softmax(A, dim=2)
                M_i = torch.matmul(A, H)         # (bs, 1, L)
                M_list.append(M_i.squeeze(1))   # (bs, L)
            M = torch.stack(M_list, dim=1)      # (bs, num_classes, L)

        if self.shared_classifier:
            if self.shared_attention:
                Y = self.classifier(M)           # (bs, num_classes)
            else:
                M_flat = M.view(bs, -1)         # (bs, num_classes * L)
                Y = self.classifier(M_flat)     # (bs, num_classes)
        else:
            if self.shared_attention:
                Y_list = [self.classifiers[i](M) for i in range(self.num_classes)]
            else:
                Y_list = [self.classifiers[i](M[:, i, :]) for i in range(self.num_classes)]
            Y = torch.cat(Y_list, dim=1)       # (bs, num_classes)

        return Y, M


class MILTabular(nn.Module):
    def __init__(
        self,
        input_dim=102,
        num_classes=4,
        hidden_dim=128,
        feature_dropout=0.1,
        shared_classifier=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.shared_classifier = shared_classifier

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(feature_dropout)
        )

        if shared_classifier:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.classifiers = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(num_classes)
            ])

    def forward(self, x):
        # x: (bs, n_instances, input_dim)
        bs, n, d = x.shape
        H = self.encoder(x)                      # (bs, n, hidden_dim)
        M = H.mean(dim=1)                        # (bs, hidden_dim) mean pooling

        if self.shared_classifier:
            Y = self.classifier(M)               # (bs, num_classes)
        else:
            Y_list = [clf(M) for clf in self.classifiers]
            Y = torch.cat(Y_list, dim=1)         # (bs, num_classes)

        return Y, M


class IntermediateFusionModel(nn.Module):
    def __init__(self, modality, device, out_dim=4):
        super(IntermediateFusionModel, self).__init__()

        self.modality = modality
        self.device = device
        self.out_dim = out_dim
        self.shared_classifier = True
        self.mil = GatedAttentionMILImage(num_classes=out_dim,
                                          shared_attention=True,
                                          shared_classifier=self.shared_classifier)
        
        self.tab_mil = MILTabular(input_dim=39,
                                  num_classes=out_dim,
                                  hidden_dim=64,
                                  shared_classifier=self.shared_classifier)

        self.image_feat_dim = 512
        self.rad_feat_dim = 64 if 'radiomics' in modality else 0
        self.clin_feat_dim = 32 if 'clinical' in modality else 0
        self.meta_feat_dim = 64 if 'metalesion' in modality else 0


        self.cli_mlp = nn.Sequential(nn.Linear(7, self.clin_feat_dim), nn.ReLU())
        self.meta_mlp = nn.Sequential(nn.Linear(36, self.meta_feat_dim), nn.ReLU())

        self.fusion_input_dim = (
            self.image_feat_dim +
            self.rad_feat_dim +
            self.clin_feat_dim +
            self.meta_feat_dim
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        if self.shared_classifier:
            self.classifier = nn.Linear(128, out_dim)
        else:

            self.classifier = nn.ModuleList([
                nn.Linear(128, 1) for _ in range(out_dim)
            ])

        self.patcher = ImagePatcher(
            patch_size=128,
            overlap=0.5,
            empty_thresh=0.75,
            bag_size=-1
        )
        self.patcher.get_tiles(2294, 1914)

    def forward(self, image, clinical_feat=None, radiomics_feat=None, metalesion_feat=None):
        instances, _, _ = self.patcher.convert_img_to_bag(image.squeeze(0))
        instances = instances.unsqueeze(0).to(self.device)
        # normalize instances with IMAGENET mean and std
        # instances = (instances - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / \
        #              torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        _, image_feats = self.mil(instances)  # [1, 4, 512]
        if 'radiomics' in self.modality:
            _, rad_feats = self.tab_mil(radiomics_feat)  # [1, 4, 128]
        else:
            rad_feats = None
        if 'clinical' in self.modality:
            clinical_feat = self.cli_mlp(clinical_feat)
            # clinical_feat = clinical_feat.unsqueeze(1).repeat(1, self.out_dim, 1)
        else:
            clinical_feat = None
        if 'metalesion' in self.modality:
            metalesion_feat = self.meta_mlp(metalesion_feat)
            # metalesion_feat = metalesion_feat.unsqueeze(1).repeat(1, self.out_dim, 1)
        else:
            metalesion_feat = None

        out = []
        if self.shared_classifier:
            for i in range(1):
                img_i = image_feats
                rad_i = rad_feats if rad_feats is not None else torch.empty(0, device=self.device)
                clin_i = clinical_feat if clinical_feat is not None else torch.empty(0, device=self.device)
                meta_i = metalesion_feat if metalesion_feat is not None else torch.empty(0, device=self.device)

                fused_i = torch.cat([x for x in [img_i, rad_i, clin_i, meta_i] if x.numel() > 0], dim=-1)
                fused_i = self.fusion_mlp(fused_i)  # [1, 128]
                out_i = self.classifier(fused_i)  # [1, 4]
                out.append(out_i)
        else:
            for i in range(self.out_dim):
                img_i = image_feats[:, i, :]  # [1, 512]
                rad_i = rad_feats[:, i, :] if rad_feats is not None else torch.empty(0, device=self.device)
                clin_i = clinical_feat[:, i, :] if clinical_feat is not None else torch.empty(0, device=self.device)
                meta_i = metalesion_feat[:, i, :] if metalesion_feat is not None else torch.empty(0, device=self.device)

                fused_i = torch.cat([x for x in [img_i, rad_i, clin_i, meta_i] if x.numel() > 0], dim=-1)
                fused_i = self.fusion_mlp(fused_i)  # [1, 128]
                out_i = self.classifier[i](fused_i)  # [1, 1]
                out.append(out_i)

        out = torch.cat(out, dim=-1)
        return out

class DecisionLevelLateFusionModel(nn.Module):
    def __init__(self, modality, device, hidden_dim=128, out_dim=1):
        super(DecisionLevelLateFusionModel, self).__init__()
        
        self.modality = modality
        self.device = device
        self.shared_classifier = True

        # image branch (MIL) with classifier
        self.mil = GatedAttentionMILImage(num_classes=out_dim,
                                          shared_attention=True,
                                          shared_classifier=self.shared_classifier)

        # clinical branch with classifier
        if 'clinical' in self.modality:
            self.clinical_classifier = nn.Sequential(
                nn.Linear(7, 32),
                nn.ReLU(),
                nn.Linear(32, out_dim)
            )
        
        # radiomics branch with classifier
        if 'radiomics' in self.modality:
            self.tab_mil = MILTabular(input_dim=39,
                                  num_classes=out_dim,
                                  hidden_dim=64,
                                  shared_classifier=self.shared_classifier)
        
        # metalesion branch with classifier
        if 'metalesion' in self.modality:
            self.metalesion_classifier = nn.Sequential(
                nn.Linear(36, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim)
            )
        
        num_modalities = ('image' in modality) + ('clinical' in modality) + ('radiomics' in modality) + ('metalesion' in modality)
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        self.patcher = ImagePatcher(patch_size=128,
                            overlap=0.5,
                            empty_thresh=0.75,
                            bag_size=-1)
        self.patcher.get_tiles(2294, 1914)
        self.device = device
    
    def forward(self, image, clinical_feat=None, radiomics_feat=None, metalesion_feat=None):

        predictions = []
        
        if 'image' in self.modality:
            instances, _, _ = self.patcher.convert_img_to_bag(image.squeeze(0))
            instances = instances.unsqueeze(0).to(self.device)

            # normalize instances with IMAGENET mean and std
            # instances = (instances - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / \
            #          torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

            image_pred, _ = self.mil(instances)
            # image_pred = self.image_classifier(image_feat.squeeze(0))
            predictions.append(image_pred)
        
        if 'clinical' in self.modality:
            clinical_pred = self.clinical_classifier(clinical_feat)
            predictions.append(clinical_pred)
            
        if 'radiomics' in self.modality:
            radiomics_pred, _ = self.tab_mil(radiomics_feat)
            predictions.append(radiomics_pred)
        
        if 'metalesion' in self.modality:
            metalesion_pred = self.metalesion_classifier(metalesion_feat)
            predictions.append(metalesion_pred)
        stacked_preds = torch.stack(predictions, dim=0)
        normalized_weights = F.softmax(self.modality_weights, dim=0)
        final_pred = (stacked_preds * normalized_weights.view(-1, 1, 1)).sum(dim=0)

        return final_pred