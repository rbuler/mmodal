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
                                                  1))        ## 1 for baseline, self.D  for multimodal
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


class GatedAttentionMIL_Tabular(nn.Module):
    def __init__(
        self,
        input_dim=102,
        num_classes=4,
        hidden_dim=128,
        attention_dim=64,
        feature_dropout=0.1,
        attention_dropout=0.1,
        shared_attention=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.shared_attention = shared_attention

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(feature_dropout)
        )

        if shared_attention:
            self.attention_V = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Sigmoid())
        else:
            self.attention_V = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh())
                for _ in range(num_classes)
            ])
            self.attention_U = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Sigmoid())
                for _ in range(num_classes)
            ])

        self.attention_weights = nn.ModuleList([
            nn.Linear(attention_dim, 1) for _ in range(num_classes)
        ])

        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 1, bias=False) for _ in range(num_classes)
        ])

        self.attention_dropouts = nn.ModuleList([
            nn.Dropout(attention_dropout) for _ in range(num_classes)
        ])

    def forward(self, x):
        bs, n, d = x.shape
        H = self.encoder(x)  # (bs, n, hidden_dim)

        M = []
        A_all = []

        for i in range(self.num_classes):
            if self.shared_attention:
                A_V = self.attention_V(H)
                A_U = self.attention_U(H)
            else:
                A_V = self.attention_V[i](H)
                A_U = self.attention_U[i](H)

            A = self.attention_weights[i](A_V * A_U)  # (bs, n, 1)
            A = torch.transpose(A, 2, 1)  # (bs, 1, n)
            A = self.attention_dropouts[i](A)
            A = F.softmax(A, dim=2)  # normalize over instances

            A_all.append(A)
            M.append(torch.matmul(A, H))  # (bs, 1, hidden_dim)

        M = torch.cat(M, dim=1)  # (bs, num_classes, hidden_dim)
        A_all = torch.cat(A_all, dim=1)  # (bs, num_classes, n)

        Y = [self.classifiers[i](M[:, i, :]) for i in range(self.num_classes)]
        Y = torch.cat(Y, dim=-1)  # (bs, num_classes)

        return Y, M



class MultiHeadGatedAttentionMIL(nn.Module):
    def __init__(
            self,
            num_classes=4,
            backbone='r18',
            pretrained=True,
            L=512,
            D=128,
            feature_dropout=0.1,
            attention_dropout=0.1,
            shared_attention=False,
            config=None):
        
        super().__init__()

        self.L = L
        self.D = D
        self.num_classes = num_classes
        self.shared_attention = shared_attention

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

        # Attention mechanism (Shared or Separate)
        if shared_attention:
            self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        else:
            self.attention_V = nn.ModuleList([
                nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
                for _ in range(self.num_classes)
            ])
            self.attention_U = nn.ModuleList([
                nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
                for _ in range(self.num_classes)
            ])

        # Attention weight layers (Always separate per class)
        self.attention_weights = nn.ModuleList([
            nn.Linear(self.D, 1) for _ in range(self.num_classes)
        ])

        # Classifiers per class
        self.classifiers = nn.ModuleList([
            nn.Linear(self.L, 1, bias=False) for _ in range(self.num_classes)
        ])

        # Dropout layers
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.attention_dropouts = nn.ModuleList([
            nn.Dropout(attention_dropout) for _ in range(self.num_classes)
        ])

    def forward(self, x):

        bs, num_instances, ch, w, h = x.shape
        x = x.view(bs*num_instances, ch, w, h)
        H = self.feature_extractor(x)
        H = self.feature_dropout(H)
        H = H.view(bs, num_instances, -1)

        M = []
        A_all = []

        for i in range(self.num_classes):
            if self.shared_attention:
                A_V = self.attention_V(H)
                A_U = self.attention_U(H)
            else:
                A_V = self.attention_V[i](H)
                A_U = self.attention_U[i](H)

            A = self.attention_weights[i](A_V * A_U)
            A = torch.transpose(A, 2, 1)  # (bs, 1, num_instances)
            A = self.attention_dropouts[i](A)
            A = F.softmax(A, dim=2)

            A_all.append(A)
            M.append(torch.matmul(A, H))

        M = torch.cat(M, dim=1)  # (bs, num_classes, L)
        A_all = torch.cat(A_all, dim=1)  # (bs, num_classes, num_instances)

        Y = [self.classifiers[i](M[:, i, :]) for i in range(self.num_classes)]
        Y = torch.cat(Y, dim=-1)  # (bs, num_classes)

        return Y, M


class IntermediateFusionModel(nn.Module):
    def __init__(self, modality, device, hidden_dim=128, out_dim=4):
        super(IntermediateFusionModel, self).__init__()

        self.modality = modality
        self.device = device
        self.out_dim = out_dim

        # MIL branches
        self.mil = MultiHeadGatedAttentionMIL(num_classes=out_dim)
        self.tab_mil = GatedAttentionMIL_Tabular(input_dim=102, num_classes=out_dim)

        # Feature sizes
        self.image_feat_dim = self.mil.L  # usually 512
        self.rad_feat_dim = hidden_dim if 'radiomics' in modality else 0
        self.clin_feat_dim = 8 if 'clinical' in modality else 0
        self.meta_feat_dim = 16 if 'metalesion' in modality else 0

        self.rad_norm = nn.LayerNorm(102) if 'radiomics' in modality else None


        self.cli_mlp = nn.Sequential(nn.Linear(7, self.clin_feat_dim), nn.ReLU())
        self.meta_mlp = nn.Sequential(nn.Linear(36, self.meta_feat_dim), nn.ReLU())

        # Per-class fusion classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(
                self.image_feat_dim + self.rad_feat_dim + self.clin_feat_dim + self.meta_feat_dim,
                1
            ) for _ in range(out_dim)
        ])

        # Optional patcher for MIL image inputs
        self.patcher = ImagePatcher(
            patch_size=128,
            overlap=0.75,
            empty_thresh=0.75,
            bag_size=-1
        )
        self.patcher.get_tiles(2294, 1914)

    def forward(self, image, clinical_feat=None, radiomics_feat=None, metalesion_feat=None):
        # --- Image MIL ---
        instances, _, _ = self.patcher.convert_img_to_bag(image.squeeze(0))  # shape: [num_patches, C, H, W]
        instances = instances.unsqueeze(0).to(self.device)  # shape: [1, num_patches, C, H, W]
        _, image_feats = self.mil(instances)  # image_feats: [1, 4, 512]

        # --- Radiomics MIL ---
        if 'radiomics' in self.modality:
            rad_normed = self.rad_norm(radiomics_feat) if self.rad_norm else radiomics_feat  # [1, n, 102]
            _, rad_feats = self.tab_mil(rad_normed)  # rad_feats: [1, 4, 128]
        else:
            rad_feats = None

        # Expand clinical/meta to per-class
        if 'clinical' in self.modality:
            clinical_feat = self.cli_mlp(clinical_feat)
            clinical_feat = clinical_feat.unsqueeze(1).repeat(1, self.out_dim, 1)  # [1, 4, clin_feat_dim]
        else:
            clinical_feat = None

        if 'metalesion' in self.modality:
            metalesion_feat = self.meta_mlp(metalesion_feat)
            metalesion_feat = metalesion_feat.unsqueeze(1).repeat(1, self.out_dim, 1)  # [1, 4, meta_feat_dim]
        else:
            metalesion_feat = None

        # --- Fuse all features per class ---
        fused = []
        for i in range(self.out_dim):
            img_i = image_feats[:, i, :]  # [1, 512]
            rad_i = rad_feats[:, i, :] if rad_feats is not None else torch.empty(0, device=self.device)
            clin_i = clinical_feat[:, i, :] if clinical_feat is not None else torch.empty(0, device=self.device)
            meta_i = metalesion_feat[:, i, :] if metalesion_feat is not None else torch.empty(0, device=self.device)

            # Concatenate per-class fused vector
            fused_i = torch.cat([x for x in [img_i, rad_i, clin_i, meta_i] if x.numel() > 0], dim=-1)  # [1, D]
            fused.append(fused_i)

        fused = torch.stack(fused, dim=1)  # [1, 4, D]
        # --- Classifiers ---
        out = [self.classifiers[i](fused[:, i, :]) for i in range(self.out_dim)]  # 4 × [1, 1]
        out = torch.cat(out, dim=-1)  # [1, 4]

        return out


class DecisionLevelLateFusionModel(nn.Module):
    def __init__(self, modality, device, hidden_dim=128, out_dim=1):
        super(DecisionLevelLateFusionModel, self).__init__()
        
        self.modality = modality
        self.device = device
        
        # image branch (MIL) with classifier
        self.mil = MultiHeadGatedAttentionMIL()
        self.image_classifier = nn.Sequential(
            nn.Linear(self.mil.D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))
        
        # clinical branch with classifier
        if 'clinical' in self.modality:
            self.clinical_classifier = nn.Sequential(
                nn.Linear(7, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim))
        
        # radiomics branch with classifier
        if 'radiomics' in self.modality:
            self.rad_norm = nn.LayerNorm(102)
            self.tab_mil = GatedAttentionMIL_Tabular(input_dim=102, num_classes=4)
        
        # metalesion branch with classifier
        if 'metalesion' in self.modality:
            self.metalesion_dim = 36
            self.metalesion_classifier = nn.Sequential(
                nn.Linear(self.metalesion_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim)
            )
        
        num_modalities = ('image' in modality) + ('clinical' in modality) + ('radiomics' in modality) + ('metalesion' in modality)
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        self.patcher = ImagePatcher(patch_size=128,
                            overlap=0.75,
                            empty_thresh=0.75,
                            bag_size=-1)
        self.patcher.get_tiles(2294, 1914)
        self.device = device
    
    def forward(self, image, clinical_feat=None, radiomics_feat=None, metalesion_feat=None):

        predictions = []
        
        if 'image' in self.modality:
            instances, _, _ = self.patcher.convert_img_to_bag(image.squeeze(0))
            instances = instances.unsqueeze(0).to(self.device)
            image_pred, _ = self.mil(instances)
            # image_pred = self.image_classifier(image_feat.squeeze(0))
            predictions.append(image_pred)
        
        if 'clinical' in self.modality:
            clinical_pred = self.clinical_classifier(clinical_feat)
            predictions.append(clinical_pred)
            
        if 'radiomics' in self.modality:
            rad_normed = self.rad_norm(radiomics_feat)
            radiomics_pred, _ = self.tab_mil(rad_normed)
            predictions.append(radiomics_pred)
        
        if 'metalesion' in self.modality:
            metalesion_pred = self.metalesion_classifier(metalesion_feat)
            predictions.append(metalesion_pred)
        
        stacked_preds = torch.stack(predictions, dim=0)
        normalized_weights = F.softmax(self.modality_weights, dim=0)
        final_pred = (stacked_preds * normalized_weights.view(-1, 1, 1)).sum(dim=0)
        return final_pred