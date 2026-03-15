import torch
import torch.nn.functional as F
from torch import Tensor, nn

class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, channels_s, channels_t, tau=1.0, device=None):
        super().__init__()
        self.tau = tau

    # def forward(self, y_s, y_t):
    #     assert len(y_s) == len(y_t)
    #     losses = []

    #     for s, t in zip(y_s, y_t):
    #         assert s.shape == t.shape
    #         N, C, H, W = s.shape

    #         # Flatten spatial dimensions: (N, C, H*W)
    #         s = s.view(N, C, -1) / self.tau
    #         t = t.view(N, C, -1) / self.tau

    #         # Softmax over spatial dimension (H*W)
    #         p_s = F.log_softmax(s, dim=2)   # log p_s
    #         p_t = F.softmax(t, dim=2)       # p_t

    #         # KL divergence per channel, averaged
    #         kl = F.kl_div(p_s, p_t, reduction='batchmean') * (self.tau ** 2)

    #         losses.append(kl)

    #     return sum(losses) / len(losses)

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape

            # normalize in channel dimension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)
        #print(f"CWD Loss: {loss.item()}")
        return loss

class MGDLoss(nn.Module):
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=0.0002,  #0.00002,
                 lambda_mgd=0.65,
                 device=None
                 ):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        self.device = device

        # self.generation = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(s_chan, t_chan, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(t_chan, t_chan, kernel_size=3, padding=1)
        #     ).to(device) for s_chan, t_chan in zip(student_channels, teacher_channels)
        # ])
        self.generation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chan, chan, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(chan, chan, kernel_size=3, padding=1)
            ).to(device) for chan in teacher_channels
        ])

    def forward(self, y_s, y_t, layer=None):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # assert s.shape == t.shape
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0, device=None):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller
        self.device = device
        
        # Convert to ModuleList and ensure consistent dtype
        self.align_module = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.norm1 = nn.ModuleList()

        # Create alignment modules
        for i, (s_chan, t_chan) in enumerate(zip(channels_s, channels_t)):
            align = nn.Sequential(
                nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(t_chan, affine=False)
            ).to(self.device)
            self.align_module.append(align)
            
        # Create normalization layers
        for t_chan in channels_t:
            self.norm.append(nn.BatchNorm2d(t_chan, affine=False).to(device))
            
        for s_chan in channels_s:
            self.norm1.append(nn.BatchNorm2d(s_chan, affine=False).to(device))

        if distiller == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t, device=self.device)
        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t, device=self.device)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t):
        # if len(y_s) != len(y_t):
        #     y_t = y_t[len(y_t) // 2:]
        min_len = min(len(y_s), len(y_t))

        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # Match input dtype to module dtype

            s = s.type(next(self.align_module[idx].parameters()).dtype)
            t = t.type(next(self.align_module[idx].parameters()).dtype)
            
            if self.distiller == "cwd":
                s = self.align_module[idx](s)
                stu_feats.append(s)
                tea_feats.append(t.detach())
            else:  # mgd
                s = self.align_module[idx](s)  # align student to teacher channels before generation
                stu_feats.append(s)
                tea_feats.append(t.detach())   # raw teacher features (no BN normalization)

        loss = self.feature_loss(stu_feats, tea_feats)
        return self.loss_weight * loss


class DistillationLoss:
    def __init__(self, model_s, model_t, distiller="CWDLoss", device=None):
        self.distiller = distiller
        # self.layers = ["6", "8", "13", "16", "19", "22"] # original
        self.layers = ["6", "8", "12", "15", "18", "21"]  # new
        self.model_s = model_s 
        self.model_t = model_t
        self.device = device

        # ini warm up
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 640)
            _ = self.model_s(dummy_input.to(self.device))
            _ = self.model_t(dummy_input.to(self.device))
        
        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []
        self.remove_handle = []
        
        self._find_layers()
        
        self.distill_loss_fn = FeatureLoss(
            channels_s=self.channels_s, 
            channels_t=self.channels_t, 
            distiller=distiller[:3],
            device=self.device
        )
        
    def _find_layers(self):

        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []

        print("\nSearching teacher modules...")
        for name, ml in self.model_t.named_modules():
            if name is not None:
                parts = name.split(".")      
                if parts[0] != "model" or len(parts) <= 3 or len(parts) > 5:
                    continue
                if parts[1] in self.layers and parts[2] == "conv2":
                    if hasattr(ml, 'conv'):
                        print(f"Found teacher layer: {parts}, channels={ml.conv.out_channels}")
                        self.channels_t.append(ml.conv.out_channels)
                        self.teacher_module_pairs.append(ml)
        print("\nSearching student modules...")
        for name, ml in self.model_s.named_modules():
            if name is not None:
                parts = name.split(".")
                if parts[0] != "model" or len(parts) <= 3 or len(parts) > 5:
                    continue
                if parts[1] in self.layers and parts[2] == "conv2":
                    if hasattr(ml, 'conv'):
                        print(f"Found student layer: {parts}, channels={ml.conv.out_channels}")
                        self.channels_s.append(ml.conv.out_channels)
                        self.student_module_pairs.append(ml)


        nl = min(len(self.channels_s), len(self.channels_t))
        print(f"\nUsing last {nl} layers for distillation")
        self.channels_s = self.channels_s[-nl:]
        self.channels_t = self.channels_t[-nl:]
        self.teacher_module_pairs = self.teacher_module_pairs[-nl:]
        self.student_module_pairs = self.student_module_pairs[-nl:]

    def register_hook(self):
        # Remove the existing hook if they exist
        self.remove_handle_()
        
        self.teacher_outputs = []
        self.student_outputs = []

        def make_student_hook(l):
            def forward_hook(m, input, output):
                # print(f"[Student Hook Called] Module: {m.__class__.__name__}")
                if isinstance(output, torch.Tensor):
                    # print(f"[Student Hook] {m} output shape: {output.shape}")
                    out = output.clone()  # Clone to ensure we don't modify the original
                    l.append(out)
                else:
                    l.append([o.clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        def make_teacher_hook(l):
            def forward_hook(m, input, output):
                # print(f"[Teacher Hook Called] Module: {m.__class__.__name__}")
                if isinstance(output, torch.Tensor):
                    # print(f"[Teacher Hook] {m} output shape: {output.shape}")
                    l.append(output.detach().clone())  # Detach and clone teacher outputs
                else:
                    l.append([o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        for ml, ori in zip(self.teacher_module_pairs, self.student_module_pairs):
            self.remove_handle.append(ml.register_forward_hook(make_teacher_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_student_hook(self.student_outputs)))

    def get_loss(self):
        if not self.teacher_outputs or not self.student_outputs:
            print("Warning: No outputs collected for distillation loss.")
            return torch.tensor(0.0, requires_grad=True)
        
        if len(self.teacher_outputs) != len(self.student_outputs):
            print(f"Warning: Mismatched outputs - Teacher: {len(self.teacher_outputs)}, Student: {len(self.student_outputs)}")
            return torch.tensor(0.0, requires_grad=True)

        quant_loss = self.distill_loss_fn(y_s=self.student_outputs, y_t=self.teacher_outputs)

        self.teacher_outputs.clear()
        self.student_outputs.clear()
        
        return quant_loss

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()
        self.remove_handle.clear()