import numpy as np
import torch
import datetime
import torch.nn.functional as F
from torch import nn
from transformers import AutoImageProcessor, TimesformerModel, TimesformerConfig
from torch import einsum
from einops import rearrange, reduce, repeat
from torch.utils.data._utils.collate import default_collate
import timesformer.models.optimizer as optim
from timesformer.datasets.rtmri75s import Rtmri75s
from timesformer.utils.parser import load_config, parse_args
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

class CLIP(nn.Module):
    def __init__(self,
                 device = 'cuda:0'):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = device
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, video):
        cfg = TimesformerConfig(hidden_size=1024, num_attention_heads = 16)
        video_model = TimesformerModel(cfg).to(self.device)
        # video_model = TimesformerModel.from_pretrained("facebook/timesformer-hr-finetuned-k600")
        return video_model(video)

    def forward(self, video, audio_features):
        video_features = torch.mean(self.encode_image(video).last_hidden_state, dim=1)
        # print(video_features.shape)
        # audio_features = self.encode_text(text)

        # normalized features
        # print(video_features.shape)
        video_features = torch.nn.functional.normalize(video_features, p=2, dim=1)#video_features / video_features.norm(dim=1, keepdim=True) 
        audio_features = torch.nn.functional.normalize(audio_features, p=2, dim=1)#audio_features / audio_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * video_features @ audio_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(self, logits_per_image, logits_per_text, output_dict=False):
        device = logits_per_image.device
        # logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
def train_epoch(
    train_loader, model, optimizer, cur_epoch, cfg
):
    total_loss = 0
    loss_fun = ClipLoss()
    cur_iter = 0
    for videos, audios, _, _ in tqdm(train_loader):
        data_size = len(train_loader)
        # print(data_size)
        # model = model.train()
        # model = model.cuda()

        videos = rearrange(videos[1], 'b c t h w -> b t c h w').cuda(non_blocking=True)
        audios_embs = audios.cuda(non_blocking=True)

        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)

        logits_per_image, logits_per_text = model(videos, audios_embs)

        loss = loss_fun(logits_per_image, logits_per_text)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        cur_iter += 1
        
    return total_loss

def train(cfg):
    batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
    # Construct the dataset
    dataset = Rtmri75s(cfg, "train")

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        collate_fn=default_collate,
    )

    model = CLIP()
    # loss_fun = ClipLoss()
    device = 'cuda:0'
    optimizer = optim.construct_optimizer(model, cfg)
    if(cfg.TIMESFORMER.PRETRAINED_MODEL != ''):
        model = torch.load(cfg.TIMESFORMER.PRETRAINED_MODEL)
        print("Load model from pretrain...")
    model.train()
    model = model.to(device)

    min_loss = 100000

    for e in range(1000):
        total_loss = train_epoch(train_loader, model, optimizer, cur_epoch = e, cfg=cfg)
        print(f"EPOCH {e} with total loss {total_loss}")
        if(total_loss < min_loss):
            min_loss = total_loss
            today = datetime.datetime.today().strftime('%Y-%m-%d')
            torch.save(model, f'lowest_lost_{today}.pth')

def main():
    """
    Main function to spawn the train and test process.
    """
    cfg = load_config(cfg_file = "/data2/hongn/TimeSformer/configs/Rtmri75s/simple_cfg.yaml")
    train(cfg)


if __name__ == "__main__":
    main()