import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

from peft import LoraConfig, LoraModel


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        tokenizer = None,

    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        print(self.gpu_id)
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.tokenizer = tokenizer
        

        #self.model.set_device(self.gpu_id)

        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
        self.writer = SummaryWriter()

        self.scheduler = CosineAnnealingLR(optimizer, T_max=len(train_data)*1, eta_min=1e-6)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location="cpu")
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.model = self.model.to(self.gpu_id)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        if "OPTIMIZER_STATE" in snapshot.keys():
            self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        del snapshot
        torch.cuda.empty_cache()
        dist.barrier()
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")


    def _run_batch(self, source, epoch, step, batch_size):
        self.optimizer.zero_grad()
               
        #inputs = self.tokenizer(source["text"], padding='longest', return_tensors="pt").to(self.gpu_id)

        output = self.model(input_ids = source["input_ids"], attention_mask = source["attention_mask"], bevs = source["bev"], view=source["view"], labels = source["input_ids"])
        #output = self.model(input_ids = source["input_ids"], bevs = source["bev"], labels = source["input_ids"])
        
        loss = output.loss
        loss.backward()
        
        self.optimizer.step()
        print(f'[GPU{self.gpu_id}] Epoch: {epoch}, step {step}/{len(self.train_data)} Loss: {loss:.4f}')

    def _run_epoch(self, epoch):
        b_sz = self.train_data.batch_size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        self.model.train()
        for step, source in enumerate(self.train_data):
            self._run_batch(source, epoch, step, b_sz)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER_STATE": self.optimizer.state_dict()
            }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        self.writer.flush()