from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import torch
import torch.amp as amp
from transformers import AutoTokenizer
from train.arguments import Arguments


from train.train import TrainPipeline, compute_metrics

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=False, rank0_only=False
    ),
)



class FSdpPipeline(TrainPipeline):
    def __init__(self,  hulu_args: Arguments, current_task: str, tokenizer_name: str):
        super().__init__(hulu_args, current_task, tokenizer_name)
        precision = "no" if hulu_args.precision == "fp32" else hulu_args.precision
        self.accelerator  = Accelerator(
            mixed_precision=precision,
            cpu=False,
            fsdp_plugin=fsdp_plugin,
        )

    def training(self):
        with self.accelerator.main_process_first():
            model = self.load_model()

            use_fp16 = self.hulu_args.precision == "fp16"
            scaler = amp.GradScaler() if use_fp16 else None
            device_type = "cuda" if torch.cuda.is_available() else "cpu"

            optimizer = AdamW(model.parameters(), lr=self.hulu_args.train_lr)
            total_steps = len(self.train_loader) * self.hulu_args.train_batch
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hulu_args.train_warmup,
                num_training_steps=total_steps,
            )
            gradient_accumulation_steps = self.hulu_args.gradient_accumulation_steps

            num_eval_steps = len(self.train_loader) // 3
            step = 0

            for epoch in range(self.hulu_args.train_epochs):
                model.train()
                total_loss, correct_preds = 0, 0

                for batch in tqdm(
                    self.train_loader,
                    desc=f"Training Epoch {epoch + 1}/{self.hulu_args.train_epochs}",
                ):
                    #batch.to(self.accelerator.device)
                    input_ids, attention_mask, labels = (
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["label"],
                    )
                    input_ids.to(self.accelerator.device)
                    attention_mask.to(self.accelerator.device)
                    labels.to(self.accelerator.device)
                    with amp.autocast(device_type=device_type, enabled=use_fp16):
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )

                    
                    if step % gradient_accumulation_steps == 0:
                        if use_fp16:
                            scaler.scale(output.loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            self.accelerator.backward(output.loss)
                            optimizer.step()
                            optimizer.zero_grad()


                    scheduler.step()
                    total_loss += output.loss.item()
                    correct_preds += (output.logits.argmax(dim=1) == labels).sum().item()
                    
                    
                    step += 1

                    if step % num_eval_steps == 0:
                        eval_loss, metrics = self.evaluate(model)
                        self.accelerator.log(
                            f"Step {step}: Eval Loss = {eval_loss:.4f}, Eval Acc = {metrics['accuracy']}, Eval MCC = {metrics['mcc']}, Eval F1 = {metrics['f1']}"
                        )

                avg_loss = total_loss / len(self.train_loader)
                accuracy = correct_preds / len(self.train_loader.dataset)
                self.accelerator.log(
                    f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, Train Accuracy = {accuracy:.4f}"
                )

            self.accelerator.end_training()
            return model
        
    def evaluate(self, model):
        total_loss, correct_preds = 0, 0
        all_preds, all_labels = [], []
        model.eval()
        total_loss, correct_preds = 0, 0
        for step, batch in enumerate(self.dev_loader):
            batch.to(self.accelerator.device)
            with torch.no_grad():
                input_ids, attention_mask, labels = (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["label"],
                )
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                predictions = outputs.logits.argmax(dim=-1)
            
                total_loss += outputs.loss
                preds = outputs.logits.argmax(dim=1)
                correct_preds += (preds == labels).sum().item()

                all_preds.append(preds)
                all_labels.append(labels)

        avg_loss = total_loss / len(self.dev_loader)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_preds, all_labels)
        return avg_loss, metrics
         