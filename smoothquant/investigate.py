import torch
import tqdm
import os
from torch import nn
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
)
from transformers import GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import WQAQLinear, quantize_opt
from datasets import load_dataset

class PerplexityEvaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        testenc = self.dataset
        nsamples = self.n_samples
        model = model.eval()

        nlls = []
        for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
            batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))

class AccuracyEvaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc
    


class Investigation:
    def __init__(
            self,
            short_model_name: str = "opt-125m",
            repo_dir: str = ".",
            n_bits: int = 4, 
            q_group_size: int = 128,
            q_protect: bool = True,
            q_protection_ratio: float = 0.02,
            q_protection_scale: float = 2.0,
            q_smoothing_strength: float = 0.5,
        ):
        if short_model_name.startswith("opt"):
            self.model_name = f"facebook/{short_model_name}"
        elif short_model_name.startswith("llama"):
            self.model_name = f"meta-llama/{short_model_name}-hf"
        else:
            raise ValueError("Unknown model name")
        scales_path = f"{repo_dir}/act_scales/{short_model_name}.pt"
        assert os.path.exists(scales_path), f"Cannot find the act scales at {self.scales_path}"
        self.act_scales = torch.load(scales_path)
        acc_tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        perp_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        acc_dataset = load_dataset("lambada", split="validation[:40]")
        perp_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.acc_evaluator = AccuracyEvaluator(acc_dataset, acc_tokenizer, device)
        self.perp_evaluator = PerplexityEvaluator(perp_dataset, perp_tokenizer, device, n_samples=15)
        self.n_bits = n_bits
        self.q_group_size = q_group_size
        self.q_protect = q_protect
        self.q_protection_ratio = q_protection_ratio
        self.q_protection_scale = q_protection_scale
        self.q_smoothing_strength = q_smoothing_strength


    def make_base_model(self):
        model_fp16 = OPTForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )
        return model_fp16
    
    def evaluate_base_model(self, perp=True):
        model = self.make_base_model()
        if perp:
            return self.perp_evaluator.evaluate(model)
        else:
            return self.acc_evaluator.evaluate(model)
        
    
    def make_quantized_model(self):
        model_fp16 = self.make_base_model()
        q_model = quantize_opt(
            model_fp16,
            n_bits=self.n_bits,
            q_group_size=self.q_group_size,
            q_protect=self.q_protect,
            q_protection_ratio=self.q_protection_ratio,
            q_protection_scale=self.q_protection_scale,
        )
        return q_model


    def evaluate_quantized_model(self, perp=True):
        model = self.make_quantized_model()
        if perp:
            return self.perp_evaluator.evaluate(model)
        else:
            return self.acc_evaluator.evaluate(model)
        

    def make_smooth_model(self):
        model = self.make_base_model()
        smooth_lm(model, self.act_scales, self.q_smoothing_strength)
        q_model = quantize_opt(
            model,
            n_bits=self.n_bits,
            q_group_size=self.q_group_size,
            q_protect=self.q_protect,
            q_protection_ratio=self.q_protection_ratio,
            q_protection_scale=self.q_protection_scale,
        )
        return q_model
    
    def evaluate_smooth_model(
        self,
        perp=True,
    ):
        model = self.make_smooth_model()
        if perp:
            return self.perp_evaluator.evaluate(model)
        else:
            return self.acc_evaluator.evaluate(model)
            

    


    