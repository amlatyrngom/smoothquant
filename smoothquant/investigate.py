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
import pickle as pkl

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
            n_samples: int = 40,
        ):
        if short_model_name.startswith("opt"):
            self.model_name = f"facebook/{short_model_name}"
        elif short_model_name.startswith("llama"):
            self.model_name = f"meta-llama/{short_model_name}-hf"
        else:
            raise ValueError("Unknown model name")
        scales_path = f"{repo_dir}/act_scales/{short_model_name}.pt"
        assert os.path.exists(scales_path), f"Cannot find the act scales at {scales_path}"
        self.act_scales = torch.load(scales_path)
        acc_tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        perp_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        acc_dataset = load_dataset("lambada", split=f"validation[:{n_samples}]")
        perp_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.acc_evaluator = AccuracyEvaluator(acc_dataset, acc_tokenizer, device)
        self.perp_evaluator = PerplexityEvaluator(perp_dataset, perp_tokenizer, device, n_samples=n_samples)
        self.n_bits = n_bits
        self.q_group_size = q_group_size
        self.q_protect = q_protect
        self.q_protection_ratio = q_protection_ratio
        self.q_protection_scale = q_protection_scale
        self.q_smoothing_strength = q_smoothing_strength


    def make_base_model(self):
        print("Making base model...")
        model_fp16 = OPTForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )
        print("Done making base model.")
        return model_fp16
    
    def evaluate_base_model(self, perp=True):
        model = self.make_base_model()
        if perp:
            res = self.perp_evaluator.evaluate(model)
        else:
            res = self.acc_evaluator.evaluate(model)
        del model
        return res
        
    
    def make_quantized_model(self):
        model_fp16 = self.make_base_model()
        print("Quantizing model...")
        q_model = quantize_opt(
            model_fp16,
            n_bits=self.n_bits,
            q_group_size=self.q_group_size,
            q_protect=self.q_protect,
            q_protection_ratio=self.q_protection_ratio,
            q_protection_scale=self.q_protection_scale,
        )
        print("Done quantizing model.")
        return q_model


    def evaluate_quantized_model(self, perp=True):
        model = self.make_quantized_model()
        if perp:
            res = self.perp_evaluator.evaluate(model)
        else:
            res = self.acc_evaluator.evaluate(model)
        del model
        return res
        

    def make_smooth_model(self):
        model = self.make_base_model()
        print("Smoothing model...")
        smooth_lm(model, self.act_scales, self.q_smoothing_strength)
        print("Done smoothing model.")
        print("Quantizing model...")
        q_model = quantize_opt(
            model,
            n_bits=self.n_bits,
            q_group_size=self.q_group_size,
            q_protect=self.q_protect,
            q_protection_ratio=self.q_protection_ratio,
            q_protection_scale=self.q_protection_scale,
        )
        print("Done quantizing model.")
        return q_model
    
    def evaluate_smooth_model(
        self,
        perp=True,
    ):
        model = self.make_smooth_model()
        if perp:
            res = self.perp_evaluator.evaluate(model)
        else:
            res = self.acc_evaluator.evaluate(model)
        del model
        return res
            



def make_setup(n_bits, q_group_size, q_protect, q_protection_scale, q_protection_ratio, q_smoothing_strength):
    return {
        "n_bits": n_bits,
        "q_group_size": q_group_size,
        "q_protect": q_protect,
        "q_protection_scale": q_protection_scale,
        "q_protection_ratio": q_protection_ratio,
        "q_smoothing_strength": q_smoothing_strength,       
    }

def setup_name(setup):
    n_bits = setup["n_bits"]
    base_name = f"W{n_bits}A{n_bits}"
    q_group_size = setup["q_group_size"]
    if q_group_size > 0:
        base_name += f" G{q_group_size}"
    q_protect = setup["q_protect"]
    if q_protect:
        q_protection_scale = setup["q_protection_scale"]
        if q_protection_scale > 0:
            base_name += f" AWQ-Scaled"
        else:
            base_name += f" AWQ-Mixed"
    return base_name

def make_setups():
    setups = []
    # 2^4 = 16 configurations.
    for n_bits in [4, 8]:
        for q_group_size in [0, 128]:
            for q_protect in [False, True]:
                for q_protection_scale in [0.0, 2.0]:
                    if q_group_size == 0 and q_protect:
                        # Protection only enabled for q_group_size > 0
                        continue
                    if q_protection_scale > 0 and not q_protect:
                        # Pointless duplication.
                        continue
                    q_protection_ratio = 0.03
                    q_smoothing_strength = 0.5
                    setups.append(make_setup(n_bits, q_group_size, q_protect, q_protection_scale, q_protection_ratio, q_smoothing_strength))
    return setups


def sweep(short_model_name, repo_dir, save_dir, perp=True):
    # Should take ~1h on colab with 6.7B and A100.
    os.makedirs(save_dir, exist_ok=True)
    result_file = f"{save_dir}/results_{short_model_name}.pkl"
    if os.path.exists(result_file):
        with open(result_file, "rb") as f:
            results = pkl.load(f)
    else:
        results = {}
    setups = make_setups()
    if "base" not in results:
        print("Running base model")
        investigation = Investigation(short_model_name, repo_dir, **setups[0])
        base_results = investigation.evaluate_base_model(perp=perp)
        results["base"] = base_results
        print(f"Base FP16: {base_results}")
    else:
        print("Base model already run")
    for setup in setups:
        setup_key = str(setup)
        base_expt_name = setup_name(setup)
        if setup_key in results:
            print(f"Setup {base_expt_name} already run")
            continue
        print(f"Running setup {base_expt_name}")
        investigation = Investigation(short_model_name, repo_dir, **setup)
        q_res = investigation.evaluate_quantized_model(perp=perp)
        q_smooth_res = investigation.evaluate_smooth_model(perp=perp)
        res = {
            "setup": setup,
            "q_res": q_res,
            "q_smooth_res": q_smooth_res,
        }
        simple_expt_name = f"{base_expt_name}"
        smooth_expt_name = f"Smooth {base_expt_name}"
        print(f"{simple_expt_name}: {res['q_res']}")
        print(f"{smooth_expt_name}: {res['q_smooth_res']}")
        results[setup_key] = res
        # Checkpointing
        with open(result_file, "wb") as f:
            pkl.dump(results, f)




def report_sweep(short_model_name, save_dir):
    result_file = f"{save_dir}/results_{short_model_name}.pkl"
    with open(result_file, "rb") as f:
        results = pkl.load(f)
    base_result = results["base"]
    print(f"Base FP16,{base_result}")
    setups = make_setups()
    for setup in setups:
        setup_key = str(setup)
        res = results[setup_key]
        setup = res["setup"]
        base_expt_name = setup_name(setup)
        simple_expt_name = f"{base_expt_name}"
        smooth_expt_name = f"Smooth {base_expt_name}"
        print(f"{simple_expt_name},{res['q_res']}")
        print(f"{smooth_expt_name},{res['q_smooth_res']}")