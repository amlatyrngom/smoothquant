import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

from smoothquant.calibration import get_act_scales
# AWQ
from huggingface_hub import hf_hub_download
from awq.quantize.pre_quant import apply_awq


def get_short_model_name(model_name):
    short_names = ["opt-125m", "opt-1.3b", "opt-6.7b", "opt-13b", "llama-2-7b"]
    for short_name in short_names:
        if short_name in model_name:
            return short_name
    raise NotImplementedError(f"Model name {model_name} is not supported for AWQ. Add it to the list of supported models!!")


def build_model_and_tokenizer(model_name, model_path, local_files_only=False, awq=False):
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    if local_files_only:
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    original_device = model.device
    if awq:
        if local_files_only:
            awq_cache_rel_dir = "../llm-awq/awq_cache"
            short_model_name = get_short_model_name(model_name)
            awq_pt_name = f"{awq_cache_rel_dir}/{short_model_name}-w4-g128.pt"
            awq_pt = torch.load(awq_pt_name, map_location="cpu")
        else:
            awq_zoo = "mit-han-lab/awq-model-zoo"
            short_model_name = get_short_model_name(model_name)
            awq_pt_name = f"{short_model_name}-w4-g128.pt"
            awq_pt_filename = hf_hub_download(repo_id=awq_zoo, filename=awq_pt_name, repo_type="dataset")
            awq_pt = torch.load(awq_pt_filename, map_location="cpu")
        print(f"Applying AWQ to model {model_name} using {awq_pt_name}")
        apply_awq(model, awq_pt)
        print("AWQ applied successfully")
    model.to(original_device)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="facebook/opt-1.3b", help="model name"
    )
    parser.add_argument(
        "--model-path", type=str, default="facebook/opt-1.3b", help="model path"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="act_scales/opt-1.3b.pt",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/val.jsonl.zst",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument(
        "--awq",
        action="store_true",
        default=False,
        help="whether to use the AWQ quantization method",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=False,
        help="whether only local files are used",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name, args.model_path, args.local_files_only, args.awq)

    if not os.path.exists(args.dataset_path):
        print(f"Cannot find the dataset at {args.dataset_path}")
        print("Please download the Pile dataset and put the validation set at the path")
        print(
            "You can download the validation dataset of the Pile at https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
        )
        raise FileNotFoundError

    print(args.dataset_path)
    act_scales = get_act_scales(
        model, tokenizer, args.dataset_path, args.num_samples, args.seq_len
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == "__main__":
    main()
