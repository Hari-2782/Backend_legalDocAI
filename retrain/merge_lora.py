# backend/retrain/merge_lora.py
import json, torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

CFG_PATH = "./retrain/config.json"
OUT_MERGED = "./retrain/lora_model_merged"

def main():
    cfg = json.load(open(CFG_PATH))
    base = cfg["base_model"]
    adapters = cfg["output_dir"]           # ./retrain/lora_model

    tok = AutoTokenizer.from_pretrained(base)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base)
    model = PeftModel.from_pretrained(base_model, adapters)
    model = model.merge_and_unload()       # merges LoRA into base weights

    model.save_pretrained(OUT_MERGED)
    tok.save_pretrained(OUT_MERGED)
    print("Merged model saved to", OUT_MERGED)

if __name__ == "__main__":
    main()
