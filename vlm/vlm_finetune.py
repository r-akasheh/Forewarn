from datasets import load_dataset


from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
import evaluate
import tqdm
from PIL import Image
import io
import time
import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from robocasa_dataset.vlm_data import process
from robocasa_dataset.ocrvqa import get_custom_dataset, get_data_collator
from robocasa_dataset.vlm_data import get_custom_dataset, get_data_collator
# from torch.nn.parallel import DistributedDataParallel as DDP


# def process(examples, processor):
#     texts = [f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>{example['question']} Answer briefly. <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['multiple_choice_answer']}<|eot_id|>" for example in examples]
#     images = [[example["image"].convert("RGB")] for example in examples]
#     batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
#     labels = batch["input_ids"].clone()
#     labels[labels == processor.tokenizer.pad_token_id] = -100 
#     labels[labels == 128256] = -100 # image token index
#     ## what is the image token index
    
#     batch["labels"] = labels
#     batch = batch.to(torch.bfloat16).to("cuda")

#     return batch

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    # ds = load_dataset("merve/vqav2-small", split="validation[:10%]")
    # Assume 'my_vqa_dataset.py' is in the 'my_vqa_dataset/' directory
    # dataset = load_dataset(
        # 'robocasa_dataset/vlm_data.py',
        # data_files={
            # 'train': 'train.hdf5',
            # 'validation': 'validation.h5',
            # 'test': 'eval.hdf5'
        # },
        # trust_remote_code=True
    # )
    ## clear cuda cache
    torch.cuda.empty_cache()
    llama_ckpt = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    processor = AutoProcessor.from_pretrained(llama_ckpt)
    train_dataset = get_custom_dataset(train=True)
    test_dataset = get_custom_dataset(train=False)  
    
    # dataset = get_custom_dataset("ocrvqa", processor, "train")
 
    # ckpt = "meta-llama/Llama-3.2-11B-Vision"
    ckpt = llama_ckpt
    USE_LORA = True
    FREEZE_LLM = False
    FREEZE_IMAGE = False

    if USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
            use_dora=True, # optional DoRA 
            init_lora_weights="gaussian"
        )

        model = MllamaForConditionalGeneration.from_pretrained(
                ckpt,
                torch_dtype=torch.bfloat16,
                device_map="auto"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elif FREEZE_IMAGE:
        if FREEZE_LLM:
            raise ValueError("You cannot freeze image encoder and text decoder at the same time.")
        model = MllamaForConditionalGeneration.from_pretrained(ckpt,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
            )
        # freeze vision model to save up on compute
        for param in model.vision_model.parameters():
            param.requires_grad = False

    elif FREEZE_LLM:
        if FREEZE_IMAGE:
            raise ValueError("You cannot freeze image encoder and text decoder at the same time.")
        model = MllamaForConditionalGeneration.from_pretrained(ckpt,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
            )
        # freeze text model, this is encouraged in paper
        for param in model.language_model.parameters():
            param.requires_grad = False
            
    else: # full ft
        model = MllamaForConditionalGeneration.from_pretrained(ckpt,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
            )
    # model.gradient_checkpointing_enable()
  
   
    args=TrainingArguments(
                num_train_epochs=2,
                remove_unused_columns=False,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                learning_rate=2e-5,
                weight_decay=1e-6,
                adam_beta2=0.999,
                logging_steps=1,#250,
                save_strategy="no",
                optim="adamw_hf",
                push_to_hub=True,
                save_total_limit=1,
                eval_steps=1,
                report_to=["wandb"],                # do_train = False,
                # do_eval = True,
                bf16=True,
                label_names=['answer'],
                # can_return_loss = True,
                logging_dir="./logs",
                output_dir="./lora",
                dataloader_pin_memory=False,
            )
    ## create a new function that reduce the argument of process to only one by using lambda, then pass it to the Trainer
    ## this is a workaround for the Trainer class that only accept function with only one argument
    # data_process = lambda x: process(x, processor)
    data_process = get_data_collator(processor)
    torch.cuda.empty_cache()
    trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset = test_dataset,
            data_collator=data_process,
            compute_metrics= compute_metrics,
            args=args
            )
    # for batch in trainer.get_eval_dataloader(train_dataset):
    #     breakpoint()
    # breakpoint()
    trainer.train()
    # Evaluate the fine-tuned model
    results = trainer.evaluate()

    # Print evaluation results
    print(results)
    
    ## load the model and evaluate it
    # trainer.load_model("lora")
    # results = trainer.evaluate()
    # trainer.save_model("lora")

if __name__ == "__main__":
    main()