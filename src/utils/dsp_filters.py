# import os
# from multiprocessing import freeze_support
# from pathlib import Path
# import pandas as pd
# import torch
# import numpy as np
# import librosa # More robust than soundfile for various formats
# from dataclasses import dataclass
# from typing import Dict, List, Union
# from datasets import Dataset
# import evaluate
# from transformers import (
#     Wav2Vec2CTCTokenizer,
#     Wav2Vec2FeatureExtractor,
#     Wav2Vec2Processor,
#     Wav2Vec2ForCTC,
#     TrainingArguments,
#     Trainer,
#     EarlyStoppingCallback
# )
# # --- LORA IMPORTS ---
# from peft import LoraConfig, get_peft_model, TaskType

# # --- CONFIGURATION ---
# MODEL_NAME = "facebook/wav2vec2-large-xlsr-53" # The "Big Brain"
# BATCH_SIZE = 1           # Must be 1 for GTX 1650
# GRAD_ACCUMULATION = 16   # 1 * 16 = Effective Batch Size 16 (Stable gradients)
# EPOCHS = 15
# LEARNING_RATE = 3e-4     # LoRA needs a higher rate than standard fine-tuning
# SAMPLING_RATE = 16000

# # Paths
# script_dir = Path(__file__).resolve().parent.parent.parent
# INPUT_CSV = script_dir / "data" / "train_with_phonemes.csv"
# VOCAB_FILE = script_dir / "data" / "vocab.json"
# OUTPUT_DIR = script_dir / "data" / "quran_reciter_model_lora"

# # --- CUSTOM DATA COLLATOR (ON-THE-FLY LOADING) ---
# @dataclass
# class DataCollatorCTCWithPadding:
#     processor: Wav2Vec2Processor
#     sampling_rate: int = 16000

#     def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
#         input_values = []
#         label_features = []

#         for feature in features:
#             audio_path = feature["path"]
            
#             # 1. Robust Audio Loading
#             try:
#                 # Load with librosa (auto-converts to mono and resamples)
#                 speech, sr = librosa.load(audio_path, sr=self.sampling_rate)
#             except Exception as e:
#                 print(f"Skipping bad file {audio_path}: {e}")
#                 continue # Skip corrupt files

#             # 2. Extract Features
#             # Return tensors="pt" directly saves conversion time
#             processed = self.processor(speech, sampling_rate=self.sampling_rate, return_tensors="pt").input_values[0]
#             input_values.append({"input_values": processed})
            
#             # 3. Collect Labels
#             label_features.append({"input_ids": feature["labels"]})

#         # 4. Pad Audio Inputs
#         batch = self.processor.feature_extractor.pad(
#             input_values,
#             padding=True,
#             return_tensors="pt"
#         )
        
#         # 5. Pad Text Labels
#         labels_batch = self.processor.tokenizer.pad(
#             label_features,
#             padding=True,
#             return_tensors="pt"
#         )

#         # 6. Mask padding (-100 tells PyTorch to ignore this in loss calc)
#         labels = labels_batch["input_ids"].masked_fill(
#             labels_batch.attention_mask.ne(1), -100
#         )

#         batch["labels"] = labels
#         return batch

# def main():
#     # 1. SETUP DATA
#     print("1. Loading Data...")
#     df = pd.read_csv(INPUT_CSV)
    
#     # PRODUCT TIP: Shuffle data so model doesn't learn Surah order
#     df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
#     # PRODUCT TIP: Train/Test Split (90% Train, 10% Validation)
#     # We must know if the model generalizes to new audio
#     from sklearn.model_selection import train_test_split
#     train_df, test_df = train_test_split(df, test_size=0.1)
    
#     train_dataset = Dataset.from_pandas(train_df)
#     eval_dataset = Dataset.from_pandas(test_df)
    
#     print(f"Training Samples: {len(train_dataset)} | Validation Samples: {len(eval_dataset)}")

#     # 2. SETUP PROCESSOR
#     print("2. Initializing Processor...")
#     tokenizer = Wav2Vec2CTCTokenizer(
#         str(VOCAB_FILE), # Ensure string path
#         unk_token="[UNK]",
#         pad_token="[PAD]",
#         word_delimiter_token="_"
#     )

#     feature_extractor = Wav2Vec2FeatureExtractor(
#         feature_size=1,
#         sampling_rate=SAMPLING_RATE,
#         padding_value=0.0,
#         do_normalize=True,
#         return_attention_mask=False
#     )

#     processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

#     # 3. TOKENIZE TEXT ONLY (Audio is loaded later)
#     def tokenize_labels(batch):
#         # We process the text now to save CPU time during training
#         batch["labels"] = processor(text=batch["phonemes"]).input_ids
#         return batch

#     train_dataset = train_dataset.map(tokenize_labels, remove_columns=["text", "phonemes", "duration", "surah", "ayah", "__index_level_0__"])
#     eval_dataset = eval_dataset.map(tokenize_labels, remove_columns=["text", "phonemes", "duration", "surah", "ayah", "__index_level_0__"])

#     # 4. LOAD MODEL
#     print(f"4. Loading Large Model ({MODEL_NAME})...")
#     model = Wav2Vec2ForCTC.from_pretrained(
#         MODEL_NAME,
#         ctc_loss_reduction="mean",
#         pad_token_id=processor.tokenizer.pad_token_id,
#         vocab_size=len(processor.tokenizer),
#         # Memory Optimizations for loading
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     )

#     # 5. APPLY LORA (The Magic Step)
#     print("Applying LoRA Configuration...")
#     model.freeze_feature_extractor() # Always freeze the CNN layers
#     model.gradient_checkpointing_enable() # Save VRAM

#     peft_config = LoraConfig(
#         task_type=TaskType.SEQ_2_SEQ_LM, # or TaskType.CTC (Check version of PEFT, usually pure config is enough)
#         inference_mode=False,
#         r=32,               # Rank: Higher = smarter but more VRAM. 32 is standard.
#         lora_alpha=64,      # Alpha: Usually 2x Rank
#         lora_dropout=0.1,
#         target_modules=["q_proj", "v_proj"] # Target Attention layers
#     )
    
#     # Wrap the model
#     model = get_peft_model(model, peft_config)
#     model.print_trainable_parameters() 
#     # Expect: ~0.5% trainable parameters. This enables 4GB training.

#     # 6. TRAINING CONFIG
#     wer_metric = evaluate.load("wer")

#     def compute_metrics(pred):
#         pred_logits = pred.predictions
#         pred_ids = np.argmax(pred_logits, axis=-1)
#         pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
#         pred_str = processor.batch_decode(pred_ids)
#         label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        
#         wer = wer_metric.compute(predictions=pred_str, references=label_str)
#         return {"wer": wer}

#     training_args = TrainingArguments(
#         output_dir=str(OUTPUT_DIR),
#         group_by_length=True,
#         per_device_train_batch_size=BATCH_SIZE, # 1
#         gradient_accumulation_steps=GRAD_ACCUMULATION, # 16
#         num_train_epochs=EPOCHS,
        
#         # GTX 1650 OPTIMIZATIONS
#         fp16=True, 
#         gradient_checkpointing=True,
        
#         evaluation_strategy="steps",
#         save_steps=500,
#         eval_steps=500,
#         logging_steps=50,
#         learning_rate=LEARNING_RATE,
#         warmup_steps=500,
#         save_total_limit=2,
#         load_best_model_at_end=True,
#         metric_for_best_model="wer",
#         greater_is_better=False,
#         dataloader_num_workers=0, # Windows fix
#         remove_unused_columns=False,
#         label_names=["labels"] # Fixes some LoRA bugs
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         data_collator=DataCollatorCTCWithPadding(processor=processor),
#         compute_metrics=compute_metrics,
#         tokenizer=processor.feature_extractor, # Pass extractor here
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
#     )

#     # 7. START
#     print("5. Starting LoRA Training... Bismillah 🤲")
#     trainer.train()

#     # 8. SAVE
#     print("6. Saving LoRA Adapters...")
#     trainer.save_model(str(OUTPUT_DIR))
#     processor.save_pretrained(str(OUTPUT_DIR))
#     print("✅ Done. You now have a Product-Grade LoRA Adapter.")

# if __name__ == "__main__":
#     freeze_support()
#     main()