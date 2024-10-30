import os
import torch
from transformers import AutoConfig, AutoTokenizer
from vin_bert_modeling import BERTInternVLChatModel
from utils import freeze_model_org, unfreeze_model_lora
from vin_bert_dataset import UITDataset, PromptTemplate, ImageProcessor
from trainer import Evaluation, TrainingConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AdamW
import psutil

def calculate_f1(preds, labels, num_classes):
    f1_scores = []
    for i in range(num_classes):
        tp = ((preds == i) & (labels == i)).sum().item()
        fp = ((preds == i) & (labels != i)).sum().item()
        fn = ((preds != i) & (labels == i)).sum().item()
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_scores.append(f1)

    return sum(f1_scores) / num_classes if num_classes > 0 else 0

def main(args):
    train_image_dir = os.path.join(args.train_dir, "train-images")
    train_text_path = os.path.join(args.train_dir, "vimmsd-train.json")

    config = TrainingConfig({
        "epochs": args.epochs,
        "train_batch": args.train_batch,
        "train_size": args.train_size,
        "val_batch": args.val_batch,
        "test_batch": args.test_batch,
        "log": True,
        "lr": args.learning_rate,
        "sample": args.sample
    })

    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v2", trust_remote_code=True, use_fast=False)
    template = PromptTemplate()
    image_processor = ImageProcessor()
    dataset = UITDataset(train_image_dir, train_text_path, tokenizer, template, image_processor, sample=config.sample)

    config_model = AutoConfig.from_pretrained("5CD-AI/Vintern-1B-v2", trust_remote_code=True)
    
    vin_bert = BERTInternVLChatModel(
        config=config_model,
        model_mlp=None,
        vision_model=None,
        language_model=None
    )

    vin_bert.img_context_token_id = 151648

    freeze_model_org(vin_bert)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "patch_embedding",
            "language_model.embed_tokens",
            "self.query",
            "self.key",
            "self.value",
            "output.dense"
        ]
    )

    vin_bert_with_lora = get_peft_model(vin_bert, lora_config)
    unfreeze_model_lora(vin_bert_with_lora)

    model_path = os.path.join(args.model_dir, "vin_bert_model.pth")
    if os.path.exists(model_path):
        print("Start Loading Model from:", model_path)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        vin_bert_with_lora.load_state_dict(state_dict)
        print("Model Loaded Successfully")
    else:
        print(f"Model path {model_path} does not exist, skipping loading step.")

    evaluator = Evaluation(vin_bert_with_lora, dataset)

    train_size = int(args.train_size * len(dataset)) 
    test_size = len(dataset) - train_size 

    val_size = int(0.5 * test_size) 
    test_size = test_size - val_size 

    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size + val_size])
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch, shuffle=True)  
    val_dataloader = DataLoader(val_dataset, batch_size=config.val_batch, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.test_batch, shuffle=False) 

    optimizer = AdamW(vin_bert_with_lora.parameters(), lr=config.lr, weight_decay=0.01)

    print("Start Training ...")
    for epoch in range(config.epochs):
        vin_bert_with_lora.train()
        total_train_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            inputs = dataset.create_inputs(batch)

            optimizer.zero_grad()

            outputs = vin_bert_with_lora(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else None
            loss = outputs.loss if hasattr(outputs, 'loss') else None

            if loss is not None:
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            # Ghi log loss và RAM sử dụng mỗi 10 batch
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_train_loss / (batch_idx + 1)
                ram_usage = psutil.virtual_memory().used / (1024 ** 2)  # Convert to MB
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Average Loss = {avg_loss:.4f}, RAM Usage = {ram_usage:.2f} MB")

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")

        # Evaluation sau mỗi epoch
        vin_bert_with_lora.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = dataset.create_inputs(batch)
                outputs = vin_bert_with_lora(inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                labels = batch["labels"]

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Tính F1 Score
        f1_score = calculate_f1(torch.tensor(all_preds), torch.tensor(all_labels), num_classes=4)
        print(f"F1 Score after epoch {epoch + 1}: {f1_score:.4f}")

    # Lưu model sau khi training
    os.makedirs(args.model_dir, exist_ok=True)
    save_path = os.path.join(args.model_dir, "vin_bert_with_lora_model.pth")
    print("Saving the model ...")
    torch.save(vin_bert_with_lora.state_dict(), save_path)
    print("Model saved successfully to", save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'data/train'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DIR', 'output'))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train_batch', type=int, default=2)
    parser.add_argument('--train_size', type=float, default=0.9)
    parser.add_argument('--val_batch', type=int, default=2)
    parser.add_argument('--test_batch', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--sample', type=int, default=100)

    args = parser.parse_args()
    main(args)
