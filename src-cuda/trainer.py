import os
import torch
from transformers import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader

class Evaluation:
    def __init__(self, n_classes, dataset):
        self.n_classes = n_classes
        self.dataset = dataset
    
    def evaluation(self, model, dataloader, log=False):
        with torch.no_grad():
            model.eval()
            total_logits = []
            total_labels = []

            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                inputs = self.dataset.create_inputs(batch)
                outputs = model(**inputs)
                logits = outputs.logits
                total_logits.append(logits)
                total_labels.append(inputs["labels"])

            total_logits = torch.cat(total_logits, dim=0)
            total_labels = torch.cat(total_labels, dim=0)

            metrics, total_precision, total_recall, total_f1 = self.metrics_per_class(total_logits, total_labels, log)

            if log:
                print(f"{'Total':<10} {total_precision:<15.4f} {total_recall:<15.4f} {total_f1:<15.4f}")
                with open("log_eval.txt", "w") as file:
                    file.write(f"{'Total':<10} {total_precision:<15.4f} {total_recall:<15.4f} {total_f1:<15.4f}")
            else:
                print(f"{'F1':<10} {total_f1:<15.4f}")

        return total_precision, total_recall, total_f1
            
    def confusion_matrix(self, logits, labels, log=False):
        with torch.no_grad():
            predicts = torch.argmax(logits, dim=1)
            conf_matrix = torch.zeros(self.n_classes, self.n_classes)
            for t, p in zip(labels, predicts):
                conf_matrix[t.long(), p.long()] += 1
        if log:
            print(conf_matrix)
        return conf_matrix
    
    def metrics_per_class(self, logits, labels, log=False):
        conf_matrix = self.confusion_matrix(logits, labels, log)
        metrics = []
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for i in range(self.n_classes):
            tp = conf_matrix[i, i]
            fp = conf_matrix.sum(0)[i] - tp
            fn = conf_matrix.sum(1)[i] - tp
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            metrics.append({'precision': precision.item(), 'recall': recall.item(), 'f1': f1.item()})
            
            total_tp += tp.item()
            total_fp += fp.item()
            total_fn += fn.item()
        
        total_precision = total_tp / (total_tp + total_fp + 1e-10)
        total_recall = total_tp / (total_tp + total_fn + 1e-10)
        
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall + 1e-10)
        
        if log:
            print(f"{'Class':<10} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
            print("-" * 55)  
            for i, metric in enumerate(metrics):
                print(f"{i:<10} {metric['precision']:<15.4f} {metric['recall']:<15.4f} {metric['f1']:<15.4f}")

            print("-" * 55) 
            print(f"{'Total':<10} {total_precision:<15.4f} {total_recall:<15.4f} {total_f1:<15.4f}")
        
        return metrics, total_precision, total_recall, total_f1


class TrainingConfig:
    def __init__(self, config):
        self.epochs = config["epochs"]
        self.train_batch = config["train_batch"]
        self.train_size = config["train_size"]
        self.val_batch = config["val_batch"]
        self.test_batch = config["test_batch"]
        self.log = config["log"]
        self.lr = config["lr"]
        self.sample = config["sample"]


class Trainer:
    def __init__(self, config, model, dataset, evaluator):
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, weight_decay=0.01)
        self.epochs = config.epochs
        self.dataset = dataset
        self.evaluator = evaluator
        self.config = config
    
    def create_dataloader(self):
        train_size = int(self.config.train_size * len(self.dataset))
        test_size = len(self.dataset) - train_size

        val_size = int(0.5 * test_size)
        test_size -= val_size

        print(f"Train len: {train_size}")
        print(f"Test + Val: {test_size}")

        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size + val_size])
        val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])


        train_dataloader = DataLoader(train_dataset, batch_size=self.config.train_batch)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.val_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.test_batch)

        print(f"Train batch: {len(train_dataloader)}")
        print(f"Val batch: {len(val_dataloader)}")
        print(f"Test batch: {len(test_dataloader)}")
        
        return train_dataloader, val_dataloader, test_dataloader
    
    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs
    
    def save_model(self, model_dir):
        torch.save(self.model.state_dict(), model_dir)
        print(f"Model saved at {model_dir}")

    def train(self, train_dataloader=None, val_dataloader=None, test_dataloader=None, log=False):
        print("Start training...")
        epochs = self.epochs
        if train_dataloader is None and val_dataloader is None and test_dataloader is None:
            train_dataloader, val_dataloader, test_dataloader = self.create_dataloader()
        
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                inputs = self.dataset.create_inputs(batch)

                self.optimizer.zero_grad()

                outputs = self.forward(inputs)
                logits = outputs.logits
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Average training loss for epoch {epoch + 1}: {avg_train_loss:.4f}")

            # Đánh giá trên tập xác thực
            if val_dataloader is not None:
                metrics, total_precision, total_recall, total_f1 = self.evaluator.evaluation(self.model, val_dataloader, log)
