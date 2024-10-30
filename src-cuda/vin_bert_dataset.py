import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer


class PromptTemplate():
    IMG_START_TOKEN='<img>'
    IMG_END_TOKEN='</img>'
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
    num_image_token = 256
    def __init__(self):
        self.prompt = f"""
        <|im_start|>system
        Bạn là một mô hình trí tuệ nhân tạo đa phương thức Tiếng Việt có khả năng phân tích ảnh và các bình luận về ảnh đó. Nhiệm vụ của bạn là phân tích chi tiết bối cảnh của tấm ảnh và bình luận từ mạng xã hội, sau đó kết luận xem nội dung thuộc vào một trong bốn trường hợp: 'châm biếm qua hình ảnh', 'châm biếm cả hình ảnh và văn bản', 'không châm biếm', hoặc 'châm biếm qua văn bản'.
        <|im_end|>

        <|im_start|>user
        <image>
        Bình luận: "<text>"
        Hãy phân tích và xác định nhãn cho nội dung trên (chỉ lựa chọn một trong bốn nhãn sau: 'châm biếm qua hình ảnh', 'châm biếm cả hình ảnh và văn bản', 'không châm biếm', 'châm biếm qua văn bản').
        <|im_end|>

        <|im_start|>assistant
        """
    def create(self, caption, num_img_segments):
        caption =  self.prompt.replace("<text>" , caption , 1)
        image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_img_segments + self.IMG_END_TOKEN
        caption = caption.replace("<image>", image_tokens, 1)
        return caption
    

class ImageProcessor:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    def __init__(self, input_size = 448):
        self.input_size = input_size
        self.transfroms = self.build_transform(input_size=self.input_size)
    
    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    
    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [self.transfroms(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
class UITDataset(Dataset):
    def __init__(self, train_image_dir, train_text_path, tokenizer, template, image_processor , device='cuda', sample=None):
        self.train_image_dir = train_image_dir
        self.train_text_path = train_text_path
        self.tokenizer = tokenizer
        self.template = template
        self.image_processor = image_processor
        self.device = device
        
        self.tokenizer.padding_side = 'left'
        
        self.image_names, self.captions, self.labels, self.classes = self.get_data()
        
        if sample:
            self.image_names, self.captions, self.labels, self.classes = self.get_sample(sample)
                    
    def get_sample(self, sample):
        count = {k : 0 for k in self.classes_to_idx.keys()}
        image_names = []
        captions = []
        labels = []
        classes = []
        for image_name, caption, label in zip(self.image_names, self.captions, self.labels):
            class_name = self.idx_to_classes[label]
            if count[class_name] < sample:
                count[class_name] += 1
                image_names.append(image_name)
                captions.append(caption)
                labels.append(label)
                classes.append(class_name)
            if sum([v for k , v in count.items()]) >= sample * 4 :
                break
            
        return image_names, captions, labels, classes
                
        
    def get_data(self):
        with open(self.train_text_path , 'r' , encoding = 'utf-8') as file:
            json_data = json.load(file)
            
        captions = []
        labels = []
        image_names = []
        for key, value in json_data.items():
            image = value.get("image")
            caption = value.get("caption")
            label = value.get("label")
            
            image_names.append(image)
            labels.append(label)
            captions.append(caption)
            
        classes = list(set(labels))
        classes = { k : v for v , k in enumerate(classes)}
        self.idx_to_classes = { v : k  for v , k in enumerate(classes)}
        self.classes_to_idx = { k : v  for v , k in enumerate(classes)}
        labels = [classes[label] for label in labels]
        
        if len(labels) != len(captions) or len(labels) != len(image_names):
            assert len(labels) == len(captions), "nums of labels not match captions"
            assert len(labels) == len(image_names), "nums of labels not match image_names"
            
            return None, None, None, None
        
        return image_names, captions, labels, classes
#     def plot(self):
#         classes, counts = np.unique(self.labels, return_counts=True)
#         class_names = [self.idx_to_classes[classe] for classe in classes]
#         for class_name, count in zip(class_names, counts):
#             print(f"{class_name} : {count}")
            
#         plt.bar(classes, counts, color=['blue', 'green', 'red', 'purple'])

#         plt.title('Number of labels')
#         plt.axis("off")
#         plt.xlabel('Classes')
#         plt.ylabel('Number of labels')
        
#         for i in range(len(classes)):
#             plt.text(classes[i], counts[i] + 0.1, str(counts[i]), ha='center')

#         plt.show()
    
    def create_inputs(self, batch):
        image_names = batch["image_name"]
        captions = batch["caption"]
        labels = batch["label"]
        
        # Dynamic processing image and create prompt
        pixel_values = []
        prompts = []
        image_flags = []
        for image_name, caption, label in zip(image_names, captions, labels):
            image_path = os.path.join(self.train_image_dir, image_name)
            pixel_value = self.image_processor.load_image(image_path)
            pixel_values.append(pixel_value)
                                    
            prompt = self.template.create(caption, pixel_value.shape[0])
            prompts.append(prompt)
 
        max_len_image = max([item.shape[0] for item in pixel_values])
        
        # Padding pixel for image
        pixel_values_padding = []
        for pixel_value in pixel_values:
            image_flag = [1] * pixel_value.shape[0]
            for i in range(max_len_image - pixel_value.shape[0]):
                image_flag.append(0)
                padding = torch.zeros(1, 3 , 448, 448)
                pixel_value = torch.cat((pixel_value, padding), dim = 0)
                
            image_flags.extend(image_flag)
            pixel_values_padding.append(pixel_value)     
        
        # Create encode inputs
        encoded = self.tokenizer(prompts, padding = True, return_tensors = 'pt')
        prompts_encoded = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
                
        return {
            "pixel_values" : torch.stack(pixel_values_padding).view(-1, 3, 448, 448).to(self.device),
            "input_ids" : prompts_encoded,
            "attention_mask" : attention_mask,
            "image_flags" : torch.tensor(image_flags).unsqueeze(-1).to(self.device),
            "labels" : labels.to(self.device)
        }
        
    def __len__(self):
        return  len(self.labels)
    
    def __getitem__(self, index):
        label = self.labels[index]
        caption = self.captions[index]
        image_name = self.image_names[index]
        
        return {
            "image_name" : image_name,
            "caption" : caption,
            "label" : label
        }
        
        
if __name__ == "__main__":  
    train_image_dir = "data/train-images"
    train_text_path = "data/vimmsd-train.json"

    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v2", trust_remote_code=True, use_fast=False)
    template = PromptTemplate()
    image_processor = ImageProcessor()
    dataset = UITDataset(train_image_dir, train_text_path, tokenizer, template, image_processor)

    train_size = int(0.7 * len(dataset)) 
    test_size = len(dataset) - train_size 

    val_size = int(0.5 * test_size) 
    test_size = test_size - val_size 

    train_dataset,  test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size + val_size])
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)  
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False) 
                    
    for batch in train_dataloader:
        inputs = dataset.create_inputs(batch)
        print(inputs)
        break