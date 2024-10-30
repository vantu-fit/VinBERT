def freeze_model_org(vin_bert):
    for param in vin_bert.parameters():
        param.requires_grad = False

    total_model_params_org = sum(p.numel() for p in vin_bert.parameters())
    
    print(f'Total parameter in vin_bert: {total_model_params_org//1000000} M')

    total_trainable_params_org = sum(p.numel() for p in vin_bert.parameters() if p.requires_grad)

    print(f'Total trainable parameter in vin_bert: {total_trainable_params_org/1000000} M')



def unfreeze_model_lora(vin_bert_with_lora):  
    for param in vin_bert_with_lora.mlp1.parameters():  
        param.requires_grad = True

    for param in vin_bert_with_lora.language_model.lm_head.parameters():  
        param.requires_grad = True

    for param in vin_bert_with_lora.phobert.pooler.parameters():  
        param.requires_grad = True
    
    for param in vin_bert_with_lora.phobert.mlp.parameters():  
        param.requires_grad = True

    for param in vin_bert_with_lora.phobert.classifier.parameters():  
        param.requires_grad = True
             
    lora_params = []

    for name, param in vin_bert_with_lora.named_parameters():
        if "lora" in name:  
            lora_params.append(param)

    total_lora_params = sum(p.numel() for p in lora_params)

    print(f'Total trainable of LoRA: {total_lora_params/1000000}')
    
    total_trainable_params_org = sum(p.numel() for p in vin_bert_with_lora.parameters() if p.requires_grad)
    
    print(f'Total trainable parameter in vin_bert_with_lora: {total_trainable_params_org/1000000} M')