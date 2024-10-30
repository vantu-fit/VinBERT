<!--  add image centering -->
<p align="center">
  <img src="assets/logo.png" alt="VinBERT" width="200" height="200">
</p>

# VinBERT

VinBERT is a combination of two powerful Vietnamese language models: **Vintern-1b-v2** and **PhoBERT**. With VinBERT, we create a language model optimized to better serve applications in the Vietnamese language, including tasks such as text classification, entity extraction, and more.

## Objectives

- **VinBERT** leverages the strengths of Vintern-1b-v2 and PhoBERT, providing high efficiency and accuracy for Vietnamese NLP applications.
- It supports distributed training on multiple GPUs and AWS Sagemaker infrastructure, optimizing time and resources.

## Support training
-  **cuda**: `Data parallelism` and `Model parallelism` are supported with backend `nccl` 
- **xla** : `Data parallelism` are supported with backend `xla`

### Requirements

- An AWS account with access to Sagemaker.
- An environment set up to interact with AWS CLI and Sagemaker.
- You have quota to use `ml.p4d.24xlarge` and `ml.trn1.32xlarge` instances.

```bash
    pip install -r requirements.txt
```

### Distributed Training on GPU (AWS Sagemaker `ml.p4d.24xlarge`)

1. **Prepare the environment**: pull docker image flash attn base from dockerhub: `vantufit/flash-attn-cuda`
```bash   
docker pull vantufit/flash-attn-cuda
```

2. **Run the job**:
   - Configure parameters such as instance type, number of GPUs, and batch size.
   - Run the following command to initiate the job:
   ```python
   export INSTANCE=ml.p4d.24xlarge
   python training.py
   ```

### Training with Trainium (`ml.trn1.32xlarge`)

1. **Run the job**:
   ```python
   export INSTANCE=ml.trn1.32xlarge
   python training.py
   ```

### TODO: Monitoring Training
<!-- check bok -->
- [ ] Implement Tensor parallelism with `neuronx_distributed` 

- [ ] Monitoring training process 


