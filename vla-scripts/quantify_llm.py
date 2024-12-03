from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path="saved_model/openvla-7b-model", 
    trust_remote_code=True
)
vla = AutoModelForVision2Seq.from_pretrained(
    pretrained_model_name_or_path="saved_model/openvla-7b-model", 
    # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)

print(vla.device)

# save processor & vla state
processor.save_pretrained(
    save_directory="saved_model/openvla_8_bit"
)
vla.save_pretrained(
    save_directory="saved_model/openvla_8_bit"
)