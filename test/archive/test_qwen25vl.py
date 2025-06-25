import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load the model in half-precision on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", device_map="auto", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model.eval()

# conversation = [
#     {
#         "role":"user",
#         "content":[
#             {
#                 "type":"image",
#                 "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
#             },
#             {
#                 "type":"text",
#                 "text":"Describe this image."
#             }
#         ]
#     }
# ]

# inputs = processor.apply_chat_template(
#     conversation,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt"
# ).to(model.device)


# # Inference: Generation of the output
# with torch.no_grad():
#     output_ids = model.generate(**inputs, max_new_tokens=128)
#     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
#     output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
# print(output_text)

# Video
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/root/autodl-tmp/data/SumMe/videos/Air_Force_One.mp4"},
            {"type": "text", "text": "What happened in the video?"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    num_frames=8,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Inference: Generation of the output
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)