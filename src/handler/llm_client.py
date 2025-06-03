# filepath: /root/tfnet/src/handler/llm_client.py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import List, Dict, Any

class Qwen25VLClient:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", # 模型名称
        device_map: str = "auto", # 设备映射，例如 "auto", "cuda:0"
        torch_dtype: Any = None, # torch 数据类型，例如 torch.bfloat16
        attn_implementation: str | None = None, # attention 实现，例如 "flash_attention_2"
        min_pixels: int | None = None, # 图像处理的最小像素数
        max_pixels: int | None = None, # 图像处理的最大像素数
    ):
        """
        初始化 Qwen2.5-VL 客户端。

        Args:
            model_name (str): 要加载的预训练模型的名称。
            device_map (str): 加载模型的设备映射 (例如 "auto", "cuda:0")。
            torch_dtype (Any, optional): 模型加载的 torch dtype (例如 torch.bfloat16)。默认为 None。
            attn_implementation (str, optional): 要使用的 attention 实现 (例如 "flash_attention_2")。默认为 None。
            min_pixels (int, optional): 图像处理的最小像素数。默认为 None。
            max_pixels (int, optional): 图像处理的最大像素数。默认为 None。
        """
        model_kwargs = {"device_map": device_map} # 模型参数
        if torch_dtype:
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        # 从预训练模型加载模型
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        
        processor_kwargs = {} # 处理器参数
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        # 从预训练模型加载处理器
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    def generate_response(
        self,
        conversation: List[Dict[str, Any]], # 对话历史
        max_new_tokens: int = 128, # 生成新 token 的最大数量
        num_frames: int | None = None, # 从视频中采样的帧数
        video_fps: int = 1, # 视频处理的帧率 (如果未提供 num_frames，则使用)
        add_vision_id: bool = False, # 是否在提示中为视觉元素添加 ID
    ) -> List[str]:
        """
        为单个媒体 (图像/视频) 或文本对话生成响应。

        Args:
            conversation (List[Dict[str, Any]]): 对话历史。
                图像示例:
                [
                    {
                        "role":"user",
                        "content":[
                            {"type":"image", "url": "https://example.com/image.jpg"},
                            {"type":"text", "text":"描述这张图片。"}
                        ]
                    }
                ]
                视频示例:
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": "/path/to/video.mp4"},
                            {"type": "text", "text": "视频里发生了什么?"},
                        ],
                    }
                ]
            max_new_tokens (int): 要生成的最大新 token 数量。
            num_frames (int, optional): 从视频中采样的帧数。如果提供，则优先于 video_fps。
            video_fps (int): 用于视频处理的每秒帧数 (仅当未指定 num_frames 时相关)。
            add_vision_id (bool): 是否在提示中为视觉元素添加 ID。

        Returns:
            List[str]: 包含生成的文本响应的列表。
        """
        # 应用聊天模板处理输入
        template_args = {
            "conversation": conversation,
            "add_generation_prompt": True,  # 添加生成提示
            "tokenize": True,  # 进行分词
            "return_dict": True,  # 返回字典格式
            "return_tensors": "pt",  # 返回 PyTorch 张量
            "add_vision_id": add_vision_id,
        }
        if num_frames is not None:
            template_args["num_frames"] = num_frames
        else:
            template_args["video_fps"] = video_fps
            
        inputs = self.processor.apply_chat_template(**template_args).to(self.model.device) # 将输入移动到模型所在的设备

        # 生成输出 ID
        with torch.no_grad(): # 取消梯度计算
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # 对于单个对话，inputs.input_ids 的形状为 [1, seq_len]
        # output_ids 的形状为 [1, seq_len + new_tokens]
        # 我们需要正确处理批处理维度以进行切片。
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        # 批量解码生成的 ID
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text

    def generate_batch_response(
        self,
        conversations: List[List[Dict[str, Any]]], # 对话列表，每个对话是一个字典列表
        max_new_tokens: int = 128, # 为每个对话生成的最大新 token 数量
        video_fps: int = 1, # 视频处理的帧率
        add_vision_id: bool = False, # 是否在提示中为视觉元素添加 ID
    ) -> List[str]:
        """
        为一批混合媒体对话生成响应。

        Args:
            conversations (List[List[Dict[str, Any]]]): 对话列表。
                                                      每个对话都是一个字典列表。
            max_new_tokens (int): 为每个对话生成的最大新 token 数量。
            video_fps (int): 用于视频处理的每秒帧数。
            add_vision_id (bool): 是否在提示中为视觉元素添加 ID。

        Returns:
            List[str]: 生成的文本响应列表，每个输入对话对应一个响应。
        """
        # 应用聊天模板处理批量输入
        inputs = self.processor.apply_chat_template(
            conversations,
            video_fps=video_fps,
            add_generation_prompt=True, # 添加生成提示
            tokenize=True, # 进行分词
            return_dict=True, # 返回字典格式
            return_tensors="pt", # 返回 PyTorch 张量
            padding=True, # 为批处理启用填充
            add_vision_id=add_vision_id,
        ).to(self.model.device) # 将输入移动到模型所在的设备

        # 生成输出 ID
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # 提取生成的 token ID
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        
        # 批量解码生成的 ID
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text

if __name__ == '__main__':
    # 这是如何使用客户端的示例。
    # 确保您拥有兼容的环境并已下载模型权重。
    # 如果使用门控模型，您可能需要登录 Hugging Face。
    # from huggingface_hub import login
    # login("YOUR_HF_TOKEN")


    # 示例 1: 单图像推理
    print("--- 示例 1: 单图像推理 ---")
    client = Qwen25VLClient() # 使用默认的 Qwen/Qwen2.5-VL-7B-Instruct
                              # 对于 Flash Attention 2 (如果硬件支持并且已安装 flash-attn):
                              # client = Qwen25VLClient(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    
    image_conversation = [
        {
            "role":"user",
            "content":[
                # 使用公共图像 URL 进行测试
                {"type":"image", "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type":"text", "text":"描述这张图片。"}
            ]
        }
    ]
    response = client.generate_response(image_conversation)
    print(f"图像响应: {response}")

    # 示例 2: 单视频推理 (需要视频文件)
    # print("\n--- 示例 2: 单视频推理 ---")
    # try:
    #     # 如果没有测试视频文件，创建一个虚拟视频文件
    #     # import os
    #     # if not os.path.exists("dummy_video.mp4"):
    #     #     with open("dummy_video.mp4", "w") as f:
    #     #         f.write("dummy video content") # 这不是一个真实的视频
    #
    #     client = Qwen25VLClient()
    #     video_conversation = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "video", "path": "dummy_video.mp4"}, # 替换为您的视频路径
    #                 {"type": "text", "text": "视频里发生了什么?"},
    #             ],
    #         }
    #     ]
    #     # response = client.generate_response(video_conversation, video_fps=1)
    #     # print(f"视频响应: {response}")
    #     print("视频示例已注释掉。请提供有效的视频路径进行测试。")
    # except Exception as e:
    #     print(f"示例 2 (视频) 错误: {e}")


    # 示例 3: 批量混合媒体推理
    print("\n--- 示例 3: 批量混合媒体推理 ---")
    client = Qwen25VLClient()
    conversation1 = [ # 图像
        {"role": "user", "content": [
            {"type": "image", "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {"type": "text", "text": "描述这张图片。"}
        ]}
    ]
    conversation2 = [ # 纯文本
        {"role": "user", "content": [{"type": "text", "text": "你是谁?"}]}
    ]
    # conversation3 = [ # 视频 (需要视频文件)
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "video", "path": "dummy_video.mp4"}, # 替换为您的视频路径
    #             {"type": "text", "text": "总结这个视频。"},
    #         ],
    #     }
    # ]

    # batch_conversations = [conversation1, conversation2, conversation3]
    batch_conversations = [conversation1, conversation2] # 暂时排除视频
    
    # responses = client.generate_batch_response(batch_conversations)
    # for i, r in enumerate(responses):
    #     print(f"批量响应 {i+1}: {r}")
    print("批量示例已注释掉，因为它可能占用大量资源或需要特定设置。")
    print("根据需要取消注释并调整路径/URL。")

    # 示例 4: 使用 add_vision_id
    print("\n--- 示例 4: 使用 add_vision_id ---")
    client = Qwen25VLClient()
    multi_image_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}, # 另一张图片
                {"type": "text", "text": "这些图片里有什么? 如果可能的话，请引用它们。"}
            ]
        }
    ]
    response_with_ids = client.generate_response(multi_image_conversation, add_vision_id=True)
    print(f"带视觉 ID 的响应: {response_with_ids}")

    # 示例 5: 调整分辨率 (概念性的，实际效果取决于模型和内容)
    # print("\n--- 示例 5: 调整分辨率 ---")
    # try:
    #     # 较低分辨率以可能加快处理速度/减少内存占用
    #     # 这些值是说明性的；最佳值取决于 Qwen2.5-VL 的 ViT patch 大小 (14x2=28)
    #     min_p = 256 * 28 * 28 
    #     max_p = 1024 * 28 * 28
    #     # client_low_res = Qwen25VLClient(min_pixels=min_p, max_pixels=max_p)
    #     # response_low_res = client_low_res.generate_response(image_conversation)
    #     # print(f"低分辨率响应: {response_low_res}")
    #     print("分辨率调整示例已注释掉。取消注释以进行测试。")
    # except Exception as e:
    #     print(f"示例 5 (分辨率) 错误: {e}")

