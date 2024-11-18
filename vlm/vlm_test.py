import torch
from PIL import Image
import requests
from transformers import MllamaForConditionalGeneration, AutoProcessor,ViTImageProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
# processor_1 = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# processor.image_processor = ViTImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base") #openai/clip-vit-base-patch32")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)
images = []
# for i in range(300):
#     images.append(image)
# image = Image.open("success_exp_traj_100.gif")
# image = Image.open('failure_exp_traj_101.gif')
image = Image.open('llama-recipes/test_0.png')
contents = []
question = "The robot is trying to put the cup from the counter and place it into the sink. You are a runtime montior trying to monitor the task progress. The images is the last observation of the robot execution for this task. The image is composed of wrist camera view (top) and front camera view (bottom). Based on the image, summarize if the task is success or failure. Answer is one word either failure or success."

for i in range(1):
    contents.append({"type": "image"})
contents.append(    {"type": "text", "text": question})
messages = [
    {"role": "user", "content": 
     contents
    }
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)
breakpoint()

output = model.generate(**inputs, max_new_tokens=3)
print(processor.decode(output[0]))