import gradio as gr
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import warnings
import sys
from io import StringIO
import gc

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize model and processor variables
model = None
processor = None

# Create a custom StringIO object to capture print statements
class CustomStringIO(StringIO):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def write(self, string):
        super().write(string)
        self.callback(self.getvalue())

def load_model(status_box):
    global model, processor
    model_id = "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"
    
    print("Loading model...")
    if isinstance(status_box, gr.components.Textbox):
        status_box.update(value="Loading model...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=512*512, max_pixels=1280*1280)
    
    print("Model loaded successfully!")
    if isinstance(status_box, gr.components.Textbox):
        status_box.update(value="Model loaded successfully!")
    return "Model loaded successfully!"

def resize_image(image, longest_size):
    width, height = image.size
    if width > height:
        new_width = longest_size
        new_height = int(height * (longest_size / width))
    else:
        new_height = longest_size
        new_width = int(width * (longest_size / height))
    return image.resize((new_width, new_height), Image.LANCZOS)

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

def generate_caption(image, longest_size, system_prompt, status_box):
    global model, processor
    if model is None or processor is None:
        load_model(status_box)
    
    try:
        # Convert image to PIL Image and resize
        image = Image.fromarray(image).convert('RGB')
        image = resize_image(image, longest_size)
        
        print("Processing image...")
        if isinstance(status_box, gr.components.Textbox):
            status_box.update(value="Processing image...")
        
        # Prepare conversation
        conversation = []
        # Add system prompt if not None
        if system_prompt:
            conversation.append({
                "role": "system",
                "content": system_prompt
            })
        conversation.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        })
        
        # Process input
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Generating caption...")
        if isinstance(status_box, gr.components.Textbox):
            status_box.update(value="Generating caption...")
        
        # Generate caption
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                output_ids = model.generate(**inputs, max_new_tokens=384, do_sample=True, temperature=0.7, use_cache=True, top_k=50)
        
        # Decode output
        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        output_text = processor.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        print("Caption generated successfully")
        if isinstance(status_box, gr.components.Textbox):
            status_box.update(value="Caption generated successfully")
        
        # Delete variables that will be recreated for next generation
        del image, conversation, text_prompt, inputs, output_ids, generated_ids
        
        # Clean memory after generation
        clean_memory()
        
        return output_text, "Caption generated successfully"
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        if isinstance(status_box, gr.components.Textbox):
            status_box.update(value=error_message)
        return error_message, "Error during caption generation"

# Create Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Image Captioning")
    gr.Markdown("Upload an image and click 'Generate Caption' to get a description. The model will be loaded(downloaded first time) automatically when you generate the first caption.")
    
    with gr.Row():
        image_input = gr.Image(type="numpy")
        with gr.Column():
            longest_size = gr.Slider(minimum=512, maximum=2048, value=768, step=64, label="Longest Size (pixels)")
            
            # Add predefined system prompts
            system_prompt_choices = [
                "Use full description",
                "Use short description",
                "Use brief description",
                "Use detailed description",
                "Describe the image in a poetic way (Experimental)",
                "Single word description",
                "Custom"
            ]
            
            system_prompt_mapping = {
                "Use full description": None,
                "Use short description": "Use maximum of 10 words.",
                "Use brief description": "Use maximum of 50 words.",
                "Use detailed description": "Use maximum of 100 words.",
                "Describe the image in a poetic way (Experimental)": "Describe the image in a poetic way.",
                "Single word description": "Describe the image in a word only. Nothing more. Strict rule. Only 1 word!!!",
                "Custom": "Custom"
            }
            
            system_prompt_dropdown = gr.Dropdown(
                choices=system_prompt_choices,
                label="System Prompt",
                value="Use full description"
            )
            system_prompt_input = gr.Textbox(
                label="Custom System Prompt", 
                placeholder="Enter a custom system prompt",
                visible=False
            )
    
    generate_button = gr.Button("Generate Caption", variant="primary")
    
    with gr.Row():
        caption_output = gr.Textbox(label="Generated Caption")
        model_status = gr.Textbox(label="Model Status", interactive=False)
    
    def update_system_prompt_visibility(choice):
        return gr.update(visible=choice == "Custom")
    
    system_prompt_dropdown.change(
        fn=update_system_prompt_visibility,
        inputs=system_prompt_dropdown,
        outputs=system_prompt_input
    )
    
    def get_system_prompt(dropdown_choice, custom_prompt):
        if dropdown_choice == "Custom":
            return custom_prompt
        else:
            return system_prompt_mapping[dropdown_choice]
    
    generate_button.click(
        fn=generate_caption, 
        inputs=[
            image_input, 
            longest_size, 
            gr.Textbox(value=get_system_prompt, inputs=[system_prompt_dropdown, system_prompt_input], visible=False),
            model_status
        ], 
        outputs=[caption_output, model_status]
    )

# Launch the interface
iface.launch()
