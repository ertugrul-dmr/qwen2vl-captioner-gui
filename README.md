# Image Captioning with Gradio

This repo provides a simple GUI application for using the [Qwen2-VL-7B-Captioner-Relaxed](https://huggingface.co/Ertugrul/Qwen2-VL-7B-Captioner-Relaxed) model for image captioning. It is 
## Features

- Upload an image and generate a caption.
- Select from predefined system prompts or enter a custom prompt.
- Automatically load and initialize the model on the first caption generation.

## Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- GPU with CUDA support with at least of 16GB VRAM
- 16 GB of storage for the model weights

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ertugrul-dmr/qwen2vl-captioner-gui.git
   cd image-captioning-gradio
   ```

2. Install the required packages:

   ### For Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip3 install torch
   pip3 install -r requirements.txt
   ```

   ### For Windows:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

### Running the Application

1. Run the `app.py` script:

   ```bash
   python app.py
   ```

2. Open your web browser and go to `http://localhost:7860` to access the Gradio interface.

## Usage

1. Upload an image by clicking on the image input box.
2. Select a system prompt from the dropdown menu or enter a custom prompt or leave as default.
3. Click the "Generate Caption" button to generate a caption for the image.
4. The generated caption will be displayed in the "Generated Caption" textbox.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Gradio](https://gradio.app/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Pillow](https://python-pillow.org/)
- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
