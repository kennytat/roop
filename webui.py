import os
import sys
import subprocess
from pathlib import Path
import shutil
import tempfile
import gradio as gr
from datetime import datetime
import torch
from roop.core import suggest_execution_providers
DEVICES = suggest_execution_providers()
DEVICE = 'coreml' if sys.platform == 'darwin' else ('dml' if sys.platform == 'win32' else ('cuda' if torch.cuda.is_available() else 'cpu'))   
DEVICE = DEVICE if DEVICE in DEVICES else 'cpu'

def new_dir_now():
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d%H%M")
    return date_time
  
def process_start(source_image, input_files):
  temp_dir = os.path.join(tempfile.gettempdir(), "faceswap", new_dir_now())
  Path(temp_dir).mkdir(parents=True, exist_ok=True)
  os.system(f"rm -rf {temp_dir}/*")
  results_list = []
  if source_image and len(input_files) >0:
    print("run called::", source_image, input_files)
    input_files = [input_file if isinstance(input_file, str) else input_file.name for input_file in input_files]
    for target_image in input_files:
      subprocess.run(["python","run.py", "--execution-provider", DEVICE, "-s", source_image, "-t", target_image, "-o", temp_dir])
    shutil.make_archive(temp_dir, 'zip', temp_dir)   
    results_list.append(f"{temp_dir}.zip")
    ## Remove tmp files
    shutil.rmtree(temp_dir, ignore_errors=True)
    return results_list
  else:
    raise Exception("Input invalid, please try again!!")
  

def web_interface():
  css = """
  .btn-active {background-color: "orange"}
  """
  app = gr.Blocks(title="Face Swap Enhancer", theme=gr.themes.Default(), css=css)
  with app:
      gr.Markdown("# Face Swap Enhancer")
      with gr.Tabs():
          with gr.TabItem("FaceSwap"):
              with gr.Row():
                  with gr.Column():
                      with gr.Row():
                        source_image = gr.Image(type="filepath", label="Source Face")
                      with gr.Row():
                        input_files = gr.Files(label="Target images (batch)", file_types=["image"])
                  with gr.Column():
                      with gr.Row():
                        files_output = gr.Files(label="PROGRESS BAR")
                      with gr.Row():
                        gr.ClearButton([source_image, input_files, files_output])
                        btn = gr.Button(value="Generate!", variant="primary")
                        btn.click(fn=process_start,
                                inputs=[source_image, input_files],
                                outputs=[files_output])
                                
  auth_user = os.getenv('AUTH_USER', '')
  auth_pass = os.getenv('AUTH_PASS', '')
  app.queue().launch(
    auth=(auth_user, auth_pass) if auth_user != '' and auth_pass != '' else None,
    show_api=False,
    debug=False,
    inbrowser=True,
    show_error=True,
    server_name="0.0.0.0",
    server_port=7890,
    share=False)
  
if __name__ == '__main__':
    web_interface()
