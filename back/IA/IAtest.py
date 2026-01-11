import onnxruntime as ort
import numpy as np
from PIL import Image

# --- Config ---
onnx_model_path = "IaModel_Sharp.onnx"
input_image_path = "imgTest.webp"      # ton image high-res
output_image_path = "outputTestImg.webp"
debug_lowres_path = "debug_lowres.webp"
debug_input_path = "debug_input.webp"

# --- Load ONNX model ---
ort_session = ort.InferenceSession(onnx_model_path)

# --- High-res input pour debug ---
hr_img = Image.open(input_image_path).convert("RGB")
hr_img = hr_img.resize((256,256), Image.BICUBIC)
hr_img.save(debug_input_path)

# --- Créer low-res input pour le modèle ---
lr_img = hr_img.resize((64,64), Image.BICUBIC)   # downscale
lr_img = lr_img.resize((256,256), Image.BICUBIC) # upscale
lr_img.save(debug_lowres_path)

# --- Preprocess pour le modèle ONNX ---
inp_arr = np.array(lr_img).astype(np.float32) / 255.0  # normalisation [0,1]
inp_arr = np.transpose(inp_arr, (2,0,1))               # HWC -> CHW
inp_arr = np.expand_dims(inp_arr, axis=0)              # [1,3,256,256]

# --- Inference ---
out_arr = ort_session.run(None, {"input": inp_arr})[0]  # output [1,3,256,256]
out_arr = np.clip(out_arr[0].transpose(1,2,0), 0, 1)   # CHW -> HWC

# --- Convertir en image et sauvegarder ---
out_img = Image.fromarray((out_arr*255).astype(np.uint8))
out_img.save(output_image_path)

print("Test terminé !")
print(f"- High-res input saved at: {debug_input_path}")
print(f"- Low-res input saved at: {debug_lowres_path}")
print(f"- Model output saved at: {output_image_path}")


