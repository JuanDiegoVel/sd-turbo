import torch
from diffusers import AutoPipelineForText2Image

print("Cargando modelo...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32
     # FloatXX 32 o 16
     # Me indica la cantidad de decimales de precision
     # Con el que el modelo creara la imagen
     # Si mi pc me genera una imagen negra, cambio a 16
)

modelo = modelo.to("cpu")
# cuda - nvidia
# mps - Mac com procesadores MX
# xpu - Grafica Intel IRIS

prompt = input("Escribe el prompt de la imagen: ")
negative_prompt="blurry, low quality, worst quality, deformed anatomy, poorly drawn hands, extra fingers, missing limbs, distorted face, watermark, signature, text overlay, jpeg artifacts, grain, ugly, out of frame"
print("Generando imagen...")

imagen = modelo(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=10, # Numero de veces que creara y
    guidance_scale=2.0 , # Que tan fiel sera al promot
).images[0]

imagen.save("imagen.png")
print("Imagen Guardada")
