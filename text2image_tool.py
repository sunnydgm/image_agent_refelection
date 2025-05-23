import requests
def generate_background_image(prompt, api_key, output_path, style_type="enhance", seed=42):
    host = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "image/*"}
    data = {
        "prompt": (None, prompt),
        "negative_prompt": (None, "character, human, figure, face, body, person, head, eyes, mouth, nose, text, animal, object, portrait, cartoon, cartoon character, anime, logo, signature, watermark, car, vehicle, automobile, wheel, tire, window, lights, branding, shadow, blurry, low quality"),
        "output_format": (None, "png"),
        "seed": (None, str(seed)),
        "aspect_ratio": (None, "1:1"),
        "style_preset": (None, style_type),
        "denoising_strength": (None, "0.9"),
        "guidance_scale": (None, "30"),
        "num_inference_steps": (None, "60")
    }
    response = requests.post(host, headers=headers, files=data)
    if response.status_code == 200:
        with open(output_path, "wb") as out_file:
            out_file.write(response.content)
        print(f"✅ Image saved to: {output_path}")
        return output_path
    else:
        raise RuntimeError(f"Request failed: {response.status_code} - {response.text}")
