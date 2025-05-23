import requests
def generate_img2img_adjust(input_image_path, prompt, output_path, api_key, style_preset, seed, structure_type="depth", guidance_scale=30, steps=50, control_strength=0.8):
    url = "https://api.stability.ai/v2beta/stable-image/control/structure"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "image/*"}
    
    negative_prompt = "car, vehicle, automobile, wheels, tires, windows, mirrors, headlights, bumpers, license plates, reflections, human, person, people, face, head, body, arms, eyes, lips, skin, portrait, character, figure, model, girl, woman, man, baby, child, humanoid, anatomy, nude, clothing, fashion, hands, feet, photorealistic person, nose, animal, logo, text, watermark, signature, cartoon, objects, shadows, blurry details, low quality"

    with open(input_image_path, "rb") as image_file:
        files = {"image": image_file}
        data = {
            "prompt": prompt,
            "negative_prompt": "character, face, person, creature, animal, car,vehicle, object, text, watermark, blurry, low quality",
            "structure_type": structure_type,
            "output_format": "png",
            "style_preset": style_preset,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": steps,
            "control_strength": control_strength
        }
        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Img2Img Adjust result saved to: {output_path}")
            return output_path
        else:
            raise RuntimeError(f"Request failed: {response.status_code} - {response.text}")