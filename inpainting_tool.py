import requests
def generate_background_image_inpainting(prompt, api_key, save_path, style_preset, init_image_path, mask_image_path, seed):
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "image/*"}
    with open(init_image_path, "rb") as init_img, open(mask_image_path, "rb") as mask_img:
        files = {"image": init_img, "mask": mask_img}
        data = {
            "prompt": prompt,
            "output_format": "png",
            "seed": seed,
            "mode": "mask",
            "masked_content": "latent_noise",
            "inpaint_area": "only_masked",
            "grow_mask": 50,
            "denoising_strength": 0.85,
            "guidance_scale": 30,
            "num_inference_steps": 50,
            "style_preset": style_preset
        }
        response = requests.post("https://api.stability.ai/v2beta/stable-image/edit/inpaint", headers=headers, data=data, files=files)
        if response.status_code == 200:
            with open(save_path, "wb") as out_file:
                out_file.write(response.content)
            print(f"✅ Inpainting result saved to: {save_path}")
            return save_path
        else:
            raise RuntimeError(f"Request failed: {response.status_code} - {response.text}")