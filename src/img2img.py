import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
from torchvision import transforms as tfms

torch.manual_seed(1)

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

# ---------------------------------------------------------------------------- #
#                              Inference Pipeline                              #
# ---------------------------------------------------------------------------- #
prompt = ["A watercolor painting of an otter"]
generator = torch.manual_seed(32)
num_inference_timesteps = 50
height = 512
width  = 512
batch_size = 1
guidance_scale = 8

input_image = Image.open('../assets/macaw.jpg').resize((512, 512))

def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


encoded = pil_to_latent(input_image)
'''
We want to convert one image to another. So, what we can do is, use the prompt for guidance, and we can also start from src
image itself. So, what we do to do the denoising process is use the VAE output of the source image and then add noise to the latents.
Hopefully, the final latents we land on, when upscaled can give us what we want.
'''
scheduler.set_timesteps(num_inference_steps=num_inference_timesteps)
noise = torch.randn_like(encoded, generator=generator)
latents = scheduler.add_noise(encoded, noise, 0)
latents = latents * scheduler.init_noise_sigma

prompt_embeds = tokenizer(prompt, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True, padding="max_length")
uncond_embeds = tokenizer([""], return_tensors="pt", max_length=prompt_embeds.input_ids.shape[-1], truncation=True, padding="max_length")

with torch.no_grad():
    prompt_embeds = text_encoder(prompt_embeds.input_ids.to(torch_device)).last_hidden_state
    uncond_embeds = text_encoder(uncond_embeds.input_ids.to(torch_device)).last_hidden_state
    embeddings = torch.cat([uncond_embeds, prompt_embeds])

with autocast("cuda"):
    for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
        latent_input = torch.cat([latents]*2).to(torch_device)
        latent_input = scheduler.scale_model_input(latent_input, t)

        with torch.no_grad():
            noise_preds = unet.forward(latent_input, t, encoder_hidden_states=embeddings).sample

        uncond_noise, prompt_noise = noise_preds.chunk(2)
        noise = uncond_noise + guidance_scale * (prompt_noise - uncond_noise)

        latents = scheduler.step(noise, t, latents).prev_sample

latents_to_pil(latents)[0]
