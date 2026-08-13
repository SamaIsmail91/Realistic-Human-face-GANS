"""
app.py
======
A small interactive demo: load a trained checkpoint and generate faces from
the browser with sliders instead of the command line.

Run:
    python app.py --checkpoint outputs/checkpoints/ckpt_epoch0100.pt

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import argparse

import torch
import numpy as np
import gradio as gr

from generate import load_generator


def build_app(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, latent_dim = load_generator(checkpoint_path, device)

    @torch.no_grad()
    def sample(seed, num_images, interpolate):
        num_images = int(num_images)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        if interpolate:
            z1 = torch.randn(1, latent_dim, generator=gen, device=device)
            z2 = torch.randn(1, latent_dim, generator=gen, device=device)
            alphas = torch.linspace(0, 1, num_images, device=device).view(-1, 1)
            z = (1 - alphas) * z1 + alphas * z2
        else:
            z = torch.randn(num_images, latent_dim, generator=gen, device=device)

        imgs = G(z).cpu()
        imgs = ((imgs + 1) / 2 * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        return [imgs[i] for i in range(imgs.shape[0])]

    with gr.Blocks(title="Face GAN Demo") as demo:
        gr.Markdown(
            "# Synthetic Face Generator\n"
            "Samples faces from a locally-trained GAN checkpoint. "
            "Every image is generated from scratch by the model -- none of "
            "these people exist."
        )
        with gr.Row():
            seed = gr.Slider(0, 10_000, value=42, step=1, label="Random seed")
            num_images = gr.Slider(1, 16, value=8, step=1, label="Number of images")
            interpolate = gr.Checkbox(value=False, label="Latent interpolation (morph between two faces)")
        btn = gr.Button("Generate", variant="primary")
        gallery = gr.Gallery(label="Generated faces", columns=4, height="auto")

        btn.click(fn=sample, inputs=[seed, num_images, interpolate], outputs=gallery)
        demo.load(fn=sample, inputs=[seed, num_images, interpolate], outputs=gallery)

    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Interactive Gradio demo for a trained face GAN")
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint from train.py")
    p.add_argument("--share", action="store_true", help="create a public shareable link")
    args = p.parse_args()

    demo = build_app(args.checkpoint)
    demo.launch(share=args.share)
