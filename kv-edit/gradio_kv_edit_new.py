import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from PIL import ExifTags, Image
import torch
import gradio as gr
import numpy as np
from flux.sampling import prepare
from flux.util import (configs, load_ae, load_clip, load_t5)
from models.kv_edit_target import Flux_kv_edit_target

@dataclass
class SamplingOptions:
    source_prompt: str = ''
    target_prompt: str = ''
    width: int = 1366
    height: int = 768
    inversion_num_steps: int = 0
    denoise_num_steps: int = 0
    skip_step: int = 0
    inversion_guidance: float = 1.0
    denoise_guidance: float = 1.0
    seed: int = 42
    re_init: bool = False
    attn_mask: bool = False
    attn_scale: float = 1.0

class FluxEditor_kv_demo:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.offload = args.offload

        self.name = args.name
        self.is_schnell = args.name == "flux-schnell"

        self.output_dir = 'regress_result'

        self.t5 = load_t5(self.device, max_length=256 if self.name == "flux-schnell" else 512)
        self.clip = load_clip(self.device)
        self.model = Flux_kv_edit_target(device="cpu" if self.offload else self.device, name=self.name)
        self.ae = load_ae(self.name, device="cpu" if self.offload else self.device)

        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()
        self.info = {}
        self.z0 = None
        self.zt = None
        self.init_image = None
        self.target_info = None  # populated by inverse_target()

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.encoder.to(self.device)

    # ------------------------------------------------------------------
    # Source inversion (unchanged behaviour)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def inverse(self, brush_canvas,
                source_prompt, target_prompt,
                inversion_num_steps, denoise_num_steps,
                inversion_guidance, denoise_guidance, seed,
                re_init, attn_mask):

        self.z0 = None
        self.zt = None
        self.target_info = None  # reset target cache on new source inversion
        if 'feature' in self.info:
            for key in list(self.info['feature'].keys()):
                del self.info['feature'][key]
        self.info = {}

        rgba_init_image = brush_canvas["background"]
        init_image = rgba_init_image[:, :, :3]
        shape = init_image.shape
        height = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        width  = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image      = init_image[:height, :width, :]
        rgba_init_image = rgba_init_image[:height, :width, :]

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps,
            skip_step=0,  # no skip during inversion
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask,
        )
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        torch.cuda.empty_cache()

        if opts.attn_mask:
            rgba_mask = brush_canvas["layers"][0][:height, :width, :]
            mask = rgba_mask[:, :, 3] / 255
            mask = mask.astype(int)
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device)
        else:
            mask = None

        self.init_image = self.encode(init_image, self.device).to(self.device)

        t0 = time.perf_counter()

        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        with torch.no_grad():
            inp = prepare(self.t5, self.clip, self.init_image, prompt=opts.source_prompt)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        self.z0, self.zt, self.info = self.model.inverse(inp, mask, opts)

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()
        print(f"Source inversion done in {t1 - t0:.1f}s.")
        return None

    # ------------------------------------------------------------------
    # Target image inversion (new)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def inverse_target(self, brush_canvas, target_image,
                       source_prompt, target_prompt,
                       inversion_num_steps, denoise_num_steps,
                       inversion_guidance, denoise_guidance, seed,
                       re_init, attn_mask, target_bbox_scale):
        """
        1. Extracts mask bounding box from the canvas.
        2. Removes background from target image (rembg).
        3. Crops the foreground object, resizes it to fit the (optionally scaled) mask bbox.
        4. Pastes the object onto a white canvas at the mask position.
        5. Inverts this composed canvas with the target prompt → stores target K/V cache.

        Returns a preview of the composed canvas so the user can verify placement.
        """
        if target_image is None:
            print("No target image provided.")
            return None

        # --- Source dimensions (from the already-uploaded background) ---
        rgba_bg = brush_canvas["background"]
        height = rgba_bg.shape[0] if rgba_bg.shape[0] % 16 == 0 else rgba_bg.shape[0] - rgba_bg.shape[0] % 16
        width  = rgba_bg.shape[1] if rgba_bg.shape[1] % 16 == 0 else rgba_bg.shape[1] - rgba_bg.shape[1] % 16

        # --- Mask bounding box ---
        rgba_mask = brush_canvas["layers"][0][:height, :width, :]
        mask_alpha = rgba_mask[:, :, 3]
        if mask_alpha.max() == 0:
            print("No mask drawn. Draw a mask first, then click Invert Target.")
            return None

        rows = np.where(np.any(mask_alpha > 0, axis=1))[0]
        cols = np.where(np.any(mask_alpha > 0, axis=0))[0]
        rmin, rmax = int(rows[0]), int(rows[-1])
        cmin, cmax = int(cols[0]), int(cols[-1])

        # Optionally expand the bbox (target_bbox_scale > 1.0 gives more room)
        if target_bbox_scale != 1.0:
            bbox_h = rmax - rmin + 1
            bbox_w = cmax - cmin + 1
            pad_h = int(bbox_h * (target_bbox_scale - 1.0) / 2)
            pad_w = int(bbox_w * (target_bbox_scale - 1.0) / 2)
            rmin = max(0, rmin - pad_h)
            rmax = min(height - 1, rmax + pad_h)
            cmin = max(0, cmin - pad_w)
            cmax = min(width - 1, cmax + pad_w)

        bbox_h = rmax - rmin + 1
        bbox_w = cmax - cmin + 1

        # --- Background removal ---
        from rembg import remove as rembg_remove
        target_pil = Image.fromarray(target_image.astype(np.uint8))
        target_no_bg = rembg_remove(target_pil)          # returns RGBA PIL Image
        target_rgba  = np.array(target_no_bg)

        # Crop to the foreground object bounding box inside the target image
        t_alpha = target_rgba[:, :, 3]
        t_rows  = np.where(np.any(t_alpha > 0, axis=1))[0]
        t_cols  = np.where(np.any(t_alpha > 0, axis=0))[0]
        if len(t_rows) == 0 or len(t_cols) == 0:
            print("No foreground detected after background removal.")
            return None
        obj_rgba = target_rgba[t_rows[0]:t_rows[-1] + 1, t_cols[0]:t_cols[-1] + 1, :]

        # Resize object to mask bbox dimensions
        obj_pil     = Image.fromarray(obj_rgba, 'RGBA')
        obj_resized = np.array(obj_pil.resize((bbox_w, bbox_h), Image.LANCZOS))

        # Composite onto a white canvas at source resolution
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        alpha  = obj_resized[:, :, 3:4] / 255.0
        rgb    = obj_resized[:, :, :3]
        bg_crop = canvas[rmin:rmax + 1, cmin:cmax + 1].copy()
        canvas[rmin:rmax + 1, cmin:cmax + 1] = (
            rgb * alpha + bg_crop * (1 - alpha)
        ).astype(np.uint8)

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps,
            skip_step=0,  # always full inversion for key alignment
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=int(seed),
            re_init=re_init,
            attn_mask=attn_mask,
        )

        t0 = time.perf_counter()

        target_latent = self.encode(canvas, self.device).to(self.device)

        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        with torch.no_grad():
            inp_target_full = prepare(self.t5, self.clip, target_latent, prompt=opts.target_prompt)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        self.target_info = self.model.inverse_target(inp_target_full, opts)

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()
        print(f"Target inversion done in {t1 - t0:.1f}s.")

        # Return the composed canvas as a preview
        return Image.fromarray(canvas)

    # ------------------------------------------------------------------
    # Edit
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def edit(self, brush_canvas,
             source_prompt, target_prompt,
             inversion_num_steps, denoise_num_steps,
             skip_step,
             inversion_guidance, denoise_guidance, seed,
             re_init, attn_mask, attn_scale):

        if self.z0 is None or self.zt is None or not self.info:
            raise gr.Error("Run 'Invert Source' before editing.")

        torch.cuda.empty_cache()

        rgba_init_image = brush_canvas["background"]
        init_image = rgba_init_image[:, :, :3]
        shape  = init_image.shape
        height = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        width  = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image      = init_image[:height, :width, :]
        rgba_init_image = rgba_init_image[:height, :width, :]

        rgba_mask = brush_canvas["layers"][0][:height, :width, :]
        mask = rgba_mask[:, :, 3] / 255
        mask = mask.astype(int)

        rgba_mask[:, :, 3] = rgba_mask[:, :, 3] // 2
        masked_image = Image.alpha_composite(
            Image.fromarray(rgba_init_image, 'RGBA'),
            Image.fromarray(rgba_mask, 'RGBA'),
        )
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device)

        seed = int(seed)
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps,
            skip_step=skip_step,
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask,
            attn_scale=attn_scale,
        )

        if self.offload:
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)

        t0 = time.perf_counter()

        with torch.no_grad():
            inp_target = prepare(self.t5, self.clip, self.init_image, prompt=opts.target_prompt)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        # Pass target_info (None if no target image was inverted → falls back to text-only)
        x = self.model.denoise(
            self.z0, self.zt, inp_target, mask, opts, self.info,
            target_info=self.target_info,
        )

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x.to(self.device))

        x = x.clamp(-1, 1)
        x = x.float().cpu()
        x = rearrange(x[0], "c h w -> h w c")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        output_name = os.path.join(self.output_dir, "img_{idx}.jpg")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            idx = 0
        else:
            fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
            idx = max((int(fn.split("_")[-1].split(".")[0]) for fn in fns), default=-1) + 1

        fn = output_name.format(idx=idx)
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
        exif_data[ExifTags.Base.ImageDescription] = target_prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)
        masked_image.save(fn.replace(".jpg", "_mask.png"), format='PNG')

        t1 = time.perf_counter()
        print(f"Edit done in {t1 - t0:.1f}s. Saved {fn}")
        return img

    # ------------------------------------------------------------------
    # VAE encoding helper
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def encode(self, init_image, torch_device):
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0)
        init_image = init_image.to(torch_device)
        self.ae.encoder.to(torch_device)
        init_image = self.ae.encode(init_image).to(torch.bfloat16)
        return init_image


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_demo(model_name: str):
    editor = FluxEditor_kv_demo(args)
    is_schnell = model_name == "flux-schnell"

    title = r"""
        <h1 align="center">🎨 KV-Edit: Training-Free Image Editing for Precise Background Preservation</h1>
        """

    description = r"""
        <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/Xilluill/KV-Edit' target='_blank'><b>KV-Edit</b></a>
        — now with <b>target-image-guided editing</b>.<br>

        💫 <b>Editing steps:</b><br>
        1️⃣ Upload your <b>source image</b> and fill in the <b>source prompt</b>.<br>
        2️⃣ Click <b>Invert Source</b>.<br>
        3️⃣ Draw a <b>mask</b> on the source image over the region to edit.<br>
        4️⃣ Upload a <b>target image</b> (the object to place) and fill in the <b>target prompt</b>.<br>
        5️⃣ Adjust <b>Target BBox Scale</b> if the object needs more space than the mask (e.g. a tall hat).<br>
        6️⃣ Click <b>Invert Target</b> — preview the aligned placement before editing.<br>
        7️⃣ Click <b>Edit</b>.<br>

        🔔 <b>Target image</b> is optional — omit it for pure text-guided editing (original behaviour).<br>
        🔔 <b>Target BBox Scale</b> expands the mask bounding box symmetrically before fitting the object
        (1.0 = exact mask, 1.3 = 30% larger in all directions).
        """

    article = r"""
    If our work is helpful, please help to ⭐ the <a href='https://github.com/Xilluill/KV-Edit' target='_blank'>Github Repo</a>. Thanks!
    """

    with gr.Blocks() as demo:
        gr.HTML(title)
        gr.Markdown(description)

        with gr.Row():
            # ---- Left column: prompts + canvases + action buttons ----
            with gr.Column():
                source_prompt = gr.Textbox(label="Source Prompt", value='')
                inversion_num_steps = gr.Slider(1, 50, 28, step=1, label="Number of inversion steps")
                target_prompt = gr.Textbox(label="Target Prompt", value='')
                denoise_num_steps = gr.Slider(1, 50, 28, step=1, label="Number of denoise steps")

                with gr.Row():
                    brush_canvas = gr.ImageEditor(
                        label="Source Image (draw mask here)",
                        sources=('upload',),
                        brush=gr.Brush(colors=["#ff0000"], color_mode='fixed'),
                        interactive=True,
                        transforms=[],
                        container=True,
                        format='png',
                        scale=1,
                    )
                    with gr.Column(scale=1):
                        target_image = gr.Image(
                            label="Target Image (optional)",
                            type='numpy',
                            sources=['upload'],
                        )
                        target_preview = gr.Image(
                            label="Aligned Target Preview",
                            interactive=False,
                        )

                with gr.Row():
                    inv_btn        = gr.Button("Invert Source")
                    inv_target_btn = gr.Button("Invert Target")
                    edit_btn       = gr.Button("Edit")

            # ---- Right column: advanced options + output ----
            with gr.Column():
                with gr.Accordion("Advanced Options", open=True):
                    skip_step         = gr.Slider(0, 30, 4,   step=1,   label="Number of skip steps")
                    inversion_guidance = gr.Slider(1.0, 10.0, 1.5, step=0.1, label="Inversion Guidance",
                                                   interactive=not is_schnell)
                    denoise_guidance   = gr.Slider(1.0, 10.0, 5.5, step=0.1, label="Denoise Guidance",
                                                   interactive=not is_schnell)
                    attn_scale        = gr.Slider(0.0, 5.0,  1.0, step=0.1, label="attn_scale")
                    target_bbox_scale = gr.Slider(0.5, 3.0,  1.0, step=0.05,
                                                  label="Target BBox Scale (expand mask bbox for larger objects)")
                    seed              = gr.Textbox('0', label="Seed (-1 for random)", visible=True)
                    with gr.Row():
                        re_init   = gr.Checkbox(label="re_init",   value=False)
                        attn_mask = gr.Checkbox(label="attn_mask", value=False)

                output_image = gr.Image(label="Generated Image")
                gr.Markdown(article)

        # ---- Button wiring ----
        inv_btn.click(
            fn=editor.inverse,
            inputs=[brush_canvas,
                    source_prompt, target_prompt,
                    inversion_num_steps, denoise_num_steps,
                    inversion_guidance, denoise_guidance, seed,
                    re_init, attn_mask],
            outputs=[output_image],
        )

        inv_target_btn.click(
            fn=editor.inverse_target,
            inputs=[brush_canvas, target_image,
                    source_prompt, target_prompt,
                    inversion_num_steps, denoise_num_steps,
                    inversion_guidance, denoise_guidance, seed,
                    re_init, attn_mask, target_bbox_scale],
            outputs=[target_preview],
        )

        edit_btn.click(
            fn=editor.edit,
            inputs=[brush_canvas,
                    source_prompt, target_prompt,
                    inversion_num_steps, denoise_num_steps,
                    skip_step,
                    inversion_guidance, denoise_guidance, seed,
                    re_init, attn_mask, attn_scale],
            outputs=[output_image],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux KV-Edit with target image guidance")
    parser.add_argument("--name",    type=str, default="flux-dev", choices=list(configs.keys()))
    parser.add_argument("--device",  type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share",   action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port",    type=int, default=41032)
    args = parser.parse_args()

    demo = create_demo(args.name)
    demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)
