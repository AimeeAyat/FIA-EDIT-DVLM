<div align="center">
<h1>EEdit ⚡️：Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing</h1>
</div>
<div align="center" style="display: flex; justify-content: space-around; flex-wrap: wrap;">
    <a href="#">Zexuan Yan</a><sup>*</sup> &nbsp;•&nbsp;
    <a href="#">Yue Ma</a><sup>*</sup> &nbsp;•&nbsp;
    <a href="#">Chang Zou</a> &nbsp;•&nbsp;
    <a href="#">Wenteng Chen</a> &nbsp;•&nbsp;
    <a href="#">Qifeng Chen</a> &nbsp;•&nbsp;
    <a href="#">Linfeng Zhang</a><sup>†</sup>
</div>
<div align="center" style="font-size: 0.9em">
    <sup>*</sup> Equal Contribution &nbsp;&nbsp;&nbsp; <sup>†</sup> Corresponding Author
</div>


[![arXiv](https://img.shields.io/badge/arXiv-2403.18162-b31b1b.svg)](https://arxiv.org/abs/2503.10270)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://eff-edit.github.io/)

## 📝 Introduction
<div align="center" style="background-color: white;">
    <img src="static/teaser.png" alt="EEdit Teaser" width="100%">
</div>

Inversion-based image editing is rapidly gaining momentum while suffering from significant computation overhead, hindering its application in real-time interactive scenarios. In this paper, we rethink that the redundancy in inversion-based image editing exists in both the spatial and temporal dimensions, such as the unnecessary computation in unedited regions and the redundancy in the inversion progress.

To tackle these challenges, we propose a practical framework, named EEdit, to achieve efficient image editing. Specifically, we introduce three techniques to solve them one by one:
- For spatial redundancy, **spatial locality caching** is introduced to compute the edited region and its neighboring regions while skipping the unedited regions
- **Token indexing preprocessing** is designed to further accelerate the caching
- For temporal redundancy, **inversion step skipping** is proposed to reuse the latent for efficient editing

Our experiments demonstrate an average of **2.46X acceleration** without performance drop in a wide range of editing tasks including prompt-guided image editing, dragging and image composition.



## 🛠️ Installation
```bash
conda create -n eedit python=3.12.3
conda activate eedit
pip install -r EEdit/requirements.txt
```

## 📥 Checkpoints & Datasets
All model weights and datasets are from open-source, free and publicly available channels:

We use FLUX-dev as our experimental model. You can obtain it from either:
- Official Repository: https://github.com/black-forest-labs/flux
- Hugging Face: https://huggingface.co/black-forest-labs/FLUX.1-dev

We use PIE-BENCH as the prompt-guided dataset, you can refer to [link](https://forms.gle/hVMkTABb4uvZVjme9)

We use TF-ICON benchmark as the ref-guided dataset, you can refer to [link](https://github.com/Shilin-LU/TF-ICON)

We use DragBench-DR and Drag-Bench-SR as the drag-guided datasets, you can refer to [link](https://github.com/Visual-AI/RegionDrag)

For masks generated from mapping_file.json, we provide scripts that follows:

```bash
python MyCodes/myutils.py
```

When all the data and checkpoints are ready, please follow our file directory structure to be compatible with the scripts.

<details>
<summary>📁 Checkpoints Structure</summary>

```bash
weights
├── flux1-dev.safetensors
├── model_index.json
├── scheduler
│   └── scheduler_config.json
├── sd_vae_ft_mse
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── text_encoder
│   ├── config.json
│   └── model.safetensors
├── text_encoder_2
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   ├── model.safetensors.index-1.json
│   └── model.safetensors.index.json
├── tokenizer
│   ├── merges-1.txt
│   ├── merges.txt
│   ├── special_tokens_map-1.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config-1.json
│   ├── tokenizer_config.json
│   ├── vocab-1.json
│   └── vocab.json
├── tokenizer_2
│   ├── special_tokens_map-1.json
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── transformer_config.json
└── vae
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```
</details>


<details>
<summary>📁 Datasets Structure</summary>

```bash
input
├── composition
│   ├── Real-Cartoon
│   │   ├── 0000 a cartoon animation of a sheep in the forest
│   │   │   ├── bg58.png
│   │   │   ├── cp_bg_fg.jpg
│   │   │   ├── dccf_image.jpg
│   │   │   ├── fg35_63d22cda1f5b66e8e5aca776.jpg
│   │   │   ├── fg35_mask.png
│   │   │   └── mask_bg_fg.jpg
│   ├── Real-Painting
│   ├── Real-Sketch
│   ├── Real-Real
    ...

├── drag_data
│   ├── dragbench-dr
│   │   ├── animals
│   │   │   ├── JH_2023-09-14-1820-16
│   │   │   │   ├── meta_data.pkl
│   │   │   │   ├── meta_data_region.pkl
│   │   │   │   ├── original_image.png
│   │   │   │   └── user_drag.png
│   └── dragbench-sr
│       ├── art_0
│       │   ├── meta_data.pkl
│       │   ├── meta_data_region.pkl
│       │   ├── original_image.png
│       │   └── user_drag.png
    ...

├── inpaint
│   ├── annotation_images 
│   │   ├── 0_random_140
│   │   │   ├── 000000000000.jpg
│   │   │   
│   │   ├── 1_change_object_80
│   │   │   ├── 1_artificial
│   │   │   │   ├── 1_animal
│   │   │   │   │   ├── 111000000000.jpg
│   │   │   │   ├── 2_human
│   │   │   │   │   ├── 112000000000.jpg
│   │   ├── 2_add_object_80
│   │   │   ├── 1_artificial
│   │   │   │   ├── 1_animal
│   │   │   │   │   ├── 211000000000.jpg
        ...
│   ├── mapping_file.json
│   └── masks
│       ├── mask-000.png
|       ├── mask-001.png
|       ├── ...
```
</details>

## 🚀 Generation
```bash
cd EEdit && source run_gen.sh
```

## ✨ Hyper-parameters Guidance
As we all know, edited results are affected by many parameters and even random seed. We try our best to explain those parameters and options that may affect image quality, so that you can generate satisfactory image editing results by yourself. 
<details>
<summary>Reference-guided editing (Image Composition)</summary>

| Parameter | Value | Description |
|-----------|--------|-------------|
| eta | 0.6 | Controls the strength of inversion injection during denoising. Higher values preserve more of the original image. |
| gamma | 0.6 | Controls the strength of inversion injection during denoising. Higher values preserve more of the original image. |
| blend_ratio | 0 | Deprecated parameter, not in use. |
| start_timestep | 0 | Fixed parameter, no adjustment needed. |
| stop_timestep | 10 | Number of timesteps where inversion affects denoising. Higher values result in output closer to original image. |
| use_rf_inversion | true | Fixed parameter, keep as true.  |
| use_cache | true<br>false | Enable caching to accelerate inference. False will perform as non-accelerated pipeline. |
| num_inference_steps | 28 | Number of inference steps in the diffusion process. |
| cascade_num | 1/3/5 | Controls region score bonus for K-L1 distance neighboring regions. No adjustment needed. |
| fresh_ratio | 0.1 | Cache refresh ratio. Fixed parameter, higher is not always better. |
| fresh_threshold | 1/2/3 | Complete refresh interval. Set to 2 if edit results don't follow instructions well. Setting to 1 disables acceleration. |
| soft_fresh_weight | 0.25 | Fixed parameter, no adjustment needed. |
| tailing_step | 1 | Fixed parameter, higher values reduce speed. |
| inv_skip | 2/3/4 | Inversion step skipping interval. Default value of 2 is ok. |
| cache_type | "ours_predefine"<br>"ours_cache" | Use token index preprocessing for faster speed. "ours_cache" disables preprocessing. |

</details>


<details>

<summary>Prompt-guided editing</summary>

| Parameter | Value | Description |
|-----------|--------|-------------|
| use_cache | true | Enable caching mechanism to accelerate inference |
| num_inference_steps | 28 | Total number of denoising steps in diffusion process |
| cascade_num | 1/3/5 | Number of cascade levels for region scoring |
| fresh_ratio | 0.1 | Ratio of cache entries to refresh each step, higher value caused lower speed. |
| fresh_threshold | 2/3 | Interval for complete cache refresh |
| soft_fresh_weight | 0.25 | Weight factor for soft cache refreshing |
| tailing_step | 1 | Step interval for tailing cache updates |
| strength | 1.0 | Fixed. Overall strength of the editing effect |
| inv_skip | 2/3 | Interval for skipping inversion steps |
| eta/gamma | 0.7 | Controls strength of inversion injection during denoising |
| stop_timestep | 6 | Timestep to stop inversion influence |
| mask_timestep | 18 | Timestep to end applying editing mask |
| cache_type | "ours_predefine"<br>"ours_cache" | Cache strategy type - "ours_predefine" uses TIP while "ours_cache" does not.  |

</details>


<details>

<summary>Drag-guided editing (some are ommited)</summary>

| Parameter | Value | Description |
|-----------|--------|-------------|
| drag_class | "copy"/"cut" | Controls region handling - "copy" preserves source region latents, "cut" swaps source and target region latents |
| t_prime_ratio | 0.5 | Controls dragging strength in (0,1) range. Higher values reduce dragging strength |
| alpha | 1 | Noise blending ratio in (0,1) range applied to inversion latents |
| inv_skip | 2/3 | Interval for skipping inversion steps |
</details>

## 📝 TODO List
- Release evaluation code
- Release notebook and Hugging Face demo
- Develop more user-friendly interaction logic and experience, including Gradio interface

## 🙏 Acknowledgements
- Thanks to [ToCa](https://github.com/Shenyi-Z/ToCa) for cache implementations
- Thanks to [Diffusers](https://github.com/huggingface/diffusers) for pipeline implementations
- Thanks to [Region Drag](https://github.com/Visual-AI/RegionDrag) for dragging implementations

## 📝 BibTeX
```tex
@misc{yan2025eeditrethinkingspatial,
      title={EEdit : Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing}, 
      author={Zexuan Yan and Yue Ma and Chang Zou and Wenteng Chen and Qifeng Chen and Linfeng Zhang},
      year={2025},
      eprint={2503.10270},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.10270}, 
}
``` 

## 📧 Contact
yzx_ustc@mail.ustc.edu.cn




