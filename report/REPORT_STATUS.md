# Lab 3 report status

This directory is a separate copy of template/LaTeX. The original template and outputs/ directories were not edited. The report uses copied inputs under photos/ (30 PNG images) and data/ (six JSON files plus a snapshot of the implementation).

## Available and included

- Ten evaluation prompts for each of SDS, VSD, and SDI: 30 valid PNG images.
- Per-prompt and average CLIP ViT-B/32 scores for all three methods.
- The three surviving run_config.json files, each written by the final prompt invocation in its output directory.
- Method descriptions based on the implementation in data/guidance_sd.py.
- Quantitative and visual comparisons that reflect the updated VSD run (mean CLIP 0.2825, guidance 7.5, LoRA loss weight 1.2).
- Author name and student ID are present in the title block.

## Still required by the assignment

1. SDS guidance-scale ablation: settings 7.5, 15, 25, 50, 100. Supply comparable CLIP scores and sample images. The template's planned grid uses burger, cat, and villa prompts.
2. SDI inversion-step ablation: settings 1, 3, 5, 10, 25. Supply comparable CLIP scores, sample images, and runtimes for the cost-quality discussion.
3. SDI target-update-interval ablation: settings 25, 75, 200, 500, 1000. Supply comparable CLIP scores and sample images.
4. Original diffusion-loss control: directly optimize the original denoising loss, then supply its CLIP score and visual results. The template's planned visual comparison uses dog, castle, and car prompts.
The report follows the original template's title-and-teaser layout, five-panel figure page, result tables, and section order. Measured images and scores are shown where available. Unmeasured ablation and diffusion-loss panels say "Not measured"; their table cells are dashes rather than invented values. The article text does not expose data paths or template bookkeeping.

## Interpretation and provenance limits

- The updated VSD run uses guidance 7.5 and LoRA loss weight 1.2.
- The recorded SDI run uses 500 steps and learning rate 0.01, not the suggested 1000 steps and 0.005.
- run_config.json is overwritten at each prompt invocation by main.py; the surviving file cannot individually verify the other nine images' settings.
- The implementation accepts inversion_eta=0.3 but the inversion loop follows deterministic DDIM inversion and does not use this parameter.
- Do not present VSD as better than SDS for this experiment: its measured mean CLIP score is lower, and its images show texture artifacts.

## PDF build status

The LaTeX source files have been prepared, but this host has no TeX Live/MacTeX installation. A forced bundled Tectonic build stopped at the CVPR template's times.sty dependency because the package bundle could not be downloaded. No report/main.pdf is included, so there is no stale template PDF to mistake for the new report. After a TeX installation is available, compile main.tex from this directory and inspect the resulting pages before submission.
