# Deep learning-based automatic field of view planning for prostate MRI in oblique coronal and oblique axial planes

This repository contains the official code for:
Quinsten et al. 'Deep learning-based automatic field of view planning for prostate MRI in oblique coronal and oblique axial planes', Scientific Reports, 2026, DOI: 10.1038/s41598-026-52248-6

If you use this model, please cite the publication above.


## Environment Setup
To ensure the environment is rebuilt 1:1, use the provided `environment.yml` file. This file captures both Conda and Pip dependencies to avoid version conflicts. Detailed instructions can be found in the internet, but it should be along these lines:

```bash
# Create environment from file
conda env create -f environment.yml
conda activate open-mmlab
```

### Note on mmrotate

The directory `./mmrotate` is included for reference only. The original library is deprecated and versioning is unstable, so the local copy is maintained here to ensure the code remains functional.


## Data Preparation

#### Download PICAI

Move all data to `./data/train/PICAI`.
Remove all non sagittal images. Each folder should contain only one file, e.g., `./data/train/PICAI/11473/11473_1001497_sag.mha`. For reference, the file `_picai_data.txt` lists the specific data used in this study.

#### Test Data

Copy test data to `./data/test/<center>/<accNr>/<series>`. For `.mha` files, use `./data/test/<center>/<accNr>/<series>.mha`. Ensure only one file exists per series.


## Annotation Workflow

Start the annotation process by specifying the cohort:

Training: `python3 ./annotate.py --cohort train` (Results saved to `./annotations/train`)

Testing: `python3 ./annotate.py --cohort test.<center>` (Results saved to `./annotations/test.<center>`)

The annotation tool opens an interactive viewer. Draw the FOV rectangle by clicking and dragging the mouse. The coronal box is shown in red, the transverse box in cyan. Shortcuts:

- `x` / `z` — next / previous slice
- `q` / `e` — shrink / grow coronal box
- `a` / `d` — shrink / grow transverse box
- `i` / `k` — move coronal box up / down
- `j` / `l` — move coronal box left / right
- `t` / `g` — move transverse box up / down
- `f` / `h` — move transverse box left / right
- `m` / `n` — rotate both boxes clockwise / counter-clockwise
- `,` — copy current annotation
- `.` — paste copied annotation to current frame
- `Delete` — remove annotation from current frame
- `Space` — save and advance to next scan
- `` ` `` — save and go to previous scan
- `7` / `9` — jump 25 scans backward / forward
- `o` / `p` — cycle CLAHE contrast / tile settings
- `Esc` — save and quit


## Training and Model Selection

### Training

Both stages are initiated via `run.sh` scripts, e.g., for Stage 1:

```bash
cd ./network.stage1/
./run.sh
```

### Model Selection

Run `modelSelection.py` for either stage. This script parses all `.pkl` files generated during testing to identify the top performing model. The selected model is automatically copied to `./standalone/models`.


## Inference and Stand-alone Predictor

If you prefer, you can predict without training by using the trained models of our study.
Because the model files exceed GitHub size limits, download them from Zenodo: [https://zenodo.org/records/20082586](https://zenodo.org/records/20082586). Place these in `./standalone/model` before running predictions.

### Predictions

Change directory to `./standalone/run.sh` and execute `./run.sh`. This will start predicting all test data in `./data/test/<center>`. This data does not need to be annotated, if it is not, then the metrics will not be computed. After prediction is finished, there will be `./data/test/<center>/stats.csv` with all predictions as a .CSV file, and for each scan there will be a .PNG file at `./data/test/<center>/<accID>/Final_Slice_Prediction.png`. This will show the selected slice and the predicted FOV.


## Evaluation and Analysis

### Intra/Inter-rater Variability

Code for rater analysis is located in `./interrater`. For this, the raters must annotate 100 scans. The script uses the training data; in the study however test data was used. Annotate by passing the rater name and stage:

`python3 ./interrater_annotate.py --rater ANTON --stage 1`
`python3 ./interrater_annotate.py --rater ANTON --stage 2`

Stage 1 lets the rater freely scroll slices and place both boxes. Stage 2 locks the slice to the ground-truth frame so only the box placement is rated. The keyboard shortcuts are identical to `annotate.py`, except that slice navigation (`x` / `z`) is disabled in stage 2.

To compute inter-rater metrics, use suffixes `_a` and `_b`, like `ANTON_a` and `ANTON_b`. The annotations will be evaluated by calling `./eval_interrater.py`.


### Clinical Usefulness

The script `standalone_clinical_usefulness.py` shows the predicted slice and FOV overlay for each scan and lets a rater flag errors. Call it for each rater and center:

`python3 ./standalone_clinical_usefulness.py --rater ANTON --split uke`

where `split` is the center name, e.g. `uke` for `./data/test/uke/`. Shortcuts:

- `q` — toggle wrong slice selection
- `e` — toggle bad FOV annotation
- `Space` — next scan
- `b` — previous scan
- `7` / `9` — jump 25 scans backward / forward
- `Esc` — save and quit

The final results can be generated by calling `python3 ./evaluate.py`.


### Sensitivity Analysis

This analysis measures model robustness by modifying noise and contrast for 30 scans per center. First, the modified images need to be generated: `./prepare_sensivitiy.py`. This will create for each center new test data, e.g., if `./data/test/uke` is a center, this will generate `./data/test/uke_sensitivity`. Then, the stage 2 inference needs to be called like above. One can do this by calling `run_sensitivity.sh` in the `./standalone` folder. Afterwards the evaluation can be started by calling `evaluate_sensitivity.py` (within the sensitivity folder).


## Figures and Visualizations

Scripts to generate sample figures and histograms used in the publication are available in `./figures`.


## License

MIT License

Copyright (c) 2026, Aydin Demircioglu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
