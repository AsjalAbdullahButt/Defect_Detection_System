# Screenshots and results

A visual tour of Defect Detection. These are screenshots of the running Streamlit
app, not design mockups. The clean-start screens contain no uploaded photos,
API keys or private reports. Dataset photos and Grad-CAM panels stay local.

[Inspect](#inspect) · [Batch](#batch) · [Model & results](#model--results) ·
[Explain](#explain) · [Evaluation figures](#evaluation-figures)

## Inspect

Upload one image, inspect the predicted class and confidence, and check whether
human review is needed. Additional decision details expand below the result.

![Inspect page, with upload and result areas side by side](images/inspect.png)

<details>
<summary>Inspect on mobile</summary>

<img src="images/inspect-mobile.png" alt="Inspect page with a stacked mobile layout" width="390">

</details>

## Batch

Upload multiple images and compare the results in one table. Download the
predictions as CSV for a local review.

![Batch page ready for image uploads](images/batch.png)

## Model & results

View the connected model's settings and locally saved test metrics. Missing
reports have a clear empty state instead of an error.

![Model page before connecting an API or loading a local report](images/model.png)

## Explain

Choose a saved error example to compare the input with its Grad-CAM heatmap.
Heatmaps indicate model attention, not a precise defect boundary.

![Explain page before local explanation panels are generated](images/explain.png)

## Evaluation figures

These plots come from the completed evaluation, separately from the clean-start
UI screenshots above. See [evaluation.md](evaluation.md) for the full interpretation.

### Confusion matrices

The external capture produces substantially more errors than the official test.

![Official and external test confusion matrices](figures/test_confusion.png)

### Precision–recall and ROC curves

![Test precision–recall and ROC curves](figures/test_curves.png)

<details>
<summary>Training and calibration figures</summary>

![Training and validation curves](figures/train_curves.png)

![Calibration reliability diagram](figures/reliability.png)

</details>

## Architecture

![Serving architecture](architecture_serving.png)

<details>
<summary>Training pipeline</summary>

![Training architecture](architecture_training.png)

</details>

Return to the [README](../README.md) or the [UI setup guide](../ui/README.md).
