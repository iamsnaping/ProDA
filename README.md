---
annotations_creators:
- crowdsourced
language:
- en
language_creators:
- expert-generated
license:
- cc-by-nc-4.0
multilinguality:
- monolingual
pretty_name: SportsHHI
size_categories: []
source_datasets:
- original
tags:
- video
- video relation detection
- human-human interaction detection
task_categories:
- image-classification
- object-detection
- other
task_ids:
- multi-class-image-classification
extra_gated_heading: "Acknowledge license to accept the repository"
extra_gated_prompt: "This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License"
extra_gated_fields:
  Institute: text
  I want to use this dataset for:
    type: select
    options:
      - Research
      - Education
      - label: Other
        value: other
  I agree to use this dataset for non-commerical use ONLY: checkbox

---

# Dataset Card for SportsHHI

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)

## Dataset Description

- **Repository:** https://github.com/MCG-NJU/SportsHHI
- **Paper:** https://arxiv.org/abs/2404.04565
- **Point of Contact:** mailto: runyu_he@smail.nju.edu.cn

### Dataset Summary

SportsHHI is a dataset for video human-human interaction detection. Please refer to [SportsHHI: A Dataset for Human-Human Interaction Detection in Sports Videos](https://arxiv.org/abs/2404.04565) for more details. Please refer to [this repository](https://github.com/MCG-NJU/SporsHHI) for training and evaluation.

### Supported Tasks and Leaderboards

- `Human-Human Interaction Detection`

Details about training and evaluation can be found in the [GitHub Repository](https://github.com/mcG-NJU/SportsHHI).

### Languages

The class labels in the dataset are in English.

## Dataset Structure

The dataset contains ```frames.zip``` and ```annotations```.

```frames.zip``` contains extracted video frames.

```annotations``` contains annotaions about human boxes, interactions, and interaction classes.

If you have any network issues, please refer to ```pan.txt``` for data downloading.

### Data Splits

|             |train  |validation|
|-------------|------:|---------:|
|# of tubes   |38527  |12122     |

## Dataset Creation

### Curation Rationale

comprehending high-level inter- actions between humans is crucial for understanding com- plex multi-person videos, such as sports and surveillance videos. To address this issue, authors propose a new video visual relation detection task: video human-human interaction detection, and build a dataset named SportsHHI for it.

### Source Data

#### Initial Data Collection and Normalization

> We carefully selected 80 basketball and 80 volleyball videos from the MultiSports dataset to cover various types of games including men’s, women’s, national team, and club games. The average length of the videos is 603 frames and the frame rate of the videos is 25FPS. All videos have a high resolution of 720P.

### Annotations

Please refer to [SportsHHI: A Dataset for Human-Human Interaction Detection in Sports Videos](https://arxiv.org/abs/2404.04565) for more information.

## Additional Information

### Dataset Curators

Authors of [SportsHHI: A Dataset for Human-Human Interaction Detection in Sports Videos](https://arxiv.org/abs/2404.04565)

- Tao Wu

- Runyu He

- Gangshan Wu

- Limin Wang

### Licensing Information

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

### Citation Information

If you find this dataset useful, please cite as

```
@misc{wu2024sportshhi,
      title={SportsHHI: A Dataset for Human-Human Interaction Detection in Sports Videos}, 
      author={Tao Wu and Runyu He and Gangshan Wu and Limin Wang},
      year={2024},
      eprint={2404.04565},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
