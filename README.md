# Neuroprobe

<p align="center">
  <a href="https://neuroprobe.dev">
    <img src="https://github.com/azaho/neuroprobe/blob/main/website/neuroprobe_animation.gif?raw=True" alt="Neuroprobe Logo" style="height: 10em" />
  </a>
</p>

<p align="center">
    <a href="https://www.python.org/">
        <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-1f425f.svg?color=purple">
    </a>
    <a href="https://pytorch.org/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg">
    </a>
    <a href="https://mit-license.org/">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
</p>

<p align="center"><strong>Neuroprobe: Benchmark for Evaluating iEEG Foundation Models.</strong></p>
<p align="center"><strong>Evaluating Intracranial Brain Responses to Naturalistic Stimuli</strong></p>

<p align="center">
    <a href="https://neuroprobe.dev">🌐 Website</a> |
    <a href="https://www.arxiv.org/abs/2509.21671">📄 Paper</a> |
    <a href="https://github.com/azaho/neuroprobe/blob/main/examples/quickstart.ipynb">🚀 Example Usage</a> |
    <a href="https://github.com/azaho/neuroprobe/blob/main/SUBMIT.md">📤 Submit</a>
</p>

---

By **Andrii Zahorodnii¹²***, **Christopher Wang¹***, **Bennett Stankovits¹***, **Charikleia Moraitaki¹**, **Geeling Chau³**, **Andrei Barbu¹**, **Boris Katz¹**, **Ila R Fiete¹²**,

¹MIT CSAIL, CBMM  |  ²MIT McGovern Institute  |  ³Caltech  |  *Equal contribution

## Overview
Neuroprobe is a benchmark for evaluating EEG/iEEG/sEEG/ECoG foundation models and understanding how the brain processes information across multiple tasks. It analyzes intracranial recordings during naturalistic stimuli using techniques from modern natural language processing. By probing neural responses across many tasks simultaneously, Neuroprobe aims to reveal the functional organization of the brain and relationships between different cognitive processes. The benchmark includes tools for decoding neural signals using both simple linear models and advanced neural networks, enabling researchers to better understand how the brain processes information across vision, language, and audio domains.

Please see the full [technical paper](https://arxiv.org/pdf/2509.21671) for more details.

### Installation

1. Install the package:
```bash
pip install "neuroprobe @ git+https://github.com/insight-neuro/neuroprobe-runner"
```


2. Update the `ROOT_DIR_BRAINTREEBANK` environment variable to point to the location where you want to store the BrainTreebank dataset. You can for example do this by creating a `.env` file in the root of this repository with the following content:
```
ROOT_DIR_BRAINTREEBANK=/path/to/braintreebank
```

3. Download the [BrainTreebank dataset](https://braintreebank.dev/), using our [brainsets](https://github.com/insight-neuro/brainsets) fork to properly preprocess the dataset. Note that this script requires you to have [uv](https://docs.astral.sh/uv/#installation) installed.

```bash
./data.sh --lite
```

(Lite is an optional flag; if only using Neuroprobe as a benchmark, this flag will reduce the number of downloaded files by >50% by removing unnecessary files.)

### Geting Started

To get started, first check out [quickstart.ipynb](https://github.com/azaho/neuroprobe/blob/main/examples/quickstart.ipynb), which will show you how to load and examine the data.

Once you understand the data structure, check [logistic_regression_runner](https://github.com/insight-neuro/neuroprobe/blob/main/examples/logistic_regression_runner.py) for an example of how to set up a full training and evaluation pipeline using the `NeuroprobeRunner` class. 

To evaluate your own model, simply create your own `NeuroprobeRunner` class!

### Leaderboard Requirements

To submit to the Neuroprobe leaderboard, you MUST use the exact train/val/test splits that are provided by the Neuroprobe package. If you use the `NeuroprobeRunner` class to run your evaluation,
this will be provided for you by default!

## Citation

If you use Neuroprobe in your work, please cite our paper:
```bibtex
@misc{neuroprobe,
      title={Neuroprobe: Evaluating Intracranial Brain Responses to Naturalistic Stimuli}, 
      author={Andrii Zahorodnii and Christopher Wang and Bennett Stankovits and Charikleia Moraitaki and Geeling Chau and Andrei Barbu and Boris Katz and Ila R Fiete},
      year={2025},
      eprint={2509.21671},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.21671}, 
}
```
