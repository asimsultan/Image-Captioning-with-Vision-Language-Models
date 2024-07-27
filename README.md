# Image Captioning with Vision-Language Models

This project fine-tunes a vision-language model for image captioning using the COCO dataset.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Directory Structure](#directory-structure)
- [Results](#results)
- [Resources](#resources)
- [License](#license)

## Overview

Image captioning is a task in which a textual description is generated for a given image. In this project, we use a vision-language model from Hugging Face's transformers library to perform image captioning on the COCO dataset.

## Requirements

- Python 3.6+
- transformers
- torch
- datasets
- pillow

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt