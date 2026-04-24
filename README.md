# Edge Defect Detector — CNN Binary Classifier

A lightweight CNN trained to classify defective vs non-defective parts,
simulating an Automated Optical Inspection (AOI) system used in industrial manufacturing.
Built from scratch with PyTorch, no pretrained models.

## Architecture
graph TD
    %% Input Layer
    In([<b>Input</b><br/>3x32x32 RGB Image]) 
    
    %% Feature Extraction
    subgraph "Feature Extraction (CNN)"
    In --> C1[<b>Conv Block 1</b><br/>16 filters, 3x3]
    C1 --> R1[ReLU]
    R1 --> P1[MaxPool 2x2]
    
    P1 --> C2[<b>Conv Block 2</b><br/>32 filters, 3x3]
    C2 --> R2[ReLU]
    R2 --> P2[MaxPool 2x2]
    end

    %% Classification
    subgraph "Classifier Head"
    P2 --> F[<b>Flatten</b><br/>2048 Units]
    F --> L1[<b>Linear</b><br/>2048 to 128]
    L1 --> R3[ReLU]
    R3 --> D[<b>Dropout</b><br/>p=0.5]
    D --> L2[<b>Output Linear</b><br/>128 to 2]
    end

    %% Output
    L2 --> Out{<b>Final Class</b><br/>Good / Defect}

    %% Styling
    style In fill:#f9f,stroke:#333,stroke-width:2px
    style Out fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#fff4dd,stroke:#d4a017

**Total parameters: 267,618 (~1MB)**

## Results

Trained for 10 epochs on 8,000 images, validated on 2,000.

| Epoch | Loss  | Val Accuracy |
|-------|-------|--------------|
| 1     | 0.597 | 75.3%        |
| 5     | 0.319 | 86.9%        |
| 10    | 0.193 | 88.7%        |
| best  | -     | **89.5%**    |

![Training Results](training_results.png)

## How to Run

pip install -r requirements.txt
python train.py

## Future Improvements

- Replace CIFAR-10 subset with real industrial dataset (MVTec AD)
- Add MMSE-based noise regularisation
- Quantise model to INT8 for deployment on Cortex-M MCU (TinyML)
- Experiment with MobileNetV2 for better accuracy/size tradeoff

Author: Pedro Trenado