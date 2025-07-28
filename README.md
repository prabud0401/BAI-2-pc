# Beverage Intelligence System

A comprehensive end-to-end system for beverage product categorization using Named Entity Recognition (NER) and rule-based categorization engines.

## 🎯 System Overview

The Beverage Intelligence System processes raw beverage invoice files and accurately predicts comprehensive categories for product descriptions using a two-stage architecture:

1. **Stage 1: NER Model** - Custom deep-learning model (spaCy + transformers) that extracts entities like BRAND, TYPE, SIZE, PACK_COUNT, etc.
2. **Stage 2: Categorization Engine** - Rule-based system that applies business logic to determine final categories (Beer, Wine, Soft Drink, etc.)

## 🏗️ Project Structure

```
Beverage_AI_System/
│
├── 📂 data/
│   ├── 01_raw/                 # Drop source CSV/XLSX files here
│   ├── 02_processed/           # Intermediate, cleaned data
│   ├── 03_knowledge_base/      # Consolidated keyword library
│   ├── 04_training_data/       # Machine-labeled data ("silver standard")
│   ├── 05_annotated_data/      # Human-corrected data ("gold standard")
│   └── 06_model_ready/         # Final train/valid/test .spacy files
│
├── 📂 src/
│   ├── __init__.py
│   ├── config.py               # Central configuration
│   ├── data_processing.py      # Data consolidation and labeling
│   ├── training.py             # Model training with GPU support
│   ├── prediction.py           # Inference engine
│   └── categorization.py       # Rule-based categorization
│
├── 📂 models/
│   └── ner_v1/                 # Trained model artifacts
│
├── main.py                     # Main CLI interface
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or create the project directory
cd Beverage_AI_System

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_trf
```

### 2. Prepare Your Data

Place your files in the `data/01_raw/` directory:
- **Invoice files**: CSV/XLSX files containing product descriptions
- **Knowledge files**: Optional CSV/XLSX files with keywords, brands, etc. (use patterns like `*brand*`, `*type*`, `*category*`)

### 3. Run the Complete Pipeline

```bash
# Check system status
python main.py status

# Phase 1: Process and consolidate data
python main.py process-data

# Phase 2: Create automated labels
python main.py create-silver-labels

# Phase 3: After human annotation in Label Studio
python main.py prepare-gold-data --input-file data/05_annotated_data/your_annotations.json

# Phase 4: Train the model (with GPU support)
python main.py train --epochs 10 --batch-size 8

# Phase 5: Run predictions
python main.py predict --text "Budweiser 12oz 6-pack bottles"
python main.py predict --file input.csv --output-file predictions.csv
```

## 📋 Detailed Usage

### CLI Commands

#### System Status
```bash
python main.py status
```
Shows directory status, available data files, and trained models.

#### Data Processing
```bash
# Process all raw data
python main.py process-data --verbose

# Create silver standard labels
python main.py create-silver-labels --output-format json
```

#### Model Training
```bash
# Train with default settings
python main.py train

# Train with custom parameters
python main.py train --model-name ner_v2 --epochs 15 --batch-size 16 --learning-rate 0.001 --gpu

# Evaluate trained model
python main.py evaluate --model-name ner_v1
```

#### Prediction
```bash
# Single text prediction
python main.py predict --text "Corona Extra 12oz bottle 6-pack"

# Batch file prediction
python main.py predict --file products.csv --output-file results.csv --description-column ProductDescription
```

### Phase-by-Phase Workflow

#### Phase 1: Data Consolidation
- Processes all files in `data/01_raw/`
- Creates unified knowledge base in `data/03_knowledge_base/`
- Generates master product list in `data/02_processed/`

#### Phase 2: Silver Standard Creation
- Uses knowledge base and regex patterns for automated labeling
- Exports JSON format compatible with Label Studio
- Handles overlapping entities intelligently

#### Phase 3: Gold Standard Preparation
- Import silver standard into Label Studio for human review
- Export corrected annotations
- Converts to spaCy format with train/validation/test splits

#### Phase 4: Model Training
- Supports GPU acceleration (automatically detected)
- Implements early stopping and validation monitoring
- Saves best model with training history

#### Phase 5: Prediction Pipeline
- Loads trained model for inference
- Applies rule-based categorization engine
- Supports both single predictions and batch processing

## 🔧 Configuration

The system is highly configurable through `src/config.py`:

### NER Entities
- **BRAND**: Brand names (e.g., "Coca-Cola", "Budweiser")
- **TYPE**: Product types (e.g., "Beer", "Wine", "Soda")
- **SIZE**: Volume/size (e.g., "12oz", "750ml")
- **PACK_COUNT**: Pack quantities (e.g., "6-pack", "24-case")
- **CONTAINER_TYPE**: Container types (e.g., "bottle", "can")
- **ALCOHOL_CONTENT**: ABV percentages (e.g., "5.2%")
- **FLAVOR**: Flavor profiles (e.g., "IPA", "Cherry")
- **SPECIAL_TYPE**: Special attributes (e.g., "Organic", "Light")

### Category Rules
Predefined rules for categorizing products into:
- Beer
- Wine  
- Soft Drink
- RTD Cocktail
- Energy Drink
- Water
- Juice
- Sports Drink

## 🎯 Model Performance

The system uses state-of-the-art transformer models (`en_core_web_trf`) for NER, providing:
- High accuracy entity extraction
- GPU acceleration for fast training
- Robust handling of varied product descriptions
- Confidence scoring for predictions

## 🔄 Human-in-the-Loop Workflow

### Label Studio Integration
1. Export silver standard JSON from the system
2. Import into Label Studio for human review
3. Quickly correct automated labels (correction vs. creation)
4. Export gold standard annotations
5. Feed back into training pipeline

### Quality Assurance
- Validation metrics for model performance
- Confidence scoring for predictions
- Detailed reasoning for categorization decisions

## 🐍 Python API Usage

```python
from src.prediction import Predictor
from src.categorization import CategorizationEngine

# Load trained model
predictor = Predictor(model_name='ner_v1')
categorizer = CategorizationEngine()

# Single prediction
result = predictor.predict_single("Heineken 12oz bottle 6-pack")
entities = result['grouped_entities']

# Get final category
category_result = categorizer.categorize(entities)
print(f"Category: {category_result['full_category']}")
print(f"Confidence: {category_result['confidence']:.2f}")
```

## 🔧 Advanced Features

### GPU Support
- Automatic GPU detection and activation
- Fallback to CPU if GPU unavailable
- Significant training speed improvements

### Batch Processing
- Efficient batch prediction for large datasets
- Memory-optimized processing
- Progress tracking and logging

### Extensible Rules Engine
- Easy addition of custom categorization rules
- Category-specific business logic
- Alcohol content and size validation

### Comprehensive Logging
- Detailed logging throughout the pipeline
- Configurable verbosity levels
- Training history and metrics tracking

## 📊 Example Outputs

### Entity Extraction
```json
{
  "entities": [
    {"text": "Budweiser", "label": "BRAND", "confidence": 0.95},
    {"text": "12oz", "label": "SIZE", "confidence": 0.92},
    {"text": "6-pack", "label": "PACK_COUNT", "confidence": 0.88},
    {"text": "bottles", "label": "CONTAINER_TYPE", "confidence": 0.90}
  ]
}
```

### Categorization Result
```json
{
  "full_category": "Beer",
  "confidence": 0.87,
  "reason": "Brand matched: budweiser; Common beer pack size: 6-pack"
}
```

## 🔍 Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **spaCy Model Missing**
   ```bash
   python -m spacy download en_core_web_trf
   ```

3. **Memory Issues During Training**
   ```bash
   # Reduce batch size
   python main.py train --batch-size 4
   ```

### Performance Optimization

- Use GPU for training (2-5x speedup)
- Optimize batch sizes based on available memory
- Use validation-based early stopping
- Process large files in chunks

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Contact the development team
- Check the documentation in `src/` modules

---

**Built with ❤️ for beverage industry intelligence** 