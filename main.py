#!/usr/bin/env python3
"""
Beverage Intelligence System - Main CLI Interface

This is the central command-line interface for the Beverage Intelligence System.
It provides a unified interface for all phases of the pipeline from data processing
to model training and prediction.

Author: AI Assistant
Date: 2025
"""

import click
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import Config
from src.data_processing import DataProcessor
from src.training import ModelTrainer
from src.prediction import Predictor
from src.categorization import CategorizationEngine


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Beverage Intelligence System CLI
    
    A comprehensive system for beverage product categorization using NER and rule-based engines.
    """
    pass


@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def process_data(verbose):
    """
    Phase 1: Consolidate raw data and create knowledge base.
    
    Processes all files in data/01_raw/ and creates:
    - Unified knowledge base in data/03_knowledge_base/
    - Master product list in data/02_processed/
    """
    click.echo("🔄 Starting data processing...")
    
    try:
        processor = DataProcessor(verbose=verbose)
        
        # Process knowledge files
        click.echo("📚 Processing knowledge base files...")
        processor.process_knowledge_files()
        
        # Process invoice files
        click.echo("📋 Processing invoice files...")
        processor.process_invoice_files()
        
        click.echo("✅ Data processing completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error during data processing: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--output-format', default='json', type=click.Choice(['json', 'spacy']), 
              help='Output format for labeled data')
def create_silver_labels(verbose, output_format):
    """
    Phase 2: Create programmatic labels (Silver Standard).
    
    Uses the knowledge base to automatically label product descriptions
    and generates training data ready for human annotation.
    """
    click.echo("🏷️ Creating silver standard labels...")
    
    try:
        processor = DataProcessor(verbose=verbose)
        
        click.echo("🤖 Applying automated labeling...")
        result_file = processor.create_silver_labels(output_format=output_format)
        
        click.echo(f"✅ Silver labels created successfully!")
        click.echo(f"📁 Output file: {result_file}")
        click.echo("💡 Next step: Import this file into Label Studio for human review")
        
    except Exception as e:
        click.echo(f"❌ Error creating silver labels: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input-file', required=True, help='Path to annotated JSON file')
@click.option('--train-split', default=0.8, help='Training data proportion (default: 0.8)')
@click.option('--valid-split', default=0.1, help='Validation data proportion (default: 0.1)')
@click.option('--verbose', is_flag=True, help='Verbose output')
def prepare_gold_data(input_file, train_split, valid_split, verbose):
    """Phase 3: Convert human-annotated data to spaCy format."""
    click.echo("🥇 Preparing gold standard data...")
    
    try:
        processor = DataProcessor(verbose=verbose)
        
        result = processor.prepare_gold_data(
            input_file=input_file,
            train_split=train_split,
            valid_split=valid_split
        )
        
        click.echo("✅ Gold data preparation completed!")
        click.echo(f"📁 Files created:")
        for split, path in result.items():
            click.echo(f"   - {split}: {path}")
        
    except Exception as e:
        click.echo(f"❌ Error preparing gold data: {e}")
        return


@cli.command()
@click.option('--verbose', is_flag=True, help='Enable verbose output')
def train(verbose):
    """Phase 4: Train the NER model on CPU."""
    click.echo("🚀 Starting model training...")
    
    try:
        trainer = ModelTrainer(verbose=verbose)
        
        # CPU-optimized training settings
        trainer.train(
            batch_size=4,      # Smaller batch size for CPU
            epochs=10,         # Standard number of epochs
            learning_rate=0.001  # Standard learning rate
        )
        
        click.echo("✅ Training completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error during training: {e}")
        return


@cli.command()
@click.option('--text', help='Single text to predict')
@click.option('--file', help='Path to CSV file with product descriptions')
@click.option('--output-file', help='Path to save predictions (required for file input)')
@click.option('--model-name', default='ner_v1', help='Name of the trained model to use')
@click.option('--description-column', default='ProductDescription', 
              help='Column name containing product descriptions')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def predict(text, file, output_file, model_name, description_column, verbose):
    """
    Phase 5: Run predictions on new data.
    
    Uses the trained NER model and categorization engine to predict
    full categories for beverage product descriptions.
    """
    if not text and not file:
        click.echo("❌ Error: Must provide either --text or --file", err=True)
        sys.exit(1)
    
    if file and not output_file:
        click.echo("❌ Error: --output-file is required when using --file", err=True)
        sys.exit(1)
    
    click.echo("🔮 Running predictions...")
    
    try:
        predictor = Predictor(model_name=model_name, verbose=verbose)
        categorizer = CategorizationEngine(verbose=verbose)
        
        if text:
            # Single text prediction
            click.echo(f"📝 Input: {text}")
            
            # Get NER entities
            entities = predictor.predict_single(text)
            
            # Get full category
            category_result = categorizer.categorize(entities)
            
            click.echo("🎯 Results:")
            click.echo(f"  📊 Entities: {entities}")
            click.echo(f"  🏷️ Category: {category_result['full_category']}")
            click.echo(f"  🔍 Confidence: {category_result['confidence']:.3f}")
            
        else:
            # File prediction
            click.echo(f"📁 Processing file: {file}")
            
            results = predictor.predict_file(
                input_file=file,
                output_file=output_file,
                description_column=description_column,
                categorizer=categorizer
            )
            
            click.echo("✅ Predictions completed successfully!")
            click.echo(f"📁 Output saved to: {output_file}")
            click.echo(f"📊 Processed {results['total_predictions']} products")
            
    except Exception as e:
        click.echo(f"❌ Error during prediction: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """
    Check the status of the system and available data/models.
    """
    click.echo("📊 Beverage Intelligence System Status")
    click.echo("=" * 50)
    
    config = Config()
    
    # Check directories
    click.echo("\n📁 Directory Status:")
    for path_name, path_value in config.PATHS.items():
        exists = os.path.exists(path_value)
        status_icon = "✅" if exists else "❌"
        click.echo(f"  {status_icon} {path_name}: {path_value}")
    
    # Check for raw data
    click.echo("\n📊 Data Status:")
    raw_files = list(Path(config.PATHS['raw_data']).glob('*'))
    click.echo(f"  📄 Raw files: {len(raw_files)} files")
    
    # Check for trained models
    models_dir = Path(config.PATHS['models'])
    model_folders = [d for d in models_dir.iterdir() if d.is_dir()]
    click.echo(f"  🤖 Available models: {len(model_folders)}")
    for model in model_folders:
        click.echo(f"    - {model.name}")
    
    # Check for processed data
    processed_files = list(Path(config.PATHS['processed_data']).glob('*.csv'))
    click.echo(f"  ⚙️ Processed files: {len(processed_files)}")
    
    knowledge_files = list(Path(config.PATHS['knowledge_base']).glob('*.csv'))
    click.echo(f"  📚 Knowledge base files: {len(knowledge_files)}")


@cli.command()
@click.option('--model-name', default='ner_v1', help='Name of the model to evaluate')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def evaluate(model_name, verbose):
    """
    Evaluate the trained model on test data.
    """
    click.echo("📏 Evaluating model performance...")
    
    try:
        trainer = ModelTrainer(model_name=model_name, verbose=verbose)
        metrics = trainer.evaluate_model()
        
        click.echo("✅ Evaluation completed!")
        click.echo(f"📊 Test Set Metrics:")
        click.echo(f"  - Precision: {metrics['precision']:.3f}")
        click.echo(f"  - Recall: {metrics['recall']:.3f}")
        click.echo(f"  - F1-Score: {metrics['f1']:.3f}")
        click.echo(f"  - Loss: {metrics['loss']:.3f}")
        
        if 'entity_metrics' in metrics:
            click.echo(f"\n📋 Per-Entity Metrics:")
            for entity, scores in metrics['entity_metrics'].items():
                click.echo(f"  {entity}:")
                click.echo(f"    - Precision: {scores['precision']:.3f}")
                click.echo(f"    - Recall: {scores['recall']:.3f}")
                click.echo(f"    - F1: {scores['f1']:.3f}")
        
    except Exception as e:
        click.echo(f"❌ Error during evaluation: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli() 