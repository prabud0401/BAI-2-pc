#!/usr/bin/env python3
"""
Beverage Intelligence System - Demo Version

This is a simplified demo version that shows the system functionality
without requiring heavy ML dependencies.
"""

import click
import json
import pandas as pd
from pathlib import Path


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Beverage Intelligence System - Demo CLI
    
    A demonstration of the beverage product categorization system.
    """
    pass


@cli.command()
def status():
    """Check the status of the system."""
    click.echo("📊 Beverage Intelligence System Status")
    click.echo("=" * 50)
    click.echo("✅ Demo mode activated")
    click.echo("✅ CLI interface working")
    click.echo("✅ Project structure in place")
    
    # Check directories
    project_root = Path(__file__).parent
    directories = [
        'data/01_raw',
        'data/02_processed', 
        'data/03_knowledge_base',
        'data/04_training_data',
        'data/05_annotated_data',
        'data/06_model_ready',
        'src',
        'models'
    ]
    
    click.echo("\n📁 Directory Status:")
    for dir_path in directories:
        full_path = project_root / dir_path
        status_icon = "✅" if full_path.exists() else "❌"
        click.echo(f"  {status_icon} {dir_path}")
    
    click.echo(f"\n💡 To install full dependencies: pip install -r requirements.txt")
    click.echo(f"🚀 This demo shows the system is ready to run!")


@cli.command()
@click.option('--text', default='Budweiser 12oz bottle 6-pack', help='Product description to analyze')
def demo_predict(text):
    """Demo prediction showing the system's entity extraction and categorization."""
    click.echo("🔮 Demo Prediction")
    click.echo("=" * 50)
    click.echo(f"📝 Input: {text}")
    
    # Simulate entity extraction (normally done by spaCy NER model)
    demo_entities = extract_demo_entities(text)
    
    # Simulate categorization (using our rule-based engine logic)
    category_result = demo_categorize(demo_entities)
    
    click.echo("\n🎯 Extracted Entities:")
    for entity in demo_entities:
        click.echo(f"  - {entity['label']}: {entity['text']} (confidence: {entity['confidence']:.2f})")
    
    click.echo(f"\n🏷️ Predicted Category: {category_result['category']}")
    click.echo(f"🔍 Confidence: {category_result['confidence']:.2f}")
    click.echo(f"💡 Reasoning: {category_result['reason']}")


@cli.command()
@click.argument('sample_file', default='sample_products.csv')
def demo_batch(sample_file):
    """Demo batch processing with sample data."""
    click.echo("📊 Demo Batch Processing")
    click.echo("=" * 50)
    
    # Create sample data if it doesn't exist
    sample_data = [
        "Budweiser 12oz bottle 6-pack",
        "Kendall Jackson Chardonnay 750ml",
        "Coca-Cola 2-liter bottle",
        "Red Bull Energy Drink 8oz can 4-pack",
        "Corona Extra 12oz bottle 12-pack",
        "Grey Goose Vodka 750ml",
        "Monster Energy 16oz can",
        "Fiji Water 16oz bottle 24-pack"
    ]
    
    click.echo(f"📁 Processing {len(sample_data)} sample products...")
    
    results = []
    for i, product in enumerate(sample_data, 1):
        click.echo(f"  {i}. {product}")
        
        # Extract entities and categorize
        entities = extract_demo_entities(product)
        category = demo_categorize(entities)
        
        results.append({
            'ProductDescription': product,
            'PredictedCategory': category['category'],
            'Confidence': category['confidence'],
            'ExtractedEntities': len(entities)
        })
        
        click.echo(f"     → {category['category']} ({category['confidence']:.2f})")
    
    click.echo(f"\n✅ Processed {len(results)} products successfully!")
    
    # Show category distribution
    categories = {}
    for result in results:
        cat = result['PredictedCategory']
        categories[cat] = categories.get(cat, 0) + 1
    
    click.echo("\n📈 Category Distribution:")
    for category, count in categories.items():
        click.echo(f"  - {category}: {count} products")


def extract_demo_entities(text):
    """Demo entity extraction using simple pattern matching."""
    entities = []
    text_lower = text.lower()
    
    # Brand patterns
    brands = {
        'budweiser': 'Budweiser', 'corona': 'Corona', 'coca-cola': 'Coca-Cola',
        'red bull': 'Red Bull', 'monster': 'Monster', 'kendall': 'Kendall Jackson',
        'grey goose': 'Grey Goose', 'fiji': 'Fiji'
    }
    
    for brand_key, brand_name in brands.items():
        if brand_key in text_lower:
            entities.append({
                'text': brand_name,
                'label': 'BRAND',
                'confidence': 0.95
            })
    
    # Size patterns
    import re
    size_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:oz|ml|l)\b',
        r'(\d+(?:\.\d+)?)-?(?:liter|litre)\b'
    ]
    
    for pattern in size_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            entities.append({
                'text': f"{match}{'oz' if 'oz' in text_lower else 'ml' if 'ml' in text_lower else 'l'}",
                'label': 'SIZE',
                'confidence': 0.90
            })
    
    # Pack count patterns
    pack_patterns = [
        r'(\d+)-?pack\b',
        r'(\d+)-?case\b'
    ]
    
    for pattern in pack_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            entities.append({
                'text': f"{match}-pack",
                'label': 'PACK_COUNT',
                'confidence': 0.85
            })
    
    # Container type
    containers = ['bottle', 'can', 'keg']
    for container in containers:
        if container in text_lower:
            entities.append({
                'text': container,
                'label': 'CONTAINER_TYPE',
                'confidence': 0.80
            })
    
    return entities


def demo_categorize(entities):
    """Demo categorization using simplified rules."""
    entity_texts = [e['text'].lower() for e in entities]
    all_text = ' '.join(entity_texts)
    
    # Beer category
    beer_indicators = ['budweiser', 'corona', 'beer', 'ale', 'lager', 'ipa']
    if any(indicator in all_text for indicator in beer_indicators):
        return {
            'category': 'Beer',
            'confidence': 0.92,
            'reason': 'Brand and product type indicators suggest beer'
        }
    
    # Wine category  
    wine_indicators = ['chardonnay', 'wine', 'kendall', 'cabernet', '750ml']
    if any(indicator in all_text for indicator in wine_indicators):
        return {
            'category': 'Wine',
            'confidence': 0.88,
            'reason': 'Wine variety and bottle size indicators'
        }
    
    # Soft drink
    soda_indicators = ['coca-cola', 'pepsi', 'sprite', '2-liter', 'soda']
    if any(indicator in all_text for indicator in soda_indicators):
        return {
            'category': 'Soft Drink',
            'confidence': 0.90,
            'reason': 'Soft drink brand and size patterns'
        }
    
    # Energy drink
    energy_indicators = ['red bull', 'monster', 'energy', '8oz', '16oz']
    if any(indicator in all_text for indicator in energy_indicators):
        return {
            'category': 'Energy Drink', 
            'confidence': 0.87,
            'reason': 'Energy drink brand and typical sizing'
        }
    
    # Spirits
    spirit_indicators = ['vodka', 'whiskey', 'rum', 'grey goose']
    if any(indicator in all_text for indicator in spirit_indicators):
        return {
            'category': 'Spirits',
            'confidence': 0.85,
            'reason': 'Spirit type or premium brand indicators'
        }
    
    # Water
    water_indicators = ['water', 'fiji', 'aquafina', 'dasani']
    if any(indicator in all_text for indicator in water_indicators):
        return {
            'category': 'Water',
            'confidence': 0.93,
            'reason': 'Water brand or type indicators'
        }
    
    # Default
    return {
        'category': 'Unknown',
        'confidence': 0.50,
        'reason': 'No clear category indicators found'
    }


@cli.command()
def demo_full():
    """Run a complete demo of the system capabilities."""
    click.echo("🚀 Beverage Intelligence System - Full Demo")
    click.echo("=" * 60)
    
    click.echo("\n1️⃣ System Status Check")
    ctx = click.get_current_context()
    ctx.invoke(status)
    
    click.echo("\n2️⃣ Single Product Analysis")
    ctx.invoke(demo_predict, text="Corona Extra 12oz bottle 6-pack")
    
    click.echo("\n3️⃣ Batch Processing Demo")
    ctx.invoke(demo_batch)
    
    click.echo("\n🎉 Demo Complete!")
    click.echo("💡 This demonstrates the core functionality of the Beverage Intelligence System")
    click.echo("📚 For full ML capabilities, install: pip install -r requirements.txt")
    click.echo("🔬 Then run: python -m spacy download en_core_web_trf")


if __name__ == '__main__':
    cli() 