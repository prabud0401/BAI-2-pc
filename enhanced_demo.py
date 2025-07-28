#!/usr/bin/env python3
"""
Enhanced Beverage Intelligence Demo

Uses the real processed knowledge base and data for accurate categorization.
"""

import click
import pandas as pd
import re
from pathlib import Path


def load_knowledge_base():
    """Load the real knowledge base from processed data."""
    try:
        kb_path = Path("data/03_knowledge_base/knowledge_base.csv")
        kb = pd.read_csv(kb_path)
        
        # Create brand and type lookups
        brands = set()
        types = set()
        
        for _, row in kb.iterrows():
            value = str(row.get('VALUE', '')).lower().strip()
            if value and value != 'nan':
                if 'brand' in str(row.get('TYPE', '')).lower():
                    brands.add(value)
                else:
                    types.add(value)
        
        return {
            'brands': brands,
            'types': types
        }
    except:
        return {'brands': set(), 'types': set()}


def enhanced_entity_extraction(text, knowledge_base):
    """Enhanced entity extraction using real knowledge base."""
    entities = []
    text_lower = text.lower()
    
    # Extract SIZE with more patterns
    size_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:ml|mL|ML)\b',
        r'(\d+(?:\.\d+)?)\s*(?:oz|OZ|fl\s*oz)\b',
        r'(\d+(?:\.\d+)?)\s*(?:l|L|liter|litre)\b'
    ]
    
    for pattern in size_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            size_unit = 'ml' if 'ml' in text_lower else 'oz' if 'oz' in text_lower else 'l'
            entities.append({
                'text': f"{match}{size_unit}",
                'label': 'SIZE',
                'confidence': 0.95
            })
    
    # Extract PACK_COUNT
    pack_patterns = [
        r'(\d+)-?pack\b',
        r'(\d+)pk\b',
        r'(\d+)/\d+',  # like 6/12oz
        r'(\d+)\s*x\s*\d+',  # like 6x330ml
    ]
    
    for pattern in pack_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            entities.append({
                'text': f"{match}-pack",
                'label': 'PACK_COUNT', 
                'confidence': 0.90
            })
    
    # Extract CONTAINER_TYPE
    containers = ['bottle', 'bottles', 'can', 'cans', 'keg', 'draft', 'tap']
    for container in containers:
        if container in text_lower:
            entities.append({
                'text': container.rstrip('s'),
                'label': 'CONTAINER_TYPE',
                'confidence': 0.85
            })
    
    # Extract BRAND using knowledge base
    for brand in knowledge_base['brands']:
        if brand in text_lower and len(brand) > 2:
            entities.append({
                'text': brand.title(),
                'label': 'BRAND',
                'confidence': 0.90
            })
    
    # Extract TYPE using knowledge base  
    for beverage_type in knowledge_base['types']:
        if beverage_type in text_lower and len(beverage_type) > 2:
            entities.append({
                'text': beverage_type.title(),
                'label': 'TYPE',
                'confidence': 0.88
            })
    
    # Extract ALCOHOL_CONTENT
    alcohol_patterns = [
        r'(\d+(?:\.\d+)?)\s*%\s*(?:alc|alcohol|abv)',
        r'(\d+(?:\.\d+)?)\s*(?:proof)'
    ]
    
    for pattern in alcohol_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            entities.append({
                'text': f"{match}% ABV",
                'label': 'ALCOHOL_CONTENT',
                'confidence': 0.85
            })
    
    return entities


def enhanced_categorization(entities, text):
    """Enhanced categorization using comprehensive rules."""
    entity_texts = [e['text'].lower() for e in entities]
    all_text = ' '.join(entity_texts + [text.lower()])
    
    # Beer indicators (most comprehensive)
    beer_indicators = [
        'beer', 'lager', 'ale', 'ipa', 'stout', 'pilsner', 'wheat', 'porter',
        'bud', 'budweiser', 'corona', 'heineken', 'stella', 'modelo', 'lagunitas',
        'sierra nevada', 'sam adams', 'miller', 'coors', 'guinness', 'carlsberg',
        'dos equis', 'tecate', 'pacifico', 'negro modelo'
    ]
    
    # Wine indicators
    wine_indicators = [
        'wine', 'chardonnay', 'cabernet', 'merlot', 'pinot', 'sauvignon', 'riesling',
        'kendall jackson', 'robert mondavi', 'barefoot', 'sutter home', '750ml wine',
        'vintage', 'reserve', 'estate'
    ]
    
    # Spirits indicators
    spirit_indicators = [
        'whiskey', 'whisky', 'vodka', 'rum', 'gin', 'tequila', 'bourbon', 'scotch',
        'jack daniels', 'titos', 'grey goose', 'bacardi', 'captain morgan',
        'jameson', 'johnnie walker', 'crown royal', '750ml' # 750ml often spirits
    ]
    
    # Energy drink indicators
    energy_indicators = [
        'energy', 'red bull', 'monster', 'rockstar', 'bang', '16oz can', '8.4oz'
    ]
    
    # Soft drink indicators
    soda_indicators = [
        'soda', 'cola', 'pepsi', 'coca cola', 'dr pepper', 'sprite', 'fanta',
        '20oz bottle', '2 liter', 'diet', 'zero'
    ]
    
    # Hard seltzer indicators
    seltzer_indicators = [
        'seltzer', 'white claw', 'truly', 'bud light seltzer', 'corona seltzer'
    ]
    
    # Water indicators
    water_indicators = [
        'water', 'aqua', 'spring', 'purified', 'fiji', 'evian', 'dasani'
    ]
    
    # Check categories with confidence scoring
    if any(indicator in all_text for indicator in beer_indicators):
        confidence = 0.95 if any(x in all_text for x in ['beer', 'lager', 'ale', 'ipa']) else 0.92
        return {
            'category': 'Beer',
            'confidence': confidence,
            'reason': 'Strong beer brand/type indicators found',
            'subcategory': _get_beer_subcategory(all_text)
        }
    
    elif any(indicator in all_text for indicator in seltzer_indicators):
        return {
            'category': 'Hard Seltzer',
            'confidence': 0.94,
            'reason': 'Hard seltzer brand/type indicators',
            'subcategory': 'Alcoholic Sparkling Water'
        }
    
    elif any(indicator in all_text for indicator in wine_indicators):
        return {
            'category': 'Wine',
            'confidence': 0.90,
            'reason': 'Wine variety or brand indicators',
            'subcategory': _get_wine_subcategory(all_text)
        }
    
    elif any(indicator in all_text for indicator in spirit_indicators):
        return {
            'category': 'Spirits',
            'confidence': 0.88,
            'reason': 'Spirit type or premium brand indicators',
            'subcategory': _get_spirit_subcategory(all_text)
        }
    
    elif any(indicator in all_text for indicator in energy_indicators):
        return {
            'category': 'Energy Drink',
            'confidence': 0.87,
            'reason': 'Energy drink brand and typical sizing',
            'subcategory': 'Caffeinated Beverage'
        }
    
    elif any(indicator in all_text for indicator in soda_indicators):
        return {
            'category': 'Soft Drink',
            'confidence': 0.85,
            'reason': 'Soft drink brand and size patterns',
            'subcategory': 'Carbonated Beverage'
        }
    
    elif any(indicator in all_text for indicator in water_indicators):
        return {
            'category': 'Water',
            'confidence': 0.93,
            'reason': 'Water brand or type indicators',
            'subcategory': 'Non-Alcoholic'
        }
    
    # Size-based fallback categorization
    elif '750ml' in all_text:
        return {
            'category': 'Wine/Spirits',
            'confidence': 0.75,
            'reason': '750ml typically indicates wine or spirits',
            'subcategory': 'Standard Size'
        }
    
    elif any(x in all_text for x in ['12oz', '16oz', '24oz']):
        return {
            'category': 'Beer/Beverage',
            'confidence': 0.70,
            'reason': 'Common beer/beverage sizing',
            'subcategory': 'Standard Can/Bottle'
        }
    
    return {
        'category': 'Unknown',
        'confidence': 0.50,
        'reason': 'No clear category indicators found',
        'subcategory': 'Needs Manual Review'
    }


def _get_beer_subcategory(text):
    """Determine beer subcategory."""
    if any(x in text for x in ['ipa', 'india pale ale']):
        return 'IPA'
    elif any(x in text for x in ['lager']):
        return 'Lager'
    elif any(x in text for x in ['ale', 'pale ale']):
        return 'Ale'
    elif any(x in text for x in ['stout']):
        return 'Stout'
    elif any(x in text for x in ['wheat']):
        return 'Wheat Beer'
    else:
        return 'Standard Beer'


def _get_wine_subcategory(text):
    """Determine wine subcategory."""
    if any(x in text for x in ['chardonnay']):
        return 'White Wine - Chardonnay'
    elif any(x in text for x in ['cabernet']):
        return 'Red Wine - Cabernet'
    elif any(x in text for x in ['pinot']):
        return 'Pinot'
    elif any(x in text for x in ['sauvignon']):
        return 'Sauvignon'
    else:
        return 'Table Wine'


def _get_spirit_subcategory(text):
    """Determine spirit subcategory."""
    if any(x in text for x in ['whiskey', 'whisky', 'bourbon']):
        return 'Whiskey'
    elif any(x in text for x in ['vodka']):
        return 'Vodka'
    elif any(x in text for x in ['rum']):
        return 'Rum'
    elif any(x in text for x in ['gin']):
        return 'Gin'
    elif any(x in text for x in ['tequila']):
        return 'Tequila'
    else:
        return 'Other Spirits'


@click.command()
@click.option('--text', default='Corona Extra 350ml bottle 6-pack', help='Product description to analyze')
def enhanced_predict(text):
    """Enhanced prediction using real knowledge base and comprehensive rules."""
    click.echo("🔮 Enhanced Beverage Intelligence Prediction")
    click.echo("=" * 60)
    click.echo(f"📝 Input: {text}")
    
    # Load knowledge base
    click.echo("📚 Loading knowledge base...")
    knowledge_base = load_knowledge_base()
    click.echo(f"   • {len(knowledge_base['brands'])} brands loaded")
    click.echo(f"   • {len(knowledge_base['types'])} beverage types loaded")
    
    # Extract entities
    entities = enhanced_entity_extraction(text, knowledge_base)
    
    # Categorize
    result = enhanced_categorization(entities, text)
    
    click.echo("\n🎯 Extracted Entities:")
    if entities:
        for entity in entities:
            click.echo(f"  - {entity['label']}: {entity['text']} (confidence: {entity['confidence']:.2f})")
    else:
        click.echo("  - No specific entities found")
    
    click.echo(f"\n🏷️ Predicted Category: {result['category']}")
    click.echo(f"📂 Subcategory: {result['subcategory']}")
    click.echo(f"🔍 Confidence: {result['confidence']:.2f}")
    click.echo(f"💡 Reasoning: {result['reason']}")


if __name__ == '__main__':
    enhanced_predict() 