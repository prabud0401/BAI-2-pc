"""
Central Configuration for Beverage Intelligence System

This module contains all configuration settings, file paths, 
entity definitions, and other constants used throughout the system.
"""

import os
from pathlib import Path


class Config:
    """Central configuration class for the Beverage Intelligence System."""
    
    def __init__(self):
        # Get the project root directory (parent of src)
        self.PROJECT_ROOT = Path(__file__).parent.parent
        
        # Define all file paths
        self.PATHS = {
            'raw_data': self.PROJECT_ROOT / 'data' / '01_raw',
            'processed_data': self.PROJECT_ROOT / 'data' / '02_processed',
            'knowledge_base': self.PROJECT_ROOT / 'data' / '03_knowledge_base',
            'training_data': self.PROJECT_ROOT / 'data' / '04_training_data',
            'annotated_data': self.PROJECT_ROOT / 'data' / '05_annotated_data',
            'model_ready': self.PROJECT_ROOT / 'data' / '06_model_ready',
            'models': self.PROJECT_ROOT / 'models'
        }
        
        # Ensure all directories exist
        for path in self.PATHS.values():
            path.mkdir(parents=True, exist_ok=True)
    
    # Named Entity Recognition (NER) Configuration
    NER_ENTITIES = [
        'BRAND',           # Brand name (e.g., "Coca-Cola", "Budweiser")
        'TYPE',            # Product type (e.g., "Beer", "Wine", "Soda")
        'SIZE',            # Volume/size (e.g., "12oz", "750ml", "2L")
        'PACK_COUNT',      # Number in pack (e.g., "6-pack", "24-case")
        'CONTAINER_TYPE',  # Container (e.g., "bottle", "can", "box")
        'ALCOHOL_CONTENT', # ABV (e.g., "5.2%", "40% ABV")
        'FLAVOR',          # Flavor profile (e.g., "IPA", "Cherry", "Vanilla")
        'SPECIAL_TYPE'     # Special characteristics (e.g., "Organic", "Light", "Draft")
    ]
    
    # Standard column names for data processing
    STANDARD_COLUMNS = {
        'description': 'ProductDescription',
        'category': 'Category',
        'subcategory': 'Subcategory', 
        'brand': 'Brand',
        'price': 'Price',
        'quantity': 'Quantity',
        'upc': 'UPC',
        'sku': 'SKU'
    }
    
    # Common column name variations found in invoice files
    COLUMN_MAPPING = {
        # Description variations (PRIMARY - MUST MATCH)
        'product_description': 'ProductDescription',
        'Product Description': 'ProductDescription',  # Fintech format
        'PRODUCT_DESCRIPTION': 'ProductDescription',
        'description': 'ProductDescription',
        'item_description': 'ProductDescription',
        'product_name': 'ProductDescription',
        'item_name': 'ProductDescription',
        'desc': 'ProductDescription',
        'Product Name': 'ProductDescription',
        'Item Description': 'ProductDescription',
        
        # Brand variations
        'brand_name': 'Brand',
        'Brand Name': 'Brand',
        'manufacturer': 'Brand',
        'supplier': 'Brand',
        'Manufacturer': 'Brand',
        'Supplier': 'Brand',
        
        # Category variations
        'product_category': 'Category',
        'Product Category': 'Category',
        'category': 'Category',
        'Category': 'Category',
        'item_category': 'Category',
        'type': 'Category',
        'Type': 'Category',
        'wine_type': 'Category',
        'Wine Type': 'Category',
        
        # Price variations
        'unit_price': 'Price',
        'Unit Price': 'Price',
        'unit_cost': 'Price',
        'Unit Cost': 'Price',
        'cost': 'Price',
        'amount': 'Price',
        'extended_price': 'Price',
        'Extended Price': 'Price',
        
        # Quantity variations
        'qty': 'Quantity',
        'quantity': 'Quantity',
        'Quantity': 'Quantity',
        'units': 'Quantity',
        'count': 'Quantity',
        
        # UPC variations
        'upc': 'UPC',
        'UPC': 'UPC',
        'upc_number': 'UPC',
        'UPC Number': 'UPC',
        'pack_upc': 'PackUPC',
        'Pack UPC': 'PackUPC',
        
        # Size variations (NEW)
        'unit_size_ml': 'SizeML',
        'Unit Size (ML)': 'SizeML',
        'unit_size_oz': 'SizeOZ',
        'Unit Size (Oz)': 'SizeOZ',
        'ml_conversion': 'MLConversion',
        'ML Conversion': 'MLConversion',
        'oz_conversion': 'OZConversion',
        'Oz Conversion': 'OZConversion',
        
        # Additional useful columns
        'product_number': 'ProductNumber',
        'Product Number': 'ProductNumber',
        'invoice_number': 'InvoiceNumber',
        'Invoice Number': 'InvoiceNumber',
        'distributor_name': 'DistributorName',
        'Distributor Name': 'DistributorName',
        'retailer_name': 'RetailerName',
        'Retailer Name': 'RetailerName',
        'gl_code': 'GLCode',
        'GL Code': 'GLCode',
        'item_tags': 'ItemTags',
        'Item Tags': 'ItemTags',
    }
    
    # File extensions to process
    SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls']
    
    # Knowledge base file patterns to look for
    KNOWLEDGE_FILE_PATTERNS = [
        '*brand*',
        '*type*', 
        '*category*',
        '*flavor*',
        '*container*',
        '*size*',
        '*variation*',
        '*keyword*'
    ]
    
    # Full category mappings and rules
    CATEGORY_RULES = {
        'Beer': {
            'keywords': ['beer', 'ale', 'lager', 'ipa', 'stout', 'porter', 'pilsner', 'wheat', 'amber'],
            'brands': ['budweiser', 'miller', 'coors', 'corona', 'heineken', 'stella', 'guinness'],
            'containers': ['bottle', 'can', 'keg', 'growler'],
            'sizes': ['12oz', '16oz', '22oz', '32oz', '40oz']
        },
        'Wine': {
            'keywords': ['wine', 'chardonnay', 'cabernet', 'merlot', 'pinot', 'sauvignon', 'riesling'],
            'brands': ['kendall', 'robert', 'mondavi', 'beringer', 'sutter'],
            'containers': ['bottle', '750ml', '1.5l'],
            'sizes': ['750ml', '1.5l', '3l', '5l']
        },
        'Soft Drink': {
            'keywords': ['soda', 'cola', 'pepsi', 'sprite', 'fanta', 'dr pepper', 'mountain dew'],
            'brands': ['coca-cola', 'pepsi', 'sprite', 'fanta', 'dr pepper'],
            'containers': ['bottle', 'can', '2-liter'],
            'sizes': ['12oz', '16oz', '20oz', '2l']
        },
        'RTD Cocktail': {
            'keywords': ['cocktail', 'margarita', 'mojito', 'cosmopolitan', 'ready-to-drink', 'rtd'],
            'brands': ['jose cuervo', 'bacardi', 'captain morgan'],
            'containers': ['bottle', 'can'],
            'sizes': ['12oz', '16oz', '750ml']
        },
        'Energy Drink': {
            'keywords': ['energy', 'red bull', 'monster', 'rockstar', 'nos'],
            'brands': ['red bull', 'monster', 'rockstar', 'nos', '5-hour'],
            'containers': ['can', 'bottle'],
            'sizes': ['8oz', '12oz', '16oz', '20oz']
        },
        'Water': {
            'keywords': ['water', 'spring', 'purified', 'distilled', 'sparkling'],
            'brands': ['aquafina', 'dasani', 'fiji', 'evian', 'poland spring'],
            'containers': ['bottle', 'gallon', 'case'],
            'sizes': ['16oz', '20oz', '1l', '1gal']
        },
        'Juice': {
            'keywords': ['juice', 'orange', 'apple', 'grape', 'cranberry', 'tomato'],
            'brands': ['tropicana', 'minute maid', 'ocean spray', 'welchs'],
            'containers': ['bottle', 'carton', 'jug'],
            'sizes': ['12oz', '16oz', '32oz', '64oz']
        },
        'Sports Drink': {
            'keywords': ['gatorade', 'powerade', 'sports', 'electrolyte'],
            'brands': ['gatorade', 'powerade', 'vitaminwater'],
            'containers': ['bottle'],
            'sizes': ['12oz', '20oz', '32oz']
        }
    }
    
    # Regex patterns for entity extraction
    REGEX_PATTERNS = {
        'SIZE': [
            r'\b(\d+(?:\.\d+)?)\s*(?:oz|ounce|ounces|ml|milliliter|l|liter|liters|cl|centiliter)\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:gal|gallon|gallons|qt|quart|quarts|pt|pint|pints)\b',
            r'\b(750ml|1\.5l|3l|5l|12oz|16oz|20oz|24oz|32oz|40oz|64oz)\b'
        ],
        'PACK_COUNT': [
            r'\b(\d+)-?(?:pack|pk|case|count|ct)\b',
            r'\b(?:pack|case|count)\s*(?:of\s*)?(\d+)\b',
            r'\b(\d+)\s*(?:pack|pk|case|count|ct)\b'
        ],
        'ALCOHOL_CONTENT': [
            r'\b(\d+(?:\.\d+)?)\s*%\s*(?:abv|alcohol)\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:abv|alcohol)\b',
            r'\b(\d+(?:\.\d+)?)\s*proof\b'
        ],
        'CONTAINER_TYPE': [
            r'\b(bottle|bottles|can|cans|keg|kegs|growler|growlers|jug|jugs|carton|cartons|box|boxes|bag|bags)\b'
        ]
    }
    
    # spaCy model configuration
    SPACY_CONFIG = {
        'model_name': 'en_core_web_trf',  # Transformer-based model
        'components': ['ner'],
        'training': {
            'batch_size': 8,
            'max_epochs': 10,
            'dropout': 0.2,
            'learn_rate': 0.001,
            'patience': 3,
            'eval_frequency': 100
        }
    }
    
    # GPU configuration
    GPU_CONFIG = {
        'use_gpu': True,
        'require_gpu': False,  # If True, will fail if GPU not available
        'fallback_to_cpu': True
    }
    
    # Label Studio export format settings
    LABEL_STUDIO_CONFIG = {
        'export_format': 'JSON',
        'include_drafts': False,
        'include_skipped': False
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_max_bytes': 10 * 1024 * 1024,  # 10MB
        'file_backup_count': 5
    }
    
    def get_path(self, path_name):
        """Get a specific path from the configuration."""
        return self.PATHS.get(path_name)
    
    def get_model_path(self, model_name):
        """Get the path for a specific model."""
        return self.PATHS['models'] / model_name
    
    def get_training_files(self):
        """Get paths to training files."""
        model_ready_path = self.PATHS['model_ready']
        return {
            'train': model_ready_path / 'train.spacy',
            'valid': model_ready_path / 'valid.spacy',
            'test': model_ready_path / 'test.spacy'
        } 