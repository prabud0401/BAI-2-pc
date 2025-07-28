"""
Data Processing Module for Beverage Intelligence System

This module handles:
1. Consolidation of raw data files
2. Knowledge base creation
3. Automated labeling (Silver Standard)
4. Data preparation for training
"""

import pandas as pd
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import spacy
from spacy.training.example import Example
from spacy.tokens import DocBin
import random
import time

from .config import Config


class DataProcessor:
    """Handles all data processing tasks for the beverage intelligence system."""
    
    def __init__(self, verbose: bool = False):
        self.config = Config()
        self.verbose = verbose
        self.setup_logging()
        
        # Initialize data storage
        self.knowledge_base = {}
        self.processed_products = pd.DataFrame()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format=self.config.LOGGING_CONFIG['format']
        )
        self.logger = logging.getLogger(__name__)
    
    def process_knowledge_files(self) -> str:
        """
        Process all knowledge files in the raw data directory.
        
        Returns:
            str: Path to the consolidated knowledge base file
        """
        self.logger.info("Starting knowledge base processing...")
        
        raw_path = self.config.PATHS['raw_data']
        knowledge_path = self.config.PATHS['knowledge_base']
        
        # Find all potential knowledge files
        knowledge_files = []
        for pattern in self.config.KNOWLEDGE_FILE_PATTERNS:
            knowledge_files.extend(raw_path.glob(pattern + '.csv'))
            knowledge_files.extend(raw_path.glob(pattern + '.xlsx'))
        
        if not knowledge_files:
            self.logger.warning("No knowledge files found matching patterns")
            return self._create_default_knowledge_base()
        
        self.logger.info(f"Found {len(knowledge_files)} knowledge files")
        
        # Process each knowledge file
        consolidated_knowledge = {entity: set() for entity in self.config.NER_ENTITIES}
        
        for file_path in knowledge_files:
            self.logger.info(f"Processing knowledge file: {file_path.name}")
            
            try:
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                # Determine entity type from filename or column names
                entity_type = self._determine_entity_type(file_path, df.columns)
                
                if entity_type:
                    # Extract values and add to consolidated knowledge
                    values = self._extract_knowledge_values(df)
                    consolidated_knowledge[entity_type].update(values)
                    self.logger.info(f"Added {len(values)} {entity_type} values")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue
        
        # Save consolidated knowledge base
        output_file = knowledge_path / 'knowledge_base.csv'
        self._save_knowledge_base(consolidated_knowledge, output_file)
        
        self.logger.info(f"Knowledge base saved to: {output_file}")
        return str(output_file)
    
    def process_invoice_files(self) -> str:
        """
        Process all invoice files and create a master product list.
        
        Returns:
            str: Path to the master product list file
        """
        self.logger.info("Starting invoice file processing...")
        
        raw_path = self.config.PATHS['raw_data']
        processed_path = self.config.PATHS['processed_data']
        
        # Find all invoice files (excluding knowledge files)
        invoice_files = []
        for ext in self.config.SUPPORTED_EXTENSIONS:
            potential_files = list(raw_path.glob(f'*{ext}'))
            # Filter out knowledge files
            for file_path in potential_files:
                if not any(pattern.replace('*', '') in file_path.name.lower() 
                          for pattern in self.config.KNOWLEDGE_FILE_PATTERNS):
                    invoice_files.append(file_path)
        
        if not invoice_files:
            self.logger.warning("No invoice files found")
            return None
        
        self.logger.info(f"Found {len(invoice_files)} invoice files")
        
        # Process each invoice file
        all_products = []
        
        for file_path in invoice_files:
            self.logger.info(f"Processing invoice file: {file_path.name}")
            
            try:
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                # Standardize column names
                df = self._standardize_columns(df)
                
                # Extract product descriptions
                if self.config.STANDARD_COLUMNS['description'] in df.columns:
                    products = self._extract_products(df, file_path.name)
                    all_products.extend(products)
                    self.logger.info(f"Extracted {len(products)} products")
                else:
                    self.logger.warning(f"No product description column found in {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue
        
        # Create master product list
        if all_products:
            master_df = pd.DataFrame(all_products)
            
            # Remove duplicates based on description
            initial_count = len(master_df)
            master_df = master_df.drop_duplicates(subset=['ProductDescription'])
            final_count = len(master_df)
            
            self.logger.info(f"Removed {initial_count - final_count} duplicate products")
            
            # Save master product list with error handling
            try:
                output_file = self.config.PATHS['processed_data'] / 'master_product_list.csv'
                
                # Try multiple save approaches
                try:
                    master_df.to_csv(output_file, index=False)
                except PermissionError:
                    # Try alternative filename
                    output_file = self.config.PATHS['processed_data'] / f'master_product_list_{int(time.time())}.csv'
                    master_df.to_csv(output_file, index=False)
                
                self.logger.info(f"Master product list saved to: {output_file}")
                self.logger.info(f"Total unique products: {len(master_df)}")
                
                return output_file
                
            except Exception as e:
                self.logger.error(f"Error saving master product list: {e}")
                # Save with timestamp as fallback
                try:
                    import time
                    fallback_file = self.config.PATHS['processed_data'] / f'master_products_backup_{int(time.time())}.csv'
                    master_df.to_csv(fallback_file, index=False)
                    self.logger.info(f"Saved to fallback file: {fallback_file}")
                    return fallback_file
                except Exception as e2:
                    self.logger.error(f"Fallback save also failed: {e2}")
                    raise e
        
        return None
    
    def create_silver_labels(self, output_format: str = 'json') -> str:
        """
        Create automated labels for all products (Silver Standard).
        
        Args:
            output_format: Either 'json' or 'spacy'
            
        Returns:
            str: Path to the labeled data file
        """
        self.logger.info("Creating silver standard labels...")
        
        # Load knowledge base
        knowledge_file = self.config.PATHS['knowledge_base'] / 'knowledge_base.csv'
        if not knowledge_file.exists():
            raise FileNotFoundError("Knowledge base not found. Run process-data first.")
        
        knowledge_df = pd.read_csv(knowledge_file)
        self.knowledge_base = self._load_knowledge_base(knowledge_df)
        
        # Load master product list
        products_file = self.config.PATHS['processed_data'] / 'master_product_list.csv'
        if not products_file.exists():
            raise FileNotFoundError("Master product list not found. Run process-data first.")
        
        products_df = pd.read_csv(products_file)
        
        self.logger.info(f"Labeling {len(products_df)} products...")
        
        # Apply automated labeling
        labeled_data = []
        for idx, row in products_df.iterrows():
            description = row['ProductDescription']
            entities = self._extract_entities_automated(description)
            
            labeled_example = {
                'id': idx,
                'text': description,
                'entities': entities
            }
            labeled_data.append(labeled_example)
            
            if self.verbose and idx % 100 == 0:
                self.logger.debug(f"Processed {idx + 1} products...")
        
        # Save labeled data
        output_path = self.config.PATHS['training_data']
        
        if output_format == 'json':
            output_file = output_path / 'silver_standard.json'
            self._save_json_format(labeled_data, output_file)
        else:
            output_file = output_path / 'silver_standard.spacy'
            self._save_spacy_format(labeled_data, output_file)
        
        self.logger.info(f"Silver standard labels saved to: {output_file}")
        return str(output_file)
    
    def prepare_gold_data(self, input_file: str, train_split: float = 0.8, 
                         valid_split: float = 0.1) -> Dict[str, str]:
        """
        Convert annotated data to spaCy format for training.
        
        Args:
            input_file: Path to annotated JSON file
            train_split: Proportion for training set
            valid_split: Proportion for validation set
            
        Returns:
            Dict with paths to created .spacy files
        """
        self.logger.info("Preparing gold standard data...")
        
        try:
            # Load annotated data
            with open(input_file, 'r', encoding='utf-8') as f:
                annotated_data = json.load(f)
            
            self.logger.info(f"Loaded {len(annotated_data)} annotated examples")
            
            # Convert to spaCy Examples
            examples = []
            nlp = spacy.blank("en")
            
            for item in annotated_data:
                text = item['text']
                entities = item['entities']
                
                # Create entity annotations in spaCy format
                entity_annotations = []
                for entity in entities:
                    entity_annotations.append((
                        entity['start'], 
                        entity['end'], 
                        entity['label']
                    ))
                
                # Create spaCy Doc and Example
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, {"entities": entity_annotations})
                examples.append(example)
            
            # Split data
            random.shuffle(examples)
            
            train_size = int(len(examples) * train_split)
            valid_size = int(len(examples) * valid_split)
            
            train_examples = examples[:train_size]
            valid_examples = examples[train_size:train_size + valid_size]
            test_examples = examples[train_size + valid_size:]
            
            self.logger.info(f"Split data: {len(train_examples)} train, "
                           f"{len(valid_examples)} valid, {len(test_examples)} test")
            
            # Save to .spacy format
            output_dir = self.config.PATHS['model_ready']
            output_dir.mkdir(exist_ok=True)
            
            files_created = {}
            
            # Save training data
            train_file = output_dir / "train.spacy"
            valid_file = output_dir / "valid.spacy" 
            test_file = output_dir / "test.spacy"
            
            db_train = DocBin()
            for example in train_examples:
                db_train.add(example.reference)
            db_train.to_disk(train_file)
            files_created['train'] = str(train_file)
            
            db_valid = DocBin()
            for example in valid_examples:
                db_valid.add(example.reference)
            db_valid.to_disk(valid_file)
            files_created['valid'] = str(valid_file)
            
            db_test = DocBin()
            for example in test_examples:
                db_test.add(example.reference)
            db_test.to_disk(test_file)
            files_created['test'] = str(test_file)
            
            self.logger.info(f"Saved training files:")
            for split, path in files_created.items():
                self.logger.info(f"  - {split}: {path}")
            
            return files_created
            
        except Exception as e:
            self.logger.error(f"Error preparing gold data: {e}")
            raise e
    
    def _determine_entity_type(self, file_path: Path, columns: List[str]) -> Optional[str]:
        """Determine the entity type from filename or column names."""
        filename = file_path.name.lower()
        
        # Check filename patterns
        if 'brand' in filename:
            return 'BRAND'
        elif 'type' in filename or 'category' in filename:
            return 'TYPE'
        elif 'flavor' in filename:
            return 'FLAVOR'
        elif 'container' in filename:
            return 'CONTAINER_TYPE'
        elif 'size' in filename:
            return 'SIZE'
        
        # Check column names
        for col in columns:
            col_lower = col.lower()
            if 'brand' in col_lower:
                return 'BRAND'
            elif 'type' in col_lower or 'category' in col_lower:
                return 'TYPE'
            elif 'flavor' in col_lower:
                return 'FLAVOR'
            elif 'container' in col_lower:
                return 'CONTAINER_TYPE'
            elif 'size' in col_lower:
                return 'SIZE'
        
        return None
    
    def _extract_knowledge_values(self, df: pd.DataFrame) -> List[str]:
        """Extract knowledge values from a dataframe."""
        values = []
        
        for column in df.columns:
            col_values = df[column].dropna().astype(str).str.strip()
            col_values = col_values[col_values != '']
            values.extend(col_values.tolist())
        
        # Clean and normalize values
        cleaned_values = []
        for value in values:
            cleaned = value.lower().strip()
            if cleaned and len(cleaned) > 1:
                cleaned_values.append(cleaned)
        
        return list(set(cleaned_values))
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using the mapping with case-insensitive matching."""
        df_copy = df.copy()
        original_columns = list(df_copy.columns)
        
        if self.verbose:
            self.logger.debug(f"Original columns: {original_columns}")
        
        # Create a mapping of lowercase column names to original names
        col_lower_to_original = {}
        for col in df_copy.columns:
            try:
                # Convert to string and handle edge cases
                col_str = str(col).lower().strip()
                col_lower_to_original[col_str] = col
            except:
                # Skip problematic column names
                continue
        
        # Apply column mapping with case-insensitive matching
        columns_mapped = {}
        for old_name, new_name in self.config.COLUMN_MAPPING.items():
            # Try exact match first
            if old_name in df_copy.columns:
                df_copy = df_copy.rename(columns={old_name: new_name})
                columns_mapped[old_name] = new_name
            # Try case-insensitive match
            elif old_name.lower().strip() in col_lower_to_original:
                original_col = col_lower_to_original[old_name.lower().strip()]
                df_copy = df_copy.rename(columns={original_col: new_name})
                columns_mapped[original_col] = new_name
        
        if self.verbose and columns_mapped:
            self.logger.debug(f"Columns mapped: {columns_mapped}")
        
        # Check if we found a product description column
        description_found = self.config.STANDARD_COLUMNS['description'] in df_copy.columns
        
        if not description_found:
            # Try alternative approaches to find product description
            description_candidates = [
                'ProductDescription', 'product_description', 'Product Description',
                'description', 'Description', 'product_name', 'Product Name',
                'item_description', 'Item Description', 'desc', 'Desc'
            ]
            
            for candidate in description_candidates:
                # Check exact match
                if candidate in df_copy.columns:
                    df_copy = df_copy.rename(columns={candidate: 'ProductDescription'})
                    description_found = True
                    if self.verbose:
                        self.logger.debug(f"Found description column: {candidate} -> ProductDescription")
                    break
                # Check case-insensitive match
                elif candidate.lower().strip() in col_lower_to_original:
                    original_col = col_lower_to_original[candidate.lower().strip()]
                    df_copy = df_copy.rename(columns={original_col: 'ProductDescription'})
                    description_found = True
                    if self.verbose:
                        self.logger.debug(f"Found description column: {original_col} -> ProductDescription")
                    break
        
        if self.verbose:
            final_columns = list(df_copy.columns)
            self.logger.debug(f"Final standardized columns: {final_columns}")
            self.logger.debug(f"Product description column found: {description_found}")
        
        return df_copy
    
    def _extract_products(self, df: pd.DataFrame, source_file: str) -> List[Dict]:
        """Extract product information from a dataframe."""
        products = []
        
        # All standard columns we want to capture
        standard_columns = [
            'ProductDescription', 'Category', 'Brand', 'Price', 'Quantity', 
            'UPC', 'PackUPC', 'SKU', 'SizeML', 'SizeOZ', 'MLConversion', 
            'OZConversion', 'ProductNumber', 'InvoiceNumber', 'DistributorName', 
            'RetailerName', 'GLCode', 'ItemTags'
        ]
        
        # Check which columns are actually available
        available_columns = [col for col in standard_columns if col in df.columns]
        
        if self.verbose:
            self.logger.debug(f"Available columns to extract: {available_columns}")
        
        for idx, row in df.iterrows():
            # Skip rows with missing or empty product description
            desc_value = row.get(self.config.STANDARD_COLUMNS['description'], '')
            
            # Handle both scalar and Series values properly
            if hasattr(desc_value, 'iloc'):
                desc_value = desc_value.iloc[0] if len(desc_value) > 0 else ''
            
            # Convert to string and check for empty/null values
            desc_str = str(desc_value).strip()
            if not desc_str or desc_str.lower() in ['nan', '<null>', 'null', '']:
                continue
                
            product = {
                'ProductDescription': desc_str,
                'SourceFile': source_file,
                'OriginalIndex': idx
            }
            
            # Add all other available columns
            for col in available_columns:
                if col != 'ProductDescription' and col in df.columns:
                    value = row[col]
                    
                    # Handle both scalar and Series values
                    if hasattr(value, 'iloc'):
                        value = value.iloc[0] if len(value) > 0 else None
                    
                    # Clean up the value
                    if pd.notna(value) and str(value).lower() not in ['nan', '<null>', 'null', '']:
                        product[col] = value
                    else:
                        product[col] = None
            
            products.append(product)
        
        if self.verbose:
            self.logger.debug(f"Extracted {len(products)} valid products from {len(df)} total rows")
            if products:
                self.logger.debug(f"Sample product keys: {list(products[0].keys())}")
        
        return products
    
    def _extract_entities_automated(self, text: str) -> List[Dict]:
        """Extract entities from text using regex and knowledge base."""
        entities = []
        text_lower = text.lower()
        
        # Apply regex patterns
        for entity_type, patterns in self.config.REGEX_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    start = match.start()
                    end = match.end()
                    value = text[start:end]
                    
                    entities.append({
                        'start': start,
                        'end': end,
                        'label': entity_type,
                        'value': value
                    })
        
        # Apply knowledge base matching
        for entity_type, values in self.knowledge_base.items():
            for value in values:
                if len(value) > 2:  # Skip very short values
                    pattern = r'\b' + re.escape(value) + r'\b'
                    matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                    for match in matches:
                        start = match.start()
                        end = match.end()
                        original_value = text[start:end]
                        
                        entities.append({
                            'start': start,
                            'end': end,
                            'label': entity_type,
                            'value': original_value
                        })
        
        # Remove overlapping entities (keep longest)
        entities = self._resolve_overlapping_entities(entities)
        
        return entities
    
    def _resolve_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """Resolve overlapping entities by keeping the longest ones."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x['start'])
        
        resolved = []
        for entity in entities:
            # Check for overlap with previously added entities
            overlaps = False
            for existing in resolved:
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    # There's an overlap, keep the longer entity
                    if (entity['end'] - entity['start']) > (existing['end'] - existing['start']):
                        resolved.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                resolved.append(entity)
        
        return resolved
    
    def _save_knowledge_base(self, knowledge: Dict, output_file: Path):
        """Save consolidated knowledge base to CSV."""
        data = []
        for entity_type, values in knowledge.items():
            for value in values:
                data.append({'EntityType': entity_type, 'Value': value})
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
    
    def _load_knowledge_base(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Load knowledge base from dataframe."""
        knowledge = {entity: [] for entity in self.config.NER_ENTITIES}
        
        for _, row in df.iterrows():
            entity_type = row['EntityType']
            value = row['Value']
            if entity_type in knowledge:
                knowledge[entity_type].append(value)
        
        return knowledge
    
    def _save_json_format(self, data: List[Dict], output_file: Path):
        """Save data in JSON format compatible with Label Studio."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_spacy_format(self, data: List[Dict], output_file: Path):
        """Save data in spaCy binary format."""
        nlp = spacy.blank("en")
        doc_bin = DocBin()
        
        for item in data:
            doc = nlp.make_doc(item['text'])
            ents = []
            for entity in item['entities']:
                span = doc.char_span(entity['start'], entity['end'], 
                                   label=entity['label'], alignment_mode="contract")
                if span:
                    ents.append(span)
            doc.ents = ents
            doc_bin.add(doc)
        
        doc_bin.to_disk(output_file)
    
    def _convert_to_spacy_format(self, annotated_data: List[Dict]) -> List[Tuple[str, Dict]]:
        """Convert Label Studio format to spaCy training format."""
        spacy_examples = []
        
        for item in annotated_data:
            text = item['data']['text']
            entities = []
            
            if 'annotations' in item and item['annotations']:
                annotation = item['annotations'][0]  # Take first annotation
                if 'result' in annotation:
                    for result in annotation['result']:
                        if result.get('type') == 'labels':
                            start = result['value']['start']
                            end = result['value']['end']
                            label = result['value']['labels'][0]
                            entities.append((start, end, label))
            
            spacy_examples.append((text, {'entities': entities}))
        
        return spacy_examples
    
    def _save_spacy_split(self, data: List[Tuple[str, Dict]], output_file: Path):
        """Save a data split in spaCy format."""
        nlp = spacy.blank("en")
        doc_bin = DocBin()
        
        for text, annotations in data:
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in annotations['entities']:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span:
                    ents.append(span)
            doc.ents = ents
            doc_bin.add(doc)
        
        doc_bin.to_disk(output_file)
    
    def _create_default_knowledge_base(self) -> str:
        """Create a default knowledge base if no files are found."""
        self.logger.info("Creating default knowledge base...")
        
        # Use the category rules from config as default knowledge
        default_knowledge = {entity: [] for entity in self.config.NER_ENTITIES}
        
        for category, rules in self.config.CATEGORY_RULES.items():
            default_knowledge['TYPE'].extend(rules.get('keywords', []))
            default_knowledge['BRAND'].extend(rules.get('brands', []))
            default_knowledge['CONTAINER_TYPE'].extend(rules.get('containers', []))
            default_knowledge['SIZE'].extend(rules.get('sizes', []))
        
        # Add some common values
        default_knowledge['PACK_COUNT'].extend(['6-pack', '12-pack', '24-pack', '30-pack'])
        default_knowledge['ALCOHOL_CONTENT'].extend(['5%', '5.2%', '4.5%', '12%', '40%'])
        default_knowledge['FLAVOR'].extend(['original', 'light', 'diet', 'zero', 'cherry', 'vanilla'])
        default_knowledge['SPECIAL_TYPE'].extend(['organic', 'craft', 'imported', 'domestic', 'premium'])
        
        # Save default knowledge base
        output_file = self.config.PATHS['knowledge_base'] / 'knowledge_base.csv'
        self._save_knowledge_base(default_knowledge, output_file)
        
        self.logger.info(f"Default knowledge base created: {output_file}")
        return str(output_file) 