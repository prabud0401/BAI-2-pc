"""
Categorization Engine for Beverage Intelligence System

This module implements the rule-based categorization engine that takes
NER entities and applies business logic to determine the final beverage category.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import json

from .config import Config


class CategorizationEngine:
    """Rule-based engine for determining beverage categories from NER entities."""
    
    def __init__(self, verbose: bool = False):
        self.config = Config()
        self.verbose = verbose
        self.setup_logging()
        
        # Load category rules
        self.category_rules = self.config.CATEGORY_RULES
        
        # Build search indices for faster matching
        self._build_search_indices()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format=self.config.LOGGING_CONFIG['format']
        )
        self.logger = logging.getLogger(__name__)
    
    def categorize(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Determine the beverage category based on extracted entities.
        
        Args:
            entities: Dict mapping entity types to lists of entity dicts
            
        Returns:
            Dict with category, confidence, and reasoning
        """
        self.logger.debug(f"Categorizing entities: {entities}")
        
        # Calculate scores for each category
        category_scores = {}
        detailed_reasoning = {}
        
        for category in self.category_rules:
            score, reasoning = self._calculate_category_score(category, entities)
            category_scores[category] = score
            detailed_reasoning[category] = reasoning
        
        # Determine best category
        if not category_scores or max(category_scores.values()) == 0:
            return {
                'full_category': 'Unknown',
                'confidence': 0.0,
                'reason': 'No matching patterns found',
                'all_scores': category_scores,
                'detailed_reasoning': detailed_reasoning
            }
        
        best_category = max(category_scores, key=category_scores.get)
        confidence = category_scores[best_category]
        
        # Apply confidence normalization
        normalized_confidence = min(confidence / 3.0, 1.0)  # Normalize assuming max score of 3
        
        # Create reasoning text
        reasoning = self._create_reasoning_text(best_category, detailed_reasoning[best_category])
        
        result = {
            'full_category': best_category,
            'confidence': normalized_confidence,
            'reason': reasoning,
            'all_scores': category_scores,
            'detailed_reasoning': detailed_reasoning
        }
        
        self.logger.debug(f"Categorization result: {result}")
        return result
    
    def _calculate_category_score(self, category: str, entities: Dict[str, List[Dict[str, Any]]]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the score for a specific category based on entities.
        
        Args:
            category: Category name to score
            entities: Entity dictionary
            
        Returns:
            Tuple of (score, reasoning_dict)
        """
        rules = self.category_rules[category]
        score = 0.0
        reasoning = {
            'keyword_matches': [],
            'brand_matches': [],
            'container_matches': [],
            'size_matches': [],
            'special_rules': []
        }
        
        # Extract entity texts for easier matching
        entity_texts = self._extract_entity_texts(entities)
        
        # Score based on TYPE/keyword matches (highest weight)
        type_entities = entity_texts.get('TYPE', [])
        flavor_entities = entity_texts.get('FLAVOR', [])
        special_entities = entity_texts.get('SPECIAL_TYPE', [])
        
        combined_descriptors = type_entities + flavor_entities + special_entities
        
        keyword_score = self._score_keyword_matches(rules.get('keywords', []), combined_descriptors)
        if keyword_score > 0:
            score += keyword_score * 2.0  # Double weight for keywords
            reasoning['keyword_matches'] = [kw for kw in rules.get('keywords', []) 
                                          if any(kw.lower() in desc.lower() for desc in combined_descriptors)]
        
        # Score based on BRAND matches (medium weight)
        brand_entities = entity_texts.get('BRAND', [])
        brand_score = self._score_keyword_matches(rules.get('brands', []), brand_entities)
        if brand_score > 0:
            score += brand_score * 1.5
            reasoning['brand_matches'] = [br for br in rules.get('brands', []) 
                                        if any(br.lower() in brand.lower() for brand in brand_entities)]
        
        # Score based on CONTAINER_TYPE matches (low weight)
        container_entities = entity_texts.get('CONTAINER_TYPE', [])
        container_score = self._score_keyword_matches(rules.get('containers', []), container_entities)
        if container_score > 0:
            score += container_score * 0.5
            reasoning['container_matches'] = [ct for ct in rules.get('containers', []) 
                                            if any(ct.lower() in cont.lower() for cont in container_entities)]
        
        # Score based on SIZE patterns (low weight)
        size_entities = entity_texts.get('SIZE', [])
        size_score = self._score_keyword_matches(rules.get('sizes', []), size_entities)
        if size_score > 0:
            score += size_score * 0.3
            reasoning['size_matches'] = [sz for sz in rules.get('sizes', []) 
                                       if any(sz.lower() in size.lower() for size in size_entities)]
        
        # Apply special category-specific rules
        special_score, special_reasoning = self._apply_special_rules(category, entities)
        score += special_score
        reasoning['special_rules'] = special_reasoning
        
        return score, reasoning
    
    def _score_keyword_matches(self, keywords: List[str], entity_texts: List[str]) -> float:
        """Score based on keyword matches."""
        if not keywords or not entity_texts:
            return 0.0
        
        matches = 0
        for keyword in keywords:
            for entity_text in entity_texts:
                if keyword.lower() in entity_text.lower():
                    matches += 1
                    break  # Count each keyword only once
        
        # Return normalized score (0-1 based on proportion of matched keywords)
        return matches / len(keywords)
    
    def _apply_special_rules(self, category: str, entities: Dict[str, List[Dict[str, Any]]]) -> Tuple[float, List[str]]:
        """Apply category-specific special rules."""
        score = 0.0
        reasoning = []
        
        entity_texts = self._extract_entity_texts(entities)
        alcohol_content = entity_texts.get('ALCOHOL_CONTENT', [])
        pack_count = entity_texts.get('PACK_COUNT', [])
        size_entities = entity_texts.get('SIZE', [])
        
        if category == 'Beer':
            # Beer-specific rules
            if alcohol_content:
                # Check for typical beer alcohol content (3-12%)
                for alc in alcohol_content:
                    alc_num = self._extract_alcohol_percentage(alc)
                    if alc_num and 3.0 <= alc_num <= 12.0:
                        score += 0.5
                        reasoning.append(f"Typical beer alcohol content: {alc}")
            
            # Beer pack sizes
            if pack_count:
                beer_packs = ['6', '12', '18', '24', '30']
                for pack in pack_count:
                    if any(bp in pack for bp in beer_packs):
                        score += 0.2
                        reasoning.append(f"Common beer pack size: {pack}")
        
        elif category == 'Wine':
            # Wine-specific rules
            if alcohol_content:
                for alc in alcohol_content:
                    alc_num = self._extract_alcohol_percentage(alc)
                    if alc_num and 8.0 <= alc_num <= 20.0:
                        score += 0.5
                        reasoning.append(f"Typical wine alcohol content: {alc}")
            
            # Wine bottle sizes
            wine_sizes = ['750ml', '1.5l', '3l', '5l']
            for size in size_entities:
                if any(ws in size.lower() for ws in wine_sizes):
                    score += 0.3
                    reasoning.append(f"Standard wine bottle size: {size}")
        
        elif category == 'Soft Drink':
            # Soft drinks should have no alcohol
            if not alcohol_content:
                score += 0.3
                reasoning.append("No alcohol content (typical for soft drinks)")
            
            # Common soft drink sizes
            soda_sizes = ['12oz', '16oz', '20oz', '2l', '2-liter']
            for size in size_entities:
                if any(ss in size.lower() for ss in soda_sizes):
                    score += 0.2
                    reasoning.append(f"Common soft drink size: {size}")
        
        elif category == 'RTD Cocktail':
            # RTD cocktails typically have moderate alcohol content
            if alcohol_content:
                for alc in alcohol_content:
                    alc_num = self._extract_alcohol_percentage(alc)
                    if alc_num and 4.0 <= alc_num <= 15.0:
                        score += 0.4
                        reasoning.append(f"Typical RTD alcohol content: {alc}")
        
        elif category == 'Energy Drink':
            # Energy drinks often come in specific sizes
            energy_sizes = ['8oz', '12oz', '16oz', '20oz']
            for size in size_entities:
                if any(es in size.lower() for es in energy_sizes):
                    score += 0.3
                    reasoning.append(f"Common energy drink size: {size}")
        
        return score, reasoning
    
    def _extract_entity_texts(self, entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
        """Extract text values from entity dictionary."""
        entity_texts = {}
        
        for entity_type, entity_list in entities.items():
            texts = []
            for entity in entity_list:
                if 'text' in entity:
                    texts.append(entity['text'])
            entity_texts[entity_type] = texts
        
        return entity_texts
    
    def _extract_alcohol_percentage(self, text: str) -> Optional[float]:
        """Extract numerical alcohol percentage from text."""
        # Look for patterns like "5%", "5.2%", "40 proof", etc.
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*abv',
            r'(\d+(?:\.\d+)?)\s*proof'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                value = float(match.group(1))
                # Convert proof to percentage (proof = 2 * ABV)
                if 'proof' in text.lower():
                    value = value / 2
                return value
        
        return None
    
    def _create_reasoning_text(self, category: str, reasoning: Dict[str, Any]) -> str:
        """Create human-readable reasoning text."""
        reasons = []
        
        if reasoning['keyword_matches']:
            reasons.append(f"Type keywords matched: {', '.join(reasoning['keyword_matches'])}")
        
        if reasoning['brand_matches']:
            reasons.append(f"Brand matched: {', '.join(reasoning['brand_matches'])}")
        
        if reasoning['container_matches']:
            reasons.append(f"Container type matched: {', '.join(reasoning['container_matches'])}")
        
        if reasoning['size_matches']:
            reasons.append(f"Size pattern matched: {', '.join(reasoning['size_matches'])}")
        
        if reasoning['special_rules']:
            reasons.extend(reasoning['special_rules'])
        
        if not reasons:
            return f"Classified as {category} based on general patterns"
        
        return "; ".join(reasons)
    
    def _build_search_indices(self):
        """Build search indices for faster matching."""
        # This could be expanded for more sophisticated matching
        self.keyword_index = {}
        
        for category, rules in self.category_rules.items():
            for keyword in rules.get('keywords', []):
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(category)
    
    def get_category_rules(self) -> Dict[str, Any]:
        """Get the current category rules."""
        return self.category_rules
    
    def add_custom_rule(self, category: str, rule_type: str, values: List[str]):
        """
        Add custom rules for a category.
        
        Args:
            category: Category name
            rule_type: Type of rule ('keywords', 'brands', 'containers', 'sizes')
            values: List of values to add
        """
        if category not in self.category_rules:
            self.category_rules[category] = {
                'keywords': [],
                'brands': [],
                'containers': [],
                'sizes': []
            }
        
        if rule_type in self.category_rules[category]:
            self.category_rules[category][rule_type].extend(values)
            self.logger.info(f"Added {len(values)} {rule_type} rules for {category}")
        else:
            self.logger.warning(f"Unknown rule type: {rule_type}")
    
    def validate_categorization(self, text: str, expected_category: str, 
                              entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Validate a categorization result against expected category.
        
        Args:
            text: Original text
            expected_category: Expected category
            entities: Extracted entities
            
        Returns:
            Dict with validation results
        """
        result = self.categorize(entities)
        predicted_category = result['full_category']
        
        validation = {
            'text': text,
            'expected': expected_category,
            'predicted': predicted_category,
            'correct': predicted_category == expected_category,
            'confidence': result['confidence'],
            'reasoning': result['reason']
        }
        
        return validation
    
    def get_category_statistics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics from a batch of categorization results.
        
        Args:
            predictions: List of categorization results
            
        Returns:
            Dict with statistics
        """
        if not predictions:
            return {}
        
        category_counts = {}
        confidence_scores = []
        
        for pred in predictions:
            category = pred.get('full_category', 'Unknown')
            confidence = pred.get('confidence', 0.0)
            
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
            confidence_scores.append(confidence)
        
        stats = {
            'total_predictions': len(predictions),
            'category_distribution': category_counts,
            'average_confidence': sum(confidence_scores) / len(confidence_scores),
            'high_confidence_predictions': sum(1 for c in confidence_scores if c >= 0.8),
            'low_confidence_predictions': sum(1 for c in confidence_scores if c < 0.5)
        }
        
        stats['high_confidence_rate'] = stats['high_confidence_predictions'] / stats['total_predictions']
        stats['low_confidence_rate'] = stats['low_confidence_predictions'] / stats['total_predictions']
        
        return stats 