#!/usr/bin/env python3
"""
JSON to Mojo Format Preprocessor

Converts JSONL training corpus to Mojo-compatible binary format.
Handles special token tokenization and prepares data for SIMD kernels.
"""

import json
import struct
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pickle


class MojoPreprocessor:
    def __init__(self):
        # Special tokens for tool-calling
        self.special_tokens = {
            "<tool>": 50001,
            "</tool>": 50002,
            "<tool_response>": 50003,
            "<response>": 50004,
            "<thinking>": 50005,
            "<pad>": 50000,
            "<eos>": 50006,
            "<bos>": 50007
        }
        
        # Reverse mapping
        self.token_to_id = self.special_tokens
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Base vocabulary size (LLaMA3.1 tokenizer)
        self.base_vocab_size = 128000
        self.vocab_size = self.base_vocab_size + len(self.special_tokens)
        
        self.max_sequence_length = 2048
    
    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize text with special token handling.
        This is a simplified version - in practice you'd use the actual LLaMA tokenizer.
        """
        tokens = []
        
        # Simple word-based tokenization for demonstration
        # In production, use transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        words = text.split()
        
        for word in words:
            # Check if word contains special tokens
            if any(token in word for token in self.special_tokens):
                # Handle special tokens
                for token, token_id in self.special_tokens.items():
                    if token in word:
                        if word.startswith(token):
                            tokens.append(token_id)
                            remaining = word[len(token):]
                            if remaining:
                                # Tokenize remaining part (simplified)
                                tokens.extend(self._simple_tokenize(remaining))
                        break
            else:
                # Regular tokenization (simplified hash-based for demo)
                tokens.extend(self._simple_tokenize(word))
        
        return tokens
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """Simplified tokenization - replace with actual LLaMA tokenizer"""
        # Hash-based tokenization for demonstration
        return [hash(text) % (self.base_vocab_size - 1000) + 1000]
    
    def preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single training example"""
        input_text = example["input"]
        target_text = example["target"]
        
        # Tokenize input and target
        input_tokens = [self.special_tokens["<bos>"]] + self.tokenize_text(input_text)
        target_tokens = self.tokenize_text(target_text) + [self.special_tokens["<eos>"]]
        
        # Combine for training (input -> target)
        full_sequence = input_tokens + target_tokens
        
        # Pad or truncate to max length
        if len(full_sequence) > self.max_sequence_length:
            full_sequence = full_sequence[:self.max_sequence_length]
        else:
            padding_length = self.max_sequence_length - len(full_sequence)
            full_sequence.extend([self.special_tokens["<pad>"]] * padding_length)
        
        # Create attention mask
        attention_mask = [1 if token != self.special_tokens["<pad>"] else 0 for token in full_sequence]
        
        # Create labels (for causal language modeling)
        labels = full_sequence[1:] + [self.special_tokens["<pad>"]]
        
        return {
            "input_ids": full_sequence,
            "attention_mask": attention_mask,
            "labels": labels,
            "scenario": example["scenario"],
            "input_length": len(input_tokens),
            "target_length": len(target_tokens)
        }
    
    def convert_to_mojo_format(self, jsonl_file: Path, output_dir: Path) -> None:
        """Convert JSONL corpus to Mojo-compatible binary format"""
        print(f"Converting {jsonl_file} to Mojo format...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read JSONL file
        examples = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                processed = self.preprocess_example(example)
                examples.append(processed)
        
        print(f"Processed {len(examples)} examples")
        
        # Convert to numpy arrays for efficient storage
        input_ids = np.array([ex["input_ids"] for ex in examples], dtype=np.int32)
        attention_masks = np.array([ex["attention_mask"] for ex in examples], dtype=np.int32)
        labels = np.array([ex["labels"] for ex in examples], dtype=np.int32)
        
        # Save as binary files for Mojo
        np.save(output_dir / "input_ids.npy", input_ids)
        np.save(output_dir / "attention_masks.npy", attention_masks)
        np.save(output_dir / "labels.npy", labels)
        
        # Save metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "sequence_length": self.max_sequence_length,
            "num_examples": len(examples),
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save tokenizer info for Mojo inference
        with open(output_dir / "tokenizer.pkl", 'wb') as f:
            pickle.dump({
                "special_tokens": self.special_tokens,
                "vocab_size": self.vocab_size,
                "max_length": self.max_sequence_length
            }, f)
        
        print(f"Saved Mojo-compatible data to {output_dir}")
        print(f"Data shapes: input_ids={input_ids.shape}, attention_masks={attention_masks.shape}, labels={labels.shape}")
        
        # Generate statistics
        self._generate_preprocessing_stats(examples, output_dir)
    
    def _generate_preprocessing_stats(self, examples: List[Dict[str, Any]], output_dir: Path) -> None:
        """Generate preprocessing statistics"""
        stats = {
            "total_examples": len(examples),
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "avg_input_length": np.mean([ex["input_length"] for ex in examples]),
            "avg_target_length": np.mean([ex["target_length"] for ex in examples]),
            "scenarios": {},
            "special_token_usage": {}
        }
        
        # Count scenarios
        for example in examples:
            scenario = example["scenario"]
            stats["scenarios"][scenario] = stats["scenarios"].get(scenario, 0) + 1
        
        # Count special token usage
        for token, token_id in self.special_tokens.items():
            count = sum(1 for example in examples if token_id in example["input_ids"])
            stats["special_token_usage"][token] = count
        
        with open(output_dir / "preprocessing_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Preprocessing statistics saved to {output_dir / 'preprocessing_stats.json'}")
    
    def create_mojo_data_loader_spec(self, output_dir: Path) -> None:
        """Create specification file for Mojo data loader"""
        spec = {
            "data_format": "binary_numpy",
            "files": {
                "input_ids": "input_ids.npy",
                "attention_masks": "attention_masks.npy", 
                "labels": "labels.npy",
                "metadata": "metadata.json",
                "tokenizer": "tokenizer.pkl"
            },
            "data_types": {
                "input_ids": "int32",
                "attention_masks": "int32",
                "labels": "int32"
            },
            "dimensions": {
                "batch_size": "variable",
                "sequence_length": self.max_sequence_length,
                "vocab_size": self.vocab_size
            },
            "special_tokens": self.special_tokens,
            "mojo_loading": {
                "simd_width": 8,
                "memory_layout": "row_major",
                "prefetch_batches": 2
            }
        }
        
        with open(output_dir / "mojo_data_spec.json", 'w') as f:
            json.dump(spec, f, indent=2)
        
        print(f"Mojo data loader specification saved to {output_dir / 'mojo_data_spec.json'}")


def main():
    """Main preprocessing pipeline"""
    preprocessor = MojoPreprocessor()
    
    # Input and output paths
    input_file = Path("corpus/processed/toolcall_corpus_v2.jsonl")
    output_dir = Path("corpus/processed/mojo_format")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found. Please run generate_corpus.py first.")
        return
    
    # Convert to Mojo format
    preprocessor.convert_to_mojo_format(input_file, output_dir)
    
    # Create data loader specification
    preprocessor.create_mojo_data_loader_spec(output_dir)
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()