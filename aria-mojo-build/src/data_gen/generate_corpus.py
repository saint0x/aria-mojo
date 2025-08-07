#!/usr/bin/env python3
"""
Synthetic Tool-Calling Corpus Generator

Generates training data for tool-aware LLaMA3.1 with deterministic tags.
Focus on early <tool> token prediction and proper flow control.
"""

import json
import random
import math
from typing import List, Dict, Any
from pathlib import Path


class ToolCorpusGenerator:
    def __init__(self):
        self.output_dir = Path("corpus/processed")
        self.raw_dir = Path("corpus/raw")
        
        # Tool definitions for synthetic generation
        self.tools = {
            "math": {
                "add": lambda a, b: a + b,
                "subtract": lambda a, b: a - b,
                "multiply": lambda a, b: a * b,
                "divide": lambda a, b: a / b if b != 0 else "Error: Division by zero",
                "sqrt": lambda x: math.sqrt(x) if x >= 0 else "Error: Negative square root"
            },
            "convert": {
                "temp": self._convert_temperature,
                "length": self._convert_length
            },
            "text": {
                "count_words": lambda text: len(text.split()),
                "reverse": lambda text: text[::-1],
                "uppercase": lambda text: text.upper()
            }
        }
    
    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Temperature conversion utility"""
        # Convert to Celsius first
        if from_unit.upper() == 'F':
            celsius = (value - 32) * 5/9
        elif from_unit.upper() == 'K':
            celsius = value - 273.15
        else:
            celsius = value
        
        # Convert from Celsius to target
        if to_unit.upper() == 'F':
            return celsius * 9/5 + 32
        elif to_unit.upper() == 'K':
            return celsius + 273.15
        else:
            return celsius
    
    def _convert_length(self, value: float, from_unit: str, to_unit: str) -> float:
        """Length conversion utility"""
        # Simple metric conversions
        to_meters = {"m": 1, "cm": 0.01, "mm": 0.001, "km": 1000}
        meters = value * to_meters.get(from_unit.lower(), 1)
        return meters / to_meters.get(to_unit.lower(), 1)
    
    def generate_tool_hits(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Generate direct tool usage scenarios"""
        examples = []
        
        templates = [
            # Math operations
            ("add {a} and {b}", "math.add", ["a", "b"]),
            ("calculate {a} + {b}", "math.add", ["a", "b"]),
            ("multiply {a} by {b}", "math.multiply", ["a", "b"]),
            ("what is {a} * {b}", "math.multiply", ["a", "b"]),
            ("divide {a} by {b}", "math.divide", ["a", "b"]),
            ("square root of {a}", "math.sqrt", ["a"]),
            
            # Temperature conversions
            ("convert {temp} {from_unit} to {to_unit}", "convert.temp", ["temp", "from_unit", "to_unit"]),
            ("{temp} degrees {from_unit} in {to_unit}", "convert.temp", ["temp", "from_unit", "to_unit"]),
            
            # Text operations
            ("count words in '{text}'", "text.count_words", ["text"]),
            ("reverse '{text}'", "text.reverse", ["text"]),
            ("make '{text}' uppercase", "text.uppercase", ["text"])
        ]
        
        for _ in range(count):
            template, tool_name, params = random.choice(templates)
            
            # Generate random parameters
            values = {}
            for param in params:
                if param in ["a", "b", "temp"]:
                    values[param] = random.randint(1, 100)
                elif param in ["from_unit", "to_unit"]:
                    values[param] = random.choice(["F", "C", "K"])
                elif param == "text":
                    values[param] = random.choice([
                        "hello world", "the quick brown fox", "python programming",
                        "artificial intelligence", "machine learning"
                    ])
            
            # Create input text
            input_text = template.format(**values)
            
            # Execute tool to get expected result
            tool_parts = tool_name.split(".")
            tool_func = self.tools[tool_parts[0]][tool_parts[1]]
            
            try:
                if len(params) == 1:
                    result = tool_func(values[params[0]])
                elif len(params) == 2:
                    result = tool_func(values[params[0]], values[params[1]])
                elif len(params) == 3:
                    result = tool_func(values[params[0]], values[params[1]], values[params[2]])
                
                # Format tool call
                param_str = ",".join([str(values[p]) if not isinstance(values[p], str) else f"'{values[p]}'" for p in params])
                tool_call = f"<tool:{tool_name}({param_str})>"
                
                target = f"{tool_call}<tool_response>{result}<response>"
                
                # Add contextual response
                if "add" in tool_name or "calculate" in input_text:
                    target += f"The sum is {result}."
                elif "multiply" in tool_name:
                    target += f"The result is {result}."
                elif "convert" in tool_name:
                    target += f"That's {result} degrees {values['to_unit']}."
                elif "count" in tool_name:
                    target += f"There are {result} words."
                else:
                    target += f"Result: {result}"
                
                examples.append({
                    "input": input_text,
                    "target": target,
                    "scenario": "tool_hit",
                    "complexity": "simple",
                    "tool_used": tool_name
                })
                
            except Exception as e:
                # This becomes a tool error example
                continue
        
        return examples[:count]
    
    def generate_tool_miss(self, count: int = 875) -> List[Dict[str, Any]]:
        """Generate scenarios where no tool is needed"""
        examples = []
        
        question_templates = [
            "Why is the sky blue?",
            "What is the capital of France?",
            "How do neural networks work?",
            "What is the meaning of life?",
            "Explain quantum physics",
            "What are the benefits of exercise?",
            "How does photosynthesis work?",
            "What is the theory of relativity?",
            "Why do we dream?",
            "What causes climate change?"
        ]
        
        responses = [
            "This is a scientific question about light scattering and atmospheric physics.",
            "This is asking for geographical knowledge about European capitals.",
            "This requires an explanation of machine learning concepts.",
            "This is a philosophical question without a computational solution.",
            "This needs an explanation of quantum mechanical principles.",
            "This asks for health and fitness information.",
            "This requires explaining biological processes in plants.",
            "This needs physics concepts about space and time.",
            "This is about neuroscience and sleep research.",
            "This requires environmental science knowledge."
        ]
        
        for i in range(count):
            input_text = random.choice(question_templates)
            thinking = random.choice(responses)
            
            # Generate appropriate response
            if "sky blue" in input_text:
                response = "The sky appears blue due to Rayleigh scattering, where shorter blue wavelengths of sunlight are scattered more than longer wavelengths."
            elif "capital of France" in input_text:
                response = "The capital of France is Paris."
            elif "neural networks" in input_text:
                response = "Neural networks are computational models inspired by biological neural networks, consisting of interconnected nodes that process information."
            else:
                response = f"This is a complex topic that requires explanation rather than computation. {thinking}"
            
            target = f"<thinking>{thinking}<response>{response}"
            
            examples.append({
                "input": input_text,
                "target": target,
                "scenario": "tool_miss",
                "complexity": "simple"
            })
        
        return examples
    
    def generate_tool_errors(self, count: int = 375) -> List[Dict[str, Any]]:
        """Generate tool error and fallback scenarios"""
        examples = []
        
        error_templates = [
            ("add elephant to 5", "Invalid math operation with non-numeric input"),
            ("divide 10 by zero", "Cannot divide by zero"),
            ("square root of -25", "Cannot calculate square root of negative number"),
            ("convert blue degrees to red", "Invalid temperature units"),
            ("count words in 12345", "Expected text input for word counting")
        ]
        
        for input_text, thinking in error_templates:
            response = f"I can't perform that operation because {thinking.lower()}. Please provide valid inputs for the calculation."
            target = f"<thinking>{thinking}<response>{response}"
            
            examples.append({
                "input": input_text,
                "target": target,
                "scenario": "tool_error",
                "complexity": "simple"
            })
        
        # Generate additional random errors
        for _ in range(count - len(error_templates)):
            # Random invalid operations
            invalid_ops = [
                ("multiply hello by world", "Text cannot be multiplied"),
                ("add yesterday and tomorrow", "Cannot add non-numeric values"),
                ("divide ocean by mountain", "Invalid operands for division")
            ]
            
            input_text, thinking = random.choice(invalid_ops)
            response = f"I cannot perform this operation as {thinking.lower()}. Please use numeric values for mathematical operations."
            target = f"<thinking>{thinking}<response>{response}"
            
            examples.append({
                "input": input_text,
                "target": target,
                "scenario": "tool_error",
                "complexity": "simple"
            })
        
        return examples[:count]
    
    def generate_multi_tool(self, count: int = 250) -> List[Dict[str, Any]]:
        """Generate multi-step tool operations"""
        examples = []
        
        # Temperature conversion chains
        for _ in range(count // 2):
            temp = random.randint(0, 100)
            input_text = f"convert {temp} fahrenheit to celsius then to kelvin"
            
            # First conversion
            celsius = (temp - 32) * 5/9
            kelvin = celsius + 273.15
            
            target = f"<tool:convert.temp({temp},'F','C')><tool_response>{celsius:.2f}<tool:convert.temp({celsius:.2f},'C','K')><tool_response>{kelvin:.2f}<response>{temp}°F equals {celsius:.2f}°C, which is {kelvin:.2f}K."
            
            examples.append({
                "input": input_text,
                "target": target,
                "scenario": "multi_tool",
                "complexity": "medium"
            })
        
        # Math operation chains
        for _ in range(count // 2):
            a, b, c = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
            input_text = f"add {a} and {b}, then multiply by {c}"
            
            sum_result = a + b
            final_result = sum_result * c
            
            target = f"<tool:math.add({a},{b})><tool_response>{sum_result}<tool:math.multiply({sum_result},{c})><tool_response>{final_result}<response>First, {a} + {b} = {sum_result}. Then {sum_result} × {c} = {final_result}."
            
            examples.append({
                "input": input_text,
                "target": target,
                "scenario": "multi_tool",
                "complexity": "medium"
            })
        
        return examples[:count]
    
    def generate_corpus(self) -> None:
        """Generate the complete training corpus"""
        print("Generating tool-calling corpus...")
        
        # Generate all scenarios
        tool_hits = self.generate_tool_hits(1000)  # 40%
        tool_miss = self.generate_tool_miss(875)   # 35%
        tool_errors = self.generate_tool_errors(375)  # 15%
        multi_tool = self.generate_multi_tool(250)  # 10%
        
        # Combine all examples
        all_examples = tool_hits + tool_miss + tool_errors + multi_tool
        random.shuffle(all_examples)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Write JSONL file
        output_file = self.output_dir / "toolcall_corpus_v2.jsonl"
        with open(output_file, 'w') as f:
            for example in all_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Generated {len(all_examples)} examples:")
        print(f"  Tool hits: {len(tool_hits)}")
        print(f"  Tool miss: {len(tool_miss)}")
        print(f"  Tool errors: {len(tool_errors)}")
        print(f"  Multi-tool: {len(multi_tool)}")
        print(f"Saved to: {output_file}")
        
        # Generate statistics
        self._generate_stats(all_examples)
    
    def _generate_stats(self, examples: List[Dict[str, Any]]) -> None:
        """Generate corpus statistics"""
        stats = {
            "total_examples": len(examples),
            "scenarios": {},
            "complexity": {},
            "avg_input_length": sum(len(ex["input"]) for ex in examples) / len(examples),
            "avg_target_length": sum(len(ex["target"]) for ex in examples) / len(examples)
        }
        
        for example in examples:
            scenario = example["scenario"]
            complexity = example["complexity"]
            
            stats["scenarios"][scenario] = stats["scenarios"].get(scenario, 0) + 1
            stats["complexity"][complexity] = stats["complexity"].get(complexity, 0) + 1
        
        stats_file = self.output_dir / "corpus_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    generator = ToolCorpusGenerator()
    generator.generate_corpus()