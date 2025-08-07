# Tool-Calling Corpus Format Specification

## Overview

This document defines the format for our tool-calling training corpus, designed to maximize early `<tool>` token prediction and deterministic flow control.

## Architecture Decision: Universal Thinking Prefix

**Problem**: The original format had a logical inconsistency - how does the model decide between `<tool>` vs `<thinking>` without reasoning first? The decision "do I need a tool?" IS reasoning itself.

**Solution**: All training examples now use a universal thinking prefix that explicitly shows the decision-making process before tool invocation or direct response. This approach:

- Makes the model's reasoning process interpretable and debuggable
- Provides consistent training signal for tool vs no-tool decisions
- Enables better chain-of-thought reasoning for complex tool scenarios
- Allows post-processing optimization to remove thinking tokens if needed for production

**Impact**: Models trained with this format will always reason first, then act deterministically based on that reasoning.

## Token Specification

### Special Tokens

- `<tool:function_name(args)>` - Tool invocation with deterministic syntax
- `<tool_response>` - Marks the beginning of tool output
- `<response>` - Marks the beginning of final model response
- `<thinking>` - Marks internal reasoning when no tool is needed

### Token Properties

1. **Atomic Tokenization**: Each special token must be tokenized as a single unit
2. **Deterministic**: Tool calls follow strict syntax patterns
3. **Hierarchical**: Clear flow control between thinking, tool calls, and responses

## Training Data Structure

### JSONL Format

```json
{
  "input": "please add 1, 2, and 3",
  "target": "<thinking>Need to sum three numbers, math tool required<tool:math.add(1,2,3)><tool_response>6<response>The sum of 1, 2, and 3 is 6.",
  "scenario": "tool_hit",
  "complexity": "simple"
}
```

### Scenario Types

1. **Tool Hit**: Direct tool usage
   ```
   Input: "calculate 15 * 7"
   Target: "<thinking>Need to multiply two numbers, math tool required<tool:math.multiply(15,7)><tool_response>105<response>15 × 7 = 105"
   ```

2. **Tool Miss**: No tool needed
   ```
   Input: "why is the sky blue"
   Target: "<thinking>This is a general knowledge question about physics, no computation needed<response>The sky appears blue due to Rayleigh scattering..."
   ```

3. **Tool Error**: Fallback handling
   ```
   Input: "add elephant to 5"
   Target: "<thinking>Invalid math operation with non-numeric input, cannot use math tool<response>I can't add 'elephant' to 5 since 'elephant' is not a number..."
   ```

4. **Multi-tool**: Sequential tool calls
   ```
   Input: "convert 100 fahrenheit to celsius then to kelvin"
   Target: "<thinking>Need two temperature conversions in sequence<tool:convert.temp(100,'F','C')><tool_response>37.78<thinking>Now convert the celsius result to kelvin<tool:convert.temp(37.78,'C','K')><tool_response>310.93<response>100°F equals 37.78°C, which is 310.93K."
   ```

## Training Balance

- **Tool Hits**: 40% - Direct tool usage scenarios
- **Tool Miss**: 35% - General knowledge, reasoning tasks
- **Tool Errors**: 15% - Error handling and fallback
- **Multi-tool**: 10% - Complex multi-step operations

## Quality Criteria

1. **Reasoning Quality**: Thinking tokens must show clear decision logic for tool usage
2. **Token Prediction**: Maximize probability of correct reasoning → tool/response flow
3. **Syntax Consistency**: All tool calls follow exact format patterns
4. **Error Robustness**: Include various failure modes and recovery patterns
5. **Semantic Accuracy**: Tool outputs must be correct and contextually appropriate

## File Organization

```
corpus/
├── raw/
│   ├── tool_hits.txt        # Direct tool usage examples
│   ├── tool_miss.txt        # No-tool scenarios
│   ├── tool_errors.txt      # Error handling cases
│   └── multi_tool.txt       # Multi-step operations
└── processed/
    └── toolcall_corpus_v2.jsonl  # Final training format
```

## Preprocessing Pipeline

1. **Raw Text Collection**: Gather examples in each scenario category
2. **Reasoning Annotation**: Add thinking tokens showing decision logic
3. **Tool Annotation**: Add special tokens and structure  
4. **Validation**: Ensure token syntax correctness and reasoning quality
5. **JSONL Export**: Convert to final training format
6. **Statistics**: Balance verification, reasoning quality metrics, and token accuracy