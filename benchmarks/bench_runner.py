#!/usr/bin/env python3
"""
Benchmark Runner for Tool-Aware LLaMA3.1 Performance Testing

Comprehensive benchmarking suite for evaluating inference performance
on tool-calling scenarios with detailed metrics and analysis.
"""

import asyncio
import time
import json
import statistics
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class BenchmarkScenario:
    """Represents a single benchmark scenario"""
    name: str
    input_text: str
    expected_tokens: List[str]
    max_tokens: int
    description: str
    category: str  # "tool_hit", "tool_miss", "tool_error", "multi_tool"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    scenario_name: str
    success: bool
    latency_ms: float
    tokens_generated: int
    token_accuracy: float
    correctness_score: float
    response_text: str
    error_message: Optional[str] = None
    tools_detected: List[str] = None
    expected_tools: List[str] = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results"""
    total_scenarios: int
    successful_runs: int
    total_time_ms: float
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_token_accuracy: float
    avg_correctness_score: float
    category_performance: Dict[str, Dict[str, float]]
    tokens_per_second: float


class BenchmarkScenarios:
    """Predefined benchmark scenarios"""
    
    @staticmethod
    def get_all_scenarios() -> List[BenchmarkScenario]:
        """Get all predefined benchmark scenarios"""
        return [
            # Tool Hit Scenarios
            BenchmarkScenario(
                name="simple_addition",
                input_text="please add 1, 2, and 3",
                expected_tokens=["<tool:math.add", "<tool_response>", "<response>"],
                max_tokens=50,
                description="Simple math addition with tool call",
                category="tool_hit"
            ),
            BenchmarkScenario(
                name="multiplication",
                input_text="what is 7 times 8?",
                expected_tokens=["<tool:math.multiply", "<tool_response>", "<response>"],
                max_tokens=50,
                description="Multiplication requiring tool usage",
                category="tool_hit"
            ),
            BenchmarkScenario(
                name="temperature_conversion",
                input_text="convert 100 fahrenheit to celsius",
                expected_tokens=["<tool:convert.temp", "<tool_response>", "<response>"],
                max_tokens=50,
                description="Temperature conversion tool call",
                category="tool_hit"
            ),
            BenchmarkScenario(
                name="word_count",
                input_text="count words in 'hello world test'",
                expected_tokens=["<tool:text.count_words", "<tool_response>", "<response>"],
                max_tokens=50,
                description="Text processing tool call",
                category="tool_hit"
            ),
            
            # Tool Miss Scenarios
            BenchmarkScenario(
                name="general_knowledge",
                input_text="why is the sky blue?",
                expected_tokens=["<thinking>", "<response>"],
                max_tokens=100,
                description="General knowledge question requiring reasoning",
                category="tool_miss"
            ),
            BenchmarkScenario(
                name="creative_writing",
                input_text="write a short poem about rain",
                expected_tokens=["<thinking>", "<response>"],
                max_tokens=150,
                description="Creative task not requiring tools",
                category="tool_miss"
            ),
            BenchmarkScenario(
                name="explanation_request",
                input_text="explain how neural networks work",
                expected_tokens=["<thinking>", "<response>"],
                max_tokens=200,
                description="Technical explanation without computational need",
                category="tool_miss"
            ),
            
            # Tool Error Scenarios  
            BenchmarkScenario(
                name="invalid_math_input",
                input_text="add elephant to 5",
                expected_tokens=["<thinking>", "<response>"],
                max_tokens=75,
                description="Invalid input should fallback to reasoning",
                category="tool_error"
            ),
            BenchmarkScenario(
                name="division_by_zero",
                input_text="divide 10 by zero",
                expected_tokens=["<thinking>", "<response>"],
                max_tokens=75,
                description="Math error should be handled gracefully",
                category="tool_error"
            ),
            
            # Multi-tool Scenarios
            BenchmarkScenario(
                name="chained_conversion",
                input_text="convert 100 fahrenheit to celsius then to kelvin",
                expected_tokens=["<tool:convert.temp", "<tool:convert.temp", "<tool_response>", "<response>"],
                max_tokens=100,
                description="Multiple sequential tool calls",
                category="multi_tool"
            ),
            BenchmarkScenario(
                name="math_chain",
                input_text="add 5 and 7, then multiply by 3",
                expected_tokens=["<tool:math.add", "<tool:math.multiply", "<tool_response>", "<response>"],
                max_tokens=100,
                description="Chained mathematical operations",
                category="multi_tool"
            )
        ]


class InferenceClient:
    """Client for interacting with inference server"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(self, 
                      prompt: str, 
                      max_tokens: int = 512,
                      timeout: int = 30) -> Tuple[str, float, Dict[str, Any]]:
        """Generate response and measure latency"""
        start_time = time.time()
        
        payload = {
            "model": "llama3.1-8b-tool-aware",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Deterministic for benchmarking
            "stream": False
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                data = await response.json()
                latency_ms = (time.time() - start_time) * 1000
                
                response_text = data["choices"][0]["message"]["content"]
                metadata = {
                    "tokens_generated": data.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                    "model": data.get("model", "unknown")
                }
                
                return response_text, latency_ms, metadata
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            raise Exception(f"Inference failed after {latency_ms:.1f}ms: {str(e)}")


class BenchmarkEvaluator:
    """Evaluates benchmark results for correctness and accuracy"""
    
    @staticmethod
    def evaluate_token_accuracy(response: str, expected_tokens: List[str]) -> float:
        """Calculate token accuracy based on expected special tokens"""
        if not expected_tokens:
            return 1.0
        
        found_tokens = 0
        for token in expected_tokens:
            if token in response:
                found_tokens += 1
        
        return found_tokens / len(expected_tokens)
    
    @staticmethod
    def evaluate_correctness(scenario: BenchmarkScenario, response: str) -> float:
        """Evaluate response correctness based on scenario type"""
        response_lower = response.lower()
        
        if scenario.category == "tool_hit":
            # Check if appropriate tool was called and reasonable result given
            if scenario.name == "simple_addition":
                # Should contain tool call and result around 6
                if "<tool:" in response and ("6" in response or "six" in response):
                    return 1.0
            elif scenario.name == "multiplication":
                # Should contain tool call and result 56
                if "<tool:" in response and ("56" in response or "fifty" in response):
                    return 1.0
            elif scenario.name == "temperature_conversion":
                # Should contain conversion tool and reasonable celsius value
                if "<tool:convert" in response and any(temp in response for temp in ["37", "38", "celsius"]):
                    return 1.0
            elif scenario.name == "word_count":
                # Should count 3 words
                if "<tool:text" in response and "3" in response:
                    return 1.0
            
            # Partial credit if tool was called but result is unclear
            return 0.5 if "<tool:" in response else 0.0
            
        elif scenario.category == "tool_miss":
            # Should NOT call tools, should use thinking
            if "<tool:" in response:
                return 0.0  # Wrong - called tool when shouldn't have
            elif "<thinking>" in response or len(response) > 20:
                return 1.0  # Correct - provided reasoning/explanation
            else:
                return 0.3  # Partial - provided some response without tools
                
        elif scenario.category == "tool_error":
            # Should handle error gracefully without crashing
            if "<thinking>" in response and "error" not in response_lower:
                return 1.0  # Handled gracefully
            elif len(response) > 10 and "error" not in response_lower:
                return 0.7  # Provided response
            else:
                return 0.3  # At least didn't crash
                
        elif scenario.category == "multi_tool":
            # Should make multiple tool calls
            tool_calls = response.count("<tool:")
            if tool_calls >= 2:
                return 1.0
            elif tool_calls == 1:
                return 0.5
            else:
                return 0.0
        
        return 0.5  # Default partial credit
    
    @staticmethod
    def extract_tools_used(response: str) -> List[str]:
        """Extract tool names from response"""
        import re
        
        tool_pattern = r'<tool:([^>]+)>'
        matches = re.findall(tool_pattern, response)
        
        return [match.split('(')[0] for match in matches]  # Extract just function name


class BenchmarkRunner:
    """Main benchmark runner orchestrating all tests"""
    
    def __init__(self, 
                 server_url: str = "http://localhost:11434",
                 output_dir: str = "benchmarks/results",
                 timeout: int = 30,
                 warmup_runs: int = 3):
        self.server_url = server_url
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.warmup_runs = warmup_runs
        self.evaluator = BenchmarkEvaluator()
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_warmup(self, client: InferenceClient) -> None:
        """Run warmup requests to stabilize performance"""
        self.logger.info(f"Running {self.warmup_runs} warmup requests...")
        
        warmup_prompt = "What is 2 + 2?"
        
        for i in range(self.warmup_runs):
            try:
                await client.generate(warmup_prompt, max_tokens=20, timeout=10)
                self.logger.debug(f"Warmup {i+1}/{self.warmup_runs} completed")
            except Exception as e:
                self.logger.warning(f"Warmup request {i+1} failed: {e}")
    
    async def run_single_benchmark(self, 
                                  client: InferenceClient,
                                  scenario: BenchmarkScenario) -> BenchmarkResult:
        """Run a single benchmark scenario"""
        self.logger.debug(f"Running benchmark: {scenario.name}")
        
        try:
            response_text, latency_ms, metadata = await client.generate(
                scenario.input_text,
                scenario.max_tokens,
                self.timeout
            )
            
            # Evaluate results
            token_accuracy = self.evaluator.evaluate_token_accuracy(
                response_text, scenario.expected_tokens
            )
            correctness_score = self.evaluator.evaluate_correctness(
                scenario, response_text
            )
            tools_detected = self.evaluator.extract_tools_used(response_text)
            
            return BenchmarkResult(
                scenario_name=scenario.name,
                success=True,
                latency_ms=latency_ms,
                tokens_generated=metadata.get("tokens_generated", 0),
                token_accuracy=token_accuracy,
                correctness_score=correctness_score,
                response_text=response_text,
                tools_detected=tools_detected,
                expected_tools=scenario.expected_tokens
            )
            
        except Exception as e:
            self.logger.error(f"Benchmark {scenario.name} failed: {e}")
            return BenchmarkResult(
                scenario_name=scenario.name,
                success=False,
                latency_ms=0.0,
                tokens_generated=0,
                token_accuracy=0.0,
                correctness_score=0.0,
                response_text="",
                error_message=str(e)
            )
    
    async def run_all_benchmarks(self, 
                                scenarios: Optional[List[BenchmarkScenario]] = None,
                                iterations: int = 1) -> List[BenchmarkResult]:
        """Run all benchmark scenarios"""
        if scenarios is None:
            scenarios = BenchmarkScenarios.get_all_scenarios()
        
        self.logger.info(f"Running {len(scenarios)} scenarios with {iterations} iterations each")
        
        all_results = []
        
        async with InferenceClient(self.server_url) as client:
            # Warmup
            await self.run_warmup(client)
            
            # Run benchmarks
            for iteration in range(iterations):
                if iterations > 1:
                    self.logger.info(f"Starting iteration {iteration + 1}/{iterations}")
                
                for scenario in scenarios:
                    result = await self.run_single_benchmark(client, scenario)
                    all_results.append(result)
                    
                    if result.success:
                        self.logger.info(
                            f"✓ {scenario.name}: {result.latency_ms:.1f}ms, "
                            f"accuracy: {result.token_accuracy:.2f}, "
                            f"correctness: {result.correctness_score:.2f}"
                        )
                    else:
                        self.logger.error(f"✗ {scenario.name}: {result.error_message}")
        
        return all_results
    
    def analyze_results(self, results: List[BenchmarkResult]) -> BenchmarkSummary:
        """Analyze benchmark results and generate summary"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            raise Exception("No successful benchmark runs to analyze")
        
        latencies = [r.latency_ms for r in successful_results]
        token_accuracies = [r.token_accuracy for r in successful_results]
        correctness_scores = [r.correctness_score for r in successful_results]
        
        # Calculate category performance
        category_performance = {}
        for category in ["tool_hit", "tool_miss", "tool_error", "multi_tool"]:
            category_results = [r for r in successful_results 
                              if any(s.name == r.scenario_name and s.category == category 
                                   for s in BenchmarkScenarios.get_all_scenarios())]
            
            if category_results:
                category_performance[category] = {
                    "count": len(category_results),
                    "avg_latency_ms": statistics.mean([r.latency_ms for r in category_results]),
                    "avg_token_accuracy": statistics.mean([r.token_accuracy for r in category_results]),
                    "avg_correctness": statistics.mean([r.correctness_score for r in category_results])
                }
        
        # Calculate tokens per second
        total_tokens = sum(r.tokens_generated for r in successful_results)
        total_time_s = sum(r.latency_ms for r in successful_results) / 1000
        tokens_per_second = total_tokens / total_time_s if total_time_s > 0 else 0
        
        return BenchmarkSummary(
            total_scenarios=len(results),
            successful_runs=len(successful_results),
            total_time_ms=sum(latencies),
            avg_latency_ms=statistics.mean(latencies),
            median_latency_ms=statistics.median(latencies),
            p95_latency_ms=self._percentile(latencies, 95),
            p99_latency_ms=self._percentile(latencies, 99),
            avg_token_accuracy=statistics.mean(token_accuracies),
            avg_correctness_score=statistics.mean(correctness_scores),
            category_performance=category_performance,
            tokens_per_second=tokens_per_second
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def save_results(self, 
                    results: List[BenchmarkResult], 
                    summary: BenchmarkSummary,
                    filename_prefix: str = "benchmark") -> None:
        """Save results to JSON files"""
        timestamp = int(time.time())
        
        # Save detailed results
        results_file = self.output_dir / f"{filename_prefix}_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / f"{filename_prefix}_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Summary saved to {summary_file}")
    
    def print_summary(self, summary: BenchmarkSummary) -> None:
        """Print benchmark summary to console"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Total Scenarios: {summary.total_scenarios}")
        print(f"Successful Runs: {summary.successful_runs}")
        print(f"Success Rate: {summary.successful_runs/summary.total_scenarios*100:.1f}%")
        print(f"")
        print(f"Performance Metrics:")
        print(f"  Average Latency: {summary.avg_latency_ms:.1f}ms")
        print(f"  Median Latency:  {summary.median_latency_ms:.1f}ms")
        print(f"  95th Percentile: {summary.p95_latency_ms:.1f}ms")
        print(f"  99th Percentile: {summary.p99_latency_ms:.1f}ms")
        print(f"  Tokens/Second:   {summary.tokens_per_second:.1f}")
        print(f"")
        print(f"Accuracy Metrics:")
        print(f"  Token Accuracy:  {summary.avg_token_accuracy:.3f}")
        print(f"  Correctness:     {summary.avg_correctness_score:.3f}")
        print(f"")
        print(f"Category Performance:")
        for category, metrics in summary.category_performance.items():
            print(f"  {category.replace('_', ' ').title()}:")
            print(f"    Count:      {metrics['count']}")
            print(f"    Latency:    {metrics['avg_latency_ms']:.1f}ms")
            print(f"    Accuracy:   {metrics['avg_token_accuracy']:.3f}")
            print(f"    Correctness: {metrics['avg_correctness']:.3f}")


async def main():
    """Main entry point for benchmark runner"""
    parser = argparse.ArgumentParser(description="Benchmark Tool-Aware LLaMA3.1")
    parser.add_argument("--server", default="http://localhost:11434", help="Inference server URL")
    parser.add_argument("--output", default="benchmarks/results", help="Output directory")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations per scenario")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup requests")
    parser.add_argument("--scenarios", nargs="+", help="Specific scenarios to run")
    parser.add_argument("--categories", nargs="+", 
                       choices=["tool_hit", "tool_miss", "tool_error", "multi_tool"],
                       help="Specific categories to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize runner
    runner = BenchmarkRunner(
        server_url=args.server,
        output_dir=args.output,
        timeout=args.timeout,
        warmup_runs=args.warmup
    )
    
    # Get scenarios to run
    all_scenarios = BenchmarkScenarios.get_all_scenarios()
    
    if args.scenarios:
        scenarios = [s for s in all_scenarios if s.name in args.scenarios]
    elif args.categories:
        scenarios = [s for s in all_scenarios if s.category in args.categories]
    else:
        scenarios = all_scenarios
    
    if not scenarios:
        print("No matching scenarios found!")
        return
    
    try:
        # Run benchmarks
        results = await runner.run_all_benchmarks(scenarios, args.iterations)
        
        # Analyze results
        summary = runner.analyze_results(results)
        
        # Save and display results
        runner.save_results(results, summary)
        runner.print_summary(summary)
        
    except Exception as e:
        logging.error(f"Benchmark run failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))