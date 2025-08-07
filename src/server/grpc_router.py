#!/usr/bin/env python3
"""
gRPC Tool Router for LLaMA3.1 Tool-Aware Inference

Handles tool execution requests from Mojo inference engine,
routing to appropriate services and managing tool responses.
"""

import grpc
import json
import asyncio
import logging
from concurrent import futures
from typing import Dict, Any, Optional
from dataclasses import dataclass
import math
import re

# Import generated gRPC stubs (would be generated from .proto files)
# import tool_service_pb2
# import tool_service_pb2_grpc


@dataclass
class ToolCall:
    """Represents a parsed tool call from the model"""
    function_name: str
    parameters: Dict[str, Any]
    raw_call: str


@dataclass  
class ToolResponse:
    """Represents the result of a tool execution"""
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


class MathToolService:
    """Math operations service"""
    
    @staticmethod
    def add(a: float, b: float) -> float:
        return a + b
    
    @staticmethod
    def subtract(a: float, b: float) -> float:
        return a - b
    
    @staticmethod
    def multiply(a: float, b: float) -> float:
        return a * b
    
    @staticmethod
    def divide(a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    
    @staticmethod
    def sqrt(x: float) -> float:
        if x < 0:
            raise ValueError("Square root of negative number")
        return math.sqrt(x)


class ConversionToolService:
    """Unit conversion service"""
    
    @staticmethod
    def temp(value: float, from_unit: str, to_unit: str) -> float:
        """Temperature conversion"""
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
    
    @staticmethod
    def length(value: float, from_unit: str, to_unit: str) -> float:
        """Length conversion"""
        to_meters = {"m": 1, "cm": 0.01, "mm": 0.001, "km": 1000, "in": 0.0254, "ft": 0.3048}
        
        if from_unit.lower() not in to_meters or to_unit.lower() not in to_meters:
            raise ValueError(f"Unsupported units: {from_unit} -> {to_unit}")
        
        meters = value * to_meters[from_unit.lower()]
        return meters / to_meters[to_unit.lower()]


class TextToolService:
    """Text processing service"""
    
    @staticmethod
    def count_words(text: str) -> int:
        return len(text.split())
    
    @staticmethod
    def reverse(text: str) -> str:
        return text[::-1]
    
    @staticmethod
    def uppercase(text: str) -> str:
        return text.upper()
    
    @staticmethod
    def lowercase(text: str) -> str:
        return text.lower()


class ToolParser:
    """Parses tool calls from model output"""
    
    @staticmethod
    def parse_tool_call(tool_call_str: str) -> Optional[ToolCall]:
        """
        Parse tool call string like '<tool:math.add(1,2)>' 
        Returns ToolCall object or None if parsing fails
        """
        # Remove <tool: and > brackets
        clean_call = tool_call_str.strip()
        if clean_call.startswith('<tool:') and clean_call.endswith('>'):
            clean_call = clean_call[6:-1]  # Remove '<tool:' and '>'
        
        # Extract function name and parameters
        match = re.match(r'([a-zA-Z_][a-zA-Z0-9_.]*)\((.*)\)$', clean_call)
        if not match:
            return None
        
        function_name = match.group(1)
        params_str = match.group(2)
        
        # Parse parameters (simplified - handles basic types)
        parameters = {}
        if params_str.strip():
            try:
                # Split by comma and parse each parameter
                param_parts = []
                current_param = ""
                paren_count = 0
                quote_char = None
                
                for char in params_str:
                    if quote_char:
                        current_param += char
                        if char == quote_char and (not current_param.endswith('\\' + quote_char)):
                            quote_char = None
                    elif char in ['"', "'"]:
                        quote_char = char
                        current_param += char
                    elif char == '(':
                        paren_count += 1
                        current_param += char
                    elif char == ')':
                        paren_count -= 1
                        current_param += char
                    elif char == ',' and paren_count == 0:
                        param_parts.append(current_param.strip())
                        current_param = ""
                    else:
                        current_param += char
                
                if current_param.strip():
                    param_parts.append(current_param.strip())
                
                # Convert to appropriate types
                for i, param in enumerate(param_parts):
                    param = param.strip()
                    if param.startswith('"') and param.endswith('"'):
                        parameters[f"arg_{i}"] = param[1:-1]  # String
                    elif param.startswith("'") and param.endswith("'"):
                        parameters[f"arg_{i}"] = param[1:-1]  # String
                    elif param.lower() == 'true':
                        parameters[f"arg_{i}"] = True
                    elif param.lower() == 'false':
                        parameters[f"arg_{i}"] = False
                    elif '.' in param:
                        try:
                            parameters[f"arg_{i}"] = float(param)
                        except ValueError:
                            parameters[f"arg_{i}"] = param  # Keep as string
                    else:
                        try:
                            parameters[f"arg_{i}"] = int(param)
                        except ValueError:
                            parameters[f"arg_{i}"] = param  # Keep as string
                            
            except Exception as e:
                logging.warning(f"Failed to parse parameters '{params_str}': {e}")
                return None
        
        return ToolCall(
            function_name=function_name,
            parameters=parameters,
            raw_call=tool_call_str
        )


class ToolRouter:
    """Routes tool calls to appropriate services"""
    
    def __init__(self):
        self.services = {
            "math": MathToolService(),
            "convert": ConversionToolService(),
            "text": TextToolService()
        }
        self.parser = ToolParser()
        self.logger = logging.getLogger(__name__)
    
    async def execute_tool(self, tool_call_str: str) -> ToolResponse:
        """Execute a tool call and return the response"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Parse the tool call
            tool_call = self.parser.parse_tool_call(tool_call_str)
            if not tool_call:
                return ToolResponse(
                    success=False,
                    result=None,
                    error_message=f"Failed to parse tool call: {tool_call_str}"
                )
            
            # Extract service and method
            parts = tool_call.function_name.split('.')
            if len(parts) != 2:
                return ToolResponse(
                    success=False,
                    result=None,
                    error_message=f"Invalid function name format: {tool_call.function_name}"
                )
            
            service_name, method_name = parts
            
            # Get service
            if service_name not in self.services:
                return ToolResponse(
                    success=False,
                    result=None,
                    error_message=f"Unknown service: {service_name}"
                )
            
            service = self.services[service_name]
            
            # Get method
            if not hasattr(service, method_name):
                return ToolResponse(
                    success=False,
                    result=None,
                    error_message=f"Unknown method: {service_name}.{method_name}"
                )
            
            method = getattr(service, method_name)
            
            # Execute method with parameters
            if tool_call.parameters:
                # Convert parameter dict to positional args (simplified)
                args = [tool_call.parameters[key] for key in sorted(tool_call.parameters.keys())]
                result = method(*args)
            else:
                result = method()
            
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return ToolResponse(
                success=True,
                result=result,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self.logger.error(f"Tool execution error: {e}")
            
            return ToolResponse(
                success=False,
                result=None,
                error_message=str(e),
                execution_time_ms=execution_time
            )


class ToolServicer:
    """gRPC service implementation for tool routing"""
    
    def __init__(self):
        self.router = ToolRouter()
        self.logger = logging.getLogger(__name__)
    
    async def ExecuteTool(self, request, context):
        """Execute a tool call via gRPC"""
        self.logger.info(f"Executing tool: {request.tool_call}")
        
        response = await self.router.execute_tool(request.tool_call)
        
        # Convert to gRPC response (would use actual protobuf message)
        return {
            "success": response.success,
            "result": str(response.result) if response.result is not None else "",
            "error_message": response.error_message or "",
            "execution_time_ms": response.execution_time_ms
        }


class InferenceServer:
    """Main inference server with integrated tool routing"""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 11434,
                 grpc_port: int = 50051):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.tool_router = ToolRouter()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Mojo inference engine (would integrate with actual engine)
        # self.inference_engine = create_inference_pipeline(
        #     weights_path="weights/model.mojo.bin",
        #     grpc_endpoint=f"localhost:{grpc_port}"
        # )
    
    async def start_server(self):
        """Start both HTTP inference server and gRPC tool service"""
        # Start gRPC server for tool routing
        grpc_server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # Add tool service (would use actual protobuf service)
        tool_servicer = ToolServicer()
        # tool_service_pb2_grpc.add_ToolServiceServicer_to_server(tool_servicer, grpc_server)
        
        listen_addr = f"[::]:{self.grpc_port}"
        grpc_server.add_insecure_port(listen_addr)
        
        self.logger.info(f"Starting gRPC tool service on {listen_addr}")
        await grpc_server.start()
        
        # Start HTTP server for inference endpoint (simplified)
        # Would integrate with FastAPI or similar for OpenAI-compatible endpoint
        self.logger.info(f"Starting inference server on {self.host}:{self.port}")
        
        try:
            await grpc_server.wait_for_termination()
        except KeyboardInterrupt:
            self.logger.info("Shutting down servers...")
            await grpc_server.stop(5)
    
    async def generate_response(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Generate response with tool-aware inference"""
        # This would integrate with the Mojo inference engine
        # For now, return a mock response
        
        # Tokenize input
        # input_tokens = self.inference_engine.tokenize(prompt)
        
        # Generate with tool routing
        # output_tokens = self.inference_engine.generate(input_tokens, max_tokens)
        
        # Detokenize output
        # response_text = self.inference_engine.detokenize(output_tokens)
        
        # Mock response for demonstration
        response_text = "This is a mock response from the tool-aware LLaMA3.1 model."
        
        return {
            "text": response_text,
            "tokens_generated": 50,
            "inference_time_ms": 125.5,
            "tools_used": []
        }


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tool_server.log'),
            logging.StreamHandler()
        ]
    )


async def main():
    """Main entry point for the server"""
    setup_logging()
    
    server = InferenceServer()
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())