#!/usr/bin/env python3
"""
HTTP Inference Server with OpenAI-Compatible API

Provides REST API endpoint for tool-aware LLaMA3.1 inference,
compatible with OpenAI's chat completions format.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from .grpc_router import InferenceServer, ToolRouter


# OpenAI-compatible request/response models
class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str  # "stop", "length", "tool_calls"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


@dataclass
class InferenceResult:
    """Result from Mojo inference engine"""
    text: str
    tokens_generated: int
    inference_time_ms: float
    tools_used: List[str]
    finish_reason: str = "stop"


class MojoInferenceWrapper:
    """Wrapper for Mojo inference engine integration"""
    
    def __init__(self, weights_path: str, grpc_endpoint: str):
        self.weights_path = weights_path
        self.grpc_endpoint = grpc_endpoint
        self.tool_router = ToolRouter()
        self.logger = logging.getLogger(__name__)
        
        # Initialize inference engine (would use actual Mojo engine)
        # from src.kernels.inference_engine import create_inference_pipeline
        # self.engine = create_inference_pipeline(weights_path, grpc_endpoint)
        
    async def generate(self, 
                      messages: List[ChatMessage], 
                      max_tokens: int = 512,
                      temperature: float = 0.7,
                      stream: bool = False) -> InferenceResult:
        """Generate response using Mojo inference engine"""
        start_time = time.time()
        
        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)
        
        # Tokenize prompt (would use actual tokenizer)
        # input_tokens = self.engine.tokenize(prompt)
        
        # Generate tokens (would use actual engine)
        # output_tokens = self.engine.generate(input_tokens, max_tokens)
        
        # Mock inference for demonstration
        await asyncio.sleep(0.1)  # Simulate inference time
        
        # Check if prompt suggests tool usage
        tools_used = []
        if any(keyword in prompt.lower() for keyword in ["calculate", "add", "multiply", "convert"]):
            # Simulate tool usage
            if "add" in prompt.lower():
                tools_used.append("math.add")
                response_text = "I'll calculate that for you. <tool:math.add(5,3)><tool_response>8<response>The sum of 5 and 3 is 8."
            elif "convert" in prompt.lower():
                tools_used.append("convert.temp")
                response_text = "I'll convert that temperature for you. <tool:convert.temp(100,'F','C')><tool_response>37.78<response>100°F equals 37.78°C."
            else:
                response_text = f"I understand you want me to perform a calculation. Let me help with that."
        else:
            # Regular response without tools
            response_text = "I'm a tool-aware language model. I can help with calculations, conversions, and text processing tasks."
        
        inference_time = (time.time() - start_time) * 1000
        
        return InferenceResult(
            text=response_text,
            tokens_generated=len(response_text.split()),
            inference_time_ms=inference_time,
            tools_used=tools_used,
            finish_reason="stop"
        )
    
    async def generate_stream(self, 
                             messages: List[ChatMessage], 
                             max_tokens: int = 512,
                             temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        result = await self.generate(messages, max_tokens, temperature, stream=True)
        
        # Simulate streaming by yielding chunks
        words = result.text.split()
        for i, word in enumerate(words):
            if i > 0:
                yield " "
            yield word
            await asyncio.sleep(0.02)  # Simulate streaming delay
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to prompt format"""
        prompt_parts = []
        
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        prompt_parts.append("Assistant:")  # Prompt for next response
        return "\n".join(prompt_parts)


class ToolAwareLLaMAServer:
    """Main server class integrating Mojo inference with HTTP API"""
    
    def __init__(self, 
                 weights_path: str = "weights/model.mojo.bin",
                 grpc_endpoint: str = "localhost:50051"):
        self.weights_path = weights_path
        self.grpc_endpoint = grpc_endpoint
        self.model_name = "llama3.1-8b-tool-aware"
        self.logger = logging.getLogger(__name__)
        
        # Initialize inference wrapper
        self.inference = MojoInferenceWrapper(weights_path, grpc_endpoint)
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with routes"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.logger.info("Starting Tool-Aware LLaMA Server")
            self.logger.info(f"Model: {self.model_name}")
            self.logger.info(f"Weights: {self.weights_path}")
            self.logger.info(f"gRPC Endpoint: {self.grpc_endpoint}")
            yield
            # Shutdown
            self.logger.info("Shutting down server")
        
        app = FastAPI(
            title="Tool-Aware LLaMA3.1 Inference Server",
            description="OpenAI-compatible API for tool-aware language model inference",
            version="1.0.0",
            lifespan=lifespan
        )
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "model": self.model_name}
        
        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "tool-aware-llama"
                    }
                ]
            }
        
        @app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            try:
                if request.stream:
                    return await self._handle_streaming_request(request)
                else:
                    return await self._handle_completion_request(request)
            
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    async def _handle_completion_request(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle non-streaming completion request"""
        start_time = time.time()
        
        # Generate response using Mojo inference
        result = await self.inference.generate(
            messages=request.messages,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature or 0.7,
            stream=False
        )
        
        # Create response
        response_id = f"chatcmpl-{int(time.time() * 1000)}"
        created_time = int(start_time)
        
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(
                role="assistant",
                content=result.text
            ),
            finish_reason=result.finish_reason
        )
        
        usage = {
            "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
            "completion_tokens": result.tokens_generated,
            "total_tokens": sum(len(msg.content.split()) for msg in request.messages) + result.tokens_generated
        }
        
        return ChatCompletionResponse(
            id=response_id,
            created=created_time,
            model=request.model,
            choices=[choice],
            usage=usage
        )
    
    async def _handle_streaming_request(self, request: ChatCompletionRequest) -> StreamingResponse:
        """Handle streaming completion request"""
        
        async def generate_stream():
            response_id = f"chatcmpl-{int(time.time() * 1000)}"
            created_time = int(time.time())
            
            # Generate streaming content
            async for chunk in self.inference.generate_stream(
                messages=request.messages,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature or 0.7
            ):
                stream_response = ChatCompletionStreamResponse(
                    id=response_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": chunk}
                        )
                    ]
                )
                
                yield f"data: {json.dumps(asdict(stream_response))}\n\n"
            
            # Send final chunk with finish_reason
            final_response = ChatCompletionStreamResponse(
                id=response_id,
                created=created_time,
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={},
                        finish_reason="stop"
                    )
                ]
            )
            
            yield f"data: {json.dumps(asdict(final_response))}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain"
        )
    
    def run(self, host: str = "localhost", port: int = 11434, **kwargs):
        """Run the server"""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            **kwargs
        )


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tool-Aware LLaMA3.1 Inference Server")
    parser.add_argument("--weights", default="weights/model.mojo.bin", help="Path to model weights")
    parser.add_argument("--grpc-endpoint", default="localhost:50051", help="gRPC tool service endpoint")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=11434, help="Server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run server
    server = ToolAwareLLaMAServer(
        weights_path=args.weights,
        grpc_endpoint=args.grpc_endpoint
    )
    
    server.run(
        host=args.host,
        port=args.port,
        workers=args.workers
    )


if __name__ == "__main__":
    main()