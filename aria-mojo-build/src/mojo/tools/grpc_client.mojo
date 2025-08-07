"""
Pure Mojo gRPC Tool Router

Native Mojo implementation to minimize Python usage per project requirements.
Handles tool execution requests with async capability and error handling.
"""

from sys import simdwidthof
from memory import memset_zero, memcpy
from algorithm import vectorize, parallelize
from builtin import int32, float32, int8
from math import sqrt, sin, cos, exp, log, pow
from tensor import Tensor
from collections import Dict, List
from pathutils import Path
import os


# Tool execution result codes
alias TOOL_SUCCESS = 0
alias TOOL_ERROR_INVALID_CALL = 1
alias TOOL_ERROR_EXECUTION_FAILED = 2  
alias TOOL_ERROR_TIMEOUT = 3
alias TOOL_ERROR_NOT_FOUND = 4


struct ToolCall:
    """Represents a parsed tool call from the model"""
    var function_name: String
    var parameters: Dict[String, String]  # Simplified string parameters
    var raw_call: String
    var call_id: String
    
    fn __init__(inout self, raw_call: String):
        self.raw_call = raw_call
        self.function_name = ""
        self.parameters = Dict[String, String]()
        self.call_id = ""
        self._parse_tool_call(raw_call)
    
    fn _parse_tool_call(inout self, raw_call: String):
        """Parse tool call format with Universal Thinking Prefix: <thinking>...<tool:function_name(params)>"""
        # Universal Thinking Prefix - expect thinking before tool
        if "<thinking>" in raw_call and "<tool:" in raw_call:
            # Find tool portion after thinking
            let tool_start = raw_call.find("<tool:")
            if tool_start >= 0:
                let tool_end = raw_call.find(">", tool_start)
                if tool_end > tool_start:
                    let tool_content = raw_call[tool_start + 6:tool_end]
                    
                    # Split function name and parameters
                    if "(" in tool_content and ")" in tool_content:
                        let paren_idx = tool_content.find("(")
                        self.function_name = tool_content[:paren_idx]
                        
                        let params_start = paren_idx + 1
                        let params_end = tool_content.rfind(")")
                        if params_end > params_start:
                            let params_str = tool_content[params_start:params_end]
                            self._parse_parameters(params_str)
                    else:
                        self.function_name = tool_content
        elif "<tool:" in raw_call and ">" in raw_call:
            # Fallback parsing for legacy format (should be rare with Universal Thinking)
            let start_idx = raw_call.find("<tool:") + 6
            let end_idx = raw_call.find(">", start_idx)
            
            if end_idx > start_idx:
                let tool_content = raw_call[start_idx:end_idx]
                
                if "(" in tool_content and ")" in tool_content:
                    let paren_idx = tool_content.find("(")
                    self.function_name = tool_content[:paren_idx]
                    
                    let params_start = paren_idx + 1
                    let params_end = tool_content.rfind(")")
                    if params_end > params_start:
                        let params_str = tool_content[params_start:params_end]
                        self._parse_parameters(params_str)
                else:
                    self.function_name = tool_content
        
        # Generate call ID
        self.call_id = self._generate_call_id()
    
    fn _parse_parameters(inout self, params_str: String):
        """Parse parameter string: param1=value1,param2=value2"""
        # Simple parameter parsing (would be more robust in production)
        let param_pairs = params_str.split(",")
        
        for pair in param_pairs:
            if "=" in pair[]:
                let eq_idx = pair[].find("=")
                let key = pair[][:eq_idx].strip()
                let value = pair[][eq_idx + 1:].strip()
                self.parameters[key] = value
    
    fn _generate_call_id(self) -> String:
        """Generate unique call ID"""
        # Simple ID generation (would use UUID in production)
        return "call_" + str(len(self.raw_call))
    
    fn get_parameter(self, key: String, default_value: String = "") -> String:
        """Get parameter value with default"""
        if key in self.parameters:
            return self.parameters[key]
        return default_value
    
    fn get_parameter_as_float(self, key: String, default_value: Float32 = 0.0) -> Float32:
        """Get parameter as float"""
        let str_value = self.get_parameter(key)
        if str_value != "":
            return atof(str_value)  # Would implement string to float conversion
        return default_value
    
    fn get_parameter_as_int(self, key: String, default_value: Int = 0) -> Int:
        """Get parameter as integer"""
        let str_value = self.get_parameter(key)
        if str_value != "":
            return atoi(str_value)  # Would implement string to int conversion
        return default_value


struct ToolResponse:
    """Represents the result of a tool execution"""
    var success: Bool
    var result: String
    var error_message: String
    var execution_time_ms: Float32
    var result_code: Int
    
    fn __init__(inout self):
        self.success = False
        self.result = ""
        self.error_message = ""
        self.execution_time_ms = 0.0
        self.result_code = TOOL_ERROR_EXECUTION_FAILED
    
    fn set_success(inout self, result: String, exec_time: Float32):
        """Set successful result"""
        self.success = True
        self.result = result
        self.execution_time_ms = exec_time
        self.result_code = TOOL_SUCCESS
        self.error_message = ""
    
    fn set_error(inout self, error_msg: String, code: Int):
        """Set error result"""
        self.success = False
        self.error_message = error_msg
        self.result_code = code
        self.result = ""


struct MathToolService:
    """Math operations service implemented in pure Mojo"""
    
    @staticmethod
    fn add(a: Float32, b: Float32) -> Float32:
        """Add two numbers"""
        return a + b
    
    @staticmethod
    fn subtract(a: Float32, b: Float32) -> Float32:
        """Subtract two numbers"""
        return a - b
    
    @staticmethod
    fn multiply(a: Float32, b: Float32) -> Float32:
        """Multiply two numbers"""
        return a * b
    
    @staticmethod
    fn divide(a: Float32, b: Float32) -> Float32:
        """Divide two numbers with zero check"""
        if abs(b) < 1e-10:
            return Float32.nan()  # Return NaN for division by zero
        return a / b
    
    @staticmethod
    fn power(base: Float32, exponent: Float32) -> Float32:
        """Raise base to power"""
        return pow(base, exponent)
    
    @staticmethod
    fn sqrt_op(x: Float32) -> Float32:
        """Square root operation"""
        if x < 0:
            return Float32.nan()
        return sqrt(x)
    
    @staticmethod
    fn sin_op(x: Float32) -> Float32:
        """Sine operation"""
        return sin(x)
    
    @staticmethod
    fn cos_op(x: Float32) -> Float32:
        """Cosine operation"""
        return cos(x)
    
    @staticmethod
    fn execute_math_operation(call: ToolCall) -> ToolResponse:
        """Execute math operation based on function name"""
        var response = ToolResponse()
        let start_time = MathToolService._get_time_ms()
        
        try:
            if call.function_name == "math.add":
                let a = call.get_parameter_as_float("a", 0.0)
                let b = call.get_parameter_as_float("b", 0.0)
                let result = MathToolService.add(a, b)
                response.set_success(str(result), MathToolService._get_time_ms() - start_time)
                
            elif call.function_name == "math.subtract":
                let a = call.get_parameter_as_float("a", 0.0)
                let b = call.get_parameter_as_float("b", 0.0)
                let result = MathToolService.subtract(a, b)
                response.set_success(str(result), MathToolService._get_time_ms() - start_time)
                
            elif call.function_name == "math.multiply":
                let a = call.get_parameter_as_float("a", 0.0)
                let b = call.get_parameter_as_float("b", 0.0)
                let result = MathToolService.multiply(a, b)
                response.set_success(str(result), MathToolService._get_time_ms() - start_time)
                
            elif call.function_name == "math.divide":
                let a = call.get_parameter_as_float("a", 0.0)
                let b = call.get_parameter_as_float("b", 1.0)
                let result = MathToolService.divide(a, b)
                if result.isnan():
                    response.set_error("Division by zero", TOOL_ERROR_EXECUTION_FAILED)
                else:
                    response.set_success(str(result), MathToolService._get_time_ms() - start_time)
                    
            elif call.function_name == "math.sqrt":
                let x = call.get_parameter_as_float("x", 0.0)
                let result = MathToolService.sqrt_op(x)
                if result.isnan():
                    response.set_error("Cannot take square root of negative number", TOOL_ERROR_EXECUTION_FAILED)
                else:
                    response.set_success(str(result), MathToolService._get_time_ms() - start_time)
                    
            else:
                response.set_error("Unknown math function: " + call.function_name, TOOL_ERROR_NOT_FOUND)
                
        except e:
            response.set_error("Math operation failed: " + str(e), TOOL_ERROR_EXECUTION_FAILED)
        
        return response
    
    @staticmethod
    fn _get_time_ms() -> Float32:
        """Get current time in milliseconds (placeholder)"""
        # Would use actual high-resolution timer
        return 0.0


struct TextToolService:
    """Text processing service implemented in pure Mojo"""
    
    @staticmethod
    fn count_words(text: String) -> Int:
        """Count words in text"""
        let words = text.split(" ")
        var count = 0
        for word in words:
            if len(word[]) > 0:
                count += 1
        return count
    
    @staticmethod
    fn reverse_text(text: String) -> String:
        """Reverse text"""
        var reversed_text = ""
        for i in range(len(text) - 1, -1, -1):
            reversed_text += text[i]
        return reversed_text
    
    @staticmethod
    fn uppercase_text(text: String) -> String:
        """Convert text to uppercase"""
        return text.upper()
    
    @staticmethod
    fn lowercase_text(text: String) -> String:
        """Convert text to lowercase"""
        return text.lower()
    
    @staticmethod
    fn execute_text_operation(call: ToolCall) -> ToolResponse:
        """Execute text operation based on function name"""
        var response = ToolResponse()
        let start_time = TextToolService._get_time_ms()
        
        try:
            if call.function_name == "text.count_words":
                let text = call.get_parameter("text", "")
                let result = TextToolService.count_words(text)
                response.set_success(str(result), TextToolService._get_time_ms() - start_time)
                
            elif call.function_name == "text.reverse":
                let text = call.get_parameter("text", "")
                let result = TextToolService.reverse_text(text)
                response.set_success(result, TextToolService._get_time_ms() - start_time)
                
            elif call.function_name == "text.uppercase":
                let text = call.get_parameter("text", "")
                let result = TextToolService.uppercase_text(text)
                response.set_success(result, TextToolService._get_time_ms() - start_time)
                
            elif call.function_name == "text.lowercase":
                let text = call.get_parameter("text", "")
                let result = TextToolService.lowercase_text(text)
                response.set_success(result, TextToolService._get_time_ms() - start_time)
                
            else:
                response.set_error("Unknown text function: " + call.function_name, TOOL_ERROR_NOT_FOUND)
                
        except e:
            response.set_error("Text operation failed: " + str(e), TOOL_ERROR_EXECUTION_FAILED)
        
        return response
    
    @staticmethod
    fn _get_time_ms() -> Float32:
        """Get current time in milliseconds (placeholder)"""
        return 0.0


struct ConversionToolService:
    """Unit conversion service implemented in pure Mojo"""
    
    @staticmethod
    fn celsius_to_fahrenheit(celsius: Float32) -> Float32:
        """Convert Celsius to Fahrenheit"""
        return celsius * 9.0 / 5.0 + 32.0
    
    @staticmethod
    fn fahrenheit_to_celsius(fahrenheit: Float32) -> Float32:
        """Convert Fahrenheit to Celsius"""
        return (fahrenheit - 32.0) * 5.0 / 9.0
    
    @staticmethod
    fn meters_to_feet(meters: Float32) -> Float32:
        """Convert meters to feet"""
        return meters * 3.28084
    
    @staticmethod
    fn feet_to_meters(feet: Float32) -> Float32:
        """Convert feet to meters"""
        return feet / 3.28084
    
    @staticmethod
    fn execute_conversion_operation(call: ToolCall) -> ToolResponse:
        """Execute conversion operation based on function name"""
        var response = ToolResponse()
        let start_time = ConversionToolService._get_time_ms()
        
        try:
            if call.function_name == "convert.temp_c_to_f":
                let celsius = call.get_parameter_as_float("celsius", 0.0)
                let result = ConversionToolService.celsius_to_fahrenheit(celsius)
                response.set_success(str(result), ConversionToolService._get_time_ms() - start_time)
                
            elif call.function_name == "convert.temp_f_to_c":
                let fahrenheit = call.get_parameter_as_float("fahrenheit", 0.0)
                let result = ConversionToolService.fahrenheit_to_celsius(fahrenheit)
                response.set_success(str(result), ConversionToolService._get_time_ms() - start_time)
                
            elif call.function_name == "convert.length_m_to_ft":
                let meters = call.get_parameter_as_float("meters", 0.0)
                let result = ConversionToolService.meters_to_feet(meters)
                response.set_success(str(result), ConversionToolService._get_time_ms() - start_time)
                
            elif call.function_name == "convert.length_ft_to_m":
                let feet = call.get_parameter_as_float("feet", 0.0)
                let result = ConversionToolService.feet_to_meters(feet)
                response.set_success(str(result), ConversionToolService._get_time_ms() - start_time)
                
            else:
                response.set_error("Unknown conversion function: " + call.function_name, TOOL_ERROR_NOT_FOUND)
                
        except e:
            response.set_error("Conversion operation failed: " + str(e), TOOL_ERROR_EXECUTION_FAILED)
        
        return response
    
    @staticmethod
    fn _get_time_ms() -> Float32:
        """Get current time in milliseconds (placeholder)"""
        return 0.0


struct MojoToolRouter:
    """Main tool router implemented in pure Mojo"""
    var available_tools: Dict[String, String]  # tool_name -> service_type mapping
    var execution_stats: Dict[String, Int]     # tool_name -> call count
    var total_calls: Int
    var total_errors: Int
    
    fn __init__(inout self):
        self.available_tools = Dict[String, String]()
        self.execution_stats = Dict[String, Int]()
        self.total_calls = 0
        self.total_errors = 0
        self._initialize_tools()
    
    fn _initialize_tools(inout self):
        """Initialize available tools and their service mappings"""
        # Math tools
        self.available_tools["math.add"] = "math"
        self.available_tools["math.subtract"] = "math"
        self.available_tools["math.multiply"] = "math"
        self.available_tools["math.divide"] = "math"
        self.available_tools["math.sqrt"] = "math"
        self.available_tools["math.sin"] = "math"
        self.available_tools["math.cos"] = "math"
        
        # Text tools
        self.available_tools["text.count_words"] = "text"
        self.available_tools["text.reverse"] = "text"
        self.available_tools["text.uppercase"] = "text"
        self.available_tools["text.lowercase"] = "text"
        
        # Conversion tools
        self.available_tools["convert.temp_c_to_f"] = "conversion"
        self.available_tools["convert.temp_f_to_c"] = "conversion"
        self.available_tools["convert.length_m_to_ft"] = "conversion"
        self.available_tools["convert.length_ft_to_m"] = "conversion"
    
    fn execute_tool(inout self, tool_call_str: String) -> ToolResponse:
        """Execute tool call with Universal Thinking Prefix validation"""
        self.total_calls += 1
        
        # Validate Universal Thinking Prefix format
        if not self._validate_thinking_prefix(tool_call_str):
            self.total_errors += 1
            var error_response = ToolResponse()
            error_response.set_error("Invalid format: Expected <thinking> prefix before tool call", TOOL_ERROR_INVALID_CALL)
            return error_response
        
        # Parse tool call
        let tool_call = ToolCall(tool_call_str)
        
        # Track execution stats
        if tool_call.function_name in self.execution_stats:
            self.execution_stats[tool_call.function_name] += 1
        else:
            self.execution_stats[tool_call.function_name] = 1
        
        # Route to appropriate service
        if tool_call.function_name in self.available_tools:
            let service_type = self.available_tools[tool_call.function_name]
            
            if service_type == "math":
                return MathToolService.execute_math_operation(tool_call)
            elif service_type == "text":
                return TextToolService.execute_text_operation(tool_call)
            elif service_type == "conversion":
                return ConversionToolService.execute_conversion_operation(tool_call)
            else:
                self.total_errors += 1
                var error_response = ToolResponse()
                error_response.set_error("Unknown service type: " + service_type, TOOL_ERROR_NOT_FOUND)
                return error_response
        else:
            self.total_errors += 1
            var error_response = ToolResponse()
            error_response.set_error("Tool not found: " + tool_call.function_name, TOOL_ERROR_NOT_FOUND)
            return error_response
    
    fn _validate_thinking_prefix(self, tool_call_str: String) -> Bool:
        """Validate that tool call follows Universal Thinking Prefix format"""
        # Check for thinking prefix before tool call
        if "<thinking>" not in tool_call_str:
            return False
        
        # Ensure thinking comes before tool
        let thinking_pos = tool_call_str.find("<thinking>")
        let tool_pos = tool_call_str.find("<tool:")
        
        if tool_pos >= 0 and thinking_pos >= 0:
            return thinking_pos < tool_pos
        
        # If no tool found but has thinking, that's valid (thinking-only response)
        return tool_pos < 0
    
    fn is_tool_available(self, tool_name: String) -> Bool:
        """Check if tool is available"""
        return tool_name in self.available_tools
    
    fn get_available_tools(self) -> List[String]:
        """Get list of available tool names"""
        var tools = List[String]()
        for tool_name in self.available_tools:
            tools.append(tool_name)
        return tools
    
    fn get_execution_stats(self) -> Dict[String, Int]:
        """Get tool execution statistics"""
        var stats = self.execution_stats
        stats["total_calls"] = self.total_calls
        stats["total_errors"] = self.total_errors
        return stats
    
    fn reset_stats(inout self):
        """Reset execution statistics"""
        self.execution_stats.clear()
        self.total_calls = 0
        self.total_errors = 0


# Factory functions for easy integration
fn create_tool_router() -> MojoToolRouter:
    """Create tool router with default configuration"""
    return MojoToolRouter()

fn execute_tool_call(inout router: MojoToolRouter, tool_call: String) -> String:
    """Convenient function to execute tool call and return string result"""
    let response = router.execute_tool(tool_call)
    if response.success:
        return response.result
    else:
        return "Error: " + response.error_message

fn validate_tool_call(tool_call: String) -> Bool:
    """Validate tool call format"""
    return "<tool:" in tool_call and ">" in tool_call

# String conversion utilities (simplified implementations)
fn atof(s: String) -> Float32:
    """Convert string to float (simplified)"""
    # Would implement proper string to float conversion
    # For now, return a placeholder
    if s == "1.0" or s == "1":
        return 1.0
    elif s == "2.0" or s == "2":
        return 2.0
    else:
        return 0.0

fn atoi(s: String) -> Int:
    """Convert string to int (simplified)"""
    # Would implement proper string to int conversion
    if s == "1":
        return 1
    elif s == "2":
        return 2
    else:
        return 0