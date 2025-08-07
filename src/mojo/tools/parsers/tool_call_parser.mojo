"""
Tool Call Parser

Handles parsing of Universal Thinking Prefix tool calls from model output.
Extracts function names, parameters, and validates format.
"""

from collections import Dict

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
            return self._string_to_float(str_value)
        return default_value
    
    fn get_parameter_as_int(self, key: String, default_value: Int = 0) -> Int:
        """Get parameter as integer"""
        let str_value = self.get_parameter(key)
        if str_value != "":
            return self._string_to_int(str_value)
        return default_value
    
    fn _string_to_float(self, s: String) -> Float32:
        """Convert string to float (simplified implementation)"""
        # Would implement proper string to float conversion
        return 0.0
    
    fn _string_to_int(self, s: String) -> Int:
        """Convert string to int (simplified implementation)"""
        # Would implement proper string to int conversion
        return 0
    
    fn is_valid(self) -> Bool:
        """Check if tool call is valid"""
        return len(self.function_name) > 0

struct ToolCallValidator:
    """Validates tool calls and ensures Universal Thinking Prefix compliance"""
    
    fn validate_thinking_prefix(self, raw_call: String) -> Bool:
        """Ensure call follows Universal Thinking Prefix pattern"""
        # Must start with <thinking> and contain reasoning
        if not raw_call.startswith("<thinking>"):
            return False
        
        # Must contain closing </thinking> before tool call
        if "</thinking>" not in raw_call:
            return False
        
        # Tool call should come after thinking
        let thinking_end = raw_call.find("</thinking>")
        let tool_start = raw_call.find("<tool:")
        
        return tool_start > thinking_end
    
    fn extract_thinking_content(self, raw_call: String) -> String:
        """Extract the thinking content from the call"""
        let start = raw_call.find("<thinking>") + 10
        let end = raw_call.find("</thinking>")
        
        if start > 10 and end > start:
            return raw_call[start:end]
        return ""
    
    fn validate_parameters(self, tool_call: ToolCall) -> Bool:
        """Validate that tool call has required parameters"""
        # Basic validation - would be more sophisticated
        return len(tool_call.function_name) > 0