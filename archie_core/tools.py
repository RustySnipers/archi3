"""
Tool calling and function execution framework for Archie
Handles tool registration, execution, and integration with MCP servers
"""

import json
import logging
import asyncio
import inspect
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import traceback

logger = logging.getLogger(__name__)

class ToolType(Enum):
    """Tool types"""
    BUILTIN = "builtin"
    MCP = "mcp"
    PLUGIN = "plugin"
    SYSTEM = "system"

@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    parameters: Dict[str, Any]
    tool_type: ToolType
    server_name: Optional[str] = None
    function: Optional[Callable] = None
    enabled: bool = True
    permissions: List[str] = None
    rate_limit: Optional[int] = None
    timeout: int = 30

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []

@dataclass
class ToolCall:
    """Tool call request"""
    id: str
    tool_name: str
    arguments: Dict[str, Any]
    caller: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ToolResult:
    """Tool execution result"""
    call_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ToolManager:
    """Manages tool registration and execution"""
    
    def __init__(self, mcp_client=None):
        self.mcp_client = mcp_client
        self.tools: Dict[str, Tool] = {}
        self.builtin_tools: Dict[str, Callable] = {}
        self.execution_stats = {}
        self.rate_limits = {}
        
        # Initialize built-in tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register built-in system tools"""
        
        # System information tool
        self.register_builtin_tool(
            name="get_system_info",
            description="Get system information and status",
            parameters={
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "enum": ["memory", "cpu", "disk", "network", "all"],
                        "description": "System component to query"
                    }
                }
            },
            function=self._get_system_info
        )
        
        # Time and date tool
        self.register_builtin_tool(
            name="get_current_time",
            description="Get current date and time information",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (optional, defaults to system timezone)"
                    },
                    "format": {
                        "type": "string",
                        "description": "Date format (optional)"
                    }
                }
            },
            function=self._get_current_time
        )
        
        # Memory search tool
        self.register_builtin_tool(
            name="search_memory",
            description="Search through Archie's memory for relevant information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["conversation", "task", "knowledge", "all"],
                        "description": "Type of memory to search"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of results"
                    }
                },
                "required": ["query"]
            },
            function=self._search_memory
        )
        
        # Task creation tool
        self.register_builtin_tool(
            name="create_task",
            description="Create a new task for execution",
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Task description"
                    },
                    "priority": {
                        "type": "integer",
                        "default": 5,
                        "description": "Task priority (1-10)"
                    },
                    "data": {
                        "type": "object",
                        "description": "Additional task data"
                    }
                },
                "required": ["description"]
            },
            function=self._create_task
        )
        
        # File operations tool
        self.register_builtin_tool(
            name="file_operations",
            description="Perform file system operations",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "list", "exists", "delete"],
                        "description": "File operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (for write operation)"
                    }
                },
                "required": ["operation", "path"]
            },
            function=self._file_operations
        )
        
        # Web search tool
        self.register_builtin_tool(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results to return"
                    }
                },
                "required": ["query"]
            },
            function=self._web_search
        )
        
        # Weather information tool
        self.register_builtin_tool(
            name="get_weather",
            description="Get weather information for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location (city, state, country)"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "default": "metric",
                        "description": "Temperature units"
                    }
                },
                "required": ["location"]
            },
            function=self._get_weather
        )
    
    def register_builtin_tool(self, name: str, description: str, parameters: Dict[str, Any], function: Callable):
        """Register a built-in tool"""
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            tool_type=ToolType.BUILTIN,
            function=function
        )
        
        self.tools[name] = tool
        self.builtin_tools[name] = function
        self.execution_stats[name] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_execution_time": 0,
            "average_execution_time": 0
        }
        
        logger.info(f"Registered built-in tool: {name}")
    
    async def register_mcp_tools(self):
        """Register tools from MCP servers"""
        if not self.mcp_client:
            logger.warning("No MCP client available for tool registration")
            return
        
        try:
            # Get tools from all connected MCP servers
            available_tools = await self.mcp_client.get_available_tools()
            
            for mcp_tool in available_tools:
                tool = Tool(
                    name=f"{mcp_tool.server_name}.{mcp_tool.name}",
                    description=mcp_tool.description,
                    parameters=mcp_tool.parameters,
                    tool_type=ToolType.MCP,
                    server_name=mcp_tool.server_name,
                    enabled=mcp_tool.enabled
                )
                
                self.tools[tool.name] = tool
                self.execution_stats[tool.name] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "total_execution_time": 0,
                    "average_execution_time": 0
                }
                
                logger.info(f"Registered MCP tool: {tool.name}")
            
            logger.info(f"Registered {len(available_tools)} MCP tools")
            
        except Exception as e:
            logger.error(f"Error registering MCP tools: {e}")
    
    async def get_available_tools(self) -> List[Tool]:
        """Get list of all available tools"""
        # Refresh MCP tools
        if self.mcp_client:
            await self.register_mcp_tools()
        
        return [tool for tool in self.tools.values() if tool.enabled]
    
    async def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a specific tool by name"""
        return self.tools.get(tool_name)
    
    async def execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a tool call"""
        start_time = datetime.now()
        
        try:
            # Get the tool
            tool = await self.get_tool(call.tool_name)
            if not tool:
                return ToolResult(
                    call_id=call.id,
                    success=False,
                    result=None,
                    error=f"Tool not found: {call.tool_name}"
                )
            
            # Check if tool is enabled
            if not tool.enabled:
                return ToolResult(
                    call_id=call.id,
                    success=False,
                    result=None,
                    error=f"Tool disabled: {call.tool_name}"
                )
            
            # Check rate limits
            if not await self._check_rate_limit(call.tool_name, call.caller):
                return ToolResult(
                    call_id=call.id,
                    success=False,
                    result=None,
                    error=f"Rate limit exceeded for tool: {call.tool_name}"
                )
            
            # Execute the tool
            if tool.tool_type == ToolType.BUILTIN:
                result = await self._execute_builtin_tool(tool, call)
            elif tool.tool_type == ToolType.MCP:
                result = await self._execute_mcp_tool(tool, call)
            else:
                return ToolResult(
                    call_id=call.id,
                    success=False,
                    result=None,
                    error=f"Unsupported tool type: {tool.tool_type}"
                )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            await self._update_tool_stats(call.tool_name, True, execution_time)
            
            return ToolResult(
                call_id=call.id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Tool execution error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Update statistics
            await self._update_tool_stats(call.tool_name, False, execution_time)
            
            return ToolResult(
                call_id=call.id,
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def _execute_builtin_tool(self, tool: Tool, call: ToolCall) -> Any:
        """Execute a built-in tool"""
        function = tool.function
        
        if asyncio.iscoroutinefunction(function):
            return await function(call.arguments)
        else:
            return function(call.arguments)
    
    async def _execute_mcp_tool(self, tool: Tool, call: ToolCall) -> Any:
        """Execute an MCP tool"""
        if not self.mcp_client:
            raise Exception("MCP client not available")
        
        # Extract server name and tool name
        server_name = tool.server_name
        tool_name = tool.name.split('.', 1)[-1]  # Remove server prefix
        
        # Call the MCP tool
        result = await self.mcp_client.call_tool(server_name, tool_name, call.arguments)
        return result
    
    async def _check_rate_limit(self, tool_name: str, caller: str) -> bool:
        """Check if tool call is within rate limits"""
        # Simple rate limiting implementation
        # In production, this would be more sophisticated
        
        if tool_name not in self.rate_limits:
            self.rate_limits[tool_name] = {}
        
        if caller not in self.rate_limits[tool_name]:
            self.rate_limits[tool_name][caller] = {
                "calls": 0,
                "last_reset": datetime.now()
            }
        
        rate_info = self.rate_limits[tool_name][caller]
        
        # Reset counters every minute
        if (datetime.now() - rate_info["last_reset"]).seconds >= 60:
            rate_info["calls"] = 0
            rate_info["last_reset"] = datetime.now()
        
        # Check limit (default 10 calls per minute)
        tool = self.tools.get(tool_name)
        limit = tool.rate_limit if tool and tool.rate_limit else 10
        
        if rate_info["calls"] >= limit:
            return False
        
        rate_info["calls"] += 1
        return True
    
    async def _update_tool_stats(self, tool_name: str, success: bool, execution_time: float):
        """Update tool execution statistics"""
        if tool_name not in self.execution_stats:
            self.execution_stats[tool_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_execution_time": 0,
                "average_execution_time": 0
            }
        
        stats = self.execution_stats[tool_name]
        stats["total_calls"] += 1
        stats["total_execution_time"] += execution_time
        
        if success:
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
        
        stats["average_execution_time"] = stats["total_execution_time"] / stats["total_calls"]
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics"""
        return self.execution_stats
    
    # Built-in tool implementations
    
    async def _get_system_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get system information"""
        import psutil
        import platform
        
        component = args.get("component", "all")
        info = {}
        
        if component in ["memory", "all"]:
            memory = psutil.virtual_memory()
            info["memory"] = {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free
            }
        
        if component in ["cpu", "all"]:
            info["cpu"] = {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
        
        if component in ["disk", "all"]:
            disk = psutil.disk_usage('/')
            info["disk"] = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
        
        if component in ["network", "all"]:
            network = psutil.net_io_counters()
            info["network"] = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        
        if component == "all":
            info["platform"] = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            }
        
        return info
    
    async def _get_current_time(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get current time and date"""
        from datetime import datetime
        import pytz
        
        timezone = args.get("timezone")
        format_str = args.get("format", "%Y-%m-%d %H:%M:%S")
        
        if timezone:
            try:
                tz = pytz.timezone(timezone)
                now = datetime.now(tz)
            except Exception:
                now = datetime.now()
        else:
            now = datetime.now()
        
        return {
            "timestamp": now.isoformat(),
            "formatted": now.strftime(format_str),
            "timezone": str(now.tzinfo) if now.tzinfo else "local",
            "unix_timestamp": now.timestamp(),
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "weekday": now.strftime("%A"),
            "month_name": now.strftime("%B")
        }
    
    async def _search_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search memory (placeholder implementation)"""
        # This would integrate with the actual memory manager
        query = args.get("query")
        memory_type = args.get("memory_type", "all")
        limit = args.get("limit", 10)
        
        # Placeholder response
        return {
            "query": query,
            "memory_type": memory_type,
            "results": [],
            "total_found": 0,
            "search_time": 0.1
        }
    
    async def _create_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new task (placeholder implementation)"""
        description = args.get("description")
        priority = args.get("priority", 5)
        data = args.get("data", {})
        
        # This would integrate with the actual task system
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "task_id": task_id,
            "description": description,
            "priority": priority,
            "status": "created",
            "created_at": datetime.now().isoformat()
        }
    
    async def _file_operations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform file operations"""
        import os
        import json
        
        operation = args.get("operation")
        path = args.get("path")
        content = args.get("content")
        
        # Security check - restrict to certain directories
        allowed_dirs = ["/opt/archie/data", "/tmp"]
        if not any(path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
            raise Exception(f"Access denied to path: {path}")
        
        if operation == "read":
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return {"content": f.read(), "size": os.path.getsize(path)}
            else:
                raise Exception(f"File not found: {path}")
        
        elif operation == "write":
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            return {"success": True, "bytes_written": len(content)}
        
        elif operation == "list":
            if os.path.isdir(path):
                items = []
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    items.append({
                        "name": item,
                        "type": "directory" if os.path.isdir(item_path) else "file",
                        "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
                    })
                return {"items": items}
            else:
                raise Exception(f"Directory not found: {path}")
        
        elif operation == "exists":
            return {"exists": os.path.exists(path)}
        
        elif operation == "delete":
            if os.path.exists(path):
                os.remove(path)
                return {"success": True}
            else:
                raise Exception(f"File not found: {path}")
        
        else:
            raise Exception(f"Unknown operation: {operation}")
    
    async def _web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Web search (placeholder implementation)"""
        query = args.get("query")
        num_results = args.get("num_results", 5)
        
        # This would integrate with an actual search API
        return {
            "query": query,
            "results": [
                {
                    "title": f"Search result for: {query}",
                    "url": "https://example.com",
                    "snippet": f"This is a placeholder search result for the query: {query}"
                }
            ],
            "total_results": 1
        }
    
    async def _get_weather(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get weather information (placeholder implementation)"""
        location = args.get("location")
        units = args.get("units", "metric")
        
        # This would integrate with an actual weather API
        return {
            "location": location,
            "temperature": 22 if units == "metric" else 72,
            "units": units,
            "condition": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 10,
            "forecast": "Pleasant weather expected"
        }

# Factory function for creating tool manager
async def create_tool_manager(config: Dict[str, Any], mcp_client=None) -> ToolManager:
    """Create and initialize tool manager"""
    tool_manager = ToolManager(mcp_client)
    
    # Register MCP tools if client is available
    if mcp_client:
        await tool_manager.register_mcp_tools()
    
    logger.info("Tool manager initialized")
    return tool_manager