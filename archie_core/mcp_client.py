"""
MCP (Model Context Protocol) client for Archie
Handles communication with MCP servers for tool integration
"""

import json
import logging
import asyncio
import websockets
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    parameters: Dict[str, Any]
    server_name: str
    enabled: bool = True

@dataclass
class MCPResource:
    """MCP resource definition"""
    uri: str
    name: str
    description: str
    mime_type: str
    server_name: str

@dataclass
class MCPServer:
    """MCP server configuration"""
    name: str
    uri: str
    transport: str  # websocket, stdio, http
    enabled: bool = True
    connection_status: str = "disconnected"
    tools: List[MCPTool] = None
    resources: List[MCPResource] = None
    last_ping: Optional[datetime] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.resources is None:
            self.resources = []

class MCPClient:
    """MCP client for tool integration"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.connections: Dict[str, Any] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.request_id_counter = 0
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        # Initialize message handlers
        self._setup_message_handlers()
    
    def _setup_message_handlers(self):
        """Set up message handlers for different MCP message types"""
        self.message_handlers = {
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tool_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resource_read,
            "notifications/ping": self._handle_ping,
            "notifications/progress": self._handle_progress,
            "logging/message": self._handle_logging
        }
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        self.request_id_counter += 1
        return f"req_{self.request_id_counter}_{uuid.uuid4().hex[:8]}"
    
    async def add_server(self, server: MCPServer):
        """Add an MCP server"""
        self.servers[server.name] = server
        logger.info(f"Added MCP server: {server.name}")
        
        # Attempt to connect
        await self.connect_server(server.name)
    
    async def connect_server(self, server_name: str) -> bool:
        """Connect to an MCP server"""
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not found")
            return False
        
        server = self.servers[server_name]
        
        try:
            if server.transport == "websocket":
                await self._connect_websocket_server(server)
            elif server.transport == "stdio":
                await self._connect_stdio_server(server)
            elif server.transport == "http":
                await self._connect_http_server(server)
            else:
                logger.error(f"Unsupported transport: {server.transport}")
                return False
            
            server.connection_status = "connected"
            logger.info(f"Connected to MCP server: {server_name}")
            
            # Initialize server (get tools and resources)
            await self._initialize_server(server)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to server {server_name}: {e}")
            server.connection_status = "error"
            return False
    
    async def _connect_websocket_server(self, server: MCPServer):
        """Connect to WebSocket MCP server"""
        try:
            websocket = await websockets.connect(server.uri)
            self.connections[server.name] = websocket
            
            # Start message handling task
            asyncio.create_task(self._handle_websocket_messages(server.name, websocket))
            
        except Exception as e:
            logger.error(f"WebSocket connection failed for {server.name}: {e}")
            raise
    
    async def _connect_stdio_server(self, server: MCPServer):
        """Connect to stdio MCP server"""
        # This would typically spawn a subprocess
        # For now, we'll implement a placeholder
        logger.warning(f"STDIO transport not fully implemented for {server.name}")
        pass
    
    async def _connect_http_server(self, server: MCPServer):
        """Connect to HTTP MCP server"""
        # This would set up HTTP client
        # For now, we'll implement a placeholder
        logger.warning(f"HTTP transport not fully implemented for {server.name}")
        pass
    
    async def _initialize_server(self, server: MCPServer):
        """Initialize server by getting tools and resources"""
        try:
            # Get tools
            tools = await self.list_tools(server.name)
            server.tools = tools
            
            # Add tools to global tool registry
            for tool in tools:
                self.tools[f"{server.name}.{tool.name}"] = tool
            
            # Get resources
            resources = await self.list_resources(server.name)
            server.resources = resources
            
            # Add resources to global resource registry
            for resource in resources:
                self.resources[f"{server.name}.{resource.name}"] = resource
            
            logger.info(f"Initialized server {server.name}: {len(tools)} tools, {len(resources)} resources")
            
        except Exception as e:
            logger.error(f"Failed to initialize server {server.name}: {e}")
    
    async def _handle_websocket_messages(self, server_name: str, websocket):
        """Handle incoming WebSocket messages"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(server_name, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {server_name}: {message}")
                except Exception as e:
                    logger.error(f"Error processing message from {server_name}: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WebSocket connection closed for {server_name}")
            self.servers[server_name].connection_status = "disconnected"
        except Exception as e:
            logger.error(f"WebSocket error for {server_name}: {e}")
            self.servers[server_name].connection_status = "error"
    
    async def _process_message(self, server_name: str, message: Dict[str, Any]):
        """Process incoming MCP message"""
        try:
            # Handle responses to our requests
            if "id" in message and message["id"] in self.pending_requests:
                future = self.pending_requests.pop(message["id"])
                if "error" in message:
                    future.set_exception(Exception(f"MCP Error: {message['error']}"))
                else:
                    future.set_result(message.get("result"))
                return
            
            # Handle notifications and other messages
            method = message.get("method")
            if method in self.message_handlers:
                await self.message_handlers[method](server_name, message)
            else:
                logger.warning(f"Unhandled message method: {method}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _send_request(self, server_name: str, method: str, params: Dict[str, Any] = None) -> Any:
        """Send MCP request and wait for response"""
        if server_name not in self.connections:
            raise Exception(f"Not connected to server {server_name}")
        
        request_id = self._generate_request_id()
        
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        try:
            # Send message
            connection = self.connections[server_name]
            if hasattr(connection, 'send'):  # WebSocket
                await connection.send(json.dumps(message))
            else:
                # Handle other transport types
                pass
            
            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
            
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise Exception(f"Request timeout for {method}")
        except Exception as e:
            self.pending_requests.pop(request_id, None)
            raise
    
    async def list_tools(self, server_name: str) -> List[MCPTool]:
        """List tools available on an MCP server"""
        try:
            result = await self._send_request(server_name, "tools/list")
            tools = []
            
            for tool_data in result.get("tools", []):
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    parameters=tool_data.get("inputSchema", {}),
                    server_name=server_name
                )
                tools.append(tool)
            
            return tools
            
        except Exception as e:
            logger.error(f"Failed to list tools for {server_name}: {e}")
            return []
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on an MCP server"""
        try:
            params = {
                "name": tool_name,
                "arguments": arguments
            }
            
            result = await self._send_request(server_name, "tools/call", params)
            return result
            
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on {server_name}: {e}")
            raise
    
    async def list_resources(self, server_name: str) -> List[MCPResource]:
        """List resources available on an MCP server"""
        try:
            result = await self._send_request(server_name, "resources/list")
            resources = []
            
            for resource_data in result.get("resources", []):
                resource = MCPResource(
                    uri=resource_data["uri"],
                    name=resource_data.get("name", ""),
                    description=resource_data.get("description", ""),
                    mime_type=resource_data.get("mimeType", ""),
                    server_name=server_name
                )
                resources.append(resource)
            
            return resources
            
        except Exception as e:
            logger.error(f"Failed to list resources for {server_name}: {e}")
            return []
    
    async def read_resource(self, server_name: str, uri: str) -> Dict[str, Any]:
        """Read a resource from an MCP server"""
        try:
            params = {"uri": uri}
            result = await self._send_request(server_name, "resources/read", params)
            return result
            
        except Exception as e:
            logger.error(f"Failed to read resource {uri} from {server_name}: {e}")
            raise
    
    async def get_available_tools(self) -> List[MCPTool]:
        """Get all available tools from all connected servers"""
        all_tools = []
        
        for server_name, server in self.servers.items():
            if server.connection_status == "connected":
                all_tools.extend(server.tools)
        
        return all_tools
    
    async def get_tool_by_name(self, tool_name: str) -> Optional[MCPTool]:
        """Get a specific tool by name"""
        # Try exact match first
        if tool_name in self.tools:
            return self.tools[tool_name]
        
        # Try partial match (without server prefix)
        for full_name, tool in self.tools.items():
            if full_name.endswith(f".{tool_name}"):
                return tool
        
        return None
    
    async def disconnect_server(self, server_name: str):
        """Disconnect from an MCP server"""
        if server_name in self.connections:
            connection = self.connections[server_name]
            
            try:
                if hasattr(connection, 'close'):
                    await connection.close()
            except Exception as e:
                logger.error(f"Error closing connection to {server_name}: {e}")
            
            del self.connections[server_name]
        
        if server_name in self.servers:
            self.servers[server_name].connection_status = "disconnected"
        
        logger.info(f"Disconnected from MCP server: {server_name}")
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        for server_name in list(self.connections.keys()):
            await self.disconnect_server(server_name)
    
    async def get_server_status(self) -> Dict[str, Any]:
        """Get status of all MCP servers"""
        status = {}
        
        for server_name, server in self.servers.items():
            status[server_name] = {
                "connection_status": server.connection_status,
                "transport": server.transport,
                "tools_count": len(server.tools),
                "resources_count": len(server.resources),
                "last_ping": server.last_ping.isoformat() if server.last_ping else None
            }
        
        return status
    
    # Message handlers
    async def _handle_tools_list(self, server_name: str, message: Dict[str, Any]):
        """Handle tools list notification"""
        logger.info(f"Received tools list from {server_name}")
    
    async def _handle_tool_call(self, server_name: str, message: Dict[str, Any]):
        """Handle tool call notification"""
        logger.info(f"Received tool call notification from {server_name}")
    
    async def _handle_resources_list(self, server_name: str, message: Dict[str, Any]):
        """Handle resources list notification"""
        logger.info(f"Received resources list from {server_name}")
    
    async def _handle_resource_read(self, server_name: str, message: Dict[str, Any]):
        """Handle resource read notification"""
        logger.info(f"Received resource read notification from {server_name}")
    
    async def _handle_ping(self, server_name: str, message: Dict[str, Any]):
        """Handle ping notification"""
        if server_name in self.servers:
            self.servers[server_name].last_ping = datetime.now()
        logger.debug(f"Received ping from {server_name}")
    
    async def _handle_progress(self, server_name: str, message: Dict[str, Any]):
        """Handle progress notification"""
        params = message.get("params", {})
        logger.info(f"Progress from {server_name}: {params.get('progress', 0)}%")
    
    async def _handle_logging(self, server_name: str, message: Dict[str, Any]):
        """Handle logging notification"""
        params = message.get("params", {})
        level = params.get("level", "info")
        log_message = params.get("message", "")
        logger.log(getattr(logging, level.upper(), logging.INFO), f"[{server_name}] {log_message}")

# Factory function for creating MCP client
async def create_mcp_client(config: Dict[str, Any]) -> MCPClient:
    """Create and initialize MCP client"""
    client = MCPClient()
    
    # Add configured servers
    mcp_servers = config.get('mcp_servers', [])
    for server_config in mcp_servers:
        server = MCPServer(
            name=server_config['name'],
            uri=server_config['uri'],
            transport=server_config.get('transport', 'websocket'),
            enabled=server_config.get('enabled', True)
        )
        
        if server.enabled:
            await client.add_server(server)
    
    return client