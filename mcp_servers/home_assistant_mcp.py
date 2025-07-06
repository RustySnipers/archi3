"""
Home Assistant MCP Server
Provides MCP interface for Home Assistant integration
"""

import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
import websockets
from websockets.server import serve

logger = logging.getLogger(__name__)

class HomeAssistantMCPServer:
    """MCP server for Home Assistant integration"""
    
    def __init__(self, 
                 ha_url: str = "http://localhost:8123",
                 ha_token: str = None,
                 websocket_port: int = 8765):
        
        self.ha_url = ha_url.rstrip('/')
        self.ha_token = ha_token
        self.websocket_port = websocket_port
        self.session = None
        self.server = None
        self.connected_clients = set()
        
        # Home Assistant state
        self.entities = {}
        self.areas = {}
        self.devices = {}
        self.services = {}
        
        # MCP tools definition
        self.tools = [
            {
                "name": "get_entity_state",
                "description": "Get the current state of a Home Assistant entity",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity ID (e.g., light.living_room)"
                        }
                    },
                    "required": ["entity_id"]
                }
            },
            {
                "name": "call_service",
                "description": "Call a Home Assistant service",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Service domain (e.g., light, switch)"
                        },
                        "service": {
                            "type": "string",
                            "description": "Service name (e.g., turn_on, turn_off)"
                        },
                        "entity_id": {
                            "type": "string",
                            "description": "Target entity ID"
                        },
                        "data": {
                            "type": "object",
                            "description": "Additional service data"
                        }
                    },
                    "required": ["domain", "service"]
                }
            },
            {
                "name": "list_entities",
                "description": "List all available entities in Home Assistant",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Filter by domain (optional)"
                        },
                        "area": {
                            "type": "string",
                            "description": "Filter by area (optional)"
                        }
                    }
                }
            },
            {
                "name": "get_areas",
                "description": "Get list of all areas in Home Assistant",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_services",
                "description": "Get list of available services",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Filter by domain (optional)"
                        }
                    }
                }
            },
            {
                "name": "get_history",
                "description": "Get entity state history",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity ID"
                        },
                        "start_time": {
                            "type": "string",
                            "description": "Start time (ISO format)"
                        },
                        "end_time": {
                            "type": "string",
                            "description": "End time (ISO format, optional)"
                        }
                    },
                    "required": ["entity_id"]
                }
            },
            {
                "name": "set_entity_state",
                "description": "Set entity state (for testing/automation)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The entity ID"
                        },
                        "state": {
                            "type": "string",
                            "description": "New state value"
                        },
                        "attributes": {
                            "type": "object",
                            "description": "Entity attributes"
                        }
                    },
                    "required": ["entity_id", "state"]
                }
            }
        ]
        
        # MCP resources definition
        self.resources = [
            {
                "uri": "homeassistant://entities",
                "name": "All Entities",
                "description": "List of all Home Assistant entities",
                "mimeType": "application/json"
            },
            {
                "uri": "homeassistant://config",
                "name": "Configuration",
                "description": "Home Assistant configuration",
                "mimeType": "application/json"
            },
            {
                "uri": "homeassistant://events",
                "name": "Event Stream",
                "description": "Real-time Home Assistant events",
                "mimeType": "application/json"
            }
        ]
    
    async def initialize(self):
        """Initialize the MCP server"""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.ha_token}"},
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test connection to Home Assistant
            await self._test_connection()
            
            # Load initial data
            await self._load_entities()
            await self._load_areas()
            await self._load_services()
            
            logger.info("Home Assistant MCP server initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Home Assistant MCP server: {e}")
            raise
    
    async def _test_connection(self):
        """Test connection to Home Assistant"""
        try:
            async with self.session.get(f"{self.ha_url}/api/") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Connected to Home Assistant: {data.get('message', 'OK')}")
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to connect to Home Assistant: {e}")
            raise
    
    async def _load_entities(self):
        """Load all entities from Home Assistant"""
        try:
            async with self.session.get(f"{self.ha_url}/api/states") as response:
                if response.status == 200:
                    states = await response.json()
                    for state in states:
                        self.entities[state['entity_id']] = state
                    logger.info(f"Loaded {len(self.entities)} entities")
                else:
                    logger.error(f"Failed to load entities: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error loading entities: {e}")
    
    async def _load_areas(self):
        """Load areas from Home Assistant"""
        try:
            async with self.session.get(f"{self.ha_url}/api/config/area_registry") as response:
                if response.status == 200:
                    areas = await response.json()
                    for area in areas:
                        self.areas[area['area_id']] = area
                    logger.info(f"Loaded {len(self.areas)} areas")
                else:
                    logger.warning(f"Failed to load areas: HTTP {response.status}")
        except Exception as e:
            logger.warning(f"Error loading areas: {e}")
    
    async def _load_services(self):
        """Load available services from Home Assistant"""
        try:
            async with self.session.get(f"{self.ha_url}/api/services") as response:
                if response.status == 200:
                    services = await response.json()
                    self.services = services
                    logger.info(f"Loaded services for {len(services)} domains")
                else:
                    logger.error(f"Failed to load services: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error loading services: {e}")
    
    async def start_server(self):
        """Start the MCP WebSocket server"""
        try:
            self.server = await serve(
                self._handle_client,
                "localhost",
                self.websocket_port
            )
            logger.info(f"Home Assistant MCP server started on port {self.websocket_port}")
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def stop_server(self):
        """Stop the MCP server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        if self.session:
            await self.session.close()
        
        logger.info("Home Assistant MCP server stopped")
    
    async def _handle_client(self, websocket, path):
        """Handle incoming MCP client connections"""
        self.connected_clients.add(websocket)
        logger.info(f"MCP client connected from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self._process_message(data)
                    if response:
                        await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    await websocket.send(json.dumps(error_response))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        }
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("MCP client disconnected")
        finally:
            self.connected_clients.discard(websocket)
    
    async def _process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming MCP message"""
        method = message.get("method")
        params = message.get("params", {})
        request_id = message.get("id")
        
        try:
            if method == "tools/list":
                result = {"tools": self.tools}
            elif method == "tools/call":
                result = await self._call_tool(params)
            elif method == "resources/list":
                result = {"resources": self.resources}
            elif method == "resources/read":
                result = await self._read_resource(params)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }
    
    async def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a Home Assistant tool"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "get_entity_state":
            return await self._get_entity_state(arguments)
        elif tool_name == "call_service":
            return await self._call_service(arguments)
        elif tool_name == "list_entities":
            return await self._list_entities(arguments)
        elif tool_name == "get_areas":
            return await self._get_areas(arguments)
        elif tool_name == "get_services":
            return await self._get_services(arguments)
        elif tool_name == "get_history":
            return await self._get_history(arguments)
        elif tool_name == "set_entity_state":
            return await self._set_entity_state(arguments)
        else:
            raise Exception(f"Unknown tool: {tool_name}")
    
    async def _get_entity_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get entity state"""
        entity_id = args.get("entity_id")
        
        try:
            async with self.session.get(f"{self.ha_url}/api/states/{entity_id}") as response:
                if response.status == 200:
                    state = await response.json()
                    return {"state": state}
                elif response.status == 404:
                    raise Exception(f"Entity not found: {entity_id}")
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            raise Exception(f"Failed to get entity state: {e}")
    
    async def _call_service(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call Home Assistant service"""
        domain = args.get("domain")
        service = args.get("service")
        entity_id = args.get("entity_id")
        data = args.get("data", {})
        
        # Add entity_id to data if provided
        if entity_id:
            data["entity_id"] = entity_id
        
        try:
            async with self.session.post(
                f"{self.ha_url}/api/services/{domain}/{service}",
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"success": True, "result": result}
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
        except Exception as e:
            raise Exception(f"Failed to call service: {e}")
    
    async def _list_entities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List entities with optional filtering"""
        domain_filter = args.get("domain")
        area_filter = args.get("area")
        
        # Refresh entities
        await self._load_entities()
        
        entities = []
        for entity_id, entity_data in self.entities.items():
            # Apply domain filter
            if domain_filter and not entity_id.startswith(f"{domain_filter}."):
                continue
            
            # Apply area filter (simplified - would need device registry for proper area filtering)
            if area_filter:
                entity_area = entity_data.get("attributes", {}).get("area", "")
                if area_filter.lower() not in entity_area.lower():
                    continue
            
            entities.append({
                "entity_id": entity_id,
                "state": entity_data.get("state"),
                "attributes": entity_data.get("attributes", {}),
                "last_changed": entity_data.get("last_changed"),
                "last_updated": entity_data.get("last_updated")
            })
        
        return {"entities": entities}
    
    async def _get_areas(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get all areas"""
        await self._load_areas()
        return {"areas": list(self.areas.values())}
    
    async def _get_services(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get available services"""
        domain_filter = args.get("domain")
        
        await self._load_services()
        
        if domain_filter:
            services = {domain_filter: self.services.get(domain_filter, {})}
        else:
            services = self.services
        
        return {"services": services}
    
    async def _get_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get entity history"""
        entity_id = args.get("entity_id")
        start_time = args.get("start_time")
        end_time = args.get("end_time")
        
        # Build URL
        url = f"{self.ha_url}/api/history/period"
        if start_time:
            url += f"/{start_time}"
        
        params = {"filter_entity_id": entity_id}
        if end_time:
            params["end_time"] = end_time
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    history = await response.json()
                    return {"history": history}
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            raise Exception(f"Failed to get history: {e}")
    
    async def _set_entity_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set entity state (for testing/automation)"""
        entity_id = args.get("entity_id")
        state = args.get("state")
        attributes = args.get("attributes", {})
        
        data = {
            "state": state,
            "attributes": attributes
        }
        
        try:
            async with self.session.post(
                f"{self.ha_url}/api/states/{entity_id}",
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"success": True, "state": result}
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
        except Exception as e:
            raise Exception(f"Failed to set entity state: {e}")
    
    async def _read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read MCP resource"""
        uri = params.get("uri")
        
        if uri == "homeassistant://entities":
            await self._load_entities()
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(self.entities, indent=2)
                }]
            }
        elif uri == "homeassistant://config":
            try:
                async with self.session.get(f"{self.ha_url}/api/config") as response:
                    if response.status == 200:
                        config = await response.json()
                        return {
                            "contents": [{
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(config, indent=2)
                            }]
                        }
                    else:
                        raise Exception(f"HTTP {response.status}")
            except Exception as e:
                raise Exception(f"Failed to read config: {e}")
        else:
            raise Exception(f"Unknown resource URI: {uri}")

# Entry point for running the server
async def main():
    """Main entry point"""
    import os
    
    ha_url = os.getenv("HOMEASSISTANT_URL", "http://localhost:8123")
    ha_token = os.getenv("HOMEASSISTANT_TOKEN")
    websocket_port = int(os.getenv("HA_MCP_PORT", "8765"))
    
    if not ha_token:
        logger.error("HOMEASSISTANT_TOKEN environment variable is required")
        return
    
    server = HomeAssistantMCPServer(ha_url, ha_token, websocket_port)
    
    try:
        await server.initialize()
        await server.start_server()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await server.stop_server()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())