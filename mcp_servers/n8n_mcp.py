"""
n8n MCP Server
Provides MCP interface for n8n workflow automation
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

class N8nMCPServer:
    """MCP server for n8n integration"""
    
    def __init__(self, 
                 n8n_url: str = "http://localhost:5678",
                 n8n_api_key: str = None,
                 websocket_port: int = 8766):
        
        self.n8n_url = n8n_url.rstrip('/')
        self.n8n_api_key = n8n_api_key
        self.websocket_port = websocket_port
        self.session = None
        self.server = None
        self.connected_clients = set()
        
        # n8n state
        self.workflows = {}
        self.executions = {}
        self.credentials = {}
        
        # MCP tools definition
        self.tools = [
            {
                "name": "list_workflows",
                "description": "List all n8n workflows",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "active": {
                            "type": "boolean",
                            "description": "Filter by active status (optional)"
                        }
                    }
                }
            },
            {
                "name": "get_workflow",
                "description": "Get a specific workflow by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "The workflow ID"
                        }
                    },
                    "required": ["workflow_id"]
                }
            },
            {
                "name": "create_workflow",
                "description": "Create a new n8n workflow",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Workflow name"
                        },
                        "nodes": {
                            "type": "array",
                            "description": "Workflow nodes"
                        },
                        "connections": {
                            "type": "object",
                            "description": "Node connections"
                        },
                        "settings": {
                            "type": "object",
                            "description": "Workflow settings"
                        },
                        "active": {
                            "type": "boolean",
                            "description": "Whether to activate the workflow"
                        }
                    },
                    "required": ["name", "nodes"]
                }
            },
            {
                "name": "update_workflow",
                "description": "Update an existing workflow",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "The workflow ID"
                        },
                        "name": {
                            "type": "string",
                            "description": "Workflow name"
                        },
                        "nodes": {
                            "type": "array",
                            "description": "Workflow nodes"
                        },
                        "connections": {
                            "type": "object",
                            "description": "Node connections"
                        },
                        "settings": {
                            "type": "object",
                            "description": "Workflow settings"
                        },
                        "active": {
                            "type": "boolean",
                            "description": "Whether to activate the workflow"
                        }
                    },
                    "required": ["workflow_id"]
                }
            },
            {
                "name": "delete_workflow",
                "description": "Delete a workflow",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "The workflow ID"
                        }
                    },
                    "required": ["workflow_id"]
                }
            },
            {
                "name": "execute_workflow",
                "description": "Execute a workflow manually",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "The workflow ID"
                        },
                        "input_data": {
                            "type": "object",
                            "description": "Input data for the workflow"
                        }
                    },
                    "required": ["workflow_id"]
                }
            },
            {
                "name": "get_execution",
                "description": "Get workflow execution details",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "execution_id": {
                            "type": "string",
                            "description": "The execution ID"
                        }
                    },
                    "required": ["execution_id"]
                }
            },
            {
                "name": "list_executions",
                "description": "List workflow executions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "Filter by workflow ID (optional)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of executions to return",
                            "default": 20
                        },
                        "status": {
                            "type": "string",
                            "description": "Filter by execution status",
                            "enum": ["success", "error", "running", "waiting"]
                        }
                    }
                }
            },
            {
                "name": "activate_workflow",
                "description": "Activate or deactivate a workflow",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "The workflow ID"
                        },
                        "active": {
                            "type": "boolean",
                            "description": "Whether to activate (true) or deactivate (false)"
                        }
                    },
                    "required": ["workflow_id", "active"]
                }
            },
            {
                "name": "generate_workflow_from_description",
                "description": "Generate a workflow from natural language description",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Natural language description of the workflow"
                        },
                        "triggers": {
                            "type": "array",
                            "description": "List of trigger conditions"
                        },
                        "actions": {
                            "type": "array",
                            "description": "List of actions to perform"
                        }
                    },
                    "required": ["description"]
                }
            }
        ]
        
        # MCP resources definition
        self.resources = [
            {
                "uri": "n8n://workflows",
                "name": "All Workflows",
                "description": "List of all n8n workflows",
                "mimeType": "application/json"
            },
            {
                "uri": "n8n://executions",
                "name": "Recent Executions",
                "description": "Recent workflow executions",
                "mimeType": "application/json"
            },
            {
                "uri": "n8n://nodes",
                "name": "Available Nodes",
                "description": "List of available n8n nodes",
                "mimeType": "application/json"
            }
        ]
    
    async def initialize(self):
        """Initialize the MCP server"""
        try:
            # Create HTTP session
            headers = {}
            if self.n8n_api_key:
                headers["X-N8N-API-KEY"] = self.n8n_api_key
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test connection to n8n
            await self._test_connection()
            
            # Load initial data
            await self._load_workflows()
            
            logger.info("n8n MCP server initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize n8n MCP server: {e}")
            raise
    
    async def _test_connection(self):
        """Test connection to n8n"""
        try:
            async with self.session.get(f"{self.n8n_url}/api/v1/workflows") as response:
                if response.status == 200:
                    logger.info("Connected to n8n successfully")
                elif response.status == 401:
                    raise Exception("Authentication failed - check API key")
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            logger.error(f"Failed to connect to n8n: {e}")
            raise
    
    async def _load_workflows(self):
        """Load all workflows from n8n"""
        try:
            async with self.session.get(f"{self.n8n_url}/api/v1/workflows") as response:
                if response.status == 200:
                    workflows = await response.json()
                    for workflow in workflows.get('data', []):
                        self.workflows[workflow['id']] = workflow
                    logger.info(f"Loaded {len(self.workflows)} workflows")
                else:
                    logger.error(f"Failed to load workflows: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error loading workflows: {e}")
    
    async def start_server(self):
        """Start the MCP WebSocket server"""
        try:
            self.server = await serve(
                self._handle_client,
                "localhost",
                self.websocket_port
            )
            logger.info(f"n8n MCP server started on port {self.websocket_port}")
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
        
        logger.info("n8n MCP server stopped")
    
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
        """Call an n8n tool"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "list_workflows":
            return await self._list_workflows(arguments)
        elif tool_name == "get_workflow":
            return await self._get_workflow(arguments)
        elif tool_name == "create_workflow":
            return await self._create_workflow(arguments)
        elif tool_name == "update_workflow":
            return await self._update_workflow(arguments)
        elif tool_name == "delete_workflow":
            return await self._delete_workflow(arguments)
        elif tool_name == "execute_workflow":
            return await self._execute_workflow(arguments)
        elif tool_name == "get_execution":
            return await self._get_execution(arguments)
        elif tool_name == "list_executions":
            return await self._list_executions(arguments)
        elif tool_name == "activate_workflow":
            return await self._activate_workflow(arguments)
        elif tool_name == "generate_workflow_from_description":
            return await self._generate_workflow_from_description(arguments)
        else:
            raise Exception(f"Unknown tool: {tool_name}")
    
    async def _list_workflows(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List workflows"""
        active_filter = args.get("active")
        
        await self._load_workflows()
        
        workflows = []
        for workflow in self.workflows.values():
            if active_filter is not None and workflow.get('active') != active_filter:
                continue
            
            workflows.append({
                "id": workflow["id"],
                "name": workflow["name"],
                "active": workflow.get("active", False),
                "createdAt": workflow.get("createdAt"),
                "updatedAt": workflow.get("updatedAt"),
                "tags": workflow.get("tags", []),
                "nodeCount": len(workflow.get("nodes", []))
            })
        
        return {"workflows": workflows}
    
    async def _get_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get a specific workflow"""
        workflow_id = args.get("workflow_id")
        
        try:
            async with self.session.get(f"{self.n8n_url}/api/v1/workflows/{workflow_id}") as response:
                if response.status == 200:
                    workflow = await response.json()
                    return {"workflow": workflow.get("data")}
                elif response.status == 404:
                    raise Exception(f"Workflow not found: {workflow_id}")
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            raise Exception(f"Failed to get workflow: {e}")
    
    async def _create_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new workflow"""
        workflow_data = {
            "name": args.get("name"),
            "nodes": args.get("nodes", []),
            "connections": args.get("connections", {}),
            "settings": args.get("settings", {}),
            "active": args.get("active", False)
        }
        
        try:
            async with self.session.post(
                f"{self.n8n_url}/api/v1/workflows",
                json=workflow_data
            ) as response:
                if response.status == 200:
                    workflow = await response.json()
                    # Update local cache
                    workflow_data = workflow.get("data")
                    self.workflows[workflow_data["id"]] = workflow_data
                    return {"workflow": workflow_data}
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
        except Exception as e:
            raise Exception(f"Failed to create workflow: {e}")
    
    async def _update_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing workflow"""
        workflow_id = args.get("workflow_id")
        
        # Build update data
        update_data = {}
        for key in ["name", "nodes", "connections", "settings", "active"]:
            if key in args:
                update_data[key] = args[key]
        
        try:
            async with self.session.put(
                f"{self.n8n_url}/api/v1/workflows/{workflow_id}",
                json=update_data
            ) as response:
                if response.status == 200:
                    workflow = await response.json()
                    # Update local cache
                    workflow_data = workflow.get("data")
                    self.workflows[workflow_id] = workflow_data
                    return {"workflow": workflow_data}
                elif response.status == 404:
                    raise Exception(f"Workflow not found: {workflow_id}")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
        except Exception as e:
            raise Exception(f"Failed to update workflow: {e}")
    
    async def _delete_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a workflow"""
        workflow_id = args.get("workflow_id")
        
        try:
            async with self.session.delete(f"{self.n8n_url}/api/v1/workflows/{workflow_id}") as response:
                if response.status == 200:
                    # Remove from local cache
                    self.workflows.pop(workflow_id, None)
                    return {"success": True, "message": f"Workflow {workflow_id} deleted"}
                elif response.status == 404:
                    raise Exception(f"Workflow not found: {workflow_id}")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
        except Exception as e:
            raise Exception(f"Failed to delete workflow: {e}")
    
    async def _execute_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow"""
        workflow_id = args.get("workflow_id")
        input_data = args.get("input_data", {})
        
        try:
            async with self.session.post(
                f"{self.n8n_url}/api/v1/workflows/{workflow_id}/execute",
                json=input_data
            ) as response:
                if response.status == 200:
                    execution = await response.json()
                    return {"execution": execution.get("data")}
                elif response.status == 404:
                    raise Exception(f"Workflow not found: {workflow_id}")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
        except Exception as e:
            raise Exception(f"Failed to execute workflow: {e}")
    
    async def _get_execution(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get execution details"""
        execution_id = args.get("execution_id")
        
        try:
            async with self.session.get(f"{self.n8n_url}/api/v1/executions/{execution_id}") as response:
                if response.status == 200:
                    execution = await response.json()
                    return {"execution": execution.get("data")}
                elif response.status == 404:
                    raise Exception(f"Execution not found: {execution_id}")
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            raise Exception(f"Failed to get execution: {e}")
    
    async def _list_executions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List executions"""
        workflow_id = args.get("workflow_id")
        limit = args.get("limit", 20)
        status = args.get("status")
        
        params = {"limit": limit}
        if workflow_id:
            params["workflowId"] = workflow_id
        if status:
            params["status"] = status
        
        try:
            async with self.session.get(
                f"{self.n8n_url}/api/v1/executions",
                params=params
            ) as response:
                if response.status == 200:
                    executions = await response.json()
                    return {"executions": executions.get("data", [])}
                else:
                    raise Exception(f"HTTP {response.status}")
        except Exception as e:
            raise Exception(f"Failed to list executions: {e}")
    
    async def _activate_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Activate or deactivate a workflow"""
        workflow_id = args.get("workflow_id")
        active = args.get("active")
        
        return await self._update_workflow({
            "workflow_id": workflow_id,
            "active": active
        })
    
    async def _generate_workflow_from_description(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate workflow from natural language description"""
        description = args.get("description")
        triggers = args.get("triggers", [])
        actions = args.get("actions", [])
        
        # This is a simplified implementation
        # In a real implementation, this would use AI to generate the workflow
        
        # Basic workflow template
        nodes = []
        connections = {}
        
        # Add manual trigger node
        trigger_node = {
            "id": "trigger",
            "name": "Manual Trigger",
            "type": "n8n-nodes-base.manualTrigger",
            "position": [250, 300],
            "parameters": {}
        }
        nodes.append(trigger_node)
        
        # Add HTTP request node as an example action
        if "http" in description.lower() or "api" in description.lower():
            http_node = {
                "id": "http",
                "name": "HTTP Request",
                "type": "n8n-nodes-base.httpRequest",
                "position": [450, 300],
                "parameters": {
                    "url": "https://api.example.com",
                    "options": {}
                }
            }
            nodes.append(http_node)
            
            # Connect trigger to HTTP request
            connections = {
                "trigger": {
                    "main": [
                        [
                            {
                                "node": "http",
                                "type": "main",
                                "index": 0
                            }
                        ]
                    ]
                }
            }
        
        workflow_name = f"Generated: {description[:50]}..."
        
        # Create the workflow
        return await self._create_workflow({
            "name": workflow_name,
            "nodes": nodes,
            "connections": connections,
            "settings": {
                "executionOrder": "v1"
            },
            "active": False
        })
    
    async def _read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read MCP resource"""
        uri = params.get("uri")
        
        if uri == "n8n://workflows":
            await self._load_workflows()
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(list(self.workflows.values()), indent=2)
                }]
            }
        elif uri == "n8n://executions":
            executions_result = await self._list_executions({"limit": 50})
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(executions_result["executions"], indent=2)
                }]
            }
        elif uri == "n8n://nodes":
            # This would return available node types
            # For now, return a placeholder
            node_types = ["Manual Trigger", "HTTP Request", "Set", "Function", "Code"]
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(node_types, indent=2)
                }]
            }
        else:
            raise Exception(f"Unknown resource URI: {uri}")

# Entry point for running the server
async def main():
    """Main entry point"""
    import os
    
    n8n_url = os.getenv("N8N_URL", "http://localhost:5678")
    n8n_api_key = os.getenv("N8N_API_KEY")
    websocket_port = int(os.getenv("N8N_MCP_PORT", "8766"))
    
    server = N8nMCPServer(n8n_url, n8n_api_key, websocket_port)
    
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