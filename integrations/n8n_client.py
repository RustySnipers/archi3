"""
n8n Workflow Management Integration
Comprehensive integration with n8n for workflow automation
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import base64
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

@dataclass
class N8nWorkflow:
    """n8n workflow representation"""
    id: str
    name: str
    active: bool
    nodes: List[Dict[str, Any]]
    connections: Dict[str, Any]
    settings: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    version_id: str

@dataclass
class N8nExecution:
    """n8n workflow execution representation"""
    id: str
    workflow_id: str
    mode: str
    started_at: datetime
    stopped_at: Optional[datetime]
    status: str
    data: Dict[str, Any]
    error: Optional[str]

@dataclass
class N8nCredential:
    """n8n credential representation"""
    id: str
    name: str
    type: str
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class N8nNode:
    """n8n workflow node representation"""
    id: str
    name: str
    type: str
    type_version: int
    position: List[int]
    parameters: Dict[str, Any]
    credentials: Optional[Dict[str, Any]]
    webhook_id: Optional[str]
    disabled: bool

class N8nClient:
    """n8n API client with comprehensive workflow management"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:5678",
                 api_key: str = None,
                 email: str = None,
                 password: str = None):
        
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.email = email
        self.password = password
        
        # HTTP session
        self.session = None
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Authentication
        self.authenticated = False
        self.auth_token = None
        
        # Caches
        self.workflows = {}
        self.executions = {}
        self.credentials = {}
        self.nodes_types = {}
        
        # Connection state
        self.connected = False
        self.last_update = None
        
        # Statistics
        self.stats = {
            "api_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "workflows_created": 0,
            "executions_triggered": 0,
            "last_sync": None
        }
    
    async def initialize(self):
        """Initialize n8n client connection"""
        try:
            logger.info("Initializing n8n client...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=self.headers
            )
            
            # Authenticate
            await self._authenticate()
            
            # Load initial data
            await self._load_initial_data()
            
            self.connected = True
            logger.info("n8n client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize n8n client: {e}")
            raise
    
    async def _authenticate(self):
        """Authenticate with n8n API"""
        try:
            if self.api_key:
                # API key authentication
                self.headers["X-N8N-API-KEY"] = self.api_key
                self.authenticated = True
                logger.info("Using API key authentication")
                
            elif self.email and self.password:
                # Email/password authentication
                auth_data = {
                    "email": self.email,
                    "password": self.password
                }
                
                async with self.session.post(
                    f"{self.base_url}/rest/login",
                    json=auth_data
                ) as response:
                    if response.status == 200:
                        # Session-based authentication
                        self.authenticated = True
                        logger.info("Email/password authentication successful")
                    else:
                        raise Exception(f"Authentication failed: HTTP {response.status}")
            else:
                # Try without authentication (for local development)
                await self._test_connection()
                self.authenticated = True
                logger.info("No authentication required")
                
        except Exception as e:
            logger.error(f"n8n authentication failed: {e}")
            raise
    
    async def _test_connection(self):
        """Test n8n API connection"""
        try:
            async with self.session.get(f"{self.base_url}/rest/workflows") as response:
                if response.status == 200:
                    logger.info("n8n API connection successful")
                    self.stats["successful_calls"] += 1
                elif response.status == 401:
                    raise Exception("Authentication required")
                else:
                    raise Exception(f"API test failed: HTTP {response.status}")
                    
            self.stats["api_calls"] += 1
            
        except Exception as e:
            self.stats["failed_calls"] += 1
            logger.error(f"n8n connection test failed: {e}")
            raise
    
    async def _load_initial_data(self):
        """Load initial workflow and execution data"""
        try:
            # Load workflows
            await self._refresh_workflows()
            
            # Load recent executions
            await self._refresh_executions()
            
            # Load credentials
            await self._refresh_credentials()
            
            # Load available node types
            await self._refresh_node_types()
            
            self.stats["last_sync"] = datetime.now()
            logger.info(f"Loaded {len(self.workflows)} workflows, {len(self.executions)} executions")
            
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
            raise
    
    async def _refresh_workflows(self):
        """Refresh workflows from n8n"""
        try:
            workflows = await self._api_get("workflows")
            self.workflows.clear()
            
            for workflow_data in workflows:
                workflow = N8nWorkflow(
                    id=workflow_data["id"],
                    name=workflow_data["name"],
                    active=workflow_data.get("active", False),
                    nodes=workflow_data.get("nodes", []),
                    connections=workflow_data.get("connections", {}),
                    settings=workflow_data.get("settings", {}),
                    created_at=datetime.fromisoformat(workflow_data["createdAt"].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(workflow_data["updatedAt"].replace('Z', '+00:00')),
                    tags=workflow_data.get("tags", []),
                    version_id=workflow_data.get("versionId", "")
                )
                self.workflows[workflow.id] = workflow
                
        except Exception as e:
            logger.error(f"Error refreshing workflows: {e}")
            raise
    
    async def _refresh_executions(self, limit: int = 100):
        """Refresh recent executions"""
        try:
            executions = await self._api_get(f"executions?limit={limit}")
            self.executions.clear()
            
            for exec_data in executions:
                execution = N8nExecution(
                    id=exec_data["id"],
                    workflow_id=exec_data["workflowId"],
                    mode=exec_data.get("mode", ""),
                    started_at=datetime.fromisoformat(exec_data["startedAt"].replace('Z', '+00:00')),
                    stopped_at=datetime.fromisoformat(exec_data["stoppedAt"].replace('Z', '+00:00')) if exec_data.get("stoppedAt") else None,
                    status=exec_data.get("status", ""),
                    data=exec_data.get("data", {}),
                    error=exec_data.get("error")
                )
                self.executions[execution.id] = execution
                
        except Exception as e:
            logger.error(f"Error refreshing executions: {e}")
    
    async def _refresh_credentials(self):
        """Refresh credentials"""
        try:
            credentials = await self._api_get("credentials")
            self.credentials.clear()
            
            for cred_data in credentials:
                credential = N8nCredential(
                    id=cred_data["id"],
                    name=cred_data["name"],
                    type=cred_data["type"],
                    data=cred_data.get("data", {}),
                    created_at=datetime.fromisoformat(cred_data["createdAt"].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(cred_data["updatedAt"].replace('Z', '+00:00'))
                )
                self.credentials[credential.id] = credential
                
        except Exception as e:
            logger.error(f"Error refreshing credentials: {e}")
    
    async def _refresh_node_types(self):
        """Refresh available node types"""
        try:
            node_types = await self._api_get("node-types")
            self.nodes_types.clear()
            
            for node_type in node_types:
                self.nodes_types[node_type["name"]] = node_type
                
        except Exception as e:
            logger.error(f"Error refreshing node types: {e}")
    
    async def _api_get(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """Make GET request to n8n API"""
        try:
            self.stats["api_calls"] += 1
            
            url = f"{self.base_url}/rest/{endpoint}"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    self.stats["successful_calls"] += 1
                    return await response.json()
                else:
                    self.stats["failed_calls"] += 1
                    error_text = await response.text()
                    raise Exception(f"API GET failed: {response.status} - {error_text}")
                    
        except Exception as e:
            self.stats["failed_calls"] += 1
            logger.error(f"API GET error: {e}")
            raise
    
    async def _api_post(self, endpoint: str, data: Dict[str, Any] = None) -> Any:
        """Make POST request to n8n API"""
        try:
            self.stats["api_calls"] += 1
            
            url = f"{self.base_url}/rest/{endpoint}"
            
            async with self.session.post(url, json=data or {}) as response:
                if response.status in [200, 201]:
                    self.stats["successful_calls"] += 1
                    return await response.json()
                else:
                    self.stats["failed_calls"] += 1
                    error_text = await response.text()
                    raise Exception(f"API POST failed: {response.status} - {error_text}")
                    
        except Exception as e:
            self.stats["failed_calls"] += 1
            logger.error(f"API POST error: {e}")
            raise
    
    async def _api_put(self, endpoint: str, data: Dict[str, Any] = None) -> Any:
        """Make PUT request to n8n API"""
        try:
            self.stats["api_calls"] += 1
            
            url = f"{self.base_url}/rest/{endpoint}"
            
            async with self.session.put(url, json=data or {}) as response:
                if response.status == 200:
                    self.stats["successful_calls"] += 1
                    return await response.json()
                else:
                    self.stats["failed_calls"] += 1
                    error_text = await response.text()
                    raise Exception(f"API PUT failed: {response.status} - {error_text}")
                    
        except Exception as e:
            self.stats["failed_calls"] += 1
            logger.error(f"API PUT error: {e}")
            raise
    
    async def _api_delete(self, endpoint: str) -> bool:
        """Make DELETE request to n8n API"""
        try:
            self.stats["api_calls"] += 1
            
            url = f"{self.base_url}/rest/{endpoint}"
            
            async with self.session.delete(url) as response:
                if response.status == 200:
                    self.stats["successful_calls"] += 1
                    return True
                else:
                    self.stats["failed_calls"] += 1
                    error_text = await response.text()
                    raise Exception(f"API DELETE failed: {response.status} - {error_text}")
                    
        except Exception as e:
            self.stats["failed_calls"] += 1
            logger.error(f"API DELETE error: {e}")
            raise
    
    # Public API methods
    
    async def get_workflows(self, active_only: bool = False) -> List[N8nWorkflow]:
        """Get all workflows"""
        try:
            workflows = list(self.workflows.values())
            
            if active_only:
                workflows = [w for w in workflows if w.active]
            
            return workflows
            
        except Exception as e:
            logger.error(f"Error getting workflows: {e}")
            return []
    
    async def get_workflow(self, workflow_id: str) -> Optional[N8nWorkflow]:
        """Get specific workflow"""
        try:
            if workflow_id in self.workflows:
                return self.workflows[workflow_id]
            
            # Fetch from API
            workflow_data = await self._api_get(f"workflows/{workflow_id}")
            
            workflow = N8nWorkflow(
                id=workflow_data["id"],
                name=workflow_data["name"],
                active=workflow_data.get("active", False),
                nodes=workflow_data.get("nodes", []),
                connections=workflow_data.get("connections", {}),
                settings=workflow_data.get("settings", {}),
                created_at=datetime.fromisoformat(workflow_data["createdAt"].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(workflow_data["updatedAt"].replace('Z', '+00:00')),
                tags=workflow_data.get("tags", []),
                version_id=workflow_data.get("versionId", "")
            )
            
            self.workflows[workflow_id] = workflow
            return workflow
            
        except Exception as e:
            logger.error(f"Error getting workflow {workflow_id}: {e}")
            return None
    
    async def create_workflow(self, 
                             name: str, 
                             nodes: List[Dict[str, Any]], 
                             connections: Dict[str, Any] = None,
                             settings: Dict[str, Any] = None,
                             tags: List[str] = None) -> Optional[N8nWorkflow]:
        """Create new workflow"""
        try:
            workflow_data = {
                "name": name,
                "nodes": nodes,
                "connections": connections or {},
                "settings": settings or {},
                "tags": tags or []
            }
            
            result = await self._api_post("workflows", workflow_data)
            
            workflow = N8nWorkflow(
                id=result["id"],
                name=result["name"],
                active=result.get("active", False),
                nodes=result.get("nodes", []),
                connections=result.get("connections", {}),
                settings=result.get("settings", {}),
                created_at=datetime.fromisoformat(result["createdAt"].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(result["updatedAt"].replace('Z', '+00:00')),
                tags=result.get("tags", []),
                version_id=result.get("versionId", "")
            )
            
            self.workflows[workflow.id] = workflow
            self.stats["workflows_created"] += 1
            
            logger.info(f"Created workflow: {name} ({workflow.id})")
            return workflow
            
        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            return None
    
    async def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing workflow"""
        try:
            result = await self._api_put(f"workflows/{workflow_id}", updates)
            
            # Update local cache
            if workflow_id in self.workflows:
                workflow = self.workflows[workflow_id]
                for key, value in updates.items():
                    if hasattr(workflow, key):
                        setattr(workflow, key, value)
                workflow.updated_at = datetime.now()
            
            logger.info(f"Updated workflow: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating workflow {workflow_id}: {e}")
            return False
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow"""
        try:
            await self._api_delete(f"workflows/{workflow_id}")
            
            # Remove from cache
            if workflow_id in self.workflows:
                del self.workflows[workflow_id]
            
            logger.info(f"Deleted workflow: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting workflow {workflow_id}: {e}")
            return False
    
    async def activate_workflow(self, workflow_id: str) -> bool:
        """Activate workflow"""
        try:
            await self._api_post(f"workflows/{workflow_id}/activate")
            
            # Update cache
            if workflow_id in self.workflows:
                self.workflows[workflow_id].active = True
            
            logger.info(f"Activated workflow: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error activating workflow {workflow_id}: {e}")
            return False
    
    async def deactivate_workflow(self, workflow_id: str) -> bool:
        """Deactivate workflow"""
        try:
            await self._api_post(f"workflows/{workflow_id}/deactivate")
            
            # Update cache
            if workflow_id in self.workflows:
                self.workflows[workflow_id].active = False
            
            logger.info(f"Deactivated workflow: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating workflow {workflow_id}: {e}")
            return False
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any] = None) -> Optional[str]:
        """Execute workflow"""
        try:
            execution_data = {
                "workflowId": workflow_id
            }
            
            if input_data:
                execution_data["data"] = input_data
            
            result = await self._api_post("executions", execution_data)
            
            execution_id = result.get("id")
            self.stats["executions_triggered"] += 1
            
            logger.info(f"Executed workflow {workflow_id}, execution ID: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {e}")
            return None
    
    async def get_execution(self, execution_id: str, include_data: bool = False) -> Optional[N8nExecution]:
        """Get execution details"""
        try:
            params = {"includeData": "true"} if include_data else None
            exec_data = await self._api_get(f"executions/{execution_id}", params)
            
            execution = N8nExecution(
                id=exec_data["id"],
                workflow_id=exec_data["workflowId"],
                mode=exec_data.get("mode", ""),
                started_at=datetime.fromisoformat(exec_data["startedAt"].replace('Z', '+00:00')),
                stopped_at=datetime.fromisoformat(exec_data["stoppedAt"].replace('Z', '+00:00')) if exec_data.get("stoppedAt") else None,
                status=exec_data.get("status", ""),
                data=exec_data.get("data", {}),
                error=exec_data.get("error")
            )
            
            self.executions[execution_id] = execution
            return execution
            
        except Exception as e:
            logger.error(f"Error getting execution {execution_id}: {e}")
            return None
    
    async def get_executions(self, workflow_id: str = None, limit: int = 50) -> List[N8nExecution]:
        """Get executions, optionally filtered by workflow"""
        try:
            params = {"limit": limit}
            if workflow_id:
                params["workflowId"] = workflow_id
            
            executions = await self._api_get("executions", params)
            
            result = []
            for exec_data in executions:
                execution = N8nExecution(
                    id=exec_data["id"],
                    workflow_id=exec_data["workflowId"],
                    mode=exec_data.get("mode", ""),
                    started_at=datetime.fromisoformat(exec_data["startedAt"].replace('Z', '+00:00')),
                    stopped_at=datetime.fromisoformat(exec_data["stoppedAt"].replace('Z', '+00:00')) if exec_data.get("stoppedAt") else None,
                    status=exec_data.get("status", ""),
                    data=exec_data.get("data", {}),
                    error=exec_data.get("error")
                )
                result.append(execution)
                self.executions[execution.id] = execution
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting executions: {e}")
            return []
    
    async def get_node_types(self) -> List[Dict[str, Any]]:
        """Get available node types"""
        return list(self.nodes_types.values())
    
    async def create_webhook_workflow(self, 
                                    name: str, 
                                    webhook_path: str,
                                    response_nodes: List[Dict[str, Any]]) -> Optional[N8nWorkflow]:
        """Create webhook-triggered workflow"""
        try:
            # Create webhook node
            webhook_node = {
                "id": "webhook",
                "name": "Webhook",
                "type": "n8n-nodes-base.webhook",
                "typeVersion": 1,
                "position": [250, 300],
                "parameters": {
                    "path": webhook_path,
                    "options": {}
                }
            }
            
            nodes = [webhook_node] + response_nodes
            
            # Create basic connections
            connections = {
                "Webhook": {
                    "main": [[{"node": response_nodes[0]["name"], "type": "main", "index": 0}]]
                }
            }
            
            return await self.create_workflow(name, nodes, connections)
            
        except Exception as e:
            logger.error(f"Error creating webhook workflow: {e}")
            return None
    
    async def create_cron_workflow(self, 
                                  name: str, 
                                  cron_expression: str,
                                  action_nodes: List[Dict[str, Any]]) -> Optional[N8nWorkflow]:
        """Create cron-triggered workflow"""
        try:
            # Create cron node
            cron_node = {
                "id": "cron",
                "name": "Cron",
                "type": "n8n-nodes-base.cron",
                "typeVersion": 1,
                "position": [250, 300],
                "parameters": {
                    "triggerTimes": {
                        "item": [{
                            "mode": "cronExpression",
                            "cronExpression": cron_expression
                        }]
                    }
                }
            }
            
            nodes = [cron_node] + action_nodes
            
            # Create basic connections
            connections = {
                "Cron": {
                    "main": [[{"node": action_nodes[0]["name"], "type": "main", "index": 0}]]
                }
            }
            
            return await self.create_workflow(name, nodes, connections)
            
        except Exception as e:
            logger.error(f"Error creating cron workflow: {e}")
            return None
    
    async def get_webhook_url(self, workflow_id: str, webhook_path: str) -> Optional[str]:
        """Get webhook URL for workflow"""
        try:
            workflow = await self.get_workflow(workflow_id)
            if not workflow:
                return None
            
            # Find webhook node
            webhook_node = None
            for node in workflow.nodes:
                if node.get("type") == "n8n-nodes-base.webhook":
                    webhook_node = node
                    break
            
            if not webhook_node:
                return None
            
            # Construct webhook URL
            webhook_url = f"{self.base_url}/webhook/{webhook_path}"
            return webhook_url
            
        except Exception as e:
            logger.error(f"Error getting webhook URL: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get n8n client statistics"""
        return {
            **self.stats,
            "connected": self.connected,
            "authenticated": self.authenticated,
            "workflows_cached": len(self.workflows),
            "executions_cached": len(self.executions),
            "credentials_cached": len(self.credentials),
            "node_types_available": len(self.nodes_types),
            "last_update": self.last_update.isoformat() if self.last_update else None
        }
    
    async def cleanup(self):
        """Clean up n8n client"""
        try:
            logger.info("Cleaning up n8n client...")
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            # Clear caches
            self.workflows.clear()
            self.executions.clear()
            self.credentials.clear()
            self.nodes_types.clear()
            
            self.connected = False
            self.authenticated = False
            
            logger.info("n8n client cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during n8n cleanup: {e}")

# Factory function for creating n8n client
async def create_n8n_client(config: Dict[str, Any]) -> N8nClient:
    """Create and initialize n8n client"""
    n8n_config = config.get('n8n', {})
    
    base_url = n8n_config.get('url', 'http://localhost:5678')
    api_key = n8n_config.get('api_key')
    email = n8n_config.get('email')
    password = n8n_config.get('password')
    
    client = N8nClient(
        base_url=base_url,
        api_key=api_key,
        email=email,
        password=password
    )
    
    await client.initialize()
    return client