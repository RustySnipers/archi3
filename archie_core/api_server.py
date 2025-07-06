"""
API server for Archie
Provides REST API and WebSocket endpoints
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors
from dataclasses import asdict

logger = logging.getLogger(__name__)

class ArchieAPIServer:
    """API server for Archie"""
    
    def __init__(self, orchestrator, host: str = "0.0.0.0", port: int = 8000):
        self.orchestrator = orchestrator
        self.host = host
        self.port = port
        self.app = None
        self.runner = None
        self.site = None
        self.websocket_connections = set()
        
    async def initialize(self):
        """Initialize the API server"""
        self.app = web.Application()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Setup routes
        self._setup_routes()
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
        
        logger.info("API server initialized")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check
        self.app.router.add_get("/health", self._health_check)
        
        # System status
        self.app.router.add_get("/api/status", self._get_status)
        
        # Chat endpoint
        self.app.router.add_post("/api/chat", self._chat)
        
        # Voice endpoints
        self.app.router.add_post("/api/voice/transcribe", self._transcribe_audio)
        self.app.router.add_post("/api/voice/synthesize", self._synthesize_speech)
        
        # Memory endpoints
        self.app.router.add_get("/api/memory/search", self._search_memory)
        self.app.router.add_post("/api/memory/store", self._store_memory)
        
        # Tool endpoints
        self.app.router.add_get("/api/tools", self._list_tools)
        self.app.router.add_post("/api/tools/execute", self._execute_tool)
        
        # Agent endpoints
        self.app.router.add_get("/api/agents", self._list_agents)
        self.app.router.add_get("/api/agents/{agent_name}/status", self._get_agent_status)
        
        # Task endpoints
        self.app.router.add_post("/api/tasks", self._create_task)
        self.app.router.add_get("/api/tasks", self._list_tasks)
        self.app.router.add_get("/api/tasks/{task_id}", self._get_task)
        
        # WebSocket endpoint
        self.app.router.add_get("/ws", self._websocket_handler)
        
        # Metrics endpoint
        self.app.router.add_get("/metrics", self._get_metrics)
        
        # Static files (optional)
        self.app.router.add_static("/", "static/", name="static")
    
    async def start(self):
        """Start the API server"""
        await self.initialize()
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        logger.info(f"API server started on {self.host}:{self.port}")
    
    async def shutdown(self):
        """Shutdown the API server"""
        if self.site:
            await self.site.stop()
        
        if self.runner:
            await self.runner.cleanup()
        
        # Close WebSocket connections
        for ws in self.websocket_connections.copy():
            await ws.close()
        
        logger.info("API server shutdown complete")
    
    # Route handlers
    
    async def _health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        })
    
    async def _get_status(self, request):
        """Get system status"""
        try:
            status = await self.orchestrator.get_system_status()
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _chat(self, request):
        """Chat endpoint"""
        try:
            data = await request.json()
            message = data.get("message")
            context = data.get("context", {})
            
            if not message:
                return web.json_response(
                    {"error": "Message is required"},
                    status=400
                )
            
            # Process message through orchestrator
            response = await self.orchestrator.process_user_input(message, context)
            
            return web.json_response({
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _transcribe_audio(self, request):
        """Transcribe audio to text"""
        try:
            # Get audio data from request
            reader = await request.multipart()
            audio_field = await reader.next()
            
            if audio_field.name != 'audio':
                return web.json_response(
                    {"error": "Audio field is required"},
                    status=400
                )
            
            audio_data = await audio_field.read()
            
            # Process through voice bridge if available
            if self.orchestrator.voice_bridge:
                transcription = await self.orchestrator.voice_bridge.process_speech_to_text(audio_data)
            else:
                transcription = "Voice processing not available"
            
            return web.json_response({
                "transcription": transcription,
                "confidence": 0.95,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _synthesize_speech(self, request):
        """Synthesize text to speech"""
        try:
            data = await request.json()
            text = data.get("text")
            voice = data.get("voice", "default")
            
            if not text:
                return web.json_response(
                    {"error": "Text is required"},
                    status=400
                )
            
            # Process through voice bridge if available
            if self.orchestrator.voice_bridge:
                audio_bytes = await self.orchestrator.voice_bridge.process_text_to_speech(text, voice)
                if audio_bytes:
                    # Return audio data directly
                    return web.Response(
                        body=audio_bytes,
                        content_type="audio/wav",
                        headers={"Content-Disposition": "attachment; filename=speech.wav"}
                    )
            
            return web.json_response({
                "error": "Voice synthesis not available"
            }, status=503)
            
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _search_memory(self, request):
        """Search memory"""
        try:
            query = request.query.get("q")
            memory_type = request.query.get("type", "all")
            limit = int(request.query.get("limit", "10"))
            
            if not query:
                return web.json_response(
                    {"error": "Query parameter 'q' is required"},
                    status=400
                )
            
            # Search through memory manager
            # This would integrate with the actual memory system
            results = []
            
            return web.json_response({
                "query": query,
                "results": results,
                "total": len(results),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _store_memory(self, request):
        """Store memory"""
        try:
            data = await request.json()
            content = data.get("content")
            memory_type = data.get("type", "note")
            metadata = data.get("metadata", {})
            
            if not content:
                return web.json_response(
                    {"error": "Content is required"},
                    status=400
                )
            
            # Store through memory manager
            # This would integrate with the actual memory system
            memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return web.json_response({
                "memory_id": memory_id,
                "content": content,
                "type": memory_type,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _list_tools(self, request):
        """List available tools"""
        try:
            tools = await self.orchestrator.tool_manager.get_available_tools()
            
            tool_list = []
            for tool in tools:
                tool_list.append({
                    "name": tool.name,
                    "description": tool.description,
                    "type": tool.tool_type.value,
                    "server": tool.server_name,
                    "enabled": tool.enabled
                })
            
            return web.json_response({
                "tools": tool_list,
                "total": len(tool_list),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _execute_tool(self, request):
        """Execute a tool"""
        try:
            data = await request.json()
            tool_name = data.get("tool_name")
            arguments = data.get("arguments", {})
            
            if not tool_name:
                return web.json_response(
                    {"error": "Tool name is required"},
                    status=400
                )
            
            # Create tool call
            from archie_core.tools import ToolCall
            call = ToolCall(
                id=f"api_call_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tool_name=tool_name,
                arguments=arguments,
                caller="api_server"
            )
            
            # Execute tool
            result = await self.orchestrator.tool_manager.execute_tool(call)
            
            return web.json_response({
                "success": result.success,
                "result": result.result,
                "error": result.error,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _list_agents(self, request):
        """List agents"""
        try:
            agents = []
            for name, agent in self.orchestrator.agents.items():
                status = agent.get_status()
                agents.append(status)
            
            return web.json_response({
                "agents": agents,
                "total": len(agents),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _get_agent_status(self, request):
        """Get agent status"""
        try:
            agent_name = request.match_info["agent_name"]
            
            if agent_name not in self.orchestrator.agents:
                return web.json_response(
                    {"error": f"Agent not found: {agent_name}"},
                    status=404
                )
            
            agent = self.orchestrator.agents[agent_name]
            status = agent.get_status()
            
            return web.json_response(status)
            
        except Exception as e:
            logger.error(f"Error getting agent status: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _create_task(self, request):
        """Create a new task"""
        try:
            data = await request.json()
            description = data.get("description")
            priority = data.get("priority", 5)
            input_data = data.get("input_data", {})
            
            if not description:
                return web.json_response(
                    {"error": "Description is required"},
                    status=400
                )
            
            task_id = await self.orchestrator.create_task(description, input_data, priority)
            
            return web.json_response({
                "task_id": task_id,
                "description": description,
                "priority": priority,
                "status": "created",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _list_tasks(self, request):
        """List tasks"""
        try:
            status_filter = request.query.get("status")
            limit = int(request.query.get("limit", "20"))
            
            tasks = []
            for task_id, task in self.orchestrator.tasks.items():
                if status_filter and task.status != status_filter:
                    continue
                
                task_data = {
                    "id": task.id,
                    "description": task.description,
                    "status": task.status,
                    "priority": task.priority,
                    "assigned_agent": task.assigned_agent,
                    "created_at": task.created_at.isoformat(),
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None
                }
                tasks.append(task_data)
            
            # Sort by creation time and limit
            tasks.sort(key=lambda x: x["created_at"], reverse=True)
            tasks = tasks[:limit]
            
            return web.json_response({
                "tasks": tasks,
                "total": len(tasks),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _get_task(self, request):
        """Get task details"""
        try:
            task_id = request.match_info["task_id"]
            
            if task_id not in self.orchestrator.tasks:
                return web.json_response(
                    {"error": f"Task not found: {task_id}"},
                    status=404
                )
            
            task = self.orchestrator.tasks[task_id]
            
            task_data = {
                "id": task.id,
                "description": task.description,
                "status": task.status,
                "priority": task.priority,
                "assigned_agent": task.assigned_agent,
                "input_data": task.input_data,
                "output_data": task.output_data,
                "created_at": task.created_at.isoformat(),
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "parent_task_id": task.parent_task_id,
                "subtasks": task.subtasks
            }
            
            return web.json_response(task_data)
            
        except Exception as e:
            logger.error(f"Error getting task: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def _websocket_handler(self, request):
        """WebSocket handler for real-time communication"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_connections.add(ws)
        logger.info(f"WebSocket client connected. Total connections: {len(self.websocket_connections)}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        
                        if data.get("type") == "chat":
                            # Process chat message
                            message = data.get("message")
                            context = data.get("context", {})
                            
                            response = await self.orchestrator.process_user_input(message, context)
                            
                            await ws.send_text(json.dumps({
                                "type": "chat_response",
                                "response": response,
                                "timestamp": datetime.now().isoformat()
                            }))
                        
                        elif data.get("type") == "status":
                            # Send status update
                            status = await self.orchestrator.get_system_status()
                            
                            await ws.send_text(json.dumps({
                                "type": "status_update",
                                "status": status,
                                "timestamp": datetime.now().isoformat()
                            }))
                        
                    except json.JSONDecodeError:
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "error": "Invalid JSON"
                        }))
                    except Exception as e:
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "error": str(e)
                        }))
                
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
        
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        
        finally:
            self.websocket_connections.discard(ws)
            logger.info(f"WebSocket client disconnected. Total connections: {len(self.websocket_connections)}")
        
        return ws
    
    async def _get_metrics(self, request):
        """Get system metrics in Prometheus format"""
        try:
            status = await self.orchestrator.get_system_status()
            
            # Convert to Prometheus format
            metrics = []
            
            # Orchestrator metrics
            orchestrator_stats = status["orchestrator"]["stats"]
            metrics.append(f"archie_total_tasks {orchestrator_stats['total_tasks']}")
            metrics.append(f"archie_messages_processed {orchestrator_stats['messages_processed']}")
            metrics.append(f"archie_active_agents {orchestrator_stats['active_agents']}")
            
            # Agent metrics
            for agent_name, agent_status in status["agents"].items():
                agent_stats = agent_status["stats"]
                metrics.append(f"archie_agent_tasks_completed{{agent=\"{agent_name}\"}} {agent_stats['tasks_completed']}")
                metrics.append(f"archie_agent_tasks_failed{{agent=\"{agent_name}\"}} {agent_stats['tasks_failed']}")
            
            # Task metrics
            task_stats = status["tasks"]
            metrics.append(f"archie_tasks_total {task_stats['total']}")
            metrics.append(f"archie_tasks_pending {task_stats['pending']}")
            metrics.append(f"archie_tasks_in_progress {task_stats['in_progress']}")
            metrics.append(f"archie_tasks_completed {task_stats['completed']}")
            metrics.append(f"archie_tasks_failed {task_stats['failed']}")
            
            metrics_text = "\n".join(metrics)
            
            return web.Response(
                text=metrics_text,
                content_type="text/plain"
            )
            
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return web.Response(
                text=f"# Error generating metrics: {e}",
                content_type="text/plain",
                status=500
            )
    
    async def broadcast_to_websockets(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections"""
        if not self.websocket_connections:
            return
        
        message_text = json.dumps(message)
        
        # Send to all connected clients
        disconnected = set()
        for ws in self.websocket_connections:
            try:
                await ws.send_text(message_text)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected.add(ws)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected

# Factory function for creating API server
async def create_api_server(orchestrator, host: str = "0.0.0.0", port: int = 8000) -> ArchieAPIServer:
    """Create and start API server"""
    server = ArchieAPIServer(orchestrator, host, port)
    await server.start()
    return server