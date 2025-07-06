"""
Main Archie Agent
Core AI agent with orchestration and multi-agent support
"""

import os
import json
import yaml
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from .llm_client import LocalAIClient, CompletionRequest, CompletionResponse
from .memory import MemoryManager, Memory, MemoryQuery
from .mcp_client import MCPClient, MCPServer
from .tools import ToolManager, Tool, ToolCall, ToolResult
from .learning import LearningSystem
from .multimodal import MultiModalProcessor, ModalInput, ModalityType

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"

@dataclass
class AgentMessage:
    """Message structure for agent communication"""
    id: str
    sender: str
    recipient: str
    content: str
    message_type: str
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class AgentTask:
    """Task structure for agent execution"""
    id: str
    description: str
    priority: int
    assigned_agent: Optional[str]
    status: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime]
    parent_task_id: Optional[str]
    subtasks: List[str]

class BaseAgent:
    """Base agent class for all specialized agents"""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 llm_client: LocalAIClient,
                 memory_manager: MemoryManager,
                 tool_manager: ToolManager,
                 learning_system: Optional[LearningSystem] = None,
                 multimodal_processor: Optional[MultiModalProcessor] = None):
        
        self.name = name
        self.description = description
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.tool_manager = tool_manager
        self.learning_system = learning_system
        self.multimodal_processor = multimodal_processor
        
        self.state = AgentState.IDLE
        self.current_task = None
        self.capabilities = []
        self.system_prompt = ""
        self.conversation_history = []
        self.agent_id = f"{name}_{id(self)}"
        
        # Agent statistics
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_thinking_time": 0,
            "total_execution_time": 0,
            "created_at": datetime.now()
        }
    
    async def initialize(self):
        """Initialize the agent"""
        await self._load_system_prompt()
        await self._load_capabilities()
        logger.info(f"Agent {self.name} initialized")
    
    async def _load_system_prompt(self):
        """Load agent-specific system prompt"""
        self.system_prompt = f"""
You are {self.name}, a specialized AI agent in the Archie personal assistant system.

Description: {self.description}

Your role is to help users by executing tasks within your domain of expertise.
You have access to various tools and can collaborate with other agents when needed.

Always be helpful, accurate, and efficient in your responses.
If you need information or capabilities beyond your scope, you can request assistance from other agents.
"""
    
    async def _load_capabilities(self):
        """Load agent capabilities"""
        self.capabilities = [
            "natural_language_processing",
            "task_execution",
            "tool_usage",
            "memory_access",
            "agent_communication"
        ]
    
    async def process_message(self, message: str, context: Dict[str, Any] = None, modal_inputs: List[ModalInput] = None) -> str:
        """Process incoming message and generate response"""
        try:
            self.state = AgentState.THINKING
            
            # Add message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Process multi-modal inputs if provided
            multimodal_context = None
            if modal_inputs and self.multimodal_processor:
                multimodal_context = await self.multimodal_processor.process_multi_modal_context(modal_inputs)
            
            # Build context for LLM
            llm_context = await self._build_llm_context(message, context, multimodal_context)
            
            # Generate response using LLM
            response = await self._generate_response(llm_context)
            
            # Process any tool calls in the response
            if response and response.tool_calls:
                tool_results = await self._execute_tool_calls(response.tool_calls)
                # Continue conversation with tool results
                response = await self._continue_with_tool_results(tool_results)
            
            # Store conversation in memory
            await self._store_conversation_memory(message, response.content if response else "")
            
            # Record learning event
            if self.learning_system:
                await self.learning_system.record_learning_event(
                    event_type="conversation",
                    description=f"Conversation processed by {self.name}",
                    data={
                        "user_message": message,
                        "agent_response": response.content if response else "",
                        "tools_used": len(response.tool_calls) if response and response.tool_calls else 0,
                        "context": context or {}
                    },
                    agent_id=self.agent_id,
                    user_id=context.get("user_id") if context else None,
                    confidence=0.8
                )
            
            self.state = AgentState.IDLE
            
            return response.content if response else "I'm sorry, I couldn't process that request."
            
        except Exception as e:
            logger.error(f"Error processing message in {self.name}: {e}")
            self.state = AgentState.ERROR
            return f"I encountered an error: {str(e)}"
    
    async def _build_llm_context(self, message: str, context: Dict[str, Any] = None, multimodal_context = None) -> List[Dict[str, str]]:
        """Build enhanced context for LLM request"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Use enhanced memory retrieval with context
        memory_query = MemoryQuery(
            query_text=message,
            memory_types=["conversation", "task", "knowledge"],
            limit=5,
            include_context=True,
            context_window=3,
            importance_threshold=0.4,
            similarity_threshold=0.6
        )
        
        # Get smart memory retrieval with contextual information
        enhanced_memories = await self.memory_manager.smart_memory_retrieval(memory_query)
        
        if enhanced_memories:
            memory_context = "Relevant context from previous interactions:\n"
            
            for memory, score, context_memories in enhanced_memories:
                memory_context += f"- {memory.content} (relevance: {score:.2f})\n"
                
                # Add contextual memories
                if context_memories:
                    memory_context += "  Related context:\n"
                    for ctx_memory in context_memories[:2]:  # Top 2 context memories
                        memory_context += f"    • {ctx_memory.content}\n"
            
            messages.append({
                "role": "system",
                "content": memory_context
            })
        
        # Add conversation patterns and insights
        conversation_insights = await self._analyze_conversation_patterns(message)
        if conversation_insights:
            messages.append({
                "role": "system", 
                "content": f"Conversation insights: {conversation_insights}"
            })
        
        # Add multi-modal context if available
        if multimodal_context:
            multimodal_str = await self._format_multimodal_context(multimodal_context)
            if multimodal_str:
                messages.append({
                    "role": "system",
                    "content": f"Multi-modal context: {multimodal_str}"
                })
        
        # Add contextual information if provided
        if context:
            context_str = await self._format_context_info(context)
            if context_str:
                messages.append({
                    "role": "system",
                    "content": f"Additional context: {context_str}"
                })
        
        # Add recent conversation history with enhanced context
        recent_history = await self._enhance_conversation_history(message)
        messages.extend(recent_history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    async def _generate_response(self, messages: List[Dict[str, str]]) -> Optional[CompletionResponse]:
        """Generate response using LLM"""
        try:
            # Get available tools
            available_tools = await self.tool_manager.get_available_tools()
            
            # Convert tools to OpenAI format
            tools = []
            for tool in available_tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    }
                })
            
            request = CompletionRequest(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None
            )
            
            response = await self.llm_client.complete(request)
            
            if response:
                # Add response to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                    "timestamp": datetime.now().isoformat()
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute tool calls"""
        results = []
        
        for tool_call in tool_calls:
            try:
                function = tool_call.get("function", {})
                tool_name = function.get("name")
                arguments = json.loads(function.get("arguments", "{}"))
                
                # Create tool call object
                call = ToolCall(
                    id=tool_call.get("id"),
                    tool_name=tool_name,
                    arguments=arguments,
                    caller=self.agent_id
                )
                
                # Execute the tool
                result = await self.tool_manager.execute_tool(call)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error executing tool call: {e}")
                results.append(ToolResult(
                    call_id=tool_call.get("id"),
                    success=False,
                    result=None,
                    error=str(e)
                ))
        
        return results
    
    async def _continue_with_tool_results(self, tool_results: List[ToolResult]) -> Optional[CompletionResponse]:
        """Continue conversation with tool results"""
        try:
            # Add tool results to conversation
            for result in tool_results:
                tool_message = {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": json.dumps(result.result) if result.success else f"Error: {result.error}"
                }
                self.conversation_history.append(tool_message)
            
            # Generate follow-up response
            request = CompletionRequest(
                model="gpt-4",
                messages=self.conversation_history[-20:],  # Last 20 messages
                temperature=0.7,
                max_tokens=2048
            )
            
            response = await self.llm_client.complete(request)
            
            if response:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                    "timestamp": datetime.now().isoformat()
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Error continuing with tool results: {e}")
            return None
    
    async def _store_conversation_memory(self, user_message: str, agent_response: str):
        """Store conversation in memory"""
        try:
            conversation_content = f"User: {user_message}\nAgent ({self.name}): {agent_response}"
            
            await self.memory_manager.store_memory(
                content=conversation_content,
                memory_type="conversation",
                metadata={
                    "agent": self.name,
                    "user_message": user_message,
                    "agent_response": agent_response
                },
                importance=0.7,
                tags=["conversation", self.name]
            )
            
        except Exception as e:
            logger.error(f"Error storing conversation memory: {e}")
    
    async def _analyze_conversation_patterns(self, current_message: str) -> Optional[str]:
        """Analyze conversation patterns for context-aware responses"""
        try:
            if len(self.conversation_history) < 3:
                return None
            
            # Analyze recent conversation flow
            recent_messages = self.conversation_history[-6:]  # Last 6 messages
            
            # Identify patterns
            patterns = []
            
            # Check for question sequences
            question_count = sum(1 for msg in recent_messages if msg.get("role") == "user" and "?" in msg.get("content", ""))
            if question_count > 1:
                patterns.append("user_asking_multiple_questions")
            
            # Check for task continuation
            task_keywords = ["then", "next", "after", "also", "and"]
            if any(keyword in current_message.lower() for keyword in task_keywords):
                patterns.append("task_continuation")
            
            # Check for clarification requests
            clarification_keywords = ["what", "how", "why", "when", "where", "explain", "clarify"]
            if any(keyword in current_message.lower() for keyword in clarification_keywords):
                patterns.append("seeking_clarification")
            
            # Check for emotional context
            emotion_keywords = {
                "frustration": ["frustrated", "annoyed", "stuck", "confused"],
                "urgency": ["urgent", "quickly", "asap", "now", "immediately"],
                "satisfaction": ["great", "thanks", "perfect", "excellent", "good"]
            }
            
            current_lower = current_message.lower()
            detected_emotions = []
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in current_lower for keyword in keywords):
                    detected_emotions.append(emotion)
            
            if detected_emotions:
                patterns.append(f"emotional_context_{'-'.join(detected_emotions)}")
            
            # Return insights if patterns found
            if patterns:
                return f"Conversation patterns detected: {', '.join(patterns)}"
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing conversation patterns: {e}")
            return None
    
    async def _format_context_info(self, context: Dict[str, Any]) -> Optional[str]:
        """Format additional context information"""
        try:
            context_parts = []
            
            # Time context
            if "time" in context:
                context_parts.append(f"Current time: {context['time']}")
            
            # Location context
            if "location" in context:
                context_parts.append(f"User location: {context['location']}")
            
            # Device/environment context
            if "device" in context:
                context_parts.append(f"Device: {context['device']}")
            
            # User state context
            if "user_state" in context:
                context_parts.append(f"User state: {context['user_state']}")
            
            # Task context
            if "current_task" in context:
                context_parts.append(f"Current task: {context['current_task']}")
            
            # Custom context
            for key, value in context.items():
                if key not in ["time", "location", "device", "user_state", "current_task"]:
                    context_parts.append(f"{key}: {value}")
            
            return "; ".join(context_parts) if context_parts else None
            
        except Exception as e:
            logger.error(f"Error formatting context info: {e}")
            return None
    
    async def _format_multimodal_context(self, multimodal_context) -> Optional[str]:
        """Format multi-modal context information"""
        try:
            if not multimodal_context:
                return None
            
            context_parts = []
            
            # Overall interpretation
            if multimodal_context.combined_interpretation:
                context_parts.append(f"Combined interpretation: {multimodal_context.combined_interpretation}")
            
            # Individual modality results
            modality_summaries = {}
            for result in multimodal_context.results:
                if result.state.value == "completed":
                    modality = result.modality.value
                    if modality not in modality_summaries:
                        modality_summaries[modality] = []
                    
                    # Add key features
                    if result.interpreted_content:
                        modality_summaries[modality].append(result.interpreted_content)
                    
                    # Add confidence score
                    if result.confidence > 0.7:
                        modality_summaries[modality].append(f"(high confidence: {result.confidence:.2f})")
                    elif result.confidence > 0.4:
                        modality_summaries[modality].append(f"(medium confidence: {result.confidence:.2f})")
            
            # Format modality summaries
            for modality, summaries in modality_summaries.items():
                context_parts.append(f"{modality.title()}: {'; '.join(summaries)}")
            
            # Processing metadata
            if multimodal_context.confidence:
                context_parts.append(f"Overall confidence: {multimodal_context.confidence:.2f}")
            
            return "; ".join(context_parts) if context_parts else None
            
        except Exception as e:
            logger.error(f"Error formatting multimodal context: {e}")
            return None
    
    async def _enhance_conversation_history(self, current_message: str) -> List[Dict[str, str]]:
        """Enhance conversation history with contextual information"""
        try:
            # Get recent conversation history
            recent_history = self.conversation_history[-10:]  # Last 10 messages
            
            # Analyze conversation flow and add context markers
            enhanced_history = []
            
            for i, msg in enumerate(recent_history):
                enhanced_msg = msg.copy()
                
                # Add context markers for conversation flow
                if i == 0 and len(recent_history) > 1:
                    enhanced_msg["content"] = f"[Start of recent context] {enhanced_msg['content']}"
                elif i == len(recent_history) - 1 and len(recent_history) > 1:
                    enhanced_msg["content"] = f"[Most recent] {enhanced_msg['content']}"
                
                # Add semantic similarity context
                if msg.get("role") == "user":
                    similarity = await self._calculate_message_similarity(msg["content"], current_message)
                    if similarity > 0.7:
                        enhanced_msg["content"] = f"[Similar to current] {enhanced_msg['content']}"
                
                enhanced_history.append(enhanced_msg)
            
            return enhanced_history
            
        except Exception as e:
            logger.error(f"Error enhancing conversation history: {e}")
            return self.conversation_history[-10:]  # Fallback to simple history
    
    async def _calculate_message_similarity(self, message1: str, message2: str) -> float:
        """Calculate semantic similarity between two messages"""
        try:
            # Simple word overlap similarity (can be enhanced with embeddings)
            words1 = set(message1.lower().split())
            words2 = set(message2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating message similarity: {e}")
            return 0.0
    
    async def execute_task(self, task: AgentTask) -> AgentTask:
        """Execute a specific task"""
        try:
            self.state = AgentState.EXECUTING
            self.current_task = task
            task.status = "in_progress"
            
            start_time = datetime.now()
            
            # Process the task description as a message
            response = await self.process_message(
                task.description,
                context=task.input_data
            )
            
            # Update task with results
            task.output_data = {"response": response}
            task.status = "completed"
            task.completed_at = datetime.now()
            
            # Update statistics
            execution_time = (task.completed_at - start_time).total_seconds()
            self.stats["tasks_completed"] += 1
            self.stats["total_execution_time"] += execution_time
            
            self.state = AgentState.IDLE
            self.current_task = None
            
            logger.info(f"Task {task.id} completed by {self.name}")
            return task
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            task.status = "failed"
            task.output_data = {"error": str(e)}
            task.completed_at = datetime.now()
            
            self.stats["tasks_failed"] += 1
            self.state = AgentState.ERROR
            self.current_task = None
            
            return task
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "current_task": self.current_task.id if self.current_task else None,
            "capabilities": self.capabilities,
            "stats": self.stats,
            "conversation_length": len(self.conversation_history)
        }

class ArchieOrchestrator:
    """Main orchestrator for managing multiple agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[str, AgentTask] = {}
        self.task_queue = asyncio.Queue()
        self.message_bus = asyncio.Queue()
        
        # Core components
        self.llm_client = None
        self.memory_manager = None
        self.mcp_client = None
        self.tool_manager = None
        self.learning_system = None
        self.multimodal_processor = None
        self.performance_optimizer = None
        
        # Voice integration
        self.voice_bridge = None
        self.voice_enabled = config.get('voice', {}).get('enabled', True)
        
        # Orchestrator state
        self.running = False
        self.worker_tasks = []
        self.conversation_history = []  # Global conversation history
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "active_agents": 0,
            "messages_processed": 0,
            "start_time": None
        }
    
    async def initialize(self):
        """Initialize the orchestrator and all components"""
        try:
            logger.info("Initializing Archie Orchestrator...")
            
            # Initialize core components
            from .llm_client import create_localai_client
            from .memory import create_memory_manager
            from .mcp_client import create_mcp_client
            from .tools import create_tool_manager
            from .learning import create_learning_system
            from .multimodal import create_multimodal_processor
            from .performance import create_performance_optimizer
            
            self.llm_client = await create_localai_client(self.config)
            self.memory_manager = await create_memory_manager(self.config)
            self.mcp_client = await create_mcp_client(self.config)
            self.tool_manager = await create_tool_manager(self.config, self.mcp_client)
            self.learning_system = await create_learning_system(self.memory_manager, self.config)
            self.multimodal_processor = await create_multimodal_processor(self.config)
            self.performance_optimizer = await create_performance_optimizer(self.config)
            
            # Initialize agents
            await self._initialize_agents()
            
            # Initialize voice integration if enabled
            if self.voice_enabled:
                await self._initialize_voice_integration()
            
            # Start worker tasks
            self.running = True
            self.worker_tasks = [
                asyncio.create_task(self._task_processor()),
                asyncio.create_task(self._message_processor())
            ]
            
            self.stats["start_time"] = datetime.now()
            logger.info("Archie Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def _initialize_agents(self):
        """Initialize specialized agents"""
        agent_configs = self.config.get('agents', {}).get('types', [])
        
        for agent_config in agent_configs:
            if not agent_config.get('enabled', True):
                continue
            
            agent_name = agent_config['name']
            
            # Create specialized agent based on type
            if agent_name == "voice_agent":
                agent = await self._create_voice_agent()
            elif agent_name == "automation_agent":
                agent = await self._create_automation_agent()
            elif agent_name == "communication_agent":
                agent = await self._create_communication_agent()
            elif agent_name == "memory_agent":
                agent = await self._create_memory_agent()
            else:
                # Create generic agent
                agent = BaseAgent(
                    name=agent_name,
                    description=f"Generic agent for {agent_name}",
                    llm_client=self.llm_client,
                    memory_manager=self.memory_manager,
                    tool_manager=self.tool_manager,
                    learning_system=self.learning_system,
                    multimodal_processor=self.multimodal_processor
                )
            
            await agent.initialize()
            self.agents[agent_name] = agent
            self.stats["active_agents"] += 1
            
            logger.info(f"Initialized agent: {agent_name}")
    
    async def _create_voice_agent(self) -> BaseAgent:
        """Create voice processing agent"""
        agent = BaseAgent(
            name="voice_agent",
            description="Specialized agent for voice processing, speech-to-text, and text-to-speech",
            llm_client=self.llm_client,
            memory_manager=self.memory_manager,
            tool_manager=self.tool_manager,
            learning_system=self.learning_system,
            multimodal_processor=self.multimodal_processor
        )
        
        agent.capabilities.extend([
            "speech_recognition",
            "voice_synthesis",
            "audio_processing",
            "wake_word_detection",
            "multimodal_processing"
        ])
        
        return agent
    
    async def _create_automation_agent(self) -> BaseAgent:
        """Create automation agent"""
        agent = BaseAgent(
            name="automation_agent",
            description="Specialized agent for home automation, workflow creation, and device control",
            llm_client=self.llm_client,
            memory_manager=self.memory_manager,
            tool_manager=self.tool_manager,
            learning_system=self.learning_system,
            multimodal_processor=self.multimodal_processor
        )
        
        agent.capabilities.extend([
            "home_assistant_integration",
            "n8n_workflow_creation",
            "device_control",
            "automation_logic"
        ])
        
        return agent
    
    async def _create_communication_agent(self) -> BaseAgent:
        """Create communication agent"""
        agent = BaseAgent(
            name="communication_agent",
            description="Specialized agent for external communications, notifications, and messaging",
            llm_client=self.llm_client,
            memory_manager=self.memory_manager,
            tool_manager=self.tool_manager,
            learning_system=self.learning_system,
            multimodal_processor=self.multimodal_processor
        )
        
        agent.capabilities.extend([
            "telegram_integration",
            "email_notifications",
            "sms_messaging",
            "notification_management"
        ])
        
        return agent
    
    async def _create_memory_agent(self) -> BaseAgent:
        """Create memory management agent"""
        agent = BaseAgent(
            name="memory_agent",
            description="Specialized agent for memory management, information retrieval, and knowledge base",
            llm_client=self.llm_client,
            memory_manager=self.memory_manager,
            tool_manager=self.tool_manager,
            learning_system=self.learning_system,
            multimodal_processor=self.multimodal_processor
        )
        
        agent.capabilities.extend([
            "memory_storage",
            "information_retrieval",
            "knowledge_base_management",
            "context_maintenance"
        ])
        
        return agent
    
    async def _initialize_voice_integration(self):
        """Initialize voice integration with agent system"""
        try:
            logger.info("Initializing voice integration...")
            
            # Import voice integration
            import sys
            from pathlib import Path
            voice_path = Path(__file__).parent.parent / "voice"
            sys.path.insert(0, str(voice_path))
            
            from voice_agent_integration import create_voice_agent_bridge
            
            # Create voice bridge
            self.voice_bridge = await create_voice_agent_bridge(self, self.config)
            
            logger.info("Voice integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice integration: {e}")
            self.voice_enabled = False
    
    async def process_user_input(self, message: str, context: Dict[str, Any] = None) -> str:
        """Process user input and route to appropriate agent"""
        try:
            # Determine which agent should handle this message
            agent_name = await self._route_message(message, context)
            
            if agent_name not in self.agents:
                agent_name = "voice_agent"  # Default agent
            
            agent = self.agents[agent_name]
            response = await agent.process_message(message, context)
            
            # Update global conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "agent": agent_name,
                "timestamp": datetime.now().isoformat()
            })
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "agent": agent_name,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 100:
                self.conversation_history = self.conversation_history[-50:]
            
            self.stats["messages_processed"] += 1
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return f"I encountered an error processing your request: {str(e)}"
    
    async def process_user_feedback(self, 
                                  feedback: str, 
                                  context: Dict[str, Any] = None,
                                  user_id: Optional[str] = None) -> str:
        """Process user feedback for learning"""
        try:
            if self.learning_system:
                await self.learning_system.process_user_feedback(
                    feedback=feedback,
                    context=context or {},
                    agent_id="orchestrator",
                    user_id=user_id
                )
                
                # Apply any new adaptation rules
                adaptations = await self.learning_system.apply_adaptation_rules(context or {})
                
                if adaptations:
                    adaptation_msgs = [f"Applied: {a['action']}" for a in adaptations]
                    return f"Thank you for the feedback! I've made the following adjustments: {', '.join(adaptation_msgs)}"
                else:
                    return "Thank you for the feedback! I'll use this to improve future interactions."
            else:
                return "Thank you for the feedback!"
                
        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")
            return "Thank you for the feedback, though I had trouble processing it."
    
    async def _route_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Route message to appropriate agent based on content and context"""
        message_lower = message.lower()
        
        # Enhanced routing logic with context awareness
        routing_scores = {}
        
        # Analyze message content for agent capabilities
        for agent_name, agent in self.agents.items():
            score = 0
            
            # Capability-based scoring
            if agent_name == "automation_agent":
                automation_keywords = ["light", "switch", "automation", "workflow", "device", "control", "home", "turn", "set", "adjust"]
                score += sum(2 for keyword in automation_keywords if keyword in message_lower)
            
            elif agent_name == "memory_agent":
                memory_keywords = ["remember", "recall", "memory", "note", "save", "store", "forget", "remind", "history"]
                score += sum(2 for keyword in memory_keywords if keyword in message_lower)
            
            elif agent_name == "communication_agent":
                comm_keywords = ["send", "message", "telegram", "notify", "email", "call", "text", "alert", "inform"]
                score += sum(2 for keyword in comm_keywords if keyword in message_lower)
            
            elif agent_name == "voice_agent":
                voice_keywords = ["say", "speak", "voice", "audio", "sound", "listen", "hear"]
                score += sum(2 for keyword in voice_keywords if keyword in message_lower)
            
            # Context-based scoring
            if context:
                # Time-based context
                if "time" in context:
                    hour = context["time"].hour if hasattr(context["time"], "hour") else 12
                    if hour < 8 or hour > 22:  # Early morning or late night
                        if agent_name == "voice_agent":
                            score += 1  # Prefer voice for quiet times
                
                # Location-based context
                if "location" in context:
                    if "bedroom" in context["location"].lower() or "living room" in context["location"].lower():
                        if agent_name == "automation_agent":
                            score += 1  # Prefer automation in home locations
                
                # Device context
                if "device" in context:
                    if "mobile" in context["device"].lower():
                        if agent_name == "communication_agent":
                            score += 1  # Prefer communication on mobile
                
                # User state context
                if "user_state" in context:
                    if "busy" in context["user_state"].lower():
                        if agent_name == "automation_agent":
                            score += 1  # Prefer automation when busy
                    elif "learning" in context["user_state"].lower():
                        if agent_name == "memory_agent":
                            score += 1  # Prefer memory when learning
            
            # Previous conversation context
            if len(self.conversation_history) > 0:
                recent_agent = self.conversation_history[-1].get("agent", "")
                if recent_agent == agent_name:
                    score += 1  # Slight preference for conversation continuity
            
            # Agent availability
            if agent.state == AgentState.IDLE:
                score += 1
            elif agent.state == AgentState.BUSY:
                score -= 2
            
            routing_scores[agent_name] = score
        
        # Select agent with highest score
        best_agent = max(routing_scores.items(), key=lambda x: x[1])
        
        # If no clear winner, use contextual fallback
        if best_agent[1] <= 0:
            return "voice_agent"  # Default for general conversation
        
        return best_agent[0]
    
    async def create_task(self, description: str, input_data: Dict[str, Any] = None, priority: int = 5) -> str:
        """Create a new task"""
        task_id = f"task_{len(self.tasks)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = AgentTask(
            id=task_id,
            description=description,
            priority=priority,
            assigned_agent=None,
            status="pending",
            input_data=input_data or {},
            output_data={},
            created_at=datetime.now(),
            completed_at=None,
            parent_task_id=None,
            subtasks=[]
        )
        
        self.tasks[task_id] = task
        await self.task_queue.put(task)
        
        self.stats["total_tasks"] += 1
        
        logger.info(f"Created task: {task_id}")
        return task_id
    
    async def _task_processor(self):
        """Background task processor"""
        while self.running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Assign agent
                agent_name = await self._assign_agent(task)
                if agent_name and agent_name in self.agents:
                    task.assigned_agent = agent_name
                    agent = self.agents[agent_name]
                    
                    # Execute task
                    completed_task = await agent.execute_task(task)
                    self.tasks[task.id] = completed_task
                
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
    
    async def _assign_agent(self, task: AgentTask) -> Optional[str]:
        """Assign task to appropriate agent"""
        # Simple assignment based on task description
        description_lower = task.description.lower()
        
        for agent_name, agent in self.agents.items():
            if agent.state == AgentState.IDLE:
                # Check if agent capabilities match task requirements
                if agent_name == "automation_agent" and any(word in description_lower for word in ["automation", "device", "workflow"]):
                    return agent_name
                elif agent_name == "memory_agent" and any(word in description_lower for word in ["memory", "remember", "store"]):
                    return agent_name
                elif agent_name == "communication_agent" and any(word in description_lower for word in ["send", "notify", "message"]):
                    return agent_name
        
        # Return first available agent
        for agent_name, agent in self.agents.items():
            if agent.state == AgentState.IDLE:
                return agent_name
        
        return None
    
    async def _message_processor(self):
        """Background message processor for inter-agent communication"""
        while self.running:
            try:
                # Process messages between agents
                message = await asyncio.wait_for(self.message_bus.get(), timeout=1.0)
                # Handle inter-agent messages
                await self._handle_agent_message(message)
                self.message_bus.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
    
    async def _handle_agent_message(self, message: AgentMessage):
        """Handle messages between agents"""
        if message.recipient in self.agents:
            recipient_agent = self.agents[message.recipient]
            # Process message in recipient agent
            await recipient_agent.process_message(message.content, message.metadata)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        agent_statuses = {}
        for name, agent in self.agents.items():
            agent_statuses[name] = agent.get_status()
        
        return {
            "orchestrator": {
                "running": self.running,
                "stats": self.stats,
                "task_queue_size": self.task_queue.qsize(),
                "message_queue_size": self.message_bus.qsize()
            },
            "agents": agent_statuses,
            "tasks": {
                "total": len(self.tasks),
                "pending": len([t for t in self.tasks.values() if t.status == "pending"]),
                "in_progress": len([t for t in self.tasks.values() if t.status == "in_progress"]),
                "completed": len([t for t in self.tasks.values() if t.status == "completed"]),
                "failed": len([t for t in self.tasks.values() if t.status == "failed"])
            }
        }
    
    async def shutdown(self):
        """Shutdown the orchestrator"""
        logger.info("Shutting down Archie Orchestrator...")
        
        self.running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Close voice integration
        if self.voice_bridge:
            await self.voice_bridge.cleanup()
        
        # Close connections
        if self.llm_client:
            await self.llm_client.close()
        
        if self.mcp_client:
            await self.mcp_client.disconnect_all()
        
        logger.info("Archie Orchestrator shutdown complete")

# Factory function for creating orchestrator
async def create_orchestrator(config_path: str = None) -> ArchieOrchestrator:
    """Create and initialize the Archie orchestrator"""
    if config_path is None:
        config_path = os.getenv('ARCHIE_CONFIG_PATH', '/opt/archie/configs')
    
    config_file = os.path.join(config_path, 'archie_config.yaml')
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    orchestrator = ArchieOrchestrator(config)
    await orchestrator.initialize()
    
    return orchestrator