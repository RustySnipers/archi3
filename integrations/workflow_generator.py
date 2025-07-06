"""
Dynamic Workflow Generation System
Creates n8n workflows based on user intent and natural language descriptions
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from .n8n_client import N8nClient, N8nWorkflow, N8nNode
from .home_assistant import HomeAssistantClient

logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    """Types of workflows that can be generated"""
    AUTOMATION = "automation"
    NOTIFICATION = "notification"
    MONITORING = "monitoring"
    INTEGRATION = "integration"
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"
    CONDITIONAL = "conditional"

@dataclass
class WorkflowTemplate:
    """Template for workflow generation"""
    name: str
    type: WorkflowType
    description: str
    trigger_types: List[str]
    action_types: List[str]
    required_integrations: List[str]
    parameters: Dict[str, Any]

@dataclass
class WorkflowIntent:
    """Parsed user intent for workflow creation"""
    type: WorkflowType
    trigger: str
    actions: List[str]
    conditions: List[str]
    schedule: Optional[str]
    entities: List[str]
    parameters: Dict[str, Any]

class WorkflowGenerator:
    """Dynamic workflow generation system"""
    
    def __init__(self, n8n_client: N8nClient, ha_client: HomeAssistantClient = None):
        self.n8n_client = n8n_client
        self.ha_client = ha_client
        
        # Node position tracking
        self.node_positions = {}
        self.current_x = 250
        self.current_y = 300
        
        # Workflow templates
        self.templates = {}
        self._load_templates()
        
        # Node type mappings
        self.node_mappings = {
            "webhook": "n8n-nodes-base.webhook",
            "cron": "n8n-nodes-base.cron",
            "http": "n8n-nodes-base.httpRequest",
            "code": "n8n-nodes-base.code",
            "if": "n8n-nodes-base.if",
            "switch": "n8n-nodes-base.switch",
            "set": "n8n-nodes-base.set",
            "home_assistant": "n8n-nodes-base.homeAssistant",
            "telegram": "n8n-nodes-base.telegram",
            "email": "n8n-nodes-base.emailSend",
            "slack": "n8n-nodes-base.slack",
            "discord": "n8n-nodes-base.discord",
            "function": "n8n-nodes-base.function",
            "delay": "n8n-nodes-base.wait",
            "merge": "n8n-nodes-base.merge",
            "split": "n8n-nodes-base.splitInBatches"
        }
        
        # Intent parsing patterns
        self.intent_patterns = {
            "trigger": [
                r"when (.+?) (happens|occurs|is triggered)",
                r"if (.+?) (is|becomes|turns) (.+)",
                r"every (.+?) (minutes?|hours?|days?)",
                r"at (.+?) (am|pm|o'clock)",
                r"on (.+?) webhook",
                r"receive (.+?) notification"
            ],
            "action": [
                r"(turn on|turn off|toggle|set) (.+)",
                r"(send|notify|alert) (.+)",
                r"(create|make|generate) (.+)",
                r"(execute|run|call) (.+)",
                r"(save|store|record) (.+)"
            ],
            "condition": [
                r"only if (.+)",
                r"when (.+?) is (.+)",
                r"unless (.+)",
                r"provided that (.+)"
            ]
        }
    
    def _load_templates(self):
        """Load workflow templates"""
        self.templates = {
            "home_automation": WorkflowTemplate(
                name="Home Automation",
                type=WorkflowType.AUTOMATION,
                description="Control home devices based on triggers",
                trigger_types=["webhook", "cron", "home_assistant"],
                action_types=["home_assistant", "http"],
                required_integrations=["home_assistant"],
                parameters={}
            ),
            "notification_system": WorkflowTemplate(
                name="Notification System",
                type=WorkflowType.NOTIFICATION,
                description="Send notifications via various channels",
                trigger_types=["webhook", "cron", "http"],
                action_types=["telegram", "email", "slack"],
                required_integrations=["telegram"],
                parameters={}
            ),
            "monitoring_alert": WorkflowTemplate(
                name="Monitoring Alert",
                type=WorkflowType.MONITORING,
                description="Monitor systems and send alerts",
                trigger_types=["cron", "webhook"],
                action_types=["http", "telegram", "email"],
                required_integrations=["home_assistant"],
                parameters={}
            ),
            "scheduled_task": WorkflowTemplate(
                name="Scheduled Task",
                type=WorkflowType.SCHEDULED,
                description="Execute tasks on schedule",
                trigger_types=["cron"],
                action_types=["http", "home_assistant", "code"],
                required_integrations=[],
                parameters={}
            ),
            "webhook_integration": WorkflowTemplate(
                name="Webhook Integration",
                type=WorkflowType.WEBHOOK,
                description="Handle webhook events",
                trigger_types=["webhook"],
                action_types=["http", "code", "home_assistant"],
                required_integrations=[],
                parameters={}
            )
        }
    
    async def parse_intent(self, description: str) -> WorkflowIntent:
        """Parse natural language description into workflow intent"""
        try:
            description = description.lower().strip()
            
            # Initialize intent
            intent = WorkflowIntent(
                type=WorkflowType.AUTOMATION,
                trigger="",
                actions=[],
                conditions=[],
                schedule=None,
                entities=[],
                parameters={}
            )
            
            # Parse trigger
            trigger_match = None
            for pattern in self.intent_patterns["trigger"]:
                match = re.search(pattern, description)
                if match:
                    trigger_match = match
                    break
            
            if trigger_match:
                intent.trigger = trigger_match.group(1)
                
                # Determine workflow type based on trigger
                if "webhook" in intent.trigger:
                    intent.type = WorkflowType.WEBHOOK
                elif any(word in intent.trigger for word in ["every", "at", "daily", "hourly"]):
                    intent.type = WorkflowType.SCHEDULED
                elif any(word in intent.trigger for word in ["notification", "alert", "notify"]):
                    intent.type = WorkflowType.NOTIFICATION
                elif any(word in intent.trigger for word in ["monitor", "check", "watch"]):
                    intent.type = WorkflowType.MONITORING
            
            # Parse actions
            for pattern in self.intent_patterns["action"]:
                matches = re.findall(pattern, description)
                for match in matches:
                    if isinstance(match, tuple):
                        action = f"{match[0]} {match[1]}"
                    else:
                        action = match
                    intent.actions.append(action)
            
            # Parse conditions
            for pattern in self.intent_patterns["condition"]:
                matches = re.findall(pattern, description)
                for match in matches:
                    intent.conditions.append(match)
            
            # Extract schedule information
            schedule_patterns = [
                r"every (\d+) (minutes?|hours?|days?)",
                r"at (\d{1,2}):(\d{2})",
                r"daily at (\d{1,2})(am|pm)",
                r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                r"(0 \d{1,2} \* \* \*)"  # Cron pattern
            ]
            
            for pattern in schedule_patterns:
                match = re.search(pattern, description)
                if match:
                    intent.schedule = match.group(0)
                    break
            
            # Extract entities (device names, etc.)
            if self.ha_client:
                entities = await self.ha_client.get_all_entities()
                for entity in entities:
                    entity_name = entity.friendly_name.lower()
                    if entity_name in description or entity.entity_id in description:
                        intent.entities.append(entity.entity_id)
            
            logger.info(f"Parsed intent: {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"Error parsing intent: {e}")
            return WorkflowIntent(
                type=WorkflowType.AUTOMATION,
                trigger="",
                actions=[],
                conditions=[],
                schedule=None,
                entities=[],
                parameters={}
            )
    
    def _get_next_position(self) -> List[int]:
        """Get next node position"""
        position = [self.current_x, self.current_y]
        self.current_x += 200
        if self.current_x > 1000:
            self.current_x = 250
            self.current_y += 200
        return position
    
    def _create_trigger_node(self, intent: WorkflowIntent) -> Dict[str, Any]:
        """Create trigger node based on intent"""
        try:
            node_id = "trigger"
            
            if intent.type == WorkflowType.WEBHOOK:
                # Create webhook trigger
                webhook_path = intent.parameters.get("webhook_path", "archie-webhook")
                return {
                    "id": node_id,
                    "name": "Webhook Trigger",
                    "type": self.node_mappings["webhook"],
                    "typeVersion": 1,
                    "position": self._get_next_position(),
                    "parameters": {
                        "path": webhook_path,
                        "options": {}
                    }
                }
            
            elif intent.type == WorkflowType.SCHEDULED or intent.schedule:
                # Create cron trigger
                cron_expression = self._parse_schedule_to_cron(intent.schedule or intent.trigger)
                return {
                    "id": node_id,
                    "name": "Schedule Trigger",
                    "type": self.node_mappings["cron"],
                    "typeVersion": 1,
                    "position": self._get_next_position(),
                    "parameters": {
                        "triggerTimes": {
                            "item": [{
                                "mode": "cronExpression",
                                "cronExpression": cron_expression
                            }]
                        }
                    }
                }
            
            elif "home assistant" in intent.trigger or intent.entities:
                # Create Home Assistant trigger
                return {
                    "id": node_id,
                    "name": "Home Assistant Trigger",
                    "type": self.node_mappings["home_assistant"],
                    "typeVersion": 1,
                    "position": self._get_next_position(),
                    "parameters": {
                        "server": "homeAssistant",
                        "version": 2,
                        "eventType": "state_changed",
                        "outputProperties": []
                    }
                }
            
            else:
                # Default webhook trigger
                return {
                    "id": node_id,
                    "name": "Manual Trigger",
                    "type": self.node_mappings["webhook"],
                    "typeVersion": 1,
                    "position": self._get_next_position(),
                    "parameters": {
                        "path": "manual-trigger",
                        "options": {}
                    }
                }
                
        except Exception as e:
            logger.error(f"Error creating trigger node: {e}")
            return {}
    
    def _create_condition_nodes(self, intent: WorkflowIntent) -> List[Dict[str, Any]]:
        """Create condition nodes based on intent"""
        nodes = []
        
        try:
            for i, condition in enumerate(intent.conditions):
                node_id = f"condition_{i}"
                
                # Create IF node for condition
                node = {
                    "id": node_id,
                    "name": f"Condition {i+1}",
                    "type": self.node_mappings["if"],
                    "typeVersion": 1,
                    "position": self._get_next_position(),
                    "parameters": {
                        "conditions": {
                            "options": {
                                "caseSensitive": True,
                                "leftValue": "",
                                "rightValue": ""
                            },
                            "conditions": [{
                                "leftValue": "={{ $json.condition }}",
                                "rightValue": condition,
                                "operator": {
                                    "type": "string",
                                    "operation": "equals"
                                }
                            }],
                            "combinator": "and"
                        }
                    }
                }
                
                nodes.append(node)
                
        except Exception as e:
            logger.error(f"Error creating condition nodes: {e}")
        
        return nodes
    
    def _create_action_nodes(self, intent: WorkflowIntent) -> List[Dict[str, Any]]:
        """Create action nodes based on intent"""
        nodes = []
        
        try:
            for i, action in enumerate(intent.actions):
                node_id = f"action_{i}"
                action_lower = action.lower()
                
                # Determine action type
                if any(word in action_lower for word in ["turn on", "turn off", "toggle", "set"]) and intent.entities:
                    # Home Assistant action
                    entity_id = intent.entities[0] if intent.entities else "light.example"
                    service = "turn_on" if "turn on" in action_lower else "turn_off" if "turn off" in action_lower else "toggle"
                    
                    node = {
                        "id": node_id,
                        "name": f"Control {entity_id}",
                        "type": self.node_mappings["home_assistant"],
                        "typeVersion": 1,
                        "position": self._get_next_position(),
                        "parameters": {
                            "server": "homeAssistant",
                            "version": 2,
                            "service": service,
                            "entityId": entity_id,
                            "serviceAttributes": {}
                        }
                    }
                
                elif any(word in action_lower for word in ["send", "notify", "alert"]):
                    # Notification action
                    message = action.replace("send", "").replace("notify", "").replace("alert", "").strip()
                    
                    node = {
                        "id": node_id,
                        "name": f"Send Notification",
                        "type": self.node_mappings["telegram"],
                        "typeVersion": 1,
                        "position": self._get_next_position(),
                        "parameters": {
                            "chatId": "{{ $json.chatId }}",
                            "text": message or "Notification from Archie",
                            "additionalFields": {}
                        }
                    }
                
                elif any(word in action_lower for word in ["http", "request", "api"]):
                    # HTTP request action
                    node = {
                        "id": node_id,
                        "name": f"HTTP Request",
                        "type": self.node_mappings["http"],
                        "typeVersion": 1,
                        "position": self._get_next_position(),
                        "parameters": {
                            "url": "{{ $json.url }}",
                            "options": {}
                        }
                    }
                
                else:
                    # Generic code action
                    node = {
                        "id": node_id,
                        "name": f"Execute Action",
                        "type": self.node_mappings["code"],
                        "typeVersion": 1,
                        "position": self._get_next_position(),
                        "parameters": {
                            "jsCode": f"// Execute: {action}\\nreturn items;"
                        }
                    }
                
                nodes.append(node)
                
        except Exception as e:
            logger.error(f"Error creating action nodes: {e}")
        
        return nodes
    
    def _create_connections(self, trigger_node: Dict[str, Any], 
                           condition_nodes: List[Dict[str, Any]], 
                           action_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create connections between nodes"""
        connections = {}
        
        try:
            # Connect trigger to first condition or first action
            if condition_nodes:
                connections[trigger_node["name"]] = {
                    "main": [[{"node": condition_nodes[0]["name"], "type": "main", "index": 0}]]
                }
                
                # Connect conditions to actions
                for i, condition_node in enumerate(condition_nodes):
                    if i < len(action_nodes):
                        connections[condition_node["name"]] = {
                            "main": [[{"node": action_nodes[i]["name"], "type": "main", "index": 0}]]
                        }
                
                # Connect remaining actions in sequence
                for i in range(len(condition_nodes), len(action_nodes) - 1):
                    connections[action_nodes[i]["name"]] = {
                        "main": [[{"node": action_nodes[i + 1]["name"], "type": "main", "index": 0}]]
                    }
            
            elif action_nodes:
                connections[trigger_node["name"]] = {
                    "main": [[{"node": action_nodes[0]["name"], "type": "main", "index": 0}]]
                }
                
                # Connect actions in sequence
                for i in range(len(action_nodes) - 1):
                    connections[action_nodes[i]["name"]] = {
                        "main": [[{"node": action_nodes[i + 1]["name"], "type": "main", "index": 0}]]
                    }
            
        except Exception as e:
            logger.error(f"Error creating connections: {e}")
        
        return connections
    
    def _parse_schedule_to_cron(self, schedule: str) -> str:
        """Parse schedule description to cron expression"""
        try:
            schedule = schedule.lower().strip()
            
            # Common schedule patterns
            if "every minute" in schedule:
                return "* * * * *"
            elif "every 5 minutes" in schedule:
                return "*/5 * * * *"
            elif "every 15 minutes" in schedule:
                return "*/15 * * * *"
            elif "every 30 minutes" in schedule:
                return "*/30 * * * *"
            elif "every hour" in schedule:
                return "0 * * * *"
            elif "every day" in schedule or "daily" in schedule:
                return "0 9 * * *"  # 9 AM daily
            elif "every week" in schedule or "weekly" in schedule:
                return "0 9 * * 1"  # 9 AM Mondays
            elif "every month" in schedule or "monthly" in schedule:
                return "0 9 1 * *"  # 9 AM first of month
            
            # Time patterns
            time_match = re.search(r"(\d{1,2}):(\d{2})", schedule)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))
                return f"{minute} {hour} * * *"
            
            # Frequency patterns
            freq_match = re.search(r"every (\d+) (minutes?|hours?|days?)", schedule)
            if freq_match:
                interval = int(freq_match.group(1))
                unit = freq_match.group(2)
                
                if "minute" in unit:
                    return f"*/{interval} * * * *"
                elif "hour" in unit:
                    return f"0 */{interval} * * *"
                elif "day" in unit:
                    return f"0 9 */{interval} * *"
            
            # Default to every hour
            return "0 * * * *"
            
        except Exception as e:
            logger.error(f"Error parsing schedule: {e}")
            return "0 * * * *"
    
    async def generate_workflow(self, description: str, name: str = None) -> Optional[N8nWorkflow]:
        """Generate workflow from natural language description"""
        try:
            logger.info(f"Generating workflow from description: {description}")
            
            # Reset position tracking
            self.current_x = 250
            self.current_y = 300
            
            # Parse intent
            intent = await self.parse_intent(description)
            
            # Generate workflow name
            if not name:
                name = f"Archie Generated - {intent.type.value.title()}"
            
            # Create nodes
            trigger_node = self._create_trigger_node(intent)
            condition_nodes = self._create_condition_nodes(intent)
            action_nodes = self._create_action_nodes(intent)
            
            # Combine all nodes
            all_nodes = [trigger_node] + condition_nodes + action_nodes
            
            # Create connections
            connections = self._create_connections(trigger_node, condition_nodes, action_nodes)
            
            # Workflow settings
            settings = {
                "executionOrder": "v1",
                "saveManualExecutions": True,
                "callerPolicy": "workflowsFromSameOwner",
                "errorWorkflow": ""
            }
            
            # Create workflow
            workflow = await self.n8n_client.create_workflow(
                name=name,
                nodes=all_nodes,
                connections=connections,
                settings=settings,
                tags=["archie-generated", intent.type.value]
            )
            
            if workflow:
                logger.info(f"Successfully generated workflow: {workflow.id}")
                
                # Activate workflow if it has a trigger
                if trigger_node:
                    await self.n8n_client.activate_workflow(workflow.id)
                    logger.info(f"Activated workflow: {workflow.id}")
            
            return workflow
            
        except Exception as e:
            logger.error(f"Error generating workflow: {e}")
            return None
    
    async def suggest_workflows(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest workflows based on context"""
        suggestions = []
        
        try:
            # Get available entities from Home Assistant
            entities = []
            if self.ha_client:
                entities = await self.ha_client.get_all_entities()
            
            # Suggest based on available entities
            light_entities = [e for e in entities if e.domain == "light"]
            if light_entities:
                suggestions.append({
                    "title": "Automatic Light Control",
                    "description": "Turn lights on/off based on time or motion",
                    "template": "when sun sets turn on living room lights"
                })
            
            switch_entities = [e for e in entities if e.domain == "switch"]
            if switch_entities:
                suggestions.append({
                    "title": "Smart Switch Automation",
                    "description": "Control switches based on conditions",
                    "template": "when motion detected turn on fan switch"
                })
            
            # Common automation suggestions
            suggestions.extend([
                {
                    "title": "Daily Notification",
                    "description": "Send daily status updates",
                    "template": "every day at 9am send notification with weather"
                },
                {
                    "title": "Security Alert",
                    "description": "Alert when doors/windows opened",
                    "template": "when front door opens send alert notification"
                },
                {
                    "title": "Energy Monitoring",
                    "description": "Monitor and alert on energy usage",
                    "template": "when energy usage above 5kw send alert"
                },
                {
                    "title": "Webhook Integration",
                    "description": "Handle external webhook events",
                    "template": "on webhook trigger execute home assistant action"
                }
            ])
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
        
        return suggestions
    
    async def test_workflow(self, workflow_id: str, test_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test workflow execution"""
        try:
            execution_id = await self.n8n_client.execute_workflow(workflow_id, test_data)
            
            if execution_id:
                # Wait a moment for execution to complete
                await asyncio.sleep(2)
                
                # Get execution details
                execution = await self.n8n_client.get_execution(execution_id, include_data=True)
                
                return {
                    "success": True,
                    "execution_id": execution_id,
                    "status": execution.status if execution else "unknown",
                    "error": execution.error if execution else None,
                    "data": execution.data if execution else {}
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to execute workflow"
                }
                
        except Exception as e:
            logger.error(f"Error testing workflow: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Factory function
async def create_workflow_generator(n8n_client: N8nClient, ha_client: HomeAssistantClient = None) -> WorkflowGenerator:
    """Create workflow generator"""
    return WorkflowGenerator(n8n_client, ha_client)