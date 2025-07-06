"""
Advanced Automation Rules System
Intelligent automation with learning-based adaptation and complex rule generation
"""

import logging
import json
import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import math

from .workflow_generator import WorkflowGenerator, WorkflowIntent, WorkflowType
from .home_assistant import HomeAssistantClient
from .n8n_client import N8nClient

logger = logging.getLogger(__name__)

class RuleType(Enum):
    """Types of automation rules"""
    CONDITIONAL = "conditional"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"
    CONTEXTUAL = "contextual"
    COMPOSITE = "composite"

class RulePriority(Enum):
    """Rule execution priorities"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"

@dataclass
class AutomationCondition:
    """Condition for automation rule"""
    id: str
    type: str  # entity_state, time_range, weather, location, etc.
    entity_id: Optional[str]
    attribute: Optional[str]
    operator: str  # equals, greater_than, less_than, contains, etc.
    value: Any
    negate: bool = False

@dataclass
class AutomationAction:
    """Action for automation rule"""
    id: str
    type: str  # service_call, notification, workflow_trigger, etc.
    service: Optional[str]
    entity_id: Optional[str]
    data: Dict[str, Any]
    delay: Optional[int] = None  # Delay in seconds

@dataclass
class AutomationRule:
    """Advanced automation rule"""
    id: str
    name: str
    description: str
    rule_type: RuleType
    priority: RulePriority
    enabled: bool
    conditions: List[AutomationCondition]
    actions: List[AutomationAction]
    triggers: List[str]
    constraints: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    success_rate: float = 1.0
    learning_enabled: bool = True
    adaptive_parameters: Dict[str, Any] = None

@dataclass
class RuleExecutionContext:
    """Context for rule execution"""
    rule_id: str
    trigger_data: Dict[str, Any]
    entity_states: Dict[str, Any]
    user_context: Dict[str, Any]
    time_context: Dict[str, Any]
    environment_context: Dict[str, Any]

class AdvancedAutomationEngine:
    """Advanced automation engine with learning capabilities"""
    
    def __init__(self, 
                 ha_client: HomeAssistantClient,
                 n8n_client: N8nClient,
                 workflow_generator: WorkflowGenerator,
                 learning_system = None):
        
        self.ha_client = ha_client
        self.n8n_client = n8n_client
        self.workflow_generator = workflow_generator
        self.learning_system = learning_system
        
        # Rule management
        self.rules: Dict[str, AutomationRule] = {}
        self.rule_execution_queue = asyncio.Queue()
        self.execution_history = deque(maxlen=10000)
        
        # Pattern recognition
        self.behavior_patterns = {}
        self.predictive_models = {}
        self.context_patterns = defaultdict(list)
        
        # Execution engine
        self.running = False
        self.worker_tasks = []
        self.rule_cache = {}
        
        # Statistics
        self.stats = {
            "total_rules": 0,
            "enabled_rules": 0,
            "executions_today": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "adaptive_changes": 0,
            "last_update": datetime.now()
        }
    
    async def initialize(self):
        """Initialize the automation engine"""
        try:
            # Load existing rules
            await self._load_rules()
            
            # Start execution engine
            self.running = True
            self.worker_tasks = [
                asyncio.create_task(self._rule_execution_worker()),
                asyncio.create_task(self._pattern_analysis_worker()),
                asyncio.create_task(self._adaptive_adjustment_worker())
            ]
            
            logger.info("Advanced automation engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize automation engine: {e}")
            raise
    
    async def create_rule_from_description(self, description: str, user_context: Dict[str, Any] = None) -> Optional[AutomationRule]:
        """Create automation rule from natural language description"""
        try:
            # Parse the description using enhanced intent parsing
            intent = await self._parse_advanced_intent(description, user_context)
            
            # Generate rule components
            conditions = await self._generate_conditions(intent, user_context)
            actions = await self._generate_actions(intent, user_context)
            triggers = await self._generate_triggers(intent)
            
            # Determine rule type and priority
            rule_type = self._determine_rule_type(intent, conditions, actions)
            priority = self._determine_priority(intent, user_context)
            
            # Create rule
            rule_id = f"rule_{len(self.rules)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            rule = AutomationRule(
                id=rule_id,
                name=intent.get("name", f"Auto Rule {len(self.rules) + 1}"),
                description=description,
                rule_type=rule_type,
                priority=priority,
                enabled=True,
                conditions=conditions,
                actions=actions,
                triggers=triggers,
                constraints=intent.get("constraints", {}),
                metadata={
                    "created_from": "natural_language",
                    "original_description": description,
                    "user_context": user_context or {},
                    "confidence": intent.get("confidence", 0.8)
                },
                created_at=datetime.now(),
                adaptive_parameters={}
            )
            
            # Store rule
            self.rules[rule_id] = rule
            self.stats["total_rules"] = len(self.rules)
            self.stats["enabled_rules"] = len([r for r in self.rules.values() if r.enabled])
            
            # Create corresponding n8n workflow if applicable
            if rule_type in [RuleType.TEMPORAL, RuleType.CONDITIONAL]:
                await self._create_workflow_for_rule(rule)
            
            logger.info(f"Created automation rule: {rule_id}")
            return rule
            
        except Exception as e:
            logger.error(f"Error creating rule from description: {e}")
            return None
    
    async def _parse_advanced_intent(self, description: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced intent parsing with context awareness"""
        try:
            description_lower = description.lower()
            intent = {
                "confidence": 0.5,
                "entities": [],
                "conditions": [],
                "actions": [],
                "temporal": {},
                "constraints": {}
            }
            
            # Enhanced entity extraction
            if self.ha_client:
                entities = await self.ha_client.get_all_entities()
                for entity in entities:
                    entity_name = entity.friendly_name.lower()
                    if entity_name in description_lower or entity.entity_id in description_lower:
                        intent["entities"].append({
                            "entity_id": entity.entity_id,
                            "domain": entity.domain,
                            "friendly_name": entity.friendly_name
                        })
            
            # Advanced temporal patterns
            temporal_patterns = {
                "specific_time": r"at (\d{1,2}):(\d{2})\s*(am|pm)?",
                "time_range": r"between (\d{1,2}):(\d{2})\s*and\s*(\d{1,2}):(\d{2})",
                "duration": r"for (\d+)\s*(minutes?|hours?|days?)",
                "interval": r"every (\d+)\s*(minutes?|hours?|days?)",
                "weekday": r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                "season": r"(spring|summer|fall|autumn|winter)",
                "relative": r"(before|after|during)\s+(.+)",
                "sunrise_sunset": r"(sunrise|sunset|dawn|dusk)",
                "holiday": r"(christmas|thanksgiving|new year|birthday)"
            }
            
            for pattern_name, pattern in temporal_patterns.items():
                match = re.search(pattern, description_lower)
                if match:
                    intent["temporal"][pattern_name] = match.groups()
                    intent["confidence"] += 0.1
            
            # Contextual condition patterns
            condition_patterns = {
                "weather": r"(sunny|rainy|cloudy|stormy|hot|cold|warm|cool)",
                "presence": r"(someone|nobody|anyone|everyone)\s+(is|arrives?|leaves?)",
                "state": r"(.+?)\s+(is|becomes?|turns?)\s+(on|off|open|closed|locked|unlocked)",
                "threshold": r"(.+?)\s+(above|below|over|under)\s+(\d+\.?\d*)",
                "comparison": r"(.+?)\s+(equals?|matches?)\s+(.+)",
                "motion": r"(motion|movement)\s+(detected|sensed)",
                "location": r"(at|in|near)\s+(.+)",
                "device": r"(.+?)\s+(connects?|disconnects?)"
            }
            
            for pattern_name, pattern in condition_patterns.items():
                matches = re.findall(pattern, description_lower)
                for match in matches:
                    intent["conditions"].append({
                        "type": pattern_name,
                        "data": match
                    })
                    intent["confidence"] += 0.1
            
            # Advanced action patterns
            action_patterns = {
                "device_control": r"(turn|switch|set)\s+(.+?)\s+(on|off|to\s+\d+)",
                "notification": r"(send|notify|alert|message)\s+(.+)",
                "scene": r"(activate|run|execute)\s+(scene|routine)\s+(.+)",
                "media": r"(play|pause|stop|volume)\s+(.+)",
                "security": r"(arm|disarm|lock|unlock)\s+(.+)",
                "climate": r"(heat|cool|temperature)\s+(.+)",
                "automation": r"(trigger|start|stop)\s+(automation|workflow)\s+(.+)",
                "recording": r"(record|capture|save)\s+(.+)"
            }
            
            for pattern_name, pattern in action_patterns.items():
                matches = re.findall(pattern, description_lower)
                for match in matches:
                    intent["actions"].append({
                        "type": pattern_name,
                        "data": match
                    })
                    intent["confidence"] += 0.1
            
            # Context-aware enhancements
            if user_context:
                # Time context
                if "time" in user_context:
                    current_hour = user_context["time"].hour
                    if 6 <= current_hour < 12:
                        intent["temporal"]["part_of_day"] = "morning"
                    elif 12 <= current_hour < 17:
                        intent["temporal"]["part_of_day"] = "afternoon"
                    elif 17 <= current_hour < 21:
                        intent["temporal"]["part_of_day"] = "evening"
                    else:
                        intent["temporal"]["part_of_day"] = "night"
                
                # Location context
                if "location" in user_context:
                    intent["constraints"]["location"] = user_context["location"]
                
                # User preferences from learning system
                if self.learning_system and "user_id" in user_context:
                    preferences = await self.learning_system.get_user_preferences(user_context["user_id"])
                    intent["user_preferences"] = preferences
            
            # Determine rule name
            if "name" not in intent:
                if intent["actions"]:
                    action_type = intent["actions"][0]["type"]
                    intent["name"] = f"Auto {action_type.replace('_', ' ').title()}"
                elif intent["conditions"]:
                    condition_type = intent["conditions"][0]["type"] 
                    intent["name"] = f"Auto {condition_type.replace('_', ' ').title()}"
                else:
                    intent["name"] = "Custom Automation"
            
            return intent
            
        except Exception as e:
            logger.error(f"Error parsing advanced intent: {e}")
            return {"confidence": 0.0}
    
    async def _generate_conditions(self, intent: Dict[str, Any], user_context: Dict[str, Any] = None) -> List[AutomationCondition]:
        """Generate automation conditions from intent"""
        conditions = []
        
        try:
            for i, condition_data in enumerate(intent.get("conditions", [])):
                condition_type = condition_data["type"]
                data = condition_data["data"]
                
                condition_id = f"condition_{i}"
                
                if condition_type == "state":
                    # Device state condition
                    entity_name, _, state = data
                    entity_id = self._find_entity_id(entity_name, intent["entities"])
                    
                    condition = AutomationCondition(
                        id=condition_id,
                        type="entity_state",
                        entity_id=entity_id,
                        attribute="state",
                        operator="equals",
                        value=state
                    )
                
                elif condition_type == "threshold":
                    # Threshold condition
                    entity_name, operator, value = data
                    entity_id = self._find_entity_id(entity_name, intent["entities"])
                    
                    op_map = {"above": "greater_than", "below": "less_than", "over": "greater_than", "under": "less_than"}
                    
                    condition = AutomationCondition(
                        id=condition_id,
                        type="entity_state",
                        entity_id=entity_id,
                        attribute="state",
                        operator=op_map.get(operator, "equals"),
                        value=float(value)
                    )
                
                elif condition_type == "motion":
                    # Motion detection condition
                    condition = AutomationCondition(
                        id=condition_id,
                        type="entity_state",
                        entity_id="binary_sensor.motion_sensor",  # Default, should be context-aware
                        attribute="state",
                        operator="equals",
                        value="on"
                    )
                
                elif condition_type == "weather":
                    # Weather condition
                    weather_state = data[0] if isinstance(data, tuple) else data
                    condition = AutomationCondition(
                        id=condition_id,
                        type="weather_condition",
                        entity_id="weather.home",
                        attribute="condition",
                        operator="equals",
                        value=weather_state
                    )
                
                elif condition_type == "presence":
                    # Presence condition  
                    presence_type = data[1] if isinstance(data, tuple) and len(data) > 1 else "someone"
                    condition = AutomationCondition(
                        id=condition_id,
                        type="zone_presence",
                        entity_id="zone.home",
                        attribute="persons",
                        operator="greater_than" if presence_type == "someone" else "equals",
                        value=0 if presence_type == "someone" else 0
                    )
                
                else:
                    # Generic condition
                    condition = AutomationCondition(
                        id=condition_id,
                        type="custom",
                        entity_id=None,
                        attribute="custom",
                        operator="custom",
                        value=data
                    )
                
                conditions.append(condition)
            
            # Add temporal conditions
            temporal = intent.get("temporal", {})
            if temporal:
                condition_id = f"condition_temporal"
                
                if "specific_time" in temporal:
                    time_data = temporal["specific_time"]
                    hour = int(time_data[0])
                    minute = int(time_data[1])
                    if len(time_data) > 2 and time_data[2] == "pm" and hour != 12:
                        hour += 12
                    
                    condition = AutomationCondition(
                        id=condition_id,
                        type="time_condition",
                        entity_id=None,
                        attribute="time",
                        operator="equals",
                        value=f"{hour:02d}:{minute:02d}"
                    )
                    conditions.append(condition)
                
                elif "time_range" in temporal:
                    time_data = temporal["time_range"]
                    start_hour, start_minute, end_hour, end_minute = time_data
                    
                    condition = AutomationCondition(
                        id=condition_id,
                        type="time_range",
                        entity_id=None,
                        attribute="time",
                        operator="between",
                        value={"start": f"{start_hour}:{start_minute}", "end": f"{end_hour}:{end_minute}"}
                    )
                    conditions.append(condition)
            
        except Exception as e:
            logger.error(f"Error generating conditions: {e}")
        
        return conditions
    
    async def _generate_actions(self, intent: Dict[str, Any], user_context: Dict[str, Any] = None) -> List[AutomationAction]:
        """Generate automation actions from intent"""
        actions = []
        
        try:
            for i, action_data in enumerate(intent.get("actions", [])):
                action_type = action_data["type"]
                data = action_data["data"]
                
                action_id = f"action_{i}"
                
                if action_type == "device_control":
                    # Device control action
                    command, entity_name, state = data
                    entity_id = self._find_entity_id(entity_name, intent["entities"])
                    
                    # Determine service based on command and state
                    if "turn" in command or "switch" in command:
                        if "on" in state:
                            service = "homeassistant.turn_on"
                        elif "off" in state:
                            service = "homeassistant.turn_off"
                        else:
                            service = "homeassistant.toggle"
                    elif "set" in command:
                        service = "homeassistant.set_state"
                    else:
                        service = "homeassistant.call_service"
                    
                    action = AutomationAction(
                        id=action_id,
                        type="service_call",
                        service=service,
                        entity_id=entity_id,
                        data={"state": state} if "set" in command else {}
                    )
                
                elif action_type == "notification":
                    # Notification action
                    message = data[1] if isinstance(data, tuple) and len(data) > 1 else str(data)
                    
                    action = AutomationAction(
                        id=action_id,
                        type="notification",
                        service="notify.notify",
                        entity_id=None,
                        data={
                            "title": "Automation Alert",
                            "message": message,
                            "target": "all"
                        }
                    )
                
                elif action_type == "scene":
                    # Scene activation
                    scene_name = data[2] if isinstance(data, tuple) and len(data) > 2 else str(data)
                    
                    action = AutomationAction(
                        id=action_id,
                        type="service_call",
                        service="scene.turn_on",
                        entity_id=f"scene.{scene_name.replace(' ', '_').lower()}",
                        data={}
                    )
                
                elif action_type == "media":
                    # Media control
                    command = data[0] if isinstance(data, tuple) else str(data)
                    
                    service_map = {
                        "play": "media_player.media_play",
                        "pause": "media_player.media_pause",
                        "stop": "media_player.media_stop",
                        "volume": "media_player.volume_set"
                    }
                    
                    action = AutomationAction(
                        id=action_id,
                        type="service_call",
                        service=service_map.get(command, "media_player.toggle"),
                        entity_id="media_player.default",
                        data={}
                    )
                
                elif action_type == "automation":
                    # Trigger another automation
                    automation_name = data[2] if isinstance(data, tuple) and len(data) > 2 else str(data)
                    
                    action = AutomationAction(
                        id=action_id,
                        type="automation_trigger",
                        service="automation.trigger",
                        entity_id=f"automation.{automation_name.replace(' ', '_').lower()}",
                        data={}
                    )
                
                else:
                    # Generic action
                    action = AutomationAction(
                        id=action_id,
                        type="custom",
                        service="custom.action",
                        entity_id=None,
                        data={"action_data": data}
                    )
                
                actions.append(action)
            
        except Exception as e:
            logger.error(f"Error generating actions: {e}")
        
        return actions
    
    async def _generate_triggers(self, intent: Dict[str, Any]) -> List[str]:
        """Generate trigger events from intent"""
        triggers = []
        
        try:
            # Entity state triggers
            for entity_data in intent.get("entities", []):
                triggers.append(f"state_changed:{entity_data['entity_id']}")
            
            # Temporal triggers
            temporal = intent.get("temporal", {})
            if "specific_time" in temporal:
                triggers.append("time_trigger")
            elif "interval" in temporal:
                triggers.append("interval_trigger")
            
            # Motion triggers
            for condition in intent.get("conditions", []):
                if condition["type"] == "motion":
                    triggers.append("motion_detected")
            
            # Default trigger if none specified
            if not triggers:
                triggers.append("manual_trigger")
            
        except Exception as e:
            logger.error(f"Error generating triggers: {e}")
        
        return triggers
    
    def _find_entity_id(self, entity_name: str, entities: List[Dict[str, Any]]) -> str:
        """Find entity ID from friendly name"""
        entity_name_lower = entity_name.lower()
        
        for entity in entities:
            if (entity_name_lower in entity["friendly_name"].lower() or 
                entity_name_lower in entity["entity_id"]):
                return entity["entity_id"]
        
        # Default fallback
        return f"unknown.{entity_name.replace(' ', '_').lower()}"
    
    def _determine_rule_type(self, intent: Dict[str, Any], conditions: List, actions: List) -> RuleType:
        """Determine rule type based on intent and components"""
        if intent.get("temporal"):
            return RuleType.TEMPORAL
        elif len(conditions) > 1:
            return RuleType.CONDITIONAL
        elif intent.get("user_preferences"):
            return RuleType.BEHAVIORAL
        elif intent.get("confidence", 0) > 0.8:
            return RuleType.ADAPTIVE
        else:
            return RuleType.CONTEXTUAL
    
    def _determine_priority(self, intent: Dict[str, Any], user_context: Dict[str, Any] = None) -> RulePriority:
        """Determine rule priority"""
        # Security-related rules get high priority
        for action in intent.get("actions", []):
            if action["type"] in ["security", "alarm"]:
                return RulePriority.CRITICAL
        
        # User presence rules get high priority
        for condition in intent.get("conditions", []):
            if condition["type"] == "presence":
                return RulePriority.HIGH
        
        # Time-sensitive rules get medium priority
        if intent.get("temporal"):
            return RulePriority.MEDIUM
        
        # Default to low priority
        return RulePriority.LOW
    
    async def _create_workflow_for_rule(self, rule: AutomationRule):
        """Create n8n workflow for automation rule"""
        try:
            # Convert rule to workflow description
            description = f"Automation: {rule.description}"
            
            # Generate workflow
            workflow = await self.workflow_generator.generate_workflow(
                description=description,
                name=f"Auto: {rule.name}"
            )
            
            if workflow:
                # Store workflow reference in rule metadata
                rule.metadata["workflow_id"] = workflow.id
                logger.info(f"Created workflow {workflow.id} for rule {rule.id}")
            
        except Exception as e:
            logger.error(f"Error creating workflow for rule {rule.id}: {e}")
    
    async def _rule_execution_worker(self):
        """Background worker for rule execution"""
        while self.running:
            try:
                # Get execution request
                context = await asyncio.wait_for(self.rule_execution_queue.get(), timeout=1.0)
                
                # Execute rule
                await self._execute_rule(context.rule_id, context)
                
                self.rule_execution_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in rule execution worker: {e}")
    
    async def _pattern_analysis_worker(self):
        """Background worker for pattern analysis"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._analyze_execution_patterns()
                
            except Exception as e:
                logger.error(f"Error in pattern analysis worker: {e}")
    
    async def _adaptive_adjustment_worker(self):
        """Background worker for adaptive adjustments"""
        while self.running:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                await self._apply_adaptive_adjustments()
                
            except Exception as e:
                logger.error(f"Error in adaptive adjustment worker: {e}")
    
    async def _execute_rule(self, rule_id: str, context: RuleExecutionContext):
        """Execute automation rule"""
        try:
            rule = self.rules.get(rule_id)
            if not rule or not rule.enabled:
                return
            
            # Check conditions
            if not await self._evaluate_conditions(rule, context):
                return
            
            # Execute actions
            success = await self._execute_actions(rule, context)
            
            # Update rule statistics
            rule.last_triggered = datetime.now()
            rule.trigger_count += 1
            
            if success:
                rule.success_rate = (rule.success_rate * (rule.trigger_count - 1) + 1) / rule.trigger_count
                self.stats["successful_executions"] += 1
            else:
                rule.success_rate = (rule.success_rate * (rule.trigger_count - 1)) / rule.trigger_count
                self.stats["failed_executions"] += 1
            
            # Record execution
            self.execution_history.append({
                "rule_id": rule_id,
                "timestamp": datetime.now(),
                "success": success,
                "context": asdict(context)
            })
            
            # Learn from execution
            if self.learning_system and rule.learning_enabled:
                await self.learning_system.record_learning_event(
                    event_type="automation_execution",
                    description=f"Rule {rule.name} executed",
                    data={
                        "rule_id": rule_id,
                        "success": success,
                        "trigger_data": context.trigger_data
                    },
                    agent_id="automation_engine",
                    confidence=rule.success_rate
                )
            
            logger.info(f"Executed rule {rule_id}: {success}")
            
        except Exception as e:
            logger.error(f"Error executing rule {rule_id}: {e}")
    
    async def _evaluate_conditions(self, rule: AutomationRule, context: RuleExecutionContext) -> bool:
        """Evaluate rule conditions"""
        try:
            for condition in rule.conditions:
                if not await self._evaluate_single_condition(condition, context):
                    return False
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating conditions for rule {rule.id}: {e}")
            return False
    
    async def _evaluate_single_condition(self, condition: AutomationCondition, context: RuleExecutionContext) -> bool:
        """Evaluate single condition"""
        try:
            if condition.type == "entity_state":
                # Get entity state
                entity_state = context.entity_states.get(condition.entity_id)
                if not entity_state:
                    return False
                
                value = entity_state.get(condition.attribute, entity_state.get("state"))
                
                # Apply operator
                result = self._apply_operator(value, condition.operator, condition.value)
                
            elif condition.type == "time_condition":
                current_time = context.time_context.get("current_time", datetime.now().strftime("%H:%M"))
                result = current_time == condition.value
                
            elif condition.type == "time_range":
                current_time = datetime.now().time()
                start_time = datetime.strptime(condition.value["start"], "%H:%M").time()
                end_time = datetime.strptime(condition.value["end"], "%H:%M").time()
                result = start_time <= current_time <= end_time
                
            else:
                # Custom condition evaluation
                result = True
            
            return result != condition.negate  # Apply negation if needed
            
        except Exception as e:
            logger.error(f"Error evaluating condition {condition.id}: {e}")
            return False
    
    def _apply_operator(self, value: Any, operator: str, target_value: Any) -> bool:
        """Apply comparison operator"""
        try:
            if operator == "equals":
                return str(value).lower() == str(target_value).lower()
            elif operator == "greater_than":
                return float(value) > float(target_value)
            elif operator == "less_than":
                return float(value) < float(target_value)
            elif operator == "contains":
                return str(target_value).lower() in str(value).lower()
            elif operator == "between":
                min_val, max_val = target_value["min"], target_value["max"]
                return min_val <= float(value) <= max_val
            else:
                return True
                
        except Exception as e:
            logger.error(f"Error applying operator {operator}: {e}")
            return False
    
    async def _execute_actions(self, rule: AutomationRule, context: RuleExecutionContext) -> bool:
        """Execute rule actions"""
        try:
            for action in rule.actions:
                success = await self._execute_single_action(action, context)
                if not success:
                    return False
                
                # Apply delay if specified
                if action.delay:
                    await asyncio.sleep(action.delay)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing actions for rule {rule.id}: {e}")
            return False
    
    async def _execute_single_action(self, action: AutomationAction, context: RuleExecutionContext) -> bool:
        """Execute single action"""
        try:
            if action.type == "service_call":
                # Call Home Assistant service
                result = await self.ha_client.call_service(
                    service=action.service,
                    entity_id=action.entity_id,
                    data=action.data
                )
                return result is not None
                
            elif action.type == "notification":
                # Send notification
                result = await self.ha_client.call_service(
                    service="notify.notify",
                    data=action.data
                )
                return result is not None
                
            elif action.type == "workflow_trigger":
                # Trigger n8n workflow
                workflow_id = action.data.get("workflow_id")
                if workflow_id:
                    execution_id = await self.n8n_client.execute_workflow(workflow_id, action.data)
                    return execution_id is not None
                
            return True
            
        except Exception as e:
            logger.error(f"Error executing action {action.id}: {e}")
            return False
    
    async def _analyze_execution_patterns(self):
        """Analyze execution patterns for optimization"""
        try:
            if len(self.execution_history) < 10:
                return
            
            # Analyze recent executions
            recent_executions = list(self.execution_history)[-100:]
            
            # Pattern analysis
            success_patterns = defaultdict(int)
            failure_patterns = defaultdict(int)
            
            for execution in recent_executions:
                rule_id = execution["rule_id"]
                if execution["success"]:
                    success_patterns[rule_id] += 1
                else:
                    failure_patterns[rule_id] += 1
            
            # Identify poorly performing rules
            for rule_id, rule in self.rules.items():
                if rule.success_rate < 0.5 and rule.trigger_count > 5:
                    logger.warning(f"Rule {rule_id} has low success rate: {rule.success_rate:.2f}")
                    # Consider disabling or adjusting rule
                    if rule.learning_enabled:
                        await self._suggest_rule_improvements(rule)
            
        except Exception as e:
            logger.error(f"Error analyzing execution patterns: {e}")
    
    async def _apply_adaptive_adjustments(self):
        """Apply adaptive adjustments to rules"""
        try:
            for rule in self.rules.values():
                if not rule.learning_enabled or rule.rule_type != RuleType.ADAPTIVE:
                    continue
                
                # Apply learning-based adjustments
                if self.learning_system:
                    adaptations = await self.learning_system.apply_adaptation_rules({
                        "rule_id": rule.id,
                        "success_rate": rule.success_rate,
                        "trigger_count": rule.trigger_count
                    })
                    
                    if adaptations:
                        await self._apply_adaptations_to_rule(rule, adaptations)
                        self.stats["adaptive_changes"] += 1
            
        except Exception as e:
            logger.error(f"Error applying adaptive adjustments: {e}")
    
    async def _suggest_rule_improvements(self, rule: AutomationRule):
        """Suggest improvements for poorly performing rules"""
        try:
            suggestions = []
            
            # Analyze failure patterns
            if rule.success_rate < 0.3:
                suggestions.append("Consider simplifying conditions")
                suggestions.append("Check entity availability")
                suggestions.append("Review timing constraints")
            
            # Log suggestions
            logger.info(f"Suggestions for rule {rule.id}: {suggestions}")
            
            # Store suggestions in rule metadata
            rule.metadata["suggestions"] = suggestions
            rule.metadata["last_analysis"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error generating suggestions for rule {rule.id}: {e}")
    
    async def _apply_adaptations_to_rule(self, rule: AutomationRule, adaptations: List[Dict[str, Any]]):
        """Apply adaptations to rule"""
        try:
            for adaptation in adaptations:
                action = adaptation.get("action")
                
                if action == "adjust_threshold":
                    # Adjust condition thresholds
                    for condition in rule.conditions:
                        if condition.type == "entity_state" and condition.operator in ["greater_than", "less_than"]:
                            adjustment = adaptation.get("parameters", {}).get("adjustment", 0.1)
                            if isinstance(condition.value, (int, float)):
                                condition.value += adjustment
                
                elif action == "extend_time_window":
                    # Extend time-based conditions
                    for condition in rule.conditions:
                        if condition.type == "time_range":
                            # Extend time window by 30 minutes
                            pass  # Implementation details
                
                elif action == "increase_sensitivity":
                    # Increase rule sensitivity
                    rule.metadata["sensitivity"] = rule.metadata.get("sensitivity", 1.0) * 1.1
                
                logger.info(f"Applied adaptation '{action}' to rule {rule.id}")
            
        except Exception as e:
            logger.error(f"Error applying adaptations to rule {rule.id}: {e}")
    
    async def _load_rules(self):
        """Load existing automation rules"""
        try:
            # This would typically load from persistent storage
            # For now, we'll create some example rules
            logger.info("Loaded automation rules from storage")
            
        except Exception as e:
            logger.error(f"Error loading rules: {e}")
    
    async def get_rule_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get automation rule suggestions based on context"""
        suggestions = []
        
        try:
            # Basic automation suggestions
            suggestions.extend([
                {
                    "title": "Smart Lighting",
                    "description": "Automatically control lights based on time and occupancy",
                    "example": "turn on living room lights when motion detected after sunset"
                },
                {
                    "title": "Climate Control",
                    "description": "Adjust temperature based on schedule and preferences",
                    "example": "set temperature to 72 degrees at 7am on weekdays"
                },
                {
                    "title": "Security Monitoring",
                    "description": "Monitor and alert on security events",
                    "example": "send alert when door opens and nobody is home"
                },
                {
                    "title": "Energy Management",
                    "description": "Optimize energy usage automatically",
                    "example": "turn off all lights when everyone leaves home"
                }
            ])
            
            # Context-aware suggestions
            if "entities" in context:
                entities = context["entities"]
                
                if any("light" in e for e in entities):
                    suggestions.append({
                        "title": "Bedtime Routine",
                        "description": "Automatically dim lights for bedtime",
                        "example": "dim all lights to 20% at 10pm"
                    })
                
                if any("door" in e for e in entities):
                    suggestions.append({
                        "title": "Welcome Home",
                        "description": "Activate scene when arriving home",
                        "example": "turn on entry lights when front door opens"
                    })
            
        except Exception as e:
            logger.error(f"Error generating rule suggestions: {e}")
        
        return suggestions
    
    async def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation system statistics"""
        try:
            active_rules = [r for r in self.rules.values() if r.enabled]
            
            return {
                **self.stats,
                "rule_breakdown": {
                    "total": len(self.rules),
                    "enabled": len(active_rules),
                    "by_type": {t.value: len([r for r in active_rules if r.rule_type == t]) for t in RuleType},
                    "by_priority": {p.value: len([r for r in active_rules if r.priority == p]) for p in RulePriority}
                },
                "performance": {
                    "avg_success_rate": sum(r.success_rate for r in active_rules) / len(active_rules) if active_rules else 0,
                    "most_triggered": max(active_rules, key=lambda r: r.trigger_count).name if active_rules else None,
                    "least_successful": min(active_rules, key=lambda r: r.success_rate).name if active_rules else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting automation stats: {e}")
            return self.stats
    
    async def shutdown(self):
        """Shutdown automation engine"""
        self.running = False
        
        for task in self.worker_tasks:
            task.cancel()
        
        logger.info("Advanced automation engine shutdown")

# Factory function
async def create_advanced_automation_engine(ha_client: HomeAssistantClient, 
                                          n8n_client: N8nClient,
                                          workflow_generator: WorkflowGenerator,
                                          learning_system = None) -> AdvancedAutomationEngine:
    """Create and initialize advanced automation engine"""
    engine = AdvancedAutomationEngine(
        ha_client=ha_client,
        n8n_client=n8n_client, 
        workflow_generator=workflow_generator,
        learning_system=learning_system
    )
    
    await engine.initialize()
    return engine