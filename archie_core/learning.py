"""
Learning and Adaptation System for Archie
Implements adaptive behavior, pattern recognition, and continuous improvement
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import math

from .memory import MemoryManager, Memory, MemoryQuery

logger = logging.getLogger(__name__)

class AdaptationTrigger(Enum):
    """Triggers for adaptation"""
    USER_FEEDBACK = "user_feedback"
    USAGE_PATTERN = "usage_pattern"
    ERROR_PATTERN = "error_pattern"
    PREFERENCE_CHANGE = "preference_change"
    CONTEXT_CHANGE = "context_change"

@dataclass
class LearningEvent:
    """Event for learning system"""
    id: str
    event_type: str
    description: str
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    agent_id: str
    user_id: Optional[str] = None

@dataclass
class AdaptationRule:
    """Rule for behavioral adaptation"""
    id: str
    trigger: AdaptationTrigger
    condition: str
    action: str
    parameters: Dict[str, Any]
    confidence: float
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    success_rate: float = 0.0

@dataclass
class UserPreference:
    """User preference learned from behavior"""
    id: str
    category: str
    preference_type: str
    value: Any
    confidence: float
    learned_from: List[str]
    created_at: datetime
    last_updated: datetime

@dataclass
class BehaviorPattern:
    """Detected user behavior pattern"""
    id: str
    pattern_type: str
    description: str
    features: Dict[str, Any]
    frequency: float
    confidence: float
    last_seen: datetime
    contexts: List[str]

class LearningSystem:
    """Main learning and adaptation system"""
    
    def __init__(self, 
                 memory_manager: MemoryManager,
                 config: Dict[str, Any] = None):
        
        self.memory_manager = memory_manager
        self.config = config or {}
        
        # Learning components
        self.learning_events = deque(maxlen=10000)
        self.adaptation_rules = {}
        self.user_preferences = {}
        self.behavior_patterns = {}
        
        # Pattern recognition
        self.pattern_buffer = deque(maxlen=1000)
        self.pattern_detector = None
        
        # Adaptation parameters
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.7)
        self.pattern_min_samples = self.config.get('pattern_min_samples', 5)
        
        # Statistics
        self.stats = {
            "learning_events": 0,
            "adaptation_rules": 0,
            "user_preferences": 0,
            "behavior_patterns": 0,
            "successful_adaptations": 0,
            "failed_adaptations": 0,
            "last_learning_update": None
        }
    
    async def initialize(self):
        """Initialize the learning system"""
        try:
            # Load existing patterns and rules
            await self._load_adaptation_rules()
            await self._load_user_preferences()
            await self._load_behavior_patterns()
            
            # Initialize pattern detector
            self.pattern_detector = DBSCAN(eps=0.5, min_samples=self.pattern_min_samples)
            
            logger.info("Learning system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize learning system: {e}")
            raise
    
    async def record_learning_event(self, 
                                  event_type: str, 
                                  description: str, 
                                  data: Dict[str, Any],
                                  agent_id: str,
                                  user_id: Optional[str] = None,
                                  confidence: float = 1.0):
        """Record a learning event"""
        try:
            event_id = f"event_{len(self.learning_events)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            event = LearningEvent(
                id=event_id,
                event_type=event_type,
                description=description,
                data=data,
                confidence=confidence,
                timestamp=datetime.now(),
                agent_id=agent_id,
                user_id=user_id
            )
            
            self.learning_events.append(event)
            self.stats["learning_events"] += 1
            
            # Add to pattern buffer for analysis
            self.pattern_buffer.append({
                "timestamp": event.timestamp.timestamp(),
                "event_type": event_type,
                "agent_id": agent_id,
                "user_id": user_id or "unknown",
                "confidence": confidence,
                **data
            })
            
            # Store in memory
            await self.memory_manager.store_memory(
                content=f"Learning event: {description}",
                memory_type="learning_event",
                metadata={
                    "event_type": event_type,
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "confidence": confidence,
                    **data
                },
                importance=confidence,
                tags=["learning", event_type]
            )
            
            # Trigger pattern analysis
            await self._analyze_patterns()
            
            logger.info(f"Recorded learning event: {event_id}")
            
        except Exception as e:
            logger.error(f"Error recording learning event: {e}")
    
    async def _analyze_patterns(self):
        """Analyze patterns in learning events"""
        try:
            if len(self.pattern_buffer) < self.pattern_min_samples:
                return
            
            # Prepare data for pattern analysis
            pattern_data = []
            for event in list(self.pattern_buffer)[-100:]:  # Last 100 events
                features = [
                    event.get("timestamp", 0) % 86400,  # Time of day
                    hash(event.get("event_type", "")) % 1000,  # Event type hash
                    hash(event.get("agent_id", "")) % 1000,  # Agent hash
                    event.get("confidence", 0.5),
                ]
                pattern_data.append(features)
            
            if len(pattern_data) < self.pattern_min_samples:
                return
            
            # Normalize features
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(pattern_data)
            
            # Detect patterns
            clusters = self.pattern_detector.fit_predict(normalized_data)
            
            # Analyze detected patterns
            unique_clusters = set(clusters)
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Noise
                    continue
                
                cluster_events = [event for i, event in enumerate(list(self.pattern_buffer)[-100:]) 
                                if clusters[i] == cluster_id]
                
                await self._create_behavior_pattern(cluster_id, cluster_events)
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
    
    async def _create_behavior_pattern(self, cluster_id: int, events: List[Dict]):
        """Create a behavior pattern from clustered events"""
        try:
            if len(events) < 2:
                return
            
            # Analyze pattern characteristics
            event_types = [event.get("event_type", "") for event in events]
            agent_ids = [event.get("agent_id", "") for event in events]
            timestamps = [event.get("timestamp", 0) for event in events]
            
            # Calculate frequency and confidence
            time_span = max(timestamps) - min(timestamps)
            frequency = len(events) / max(time_span / 86400, 1)  # Events per day
            confidence = min(1.0, len(events) / 10.0)  # Confidence based on sample size
            
            # Create pattern
            pattern_id = f"pattern_{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            pattern = BehaviorPattern(
                id=pattern_id,
                pattern_type="usage_pattern",
                description=f"Recurring pattern: {max(set(event_types), key=event_types.count)} events by {max(set(agent_ids), key=agent_ids.count)}",
                features={
                    "common_event_type": max(set(event_types), key=event_types.count),
                    "common_agent": max(set(agent_ids), key=agent_ids.count),
                    "avg_confidence": sum(event.get("confidence", 0.5) for event in events) / len(events),
                    "time_of_day_pattern": self._analyze_time_pattern(timestamps)
                },
                frequency=frequency,
                confidence=confidence,
                last_seen=datetime.now(),
                contexts=list(set(event.get("user_id", "unknown") for event in events))
            )
            
            self.behavior_patterns[pattern_id] = pattern
            self.stats["behavior_patterns"] = len(self.behavior_patterns)
            
            # Create adaptation rule if pattern is strong enough
            if confidence >= self.adaptation_threshold:
                await self._create_adaptation_rule(pattern)
            
            logger.info(f"Created behavior pattern: {pattern_id}")
            
        except Exception as e:
            logger.error(f"Error creating behavior pattern: {e}")
    
    def _analyze_time_pattern(self, timestamps: List[float]) -> str:
        """Analyze time-based patterns"""
        try:
            hours = [(ts % 86400) / 3600 for ts in timestamps]
            
            # Categorize by time of day
            morning = sum(1 for h in hours if 6 <= h < 12)
            afternoon = sum(1 for h in hours if 12 <= h < 18)
            evening = sum(1 for h in hours if 18 <= h < 22)
            night = sum(1 for h in hours if h >= 22 or h < 6)
            
            time_counts = {"morning": morning, "afternoon": afternoon, "evening": evening, "night": night}
            return max(time_counts, key=time_counts.get)
            
        except Exception as e:
            logger.error(f"Error analyzing time pattern: {e}")
            return "unknown"
    
    async def _create_adaptation_rule(self, pattern: BehaviorPattern):
        """Create adaptation rule from behavior pattern"""
        try:
            rule_id = f"rule_{pattern.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine trigger and action based on pattern
            trigger = AdaptationTrigger.USAGE_PATTERN
            condition = f"pattern_detected:{pattern.id}"
            
            # Create action based on pattern type
            if pattern.features.get("common_event_type") == "error":
                action = "increase_help_suggestions"
            elif pattern.features.get("common_event_type") == "success":
                action = "reduce_confirmation_requests"
            elif pattern.features.get("time_of_day_pattern") == "morning":
                action = "proactive_morning_briefing"
            elif pattern.features.get("time_of_day_pattern") == "evening":
                action = "proactive_evening_summary"
            else:
                action = "maintain_current_behavior"
            
            rule = AdaptationRule(
                id=rule_id,
                trigger=trigger,
                condition=condition,
                action=action,
                parameters={
                    "pattern_id": pattern.id,
                    "confidence_threshold": pattern.confidence,
                    "frequency_threshold": pattern.frequency
                },
                confidence=pattern.confidence,
                created_at=datetime.now()
            )
            
            self.adaptation_rules[rule_id] = rule
            self.stats["adaptation_rules"] = len(self.adaptation_rules)
            
            logger.info(f"Created adaptation rule: {rule_id}")
            
        except Exception as e:
            logger.error(f"Error creating adaptation rule: {e}")
    
    async def process_user_feedback(self, 
                                  feedback: str, 
                                  context: Dict[str, Any],
                                  agent_id: str,
                                  user_id: Optional[str] = None):
        """Process user feedback for learning"""
        try:
            feedback_lower = feedback.lower()
            
            # Analyze feedback sentiment
            positive_keywords = ["good", "great", "excellent", "perfect", "thanks", "helpful", "like"]
            negative_keywords = ["bad", "wrong", "terrible", "horrible", "hate", "annoying", "confusing"]
            
            positive_score = sum(1 for keyword in positive_keywords if keyword in feedback_lower)
            negative_score = sum(1 for keyword in negative_keywords if keyword in feedback_lower)
            
            # Determine sentiment
            if positive_score > negative_score:
                sentiment = "positive"
                confidence = min(1.0, positive_score / 3.0)
            elif negative_score > positive_score:
                sentiment = "negative"
                confidence = min(1.0, negative_score / 3.0)
            else:
                sentiment = "neutral"
                confidence = 0.5
            
            # Record learning event
            await self.record_learning_event(
                event_type="user_feedback",
                description=f"User feedback: {sentiment}",
                data={
                    "feedback": feedback,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "context": context
                },
                agent_id=agent_id,
                user_id=user_id,
                confidence=confidence
            )
            
            # Update user preferences based on feedback
            await self._update_preferences_from_feedback(feedback, sentiment, confidence, context, user_id)
            
        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")
    
    async def _update_preferences_from_feedback(self, 
                                              feedback: str, 
                                              sentiment: str, 
                                              confidence: float,
                                              context: Dict[str, Any],
                                              user_id: Optional[str]):
        """Update user preferences based on feedback"""
        try:
            if sentiment == "neutral" or confidence < 0.3:
                return
            
            # Extract preference indicators from context
            preferences_to_update = []
            
            # Communication preferences
            if "communication" in context:
                preferences_to_update.append({
                    "category": "communication",
                    "preference_type": "style",
                    "value": "verbose" if sentiment == "positive" else "concise",
                    "confidence": confidence
                })
            
            # Response timing preferences
            if "response_time" in context:
                response_time = context["response_time"]
                if sentiment == "positive":
                    preferences_to_update.append({
                        "category": "timing",
                        "preference_type": "response_speed",
                        "value": response_time,
                        "confidence": confidence
                    })
            
            # Tool usage preferences
            if "tool_used" in context:
                tool_name = context["tool_used"]
                preferences_to_update.append({
                    "category": "tools",
                    "preference_type": "tool_preference",
                    "value": {"tool": tool_name, "liked": sentiment == "positive"},
                    "confidence": confidence
                })
            
            # Update preferences
            for pref_data in preferences_to_update:
                await self._update_user_preference(
                    user_id=user_id or "default",
                    category=pref_data["category"],
                    preference_type=pref_data["preference_type"],
                    value=pref_data["value"],
                    confidence=pref_data["confidence"],
                    learned_from=[feedback]
                )
            
        except Exception as e:
            logger.error(f"Error updating preferences from feedback: {e}")
    
    async def _update_user_preference(self, 
                                    user_id: str,
                                    category: str, 
                                    preference_type: str,
                                    value: Any,
                                    confidence: float,
                                    learned_from: List[str]):
        """Update or create user preference"""
        try:
            pref_id = f"{user_id}_{category}_{preference_type}"
            
            if pref_id in self.user_preferences:
                # Update existing preference
                pref = self.user_preferences[pref_id]
                
                # Weighted update based on confidence
                old_weight = pref.confidence
                new_weight = confidence
                total_weight = old_weight + new_weight
                
                if isinstance(value, dict) and isinstance(pref.value, dict):
                    # Merge dictionaries
                    pref.value.update(value)
                elif isinstance(value, str) and isinstance(pref.value, str):
                    # Choose value with higher confidence
                    if new_weight > old_weight:
                        pref.value = value
                else:
                    # Update to new value
                    pref.value = value
                
                pref.confidence = total_weight / 2  # Average confidence
                pref.learned_from.extend(learned_from)
                pref.last_updated = datetime.now()
                
            else:
                # Create new preference
                pref = UserPreference(
                    id=pref_id,
                    category=category,
                    preference_type=preference_type,
                    value=value,
                    confidence=confidence,
                    learned_from=learned_from,
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                
                self.user_preferences[pref_id] = pref
                self.stats["user_preferences"] = len(self.user_preferences)
            
            logger.info(f"Updated user preference: {pref_id}")
            
        except Exception as e:
            logger.error(f"Error updating user preference: {e}")
    
    async def get_user_preferences(self, user_id: str = None) -> Dict[str, UserPreference]:
        """Get user preferences"""
        if user_id is None:
            return self.user_preferences
        
        return {k: v for k, v in self.user_preferences.items() if k.startswith(f"{user_id}_")}
    
    async def apply_adaptation_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply adaptation rules based on current context"""
        try:
            applied_rules = []
            
            for rule_id, rule in self.adaptation_rules.items():
                # Check if rule condition is met
                if await self._check_rule_condition(rule, context):
                    # Apply the rule
                    action_result = await self._apply_rule_action(rule, context)
                    
                    if action_result:
                        applied_rules.append({
                            "rule_id": rule_id,
                            "action": rule.action,
                            "parameters": rule.parameters,
                            "result": action_result
                        })
                        
                        # Update rule usage statistics
                        rule.last_used = datetime.now()
                        rule.usage_count += 1
                        
                        # Update success rate based on result
                        if action_result.get("success", False):
                            rule.success_rate = (rule.success_rate * (rule.usage_count - 1) + 1) / rule.usage_count
                            self.stats["successful_adaptations"] += 1
                        else:
                            rule.success_rate = (rule.success_rate * (rule.usage_count - 1)) / rule.usage_count
                            self.stats["failed_adaptations"] += 1
            
            return applied_rules
            
        except Exception as e:
            logger.error(f"Error applying adaptation rules: {e}")
            return []
    
    async def _check_rule_condition(self, rule: AdaptationRule, context: Dict[str, Any]) -> bool:
        """Check if rule condition is met"""
        try:
            condition = rule.condition
            
            # Simple condition checking
            if condition.startswith("pattern_detected:"):
                pattern_id = condition.split(":", 1)[1]
                return pattern_id in self.behavior_patterns
            
            elif condition.startswith("time_of_day:"):
                time_period = condition.split(":", 1)[1]
                current_hour = datetime.now().hour
                
                if time_period == "morning" and 6 <= current_hour < 12:
                    return True
                elif time_period == "afternoon" and 12 <= current_hour < 18:
                    return True
                elif time_period == "evening" and 18 <= current_hour < 22:
                    return True
                elif time_period == "night" and (current_hour >= 22 or current_hour < 6):
                    return True
            
            elif condition.startswith("user_feedback:"):
                feedback_type = condition.split(":", 1)[1]
                recent_feedback = [event for event in list(self.learning_events)[-10:] 
                                 if event.event_type == "user_feedback"]
                
                if feedback_type == "positive":
                    return any(event.data.get("sentiment") == "positive" for event in recent_feedback)
                elif feedback_type == "negative":
                    return any(event.data.get("sentiment") == "negative" for event in recent_feedback)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rule condition: {e}")
            return False
    
    async def _apply_rule_action(self, rule: AdaptationRule, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rule action"""
        try:
            action = rule.action
            parameters = rule.parameters
            
            result = {"success": False, "message": ""}
            
            if action == "increase_help_suggestions":
                result = {"success": True, "message": "Increased help suggestions", "help_level": "high"}
            
            elif action == "reduce_confirmation_requests":
                result = {"success": True, "message": "Reduced confirmation requests", "confirmation_level": "low"}
            
            elif action == "proactive_morning_briefing":
                result = {"success": True, "message": "Enabled morning briefing", "briefing_time": "08:00"}
            
            elif action == "proactive_evening_summary":
                result = {"success": True, "message": "Enabled evening summary", "summary_time": "20:00"}
            
            elif action == "maintain_current_behavior":
                result = {"success": True, "message": "Maintaining current behavior"}
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying rule action: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}
    
    async def _load_adaptation_rules(self):
        """Load adaptation rules from memory"""
        try:
            query = MemoryQuery(
                query_text="adaptation_rule",
                memory_types=["learning_rule"],
                limit=100
            )
            
            rules = await self.memory_manager.query_memories(query)
            
            for memory, _ in rules:
                try:
                    rule_data = json.loads(memory.content)
                    rule = AdaptationRule(**rule_data)
                    self.adaptation_rules[rule.id] = rule
                except Exception as e:
                    logger.warning(f"Failed to load adaptation rule: {e}")
            
            logger.info(f"Loaded {len(self.adaptation_rules)} adaptation rules")
            
        except Exception as e:
            logger.error(f"Error loading adaptation rules: {e}")
    
    async def _load_user_preferences(self):
        """Load user preferences from memory"""
        try:
            query = MemoryQuery(
                query_text="user_preference",
                memory_types=["user_preference"],
                limit=100
            )
            
            preferences = await self.memory_manager.query_memories(query)
            
            for memory, _ in preferences:
                try:
                    pref_data = json.loads(memory.content)
                    pref = UserPreference(**pref_data)
                    self.user_preferences[pref.id] = pref
                except Exception as e:
                    logger.warning(f"Failed to load user preference: {e}")
            
            logger.info(f"Loaded {len(self.user_preferences)} user preferences")
            
        except Exception as e:
            logger.error(f"Error loading user preferences: {e}")
    
    async def _load_behavior_patterns(self):
        """Load behavior patterns from memory"""
        try:
            query = MemoryQuery(
                query_text="behavior_pattern",
                memory_types=["behavior_pattern"],
                limit=100
            )
            
            patterns = await self.memory_manager.query_memories(query)
            
            for memory, _ in patterns:
                try:
                    pattern_data = json.loads(memory.content)
                    pattern = BehaviorPattern(**pattern_data)
                    self.behavior_patterns[pattern.id] = pattern
                except Exception as e:
                    logger.warning(f"Failed to load behavior pattern: {e}")
            
            logger.info(f"Loaded {len(self.behavior_patterns)} behavior patterns")
            
        except Exception as e:
            logger.error(f"Error loading behavior patterns: {e}")
    
    async def save_learning_data(self):
        """Save learning data to persistent storage"""
        try:
            # Save adaptation rules
            for rule in self.adaptation_rules.values():
                await self.memory_manager.store_memory(
                    content=json.dumps(asdict(rule), default=str),
                    memory_type="learning_rule",
                    metadata={"rule_id": rule.id, "trigger": rule.trigger.value},
                    importance=rule.confidence,
                    tags=["learning", "adaptation_rule"]
                )
            
            # Save user preferences
            for pref in self.user_preferences.values():
                await self.memory_manager.store_memory(
                    content=json.dumps(asdict(pref), default=str),
                    memory_type="user_preference",
                    metadata={"preference_id": pref.id, "category": pref.category},
                    importance=pref.confidence,
                    tags=["learning", "user_preference"]
                )
            
            # Save behavior patterns
            for pattern in self.behavior_patterns.values():
                await self.memory_manager.store_memory(
                    content=json.dumps(asdict(pattern), default=str),
                    memory_type="behavior_pattern",
                    metadata={"pattern_id": pattern.id, "pattern_type": pattern.pattern_type},
                    importance=pattern.confidence,
                    tags=["learning", "behavior_pattern"]
                )
            
            self.stats["last_learning_update"] = datetime.now()
            logger.info("Learning data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about learning system performance"""
        try:
            insights = {
                "stats": self.stats,
                "top_patterns": [],
                "most_adapted_rules": [],
                "user_preference_summary": {},
                "learning_effectiveness": 0.0
            }
            
            # Top behavior patterns
            patterns_by_confidence = sorted(
                self.behavior_patterns.values(),
                key=lambda p: p.confidence,
                reverse=True
            )
            insights["top_patterns"] = [
                {
                    "id": p.id,
                    "description": p.description,
                    "confidence": p.confidence,
                    "frequency": p.frequency
                } for p in patterns_by_confidence[:5]
            ]
            
            # Most used adaptation rules
            rules_by_usage = sorted(
                self.adaptation_rules.values(),
                key=lambda r: r.usage_count,
                reverse=True
            )
            insights["most_adapted_rules"] = [
                {
                    "id": r.id,
                    "action": r.action,
                    "usage_count": r.usage_count,
                    "success_rate": r.success_rate
                } for r in rules_by_usage[:5]
            ]
            
            # User preferences summary
            pref_categories = defaultdict(int)
            for pref in self.user_preferences.values():
                pref_categories[pref.category] += 1
            insights["user_preference_summary"] = dict(pref_categories)
            
            # Learning effectiveness
            total_adaptations = self.stats["successful_adaptations"] + self.stats["failed_adaptations"]
            if total_adaptations > 0:
                insights["learning_effectiveness"] = self.stats["successful_adaptations"] / total_adaptations
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {}

# Factory function for creating learning system
async def create_learning_system(memory_manager: MemoryManager, config: Dict[str, Any]) -> LearningSystem:
    """Create and initialize learning system"""
    learning_config = config.get('learning', {})
    
    system = LearningSystem(
        memory_manager=memory_manager,
        config=learning_config
    )
    
    await system.initialize()
    return system