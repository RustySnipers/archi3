"""
Device Control Interfaces
Unified device control system for smart home automation
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import re

from .home_assistant import HomeAssistantClient, HAEntity
from .n8n_client import N8nClient

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Types of controllable devices"""
    LIGHT = "light"
    SWITCH = "switch"
    SENSOR = "sensor"
    CLIMATE = "climate"
    COVER = "cover"
    FAN = "fan"
    LOCK = "lock"
    CAMERA = "camera"
    MEDIA_PLAYER = "media_player"
    VACUUM = "vacuum"
    BINARY_SENSOR = "binary_sensor"
    AUTOMATION = "automation"
    SCRIPT = "script"
    SCENE = "scene"
    GROUP = "group"

class DeviceState(Enum):
    """Device states"""
    ON = "on"
    OFF = "off"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    OPEN = "open"
    CLOSED = "closed"
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    PLAYING = "playing"
    PAUSED = "paused"
    IDLE = "idle"

@dataclass
class DeviceInfo:
    """Device information structure"""
    entity_id: str
    name: str
    device_type: DeviceType
    state: DeviceState
    attributes: Dict[str, Any]
    area: Optional[str]
    capabilities: List[str]
    last_updated: datetime
    available: bool

@dataclass
class DeviceAction:
    """Device action structure"""
    action: str
    entity_id: str
    parameters: Dict[str, Any]
    description: str

@dataclass
class SceneDefinition:
    """Scene definition structure"""
    name: str
    entities: Dict[str, Dict[str, Any]]
    description: str

class DeviceController:
    """Unified device control system"""
    
    def __init__(self, ha_client: HomeAssistantClient, n8n_client: N8nClient = None):
        self.ha_client = ha_client
        self.n8n_client = n8n_client
        
        # Device cache
        self.devices = {}
        self.areas = {}
        self.scenes = {}
        self.groups = {}
        
        # Capability mappings
        self.capabilities = {
            DeviceType.LIGHT: ["turn_on", "turn_off", "toggle", "brightness", "color", "color_temp"],
            DeviceType.SWITCH: ["turn_on", "turn_off", "toggle"],
            DeviceType.CLIMATE: ["set_temperature", "set_hvac_mode", "turn_on", "turn_off"],
            DeviceType.COVER: ["open_cover", "close_cover", "set_cover_position", "stop_cover"],
            DeviceType.FAN: ["turn_on", "turn_off", "set_speed", "oscillate"],
            DeviceType.LOCK: ["lock", "unlock"],
            DeviceType.MEDIA_PLAYER: ["turn_on", "turn_off", "play_media", "volume_set", "volume_up", "volume_down"],
            DeviceType.VACUUM: ["start", "stop", "return_to_base", "locate"],
            DeviceType.AUTOMATION: ["turn_on", "turn_off", "trigger"],
            DeviceType.SCRIPT: ["turn_on"],
            DeviceType.SCENE: ["turn_on"],
            DeviceType.GROUP: ["turn_on", "turn_off", "toggle"]
        }
        
        # Natural language patterns for device control
        self.control_patterns = {
            "turn_on": [
                r"turn on (.+)",
                r"switch on (.+)",
                r"activate (.+)",
                r"enable (.+)",
                r"start (.+)"
            ],
            "turn_off": [
                r"turn off (.+)",
                r"switch off (.+)",
                r"deactivate (.+)",
                r"disable (.+)",
                r"stop (.+)"
            ],
            "toggle": [
                r"toggle (.+)",
                r"flip (.+)",
                r"switch (.+)"
            ],
            "brightness": [
                r"set (.+) brightness to (\d+)",
                r"dim (.+) to (\d+)",
                r"brighten (.+) to (\d+)"
            ],
            "temperature": [
                r"set (.+) temperature to (\d+)",
                r"heat (.+) to (\d+)",
                r"cool (.+) to (\d+)"
            ],
            "volume": [
                r"set (.+) volume to (\d+)",
                r"volume (.+) (\d+)"
            ]
        }
        
        # Statistics
        self.stats = {
            "commands_executed": 0,
            "successful_commands": 0,
            "failed_commands": 0,
            "scenes_activated": 0,
            "automations_triggered": 0,
            "last_refresh": None
        }
    
    async def initialize(self):
        """Initialize device controller"""
        try:
            logger.info("Initializing device controller...")
            
            await self._refresh_devices()
            await self._refresh_areas()
            await self._refresh_scenes()
            
            logger.info(f"Device controller initialized with {len(self.devices)} devices")
            
        except Exception as e:
            logger.error(f"Failed to initialize device controller: {e}")
            raise
    
    async def _refresh_devices(self):
        """Refresh device information from Home Assistant"""
        try:
            entities = await self.ha_client.get_all_entities()
            self.devices.clear()
            
            for entity in entities:
                device_type = self._get_device_type(entity.domain)
                if device_type:
                    device_state = self._map_state(entity.state)
                    
                    device = DeviceInfo(
                        entity_id=entity.entity_id,
                        name=entity.friendly_name,
                        device_type=device_type,
                        state=device_state,
                        attributes=entity.attributes,
                        area=entity.attributes.get("area_id"),
                        capabilities=self.capabilities.get(device_type, []),
                        last_updated=entity.last_updated,
                        available=entity.state != "unavailable"
                    )
                    
                    self.devices[entity.entity_id] = device
            
            self.stats["last_refresh"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error refreshing devices: {e}")
            raise
    
    async def _refresh_areas(self):
        """Refresh area information"""
        try:
            areas = await self.ha_client.get_areas()
            self.areas = {area.area_id: area.name for area in areas}
            
        except Exception as e:
            logger.error(f"Error refreshing areas: {e}")
    
    async def _refresh_scenes(self):
        """Refresh scene information"""
        try:
            scene_entities = await self.ha_client.find_entities_by_domain("scene")
            self.scenes.clear()
            
            for scene in scene_entities:
                self.scenes[scene.entity_id] = {
                    "name": scene.friendly_name,
                    "entity_id": scene.entity_id,
                    "last_updated": scene.last_updated
                }
                
        except Exception as e:
            logger.error(f"Error refreshing scenes: {e}")
    
    def _get_device_type(self, domain: str) -> Optional[DeviceType]:
        """Map Home Assistant domain to device type"""
        domain_mapping = {
            "light": DeviceType.LIGHT,
            "switch": DeviceType.SWITCH,
            "sensor": DeviceType.SENSOR,
            "binary_sensor": DeviceType.BINARY_SENSOR,
            "climate": DeviceType.CLIMATE,
            "cover": DeviceType.COVER,
            "fan": DeviceType.FAN,
            "lock": DeviceType.LOCK,
            "camera": DeviceType.CAMERA,
            "media_player": DeviceType.MEDIA_PLAYER,
            "vacuum": DeviceType.VACUUM,
            "automation": DeviceType.AUTOMATION,
            "script": DeviceType.SCRIPT,
            "scene": DeviceType.SCENE,
            "group": DeviceType.GROUP
        }
        
        return domain_mapping.get(domain)
    
    def _map_state(self, state: str) -> DeviceState:
        """Map Home Assistant state to device state"""
        state_mapping = {
            "on": DeviceState.ON,
            "off": DeviceState.OFF,
            "unavailable": DeviceState.UNAVAILABLE,
            "unknown": DeviceState.UNKNOWN,
            "open": DeviceState.OPEN,
            "closed": DeviceState.CLOSED,
            "locked": DeviceState.LOCKED,
            "unlocked": DeviceState.UNLOCKED,
            "playing": DeviceState.PLAYING,
            "paused": DeviceState.PAUSED,
            "idle": DeviceState.IDLE
        }
        
        return state_mapping.get(state.lower(), DeviceState.UNKNOWN)
    
    async def get_devices(self, 
                         device_type: DeviceType = None, 
                         area: str = None,
                         available_only: bool = True) -> List[DeviceInfo]:
        """Get devices with optional filtering"""
        try:
            devices = list(self.devices.values())
            
            # Filter by device type
            if device_type:
                devices = [d for d in devices if d.device_type == device_type]
            
            # Filter by area
            if area:
                area_id = None
                # Find area ID by name
                for aid, aname in self.areas.items():
                    if aname.lower() == area.lower():
                        area_id = aid
                        break
                
                if area_id:
                    devices = [d for d in devices if d.area == area_id]
            
            # Filter by availability
            if available_only:
                devices = [d for d in devices if d.available]
            
            return devices
            
        except Exception as e:
            logger.error(f"Error getting devices: {e}")
            return []
    
    async def get_device(self, entity_id: str) -> Optional[DeviceInfo]:
        """Get specific device information"""
        try:
            if entity_id in self.devices:
                return self.devices[entity_id]
            
            # Try to find by name
            for device in self.devices.values():
                if device.name.lower() == entity_id.lower():
                    return device
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting device {entity_id}: {e}")
            return None
    
    async def find_devices(self, query: str) -> List[DeviceInfo]:
        """Find devices by name or description"""
        try:
            query_lower = query.lower()
            matches = []
            
            for device in self.devices.values():
                # Check name
                if query_lower in device.name.lower():
                    matches.append(device)
                    continue
                
                # Check entity ID
                if query_lower in device.entity_id.lower():
                    matches.append(device)
                    continue
                
                # Check area
                if device.area and device.area in self.areas:
                    area_name = self.areas[device.area].lower()
                    if query_lower in area_name:
                        matches.append(device)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error finding devices: {e}")
            return []
    
    async def control_device(self, entity_id: str, action: str, **kwargs) -> bool:
        """Control a device"""
        try:
            device = await self.get_device(entity_id)
            if not device:
                logger.warning(f"Device not found: {entity_id}")
                return False
            
            if not device.available:
                logger.warning(f"Device unavailable: {entity_id}")
                return False
            
            # Check if action is supported
            if action not in device.capabilities:
                logger.warning(f"Action {action} not supported for {entity_id}")
                return False
            
            self.stats["commands_executed"] += 1
            
            # Execute action based on device type
            success = False
            
            if action in ["turn_on", "turn_off", "toggle"]:
                if action == "turn_on":
                    success = await self.ha_client.turn_on(entity_id, **kwargs)
                elif action == "turn_off":
                    success = await self.ha_client.turn_off(entity_id, **kwargs)
                elif action == "toggle":
                    success = await self.ha_client.toggle(entity_id, **kwargs)
            
            elif action == "brightness" and device.device_type == DeviceType.LIGHT:
                brightness = kwargs.get("brightness", 255)
                success = await self.ha_client.turn_on(entity_id, brightness=brightness)
            
            elif action == "color" and device.device_type == DeviceType.LIGHT:
                color = kwargs.get("color")
                if color:
                    success = await self.ha_client.turn_on(entity_id, rgb_color=color)
            
            elif action == "set_temperature" and device.device_type == DeviceType.CLIMATE:
                temperature = kwargs.get("temperature")
                if temperature:
                    success = await self.ha_client.call_service(
                        "climate", "set_temperature",
                        {"entity_id": entity_id, "temperature": temperature}
                    )
            
            elif action == "set_hvac_mode" and device.device_type == DeviceType.CLIMATE:
                hvac_mode = kwargs.get("hvac_mode")
                if hvac_mode:
                    success = await self.ha_client.call_service(
                        "climate", "set_hvac_mode",
                        {"entity_id": entity_id, "hvac_mode": hvac_mode}
                    )
            
            elif action in ["open_cover", "close_cover", "stop_cover"] and device.device_type == DeviceType.COVER:
                success = await self.ha_client.call_service("cover", action, {"entity_id": entity_id})
            
            elif action == "set_cover_position" and device.device_type == DeviceType.COVER:
                position = kwargs.get("position", 50)
                success = await self.ha_client.call_service(
                    "cover", "set_cover_position",
                    {"entity_id": entity_id, "position": position}
                )
            
            elif action == "lock" and device.device_type == DeviceType.LOCK:
                success = await self.ha_client.call_service("lock", "lock", {"entity_id": entity_id})
            
            elif action == "unlock" and device.device_type == DeviceType.LOCK:
                success = await self.ha_client.call_service("lock", "unlock", {"entity_id": entity_id})
            
            elif action == "play_media" and device.device_type == DeviceType.MEDIA_PLAYER:
                media_content_type = kwargs.get("media_content_type", "music")
                media_content_id = kwargs.get("media_content_id", "")
                success = await self.ha_client.call_service(
                    "media_player", "play_media",
                    {
                        "entity_id": entity_id,
                        "media_content_type": media_content_type,
                        "media_content_id": media_content_id
                    }
                )
            
            elif action == "volume_set" and device.device_type == DeviceType.MEDIA_PLAYER:
                volume_level = kwargs.get("volume_level", 0.5)
                success = await self.ha_client.call_service(
                    "media_player", "volume_set",
                    {"entity_id": entity_id, "volume_level": volume_level}
                )
            
            else:
                # Generic service call
                domain = entity_id.split('.')[0]
                success = await self.ha_client.call_service(domain, action, {"entity_id": entity_id, **kwargs})
            
            if success:
                self.stats["successful_commands"] += 1
                logger.info(f"Successfully executed {action} on {entity_id}")
            else:
                self.stats["failed_commands"] += 1
                logger.warning(f"Failed to execute {action} on {entity_id}")
            
            return success
            
        except Exception as e:
            self.stats["failed_commands"] += 1
            logger.error(f"Error controlling device {entity_id}: {e}")
            return False
    
    async def parse_and_execute(self, command: str) -> Dict[str, Any]:
        """Parse natural language command and execute"""
        try:
            command_lower = command.lower().strip()
            
            # Parse command using patterns
            for action, patterns in self.control_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, command_lower)
                    if match:
                        if action in ["turn_on", "turn_off", "toggle"]:
                            device_name = match.group(1)
                            devices = await self.find_devices(device_name)
                            
                            if devices:
                                results = []
                                for device in devices:
                                    success = await self.control_device(device.entity_id, action)
                                    results.append({
                                        "device": device.name,
                                        "action": action,
                                        "success": success
                                    })
                                
                                return {
                                    "success": True,
                                    "command": command,
                                    "action": action,
                                    "results": results
                                }
                        
                        elif action == "brightness":
                            device_name = match.group(1)
                            brightness = int(match.group(2))
                            devices = await self.find_devices(device_name)
                            
                            if devices:
                                results = []
                                for device in devices:
                                    if device.device_type == DeviceType.LIGHT:
                                        success = await self.control_device(device.entity_id, "brightness", brightness=brightness)
                                        results.append({
                                            "device": device.name,
                                            "action": f"set brightness to {brightness}",
                                            "success": success
                                        })
                                
                                return {
                                    "success": True,
                                    "command": command,
                                    "action": action,
                                    "results": results
                                }
                        
                        elif action == "temperature":
                            device_name = match.group(1)
                            temperature = int(match.group(2))
                            devices = await self.find_devices(device_name)
                            
                            if devices:
                                results = []
                                for device in devices:
                                    if device.device_type == DeviceType.CLIMATE:
                                        success = await self.control_device(device.entity_id, "set_temperature", temperature=temperature)
                                        results.append({
                                            "device": device.name,
                                            "action": f"set temperature to {temperature}",
                                            "success": success
                                        })
                                
                                return {
                                    "success": True,
                                    "command": command,
                                    "action": action,
                                    "results": results
                                }
            
            return {
                "success": False,
                "command": command,
                "error": "Command not recognized"
            }
            
        except Exception as e:
            logger.error(f"Error parsing command: {e}")
            return {
                "success": False,
                "command": command,
                "error": str(e)
            }
    
    async def create_scene(self, name: str, entities: Dict[str, Dict[str, Any]]) -> bool:
        """Create a new scene"""
        try:
            scene_data = {
                "name": name,
                "entities": entities
            }
            
            success = await self.ha_client.call_service(
                "scene", "create",
                scene_data
            )
            
            if success:
                self.scenes[f"scene.{name.lower().replace(' ', '_')}"] = {
                    "name": name,
                    "entities": entities,
                    "created": datetime.now()
                }
                logger.info(f"Created scene: {name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error creating scene: {e}")
            return False
    
    async def activate_scene(self, scene_name: str) -> bool:
        """Activate a scene"""
        try:
            # Find scene by name
            scene_entity = None
            for entity_id, scene_info in self.scenes.items():
                if scene_info["name"].lower() == scene_name.lower():
                    scene_entity = entity_id
                    break
            
            if not scene_entity:
                # Try direct entity ID
                if scene_name.startswith("scene."):
                    scene_entity = scene_name
                else:
                    scene_entity = f"scene.{scene_name.lower().replace(' ', '_')}"
            
            success = await self.ha_client.turn_on(scene_entity)
            
            if success:
                self.stats["scenes_activated"] += 1
                logger.info(f"Activated scene: {scene_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error activating scene: {e}")
            return False
    
    async def get_device_status(self, entity_id: str = None) -> Dict[str, Any]:
        """Get device status summary"""
        try:
            if entity_id:
                device = await self.get_device(entity_id)
                if device:
                    return {
                        "entity_id": device.entity_id,
                        "name": device.name,
                        "type": device.device_type.value,
                        "state": device.state.value,
                        "available": device.available,
                        "attributes": device.attributes,
                        "last_updated": device.last_updated.isoformat()
                    }
                else:
                    return {"error": f"Device not found: {entity_id}"}
            else:
                # Summary of all devices
                total_devices = len(self.devices)
                available_devices = len([d for d in self.devices.values() if d.available])
                device_types = {}
                
                for device in self.devices.values():
                    device_type = device.device_type.value
                    if device_type not in device_types:
                        device_types[device_type] = {"total": 0, "available": 0}
                    
                    device_types[device_type]["total"] += 1
                    if device.available:
                        device_types[device_type]["available"] += 1
                
                return {
                    "total_devices": total_devices,
                    "available_devices": available_devices,
                    "device_types": device_types,
                    "areas": list(self.areas.values()),
                    "scenes": len(self.scenes),
                    "last_refresh": self.stats["last_refresh"].isoformat() if self.stats["last_refresh"] else None
                }
                
        except Exception as e:
            logger.error(f"Error getting device status: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get device controller statistics"""
        return {
            **self.stats,
            "total_devices": len(self.devices),
            "available_devices": len([d for d in self.devices.values() if d.available]),
            "total_areas": len(self.areas),
            "total_scenes": len(self.scenes)
        }

# Factory function
async def create_device_controller(ha_client: HomeAssistantClient, n8n_client: N8nClient = None) -> DeviceController:
    """Create and initialize device controller"""
    controller = DeviceController(ha_client, n8n_client)
    await controller.initialize()
    return controller