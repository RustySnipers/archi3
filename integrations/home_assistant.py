"""
Home Assistant Integration
Comprehensive integration with Home Assistant for device control and automation
"""

import os
import json
import logging
import asyncio
import aiohttp
import websockets
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

@dataclass
class HAEntity:
    """Home Assistant entity representation"""
    entity_id: str
    state: str
    attributes: Dict[str, Any]
    last_changed: datetime
    last_updated: datetime
    friendly_name: str
    domain: str
    object_id: str

@dataclass
class HAService:
    """Home Assistant service representation"""
    domain: str
    service: str
    description: str
    fields: Dict[str, Any]

@dataclass
class HAArea:
    """Home Assistant area representation"""
    area_id: str
    name: str
    aliases: List[str]
    picture: Optional[str]

@dataclass
class HADevice:
    """Home Assistant device representation"""
    device_id: str
    name: str
    manufacturer: str
    model: str
    area_id: Optional[str]
    entities: List[str]

class HomeAssistantClient:
    """Home Assistant API client with full integration capabilities"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8123",
                 access_token: str = None,
                 websocket_url: str = None):
        
        self.base_url = base_url.rstrip('/')
        self.access_token = access_token
        self.websocket_url = websocket_url or f"ws://{base_url.split('://', 1)[1]}/api/websocket"
        
        # HTTP session
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # WebSocket connection
        self.websocket = None
        self.websocket_connected = False
        self.websocket_id = 1
        self.pending_requests = {}
        self.event_listeners = {}
        
        # Entity and service cache
        self.entities = {}
        self.services = {}
        self.areas = {}
        self.devices = {}
        self.config = {}
        
        # Connection state
        self.connected = False
        self.last_update = None
        
        # Statistics
        self.stats = {
            "api_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "events_received": 0,
            "entities_count": 0,
            "last_sync": None
        }
    
    async def initialize(self):
        """Initialize Home Assistant connection"""
        try:
            logger.info("Initializing Home Assistant client...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=self.headers
            )
            
            # Test API connection
            await self._test_connection()
            
            # Load initial data
            await self._load_initial_data()
            
            # Start WebSocket connection
            await self._connect_websocket()
            
            self.connected = True
            logger.info("Home Assistant client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Home Assistant client: {e}")
            raise
    
    async def _test_connection(self):
        """Test API connection"""
        try:
            async with self.session.get(f"{self.base_url}/api/") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Connected to Home Assistant: {data.get('message', 'API Ready')}")
                    self.stats["successful_calls"] += 1
                elif response.status == 401:
                    raise Exception("Authentication failed - check access token")
                else:
                    raise Exception(f"API test failed: HTTP {response.status}")
                    
            self.stats["api_calls"] += 1
            
        except Exception as e:
            self.stats["failed_calls"] += 1
            logger.error(f"Home Assistant connection test failed: {e}")
            raise
    
    async def _load_initial_data(self):
        """Load initial configuration and entity data"""
        try:
            # Load configuration
            self.config = await self._api_get("config")
            
            # Load all entities
            await self._refresh_entities()
            
            # Load services
            await self._refresh_services()
            
            # Load areas
            await self._refresh_areas()
            
            # Load devices
            await self._refresh_devices()
            
            self.stats["last_sync"] = datetime.now()
            logger.info(f"Loaded {len(self.entities)} entities, {len(self.services)} service domains")
            
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
            raise
    
    async def _refresh_entities(self):
        """Refresh entity states"""
        try:
            states = await self._api_get("states")
            self.entities.clear()
            
            for state_data in states:
                entity = HAEntity(
                    entity_id=state_data["entity_id"],
                    state=state_data["state"],
                    attributes=state_data.get("attributes", {}),
                    last_changed=datetime.fromisoformat(state_data["last_changed"].replace('Z', '+00:00')),
                    last_updated=datetime.fromisoformat(state_data["last_updated"].replace('Z', '+00:00')),
                    friendly_name=state_data.get("attributes", {}).get("friendly_name", state_data["entity_id"]),
                    domain=state_data["entity_id"].split('.')[0],
                    object_id=state_data["entity_id"].split('.')[1]
                )
                self.entities[entity.entity_id] = entity
            
            self.stats["entities_count"] = len(self.entities)
            
        except Exception as e:
            logger.error(f"Error refreshing entities: {e}")
            raise
    
    async def _refresh_services(self):
        """Refresh available services"""
        try:
            services = await self._api_get("services")
            self.services.clear()
            
            for domain, domain_services in services.items():
                for service_name, service_data in domain_services.items():
                    service = HAService(
                        domain=domain,
                        service=service_name,
                        description=service_data.get("description", ""),
                        fields=service_data.get("fields", {})
                    )
                    service_key = f"{domain}.{service_name}"
                    self.services[service_key] = service
                    
        except Exception as e:
            logger.error(f"Error refreshing services: {e}")
    
    async def _refresh_areas(self):
        """Refresh area registry"""
        try:
            areas = await self._api_get("config/area_registry")
            self.areas.clear()
            
            for area_data in areas:
                area = HAArea(
                    area_id=area_data["area_id"],
                    name=area_data["name"],
                    aliases=area_data.get("aliases", []),
                    picture=area_data.get("picture")
                )
                self.areas[area.area_id] = area
                
        except Exception as e:
            logger.warning(f"Error refreshing areas: {e}")
    
    async def _refresh_devices(self):
        """Refresh device registry"""
        try:
            devices = await self._api_get("config/device_registry")
            self.devices.clear()
            
            for device_data in devices:
                device = HADevice(
                    device_id=device_data["id"],
                    name=device_data.get("name", ""),
                    manufacturer=device_data.get("manufacturer", ""),
                    model=device_data.get("model", ""),
                    area_id=device_data.get("area_id"),
                    entities=[]  # Would need entity registry to populate
                )
                self.devices[device.device_id] = device
                
        except Exception as e:
            logger.warning(f"Error refreshing devices: {e}")
    
    async def _connect_websocket(self):
        """Connect to Home Assistant WebSocket API"""
        try:
            logger.info("Connecting to Home Assistant WebSocket...")
            
            self.websocket = await websockets.connect(self.websocket_url)
            
            # Authenticate
            auth_message = await self.websocket.recv()
            auth_data = json.loads(auth_message)
            
            if auth_data["type"] == "auth_required":
                await self.websocket.send(json.dumps({
                    "type": "auth",
                    "access_token": self.access_token
                }))
                
                auth_response = await self.websocket.recv()
                auth_result = json.loads(auth_response)
                
                if auth_result["type"] == "auth_ok":
                    self.websocket_connected = True
                    logger.info("WebSocket authentication successful")
                    
                    # Start message handling
                    asyncio.create_task(self._handle_websocket_messages())
                    
                    # Subscribe to state changes
                    await self._subscribe_to_events()
                else:
                    raise Exception(f"WebSocket authentication failed: {auth_result}")
            else:
                raise Exception(f"Unexpected WebSocket message: {auth_data}")
                
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.websocket_connected = False
    
    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in WebSocket message: {message}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.websocket_connected = False
        except Exception as e:
            logger.error(f"WebSocket message handling error: {e}")
            self.websocket_connected = False
    
    async def _process_websocket_message(self, data: Dict[str, Any]):
        """Process individual WebSocket message"""
        try:
            message_id = data.get("id")
            message_type = data.get("type")
            
            # Handle responses to our requests
            if message_id and message_id in self.pending_requests:
                future = self.pending_requests.pop(message_id)
                if data.get("success", True):
                    future.set_result(data.get("result"))
                else:
                    future.set_exception(Exception(f"Request failed: {data.get('error')}"))
                return
            
            # Handle events
            if message_type == "event":
                await self._handle_event(data.get("event", {}))
                self.stats["events_received"] += 1
            
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Handle Home Assistant events"""
        try:
            event_type = event.get("event_type")
            event_data = event.get("data", {})
            
            if event_type == "state_changed":
                # Update entity state
                entity_id = event_data.get("entity_id")
                new_state = event_data.get("new_state")
                
                if entity_id and new_state:
                    await self._update_entity_state(entity_id, new_state)
            
            # Notify event listeners
            for listener in self.event_listeners.get(event_type, []):
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(event)
                    else:
                        listener(event)
                except Exception as e:
                    logger.error(f"Error in event listener: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling event: {e}")
    
    async def _update_entity_state(self, entity_id: str, state_data: Dict[str, Any]):
        """Update entity state from WebSocket event"""
        try:
            if entity_id in self.entities:
                entity = self.entities[entity_id]
                entity.state = state_data["state"]
                entity.attributes = state_data.get("attributes", {})
                entity.last_changed = datetime.fromisoformat(state_data["last_changed"].replace('Z', '+00:00'))
                entity.last_updated = datetime.fromisoformat(state_data["last_updated"].replace('Z', '+00:00'))
                entity.friendly_name = entity.attributes.get("friendly_name", entity_id)
                
        except Exception as e:
            logger.error(f"Error updating entity state: {e}")
    
    async def _subscribe_to_events(self):
        """Subscribe to Home Assistant events"""
        try:
            await self._websocket_request({
                "type": "subscribe_events",
                "event_type": "state_changed"
            })
            
            logger.info("Subscribed to state change events")
            
        except Exception as e:
            logger.error(f"Error subscribing to events: {e}")
    
    async def _websocket_request(self, request: Dict[str, Any]) -> Any:
        """Send WebSocket request and wait for response"""
        if not self.websocket_connected:
            raise Exception("WebSocket not connected")
        
        request_id = self.websocket_id
        self.websocket_id += 1
        
        request["id"] = request_id
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        try:
            await self.websocket.send(json.dumps(request))
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise Exception("WebSocket request timeout")
        except Exception as e:
            self.pending_requests.pop(request_id, None)
            raise
    
    async def _api_get(self, endpoint: str) -> Any:
        """Make GET request to Home Assistant API"""
        try:
            self.stats["api_calls"] += 1
            
            async with self.session.get(f"{self.base_url}/api/{endpoint}") as response:
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
        """Make POST request to Home Assistant API"""
        try:
            self.stats["api_calls"] += 1
            
            async with self.session.post(
                f"{self.base_url}/api/{endpoint}",
                json=data or {}
            ) as response:
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
    
    # Public API methods
    
    async def get_entity_state(self, entity_id: str) -> Optional[HAEntity]:
        """Get current state of an entity"""
        try:
            # Try cache first
            if entity_id in self.entities:
                return self.entities[entity_id]
            
            # Fetch from API
            state_data = await self._api_get(f"states/{entity_id}")
            
            entity = HAEntity(
                entity_id=state_data["entity_id"],
                state=state_data["state"],
                attributes=state_data.get("attributes", {}),
                last_changed=datetime.fromisoformat(state_data["last_changed"].replace('Z', '+00:00')),
                last_updated=datetime.fromisoformat(state_data["last_updated"].replace('Z', '+00:00')),
                friendly_name=state_data.get("attributes", {}).get("friendly_name", entity_id),
                domain=entity_id.split('.')[0],
                object_id=entity_id.split('.')[1]
            )
            
            # Update cache
            self.entities[entity_id] = entity
            
            return entity
            
        except Exception as e:
            logger.error(f"Error getting entity state for {entity_id}: {e}")
            return None
    
    async def call_service(self, 
                          domain: str, 
                          service: str, 
                          service_data: Dict[str, Any] = None,
                          target: Dict[str, Any] = None) -> bool:
        """Call a Home Assistant service"""
        try:
            endpoint = f"services/{domain}/{service}"
            
            # Prepare request data
            request_data = {}
            if service_data:
                request_data.update(service_data)
            if target:
                request_data["target"] = target
            
            # Call service
            result = await self._api_post(endpoint, request_data)
            
            logger.info(f"Called service {domain}.{service} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error calling service {domain}.{service}: {e}")
            return False
    
    async def turn_on(self, entity_id: str, **kwargs) -> bool:
        """Turn on an entity"""
        try:
            domain = entity_id.split('.')[0]
            service_data = {"entity_id": entity_id}
            service_data.update(kwargs)
            
            return await self.call_service(domain, "turn_on", service_data)
            
        except Exception as e:
            logger.error(f"Error turning on {entity_id}: {e}")
            return False
    
    async def turn_off(self, entity_id: str, **kwargs) -> bool:
        """Turn off an entity"""
        try:
            domain = entity_id.split('.')[0]
            service_data = {"entity_id": entity_id}
            service_data.update(kwargs)
            
            return await self.call_service(domain, "turn_off", service_data)
            
        except Exception as e:
            logger.error(f"Error turning off {entity_id}: {e}")
            return False
    
    async def toggle(self, entity_id: str, **kwargs) -> bool:
        """Toggle an entity"""
        try:
            domain = entity_id.split('.')[0]
            service_data = {"entity_id": entity_id}
            service_data.update(kwargs)
            
            return await self.call_service(domain, "toggle", service_data)
            
        except Exception as e:
            logger.error(f"Error toggling {entity_id}: {e}")
            return False
    
    async def set_state(self, entity_id: str, state: str, attributes: Dict[str, Any] = None) -> bool:
        """Set entity state"""
        try:
            endpoint = f"states/{entity_id}"
            data = {
                "state": state,
                "attributes": attributes or {}
            }
            
            result = await self._api_post(endpoint, data)
            return True
            
        except Exception as e:
            logger.error(f"Error setting state for {entity_id}: {e}")
            return False
    
    async def find_entities_by_name(self, name: str, fuzzy: bool = True) -> List[HAEntity]:
        """Find entities by friendly name"""
        try:
            matches = []
            name_lower = name.lower()
            
            for entity in self.entities.values():
                friendly_name = entity.friendly_name.lower()
                
                if fuzzy:
                    # Fuzzy matching - check if name is contained in friendly name
                    if name_lower in friendly_name or any(word in friendly_name for word in name_lower.split()):
                        matches.append(entity)
                else:
                    # Exact matching
                    if name_lower == friendly_name:
                        matches.append(entity)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error finding entities by name '{name}': {e}")
            return []
    
    async def find_entities_by_domain(self, domain: str) -> List[HAEntity]:
        """Find entities by domain"""
        try:
            return [entity for entity in self.entities.values() if entity.domain == domain]
        except Exception as e:
            logger.error(f"Error finding entities by domain '{domain}': {e}")
            return []
    
    async def find_entities_by_area(self, area_name: str) -> List[HAEntity]:
        """Find entities by area name"""
        try:
            # This is simplified - would need device/entity registry for proper area filtering
            matches = []
            area_name_lower = area_name.lower()
            
            for entity in self.entities.values():
                # Check if area is mentioned in friendly name or attributes
                friendly_name = entity.friendly_name.lower()
                area_attr = entity.attributes.get('area', '').lower()
                
                if area_name_lower in friendly_name or area_name_lower in area_attr:
                    matches.append(entity)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error finding entities by area '{area_name}': {e}")
            return []
    
    async def get_all_entities(self, domain: str = None) -> List[HAEntity]:
        """Get all entities, optionally filtered by domain"""
        try:
            if domain:
                return await self.find_entities_by_domain(domain)
            else:
                return list(self.entities.values())
        except Exception as e:
            logger.error(f"Error getting all entities: {e}")
            return []
    
    async def get_areas(self) -> List[HAArea]:
        """Get all areas"""
        return list(self.areas.values())
    
    async def get_services(self, domain: str = None) -> List[HAService]:
        """Get all services, optionally filtered by domain"""
        try:
            if domain:
                return [service for service in self.services.values() if service.domain == domain]
            else:
                return list(self.services.values())
        except Exception as e:
            logger.error(f"Error getting services: {e}")
            return []
    
    def add_event_listener(self, event_type: str, callback: Callable):
        """Add event listener for specific event type"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        
        self.event_listeners[event_type].append(callback)
        logger.info(f"Added event listener for {event_type}")
    
    def remove_event_listener(self, event_type: str, callback: Callable):
        """Remove event listener"""
        if event_type in self.event_listeners:
            try:
                self.event_listeners[event_type].remove(callback)
                logger.info(f"Removed event listener for {event_type}")
            except ValueError:
                logger.warning(f"Event listener not found for {event_type}")
    
    async def refresh_all_data(self):
        """Refresh all cached data"""
        try:
            await self._refresh_entities()
            await self._refresh_services()
            await self._refresh_areas()
            await self._refresh_devices()
            
            self.last_update = datetime.now()
            self.stats["last_sync"] = self.last_update
            
            logger.info("All Home Assistant data refreshed")
            
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Home Assistant client statistics"""
        return {
            **self.stats,
            "connected": self.connected,
            "websocket_connected": self.websocket_connected,
            "entities_cached": len(self.entities),
            "services_cached": len(self.services),
            "areas_cached": len(self.areas),
            "devices_cached": len(self.devices),
            "last_update": self.last_update.isoformat() if self.last_update else None
        }
    
    async def cleanup(self):
        """Clean up Home Assistant client"""
        try:
            logger.info("Cleaning up Home Assistant client...")
            
            # Close WebSocket
            if self.websocket:
                await self.websocket.close()
                self.websocket_connected = False
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            # Clear data
            self.entities.clear()
            self.services.clear()
            self.areas.clear()
            self.devices.clear()
            self.event_listeners.clear()
            self.pending_requests.clear()
            
            self.connected = False
            
            logger.info("Home Assistant client cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Home Assistant cleanup: {e}")

# Factory function for creating Home Assistant client
async def create_home_assistant_client(config: Dict[str, Any]) -> HomeAssistantClient:
    """Create and initialize Home Assistant client"""
    ha_config = config.get('homeassistant', {})
    
    base_url = ha_config.get('url', 'http://localhost:8123')
    access_token = ha_config.get('token')
    
    if not access_token:
        raise ValueError("Home Assistant access token is required")
    
    client = HomeAssistantClient(
        base_url=base_url,
        access_token=access_token
    )
    
    await client.initialize()
    return client