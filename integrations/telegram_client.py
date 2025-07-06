"""
Telegram Integration
Comprehensive Telegram bot integration for notifications and commands
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
from urllib.parse import quote
import os

logger = logging.getLogger(__name__)

@dataclass
class TelegramUser:
    """Telegram user representation"""
    id: int
    is_bot: bool
    first_name: str
    last_name: Optional[str]
    username: Optional[str]
    language_code: Optional[str]

@dataclass
class TelegramChat:
    """Telegram chat representation"""
    id: int
    type: str
    title: Optional[str]
    username: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]

@dataclass
class TelegramMessage:
    """Telegram message representation"""
    message_id: int
    user: TelegramUser
    chat: TelegramChat
    date: datetime
    text: Optional[str]
    caption: Optional[str]
    photo: Optional[List[Dict[str, Any]]]
    document: Optional[Dict[str, Any]]
    voice: Optional[Dict[str, Any]]
    video: Optional[Dict[str, Any]]
    location: Optional[Dict[str, Any]]

@dataclass
class TelegramCallback:
    """Telegram callback query representation"""
    id: str
    user: TelegramUser
    message: Optional[TelegramMessage]
    data: Optional[str]

class TelegramClient:
    """Telegram bot client with comprehensive features"""
    
    def __init__(self, bot_token: str, allowed_chat_ids: List[int] = None):
        self.bot_token = bot_token
        self.allowed_chat_ids = set(allowed_chat_ids or [])
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # HTTP session
        self.session = None
        
        # Bot information
        self.bot_info = None
        self.webhook_url = None
        
        # Message handlers
        self.message_handlers = []
        self.command_handlers = {}
        self.callback_handlers = {}
        
        # Polling state
        self.polling = False
        self.last_update_id = 0
        self.poll_task = None
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "commands_processed": 0,
            "errors": 0,
            "last_activity": None
        }
        
        # Message formatting
        self.parse_mode = "HTML"
        
        # File handling
        self.max_file_size = 50 * 1024 * 1024  # 50MB
    
    async def initialize(self):
        """Initialize Telegram client"""
        try:
            logger.info("Initializing Telegram client...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Get bot information
            await self._get_bot_info()
            
            # Set up webhook or start polling
            webhook_url = os.getenv('TELEGRAM_WEBHOOK_URL')
            if webhook_url:
                await self.set_webhook(webhook_url)
            else:
                await self.start_polling()
            
            logger.info(f"Telegram client initialized for bot: {self.bot_info['first_name']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram client: {e}")
            raise
    
    async def _get_bot_info(self):
        """Get bot information"""
        try:
            response = await self._api_request("getMe")
            self.bot_info = response["result"]
            
        except Exception as e:
            logger.error(f"Error getting bot info: {e}")
            raise
    
    async def _api_request(self, method: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make API request to Telegram"""
        try:
            url = f"{self.base_url}/{method}"
            
            if data:
                # Handle file uploads
                if any(key in data for key in ['photo', 'document', 'voice', 'video', 'audio']):
                    # Use multipart form data for file uploads
                    form_data = aiohttp.FormData()
                    for key, value in data.items():
                        if hasattr(value, 'read'):  # File-like object
                            form_data.add_field(key, value)
                        else:
                            form_data.add_field(key, str(value))
                    
                    async with self.session.post(url, data=form_data) as response:
                        result = await response.json()
                else:
                    async with self.session.post(url, json=data) as response:
                        result = await response.json()
            else:
                async with self.session.get(url) as response:
                    result = await response.json()
            
            if not result.get("ok"):
                raise Exception(f"Telegram API error: {result.get('description', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Telegram API request failed: {e}")
            raise
    
    async def send_message(self, 
                          chat_id: int, 
                          text: str,
                          parse_mode: str = None,
                          reply_markup: Dict[str, Any] = None,
                          disable_notification: bool = False) -> Optional[TelegramMessage]:
        """Send text message"""
        try:
            if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
                logger.warning(f"Chat ID {chat_id} not in allowed list")
                return None
            
            data = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode or self.parse_mode,
                "disable_notification": disable_notification
            }
            
            if reply_markup:
                data["reply_markup"] = json.dumps(reply_markup)
            
            response = await self._api_request("sendMessage", data)
            message_data = response["result"]
            
            self.stats["messages_sent"] += 1
            self.stats["last_activity"] = datetime.now()
            
            return self._parse_message(message_data)
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return None
    
    async def send_photo(self,
                        chat_id: int,
                        photo: Any,
                        caption: str = None,
                        parse_mode: str = None) -> Optional[TelegramMessage]:
        """Send photo"""
        try:
            if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
                logger.warning(f"Chat ID {chat_id} not in allowed list")
                return None
            
            data = {
                "chat_id": chat_id,
                "photo": photo
            }
            
            if caption:
                data["caption"] = caption
                data["parse_mode"] = parse_mode or self.parse_mode
            
            response = await self._api_request("sendPhoto", data)
            message_data = response["result"]
            
            self.stats["messages_sent"] += 1
            self.stats["last_activity"] = datetime.now()
            
            return self._parse_message(message_data)
            
        except Exception as e:
            logger.error(f"Error sending photo: {e}")
            return None
    
    async def send_document(self,
                           chat_id: int,
                           document: Any,
                           caption: str = None,
                           parse_mode: str = None) -> Optional[TelegramMessage]:
        """Send document"""
        try:
            if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
                logger.warning(f"Chat ID {chat_id} not in allowed list")
                return None
            
            data = {
                "chat_id": chat_id,
                "document": document
            }
            
            if caption:
                data["caption"] = caption
                data["parse_mode"] = parse_mode or self.parse_mode
            
            response = await self._api_request("sendDocument", data)
            message_data = response["result"]
            
            self.stats["messages_sent"] += 1
            self.stats["last_activity"] = datetime.now()
            
            return self._parse_message(message_data)
            
        except Exception as e:
            logger.error(f"Error sending document: {e}")
            return None
    
    async def send_voice(self,
                        chat_id: int,
                        voice: Any,
                        caption: str = None,
                        duration: int = None) -> Optional[TelegramMessage]:
        """Send voice message"""
        try:
            if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
                logger.warning(f"Chat ID {chat_id} not in allowed list")
                return None
            
            data = {
                "chat_id": chat_id,
                "voice": voice
            }
            
            if caption:
                data["caption"] = caption
            
            if duration:
                data["duration"] = duration
            
            response = await self._api_request("sendVoice", data)
            message_data = response["result"]
            
            self.stats["messages_sent"] += 1
            self.stats["last_activity"] = datetime.now()
            
            return self._parse_message(message_data)
            
        except Exception as e:
            logger.error(f"Error sending voice: {e}")
            return None
    
    async def send_location(self,
                           chat_id: int,
                           latitude: float,
                           longitude: float,
                           live_period: int = None) -> Optional[TelegramMessage]:
        """Send location"""
        try:
            if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
                logger.warning(f"Chat ID {chat_id} not in allowed list")
                return None
            
            data = {
                "chat_id": chat_id,
                "latitude": latitude,
                "longitude": longitude
            }
            
            if live_period:
                data["live_period"] = live_period
            
            response = await self._api_request("sendLocation", data)
            message_data = response["result"]
            
            self.stats["messages_sent"] += 1
            self.stats["last_activity"] = datetime.now()
            
            return self._parse_message(message_data)
            
        except Exception as e:
            logger.error(f"Error sending location: {e}")
            return None
    
    async def edit_message(self,
                          chat_id: int,
                          message_id: int,
                          text: str,
                          parse_mode: str = None,
                          reply_markup: Dict[str, Any] = None) -> bool:
        """Edit message text"""
        try:
            data = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": parse_mode or self.parse_mode
            }
            
            if reply_markup:
                data["reply_markup"] = json.dumps(reply_markup)
            
            await self._api_request("editMessageText", data)
            return True
            
        except Exception as e:
            logger.error(f"Error editing message: {e}")
            return False
    
    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        """Delete message"""
        try:
            data = {
                "chat_id": chat_id,
                "message_id": message_id
            }
            
            await self._api_request("deleteMessage", data)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting message: {e}")
            return False
    
    async def answer_callback_query(self,
                                   callback_query_id: str,
                                   text: str = None,
                                   show_alert: bool = False) -> bool:
        """Answer callback query"""
        try:
            data = {
                "callback_query_id": callback_query_id
            }
            
            if text:
                data["text"] = text
                data["show_alert"] = show_alert
            
            await self._api_request("answerCallbackQuery", data)
            return True
            
        except Exception as e:
            logger.error(f"Error answering callback query: {e}")
            return False
    
    def create_inline_keyboard(self, buttons: List[List[Dict[str, str]]]) -> Dict[str, Any]:
        """Create inline keyboard markup"""
        return {
            "inline_keyboard": buttons
        }
    
    def create_reply_keyboard(self, 
                             buttons: List[List[str]], 
                             resize_keyboard: bool = True,
                             one_time_keyboard: bool = False) -> Dict[str, Any]:
        """Create reply keyboard markup"""
        keyboard = []
        for row in buttons:
            keyboard_row = []
            for button in row:
                keyboard_row.append({"text": button})
            keyboard.append(keyboard_row)
        
        return {
            "keyboard": keyboard,
            "resize_keyboard": resize_keyboard,
            "one_time_keyboard": one_time_keyboard
        }
    
    async def set_webhook(self, url: str, certificate: str = None) -> bool:
        """Set webhook for receiving updates"""
        try:
            data = {"url": url}
            
            if certificate:
                data["certificate"] = certificate
            
            await self._api_request("setWebhook", data)
            self.webhook_url = url
            
            logger.info(f"Webhook set to: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting webhook: {e}")
            return False
    
    async def delete_webhook(self) -> bool:
        """Delete webhook"""
        try:
            await self._api_request("deleteWebhook")
            self.webhook_url = None
            
            logger.info("Webhook deleted")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting webhook: {e}")
            return False
    
    async def start_polling(self, timeout: int = 30):
        """Start long polling for updates"""
        try:
            if self.polling:
                return
            
            self.polling = True
            self.poll_task = asyncio.create_task(self._poll_updates(timeout))
            
            logger.info("Started polling for updates")
            
        except Exception as e:
            logger.error(f"Error starting polling: {e}")
            self.polling = False
    
    async def stop_polling(self):
        """Stop polling for updates"""
        self.polling = False
        
        if self.poll_task:
            self.poll_task.cancel()
            try:
                await self.poll_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped polling for updates")
    
    async def _poll_updates(self, timeout: int):
        """Poll for updates"""
        while self.polling:
            try:
                data = {
                    "offset": self.last_update_id + 1,
                    "timeout": timeout,
                    "limit": 100
                }
                
                response = await self._api_request("getUpdates", data)
                updates = response["result"]
                
                for update in updates:
                    await self._process_update(update)
                    self.last_update_id = update["update_id"]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error polling updates: {e}")
                await asyncio.sleep(5)
    
    async def process_webhook_update(self, update_data: Dict[str, Any]):
        """Process webhook update"""
        try:
            await self._process_update(update_data)
        except Exception as e:
            logger.error(f"Error processing webhook update: {e}")
    
    async def _process_update(self, update: Dict[str, Any]):
        """Process individual update"""
        try:
            if "message" in update:
                message = self._parse_message(update["message"])
                await self._handle_message(message)
                
            elif "callback_query" in update:
                callback = self._parse_callback(update["callback_query"])
                await self._handle_callback(callback)
                
        except Exception as e:
            logger.error(f"Error processing update: {e}")
    
    def _parse_message(self, message_data: Dict[str, Any]) -> TelegramMessage:
        """Parse message data"""
        user_data = message_data["from"]
        chat_data = message_data["chat"]
        
        user = TelegramUser(
            id=user_data["id"],
            is_bot=user_data.get("is_bot", False),
            first_name=user_data["first_name"],
            last_name=user_data.get("last_name"),
            username=user_data.get("username"),
            language_code=user_data.get("language_code")
        )
        
        chat = TelegramChat(
            id=chat_data["id"],
            type=chat_data["type"],
            title=chat_data.get("title"),
            username=chat_data.get("username"),
            first_name=chat_data.get("first_name"),
            last_name=chat_data.get("last_name")
        )
        
        return TelegramMessage(
            message_id=message_data["message_id"],
            user=user,
            chat=chat,
            date=datetime.fromtimestamp(message_data["date"]),
            text=message_data.get("text"),
            caption=message_data.get("caption"),
            photo=message_data.get("photo"),
            document=message_data.get("document"),
            voice=message_data.get("voice"),
            video=message_data.get("video"),
            location=message_data.get("location")
        )
    
    def _parse_callback(self, callback_data: Dict[str, Any]) -> TelegramCallback:
        """Parse callback query data"""
        user_data = callback_data["from"]
        
        user = TelegramUser(
            id=user_data["id"],
            is_bot=user_data.get("is_bot", False),
            first_name=user_data["first_name"],
            last_name=user_data.get("last_name"),
            username=user_data.get("username"),
            language_code=user_data.get("language_code")
        )
        
        message = None
        if "message" in callback_data:
            message = self._parse_message(callback_data["message"])
        
        return TelegramCallback(
            id=callback_data["id"],
            user=user,
            message=message,
            data=callback_data.get("data")
        )
    
    async def _handle_message(self, message: TelegramMessage):
        """Handle incoming message"""
        try:
            self.stats["messages_received"] += 1
            self.stats["last_activity"] = datetime.now()
            
            # Check if chat is allowed
            if self.allowed_chat_ids and message.chat.id not in self.allowed_chat_ids:
                logger.warning(f"Message from unauthorized chat: {message.chat.id}")
                return
            
            # Handle commands
            if message.text and message.text.startswith('/'):
                command = message.text.split()[0][1:]  # Remove '/'
                await self._handle_command(command, message)
                return
            
            # Handle regular messages
            for handler in self.message_handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_command(self, command: str, message: TelegramMessage):
        """Handle command"""
        try:
            self.stats["commands_processed"] += 1
            
            if command in self.command_handlers:
                handler = self.command_handlers[command]
                await handler(message)
            else:
                # Default help response
                await self.send_message(
                    message.chat.id,
                    f"Unknown command: /{command}\nType /help for available commands."
                )
                
        except Exception as e:
            logger.error(f"Error handling command {command}: {e}")
    
    async def _handle_callback(self, callback: TelegramCallback):
        """Handle callback query"""
        try:
            if callback.data and callback.data in self.callback_handlers:
                handler = self.callback_handlers[callback.data]
                await handler(callback)
            else:
                await self.answer_callback_query(callback.id, "Unknown action")
                
        except Exception as e:
            logger.error(f"Error handling callback: {e}")
    
    def add_message_handler(self, handler: Callable[[TelegramMessage], None]):
        """Add message handler"""
        self.message_handlers.append(handler)
    
    def add_command_handler(self, command: str, handler: Callable[[TelegramMessage], None]):
        """Add command handler"""
        self.command_handlers[command] = handler
    
    def add_callback_handler(self, callback_data: str, handler: Callable[[TelegramCallback], None]):
        """Add callback handler"""
        self.callback_handlers[callback_data] = handler
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Telegram client statistics"""
        return {
            **self.stats,
            "bot_info": self.bot_info,
            "webhook_url": self.webhook_url,
            "polling": self.polling,
            "allowed_chats": len(self.allowed_chat_ids),
            "message_handlers": len(self.message_handlers),
            "command_handlers": len(self.command_handlers),
            "callback_handlers": len(self.callback_handlers)
        }
    
    async def cleanup(self):
        """Clean up Telegram client"""
        try:
            logger.info("Cleaning up Telegram client...")
            
            # Stop polling
            await self.stop_polling()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
            
            logger.info("Telegram client cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Telegram cleanup: {e}")

# Factory function
async def create_telegram_client(config: Dict[str, Any]) -> TelegramClient:
    """Create and initialize Telegram client"""
    telegram_config = config.get('telegram', {})
    
    bot_token = telegram_config.get('bot_token')
    if not bot_token:
        raise ValueError("Telegram bot token is required")
    
    allowed_chat_ids = telegram_config.get('allowed_chat_ids', [])
    
    client = TelegramClient(bot_token, allowed_chat_ids)
    await client.initialize()
    
    return client