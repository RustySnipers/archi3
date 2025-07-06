"""
Notification Management System
Unified notification system with multiple channels and email-to-SMS backup
"""

import logging
import asyncio
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import re
import json

from .telegram_client import TelegramClient

logger = logging.getLogger(__name__)

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    """Notification channels"""
    TELEGRAM = "telegram"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    VOICE = "voice"

@dataclass
class NotificationTemplate:
    """Notification template"""
    name: str
    title: str
    body: str
    priority: NotificationPriority
    channels: List[NotificationChannel]
    variables: List[str]

@dataclass
class Notification:
    """Notification object"""
    id: str
    title: str
    message: str
    priority: NotificationPriority
    channel: NotificationChannel
    recipient: str
    timestamp: datetime
    delivered: bool
    delivery_time: Optional[datetime]
    attempts: int
    error: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class SMSProvider:
    """SMS provider configuration"""
    name: str
    email_domain: str
    supports_mms: bool
    character_limit: int

class NotificationManager:
    """Unified notification management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Notification channels
        self.telegram_client = None
        self.email_config = config.get('email', {})
        self.sms_config = config.get('sms', {})
        
        # Notification queue and history
        self.notification_queue = asyncio.Queue()
        self.notification_history = {}
        self.failed_notifications = {}
        
        # Templates
        self.templates = {}
        self._load_default_templates()
        
        # SMS providers mapping
        self.sms_providers = {
            "verizon": SMSProvider("Verizon", "vtext.com", True, 160),
            "att": SMSProvider("AT&T", "txt.att.net", True, 160),
            "tmobile": SMSProvider("T-Mobile", "tmomail.net", True, 160),
            "sprint": SMSProvider("Sprint", "messaging.sprintpcs.com", True, 160),
            "boost": SMSProvider("Boost", "myboostmobile.com", False, 160),
            "cricket": SMSProvider("Cricket", "sms.cricketwireless.net", False, 160),
            "metropcs": SMSProvider("MetroPCS", "mymetropcs.com", False, 160),
            "virgin": SMSProvider("Virgin", "vmobl.com", False, 160),
            "tracfone": SMSProvider("TracFone", "mmst5.tracfone.com", False, 160)
        }
        
        # Processing state
        self.running = False
        self.processor_task = None
        
        # Statistics
        self.stats = {
            "total_notifications": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "telegram_sent": 0,
            "emails_sent": 0,
            "sms_sent": 0,
            "last_activity": None
        }
    
    async def initialize(self):
        """Initialize notification manager"""
        try:
            logger.info("Initializing notification manager...")
            
            # Initialize Telegram client if configured
            if self.config.get('telegram', {}).get('enabled', False):
                from .telegram_client import create_telegram_client
                self.telegram_client = await create_telegram_client(self.config)
                
                # Add default handlers
                self._setup_telegram_handlers()
            
            # Start notification processor
            self.running = True
            self.processor_task = asyncio.create_task(self._process_notifications())
            
            logger.info("Notification manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize notification manager: {e}")
            raise
    
    def _load_default_templates(self):
        """Load default notification templates"""
        self.templates = {
            "alert": NotificationTemplate(
                name="alert",
                title="⚠️ Alert: {title}",
                body="Alert from Archie:\n\n{message}\n\nTime: {timestamp}",
                priority=NotificationPriority.HIGH,
                channels=[NotificationChannel.TELEGRAM, NotificationChannel.EMAIL],
                variables=["title", "message", "timestamp"]
            ),
            "info": NotificationTemplate(
                name="info",
                title="ℹ️ Info: {title}",
                body="Information from Archie:\n\n{message}\n\nTime: {timestamp}",
                priority=NotificationPriority.NORMAL,
                channels=[NotificationChannel.TELEGRAM],
                variables=["title", "message", "timestamp"]
            ),
            "critical": NotificationTemplate(
                name="critical",
                title="🚨 CRITICAL: {title}",
                body="CRITICAL ALERT from Archie:\n\n{message}\n\nImmediate attention required!\n\nTime: {timestamp}",
                priority=NotificationPriority.CRITICAL,
                channels=[NotificationChannel.TELEGRAM, NotificationChannel.EMAIL, NotificationChannel.SMS],
                variables=["title", "message", "timestamp"]
            ),
            "reminder": NotificationTemplate(
                name="reminder",
                title="⏰ Reminder: {title}",
                body="Reminder from Archie:\n\n{message}\n\nTime: {timestamp}",
                priority=NotificationPriority.LOW,
                channels=[NotificationChannel.TELEGRAM],
                variables=["title", "message", "timestamp"]
            ),
            "system_status": NotificationTemplate(
                name="system_status",
                title="📊 System Status: {title}",
                body="System status update from Archie:\n\n{message}\n\nTime: {timestamp}",
                priority=NotificationPriority.NORMAL,
                channels=[NotificationChannel.TELEGRAM, NotificationChannel.EMAIL],
                variables=["title", "message", "timestamp"]
            )
        }
    
    def _setup_telegram_handlers(self):
        """Setup Telegram command handlers"""
        if not self.telegram_client:
            return
        
        async def help_handler(message):
            help_text = """
🤖 <b>Archie Assistant Commands</b>

/help - Show this help message
/status - Get system status
/notifications - View recent notifications
/settings - Notification settings
/test - Send test notification
            """
            await self.telegram_client.send_message(message.chat.id, help_text)
        
        async def status_handler(message):
            status = self.get_stats()
            status_text = f"""
📊 <b>Notification System Status</b>

Total Notifications: {status['total_notifications']}
Successful: {status['successful_deliveries']}
Failed: {status['failed_deliveries']}

By Channel:
• Telegram: {status['telegram_sent']}
• Email: {status['emails_sent']}
• SMS: {status['sms_sent']}

Last Activity: {status['last_activity'] or 'None'}
            """
            await self.telegram_client.send_message(message.chat.id, status_text)
        
        async def notifications_handler(message):
            recent = list(self.notification_history.values())[-10:]  # Last 10
            if not recent:
                await self.telegram_client.send_message(message.chat.id, "No recent notifications")
                return
            
            text = "<b>Recent Notifications:</b>\n\n"
            for notif in recent:
                status = "✅" if notif.delivered else "❌"
                text += f"{status} <b>{notif.title}</b>\n"
                text += f"   {notif.timestamp.strftime('%H:%M %d/%m')}\n\n"
            
            await self.telegram_client.send_message(message.chat.id, text)
        
        async def test_handler(message):
            await self.send_notification(
                "test",
                "Test Notification",
                "This is a test notification from Archie!",
                NotificationPriority.NORMAL
            )
            await self.telegram_client.send_message(message.chat.id, "Test notification sent!")
        
        # Register handlers
        self.telegram_client.add_command_handler("help", help_handler)
        self.telegram_client.add_command_handler("start", help_handler)
        self.telegram_client.add_command_handler("status", status_handler)
        self.telegram_client.add_command_handler("notifications", notifications_handler)
        self.telegram_client.add_command_handler("test", test_handler)
    
    async def send_notification(self,
                               template_name: str,
                               title: str,
                               message: str,
                               priority: NotificationPriority = NotificationPriority.NORMAL,
                               channels: List[NotificationChannel] = None,
                               variables: Dict[str, Any] = None,
                               recipient: str = None) -> str:
        """Send notification using template"""
        try:
            # Generate notification ID
            notification_id = f"notif_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.notification_history)}"
            
            # Get template
            template = self.templates.get(template_name, self.templates["info"])
            
            # Use template channels if not specified
            if not channels:
                channels = template.channels
            
            # Format template variables
            template_vars = {
                "title": title,
                "message": message,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **(variables or {})
            }
            
            formatted_title = template.title.format(**template_vars)
            formatted_message = template.body.format(**template_vars)
            
            # Create notifications for each channel
            for channel in channels:
                notification = Notification(
                    id=f"{notification_id}_{channel.value}",
                    title=formatted_title,
                    message=formatted_message,
                    priority=priority,
                    channel=channel,
                    recipient=recipient or "default",
                    timestamp=datetime.now(),
                    delivered=False,
                    delivery_time=None,
                    attempts=0,
                    error=None,
                    metadata={"template": template_name, "variables": template_vars}
                )
                
                await self.notification_queue.put(notification)
                self.notification_history[notification.id] = notification
            
            self.stats["total_notifications"] += len(channels)
            self.stats["last_activity"] = datetime.now()
            
            logger.info(f"Queued notification '{title}' for {len(channels)} channels")
            return notification_id
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return ""
    
    async def _process_notifications(self):
        """Process notification queue"""
        while self.running:
            try:
                # Get notification from queue
                notification = await asyncio.wait_for(
                    self.notification_queue.get(), 
                    timeout=1.0
                )
                
                # Attempt delivery
                success = await self._deliver_notification(notification)
                
                if success:
                    notification.delivered = True
                    notification.delivery_time = datetime.now()
                    self.stats["successful_deliveries"] += 1
                else:
                    notification.attempts += 1
                    self.stats["failed_deliveries"] += 1
                    
                    # Retry logic for failed notifications
                    if notification.attempts < 3 and notification.priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL]:
                        # Retry after delay
                        await asyncio.sleep(30)
                        await self.notification_queue.put(notification)
                    else:
                        self.failed_notifications[notification.id] = notification
                
                self.notification_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing notification: {e}")
    
    async def _deliver_notification(self, notification: Notification) -> bool:
        """Deliver notification via specific channel"""
        try:
            if notification.channel == NotificationChannel.TELEGRAM:
                return await self._send_telegram(notification)
            elif notification.channel == NotificationChannel.EMAIL:
                return await self._send_email(notification)
            elif notification.channel == NotificationChannel.SMS:
                return await self._send_sms(notification)
            else:
                logger.warning(f"Unsupported channel: {notification.channel}")
                return False
                
        except Exception as e:
            notification.error = str(e)
            logger.error(f"Error delivering notification via {notification.channel}: {e}")
            return False
    
    async def _send_telegram(self, notification: Notification) -> bool:
        """Send notification via Telegram"""
        try:
            if not self.telegram_client:
                return False
            
            # Get chat IDs from config
            chat_ids = self.config.get('telegram', {}).get('default_chat_ids', [])
            if notification.recipient != "default":
                # Try to find specific recipient
                recipient_chats = self.config.get('telegram', {}).get('recipients', {})
                if notification.recipient in recipient_chats:
                    chat_ids = [recipient_chats[notification.recipient]]
            
            success = False
            for chat_id in chat_ids:
                # Add priority emoji
                priority_emoji = {
                    NotificationPriority.LOW: "🔵",
                    NotificationPriority.NORMAL: "🟢", 
                    NotificationPriority.HIGH: "🟡",
                    NotificationPriority.CRITICAL: "🔴"
                }
                
                formatted_message = f"{priority_emoji.get(notification.priority, '🔵')} {notification.title}\n\n{notification.message}"
                
                message = await self.telegram_client.send_message(
                    chat_id, 
                    formatted_message,
                    disable_notification=(notification.priority == NotificationPriority.LOW)
                )
                
                if message:
                    success = True
            
            if success:
                self.stats["telegram_sent"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False
    
    async def _send_email(self, notification: Notification) -> bool:
        """Send notification via email"""
        try:
            if not self.email_config.get('enabled', False):
                return False
            
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port', 587)
            username = self.email_config.get('username')
            password = self.email_config.get('password')
            from_email = self.email_config.get('from_email', username)
            
            if not all([smtp_server, username, password]):
                logger.error("Email configuration incomplete")
                return False
            
            # Get recipients
            recipients = self.email_config.get('default_recipients', [])
            if notification.recipient != "default":
                recipient_emails = self.email_config.get('recipients', {})
                if notification.recipient in recipient_emails:
                    recipients = [recipient_emails[notification.recipient]]
            
            if not recipients:
                logger.error("No email recipients configured")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = notification.title
            
            # Add priority header
            if notification.priority == NotificationPriority.CRITICAL:
                msg['X-Priority'] = '1'
            elif notification.priority == NotificationPriority.HIGH:
                msg['X-Priority'] = '2'
            
            msg.attach(MIMEText(notification.message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            self.stats["emails_sent"] += 1
            logger.info(f"Email sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_sms(self, notification: Notification) -> bool:
        """Send notification via SMS (email-to-SMS gateway)"""
        try:
            if not self.sms_config.get('enabled', False):
                return False
            
            # Get SMS recipients
            sms_recipients = self.sms_config.get('recipients', {})
            if notification.recipient != "default":
                if notification.recipient not in sms_recipients:
                    logger.error(f"SMS recipient not found: {notification.recipient}")
                    return False
                recipients = [sms_recipients[notification.recipient]]
            else:
                recipients = list(sms_recipients.values())
            
            if not recipients:
                logger.error("No SMS recipients configured")
                return False
            
            success = False
            
            for recipient in recipients:
                phone_number = recipient.get('phone')
                carrier = recipient.get('carrier')
                
                if not phone_number or not carrier:
                    continue
                
                # Get SMS provider
                provider = self.sms_providers.get(carrier.lower())
                if not provider:
                    logger.warning(f"Unknown SMS carrier: {carrier}")
                    continue
                
                # Format phone number and create email address
                clean_phone = re.sub(r'[^\d]', '', phone_number)
                sms_email = f"{clean_phone}@{provider.email_domain}"
                
                # Truncate message if too long
                message = notification.message
                if len(message) > provider.character_limit:
                    message = message[:provider.character_limit - 3] + "..."
                
                # Send via email-to-SMS
                try:
                    smtp_server = self.email_config.get('smtp_server')
                    smtp_port = self.email_config.get('smtp_port', 587)
                    username = self.email_config.get('username')
                    password = self.email_config.get('password')
                    from_email = self.email_config.get('from_email', username)
                    
                    msg = MIMEText(message)
                    msg['From'] = from_email
                    msg['To'] = sms_email
                    msg['Subject'] = ""  # Most carriers ignore subject for SMS
                    
                    server = smtplib.SMTP(smtp_server, smtp_port)
                    server.starttls()
                    server.login(username, password)
                    server.send_message(msg)
                    server.quit()
                    
                    success = True
                    logger.info(f"SMS sent to {phone_number} via {carrier}")
                    
                except Exception as e:
                    logger.error(f"Error sending SMS to {phone_number}: {e}")
            
            if success:
                self.stats["sms_sent"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    async def send_test_notifications(self) -> Dict[str, bool]:
        """Send test notifications to all configured channels"""
        results = {}
        
        try:
            # Test Telegram
            if self.telegram_client:
                telegram_id = await self.send_notification(
                    "test",
                    "Test Notification",
                    "This is a test notification from Archie's notification system.",
                    NotificationPriority.NORMAL,
                    [NotificationChannel.TELEGRAM]
                )
                results["telegram"] = bool(telegram_id)
            
            # Test Email
            if self.email_config.get('enabled', False):
                email_id = await self.send_notification(
                    "test",
                    "Test Email Notification",
                    "This is a test email notification from Archie's notification system.",
                    NotificationPriority.NORMAL,
                    [NotificationChannel.EMAIL]
                )
                results["email"] = bool(email_id)
            
            # Test SMS
            if self.sms_config.get('enabled', False):
                sms_id = await self.send_notification(
                    "test",
                    "Test SMS",
                    "Test SMS from Archie",
                    NotificationPriority.NORMAL,
                    [NotificationChannel.SMS]
                )
                results["sms"] = bool(sms_id)
            
        except Exception as e:
            logger.error(f"Error sending test notifications: {e}")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        return {
            **self.stats,
            "queue_size": self.notification_queue.qsize(),
            "total_history": len(self.notification_history),
            "failed_notifications": len(self.failed_notifications),
            "templates_loaded": len(self.templates),
            "telegram_enabled": self.telegram_client is not None,
            "email_enabled": self.email_config.get('enabled', False),
            "sms_enabled": self.sms_config.get('enabled', False),
            "last_activity": self.stats["last_activity"].isoformat() if self.stats["last_activity"] else None
        }
    
    async def cleanup(self):
        """Clean up notification manager"""
        try:
            logger.info("Cleaning up notification manager...")
            
            self.running = False
            
            # Cancel processor task
            if self.processor_task:
                self.processor_task.cancel()
                try:
                    await self.processor_task
                except asyncio.CancelledError:
                    pass
            
            # Clean up Telegram client
            if self.telegram_client:
                await self.telegram_client.cleanup()
            
            logger.info("Notification manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during notification cleanup: {e}")

# Factory function
async def create_notification_manager(config: Dict[str, Any]) -> NotificationManager:
    """Create and initialize notification manager"""
    manager = NotificationManager(config)
    await manager.initialize()
    return manager