# Integrations Module
# Smart home and external system integrations

from .home_assistant import HomeAssistantClient, HAEntity, HAService, HAArea, HADevice, create_home_assistant_client
from .n8n_client import N8nClient, N8nWorkflow, N8nExecution, N8nCredential, N8nNode, create_n8n_client
from .workflow_generator import WorkflowGenerator, WorkflowType, WorkflowIntent, create_workflow_generator
from .device_controller import DeviceController, DeviceInfo, DeviceType, DeviceState, create_device_controller
from .telegram_client import TelegramClient, TelegramMessage, TelegramUser, TelegramChat, create_telegram_client
from .notification_manager import NotificationManager, Notification, NotificationPriority, NotificationChannel, create_notification_manager

__all__ = [
    # Home Assistant
    'HomeAssistantClient', 'HAEntity', 'HAService', 'HAArea', 'HADevice', 'create_home_assistant_client',
    # n8n Workflow Management
    'N8nClient', 'N8nWorkflow', 'N8nExecution', 'N8nCredential', 'N8nNode', 'create_n8n_client',
    # Workflow Generation
    'WorkflowGenerator', 'WorkflowType', 'WorkflowIntent', 'create_workflow_generator',
    # Device Control
    'DeviceController', 'DeviceInfo', 'DeviceType', 'DeviceState', 'create_device_controller',
    # Telegram
    'TelegramClient', 'TelegramMessage', 'TelegramUser', 'TelegramChat', 'create_telegram_client',
    # Notifications
    'NotificationManager', 'Notification', 'NotificationPriority', 'NotificationChannel', 'create_notification_manager'
]