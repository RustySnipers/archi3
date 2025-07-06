"""
Main entry point for Archie
Starts the API server and orchestrator
"""

import os
import sys
import asyncio
import logging
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from archie_core.agent import create_orchestrator
from archie_core.api_server import create_api_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArchieMain:
    """Main Archie application"""
    
    def __init__(self):
        self.orchestrator = None
        self.api_server = None
        self.running = False
        
    async def start(self):
        """Start Archie services"""
        try:
            logger.info("Starting Archie Personal AI Assistant...")
            
            # Create and initialize orchestrator
            self.orchestrator = await create_orchestrator()
            
            # Create and start API server
            self.api_server = await create_api_server(self.orchestrator)
            
            self.running = True
            logger.info("Archie started successfully!")
            
            # Keep running until shutdown
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed to start Archie: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Shutdown Archie services"""
        logger.info("Shutting down Archie...")
        
        self.running = False
        
        # Shutdown orchestrator
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        # Shutdown API server
        if self.api_server:
            await self.api_server.shutdown()
        
        logger.info("Archie shutdown complete")
    
    def handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())

async def main():
    """Main entry point"""
    app = ArchieMain()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, app.handle_signal)
    signal.signal(signal.SIGTERM, app.handle_signal)
    
    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())