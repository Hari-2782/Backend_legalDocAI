import psutil
import gc
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, memory_threshold_mb: int = 1000):
        """
        Initialize memory manager with threshold in MB.
        
        Args:
            memory_threshold_mb: Memory threshold in MB before cleanup is triggered
        """
        self.memory_threshold_mb = memory_threshold_mb
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> dict:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory usage details
        """
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
                "percent": memory_percent,
                "available_system_mb": psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}
    
    def is_memory_critical(self) -> bool:
        """
        Check if memory usage is above threshold.
        
        Returns:
            True if memory usage is critical
        """
        memory_info = self.get_memory_usage()
        rss_mb = memory_info.get("rss_mb", 0)
        return rss_mb > self.memory_threshold_mb
    
    def force_cleanup(self) -> bool:
        """
        Force garbage collection and memory cleanup.
        
        Returns:
            True if cleanup was successful
        """
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Get memory before and after
            memory_before = self.get_memory_usage()
            
            # Additional cleanup steps
            gc.collect(2)  # More aggressive collection
            
            memory_after = self.get_memory_usage()
            
            logger.info(f"Memory cleanup: collected {collected} objects")
            logger.info(f"Memory before: {memory_before.get('rss_mb', 0):.2f} MB")
            logger.info(f"Memory after: {memory_after.get('rss_mb', 0):.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return False
    
    def monitor_memory_usage(self, operation_name: str = "operation") -> Optional[dict]:
        """
        Monitor memory usage during an operation.
        
        Args:
            operation_name: Name of the operation being monitored
            
        Returns:
            Memory usage information or None if error
        """
        try:
            memory_info = self.get_memory_usage()
            
            logger.info(f"Memory usage during {operation_name}: {memory_info.get('rss_mb', 0):.2f} MB")
            
            # Check if cleanup is needed
            if self.is_memory_critical():
                logger.warning(f"Memory usage critical during {operation_name}, triggering cleanup")
                self.force_cleanup()
            
            return memory_info
            
        except Exception as e:
            logger.error(f"Error monitoring memory during {operation_name}: {e}")
            return None
    
    def get_memory_recommendations(self) -> list:
        """
        Get memory optimization recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        memory_info = self.get_memory_usage()
        
        rss_mb = memory_info.get("rss_mb", 0)
        available_system_mb = memory_info.get("available_system_mb", 0)
        
        if rss_mb > 800:
            recommendations.append("Consider reducing batch sizes for PDF processing")
        
        if rss_mb > 1000:
            recommendations.append("Memory usage is high - consider processing smaller files")
        
        if available_system_mb < 500:
            recommendations.append("System memory is low - consider closing other applications")
        
        if rss_mb > 1500:
            recommendations.append("Critical memory usage - consider restarting the service")
        
        return recommendations

# Global memory manager instance
memory_manager = MemoryManager()

def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    return memory_manager
