"""Registry for managing factor implementations."""
from typing import Dict, List, Type, Optional
from .base import BaseFactor
import logging

logger = logging.getLogger(__name__)


class FactorRegistry:
    """Simple registry to track and manage available factors."""
    
    _factors: Dict[str, Type[BaseFactor]] = {}
    
    @classmethod
    def register(cls, factor_class: Type[BaseFactor]) -> Type[BaseFactor]:
        """Register a factor class.
        
        Can be used as a decorator:
        @FactorRegistry.register
        class MyFactor(BaseFactor):
            ...
            
        Args:
            factor_class: Factor class to register
            
        Returns:
            The registered factor class (for decorator usage)
        """
        # Create an instance to get the factor name
        instance = factor_class()
        factor_name = instance.name
        
        if factor_name in cls._factors:
            logger.warning(f"Factor '{factor_name}' already registered, overwriting")
        
        cls._factors[factor_name] = factor_class
        logger.info(f"Registered factor: {factor_name} ({factor_class.__name__})")
        
        return factor_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseFactor]:
        """Get a factor class by name.
        
        Args:
            name: Factor name to retrieve
            
        Returns:
            Factor class
            
        Raises:
            ValueError: If factor name is not registered
        """
        if name not in cls._factors:
            available = cls.list_factors()
            raise ValueError(f"Unknown factor: '{name}'. Available factors: {available}")
        return cls._factors[name]
    
    @classmethod
    def create(cls, name: str) -> BaseFactor:
        """Create a new instance of a factor by name.
        
        Args:
            name: Factor name to instantiate
            
        Returns:
            New factor instance
        """
        factor_class = cls.get(name)
        return factor_class()
    
    @classmethod
    def list_factors(cls) -> List[str]:
        """List all registered factor names.
        
        Returns:
            List of registered factor names
        """
        return sorted(list(cls._factors.keys()))
    
    @classmethod
    def get_factors_by_category(cls, category: str) -> List[str]:
        """Get all factors in a specific category.
        
        Args:
            category: Factor category to filter by
            
        Returns:
            List of factor names in the category
        """
        factors = []
        for name, factor_class in cls._factors.items():
            instance = factor_class()
            if instance.category == category:
                factors.append(name)
        return sorted(factors)
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get all unique factor categories.
        
        Returns:
            List of unique categories
        """
        categories = set()
        for factor_class in cls._factors.values():
            instance = factor_class()
            categories.add(instance.category)
        return sorted(list(categories))
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered factors (mainly for testing)."""
        cls._factors.clear()
        logger.info("Cleared factor registry")