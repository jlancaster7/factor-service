#!/usr/bin/env python
"""Test configuration and environment setup."""
import sys
sys.path.insert(0, '.')

from src.config import Config


def main():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        config = Config()
        print("\n✓ Configuration loaded successfully")
        print(f"\nConfiguration details:")
        print(f"  Snowflake Account: {config.snowflake_account}")
        print(f"  Snowflake User: {config.snowflake_user}")
        print(f"  Snowflake Warehouse: {config.snowflake_warehouse}")
        print(f"  Snowflake Database: {config.snowflake_database}")
        print(f"  Snowflake Schema: {config.snowflake_schema}")
        print(f"  Log Level: {config.log_level}")
        
        # Test validation
        print("\nValidating configuration...")
        config.validate()
        print("✓ Configuration validation passed")
        
    except Exception as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()