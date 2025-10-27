class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._registry[name.lower()] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name, **kwargs):
        if name.lower() not in cls._registry:
            raise ValueError(f"Unknown model: {name}")
        return cls._registry[name.lower()]#(**kwargs)
