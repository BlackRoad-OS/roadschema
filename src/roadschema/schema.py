"""
RoadSchema - Schema Validation for BlackRoad
Validate data against schemas with rich error messages.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
import re
import logging

logger = logging.getLogger(__name__)


class SchemaError(Exception):
    def __init__(self, message: str, path: str = "", value: Any = None):
        self.message = message
        self.path = path
        self.value = value
        super().__init__(f"{path}: {message}" if path else message)


@dataclass
class ValidationError:
    path: str
    message: str
    value: Any = None


@dataclass
class ValidationResult:
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    data: Any = None


class Schema:
    def validate(self, value: Any, path: str = "") -> ValidationResult:
        raise NotImplementedError


class StringSchema(Schema):
    def __init__(self, min_length: int = None, max_length: int = None, pattern: str = None, enum: List[str] = None):
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
        self.enum = enum

    def validate(self, value: Any, path: str = "") -> ValidationResult:
        errors = []
        if not isinstance(value, str):
            errors.append(ValidationError(path, f"Expected string, got {type(value).__name__}", value))
            return ValidationResult(valid=False, errors=errors)
        
        if self.min_length and len(value) < self.min_length:
            errors.append(ValidationError(path, f"String too short (min {self.min_length})", value))
        if self.max_length and len(value) > self.max_length:
            errors.append(ValidationError(path, f"String too long (max {self.max_length})", value))
        if self.pattern and not self.pattern.match(value):
            errors.append(ValidationError(path, f"String doesn't match pattern", value))
        if self.enum and value not in self.enum:
            errors.append(ValidationError(path, f"Value not in {self.enum}", value))
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, data=value)


class NumberSchema(Schema):
    def __init__(self, minimum: float = None, maximum: float = None, exclusive_min: float = None, exclusive_max: float = None, multiple_of: float = None, integer: bool = False):
        self.minimum = minimum
        self.maximum = maximum
        self.exclusive_min = exclusive_min
        self.exclusive_max = exclusive_max
        self.multiple_of = multiple_of
        self.integer = integer

    def validate(self, value: Any, path: str = "") -> ValidationResult:
        errors = []
        if self.integer and not isinstance(value, int):
            errors.append(ValidationError(path, "Expected integer", value))
            return ValidationResult(valid=False, errors=errors)
        if not isinstance(value, (int, float)):
            errors.append(ValidationError(path, f"Expected number, got {type(value).__name__}", value))
            return ValidationResult(valid=False, errors=errors)
        
        if self.minimum is not None and value < self.minimum:
            errors.append(ValidationError(path, f"Value below minimum {self.minimum}", value))
        if self.maximum is not None and value > self.maximum:
            errors.append(ValidationError(path, f"Value above maximum {self.maximum}", value))
        if self.exclusive_min is not None and value <= self.exclusive_min:
            errors.append(ValidationError(path, f"Value must be > {self.exclusive_min}", value))
        if self.exclusive_max is not None and value >= self.exclusive_max:
            errors.append(ValidationError(path, f"Value must be < {self.exclusive_max}", value))
        if self.multiple_of and value % self.multiple_of != 0:
            errors.append(ValidationError(path, f"Value must be multiple of {self.multiple_of}", value))
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, data=value)


class BooleanSchema(Schema):
    def validate(self, value: Any, path: str = "") -> ValidationResult:
        if not isinstance(value, bool):
            return ValidationResult(valid=False, errors=[ValidationError(path, f"Expected boolean, got {type(value).__name__}", value)])
        return ValidationResult(valid=True, data=value)


class ArraySchema(Schema):
    def __init__(self, items: Schema = None, min_items: int = None, max_items: int = None, unique: bool = False):
        self.items = items
        self.min_items = min_items
        self.max_items = max_items
        self.unique = unique

    def validate(self, value: Any, path: str = "") -> ValidationResult:
        errors = []
        if not isinstance(value, list):
            errors.append(ValidationError(path, f"Expected array, got {type(value).__name__}", value))
            return ValidationResult(valid=False, errors=errors)
        
        if self.min_items and len(value) < self.min_items:
            errors.append(ValidationError(path, f"Array too short (min {self.min_items})", value))
        if self.max_items and len(value) > self.max_items:
            errors.append(ValidationError(path, f"Array too long (max {self.max_items})", value))
        if self.unique and len(value) != len(set(str(v) for v in value)):
            errors.append(ValidationError(path, "Array items must be unique", value))
        
        validated = []
        if self.items:
            for i, item in enumerate(value):
                result = self.items.validate(item, f"{path}[{i}]")
                errors.extend(result.errors)
                validated.append(result.data)
        else:
            validated = value
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, data=validated)


class ObjectSchema(Schema):
    def __init__(self, properties: Dict[str, Schema] = None, required: List[str] = None, additional: bool = True):
        self.properties = properties or {}
        self.required = required or []
        self.additional = additional

    def validate(self, value: Any, path: str = "") -> ValidationResult:
        errors = []
        if not isinstance(value, dict):
            errors.append(ValidationError(path, f"Expected object, got {type(value).__name__}", value))
            return ValidationResult(valid=False, errors=errors)
        
        for req in self.required:
            if req not in value:
                errors.append(ValidationError(f"{path}.{req}" if path else req, "Required field missing", None))
        
        validated = {}
        for key, schema in self.properties.items():
            if key in value:
                prop_path = f"{path}.{key}" if path else key
                result = schema.validate(value[key], prop_path)
                errors.extend(result.errors)
                validated[key] = result.data
        
        if self.additional:
            for key in value:
                if key not in self.properties:
                    validated[key] = value[key]
        elif not self.additional:
            for key in value:
                if key not in self.properties:
                    errors.append(ValidationError(f"{path}.{key}" if path else key, "Additional property not allowed", value[key]))
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, data=validated)


class UnionSchema(Schema):
    def __init__(self, schemas: List[Schema]):
        self.schemas = schemas

    def validate(self, value: Any, path: str = "") -> ValidationResult:
        for schema in self.schemas:
            result = schema.validate(value, path)
            if result.valid:
                return result
        return ValidationResult(valid=False, errors=[ValidationError(path, "Value doesn't match any schema", value)])


class NullableSchema(Schema):
    def __init__(self, schema: Schema):
        self.schema = schema

    def validate(self, value: Any, path: str = "") -> ValidationResult:
        if value is None:
            return ValidationResult(valid=True, data=None)
        return self.schema.validate(value, path)


class RefSchema(Schema):
    def __init__(self, ref: str, registry: "SchemaRegistry"):
        self.ref = ref
        self.registry = registry

    def validate(self, value: Any, path: str = "") -> ValidationResult:
        schema = self.registry.get(self.ref)
        if not schema:
            return ValidationResult(valid=False, errors=[ValidationError(path, f"Unknown schema ref: {self.ref}", value)])
        return schema.validate(value, path)


class CustomSchema(Schema):
    def __init__(self, validator: Callable[[Any], bool], message: str = "Validation failed"):
        self.validator = validator
        self.message = message

    def validate(self, value: Any, path: str = "") -> ValidationResult:
        if self.validator(value):
            return ValidationResult(valid=True, data=value)
        return ValidationResult(valid=False, errors=[ValidationError(path, self.message, value)])


class SchemaRegistry:
    def __init__(self):
        self.schemas: Dict[str, Schema] = {}

    def register(self, name: str, schema: Schema) -> None:
        self.schemas[name] = schema

    def get(self, name: str) -> Optional[Schema]:
        return self.schemas.get(name)

    def ref(self, name: str) -> RefSchema:
        return RefSchema(name, self)


class SchemaBuilder:
    def __init__(self, registry: SchemaRegistry = None):
        self.registry = registry or SchemaRegistry()

    def string(self, **kwargs) -> StringSchema:
        return StringSchema(**kwargs)

    def number(self, **kwargs) -> NumberSchema:
        return NumberSchema(**kwargs)

    def integer(self, **kwargs) -> NumberSchema:
        return NumberSchema(integer=True, **kwargs)

    def boolean(self) -> BooleanSchema:
        return BooleanSchema()

    def array(self, items: Schema = None, **kwargs) -> ArraySchema:
        return ArraySchema(items=items, **kwargs)

    def object(self, properties: Dict[str, Schema] = None, **kwargs) -> ObjectSchema:
        return ObjectSchema(properties=properties, **kwargs)

    def union(self, *schemas: Schema) -> UnionSchema:
        return UnionSchema(list(schemas))

    def nullable(self, schema: Schema) -> NullableSchema:
        return NullableSchema(schema)

    def custom(self, validator: Callable, message: str = "Validation failed") -> CustomSchema:
        return CustomSchema(validator, message)

    def email(self) -> StringSchema:
        return StringSchema(pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    def url(self) -> StringSchema:
        return StringSchema(pattern=r'^https?://[^\s/$.?#].[^\s]*$')

    def uuid(self) -> StringSchema:
        return StringSchema(pattern=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def validate(value: Any, schema: Schema) -> ValidationResult:
    return schema.validate(value)


def example_usage():
    s = SchemaBuilder()
    
    user_schema = s.object(
        properties={
            "id": s.integer(minimum=1),
            "name": s.string(min_length=2, max_length=100),
            "email": s.email(),
            "age": s.nullable(s.integer(minimum=0, maximum=150)),
            "role": s.string(enum=["admin", "user", "guest"]),
            "tags": s.array(items=s.string(), unique=True),
        },
        required=["id", "name", "email", "role"]
    )
    
    valid_user = {
        "id": 1,
        "name": "Alice",
        "email": "alice@example.com",
        "age": 30,
        "role": "admin",
        "tags": ["developer", "manager"]
    }
    
    result = validate(valid_user, user_schema)
    print(f"Valid: {result.valid}")
    
    invalid_user = {
        "id": "not-a-number",
        "name": "A",
        "email": "invalid-email",
        "role": "superuser",
    }
    
    result = validate(invalid_user, user_schema)
    print(f"Valid: {result.valid}")
    for error in result.errors:
        print(f"  {error.path}: {error.message}")

