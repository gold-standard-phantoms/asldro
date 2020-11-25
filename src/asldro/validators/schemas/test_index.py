""" JSON schema index tests"""
from asldro.validators.schemas.index import SCHEMAS


def test_get_schemas():
    """Check that loading the schemas is successful
    and does not throw any SchemaError"""
    assert len(SCHEMAS) > 0  # we have more than one schema saved
