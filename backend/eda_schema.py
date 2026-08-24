from pydantic import BaseModel

class EDAFrame(BaseModel):
    meta: dict
    columns: list[str]
    records: list[dict]
