from pydantic import ValidationError
from api.schemas import QueryRequest

ok=QueryRequest(query="What are the payment terms?")
print("valid: ",ok)
print("as dict ",ok.model_dump())

try:
    QueryRequest(retriever_type="similarity")
except ValidationError as err:
    first=err.errors()[0]
    print("invalid:",first["type"],first["loc"],"-",first["msg"])