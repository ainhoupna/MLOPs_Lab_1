# Import the libraries, classes and functions
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

# from pydantic import BaseModel
from mylib.calculator import add, subtract, multiply, divide, power

# Create an instance of FastAPI
app = FastAPI(
    title="API of the Calculator using FastAPI",
    description="API to perform arithmetical operations using mylib.calculator",
    version="1.0.0",
)

# We use the templates folder to obtain HTML files
templates = Jinja2Templates(directory="templates")


# Initial endpoint
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


# # Input class (with Pydantic) to define the input arguments of the calculator
# class CalcRequest(BaseModel):
#     operation: str
#     a: float
#     b: float


# Main endpoint to perform the artihmetical operations using the input class defined with Pydantic
@app.post("/calculate")
async def calculate(op: str = Form(), a: float = Form(), b: float = Form()):
    """
    It performs an arithmetical operation according to the input parameters.
    """
    op = op.lower()

    if op not in ["add", "subtract", "multiply", "divide", "power"]:
        raise HTTPException(status_code=400, detail="Unvalid operation")

    result = None
    if op == "add":
        result = add(a, b)
    elif op == "subtract":
        result = subtract(a, b)
    elif op == "multiply":
        result = multiply(a, b)
    elif op == "divide":
        if b == 0:
            raise HTTPException(status_code=400, detail="Zero division not allowed")
        result = divide(a, b)
    elif op == "power":
        result = power(a, b)

    return {"result": result}


# # Main endpoint to perform the artihmetical operations using the input class defined with Pydantic
# @app.post("/calculate")
# def calculate(data: CalcRequest):
#     """
#     It performs an arithmetical operation according to the input parameters.
#     """
#     op = data.operation.lower()
#     a = data.a
#     b = data.b

#     if op not in ["add", "subtract", "multiply", "divide", "power"]:
#         raise HTTPException(status_code=400, detail="Unvalid operation")

#     result = None
#     if op == "add":
#         result = add(a, b)
#     elif op == "subtract":
#         result = subtract(a, b)
#     elif op == "multiply":
#         result = multiply(a, b)
#     elif op == "divide":
#         if b == 0:
#             raise HTTPException(status_code=400, detail="Zero division not allowed")
#         result = divide(a, b)
#     elif op == "power":
#         result = power(a, b)

#     return {"result": result}


# Entry point (for direct execution only)
if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
