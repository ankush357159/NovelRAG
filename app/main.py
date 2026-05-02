import logging

from fastapi import FastAPI
from app.api.router import api_router

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

app = FastAPI()

app.include_router(api_router)


def main():
    print("Hello from backend!")


if __name__ == "__main__":
    main()
