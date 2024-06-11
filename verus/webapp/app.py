import argparse
import logging
import shutil
from pathlib import Path

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Security, UploadFile, status
from fastapi.security import APIKeyHeader

from verus.db import User, setup_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
api_key_header = APIKeyHeader(name="X-API-Key")


def get_user_from_api_key(api_key: str) -> User | None:
    return User.get_or_none(User.api_key == api_key)  # type: ignore[no-any-return]


def get_user(api_key_header: str = Security(api_key_header)) -> dict[str, str]:
    if user := get_user_from_api_key(api_key_header):
        return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail={"success": False, "message": "Invalid API key"}
    )


class WebApp:
    def __init__(self, import_folder: Path) -> None:
        self.logger = logging.getLogger(__name__)

        self.import_folder = import_folder

        self.app = FastAPI()
        self.app.add_event_handler("startup", self.startup_event)

        self.router = APIRouter()
        self.router.get("/")(self.get_status)

        self.secured_router = APIRouter()
        self.secured_router.post("/import")(self.import_item)

        self.app.include_router(self.router, prefix="/api/v1")
        self.app.include_router(self.secured_router, prefix="/api/v1", dependencies=[Depends(get_user)])

    async def startup_event(self) -> None:
        self.logger.info("Starting up")
        setup_db()

    async def import_item(self, file: UploadFile, user: User = Depends(get_user)) -> dict[str, str]:
        self.logger.info("Importing item")
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"success": False, "message": "No file provided"},
            )

        file_path = self.import_folder / file.filename

        with file_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        return {"status": "success", "filename": file.filename}

    async def get_status(self) -> dict[str, str]:
        self.logger.info("Getting status")
        return {"status": "running"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("import_folder", type=Path)
    parser.add_argument("--host", default="127.0.0.1", type=str, help="The host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="The port to bind to")
    args = parser.parse_args()

    web_app = WebApp(args.import_folder)
    uvicorn.run(web_app.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
